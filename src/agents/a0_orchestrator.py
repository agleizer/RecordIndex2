"""
A0 — Orquestrador / Contextualizador de Coleção.

Tipo de agente: Baseado em Objetivos (Russell & Norvig, tipo 3).

Papel no sistema:
  A0 é o que diferencia RecordIndex 2.0 de um pipeline linear. Ele recebe
  inputs brutos do usuário (diretório + metadados) e decide a estratégia de
  processamento antes de coordenar os demais agentes.

Fluxo completo (run):
  1. _classify(collection_input)  → CollectionConfig
     Determina tipo de documento (batismo/casamento/obito) e parâmetros.
     Prioridade:
       1. collection_type explícito → aceita direto
       2. keyword match no collection_name → sem LLM
       3. record_template presente → LLM analisa o template (sem HTR)
       4. Fallback: HTR em linhas do meio de páginas aleatórias → LLM
  2. Para cada imagem de página em image_dir:
     a. A1.segment_page()              → Page com Lines (crops + bboxes)
     b. _transcribe_and_segment(page)  → (pre_content, new_records)
        i.  A2 transcreve (por linha ou por página)
        ii. A3 detecta inícios de registro
        iii.Retorna conteúdo antes do primeiro início + novos Records parciais
  3. Mantém open_record entre páginas:
     - Conteúdo antes do primeiro início da página N+1 → appended ao open_record
     - Quando A3 detecta novo início: open_record é finalizado (A4/A5/A6),
       adicionado à Collection, e o último novo Record torna-se o novo open_record
     - Ao fim do pipeline: open_record remanescente é finalizado
  4. Retorna Collection completa.

Suporte a registros multi-página:
  Um registro pode começar na página N e terminar na página N+1 (ou além).
  A lógica do open_record garante que linhas/texto de continuação sejam
  acumulados no mesmo Record, independente de quantas páginas percorre.
  Inspirado em collection.detect_records() do RecordIndex v1.0 (AVL sweep global).

Interface LangGraph (Semana 5):
  __call__(state: dict) → dict  — estado contém CollectionInput, retorna Collection.
"""

import logging
import os
import time
from pathlib import Path

from pydantic import BaseModel

from src.config import Config
from src.llm_client import get_chat_model, make_structured
from src.agents.a1_line_segmentation import A1LineSegmentationAgent
from src.agents.a2_htr import A2HTRAgent
from src.agents.a3_segmentation import A3RecordSegmentationAgent
from src.agents.a4_correction import A4CorrectionAgent
from src.agents.a5_ner import A5NERAgent
from src.agents.a6_validation import A6ValidationAgent
from src.models.collection import Collection
from src.models.collection_config import CollectionConfig
from src.models.collection_input import CollectionInput
from src.models.page import Page
from src.models.record import Record

logger = logging.getLogger("recordindex.a0")

# DT-25 (16/08): teto de segurança pro modo page. Descoberto rodando a coleção
# completa (108 páginas, qwen3.5:9b): uma página com leitura corrompida pode levar
# o A3 a "explodir" dezenas de fronteiras falsas numa chamada só (caso real: 51
# registros espúrios de uma vez, mais dois casos de 6). Distribuição real observada
# nesse run: 1-3 blocos é o uso normal (102 páginas só com 3), nada entre 4 e 6,
# depois os casos corrompidos (6, 6, 51). Teto em 5 separa os dois grupos sem
# descartar páginas legítimas. Acima disso, a página é descartada (mesmo caminho
# de "nenhum registro encontrado" já existente) em vez de aceitar os blocos cegamente.
A3_MAX_BLOCKS_PER_PAGE = int(os.getenv("A3_MAX_BLOCKS_PER_PAGE", "5"))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Palavras-chave para classificação por nome de coleção (sem LLM)
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "batismo":   ["batismo", "baptismo", "batismos", "baptismos", "bautismo"],
    "casamento": ["casamento", "casamentos", "matrimonio", "matrimônio", "casament"],
    "obito":     ["obito", "óbito", "obitos", "óbitos", "morte", "mortes",
                  "falecimento", "falecimentos", "enterro", "enterros"],
}


class _ClassificationResult(BaseModel):
    """Schema de saída do LLM classificador de A0."""
    collection_type: str   # "batismo" | "casamento" | "obito"
    record_start_hint: str # palavra/expressão típica de início de registro
    reasoning: str


class A0Orchestrator:

    def __init__(self, config: Config):
        self._config = config

        self._a1 = A1LineSegmentationAgent()

        self._a2 = A2HTRAgent(
            get_chat_model(config.a2_provider, config.a2_model, config.ollama_base_url)
        )
        self._a3 = A3RecordSegmentationAgent(
            get_chat_model(config.a3_provider, config.a3_model, config.ollama_base_url)
        )
        self._a4 = A4CorrectionAgent(
            get_chat_model(config.a4_provider, config.a4_model, config.ollama_base_url)
        )
        self._a5 = A5NERAgent(
            get_chat_model(config.a5_provider, config.a5_model, config.ollama_base_url)
        )
        self._a6 = A6ValidationAgent(
            get_chat_model(config.a6_provider, config.a6_model, config.ollama_base_url),
            llm_threshold=config.a6_llm_threshold,
            num_predict=config.a6_num_predict,
        )

        # Fallback A2: re-transcreve registros com score abaixo do threshold
        self._a2_fallback: A2HTRAgent | None = None
        if config.a2_fallback_model:
            self._a2_fallback = A2HTRAgent(
                get_chat_model(config.a2_fallback_provider, config.a2_fallback_model, config.ollama_base_url)
            )
            logger.info(
                "A0: fallback A2 configurado — %s/%s (threshold=%.2f)",
                config.a2_fallback_provider, config.a2_fallback_model, config.a2_fallback_threshold,
            )
        self._fallback_threshold = config.a2_fallback_threshold

        self._classifier_chain = make_structured(
            get_chat_model(config.a0_provider, config.a0_model, config.ollama_base_url),
            _ClassificationResult,
        )

    # ------------------------------------------------------------------
    # Ponto de entrada principal
    # ------------------------------------------------------------------

    def run(self, collection_input: CollectionInput) -> Collection:
        """
        Executa o pipeline completo a partir de um diretório de imagens.

        Mantém um open_record entre páginas para suportar registros que
        cruzam a fronteira entre páginas: conteúdo antes do primeiro início
        detectado em cada página é acumulado no open_record anterior.
        """
        logger.info("A0: iniciando pipeline para '%s'", collection_input.collection_name)

        collection_config = self._classify(collection_input)
        logger.info(
            "A0: coleção classificada como '%s' (hint_início='%s')",
            collection_config.collection_type,
            collection_config.record_start_hint,
        )

        image_files = sorted(
            p for p in Path(collection_input.image_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_files:
            logger.warning("A0: nenhuma imagem encontrada em '%s'", collection_input.image_dir)
            return Collection(name=collection_config.collection_name)

        lines_dir = Path(self._config.output_dir) / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        collection = Collection(name=collection_config.collection_name)
        total_pages = len(image_files)
        pipeline_start = time.time()
        htr_scope = collection_input.htr_scope

        # Estado cross-página: registro ainda aberto (pode continuar na próx. página)
        self._record_counter = 0
        open_record: Record | None = None
        # Histórico de linhas por registro — alimentado conforme registros são fechados.
        # Usado para calibrar o A3 (avg_lines_hint) e detectar outliers pós-segmentação.
        self._lines_per_record_history: list[int] = []

        for page_idx, image_path in enumerate(image_files, start=1):
            logger.info("A0: ── PÁGINA %d/%d ── '%s'", page_idx, total_pages, image_path.name)
            t0 = time.time()

            logger.info("A0: [A1] segmentando linhas...")
            page = self._a1.segment_page(str(image_path), str(lines_dir))
            collection.add_page(page)
            logger.info("A0: [A1] %d linhas detectadas (%.1fs)", len(page.lines), time.time() - t0)

            pre_content, new_records = self._transcribe_and_segment(
                page, collection_config, htr_scope
            )

            # Acumular conteúdo pré-início no registro aberto (se houver)
            if open_record is not None and pre_content:
                if htr_scope == "line":
                    for line in pre_content:
                        open_record.add_line(line)
                    logger.info(
                        "A0: [A3] %d linha(s) de continuação adicionada(s) ao registro %d",
                        len(pre_content), open_record.id,
                    )
                else:
                    open_record.page_text = (open_record.page_text + "\n" + pre_content).strip()
                    logger.info(
                        "A0: [A3] texto de continuação (%d chars) adicionado ao registro %d",
                        len(pre_content), open_record.id,
                    )

            if not new_records:
                # Página inteira é continuação do registro anterior (ou frontispício)
                if open_record is None:
                    logger.info("A0: [A3] página %d sem inícios de registro — ignorada (frontispício/cabeçalho)", page_idx)
                else:
                    logger.info("A0: [A3] página %d inteira absorvida pelo registro %d", page_idx, open_record.id)
            else:
                logger.info("A0: [A3] %d novo(s) início(s) de registro detectado(s)", len(new_records))

                # Fechar o registro aberto (se houver) e finalizar todos exceto o último
                records_to_finalize: list[Record] = []
                if open_record is not None:
                    records_to_finalize.append(open_record)
                records_to_finalize.extend(new_records[:-1])

                for record in records_to_finalize:
                    self._finalize_record(record, collection_config)
                    collection.add_record(record)
                    n_lines = len(record.lines)
                    if n_lines > 0:
                        self._lines_per_record_history.append(n_lines)
                    logger.info(
                        "A0: ✓ registro %d finalizado (página_inicial='%s', linhas=%d)",
                        record.id, record.page_filename, n_lines,
                    )

                # O último registro fica aberto — pode continuar na próxima página
                open_record = new_records[-1]

            logger.info("A0: ── PÁGINA %d/%d concluída (%.1fs) ──", page_idx, total_pages, time.time() - t0)

        # Fechar o último registro aberto
        if open_record is not None:
            self._finalize_record(open_record, collection_config)
            collection.add_record(open_record)
            n_lines = len(open_record.lines)
            if n_lines > 0:
                self._lines_per_record_history.append(n_lines)
            logger.info("A0: ✓ último registro %d finalizado (linhas=%d)", open_record.id, n_lines)

        elapsed = time.time() - pipeline_start
        logger.info(
            "A0: ══ PIPELINE CONCLUÍDO ══ %d páginas | %d registros | %.1fs total (%.1fs/pág)",
            len(collection.pages), len(collection.records),
            elapsed, elapsed / max(total_pages, 1),
        )
        return collection

    # ------------------------------------------------------------------
    # Classificação da coleção
    # ------------------------------------------------------------------

    def _classify(self, collection_input: CollectionInput) -> CollectionConfig:
        """
        Determina o tipo da coleção e constrói CollectionConfig.

        Prioridade:
          1. collection_type explícito no input → aceita direto, sem LLM
          2. Keyword match no collection_name → sem LLM
          3. record_template presente → LLM analisa o template (sem HTR, sem A1/A2)
          4. Fallback: HTR em linhas de páginas aleatórias → LLM classifica
        """
        name = collection_input.collection_name
        hint_type = collection_input.collection_type.strip().lower()

        # --- Prioridade 1: tipo fornecido explicitamente ---
        if hint_type in ("batismo", "casamento", "obito"):
            logger.info("A0: tipo fornecido explicitamente: '%s'", hint_type)
            return self._build_config(hint_type, collection_input)

        # --- Prioridade 2: keyword match no nome ---
        detected = self._keyword_match(name)
        if detected:
            logger.info("A0: tipo detectado por keyword no nome: '%s'", detected)
            return self._build_config(detected, collection_input)

        # --- Prioridade 3: LLM analisa o template (sem HTR) ---
        if collection_input.record_template:
            logger.info("A0: classificando via template (sem HTR)...")
            return self._classify_from_template(collection_input)

        # --- Prioridade 4: HTR em páginas aleatórias → LLM ---
        logger.info("A0: tipo não inferível — amostrando páginas aleatórias com HTR...")
        return self._classify_with_llm(collection_input)

    def _keyword_match(self, name: str) -> str:
        """Retorna tipo se encontrar keyword no nome da coleção, caso contrário ''."""
        name_lower = name.lower()
        for col_type, keywords in _TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return col_type
        return ""

    def _classify_from_template(self, collection_input: CollectionInput) -> CollectionConfig:
        """
        Classifica usando o record_template fornecido pelo usuário — sem HTR, sem A1/A2.
        O LLM analisa o molde de texto e determina o tipo de registro e o padrão de início.
        """
        from langchain_core.messages import HumanMessage
        from src.prompts import get_prompt

        prompt = get_prompt("a0", "classify_from_template").format(
            collection_name=collection_input.collection_name,
            year=collection_input.year or "desconhecido",
            location=collection_input.location or "desconhecido",
            record_template=collection_input.record_template,
        )
        result: _ClassificationResult = self._classifier_chain.invoke(
            [HumanMessage(content=prompt)]
        )
        logger.info("A0 template classify: type=%s, hint='%s', reasoning=%s",
                    result.collection_type, result.record_start_hint, result.reasoning)

        detected_type = result.collection_type.strip().lower()
        if detected_type not in ("batismo", "casamento", "obito"):
            raise ValueError(
                f"A0: LLM retornou tipo inválido '{result.collection_type}' ao analisar template. "
                "Use collection_type='batismo'|'casamento'|'obito' para forçar."
            )

        hint = collection_input.record_start_hint or result.record_start_hint
        col_input_with_hint = CollectionInput(
            image_dir=collection_input.image_dir,
            collection_name=collection_input.collection_name,
            year=collection_input.year,
            location=collection_input.location,
            collection_type=detected_type,
            record_start_hint=hint,
            record_template=collection_input.record_template,
        )
        return self._build_config(detected_type, col_input_with_hint)

    def _classify_with_llm(self, collection_input: CollectionInput) -> CollectionConfig:
        """
        Fallback de classificação: transcreve linhas do meio de páginas aleatórias
        e pede ao LLM que determine o tipo e o padrão de início de registro.

        Usa páginas aleatórias (não a primeira) e linhas do meio da página
        para evitar capas, índices e registros rasgados no início.
        """
        import random
        import shutil
        import tempfile
        from langchain_core.messages import HumanMessage
        from src.prompts import get_prompt

        image_files = sorted(
            p for p in Path(collection_input.image_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_files:
            raise ValueError(
                f"A0: não foi possível classificar — nenhuma imagem em '{collection_input.image_dir}'"
            )

        # Selecionar até N páginas aleatórias (configurável via A0_CLASSIFY_SAMPLE_PAGES)
        sample_size = min(self._config.a0_classify_sample_pages, len(image_files))
        sampled_pages = random.sample(image_files, sample_size)

        tmp_lines_dir = tempfile.mkdtemp(prefix="a0_classify_")
        sample_lines_text = []
        try:
            for page_path in sampled_pages:
                page = self._a1.segment_page(str(page_path), tmp_lines_dir)
                # Filtrar apenas linhas válidas (outliers marcados por A1 são descartados)
                lines = [l for l in page.lines.values() if l.is_valid]
                if not lines:
                    continue
                # Pegar 2 linhas do meio da página (evita cabeçalho e rodapé)
                mid = len(lines) // 2
                for line in lines[max(0, mid - 1): mid + 1]:
                    self._a2.transcribe_line(line)
                    if line.htr_text:
                        sample_lines_text.append(line.htr_text)
        finally:
            shutil.rmtree(tmp_lines_dir, ignore_errors=True)

        if not sample_lines_text:
            raise ValueError("A0: transcrição de amostra retornou vazio — não é possível classificar.")

        lines_fmt = "\n".join(f"{i}: {t}" for i, t in enumerate(sample_lines_text))
        prompt = get_prompt("a0", "classify").format(
            collection_name=collection_input.collection_name,
            year=collection_input.year or "desconhecido",
            location=collection_input.location or "desconhecido",
            lines_fmt=lines_fmt,
        )

        result: _ClassificationResult = self._classifier_chain.invoke(
            [HumanMessage(content=prompt)]
        )
        logger.info("A0 LLM classify: type=%s, hint='%s', reasoning=%s",
                    result.collection_type, result.record_start_hint, result.reasoning)

        # Validar tipo retornado pelo LLM
        detected_type = result.collection_type.strip().lower()
        if detected_type not in ("batismo", "casamento", "obito"):
            raise ValueError(
                f"A0: LLM retornou tipo inválido '{result.collection_type}'. "
                "Use collection_type='batismo'|'casamento'|'obito' para forçar."
            )

        # Usar hint inferido pelo LLM se o usuário não forneceu
        hint = collection_input.record_start_hint or result.record_start_hint
        col_input_with_hint = CollectionInput(
            image_dir=collection_input.image_dir,
            collection_name=collection_input.collection_name,
            year=collection_input.year,
            location=collection_input.location,
            collection_type=detected_type,
            record_start_hint=hint,
            record_template=collection_input.record_template,
        )
        return self._build_config(detected_type, col_input_with_hint)

    def _build_config(self, collection_type: str, collection_input: CollectionInput) -> CollectionConfig:
        """Constrói CollectionConfig a partir do tipo detectado e dos inputs do usuário."""
        factories = {
            "batismo":   CollectionConfig.batismo,
            "casamento": CollectionConfig.casamento,
            "obito":     CollectionConfig.obito,
        }
        cfg = factories[collection_type](name=collection_input.collection_name)
        if collection_input.record_start_hint:
            # Hint explícito do usuário tem prioridade máxima
            cfg.record_start_hint = collection_input.record_start_hint
        elif collection_input.record_template:
            # Extrair hint das primeiras palavras fixas do template
            extracted = self._hint_from_template(collection_input.record_template)
            if extracted:
                cfg.record_start_hint = extracted
                logger.info("A0: record_start_hint extraído do template: '%s'", extracted)
        if collection_input.record_template:
            cfg.record_template = collection_input.record_template
        return cfg

    @staticmethod
    def _hint_from_template(template: str, n_words: int = 12) -> str:
        """
        Extrai o padrão de início de registro das primeiras N palavras do template.

        Preserva os placeholders (<CAMPO>) para que o LLM entenda o padrão completo.
        Ex: "Aos <DATA> de <MES> de <ANO>, eu <PADRE>, batizei..." (12 palavras).
        """
        words = template.strip().split()
        return " ".join(words[:n_words]) if words else ""

    # ------------------------------------------------------------------
    # Transcrição + Segmentação (A2 → A3) — retorna conteúdo de continuação
    # e novos Records parciais (não finalizados)
    # ------------------------------------------------------------------

    def _next_record_id(self) -> int:
        self._record_counter += 1
        return self._record_counter

    def _avg_lines_hint(self) -> int | None:
        """
        Retorna a média arredondada de linhas por registro com base nos últimos 20 registros
        fechados. Retorna None se ainda não há dados suficientes (< 3 registros).
        Só tem valor para modo line — em modo page os registros não têm linhas individuais.
        """
        history = self._lines_per_record_history
        if len(history) < self._config.a0_avg_lines_min_history:
            return None
        recent = history[-self._config.a0_avg_lines_window:]
        return round(sum(recent) / len(recent))

    @staticmethod
    def _extract_pre_text(full_text: str, first_record_text: str) -> str:
        """
        Extrai o texto que precede o primeiro bloco de registro em full_text.
        Usa os primeiros 60 chars do primeiro bloco como âncora de busca.
        Retorna string vazia se o primeiro bloco começa no início do full_text.
        """
        if not first_record_text:
            return ""
        anchor = first_record_text.strip()[:60]
        idx = full_text.find(anchor)
        if idx > 0:
            return full_text[:idx].strip()
        return ""

    def _resegment_outliers(
        self,
        records: list[Record],
        collection_config: CollectionConfig,
        avg_lines: float,
        page_filename: str,
    ) -> list[Record]:
        """
        Detecta registros com linha-count excessivo (>= avg * A3_OUTLIER_MULTIPLIER) e
        re-executa o A3 só no bloco de linhas daquele registro.

        Inspirado em collection.treat_records_outliers() do RecordIndex v1.0:
        em vez de cosine distance, usa o LLM diretamente para encontrar as fronteiras.

        Máximo 1 re-segmentação por registro — sem loop. Se o A3 ainda retornar
        1 único bloco (não conseguiu dividir), mantém o registro original.

        Apenas modo LINE — em modo page registros não têm linhas individuais.
        """
        multiplier = self._config.a3_outlier_multiplier
        if multiplier <= 0 or not records:
            return records

        threshold = avg_lines * multiplier
        result: list[Record] = []

        for record in records:
            lines = list(record.lines.values())
            if len(lines) < threshold:
                result.append(record)
                continue

            logger.info(
                "A0: [A3-OUTLIER] registro %d tem %d linhas (threshold=%.1f) — re-segmentando",
                record.id, len(lines), threshold,
            )

            # Re-executa A3 só no bloco deste registro, com hint preciso
            hint = round(avg_lines)
            boundaries = self._a3.detect_boundaries(lines, collection_config, avg_lines_hint=hint)
            starts = sorted(
                set(i for i in boundaries.record_start_indices if 0 <= i < len(lines))
            )

            if len(starts) <= 1:
                # A3 não encontrou divisão — mantém o registro original
                logger.info(
                    "A0: [A3-OUTLIER] re-segmentação não encontrou divisão — mantendo registro %d",
                    record.id,
                )
                result.append(record)
                continue

            logger.info(
                "A0: [A3-OUTLIER] registro %d dividido em %d partes em inícios %s",
                record.id, len(starts), starts,
            )
            for record_idx, start in enumerate(starts):
                end = starts[record_idx + 1] if record_idx + 1 < len(starts) else len(lines)
                new_rec = Record(id=self._next_record_id(), page_filename=page_filename)
                for line in lines[start:end]:
                    new_rec.add_line(line)
                result.append(new_rec)

        return result

    def _transcribe_and_segment(
        self,
        page: Page,
        collection_config: CollectionConfig,
        htr_scope: str,
    ) -> tuple[list | str, list[Record]]:
        """
        Executa A2 (transcrição) e A3 (detecção de inícios) em uma página.

        Retorna (pre_start_content, new_records):

          pre_start_content — conteúdo que precede o primeiro início de registro:
            line mode: list[Line] (pode ser lista vazia)
            page mode: str        (pode ser string vazia)

          new_records — Records parciais detectados nesta página (sem A4/A5/A6).
            Vazio se a página inteira é continuação do registro anterior.
            O ÚLTIMO Record pode continuar na próxima página.

        O chamador (run) é responsável por:
          - Acumular pre_start_content no open_record anterior (se houver)
          - Finalizar records completos via _finalize_record()
          - Manter o último new_record como open_record
        """
        lines = list(page.lines.values())
        valid_lines = [l for l in lines if l.is_valid]
        n_total = len(lines)
        n_valid = len(valid_lines)

        if n_total != n_valid:
            logger.info(
                "A0: [A2] %d linhas (%d válidas, %d outliers ignorados) — modo %s",
                n_total, n_valid, n_total - n_valid, htr_scope,
            )
        else:
            logger.info("A0: [A2] transcrevendo %d linhas — modo %s...", n_valid, htr_scope)

        t_a2 = time.time()

        if htr_scope == "page":
            # Uma chamada por página — Claude determina o número de linhas
            page_lines = self._a2.transcribe_page(page)
            logger.info(
                "A0: [A2] %s transcreveu %d linhas (A1 havia detectado %d válidas) (%.1fs)",
                self._config.a2_model, len(page_lines), n_valid, time.time() - t_a2,
            )
            full_text = "\n".join(page_lines)

            avg_hint = self._avg_lines_hint()  # None até termos histórico suficiente
            logger.info("A0: [A3] detectando blocos de registros — modo page (avg_hint=%s)...", avg_hint)
            t_a3 = time.time()
            blocks = self._a3.detect_page_blocks(full_text, collection_config, avg_lines_hint=avg_hint)
            record_texts = [t.strip() for t in blocks.record_texts if t.strip()]
            logger.info("A0: [A3] %d blocos detectados em %.1fs", len(record_texts), time.time() - t_a3)
            logger.debug("A0: [A3] reasoning: %s", blocks.reasoning)

            if len(record_texts) > A3_MAX_BLOCKS_PER_PAGE:
                logger.warning(
                    "A0: [A3] %d blocos na página '%s' excede o teto de segurança "
                    "(%d, DT-25) — descartando a segmentação desta página, provável "
                    "corrupção de leitura",
                    len(record_texts), page.filename, A3_MAX_BLOCKS_PER_PAGE,
                )
                return full_text, []

            if not record_texts:
                return full_text, []

            pre_text = self._extract_pre_text(full_text, record_texts[0])
            new_records: list[Record] = []
            for text in record_texts:
                record = Record(id=self._next_record_id(), page_filename=page.filename)
                record.page_text = text
                new_records.append(record)

            return pre_text, new_records

        else:
            # Modo LINE: uma chamada por linha
            for i, line in enumerate(valid_lines, start=1):
                logger.info("A0: [A2] linha %d/%d — %s", i, n_valid, line.id)
                self._a2.transcribe_line(line)
                logger.info("A0: [A2] → '%s'", (line.htr_text or "")[:80])
            logger.info("A0: [A2] concluído em %.1fs", time.time() - t_a2)

            # --- Filtro 1: linhas com transcrição vazia ---
            # --- Filtro 2: linhas com transcrição muito curta (ruído/ornamentos) ---
            min_chars = self._config.a3_min_htr_chars
            a3_lines = [
                l for l in valid_lines
                if l.htr_text
                and len(l.htr_text.strip()) >= min_chars
                and l.htr_text.strip() != "[ILEGÍVEL]"
            ]
            n_filtered = len(valid_lines) - len(a3_lines)
            if n_filtered:
                logger.info(
                    "A0: [A3] %d linha(s) filtrada(s) por texto vazio ou < %d chars antes do A3",
                    n_filtered, min_chars,
                )

            avg_hint = self._avg_lines_hint()
            logger.info("A0: [A3] detectando inícios de registro — modo line (avg_hint=%s)...", avg_hint)
            t_a3 = time.time()
            boundaries = self._a3.detect_boundaries(a3_lines, collection_config, avg_lines_hint=avg_hint)
            starts = sorted(
                set(i for i in boundaries.record_start_indices if 0 <= i < len(a3_lines))
            )
            logger.info("A0: [A3] %d inícios detectados em %.1fs", len(starts), time.time() - t_a3)
            logger.debug("A0: [A3] reasoning: %s", boundaries.reasoning)

            if not starts:
                return a3_lines, []

            pre_lines = a3_lines[:starts[0]]
            new_records = []
            for record_idx, start in enumerate(starts):
                end = starts[record_idx + 1] if record_idx + 1 < len(starts) else len(a3_lines)
                record = Record(id=self._next_record_id(), page_filename=page.filename)
                for line in a3_lines[start:end]:
                    record.add_line(line)
                new_records.append(record)

            # --- Filtro 3: re-segmentar registros outlier (merge detectado) ---
            if avg_hint and self._config.a3_outlier_multiplier > 0:
                new_records = self._resegment_outliers(
                    new_records, collection_config, float(avg_hint), page.filename
                )

            return pre_lines, new_records

    # ------------------------------------------------------------------
    # Finalização de registro completo (A4 → A5 → A6 → retry)
    # ------------------------------------------------------------------

    def _finalize_record(self, record: Record, collection_config: CollectionConfig) -> None:
        """
        Executa A4/A5/A6 em um registro completo (todas as páginas acumuladas).
        Se A6 retornar score baixo e houver fallback A2 configurado (modo line),
        re-transcreve as linhas e repete A4/A5/A6. Máximo 1 retry.
        """
        if collection_config.record_template:
            logger.info("A0: [A4] corrigindo registro %d com template...", record.id)
            self._a4.correct(record, collection_config)

        logger.info("A0: [A5] extraindo campos do registro %d...", record.id)
        self._a5.extract(record, collection_config)
        logger.info("A0: [A5] registro %d → %s", record.id, record.structured_output)

        self._a6.validate(record, collection_config)
        logger.info(
            "A0: [A6] registro %d — score=%.2f verdict=%s",
            record.id,
            record.validation.score if record.validation else 0.0,
            record.validation.verdict if record.validation else "?",
        )

        # Retry com fallback A2 (apenas modo line — page mode não tem linhas individuais)
        if self._a2_fallback and record.lines:
            val = record.validation
            if val and val.score < self._fallback_threshold:
                logger.info(
                    "A0: [RETRY] registro %d score=%.2f < %.2f — retranscrevendo com fallback",
                    record.id, val.score, self._fallback_threshold,
                )
                t_retry = time.time()
                for line in record.lines.values():
                    self._a2_fallback.transcribe_line(line)
                record.corrected_text = ""
                record.structured_output = {}
                if collection_config.record_template:
                    self._a4.correct(record, collection_config)
                self._a5.extract(record, collection_config)
                self._a6.validate(record, collection_config)
                logger.info(
                    "A0: [RETRY] registro %d → score=%.2f verdict=%s (%.1fs)",
                    record.id,
                    record.validation.score if record.validation else 0.0,
                    record.validation.verdict if record.validation else "?",
                    time.time() - t_retry,
                )

    # ------------------------------------------------------------------
    # Interface LangGraph (Semana 5)
    # ------------------------------------------------------------------

    def __call__(self, state: dict) -> dict:
        """
        Interface compatível com nós LangGraph.
        Estado espera: {"collection_input": CollectionInput}
        Estado retorna: {"collection": Collection}
        """
        collection_input = state["collection_input"]
        collection = self.run(collection_input)
        return {**state, "collection": collection}
