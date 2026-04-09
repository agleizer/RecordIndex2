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
     a. A1.segment_page()          → Page com Lines (crops + bboxes)
     b. _process_page(page, cfg)   → list[Record]
        i.  A2.transcribe_line()   por linha
        ii. A3.segment()           → agrupamento em Records
        iii.A5.extract()           por Record
  3. Retorna Collection completa.

Interface LangGraph (Semana 5):
  __call__(state: dict) → dict  — estado contém CollectionInput, retorna Collection.
  Cada etapa interna (classify / A1 / A2 / A3 / A5) será um nó do grafo.

Feedback loop A6→A0→A3 (Semana 5): não implementado aqui.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import Config
from src.llm_client import get_chat_model, make_structured
from src.agents.a1_line_segmentation import A1LineSegmentationAgent
from src.agents.a2_htr import A2HTRAgent
from src.agents.a3_segmentation import A3RecordSegmentationAgent
from src.agents.a4_correction import A4CorrectionAgent
from src.agents.a5_ner import A5NERAgent
from src.models.collection import Collection
from src.models.collection_config import CollectionConfig
from src.models.collection_input import CollectionInput
from src.models.page import Page
from src.models.record import Record

logger = logging.getLogger("recordindex.a0")

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

        1. Classifica o tipo da coleção (batismo/casamento/obito).
        2. Para cada página: A1 → A2 → A3 → A5.
        3. Retorna Collection com Records e structured_output preenchidos.
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

        for image_path in image_files:
            logger.info("A0: processando página '%s'", image_path.name)
            page = self._a1.segment_page(str(image_path), str(lines_dir))
            collection.add_page(page)

            records = self._process_page(page, collection_config)
            for record in records:
                collection.add_record(record)

        logger.info(
            "A0: pipeline concluído — %d páginas, %d registros",
            len(collection.pages),
            len(collection.records),
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

        # Selecionar até 3 páginas aleatórias
        sample_size = min(3, len(image_files))
        sampled_pages = random.sample(image_files, sample_size)

        tmp_lines_dir = tempfile.mkdtemp(prefix="a0_classify_")
        sample_lines_text = []
        try:
            for page_path in sampled_pages:
                page = self._a1.segment_page(str(page_path), tmp_lines_dir)
                lines = list(page.lines.values())
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
        # Sobrescrever hints se o usuário forneceu valores explícitos
        if collection_input.record_start_hint:
            cfg.record_start_hint = collection_input.record_start_hint
        if collection_input.record_template:
            cfg.record_template = collection_input.record_template
        return cfg

    # ------------------------------------------------------------------
    # Processamento de página (A2 → A3 → A5)
    # ------------------------------------------------------------------

    def _process_page(self, page: Page, collection_config: CollectionConfig) -> list[Record]:
        """
        Executa A2 → A3 → A5 sobre uma página já segmentada por A1.
        Retorna lista de Records com structured_output preenchido.
        """
        logger.info("A0: processando página '%s' (%d linhas)", page.filename, len(page.lines))
        lines = list(page.lines.values())

        # A2: transcrever cada linha
        for line in lines:
            logger.info("  A2: linha %s", line.id)
            self._a2.transcribe_line(line)
            logger.info("  A2: '%s'", line.htr_text)

        # A3: segmentar linhas em registros
        logger.info("A0: A3 segmentando...")
        records = self._a3.segment(lines, collection_config, page.filename)
        logger.info("A0: %d registros identificados", len(records))

        # A4: corrigir texto por template (opcional — pulado se record_template vazio)
        if collection_config.record_template:
            logger.info("A0: A4 corrigindo registros com template...")
            for record in records:
                self._a4.correct(record, collection_config)

        # A5: extrair campos de cada registro
        for record in records:
            logger.info("  A5: registro %d", record.id)
            self._a5.extract(record, collection_config)
            logger.info("  A5: %s", record.structured_output)

        return records

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
