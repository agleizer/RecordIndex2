"""
AGOrchestrator — orquestrador mínimo do modo generalista.

Modo de pipeline alternativo ao MAS completo (A0-A6). Não altera nem chama
A1/A2/A3/A4/A5 — cada página é processada por uma única chamada ao agente
generalista (AG), que faz leitura + segmentação + extração de uma vez.
A6 (validação) é reaproveitado sem alteração, só pra manter score/verdict
comparável entre os dois modos na avaliação.

Estado entre páginas (equivalente ao open_record do A0, ver
`a0_orchestrator.py::run`, mas mais simples): um único registro pode ficar
aberto entre páginas quando a AG marca complete=False no último registro
da página, ou quando ela sinaliza leading_continuation_text no topo da
página seguinte. Nesse segundo caso, o AM decide se e como mesclar.

Classificação da coleção (tipo + record_start_hint) reaproveita
A0Orchestrator._classify() por composição — não duplica essa lógica aqui,
e não altera nada em a0_orchestrator.py. A1-A6 internos desse A0Orchestrator
auxiliar não são usados fora da classificação.

Ver `2026_08_16_modulo_generalista_vs_mas/PLANO_IMPLEMENTACAO.md`.
"""

import logging
import time
from pathlib import Path

from src.config import Config
from src.llm_client import get_chat_model
from src.agents.a0_orchestrator import A0Orchestrator, SUPPORTED_EXTENSIONS
from src.agents.a6_validation import A6ValidationAgent
from src.agents.ag_generalist import AGGeneralistAgent
from src.agents.am_merge import AMMergeAgent
from src.models.collection import Collection
from src.models.collection_config import CollectionConfig
from src.models.collection_input import CollectionInput
from src.models.line import Line
from src.models.page import Page
from src.models.record import Record

logger = logging.getLogger("recordindex.ag_orchestrator")


class AGOrchestrator:

    def __init__(self, config: Config):
        self._config = config

        # Reaproveita só a etapa de classificação do A0 (_classify), por composição.
        # Não altera a0_orchestrator.py — A1-A6 deste helper não são usados aqui.
        self._classify_helper = A0Orchestrator(config)

        self._ag = AGGeneralistAgent(
            get_chat_model(config.ag_provider, config.ag_model, config.ollama_base_url)
        )
        self._am = AMMergeAgent(
            get_chat_model(config.am_provider, config.am_model, config.ollama_base_url)
        )
        self._a6 = A6ValidationAgent(
            get_chat_model(config.a6_provider, config.a6_model, config.ollama_base_url),
            llm_threshold=config.a6_llm_threshold,
            num_predict=config.a6_num_predict,
        )

        self._record_counter = 0

    def _next_record_id(self) -> int:
        self._record_counter += 1
        return self._record_counter

    def _to_record(self, gen_record, page_filename: str) -> Record:
        record = Record(id=self._next_record_id(), page_filename=page_filename)
        record.page_text = gen_record.raw_text
        record.structured_output = {
            k: str(v) for k, v in gen_record.model_dump(exclude={"raw_text", "complete"}).items()
        }
        return record

    @staticmethod
    def _track_page(record: Record, page_filename: str) -> None:
        """
        Bookkeeping puro — permite Record.last_page_filename() saber que um
        registro mesclado atravessou páginas, mesmo sem granularidade de
        linha (modo generalista não popula Record.lines com linhas reais).
        Não entra no texto concatenado: get_concatenated_text() prioriza
        page_text sobre a concatenação de lines.
        """
        marker_id = f"__page_marker__{page_filename}"
        if marker_id not in record.lines:
            record.lines[marker_id] = Line(id=marker_id, image_path="", page_filename=page_filename)

    def _finalize(self, record: Record, collection_config: CollectionConfig) -> Record:
        self._a6.validate(record, collection_config)
        logger.info(
            "AGOrchestrator: registro %d finalizado — score=%.2f verdict=%s",
            record.id,
            record.validation.score if record.validation else 0.0,
            record.validation.verdict if record.validation else "?",
        )
        return record

    def run(self, collection_input: CollectionInput) -> Collection:
        logger.info(
            "AGOrchestrator: iniciando pipeline generalista para '%s'",
            collection_input.collection_name,
        )

        collection_config: CollectionConfig = self._classify_helper._classify(collection_input)
        logger.info(
            "AGOrchestrator: coleção classificada como '%s' (hint_início='%s')",
            collection_config.collection_type,
            collection_config.record_start_hint,
        )

        image_files = sorted(
            p for p in Path(collection_input.image_dir).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_files:
            logger.warning("AGOrchestrator: nenhuma imagem encontrada em '%s'", collection_input.image_dir)
            return Collection(name=collection_config.collection_name)

        collection = Collection(name=collection_config.collection_name)
        total_pages = len(image_files)
        pipeline_start = time.time()

        open_record: Record | None = None

        for page_idx, image_path in enumerate(image_files, start=1):
            logger.info("AGOrchestrator: ── PÁGINA %d/%d ── '%s'", page_idx, total_pages, image_path.name)
            t0 = time.time()

            page = Page(filename=image_path.stem, image_path=str(image_path))
            collection.add_page(page)

            extraction = self._ag.extract_page(str(image_path), collection_config)

            if open_record is not None and extraction.leading_continuation_text.strip():
                logger.info(
                    "AGOrchestrator: [AM] texto de continuação encontrado — decidindo merge com registro %d...",
                    open_record.id,
                )
                decision = self._am.merge(
                    open_record.page_text,
                    open_record.structured_output,
                    extraction.leading_continuation_text,
                    collection_config,
                )
                if decision.is_continuation:
                    open_record.page_text = decision.merged_raw_text
                    open_record.structured_output = decision.merged_fields
                    self._track_page(open_record, page.filename)
                    logger.info("AGOrchestrator: [AM] merge aplicado ao registro %d", open_record.id)
                else:
                    logger.info(
                        "AGOrchestrator: [AM] não é continuação — registro %d mantido como estava",
                        open_record.id,
                    )

            if not extraction.records:
                logger.info(
                    "AGOrchestrator: página %d sem novos registros (continuação ou frontispício)",
                    page_idx,
                )
                logger.info("AGOrchestrator: ── PÁGINA %d/%d concluída (%.1fs) ──", page_idx, total_pages, time.time() - t0)
                continue

            to_finalize = []
            if open_record is not None:
                to_finalize.append(open_record)
            to_finalize.extend(
                self._to_record(r, page.filename) for r in extraction.records[:-1]
            )
            for record in to_finalize:
                collection.add_record(self._finalize(record, collection_config))

            last = extraction.records[-1]
            candidate = self._to_record(last, page.filename)
            if last.complete:
                collection.add_record(self._finalize(candidate, collection_config))
                open_record = None
            else:
                logger.info(
                    "AGOrchestrator: registro %d marcado complete=False — mantido aberto",
                    candidate.id,
                )
                open_record = candidate

            logger.info("AGOrchestrator: ── PÁGINA %d/%d concluída (%.1fs) ──", page_idx, total_pages, time.time() - t0)

        if open_record is not None:
            collection.add_record(self._finalize(open_record, collection_config))

        elapsed = time.time() - pipeline_start
        logger.info(
            "AGOrchestrator: ══ PIPELINE CONCLUÍDO ══ %d páginas | %d registros | %.1fs total (%.1fs/pág)",
            len(collection.pages), len(collection.records),
            elapsed, elapsed / max(total_pages, 1),
        )
        return collection
