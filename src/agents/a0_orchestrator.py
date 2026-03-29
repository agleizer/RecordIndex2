"""
A0 — Orquestrador.

Tipo de agente: Goal-Based.
Objetivo: coordenar o pipeline completo para uma coleção, chamando A2 → A3 → A5
em sequência e retornando uma Collection estruturada.

Esta implementação é Python pura — sem LangGraph. A interface __call__(state: dict)
é compatível com nós LangGraph para migração na Semana 5 sem refatoração.

Fluxo por página:
  1. Para cada linha da página: A2.transcribe_line()
  2. A3.segment() → agrupa linhas em Records
  3. Para cada Record: A5.extract() → preenche structured_output

O feedback loop A6→A0→A3/A4 (Semana 5) não está implementado aqui.
"""

import logging

from src.config import Config
from src.llm_client import get_chat_model
from src.agents.a2_htr import A2HTRAgent
from src.agents.a3_segmentation import A3RecordSegmentationAgent
from src.agents.a5_ner import A5NERAgent
from src.models.collection import Collection
from src.models.collection_config import CollectionConfig
from src.models.page import Page
from src.models.record import Record

logger = logging.getLogger("recordindex.a0")


class A0Orchestrator:

    def __init__(self, config: Config, collection_config: CollectionConfig):
        self._collection_config = collection_config

        model_a2 = get_chat_model(config.a2_provider, config.a2_model, config.ollama_base_url)
        model_a3 = get_chat_model(config.a3_provider, config.a3_model, config.ollama_base_url)
        model_a5 = get_chat_model(config.a5_provider, config.a5_model, config.ollama_base_url)

        self._a2 = A2HTRAgent(model_a2)
        self._a3 = A3RecordSegmentationAgent(model_a3)
        self._a5 = A5NERAgent(model_a5)

    def process_page(self, page: Page) -> list[Record]:
        """
        Processa uma página completa: HTR → segmentação → extração.

        Assume que page.lines já está populado com imagens de linha (image_path).
        Retorna lista de Records com structured_output preenchido.
        """
        logger.info("A0: processando página '%s' (%d linhas)", page.filename, len(page.lines))

        # A2: transcrever cada linha
        for line in page.lines:
            logger.info("  A2: transcrevendo linha %s", line.id)
            self._a2.transcribe_line(line)
            logger.info("  A2: '%s'", line.htr_text)

        # A3: segmentar linhas em registros
        logger.info("A0: segmentando registros (A3)...")
        records = self._a3.segment(page.lines, self._collection_config, page.filename)
        logger.info("A0: %d registros identificados", len(records))

        # A5: extrair campos de cada registro
        for record in records:
            logger.info("  A5: extraindo campos do registro %d", record.id)
            self._a5.extract(record, self._collection_config)
            logger.info("  A5: %s", record.structured_output)

        return records

    def process_collection(self, pages: list[Page]) -> Collection:
        """
        Processa uma lista de páginas e retorna uma Collection completa.
        """
        collection = Collection(name=self._collection_config.collection_name)
        for page in pages:
            collection.add_page(page)
            records = self.process_page(page)
            for record in records:
                collection.add_record(record)
        return collection

    def __call__(self, state: dict) -> dict:
        pages = state["pages"]
        collection = self.process_collection(pages)
        return {**state, "collection": collection}
