"""
A3 — Segmentação de Registros.

Tipo de agente: Goal-Based.
Objetivo: agrupar linhas transcritas em registros genealógicos individuais.

Dois modos de operação (htr_scope):

  line (padrão): recebe lista de Line com htr_text individual.
    O LLM detecta índices de início de cada registro (RecordBoundaries).
    O agente constrói Records agrupando as linhas pelo índice.
    Input → Output: list[Line] → list[Record com lines preenchidas]

  page: recebe o texto completo da página como bloco único.
    O LLM divide o texto nos registros que ele contém (PageRecordBlocks).
    O agente constrói Records com record.page_text — sem granularidade de linha.
    Input → Output: str → list[Record com page_text preenchido]

Métodos públicos:
  segment()            — modo line: list[Line] → list[Record com lines]
  detect_boundaries()  — modo line: expõe RecordBoundaries (índices + reasoning) sem construir Records
  segment_page()       — modo page: str → list[Record com page_text]
  detect_page_blocks() — modo page: expõe PageRecordBlocks (texts + reasoning) sem construir Records

Nada no modo line é alterado pelo modo page.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.llm_client import make_structured
from src.models.collection_config import CollectionConfig
from src.models.line import Line
from src.models.record import Record
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.a3")


class RecordBoundaries(BaseModel):
    """
    Schema de saída de A3 modo LINE.

    record_start_indices: índices (0-based) que iniciam novos registros.
    Vazio se a página não contiver registros (frontispício, cabeçalhos).
    """
    record_start_indices: list[int]
    reasoning: str


class PageRecordBlocks(BaseModel):
    """
    Schema de saída de A3 modo PAGE.

    record_texts: texto completo de cada registro identificado na página,
    preservando a transcrição original. Vazio se a página não contiver registros.
    """
    record_texts: list[str]
    reasoning: str


class A3RecordSegmentationAgent:

    def __init__(self, model: BaseChatModel):
        # Modo LINE
        self._chain = make_structured(model, RecordBoundaries)
        self._prompt_template = get_prompt("a3", "segment")
        # Modo PAGE
        self._page_chain = make_structured(model, PageRecordBlocks)
        self._page_prompt_template = get_prompt("a3", "segment_page")

    def _build_prompt(
        self, lines: list[Line], config: CollectionConfig, avg_lines_hint: int | None = None
    ) -> str:
        numbered = "\n".join(
            f"{i}: {line.best_text or '[sem texto]'}"
            for i, line in enumerate(lines)
        )
        if config.record_template:
            template_section = (
                "O molde abaixo representa a estrutura completa de um registro desta coleção. "
                "Use-o como referência para entender quantos campos e aproximadamente quantas "
                f"linhas um registro completo deve ter:\n\n{config.record_template}\n"
            )
        else:
            template_section = ""

        if avg_lines_hint:
            avg_lines_hint_section = (
                f"Com base nos registros já processados desta coleção, um registro típico tem "
                f"aproximadamente {avg_lines_hint} linhas. Use isso como calibração — se um "
                f"bloco aparente de registro tiver muito mais linhas que isso, provavelmente "
                f"contém mais de um registro.\n"
            )
        else:
            avg_lines_hint_section = ""

        return self._prompt_template.format(
            collection_type=config.collection_type,
            collection_name=config.collection_name,
            record_start_hint=config.record_start_hint,
            template_section=template_section,
            avg_lines_hint_section=avg_lines_hint_section,
            last_index=len(lines) - 1,
            numbered_lines=numbered,
        )

    def detect_boundaries(
        self,
        lines: list[Line],
        collection_config: CollectionConfig,
        avg_lines_hint: int | None = None,
    ) -> RecordBoundaries:
        """
        Chama o LLM e retorna RecordBoundaries (índices + reasoning) sem construir Records.
        Útil quando o chamador precisa do reasoning para logging ou resposta de API.

        avg_lines_hint: número médio de linhas por registro observado até agora na coleção.
          Injetado no prompt para calibrar o LLM contra merge/split excessivos.
        """
        if not lines:
            return RecordBoundaries(record_start_indices=[], reasoning="Lista de linhas vazia.")
        prompt = self._build_prompt(lines, collection_config, avg_lines_hint)
        return self._chain.invoke([HumanMessage(content=prompt)])

    def segment(self, lines: list[Line], collection_config: CollectionConfig, page_filename: str = "") -> list[Record]:
        """
        Segmenta uma lista de linhas em Records.

        Chama o LLM uma vez com todas as linhas, obtém os índices de início
        de cada registro, e constrói os Records fatiando a lista.
        """
        if not lines:
            return []

        boundaries: RecordBoundaries = self.detect_boundaries(lines, collection_config, avg_lines_hint=None)

        # Linhas antes do primeiro índice são descartadas (frontispício, cabeçalhos, etc.)
        starts = sorted(set(i for i in boundaries.record_start_indices if 0 <= i < len(lines)))
        if not starts:
            return []

        records = []
        for record_idx, start in enumerate(starts):
            end = starts[record_idx + 1] if record_idx + 1 < len(starts) else len(lines)
            record = Record(
                id=record_idx,
                page_filename=page_filename,
            )
            for line in lines[start:end]:
                record.add_line(line)
            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Modo PAGE
    # ------------------------------------------------------------------

    def _build_page_prompt(
        self, page_text: str, config: CollectionConfig, avg_lines_hint: int | None = None
    ) -> str:
        if config.record_template:
            template_section = (
                "O molde abaixo representa a estrutura completa de um registro desta coleção. "
                "Use-o como referência para entender onde um registro termina e outro começa:"
                f"\n\n{config.record_template}\n"
            )
        else:
            template_section = ""

        if avg_lines_hint:
            avg_lines_hint_section = (
                f"Com base nos registros já processados desta coleção, um registro típico tem "
                f"aproximadamente {avg_lines_hint} linhas de texto. Use isso como calibração — "
                f"se um bloco aparente de registro tiver muito mais linhas que isso, provavelmente "
                f"contém mais de um registro.\n"
            )
        else:
            avg_lines_hint_section = ""

        return self._page_prompt_template.format(
            collection_type=config.collection_type,
            collection_name=config.collection_name,
            record_start_hint=config.record_start_hint,
            template_section=template_section,
            avg_lines_hint_section=avg_lines_hint_section,
            page_text=page_text,
        )

    def detect_page_blocks(
        self,
        page_text: str,
        collection_config: CollectionConfig,
        avg_lines_hint: int | None = None,
    ) -> PageRecordBlocks:
        """
        Chama o LLM e retorna PageRecordBlocks (record_texts + reasoning) sem construir Records.
        Útil quando o chamador precisa do reasoning para logging ou resposta de API.

        avg_lines_hint: número médio de linhas por registro observado até agora na coleção.
        """
        if not page_text.strip():
            return PageRecordBlocks(record_texts=[], reasoning="Texto vazio.")
        prompt = self._build_page_prompt(page_text, collection_config, avg_lines_hint)
        return self._page_chain.invoke([HumanMessage(content=prompt)])

    def segment_page(
        self,
        page_text: str,
        collection_config: CollectionConfig,
        page_filename: str = "",
    ) -> list[Record]:
        """
        Segmenta o texto completo de uma página em Records (modo PAGE).

        Envia o texto como bloco único ao LLM, que retorna os textos
        individuais de cada registro. Cada Record resultante tem page_text
        preenchido em vez de lines — não há granularidade de linha.

        Retorna lista vazia se a página não contiver registros.
        Não altera nada do modo line (segment()).
        """
        if not page_text.strip():
            return []

        blocks: PageRecordBlocks = self.detect_page_blocks(page_text, collection_config)
        logger.debug("A3 [page]: reasoning — %s", blocks.reasoning)

        records = []
        for i, text in enumerate(blocks.record_texts):
            text = text.strip()
            if not text:
                continue
            record = Record(id=i, page_filename=page_filename)
            record.page_text = text
            records.append(record)

        return records

    def __call__(self, state: dict) -> dict:
        lines = state["lines"]
        collection_config = state["collection_config"]
        page_filename = state.get("page_filename", "")
        records = self.segment(lines, collection_config, page_filename)
        return {**state, "records": records}
