"""
A3 — Segmentação de Registros.

Tipo de agente: Goal-Based.
Objetivo: agrupar linhas transcritas em registros genealógicos individuais.

Input:  lista de linhas (com htr_text preenchido por A2)
Output: lista de Records (cada um com suas linhas agrupadas)

Estratégia: envia todas as transcrições ao LLM com contexto da coleção
e pede os índices que iniciam novos registros (RecordBoundaries).
O LLM não agrupa — apenas detecta fronteiras. O agente faz o corte.

.bind(think=False) antes de with_structured_output — obrigatório para
modelos Qwen3.5 (thinking mode + structured output são incompatíveis).
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.llm_client import make_structured
from src.models.collection_config import CollectionConfig
from src.models.line import Line
from src.models.record import Record
from src.prompts import get_prompt


class RecordBoundaries(BaseModel):
    """
    Schema de saída de A3.

    record_start_indices: lista de índices (0-based) que iniciam um novo
    registro. Ex: [0, 5, 11] → registro 0 começa na linha 0, registro 1
    na linha 5, registro 2 na linha 11.
    O índice 0 está sempre incluído (toda lista começa com pelo menos 1 registro).

    reasoning: raciocínio do modelo (uma frase curta). Útil para debug
    e para justificar academicamente as decisões de segmentação.
    """

    record_start_indices: list[int]
    reasoning: str


class A3RecordSegmentationAgent:

    def __init__(self, model: BaseChatModel):
        self._chain = make_structured(model, RecordBoundaries)
        self._prompt_template = get_prompt("a3", "segment")

    def _build_prompt(self, lines: list[Line], config: CollectionConfig) -> str:
        numbered = "\n".join(
            f"{i}: {line.best_text or '[sem texto]'}"
            for i, line in enumerate(lines)
        )
        return self._prompt_template.format(
            collection_type=config.collection_type,
            collection_name=config.collection_name,
            record_start_hint=config.record_start_hint,
            last_index=len(lines) - 1,
            numbered_lines=numbered,
        )

    def detect_boundaries(self, lines: list[Line], collection_config: CollectionConfig) -> RecordBoundaries:
        """
        Chama o LLM e retorna RecordBoundaries (índices + reasoning) sem construir Records.
        Útil quando o chamador precisa do reasoning para logging ou resposta de API.
        """
        if not lines:
            return RecordBoundaries(record_start_indices=[0], reasoning="Lista de linhas vazia.")
        prompt = self._build_prompt(lines, collection_config)
        return self._chain.invoke([HumanMessage(content=prompt)])

    def segment(self, lines: list[Line], collection_config: CollectionConfig, page_filename: str = "") -> list[Record]:
        """
        Segmenta uma lista de linhas em Records.

        Chama o LLM uma vez com todas as linhas, obtém os índices de início
        de cada registro, e constrói os Records fatiando a lista.
        """
        if not lines:
            return []

        boundaries: RecordBoundaries = self.detect_boundaries(lines, collection_config)

        # Garante que 0 está sempre incluído e a lista está ordenada e sem duplicatas
        starts = sorted(set([0] + [i for i in boundaries.record_start_indices if 0 <= i < len(lines)]))

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

    def __call__(self, state: dict) -> dict:
        lines = state["lines"]
        collection_config = state["collection_config"]
        page_filename = state.get("page_filename", "")
        records = self.segment(lines, collection_config, page_filename)
        return {**state, "records": records}
