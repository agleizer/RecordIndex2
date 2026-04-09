"""
A4 — Correção Estrutural.

Tipo de agente: Model-Based Reflex.
O "modelo interno" é o template de texto da coleção (record_template em CollectionConfig).
A4 só é chamado quando CollectionConfig.record_template está preenchido; caso contrário,
o A0 pula esta etapa.

Input:  Record com linhas transcritas por A2 + CollectionConfig com record_template
Output: Record.corrected_text preenchido com o texto corrigido pelo template

Estratégia: envia o texto concatenado do registro + o template ao LLM.
O LLM preenche os placeholders do template com os dados extraídos do texto HTR,
preservando o texto fixo do molde.

Placeholders no template seguem o formato <NOME> (ex: <DIA>, <NOME_PAI>).
Trechos opcionais seguem <OPT>...</OPT>.

Por que Model-Based Reflex:
  O agente consulta um modelo interno (o template) antes de agir — o mesmo texto
  é tratado de forma diferente dependendo do template da coleção. Não há goal
  de otimização (não é Goal-Based), apenas uma percepção → ação mediada pelo modelo.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.llm_client import disable_think
from src.models.collection_config import CollectionConfig
from src.models.record import Record
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.a4")


class A4CorrectionAgent:

    def __init__(self, model: BaseChatModel):
        self._model = disable_think(model)
        self._prompt_template = get_prompt("a4", "correct")

    def correct(self, record: Record, collection_config: CollectionConfig) -> str:
        """
        Aplica o template de correção ao texto concatenado do registro.

        Preenche record.corrected_text in-place e retorna o texto corrigido.
        Se collection_config.record_template estiver vazio, retorna '' sem chamar o LLM.
        """
        if not collection_config.record_template:
            return ""

        raw_text = " ".join(
            line.htr_text for line in record.lines.values() if line.htr_text
        )
        if not raw_text.strip():
            logger.warning("A4: registro %d sem texto HTR — correção pulada", record.id)
            return ""

        prompt = self._prompt_template.format(
            record_template=collection_config.record_template,
            text=raw_text,
        )
        result = self._model.invoke([HumanMessage(content=prompt)])
        corrected = result.content.strip()

        record.corrected_text = corrected
        logger.info("A4: registro %d corrigido (%d chars)", record.id, len(corrected))
        return corrected

    def __call__(self, state: dict) -> dict:
        record = state["current_record"]
        collection_config = state["collection_config"]
        self.correct(record, collection_config)
        return {**state, "current_record": record}
