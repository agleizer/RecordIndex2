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
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.llm_client import disable_think
from src.models.collection_config import CollectionConfig
from src.models.record import Record
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.a4")

# Padrão de comentários adicionados pelo LLM (DT-10).
# llama3.2 frequentemente adiciona notas após o texto corrigido, ignorando a instrução
# "Retorne APENAS o texto corrigido". Truncamos a partir dessas marcas.
_COMMENTARY_RE = re.compile(
    r'\n+\s*(?:nota|observa[çc][aã]o|coment[aá]rio|obs\.?)\s*[:–-].*$',
    re.IGNORECASE | re.DOTALL,
)

# Placeholders não preenchidos pelo LLM: <CAMPO> ou <CAMPO_COMPOSTO>
_PLACEHOLDER_RE = re.compile(r'<[A-Z][A-Z_]*>')


def _strip_commentary(text: str) -> str:
    """Remove notas e comentários adicionados pelo LLM após o texto corrigido."""
    return _COMMENTARY_RE.sub("", text).strip()


def _has_unfilled_placeholders(text: str) -> bool:
    """Retorna True se o texto ainda contém placeholders <CAMPO> não preenchidos."""
    return bool(_PLACEHOLDER_RE.search(text))


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

        # get_concatenated_text() sabe sobre page_text (modo page) e line concat (modo line)
        raw_text = record.get_concatenated_text()
        if not raw_text.strip():
            logger.warning("A4: registro %d sem texto — correção pulada", record.id)
            return ""

        prompt = self._prompt_template.format(
            record_template=collection_config.record_template,
            text=raw_text,
        )
        result = self._model.invoke([HumanMessage(content=prompt)])
        corrected = _strip_commentary(result.content)

        if len(corrected) < len(result.content.strip()):
            logger.debug(
                "A4: registro %d — comentário removido (%d→%d chars)",
                record.id, len(result.content.strip()), len(corrected),
            )

        if _has_unfilled_placeholders(corrected):
            logger.warning(
                "A4: registro %d — LLM não preencheu os placeholders; corrected_text descartado",
                record.id,
            )
            return ""

        record.corrected_text = corrected
        logger.info("A4: registro %d corrigido (%d chars)", record.id, len(corrected))
        return corrected

    def __call__(self, state: dict) -> dict:
        record = state["current_record"]
        collection_config = state["collection_config"]
        self.correct(record, collection_config)
        return {**state, "current_record": record}
