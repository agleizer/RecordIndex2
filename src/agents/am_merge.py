"""
AM — Agente de Merge (completa registro cortado pela borda entre páginas).

Não faz parte do pipeline MAS (A0-A6). Só é acionado pelo AGOrchestrator
(modo generalista) quando a AG sinaliza leading_continuation_text não vazio
no topo de uma página, ou seja, quando pode haver um registro aberto da
página anterior continuando ali.

Tratado como condição experimental, não como mecanismo definitivo: mesclar
duas extrações estruturadas via LLM é uma operação que o A0 (pipeline MAS)
nunca precisa fazer, ele acumula texto bruto e extrai uma vez só, quando o
registro fecha. Ver IDEIA_FLAG_COMPLETE.md em
`2026_08_09_teste_mas_vs_single_prompt/` para a análise de risco desse
mecanismo, e `2026_08_16_modulo_generalista_vs_mas/PLANO_IMPLEMENTACAO.md`
para o desenho experimental que compara os dois.
"""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.llm_client import disable_think, make_structured
from src.models.collection_config import CollectionConfig
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.am")


class MergeDecision(BaseModel):
    is_continuation: bool
    merged_fields: dict[str, str]
    merged_raw_text: str
    reasoning: str = ""


class AMMergeAgent:

    def __init__(self, model: BaseChatModel):
        self._base_model = disable_think(model)
        self._chain = make_structured(self._base_model, MergeDecision)
        self._prompt_template = get_prompt("generalist", "merge")

    def _build_prompt(
        self,
        fragment_a_text: str,
        fragment_a_fields: dict,
        fragment_b_text: str,
        collection_config: CollectionConfig,
    ) -> str:
        field_list = "\n".join(
            f"- {name}: {collection_config.field_descriptions.get(name, name)}"
            for name in collection_config.extraction_fields
        )
        return self._prompt_template.format(
            collection_type=collection_config.collection_type,
            fragment_a_text=fragment_a_text,
            fragment_a_fields=json.dumps(fragment_a_fields, ensure_ascii=False),
            fragment_b_text=fragment_b_text,
            field_list=field_list,
        )

    def _fallback_merge(
        self, prompt: str, fragment_a_text: str, fragment_a_fields: dict
    ) -> MergeDecision:
        """
        Invocado quando structured output falha. Reinvoca com instrução
        explícita de JSON. Se ainda assim falhar, assume is_continuation=False
        (mais seguro: mantém o Fragmento A como estava, sem arriscar mesclar
        errado) e mantém os campos originais.
        """
        retry_prompt = (
            prompt
            + "\n\n---\n"
            "IMPORTANTE: Responda APENAS com JSON válido, sem texto antes ou depois. "
            "Formato obrigatório:\n"
            '{"is_continuation": true|false, "merged_fields": {"campo": "valor", ...}, '
            '"merged_raw_text": "...", "reasoning": "explicação resumida"}'
        )
        raw = self._base_model.invoke([HumanMessage(content=retry_prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        logger.debug("AM fallback raw response: %s", text[:400])

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                decision = MergeDecision.model_validate(data)
                logger.warning("AM fallback OK: is_continuation=%s", decision.is_continuation)
                return decision
            except Exception:
                pass

        logger.warning(
            "AM fallback falhou — assumindo is_continuation=False (mantém fragmento A). Texto: %s",
            text[:200],
        )
        return MergeDecision(
            is_continuation=False,
            merged_fields=fragment_a_fields,
            merged_raw_text=fragment_a_text,
            reasoning=f"Fallback falhou, merge não aplicado. Texto bruto: {text[:300]}",
        )

    def merge(
        self,
        fragment_a_text: str,
        fragment_a_fields: dict,
        fragment_b_text: str,
        collection_config: CollectionConfig,
    ) -> MergeDecision:
        """
        Decide se fragment_b_text (texto do topo da página seguinte) é
        continuação do registro representado por fragment_a_text/fields e,
        se for, produz os campos e o texto mesclados.
        """
        prompt = self._build_prompt(
            fragment_a_text, fragment_a_fields, fragment_b_text, collection_config
        )
        try:
            decision = self._chain.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            logger.warning(
                "AM: structured output falhou (%s: %s) — ativando fallback",
                type(e).__name__, str(e)[:120],
            )
            decision = self._fallback_merge(prompt, fragment_a_text, fragment_a_fields)

        logger.info(
            "AM: is_continuation=%s reasoning=%s",
            decision.is_continuation, decision.reasoning[:120],
        )
        return decision
