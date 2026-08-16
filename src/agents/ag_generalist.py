"""
AG — Agente Generalista (HTR + segmentação + extração em um único prompt).

Não faz parte do pipeline MAS (A0-A6). É um modo de pipeline alternativo,
usado para comparar decomposição em agentes especializados vs. um único
agente multimodal fazendo leitura, segmentação e extração numa chamada só
por página. Ver `2026_08_16_modulo_generalista_vs_mas/PLANO_IMPLEMENTACAO.md`.

Reaproveita o mesmo contrato de CollectionConfig do A5 (schema Pydantic
dinâmico via extraction_fields) e o mesmo padrão de mensagem multimodal
do A2 (imagem em base64 + prompt).

Saída por página:
  leading_continuation_text — texto no topo da página que não começa um
    novo registro (pode ser continuação de um registro aberto na página
    anterior). Vazio se a página começa direto com um novo registro.
  records — novos registros detectados nesta página. O último pode ter
    complete=False (cortado pela borda inferior da página).
"""

import base64
import json
import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, create_model

from src.llm_client import disable_think, make_structured
from src.models.collection_config import CollectionConfig
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.ag")

_MEDIA_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".tif": "jpeg", ".tiff": "jpeg",
}


class AGGeneralistAgent:

    def __init__(self, model: BaseChatModel):
        self._base_model = disable_think(model)
        self._fallback_model = self._base_model
        # key -> (chain, schema_cls, record_field_names) — record_field_names cacheado
        # à parte pra não depender de introspecção do annotation no fallback.
        self._chain_cache: dict[str, tuple[object, type, list[str]]] = {}
        self._prompt_template = get_prompt("generalist", "single_prompt")

    # ------------------------------------------------------------------
    # Schema dinâmico (mesmo padrão do A5NERAgent._get_chain)
    # ------------------------------------------------------------------

    def _get_chain(self, collection_config: CollectionConfig):
        key = collection_config.collection_type
        if key not in self._chain_cache:
            record_fields: dict = {
                "raw_text": (str, ...),
                "complete": (bool, True),
            }
            for field_name, field_type in collection_config.extraction_fields.items():
                if field_type is str:
                    record_fields[field_name] = (str, "")
                else:
                    record_fields[field_name] = (field_type, ...)

            GeneralistRecord = create_model(f"GeneralistRecord_{key}", **record_fields)

            class GeneralistPageExtraction(BaseModel):
                leading_continuation_text: str = ""
                records: list[GeneralistRecord] = []
                reasoning: str = ""

            chain = make_structured(self._base_model, GeneralistPageExtraction)
            self._chain_cache[key] = (chain, GeneralistPageExtraction, list(record_fields.keys()))
        return self._chain_cache[key]

    def _build_prompt(self, collection_config: CollectionConfig) -> str:
        field_list = "\n".join(
            f"- {name}: {collection_config.field_descriptions.get(name, name)}"
            for name in collection_config.extraction_fields
        )
        if collection_config.record_template:
            template_section = (
                "O molde abaixo representa a estrutura completa de um registro desta "
                "coleção. Use-o como referência para entender quais campos e que fórmula "
                f"de fechamento um registro completo deve ter:\n\n{collection_config.record_template}\n"
            )
        else:
            template_section = ""

        return self._prompt_template.format(
            collection_name=collection_config.collection_name,
            collection_type=collection_config.collection_type,
            record_start_hint=collection_config.record_start_hint,
            template_section=template_section,
            field_list=field_list,
        )

    def _build_message(self, image_path: str, prompt: str) -> HumanMessage:
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ])

    # ------------------------------------------------------------------
    # Fallback (mesmo espírito do A3/A5 — modelos que ignoram json_schema)
    # ------------------------------------------------------------------

    def _fallback_extract_page(self, image_path: str, prompt: str, schema_cls: type, record_fields: list[str]):
        example_record = {f: ("" if f != "complete" else True) for f in record_fields}
        retry_prompt = (
            prompt
            + "\n\n---\n"
            "IMPORTANTE: Responda APENAS com JSON válido, sem texto antes ou depois. "
            "Formato obrigatório:\n"
            + json.dumps(
                {
                    "leading_continuation_text": "",
                    "records": [example_record],
                    "reasoning": "explicação resumida",
                },
                ensure_ascii=False,
            )
        )
        message = self._build_message(image_path, retry_prompt)
        raw = self._fallback_model.invoke([message])
        text = raw.content if hasattr(raw, "content") else str(raw)
        logger.debug("AG fallback raw response: %s", text[:400])

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                result = schema_cls.model_validate(data)
                logger.warning("AG fallback OK: %d registro(s)", len(result.records))
                return result
            except Exception:
                pass

        logger.warning("AG fallback falhou — retornando página sem registros. Texto: %s", text[:200])
        return schema_cls(
            leading_continuation_text="",
            records=[],
            reasoning=f"Fallback falhou. Texto bruto: {text[:300]}",
        )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def extract_page(self, image_path: str, collection_config: CollectionConfig):
        """
        Executa leitura + segmentação + extração de uma página inteira numa
        única chamada multimodal. Retorna uma instância de
        GeneralistPageExtraction (schema dinâmico por collection_type).
        """
        chain, schema_cls, record_fields = self._get_chain(collection_config)
        prompt = self._build_prompt(collection_config)
        message = self._build_message(image_path, prompt)

        logger.info("AG: processando página '%s'...", Path(image_path).name)
        try:
            result = chain.invoke([message])
        except Exception as e:
            logger.warning(
                "AG: structured output falhou (%s: %s) — ativando fallback",
                type(e).__name__, str(e)[:120],
            )
            result = self._fallback_extract_page(image_path, prompt, schema_cls, record_fields)

        logger.info(
            "AG: '%s' → %d registro(s), continuação=%s",
            Path(image_path).name, len(result.records), bool(result.leading_continuation_text.strip()),
        )
        return result
