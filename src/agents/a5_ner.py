"""
A5 — NER / Extração de Entidades.

Tipo de agente: Goal-Based.
Objetivo: extrair campos estruturados de um registro genealógico (nome, pai, mãe, data, etc.)

Input:  Record com linhas transcritas + CollectionConfig com campos esperados
Output: dict com campos extraídos (preenchido em record.structured_output)

Estratégia: schema Pydantic gerado dinamicamente via pydantic.create_model()
a partir de collection_config.extraction_fields. Isso permite que A5 funcione
com qualquer tipo de coleção sem alterar o código.

.bind(think=False) antes de with_structured_output — obrigatório para
modelos Qwen3.5 (thinking mode + structured output são incompatíveis).
A chain é construída por chamada (não no __init__) porque o schema varia
conforme o collection_type.
"""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import create_model

from src.llm_client import disable_think, make_structured
from src.models.collection_config import CollectionConfig
from src.models.record import Record
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.a5")


class A5NERAgent:

    MIN_TEXT_LEN = 20  # caracteres mínimos após limpeza para tentar extração
    _NOISE_TOKENS = {"[?]", "[...]", "?", "...", "[ILEGÍVEL]"}

    def __init__(self, model: BaseChatModel):
        # Modelo base — chain construída dinamicamente por collection_type
        self._base_model = model
        self._fallback_model = disable_think(model)  # usado no fallback — thinking desabilitado
        self._chain_cache: dict[str, object] = {}
        self._prompt_template = get_prompt("a5", "extract")

    def _get_chain(self, collection_config: CollectionConfig):
        """
        Retorna (e cacheia) a chain para o collection_type dado.

        O schema é construído uma vez por collection_type via pydantic.create_model().
        Campos str são opcionais (default ""); outros tipos são obrigatórios.
        """
        key = collection_config.collection_type
        if key not in self._chain_cache:
            fields = {}
            for field_name, field_type in collection_config.extraction_fields.items():
                if field_type is str:
                    fields[field_name] = (str, "")
                else:
                    fields[field_name] = (field_type, ...)

            NEROutput = create_model(f"NEROutput_{key}", **fields)
            self._chain_cache[key] = make_structured(self._base_model, NEROutput)
        return self._chain_cache[key]

    def _is_extractable(self, text: str) -> bool:
        """Retorna False se o texto for vazio, só ruído HTR, ou curto demais para extrair."""
        cleaned = text.strip()
        for token in self._NOISE_TOKENS:
            cleaned = cleaned.replace(token, "")
        return len(cleaned.strip()) >= self.MIN_TEXT_LEN

    def _build_prompt(self, collection_config: CollectionConfig, text: str) -> str:
        field_list = "\n".join(
            f"- {name}: {collection_config.field_descriptions.get(name, name)}"
            for name in collection_config.extraction_fields
        )
        return self._prompt_template.format(
            collection_type=collection_config.collection_type,
            record_text=text,
            field_list=field_list,
        )

    def _fallback_extract(
        self, prompt: str, collection_config: CollectionConfig
    ) -> dict:
        """
        Invocado quando structured output falha (ex: gemma retorna null ou markdown).
        Reinvoca o modelo com instrução explícita de JSON e tenta extrair os campos.
        """
        fields = list(collection_config.extraction_fields.keys())
        empty_example = {f: "" for f in fields}
        retry_prompt = (
            prompt
            + "\n\n---\n"
            "IMPORTANTE: Responda APENAS com JSON válido, sem texto antes ou depois. "
            f"Formato obrigatório (todos os campos são strings):\n{json.dumps(empty_example, ensure_ascii=False)}"
        )
        raw = self._fallback_model.invoke([HumanMessage(content=retry_prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        logger.debug("A5 fallback raw response: %s", text[:400])

        # Tenta extrair o primeiro objeto JSON do texto
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                result = {f: str(data.get(f, "")) for f in fields}
                logger.warning("A5 fallback OK: campos=%s", list(result.keys()))
                return result
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("A5 fallback falhou — retornando campos vazios. Texto: %s", text[:200])
        return {f: "" for f in fields}

    def extract(self, record: Record, collection_config: CollectionConfig) -> dict:
        """
        Extrai campos estruturados do registro.

        Retorna dict com os campos definidos em collection_config.extraction_fields.
        Também preenche record.structured_output in-place.
        """
        chain = self._get_chain(collection_config)
        text = record.get_concatenated_text()
        if not self._is_extractable(text):
            result = {k: "" for k in collection_config.extraction_fields}
            record.structured_output = result
            return result

        prompt = self._build_prompt(collection_config, text)
        try:
            output = chain.invoke([HumanMessage(content=prompt)])
            result = output.model_dump()
        except Exception as e:
            logger.warning(
                "A5 structured output falhou (%s: %s) — ativando fallback",
                type(e).__name__, str(e)[:120],
            )
            result = self._fallback_extract(prompt, collection_config)

        record.structured_output = result
        return result

    def __call__(self, state: dict) -> dict:
        record = state["current_record"]
        collection_config = state["collection_config"]
        self.extract(record, collection_config)
        return {**state, "current_record": record}
