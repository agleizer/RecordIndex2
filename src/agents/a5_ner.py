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

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import create_model

from src.llm_client import make_structured
from src.models.collection_config import CollectionConfig
from src.models.record import Record


class A5NERAgent:

    PROMPT_TEMPLATE = (
        "Você está extraindo informações de um registro de {collection_type} de manuscrito "
        "histórico em português brasileiro.\n\n"
        "Texto do registro:\n{record_text}\n\n"
        "Extraia os seguintes campos:\n{field_list}\n\n"
        "Se um campo não estiver presente no texto, use string vazia. "
        "Preserve os nomes exatamente como aparecem no texto, sem corrigir ortografia."
    )

    def __init__(self, model: BaseChatModel):
        # Modelo base — chain construída dinamicamente por collection_type
        self._base_model = model
        self._chain_cache: dict[str, object] = {}

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

    def extract(self, record: Record, collection_config: CollectionConfig) -> dict:
        """
        Extrai campos estruturados do registro.

        Retorna dict com os campos definidos em collection_config.extraction_fields.
        Também preenche record.structured_output in-place.
        """
        chain = self._get_chain(collection_config)
        text = record.get_concatenated_text()
        if not text.strip():
            result = {k: "" for k in collection_config.extraction_fields}
            record.structured_output = result
            return result

        field_list = "\n".join(
            f"- {name}: {collection_config.field_descriptions.get(name, name)}"
            for name in collection_config.extraction_fields
        )
        prompt = self.PROMPT_TEMPLATE.format(
            collection_type=collection_config.collection_type,
            record_text=text,
            field_list=field_list,
        )
        output = chain.invoke([HumanMessage(content=prompt)])
        result = output.model_dump()
        record.structured_output = result
        return result

    def __call__(self, state: dict) -> dict:
        record = state["current_record"]
        collection_config = state["collection_config"]
        self.extract(record, collection_config)
        return {**state, "current_record": record}
