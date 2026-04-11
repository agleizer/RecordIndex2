"""
Configuração de coleção — define tipo, campos e hints para o pipeline.

CollectionConfig é o contrato entre A0 (orquestrador) e os demais agentes:
- A3 usa record_start_hint para identificar início de registros
- A5 usa extraction_fields para construir o schema Pydantic dinamicamente

Tipos suportados: batismo, casamento, obito.
Extensível: instancie CollectionConfig diretamente para coleções customizadas.
"""

from dataclasses import dataclass, field

from src.prompts import get_field_descriptions


@dataclass
class CollectionConfig:
    """
    Configuração de uma coleção de manuscritos históricos.

    collection_type: identifica o tipo de registro ("batismo" | "casamento" | "obito")
    collection_name: nome legível da coleção (ex: "Batismos São Paulo 1850-1870")
    record_start_hint: palavra ou expressão que tipicamente inicia um registro
                       (usada por A3 no prompt de segmentação)
    extraction_fields: mapeamento campo → tipo Python para o A5 gerar o schema Pydantic
                       dinamicamente via pydantic.create_model()
    """

    collection_type: str
    collection_name: str
    record_start_hint: str
    extraction_fields: dict[str, type] = field(default_factory=dict)
    field_descriptions: dict[str, str] = field(default_factory=dict)
    record_template: str = ""        # molde com placeholders para A4 — se vazio, A4 é pulado

    @classmethod
    def batismo(cls, name: str = "Batismos") -> "CollectionConfig":
        return cls(
            collection_type="batismo",
            collection_name=name,
            record_start_hint="Aos",
            extraction_fields={
                "nome": str,
                "pai": str,
                "mae": str,
                "data": str,
            },
            field_descriptions=get_field_descriptions("batismo"),
        )

    @classmethod
    def casamento(cls, name: str = "Casamentos") -> "CollectionConfig":
        return cls(
            collection_type="casamento",
            collection_name=name,
            record_start_hint="Aos",
            extraction_fields={
                "noivo": str,
                "noiva": str,
                "pai_noivo": str,
                "mae_noivo": str,
                "pai_noiva": str,
                "mae_noiva": str,
                "data": str,
            },
            field_descriptions=get_field_descriptions("casamento"),
        )

    @classmethod
    def obito(cls, name: str = "Óbitos") -> "CollectionConfig":
        return cls(
            collection_type="obito",
            collection_name=name,
            record_start_hint="Aos",
            extraction_fields={
                "nome": str,
                "pai": str,
                "mae": str,
                "data": str,
                "idade": str,
            },
            field_descriptions=get_field_descriptions("obito"),
        )
