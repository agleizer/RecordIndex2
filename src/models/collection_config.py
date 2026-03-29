"""
Configuração de coleção — define tipo, campos e hints para o pipeline.

CollectionConfig é o contrato entre A0 (orquestrador) e os demais agentes:
- A3 usa record_start_hint para identificar início de registros
- A5 usa extraction_fields para construir o schema Pydantic dinamicamente

Tipos suportados: batismo, casamento, obito.
Extensível: instancie CollectionConfig diretamente para coleções customizadas.
"""

from dataclasses import dataclass, field


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
            field_descriptions={
                "nome": "nome da pessoa batizada",
                "pai": "nome do pai da pessoa batizada",
                "mae": "nome da mãe da pessoa batizada",
                "data": "data do batismo (dia, mês e ano)",
            },
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
            field_descriptions={
                "noivo": "nome do noivo",
                "noiva": "nome da noiva",
                "pai_noivo": "nome do pai do noivo",
                "mae_noivo": "nome da mãe do noivo",
                "pai_noiva": "nome do pai da noiva",
                "mae_noiva": "nome da mãe da noiva",
                "data": "data do casamento (dia, mês e ano)",
            },
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
            field_descriptions={
                "nome": "nome da pessoa falecida",
                "pai": "nome do pai da pessoa falecida",
                "mae": "nome da mãe da pessoa falecida",
                "data": "data do óbito (dia, mês e ano)",
                "idade": "idade da pessoa falecida",
            },
        )
