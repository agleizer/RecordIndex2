"""
Schemas Pydantic compartilhados entre a camada de API e o pipeline.

Regra de organização:
- Schemas de OUTPUT dos agentes (HTROutput, NEROutput, etc.) ficam no arquivo
  do próprio agente — são detalhes de implementação acoplados ao agente.
- Schemas aqui são tipos que cruzam a fronteira API ↔ pipeline: responses
  tipados, payloads de request complexos, estruturas retornadas por múltiplos
  endpoints.

Benefício imediato: Swagger/OpenAPI gera documentação precisa dos endpoints
em vez de mostrar "object" genérico.
"""

from pydantic import BaseModel


class AgentConfig(BaseModel):
    provider: str
    model: str


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    agents: dict[str, AgentConfig]


class TranscribeResponse(BaseModel):
    filename: str
    htr_text: str
    debug_raw: str | None = None


class SegmentRequest(BaseModel):
    """
    Request para /a3/segment: lista de textos de linhas + tipo de coleção.

    lines: textos transcritos (htr_text ou corrected_text), em ordem espacial.
    collection_type: "batismo" | "casamento" | "obito"
    """

    lines: list[str]
    collection_type: str = "batismo"


class SegmentResponse(BaseModel):
    """
    Response de /a3/segment: índices de início de cada registro.

    record_start_indices: lista de índices 0-based que iniciam cada registro.
    reasoning: raciocínio do modelo (para debug e transparência acadêmica).
    num_records: número de registros identificados.
    """

    record_start_indices: list[int]
    reasoning: str
    num_records: int


class ExtractRequest(BaseModel):
    """
    Request para /a5/extract: texto de um registro + tipo de coleção.

    record_text: texto concatenado do registro (todas as linhas).
    collection_type: "batismo" | "casamento" | "obito"
    """

    record_text: str
    collection_type: str = "batismo"


class ExtractResponse(BaseModel):
    """
    Response de /a5/extract: campos extraídos do registro.

    fields: dict com os campos extraídos (nome, pai, mãe, data, etc.)
    collection_type: tipo de coleção usado para extração.
    """

    fields: dict[str, str]
    collection_type: str


class PipelineRunRequest(BaseModel):
    """
    Request opcional para /pipeline/run — permite especificar tipo de coleção.
    Se não enviado, o endpoint usa "batismo" como padrão.
    """

    collection_type: str = "batismo"
    collection_name: str = ""
