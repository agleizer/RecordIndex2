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


class LineSegmentInfo(BaseModel):
    """Informação de uma linha detectada pelo A1."""

    id: str
    bbox: list[int]  # [x1, y1, x2, y2]


class SegmentPageResponse(BaseModel):
    """
    Response de /a1/segment: linhas detectadas em uma imagem de página.

    filename: nome do arquivo enviado.
    num_lines: número de linhas detectadas.
    lines: lista de bounding boxes (x1, y1, x2, y2).
    crops_dir: diretório onde os recortes foram salvos (dentro do container).
    """

    filename: str
    num_lines: int
    lines: list[LineSegmentInfo]
    crops_dir: str


class CorrectRequest(BaseModel):
    """
    Request para /a4/correct: texto bruto + molde de referência.

    record_text:     texto HTR concatenado do registro (com erros).
    record_template: molde com placeholders <CAMPO> e trechos <OPT>...</OPT>.
    """

    record_text: str
    record_template: str


class CorrectResponse(BaseModel):
    """
    Response de /a4/correct: texto corrigido pelo template.

    corrected_text: texto com placeholders preenchidos e texto fixo preservado.
    """

    corrected_text: str


class ValidateRequest(BaseModel):
    """
    Request para /a6/validate: campos extraídos + texto original + tipo de coleção.

    record_text:      texto concatenado do registro (HTR bruto ou corrigido).
    structured_output: campos extraídos pelo A5 {nome, pai, mãe, data, ...}.
    collection_type:  "batismo" | "casamento" | "obito"
    """

    record_text: str
    structured_output: dict[str, str]
    collection_type: str = "batismo"


class ValidateResponse(BaseModel):
    """
    Response de /a6/validate: resultado da validação do registro.

    score:        0.0 (falhou) a 1.0 (perfeito)
    verdict:      "ok" | "needs_review" | "failed"
    field_errors: {campo: [lista de erros/avisos]}
    notes:        análise do LLM (vazio se score >= 0.8)
    """

    score: float
    verdict: str
    field_errors: dict[str, list[str]]
    notes: str


class PipelineRunRequest(BaseModel):
    """
    Request para /pipeline/run — espelha CollectionInput.

    collection_name: nome legível da coleção (ex: "Porto da Cruz Batismos 1860").
    year:            ano dos registros (ex: "1860"). Ajuda na classificação.
    location:        localidade (ex: "Porto da Cruz, Madeira"). Ajuda na classificação.
    collection_type: hint opcional de tipo ("batismo" | "casamento" | "obito").
                     Se vazio, A0 infere automaticamente.
    record_start_hint: expressão típica de início de registro (ex: "Aos").
                       Se vazio, A0 usa o padrão do tipo detectado.
    """

    collection_name: str = "Coleção"
    year: str = ""
    location: str = ""
    collection_type: str = ""        # A0 infere se vazio
    record_start_hint: str = ""      # A0 usa padrão se vazio
    record_template: str = ""        # molde com placeholders para A4 — se vazio, A4 é pulado
    image_dir: str = ""              # diretório de imagens — se vazio, usa SAMPLES_DIR do config
    output_formats: list[str] = ["json"]  # formatos de saída: "json" | "csv" | "txt"
