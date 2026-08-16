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

from pydantic import BaseModel, field_validator

_VALID_HTR_SCOPES = {"line", "page"}
_VALID_PIPELINE_MODES = {"mas", "generalist"}


class AgentConfig(BaseModel):
    provider: str
    model: str


class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    agents: dict[str, AgentConfig]


class TranscribeResponse(BaseModel):
    """
    Response de /a2/transcribe.

    Modo line (htr_scope="line", padrão):
      htr_text: texto transcrito da imagem de linha enviada.

    Modo page (htr_scope="page"):
      lines: lista de textos transcritos — um item por linha que Claude identificou.
      n_returned: número de linhas que Claude transcreveu.
      Claude determina o count de linhas autonomamente (sem depender do A1).
    """

    filename: str
    htr_scope: str = "line"
    # line mode
    htr_text: str | None = None
    debug_raw: str | None = None
    # page mode
    lines: list[str] | None = None
    n_returned: int | None = None


class SegmentRequest(BaseModel):
    """
    Request para /a3/segment.

    Modo line (htr_scope="line", padrão):
      lines: textos transcritos (htr_text ou corrected_text), em ordem espacial.

    Modo page (htr_scope="page"):
      page_text: texto completo da página como bloco único.

    collection_type: "batismo" | "casamento" | "obito" (ambos os modos)
    """

    htr_scope: str = "line"
    lines: list[str] = []          # modo line
    page_text: str = ""            # modo page
    collection_type: str = "batismo"

    @field_validator("htr_scope")
    @classmethod
    def _validate_htr_scope(cls, v: str) -> str:
        if v not in _VALID_HTR_SCOPES:
            raise ValueError(f"htr_scope inválido: '{v}'. Use: line | page")
        return v


class SegmentResponse(BaseModel):
    """
    Response de /a3/segment.

    Modo line (htr_scope="line"):
      record_start_indices: índices 0-based que iniciam cada registro.

    Modo page (htr_scope="page"):
      record_texts: texto completo de cada registro identificado.

    reasoning e num_records presentes em ambos os modos.
    Campos do modo oposto são omitidos da resposta (None + exclude_none).
    """

    htr_scope: str = "line"
    # line mode (None em modo page)
    record_start_indices: list[int] | None = None
    # page mode (None em modo line)
    record_texts: list[str] | None = None
    # ambos
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
    pipeline_mode:   "mas" (padrão, pipeline A0-A6 completo) | "generalist"
                     (modo alternativo, AG+AM, ver 2026_08_16_modulo_generalista_vs_mas/).
                     Em modo "generalist", htr_scope é ignorado (AG sempre lê a página inteira).
    """

    collection_name: str = "Coleção"
    year: str = ""
    location: str = ""
    collection_type: str = ""        # A0 infere se vazio
    record_start_hint: str = ""      # A0 usa padrão se vazio
    record_template: str = ""        # molde com placeholders para A4 — se vazio, A4 é pulado
    image_dir: str = ""              # diretório de imagens — se vazio, usa SAMPLES_DIR do config
    output_formats: list[str] = ["json"]  # formatos de saída: "json" | "csv" | "txt"
    htr_scope: str = "line"          # "line": A2 por linha (padrão) | "page": A2 por página inteira
    pipeline_mode: str = "mas"       # "mas" (padrão) | "generalist"

    @field_validator("htr_scope")
    @classmethod
    def _validate_htr_scope(cls, v: str) -> str:
        if v not in _VALID_HTR_SCOPES:
            raise ValueError(f"htr_scope inválido: '{v}'. Use: line | page")
        return v

    @field_validator("pipeline_mode")
    @classmethod
    def _validate_pipeline_mode(cls, v: str) -> str:
        if v not in _VALID_PIPELINE_MODES:
            raise ValueError(f"pipeline_mode inválido: '{v}'. Use: mas | generalist")
        return v
