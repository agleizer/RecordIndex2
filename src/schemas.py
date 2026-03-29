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

from typing import Any
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
