"""
Factory de LLM providers.

Todos os providers retornam um BaseChatModel com a mesma interface —
os agentes são completamente agnósticos ao provider.

Providers suportados:
  ollama        — modelos locais via Ollama (padrão)
  ollama_cloud  — modelos hospedados no Ollama Cloud (requer OLLAMA_API_KEY)
  anthropic     — Claude via API Anthropic (requer ANTHROPIC_API_KEY)
  openai        — GPT via API OpenAI (requer OPENAI_API_KEY)

O A0 (Semana 5) usará esta factory para escalar para providers externos
em casos onde o modelo local não for suficiente.

Utilitário:
  disable_think(model) — retorna instância com think=False baked in.
  Deve ser chamado pelos agentes que usam with_structured_output com
  modelos Qwen3.5. NÃO use .bind(think=False): o binding é descartado
  quando with_structured_output delega via __getattr__ ao modelo base.
"""

import os

from langchain_core.language_models import BaseChatModel

# Endpoint fixo do Ollama Cloud — não é configurável via OLLAMA_BASE_URL porque
# esse env var já é usado pelo Ollama local (docker-compose aponta para o container).
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com"


def get_chat_model(provider: str, model: str, ollama_base_url: str = "http://localhost:11434") -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=ollama_base_url)

    if provider == "ollama_cloud":
        from langchain_ollama import ChatOllama
        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY não definida no .env — necessária para provider 'ollama_cloud'")
        return ChatOllama(
            model=model,
            base_url=_OLLAMA_CLOUD_BASE_URL,
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)

    raise ValueError(f"Provider desconhecido: '{provider}'. Use: ollama | ollama_cloud | anthropic | openai")


def disable_think(model: BaseChatModel) -> BaseChatModel:
    """
    Retorna o modelo com thinking mode desabilitado, baked in na instância.

    O nome do campo mudou entre versões do langchain_ollama:
      - 0.x: campo "think"
      - 1.x: campo "reasoning"

    Para outros providers: retorna o modelo sem alteração.
    """
    try:
        from langchain_ollama import ChatOllama
        if isinstance(model, ChatOllama):
            fields = ChatOllama.model_fields.keys()
            if "reasoning" in fields:
                return model.model_copy(update={"reasoning": False})
            elif "think" in fields:
                return model.model_copy(update={"think": False})
    except ImportError:
        pass
    return model


def make_structured(model: BaseChatModel, schema: type):
    """
    Retorna uma chain Runnable com thinking mode desabilitado e structured output.

    Usa method="json_schema" para todos os providers — passa o schema Pydantic completo
    ao modelo, que constrainge os nomes dos campos no output estruturado.

    O parâmetro reasoning/think é setado via model_copy() na instância antes de
    construir a chain — .bind() seria descartado por with_structured_output.

    Para Ollama: desabilita thinking mode antes de construir a chain (Qwen3.5/llama3.2
    com thinking ativo produz raciocínio que quebra o parser do structured output).
    Para providers não-Ollama (Anthropic, OpenAI): usa with_structured_output normalmente.
    """
    try:
        from langchain_ollama import ChatOllama
        if isinstance(model, ChatOllama):
            fields = ChatOllama.model_fields.keys()
            update = {}
            if "reasoning" in fields:
                update["reasoning"] = False
            elif "think" in fields:
                update["think"] = False
            configured = model.model_copy(update=update)
            return configured.with_structured_output(schema, method="json_schema")
    except ImportError:
        pass

    # providers não-Ollama (Anthropic, OpenAI): usar tool_use padrão — não especificar
    # method="json_schema" (específico do Ollama). Com tool_use, Claude retorna structured
    # output via function calling nativo — zero fallbacks esperados.
    return disable_think(model).with_structured_output(schema)
