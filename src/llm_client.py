"""
Factory de LLM providers.

Todos os providers retornam um BaseChatModel com a mesma interface —
os agentes são completamente agnósticos ao provider.

Providers suportados:
  ollama     — modelos locais via Ollama (padrão)
  anthropic  — Claude via API Anthropic (requer ANTHROPIC_API_KEY)
  openai     — GPT via API OpenAI (requer OPENAI_API_KEY)

O A0 (Semana 5) usará esta factory para escalar para providers externos
em casos onde o modelo local não for suficiente.

Utilitário:
  disable_think(model) — retorna instância com think=False baked in.
  Deve ser chamado pelos agentes que usam with_structured_output com
  modelos Qwen3.5. NÃO use .bind(think=False): o binding é descartado
  quando with_structured_output delega via __getattr__ ao modelo base.
"""

from langchain_core.language_models import BaseChatModel


def get_chat_model(provider: str, model: str, ollama_base_url: str = "http://localhost:11434") -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=ollama_base_url)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)

    raise ValueError(f"Provider desconhecido: '{provider}'. Use: ollama | anthropic | openai")


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

    # providers não-Ollama: with_structured_output funciona normalmente
    return disable_think(model).with_structured_output(schema, method="json_schema")
