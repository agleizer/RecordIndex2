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
"""

from langchain_core.language_models import BaseChatModel


def get_chat_model(provider: str, model: str, ollama_base_url: str = "http://localhost:11434") -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=ollama_base_url, think=False)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)

    raise ValueError(f"Provider desconhecido: '{provider}'. Use: ollama | anthropic | openai")
