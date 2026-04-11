"""
Loader de prompts a partir de prompts.yaml (raiz do projeto).

Uso:
    from src.prompts import get_prompt
    template = get_prompt("a3", "segment")
    prompt = template.format(collection_type=..., ...)

O arquivo é carregado uma vez (lazy) e cacheado em memória.
Para alterar um prompt, edite prompts.yaml — sem rebuild do container
(o arquivo é montado via volume em dev).
"""

import yaml
from pathlib import Path

_prompts: dict = {}


def _load() -> None:
    global _prompts
    if not _prompts:
        path = Path(__file__).parent.parent / "prompts.yaml"
        with open(path, encoding="utf-8") as f:
            _prompts = yaml.safe_load(f)


def get_prompt(agent: str, key: str) -> str:
    """Retorna o template de prompt para o agente e chave dados."""
    _load()
    try:
        return _prompts[agent][key]
    except KeyError:
        raise KeyError(f"Prompt não encontrado: {agent}.{key} em prompts.yaml")


def get_field_descriptions(collection_type: str) -> dict[str, str]:
    """
    Retorna as descrições de campos para o tipo de coleção dado.

    As descrições são usadas pelo A5 no prompt de extração para guiar
    o modelo a localizar cada campo no texto manuscrito.
    Editável em prompts.yaml sem rebuild do container.
    """
    _load()
    try:
        return dict(_prompts["field_descriptions"][collection_type])
    except KeyError:
        raise KeyError(
            f"field_descriptions não encontrado para '{collection_type}' em prompts.yaml"
        )
