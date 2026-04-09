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


def get_prompt(agent: str, key: str) -> str:
    """Retorna o template de prompt para o agente e chave dados."""
    global _prompts
    if not _prompts:
        path = Path(__file__).parent.parent / "prompts.yaml"
        with open(path, encoding="utf-8") as f:
            _prompts = yaml.safe_load(f)
    try:
        return _prompts[agent][key]
    except KeyError:
        raise KeyError(f"Prompt não encontrado: {agent}.{key} em prompts.yaml")
