"""
Parser para o CSV de referência da coleção.

Formato esperado (batismo):
  ID; Registo de batismo n.º N: NOME. Pai: PAI; Mãe: MÃE; YYYY-MM-DD

O CSV não é um ground truth transcricional completo — é o índice arquivístico
existente, com nome, pais e data de cada ato registrado.
"""

import re
from typing import Optional


def _parse_batismo_line(parts: list[str]) -> Optional[dict]:
    """
    parts[0] = ID arquivístico (PT/ABM/...)
    parts[1] = "Registo de batismo n.º N: NOME. Pai: PAI"
    parts[2] = "Mãe: MÃE"
    parts[3] = "YYYY-MM-DD"
    """
    if len(parts) < 4:
        return None

    # Extrai seq + nome + pai de parts[1]
    m = re.match(r"Registo de batismo n\.º (\d+): (.+?)\. Pai: (.+)", parts[1].strip())
    if not m:
        return None

    seq = int(m.group(1))
    nome = m.group(2).strip()
    pai = m.group(3).strip()

    mae = parts[2].replace("Mãe:", "").strip()
    data = parts[3].strip()

    return {
        "id": parts[0].strip(),
        "seq": seq,
        "nome": nome,
        "pai": pai,
        "mae": mae,
        "data": data,   # formato ISO: YYYY-MM-DD
    }


def _parse_casamento_line(parts: list[str]) -> Optional[dict]:
    """
    Placeholder — formato de casamento ainda não padronizado.
    Retorna None para linhas não reconhecidas.
    """
    return None


def _parse_obito_line(parts: list[str]) -> Optional[dict]:
    """
    Placeholder — formato de óbito ainda não padronizado.
    """
    return None


_PARSERS = {
    "batismo": _parse_batismo_line,
    "casamento": _parse_casamento_line,
    "obito": _parse_obito_line,
}


def parse_reference_csv(content: str, collection_type: str = "batismo") -> list[dict]:
    """
    Lê o CSV de referência e retorna lista de dicts com campos padronizados.
    Linhas em branco ou não reconhecidas são ignoradas silenciosamente.
    """
    parser = _PARSERS.get(collection_type, _parse_batismo_line)
    records = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(";")
        result = parser(parts)
        if result:
            records.append(result)

    return records
