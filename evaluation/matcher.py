"""
Alinhamento entre registros do pipeline e registros de referência.

Estratégia: best-match por campo `nome` usando Jaro-Winkler.
Para cada registro GT, percorre todos os outputs e alinha com o que
maximizar a similaridade no nome — sem exigir correspondência posicional.

Threshold padrão: 0.82
  - Nomes próprios históricos têm variação ortográfica (Joanna/Joana, Emília/Emilia)
  - JW é tolerante a sufixos, boa escolha para nomes
  - 0.82 é mais permissivo que 0.85 para capturar casos com ruído HTR
"""

import unicodedata
from rapidfuzz.distance import JaroWinkler

MATCH_THRESHOLD = 0.80
# Pré-filtro: nome JW deve ser >= isso para o registro ser candidato.
# Evita que pai+mae de um registro completamente diferente resgate uma correspondência errada.
NOME_MIN = 0.65


def _normalize(s: str) -> str:
    """Lowercase + remove acentos."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(_normalize(a), _normalize(b))


def _composite_score(gt: dict, out_so: dict) -> float:
    """
    Score composto nome+pai+mae para resolver ambiguidade de nomes comuns
    (ex: múltiplos Manuéis ou Marias na mesma coleção).

    Pesos: nome=0.40, pai=0.35, mae=0.25
    Campos ausentes no output são excluídos do cálculo (não penalizam).
    Requer nome JW >= NOME_MIN como pré-condição — impede que pai+mae
    sozinhos resgatem uma correspondência completamente errada de nome.
    """
    nome_jw = jaro_winkler(gt.get("nome", ""), out_so.get("nome", ""))
    if nome_jw < NOME_MIN:
        return nome_jw  # abaixo do pré-filtro: retorna só o score do nome

    weights = {"nome": 0.40, "pai": 0.35, "mae": 0.25}
    total_w = 0.0
    total_s = 0.0

    for field, w in weights.items():
        gt_val = gt.get(field, "") or ""
        out_val = out_so.get(field, "") or ""
        if gt_val and out_val:
            total_s += jaro_winkler(gt_val, out_val) * w
            total_w += w

    if total_w == 0.0:
        return 0.0
    return total_s / total_w


def align(gt_records: list[dict], output_records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Para cada registro GT, encontra o melhor output usando score composto
    nome+pai+mae (Jaro-Winkler). Cada output é usado no máximo uma vez (greedy).

    Score composto resolve ambiguidade de nomes comuns (Manuéis, Marias, etc.):
    mesmo que o nome seja idêntico, pais diferentes diferenciam os registros.

    Retorna:
      alignments: lista de dicts com chaves gt, output, match_score, matched
      unmatched_outputs: outputs sem correspondência GT
    """
    used = set()
    alignments = []

    for gt in gt_records:
        best_score = 0.0
        best_idx = -1

        for i, out in enumerate(output_records):
            if i in used:
                continue
            out_so = out.get("structured_output") or {}
            score = _composite_score(gt, out_so)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= MATCH_THRESHOLD and best_idx >= 0:
            used.add(best_idx)
            alignments.append({
                "gt": gt,
                "output": output_records[best_idx],
                "match_score": round(best_score, 4),
                "matched": True,
            })
        else:
            alignments.append({
                "gt": gt,
                "output": None,
                "match_score": round(best_score, 4),
                "matched": False,
            })

    unmatched_outputs = [
        out for i, out in enumerate(output_records) if i not in used
    ]

    return alignments, unmatched_outputs
