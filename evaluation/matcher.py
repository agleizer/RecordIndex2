"""
Alinhamento entre registros do pipeline e registros de referência.

Estratégia: best-match por score composto (Jaro-Winkler nos campos-chave),
dependente do tipo de coleção:
  batismo/obito  → nome (0.40) + pai (0.35) + mae (0.25)
  casamento      → noivo (0.45) + noiva (0.35) + pai_noivo (0.20)

O campo primário (primeiro peso) funciona como pré-filtro: JW < NOME_MIN
descarta o par sem calcular os demais campos, evitando que campos secundários
resgatem correspondências com nomes completamente diferentes.

Threshold final: 0.80 (score composto)
  - Nomes próprios históricos têm variação ortográfica (Joanna/Joana, Emília/Emilia)
  - JW é tolerante a prefixos, boa escolha para nomes próprios
"""

import unicodedata
from rapidfuzz.distance import JaroWinkler

MATCH_THRESHOLD = 0.80
# Pré-filtro: campo primário JW deve ser >= isso para o registro ser candidato.
# Evita que campos secundários resgatem uma correspondência completamente errada.
NOME_MIN = 0.65
# Rescue: se nome falha o pré-filtro mas a média dos campos secundários presentes
# atinge esse threshold, o par ainda é considerado (pai+mae corretos, nome errado).
SECONDARY_RESCUE_MIN = 0.85

# Pesos por tipo de coleção: (campo_primário, {campo: peso})
# O primeiro campo da lista é o "primary" — sujeito ao pré-filtro NOME_MIN.
_MATCH_WEIGHTS: dict[str, dict[str, float]] = {
    "batismo":   {"nome": 0.40, "pai": 0.35, "mae": 0.25},
    "casamento": {"noivo": 0.45, "noiva": 0.35, "pai_noivo": 0.20},
    "obito":     {"nome": 0.40, "pai": 0.35, "mae": 0.25},
}
_DEFAULT_WEIGHTS = _MATCH_WEIGHTS["batismo"]


def _normalize(s: str) -> str:
    """Lowercase + remove acentos."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(_normalize(a), _normalize(b))


def _composite_score(gt: dict, out_so: dict, weights: dict[str, float]) -> float:
    """
    Score composto para resolver ambiguidade de nomes comuns.

    weights: {campo: peso} — o primeiro campo é o primário (sujeito ao pré-filtro).
    Campos ausentes no output são excluídos do cálculo (não penalizam).

    Pré-filtro: campo primário JW >= NOME_MIN — impede que campos secundários
    sozinhos resgatem uma correspondência completamente errada.

    Rescue: se primário falha mas a média dos campos secundários presentes
    >= SECONDARY_RESCUE_MIN, o pré-filtro é ignorado e o score composto é calculado
    normalmente. Cobre o caso onde A5 extraiu a entidade errada no nome mas pai+mae
    estão corretos.
    """
    primary_field = next(iter(weights))
    primary_jw = jaro_winkler(gt.get(primary_field, ""), out_so.get(primary_field, ""))
    if primary_jw < NOME_MIN:
        # Rescue: verifica se campos secundários compensam o nome errado.
        # Caso típico: A5 extraiu entidade errada no nome mas pai+mae estão corretos.
        secondary_scores = []
        for field, _ in list(weights.items())[1:]:
            gt_val = gt.get(field, "") or ""
            out_val = out_so.get(field, "") or ""
            if gt_val and out_val:
                secondary_scores.append(jaro_winkler(gt_val, out_val))
        if not secondary_scores or (sum(secondary_scores) / len(secondary_scores)) < SECONDARY_RESCUE_MIN:
            return primary_jw  # abaixo do pré-filtro e sem rescue: retorna score do campo primário

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


def align(
    gt_records: list[dict],
    output_records: list[dict],
    collection_type: str = "batismo",
) -> tuple[list[dict], list[dict]]:
    """
    Para cada registro GT, encontra o melhor output usando score composto
    (Jaro-Winkler nos campos-chave do tipo de coleção). Cada output é usado
    no máximo uma vez (greedy).

    collection_type controla quais campos e pesos são usados:
      batismo/obito : nome+pai+mae
      casamento     : noivo+noiva+pai_noivo

    Retorna:
      alignments: lista de dicts com chaves gt, output, match_score, matched
      unmatched_outputs: outputs sem correspondência GT
    """
    weights = _MATCH_WEIGHTS.get(collection_type, _DEFAULT_WEIGHTS)
    used = set()
    alignments = []

    for gt in gt_records:
        best_score = 0.0
        best_idx = -1

        for i, out in enumerate(output_records):
            if i in used:
                continue
            out_so = out.get("structured_output") or {}
            score = _composite_score(gt, out_so, weights)
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
