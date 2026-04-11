"""
Cálculo de métricas em duas camadas:

Camada 1 — Segmentação (mede A3):
  Calculada sobre o total de registros GT vs. total de outputs, independente
  de extração.

Camada 2 — Extração (mede A5):
  Calculada APENAS sobre pares alinhados com confiança (match_score >= threshold).
  Registros não-alinhados não entram no F1 de extração — mas a taxa de não-alinhamento
  é reportada explicitamente como métrica de cobertura.

Métrica de data:
  GT tem data ISO (YYYY-MM-DD). Pipeline extrai texto livre em português.
  Comparação por mês: verifica se o mês PT do GT aparece na string extraída.
  Exact match é reportado separadamente (útil quando A5 retorna ISO diretamente).
"""

import os
import re
from matcher import jaro_winkler

# Threshold JW para considerar dois valores de campo como equivalentes (match fuzzy).
FUZZY_THRESHOLD = float(os.getenv("EVAL_FUZZY_THRESHOLD", "0.85"))

_MONTH_PT = {
    1: ["janeiro", "jan"],
    2: ["fevereiro", "fev"],
    3: ["março", "marco", "mar"],
    4: ["abril", "abr"],
    5: ["maio", "mai"],
    6: ["junho", "jun"],
    7: ["julho", "jul"],
    8: ["agosto", "ago"],
    9: ["setembro", "set"],
    10: ["outubro", "out"],
    11: ["novembro", "nov"],
    12: ["dezembro", "dez"],
}


def _compare_date(gt_iso: str, out_str: str) -> dict:
    """
    gt_iso: "1866-01-04"
    out_str: "quatro dias do mez de janeiro de mil oitocentos..." ou "1866-01-04"

    Retorna: {exact, month_match, year_match}
    """
    result = {"exact": False, "month_match": False, "year_match": False}

    if not out_str or not gt_iso:
        return result

    out_lower = out_str.lower()

    # Exact string comparison (normalizado)
    result["exact"] = gt_iso.strip() == out_str.strip()

    # Extrai mês e ano do GT
    try:
        parts = gt_iso.split("-")
        gt_year = int(parts[0])
        gt_month = int(parts[1])
    except (IndexError, ValueError):
        return result

    # Verifica ano no output (4 dígitos ou por extenso)
    year_str = str(gt_year)
    result["year_match"] = year_str in out_str

    # Verifica mês: ISO no output
    iso_match = re.search(r"(\d{4})-(\d{2})-\d{2}", out_str)
    if iso_match:
        result["month_match"] = int(iso_match.group(2)) == gt_month
        return result

    # Verifica mês por nome em português
    for alias in _MONTH_PT.get(gt_month, []):
        if alias in out_lower:
            result["month_match"] = True
            break

    return result


def compare_field(gt_value: str, out_value: str, field: str) -> dict:
    """
    Compara um campo GT com o campo extraído pelo pipeline.

    Retorna:
      present     — campo não está vazio no output
      exact       — correspondência exata (normalizada)
      fuzzy       — Jaro-Winkler >= 0.85 (para nomes) ou month_match (para data)
      jaro_winkler — score numérico (0.0–1.0)
      date_detail — {exact, month_match, year_match} apenas para campo data
    """
    result = {
        "present": bool(out_value and out_value.strip()),
        "exact": False,
        "fuzzy": False,
        "jaro_winkler": 0.0,
    }

    if not out_value or not gt_value:
        return result

    if field == "data":
        date_detail = _compare_date(gt_value, out_value)
        result["date_detail"] = date_detail
        result["exact"] = date_detail["exact"]
        result["fuzzy"] = date_detail["month_match"]
        result["jaro_winkler"] = jaro_winkler(gt_value, out_value)
    else:
        jw = jaro_winkler(gt_value, out_value)
        result["jaro_winkler"] = round(jw, 4)
        result["exact"] = gt_value.strip().lower() == out_value.strip().lower()
        result["fuzzy"] = jw >= FUZZY_THRESHOLD

    return result


def segmentation_metrics(total_gt: int, total_output: int, n_matched: int) -> dict:
    """
    Métricas de Camada 1 (segmentação).

    precision_seg: fração dos outputs que correspondem a um GT real
    recall_seg:    fração dos GTs encontrados no output
    over_seg_rate: ratio output/GT (1.0 = ideal; >1 = hiperseg; <1 = fusão)
    """
    precision = round(n_matched / total_output, 4) if total_output > 0 else 0.0
    recall = round(n_matched / total_gt, 4) if total_gt > 0 else 0.0
    over_seg = round(total_output / total_gt, 4) if total_gt > 0 else None

    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_gt": total_gt,
        "total_output": total_output,
        "matched": n_matched,
        "unmatched_gt": total_gt - n_matched,
        "unmatched_output": total_output - n_matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "over_segmentation_rate": over_seg,
    }


def extraction_metrics(alignments: list[dict], fields: list[str]) -> dict:
    """
    Métricas de Camada 2 (extração), calculadas apenas sobre pares alinhados.

    Para cada campo, trata `fuzzy` como TP.
    Registros não-alinhados NÃO entram no cálculo — reportados à parte
    como `coverage` (fração de GTs com alinhamento confiável).
    """
    matched = [a for a in alignments if a["matched"]]
    coverage = round(len(matched) / len(alignments), 4) if alignments else 0.0

    results: dict[str, dict] = {"_coverage": coverage}

    for field in fields:
        tp = fp = fn = 0
        exact_count = 0

        for pair in matched:
            gt_val = pair["gt"].get(field, "") or ""
            out_val = (pair["output"].get("structured_output") or {}).get(field, "") or ""
            cmp = compare_field(gt_val, out_val, field)

            if cmp["fuzzy"]:
                tp += 1
                if cmp["exact"]:
                    exact_count += 1
            elif cmp["present"]:
                fp += 1
            else:
                fn += 1

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
        exact_rate = round(exact_count / len(matched), 4) if matched else 0.0

        results[field] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_rate": exact_rate,
        }

    return results
