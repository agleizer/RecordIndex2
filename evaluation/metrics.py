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

# Versão do vocabulário de métricas gravado no JSON de saída.
#   1 = até 05/09/2026: slot_accuracy, effective_slot_accuracy, strict_effective_f1,
#       effective_recall, effective_f1
#   2 = a partir de 05/09/2026: matched_recovery_rate, effective_recovery_rate,
#       effective_composite_f1, legacy_effective_recall, legacy_effective_f1
# Arquivos sem a chave "metrics_schema" são schema 1. Use normalize_metrics_schema()
# para ler os dois formatos com os nomes atuais.
METRICS_SCHEMA = 2

# Mapa schema 1 -> schema 2, por campo. Usado só na leitura de arquivos antigos.
_SCHEMA_1_TO_2 = {
    "slot_accuracy": "matched_recovery_rate",
    "effective_slot_accuracy": "effective_recovery_rate",
    "strict_effective_f1": "effective_composite_f1",
    "effective_recall": "legacy_effective_recall",
    "effective_f1": "legacy_effective_f1",
}

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

    precision:          fração dos outputs que correspondem a um GT real
    recall:             fração dos GTs encontrados no output
    segmentation_ratio: ratio output/GT (1.0 = ideal; >1 = hiperseg; <1 = fusão)
                        NÃO é "taxa de sobre-segmentação" — é uma razão de contagem.
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
        "segmentation_ratio": over_seg,
    }


def extraction_metrics(alignments: list[dict], fields: list[str]) -> dict:
    """
    Métricas de Camada 2 (extração), calculadas apenas sobre pares alinhados.

    Para cada campo, trata `fuzzy` como TP.
    Registros não-alinhados NÃO entram no cálculo de precision/recall/f1 padrão —
    reportados à parte como `_coverage` (fração de GTs com alinhamento confiável).

    MÉTRICAS DE REPORTE (as duas que o relatório do TCC 2 publica):

      effective_recovery_rate = tp / total_gt
          Taxa de recuperação efetiva, o R_ef do relatório.
          "De todos os registros que existem no ground truth, em quantos o sistema
          produziu este campo corretamente?"
          O denominador é o total do GT, incluindo registros que a segmentação nem
          chegou a casar, o que integra as duas camadas em um único número.

          NÃO é uma accuracy: não há verdadeiro negativo no denominador. Chamava-se
          effective_slot_accuracy até 05/09/2026, nome incorreto pela definição.

      effective_composite_f1 = 2 × precision × effective_recovery_rate
                                   / (precision + effective_recovery_rate)
          O F1_ef do relatório. Indicador composto definido no trabalho, não uma
          métrica padrão: combina a precision medida sobre os pares casados com uma
          taxa de recuperação medida sobre o GT inteiro. Por isso NÃO deve ser
          comparado diretamente a valores de F1 da literatura.
          Chamava-se strict_effective_f1 até 05/09/2026.

    MÉTRICAS DE DIAGNÓSTICO:

      matched_recovery_rate = tp / casados
          "Dos registros que a segmentação casou, em quantos este campo saiu correto?"
          Isola a qualidade da extração do desempenho da segmentação.
          Chamava-se slot_accuracy até 05/09/2026.

      precision, recall, f1   sobre os pares casados, semântica fuzzy
      strict_recall           = tp / (tp + fn_strict)
      fn_strict               = fn + fp, semântica NER estrita: um valor presente
                                porém errado conta como comissão e como omissão

      strict_effective_recall = tp / (tp + fn_strict + unmatched_gt)
          Para campos de cardinalidade 1 é algebricamente idêntica a
          effective_recovery_rate, porque tp + fp + fn = casados. Mantida porque a
          identidade deixa de valer se algum campo passar a admitir múltiplos valores.

    MÉTRICAS LEGADAS (anteriores à semântica NER estrita de 02/05/2026):

      legacy_effective_recall = tp / (tp + fn + unmatched_gt)
      legacy_effective_f1     = 2 × precision × legacy_effective_recall
                                    / (precision + legacy_effective_recall)

      Usam fn em vez de fn_strict, ou seja, um valor presente porém errado conta
      apenas como FP. Isso infla o resultado de campos que o modelo sempre preenche
      com alguma coisa. Não reportar. Mantidas para reproduzir análises antigas.
      Chamavam-se effective_recall e effective_f1 até 05/09/2026, nomes que eram a
      principal fonte de leitura errada dos JSONs.
    """
    matched = [a for a in alignments if a["matched"]]
    unmatched_gt = len(alignments) - len(matched)
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

        # Taxas de recuperação. A efetiva (R_ef) é a métrica principal do relatório:
        # denominador é o GT inteiro, incluindo registros que a segmentação não casou.
        matched_recovery_rate = round(tp / len(matched), 4) if matched else 0.0
        effective_recovery_rate = round(tp / len(alignments), 4) if alignments else 0.0

        # F1_ef: indicador composto do trabalho, precision sobre casados combinada
        # com a taxa de recuperação efetiva sobre o GT inteiro.
        effective_composite_f1 = (
            round(
                2 * precision * effective_recovery_rate
                / (precision + effective_recovery_rate),
                4,
            )
            if (precision + effective_recovery_rate) > 0
            else 0.0
        )

        # Semântica NER estrita: valor incorreto presente conta como FP e FN.
        fn_strict = fn + fp
        strict_recall = round(tp / (tp + fn_strict), 4) if (tp + fn_strict) > 0 else 0.0
        strict_eff_recall = (
            round(tp / (tp + fn_strict + unmatched_gt), 4)
            if (tp + fn_strict + unmatched_gt) > 0
            else 0.0
        )

        # Família legada (pré 02/05/2026), inflada. Não reportar.
        legacy_eff_recall = (
            round(tp / (tp + fn + unmatched_gt), 4)
            if (tp + fn + unmatched_gt) > 0
            else 0.0
        )
        legacy_eff_f1 = (
            round(2 * precision * legacy_eff_recall / (precision + legacy_eff_recall), 4)
            if (precision + legacy_eff_recall) > 0
            else 0.0
        )

        results[field] = {
            # contagens
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fn_strict": fn_strict,
            # métricas de reporte
            "effective_recovery_rate": effective_recovery_rate,
            "effective_composite_f1": effective_composite_f1,
            # diagnóstico
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_rate": exact_rate,
            "matched_recovery_rate": matched_recovery_rate,
            "strict_recall": strict_recall,
            "strict_effective_recall": strict_eff_recall,
            # legado, não reportar
            "legacy_effective_recall": legacy_eff_recall,
            "legacy_effective_f1": legacy_eff_f1,
        }

    return results


def normalize_metrics_schema(eval_result: dict) -> dict:
    """
    Devolve um resultado de avaliação com os nomes de métrica do schema atual,
    aceitando tanto arquivos novos quanto os gravados antes de 05/09/2026.

    Serve para ler os eval_*.json históricos (as nove execuções que sustentam o
    relatório do TCC 2, entre outros) sem precisar reexecutar a avaliação.

    O schema é detectado pela chave "metrics_schema" no topo do arquivo; sua
    ausência significa schema 1. Arquivos já no schema atual passam intactos.
    Não modifica o dicionário recebido.

    Exemplo:
        d = json.load(open("eval_2026-08-28_01-16.json", encoding="utf-8"))
        d = normalize_metrics_schema(d)
        d["extraction"]["nome"]["effective_recovery_rate"]   # R_ef
        d["extraction"]["nome"]["effective_composite_f1"]    # F1_ef
    """
    if eval_result.get("metrics_schema", 1) >= METRICS_SCHEMA:
        return eval_result

    out = dict(eval_result)
    extraction = out.get("extraction")
    if not isinstance(extraction, dict):
        out["metrics_schema"] = METRICS_SCHEMA
        return out

    migrated: dict = {}
    for field, m in extraction.items():
        # "_coverage" é um float solto, não um dicionário de campo.
        if not isinstance(m, dict):
            migrated[field] = m
            continue
        migrated[field] = {_SCHEMA_1_TO_2.get(k, k): v for k, v in m.items()}

    out["extraction"] = migrated
    out["metrics_schema"] = METRICS_SCHEMA
    return out
