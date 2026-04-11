"""
RecordIndex 2.0 — Serviço de Avaliação (porta 8001)

Endpoint principal:
  POST /evaluate
    pipeline_json : arquivo JSON gerado pelo pipeline (output de /pipeline/run)
    reference_csv : CSV de referência da coleção (índice arquivístico)
    collection_type : "batismo" | "casamento" | "obito" (default: batismo)

O serviço é completamente independente do pipeline — não acessa Ollama,
não usa GPU, não chama agentes. É executado após o pipeline, offline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

OUTPUT_DIR = Path("/data/output")

from csv_parser import parse_reference_csv
from matcher import align
from metrics import compare_field, extraction_metrics, segmentation_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RecordIndex 2.0 — Evaluation Service",
    description=(
        "Avalia o output do pipeline comparando com o índice arquivístico da coleção. "
        "Métricas separadas para segmentação (A3) e extração (A5)."
    ),
    version="1.0.0",
)

# Campos por tipo de coleção
_FIELDS: dict[str, list[str]] = {
    "batismo": ["nome", "pai", "mae", "data"],
    "casamento": ["noivo", "noiva", "pai_noivo", "mae_noivo", "pai_noiva", "mae_noiva", "data"],
    "obito": ["nome", "pai", "mae", "data"],
}


@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}


@app.post("/evaluate")
async def evaluate(
    pipeline_json: UploadFile = File(..., description="JSON gerado pelo /pipeline/run"),
    reference_csv: UploadFile = File(..., description="CSV de referência da coleção"),
    collection_type: str = Form("batismo"),
):
    # --- Leitura dos arquivos ---
    try:
        pipeline_content = await pipeline_json.read()
        pipeline_data = json.loads(pipeline_content)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"pipeline_json inválido: {e}"})

    try:
        ref_content = (await reference_csv.read()).decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"reference_csv inválido: {e}"})

    # --- Parse ---
    output_records: list[dict] = pipeline_data.get("records", [])
    if isinstance(output_records, dict):
        output_records = list(output_records.values())

    gt_records = parse_reference_csv(ref_content, collection_type)
    if not gt_records:
        return JSONResponse(status_code=400, content={"error": "Nenhum registro encontrado no reference_csv"})

    fields = _FIELDS.get(collection_type, _FIELDS["batismo"])

    logger.info(
        "Avaliando: %d registros GT vs %d outputs | tipo=%s",
        len(gt_records), len(output_records), collection_type,
    )

    # --- Alinhamento ---
    alignments, unmatched_outputs = align(gt_records, output_records, collection_type)

    n_matched = sum(1 for a in alignments if a["matched"])

    # --- Métricas ---
    seg = segmentation_metrics(len(gt_records), len(output_records), n_matched)
    ext = extraction_metrics(alignments, fields)

    # --- Comparação por registro ---
    record_comparisons = []
    for pair in alignments:
        gt = pair["gt"]
        entry: dict = {
            "gt_seq": gt["seq"],
            "gt_id": gt["id"],
            "gt": {f: gt.get(f, "") for f in fields},
            "matched": pair["matched"],
            "match_score": pair["match_score"],  # score composto nome+pai+mae
        }

        if pair["matched"]:
            out = pair["output"]
            out_fields = (out.get("structured_output") or {})
            entry["output_record_id"] = out.get("id")
            entry["output"] = {f: out_fields.get(f, "") for f in fields}
            entry["output_validation"] = out.get("validation")
            entry["field_comparison"] = {
                f: compare_field(gt.get(f, ""), out_fields.get(f, ""), f)
                for f in fields
            }
        else:
            entry["output_record_id"] = None
            entry["output"] = None
            entry["output_validation"] = None
            entry["field_comparison"] = None

        record_comparisons.append(entry)

    # --- Resposta ---
    result = {
        "collection_type": collection_type,
        "segmentation": seg,
        "extraction": ext,
        "record_comparisons": record_comparisons,
        "unmatched_output_ids": [r.get("id") for r in unmatched_outputs],
    }

    # --- Persistência ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    eval_path = OUTPUT_DIR / f"eval_{timestamp}.json"
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Resultado salvo em %s", eval_path)
    except Exception as e:
        logger.warning("Não foi possível salvar eval em disco: %s", e)

    return result
