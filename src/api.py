"""
RecordIndex 2.0 — API HTTP (FastAPI).

Expõe os agentes e o pipeline via REST para testes com Postman.
Roda na porta 8000 dentro do container (mapeada para localhost:8000).

Endpoints:
  GET  /health                  — liveness check
  POST /a2/transcribe           — transcreve uma imagem de linha (testa A2 isolado)
  POST /pipeline/run            — roda pipeline completo sobre volumes/samples/
  GET  /pipeline/last-output    — retorna o último output.json gerado
"""

import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.config import Config
from src.llm_client import get_chat_model
from src.agents.a2_htr import A2HTRAgent
from src import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recordindex.api")

app = FastAPI(title="RecordIndex 2.0", version="0.1.0")
config = Config.from_env()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_url": config.ollama_base_url,
        "agents": {
            "a0": {"provider": config.a0_provider, "model": config.a0_model},
            "a2": {"provider": config.a2_provider, "model": config.a2_model},
            "a3": {"provider": config.a3_provider, "model": config.a3_model},
            "a4": {"provider": config.a4_provider, "model": config.a4_model},
            "a5": {"provider": config.a5_provider, "model": config.a5_model},
        },
    }


@app.post("/a2/transcribe")
async def a2_transcribe(file: UploadFile = File(...)):
    """
    Transcreve uma imagem de linha manuscrita via A2.

    Body: multipart/form-data com campo 'file' contendo a imagem.
    Retorna: { "filename": "...", "htr_text": "..." }
    """
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    logger.info("A2 transcribe: file=%s provider=%s model=%s", file.filename, config.a2_provider, config.a2_model)
    try:
        model = get_chat_model(config.a2_provider, config.a2_model, config.ollama_base_url)
        agent = A2HTRAgent(model)
        logger.info("A2 invoking model...")
        if config.debug:
            raw, text = agent.transcribe_debug(tmp_path)
            logger.info("A2 raw result: %r", raw)
        else:
            raw, text = None, agent.transcribe(tmp_path)
        logger.info("A2 text: %r", text)
    except Exception as e:
        logger.exception("A2 error")
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    response = {"filename": file.filename, "htr_text": text}
    if config.debug:
        response["debug_raw"] = raw
    return response


@app.post("/pipeline/run")
def pipeline_run():
    """
    Executa o pipeline sobre o conteúdo de volumes/samples/.
    Salva output em volumes/output/output.json.
    Retorna a Collection serializada.
    """
    samples_dir = Path(config.samples_dir)
    if not samples_dir.exists():
        raise HTTPException(status_code=404, detail=f"Samples dir not found: {samples_dir}")

    collection = pipeline.run(str(samples_dir), config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.json"
    result = collection.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(content=result)


@app.get("/pipeline/last-output")
def pipeline_last_output():
    """Retorna o último output.json gerado pelo pipeline."""
    output_path = Path(config.output_dir) / "output.json"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Nenhum output encontrado. Rode /pipeline/run primeiro.")
    with open(output_path, encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))
