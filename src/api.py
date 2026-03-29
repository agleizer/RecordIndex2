"""
RecordIndex 2.0 — API HTTP (FastAPI).

Expõe os agentes e o pipeline via REST para testes com Postman.
Roda na porta 8000 dentro do container (mapeada para localhost:8000).

Endpoints:
  GET  /health                  — liveness check
  POST /a2/transcribe           — transcreve uma imagem de linha (testa A2 isolado)
  POST /a3/segment              — segmenta lista de textos em registros (testa A3 isolado)
  POST /a5/extract              — extrai campos de um registro (testa A5 isolado)
  POST /pipeline/run            — roda pipeline completo (A2+A3+A5) sobre volumes/samples/
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
from src.agents.a3_segmentation import A3RecordSegmentationAgent
from src.agents.a5_ner import A5NERAgent
from src.models.collection_config import CollectionConfig
from src.models.line import Line
from src.models.record import Record
from src.schemas import (
    HealthResponse, AgentConfig, TranscribeResponse,
    SegmentRequest, SegmentResponse,
    ExtractRequest, ExtractResponse,
    PipelineRunRequest,
)
from src import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recordindex.api")

app = FastAPI(title="RecordIndex 2.0", version="0.1.0")
config = Config.from_env()


def _get_collection_config(collection_type: str, collection_name: str = "") -> CollectionConfig:
    """Retorna CollectionConfig para o tipo solicitado. Lança 400 para tipos desconhecidos."""
    factories = {
        "batismo": CollectionConfig.batismo,
        "casamento": CollectionConfig.casamento,
        "obito": CollectionConfig.obito,
    }
    if collection_type not in factories:
        raise HTTPException(
            status_code=400,
            detail=f"collection_type desconhecido: '{collection_type}'. Use: batismo | casamento | obito",
        )
    cfg = factories[collection_type]()
    if collection_name:
        cfg.collection_name = collection_name
    return cfg

if config.debug:
    logging.getLogger("httpx").setLevel(logging.DEBUG)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_url=config.ollama_base_url,
        agents={
            "a0": AgentConfig(provider=config.a0_provider, model=config.a0_model),
            "a2": AgentConfig(provider=config.a2_provider, model=config.a2_model),
            "a3": AgentConfig(provider=config.a3_provider, model=config.a3_model),
            "a4": AgentConfig(provider=config.a4_provider, model=config.a4_model),
            "a5": AgentConfig(provider=config.a5_provider, model=config.a5_model),
        },
    )


@app.post("/a2/transcribe", response_model=TranscribeResponse, response_model_exclude_none=True)
async def a2_transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
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

    return TranscribeResponse(
        filename=file.filename,
        htr_text=text,
        debug_raw=raw if config.debug else None,
    )


@app.post("/a3/segment", response_model=SegmentResponse)
def a3_segment(req: SegmentRequest) -> SegmentResponse:
    """
    Segmenta uma lista de textos de linhas em registros genealógicos.

    Body JSON: { "lines": ["linha 0", "linha 1", ...], "collection_type": "batismo" }
    Retorna: índices de início de cada registro + raciocínio do modelo.
    """
    logger.info("A3 segment: %d linhas, collection_type=%s", len(req.lines), req.collection_type)
    try:
        col_config = _get_collection_config(req.collection_type)
        model = get_chat_model(config.a3_provider, config.a3_model, config.ollama_base_url)
        agent = A3RecordSegmentationAgent(model)

        lines = [Line(id=str(i), image_path="", htr_text=text) for i, text in enumerate(req.lines)]
        boundaries = agent.detect_boundaries(lines, col_config)

        # Contar registros: número de índices válidos após normalização
        starts = sorted(set([0] + [i for i in boundaries.record_start_indices if 0 <= i < len(lines)]))
        logger.info("A3: %d registros, starts=%s, reasoning=%s", len(starts), starts, boundaries.reasoning)
    except Exception as e:
        logger.exception("A3 error")
        raise HTTPException(status_code=502, detail=str(e))

    return SegmentResponse(
        record_start_indices=boundaries.record_start_indices,
        reasoning=boundaries.reasoning,
        num_records=len(starts),
    )


@app.post("/a5/extract", response_model=ExtractResponse)
def a5_extract(req: ExtractRequest) -> ExtractResponse:
    """
    Extrai campos estruturados de um registro genealógico.

    Body JSON: { "record_text": "...", "collection_type": "batismo" }
    Retorna: campos extraídos (nome, pai, mãe, data, etc.) conforme collection_type.
    """
    logger.info("A5 extract: collection_type=%s, text_len=%d", req.collection_type, len(req.record_text))
    try:
        col_config = _get_collection_config(req.collection_type)
        model = get_chat_model(config.a5_provider, config.a5_model, config.ollama_base_url)
        agent = A5NERAgent(model)

        record = Record(id=0, page_filename="")
        line = Line(id="0", image_path="", htr_text=req.record_text)
        record.add_line(line)

        fields = agent.extract(record, col_config)
        logger.info("A5: %s", fields)
    except Exception as e:
        logger.exception("A5 error")
        raise HTTPException(status_code=502, detail=str(e))

    return ExtractResponse(fields=fields, collection_type=req.collection_type)


@app.post("/pipeline/run")
def pipeline_run(req: PipelineRunRequest = None):
    """
    Executa o pipeline completo (A2 → A3 → A5) sobre volumes/samples/.
    Salva output em volumes/output/output.json.

    Body JSON opcional: { "collection_type": "batismo", "collection_name": "..." }
    Se não enviado, usa collection_type="batismo".
    """
    samples_dir = Path(config.samples_dir)
    if not samples_dir.exists():
        raise HTTPException(status_code=404, detail=f"Samples dir not found: {samples_dir}")

    col_type = req.collection_type if req else "batismo"
    col_name = (req.collection_name if req and req.collection_name else "") or samples_dir.name
    col_config = _get_collection_config(col_type, col_name)

    logger.info("Pipeline run: dir=%s, collection_type=%s", samples_dir, col_type)
    collection = pipeline.run_with_orchestrator(str(samples_dir), config, col_config)

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
