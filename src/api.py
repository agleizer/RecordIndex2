"""
RecordIndex 2.0 — API HTTP (FastAPI).

Expõe os agentes e o pipeline via REST para testes com Postman.
Roda na porta 8000 dentro do container (mapeada para localhost:8000).

Endpoints:
  GET  /health                  — liveness check
  POST /a1/segment              — segmenta uma imagem de página em linhas (testa A1 isolado)
  POST /a2/transcribe           — transcreve uma imagem de linha (testa A2 isolado)
  POST /a3/segment              — segmenta lista de textos em registros (testa A3 isolado)
  POST /a5/extract              — extrai campos de um registro (testa A5 isolado)
  POST /pipeline/run            — pipeline completo via A0 (classifica + A1+A2+A3+A5)
  GET  /pipeline/last-output    — retorna o último output.json gerado
"""

import json
from datetime import datetime
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from src.config import Config
from src.llm_client import get_chat_model
from src.agents.a1_line_segmentation import A1LineSegmentationAgent
from src.agents.a2_htr import A2HTRAgent
from src.agents.a3_segmentation import A3RecordSegmentationAgent
from src.agents.a4_correction import A4CorrectionAgent
from src.agents.a5_ner import A5NERAgent
from src.agents.a6_validation import A6ValidationAgent
from src.models.collection_config import CollectionConfig
from src.models.collection_input import CollectionInput
from src.models.line import Line
from src.models.record import Record
from src.schemas import (
    HealthResponse, AgentConfig, TranscribeResponse,
    LineSegmentInfo, SegmentPageResponse,
    SegmentRequest, SegmentResponse,
    ExtractRequest, ExtractResponse,
    CorrectRequest, CorrectResponse,
    ValidateRequest, ValidateResponse,
    PipelineRunRequest,
)
from src import pipeline
from src.output_writer import make_base_name, write_outputs
from src.logging_config import setup_logging

config = Config.from_env()
_log_file = setup_logging(output_dir=config.output_dir)
logger = logging.getLogger("recordindex.api")

app = FastAPI(title="RecordIndex 2.0", version="0.1.0")


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


@app.post("/a1/segment", response_model=SegmentPageResponse)
async def a1_segment(file: UploadFile = File(...)) -> SegmentPageResponse:
    """
    Segmenta uma imagem de página em linhas de texto via A1 (doc-UFCN).

    Body: multipart/form-data com campo 'file' contendo a imagem de página.
    Retorna: bounding boxes das linhas detectadas e contagem total.
    Os recortes são salvos em /data/output/lines/ dentro do container.
    """
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    logger.info("A1 segment: file=%s", file.filename)
    try:
        agent = A1LineSegmentationAgent()
        crops_dir = str(Path(config.output_dir) / "lines")
        page = agent.segment_page(tmp_path, crops_dir)
        logger.info("A1: %d linhas detectadas", len(page.lines))
    except Exception as e:
        logger.exception("A1 error")
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    lines_info = [
        LineSegmentInfo(id=line.id, bbox=list(line.bbox) if line.bbox else [])
        for line in page.lines.values()
    ]
    return SegmentPageResponse(
        filename=file.filename or "",
        num_lines=len(page.lines),
        lines=lines_info,
        crops_dir=crops_dir,
    )


@app.post("/a2/transcribe", response_model=TranscribeResponse, response_model_exclude_none=True)
async def a2_transcribe(
    file: UploadFile = File(...),
    htr_scope: str = Form("line"),
) -> TranscribeResponse:
    """
    Transcreve uma imagem via A2.

    Modo line (htr_scope="line", padrão):
      Trata a imagem como uma única linha de manuscrito.
      Body: multipart/form-data com campo 'file'.
      Retorna: { "htr_text": "texto transcrito" }

    Modo page (htr_scope="page"):
      Trata a imagem como página inteira. Claude detecta e transcreve
      todas as linhas visíveis — sem depender do count do A1.
      Body: multipart/form-data com campo 'file'.
      Retorna: { "lines": ["linha 0", ...], "n_returned": N }
    """
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    logger.info(
        "A2 transcribe: file=%s scope=%s provider=%s model=%s",
        file.filename, htr_scope, config.a2_provider, config.a2_model,
    )
    try:
        model = get_chat_model(config.a2_provider, config.a2_model, config.ollama_base_url)
        agent = A2HTRAgent(model)

        if htr_scope == "page":
            from src.models.page import Page as PageModel
            page_obj = PageModel(filename=Path(tmp_path).stem, image_path=tmp_path)
            transcribed = agent.transcribe_page(page_obj)  # valid_lines=None → retorna lista
            return TranscribeResponse(
                filename=file.filename or "",
                htr_scope="page",
                lines=transcribed,
                n_returned=len(transcribed),
            )
        else:
            if config.debug:
                raw, text = agent.transcribe_debug(tmp_path)
                logger.info("A2 raw result: %r", raw)
            else:
                raw, text = None, agent.transcribe(tmp_path)
            logger.info("A2 text: %r", text)
            return TranscribeResponse(
                filename=file.filename or "",
                htr_scope="line",
                htr_text=text,
                debug_raw=raw if config.debug else None,
            )
    except Exception as e:
        logger.exception("A2 error")
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/a3/segment", response_model=SegmentResponse, response_model_exclude_none=True)
def a3_segment(req: SegmentRequest) -> SegmentResponse:
    """
    Segmenta texto em registros genealógicos via A3.

    Modo line (htr_scope="line", padrão):
      Body JSON: { "lines": ["linha 0", "linha 1", ...], "collection_type": "batismo" }
      Retorna: record_start_indices (índices 0-based que iniciam cada registro) + reasoning.

    Modo page (htr_scope="page"):
      Body JSON: { "htr_scope": "page", "page_text": "texto...", "collection_type": "batismo" }
      Retorna: record_texts (texto completo de cada registro) + reasoning.
    """
    try:
        col_config = _get_collection_config(req.collection_type)
        model = get_chat_model(config.a3_provider, config.a3_model, config.ollama_base_url)
        agent = A3RecordSegmentationAgent(model)

        if req.htr_scope == "page":
            if not req.page_text.strip():
                raise HTTPException(status_code=400, detail="page_text não pode ser vazio para htr_scope='page'")
            logger.info("A3 segment [page]: %d chars, collection_type=%s", len(req.page_text), req.collection_type)
            blocks = agent.detect_page_blocks(req.page_text, col_config)
            record_texts = [t for t in blocks.record_texts if t.strip()]
            logger.info("A3: %d registros (modo page), reasoning=%s", len(record_texts), blocks.reasoning)
            return SegmentResponse(
                htr_scope="page",
                record_texts=record_texts,
                reasoning=blocks.reasoning,
                num_records=len(record_texts),
            )
        else:
            if not req.lines:
                return SegmentResponse(
                    htr_scope="line",
                    record_start_indices=[],
                    reasoning="Lista de linhas vazia.",
                    num_records=0,
                )
            logger.info("A3 segment [line]: %d linhas, collection_type=%s", len(req.lines), req.collection_type)
            lines = [Line(id=str(i), image_path="", htr_text=text) for i, text in enumerate(req.lines)]
            boundaries = agent.detect_boundaries(lines, col_config)
            starts = sorted(set(i for i in boundaries.record_start_indices if 0 <= i < len(lines)))
            logger.info("A3: %d registros, starts=%s, reasoning=%s", len(starts), starts, boundaries.reasoning)
            return SegmentResponse(
                htr_scope="line",
                record_start_indices=boundaries.record_start_indices,
                reasoning=boundaries.reasoning,
                num_records=len(starts),
                record_texts=None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("A3 error")
        raise HTTPException(status_code=502, detail=str(e))


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


@app.post("/a4/correct", response_model=CorrectResponse)
def a4_correct(req: CorrectRequest) -> CorrectResponse:
    """
    Corrige um texto HTR usando um molde de referência com placeholders.

    Body JSON: { "record_text": "...", "record_template": "Aos <DIA> dias..." }
    Retorna: texto com placeholders preenchidos e texto fixo do molde preservado.
    """
    logger.info("A4 correct: text_len=%d, template_len=%d", len(req.record_text), len(req.record_template))
    try:
        model = get_chat_model(config.a4_provider, config.a4_model, config.ollama_base_url)
        agent = A4CorrectionAgent(model)

        record = Record(id=0, page_filename="")
        line = Line(id="0", image_path="", htr_text=req.record_text)
        record.add_line(line)

        col_config = CollectionConfig(
            collection_type="generic",
            collection_name="",
            record_start_hint="",
            record_template=req.record_template,
        )
        corrected = agent.correct(record, col_config)
        logger.info("A4: corrected_len=%d", len(corrected))
    except Exception as e:
        logger.exception("A4 error")
        raise HTTPException(status_code=502, detail=str(e))

    return CorrectResponse(corrected_text=corrected)


@app.post("/a6/validate", response_model=ValidateResponse)
def a6_validate(req: ValidateRequest) -> ValidateResponse:
    """
    Valida os campos extraídos de um registro genealógico.

    Body JSON: { "record_text": "...", "structured_output": {...}, "collection_type": "batismo" }
    Retorna: score (0–1), verdict (ok/needs_review/failed), field_errors, notes (LLM se score < 0.8).
    """
    logger.info("A6 validate: collection_type=%s", req.collection_type)
    try:
        col_config = _get_collection_config(req.collection_type)
        model = get_chat_model(config.a6_provider, config.a6_model, config.ollama_base_url)
        agent = A6ValidationAgent(model)

        record = Record(id=0, page_filename="")
        line = Line(id="0", image_path="", htr_text=req.record_text)
        record.add_line(line)
        record.structured_output = dict(req.structured_output)

        result = agent.validate(record, col_config)
        logger.info("A6: score=%.2f verdict=%s", result.score, result.verdict)
    except Exception as e:
        logger.exception("A6 error")
        raise HTTPException(status_code=502, detail=str(e))

    return ValidateResponse(
        score=result.score,
        verdict=result.verdict,
        field_errors=result.field_errors,
        notes=result.notes,
    )


@app.post("/pipeline/run")
def pipeline_run(req: PipelineRunRequest = None):
    """
    Executa o pipeline completo via A0Orchestrator sobre volumes/samples/.
    Salva output em volumes/output/output.json.

    Body JSON opcional:
      {
        "collection_name": "Porto da Cruz Batismos 1860",
        "year": "1860",
        "location": "Porto da Cruz",
        "collection_type": "",        // vazio = A0 infere automaticamente
        "record_start_hint": ""       // vazio = A0 usa padrão do tipo detectado
      }

    A0 classifica o tipo (batismo/casamento/obito) via keywords ou LLM,
    chama A1 para segmentar as páginas, e coordena A2 → A3 → A5.
    """
    # Diretório de imagens: req.image_dir > samples_dir (fallback)
    requested_dir = req.image_dir.strip() if req and req.image_dir else ""
    image_dir = Path(requested_dir) if requested_dir else Path(config.samples_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail=f"Image dir not found: {image_dir}")

    col_name = (req.collection_name if req else "") or image_dir.name
    col_input = CollectionInput(
        image_dir=str(image_dir),
        collection_name=col_name,
        year=req.year if req else "",
        location=req.location if req else "",
        collection_type=req.collection_type if req else "",
        record_start_hint=req.record_start_hint if req else "",
        record_template=req.record_template if req else "",
        htr_scope=req.htr_scope if req else "line",
    )

    logger.info("Pipeline run: dir=%s, collection_name=%s, type_hint=%s",
                image_dir, col_input.collection_name, col_input.collection_type)

    try:
        collection = pipeline.run_pipeline(config, col_input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=502, detail=str(e))

    output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_name = make_base_name(collection.name, timestamp)
    formats = req.output_formats if req else ["json"]

    written = write_outputs(collection, output_dir, formats, base_name)
    logger.info("Pipeline output: %s", {fmt: str(p) for fmt, p in written.items()})

    result = collection.to_dict()
    result["output_files"] = {fmt: p.name for fmt, p in written.items()}
    return JSONResponse(content=result)


@app.get("/logs/tail")
def logs_tail(lines: int = 100):
    """
    Retorna as últimas N linhas do arquivo de log persistido.
    Útil para monitorar o pipeline sem acessar o container.

    Query param: ?lines=100 (padrão)
    """
    if not _log_file.exists():
        raise HTTPException(status_code=404, detail="Arquivo de log ainda não criado.")
    try:
        with open(_log_file, encoding="utf-8") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]
        return {"log_file": str(_log_file), "total_lines": len(all_lines), "tail": tail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/last-output")
def pipeline_last_output():
    """Retorna o JSON do output mais recente gerado pelo pipeline."""
    output_dir = Path(config.output_dir)
    json_files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        raise HTTPException(status_code=404, detail="Nenhum output encontrado. Rode /pipeline/run primeiro.")
    with open(json_files[0], encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))
