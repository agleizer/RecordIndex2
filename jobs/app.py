"""
RecordIndex 2.0 — Serviço de Jobs (porta 8002).

Fila serial de execuções do pipeline. Recebe pedidos, enfileira, e um único
worker processa um job por vez chamando o POST /pipeline/run do serviço `app`
(síncrono, pode levar horas), segurando até cada um terminar.

Não é mensageria: a "fila" é o conjunto de arquivos <code>.json em /data/jobs,
e o worker é uma thread daemon. O back-end (app/eval) permanece intocado.

Endpoints:
  GET  /health          — liveness
  POST /jobs            — cria um job (body = PipelineRunRequest); devolve o registro
  GET  /jobs            — lista todos os jobs, mais recentes primeiro
  GET  /jobs/{code}     — registro de um job (404 se não existe)
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import store

APP_URL = os.getenv("APP_URL", "http://app:8000")
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
POLL_INTERVAL = float(os.getenv("JOBS_POLL_INTERVAL", "2"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jobs")

_lock = threading.Lock()
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


class JobConfig(BaseModel):
    """Espelha src/schemas.py::PipelineRunRequest. Enviado tal qual ao /pipeline/run."""

    htr_scope: str = "line"
    collection_name: str = "Coleção"
    year: str = ""
    location: str = ""
    collection_type: str = ""          # A0 infere se vazio
    record_start_hint: str = ""
    record_template: str = ""
    image_dir: str = ""
    output_formats: list[str] = ["json", "csv"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _count_images(image_dir: str) -> int:
    if not image_dir:
        return 0
    p = Path(image_dir)
    if not p.exists():
        return 0
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# Worker (fila serial)
# ---------------------------------------------------------------------------

def _next_queued() -> dict | None:
    """Job 'queued' mais antigo, ou None. Ordena por created_at crescente."""
    queued = [j for j in store.load_all() if j.get("status") == "queued"]
    if not queued:
        return None
    return sorted(queued, key=lambda j: j.get("created_at", ""))[0]


def _run_one(job: dict) -> None:
    code = job["code"]
    job["status"] = "running"
    job["started_at"] = _now()
    with _lock:
        store.save(job)
    log.info("job %s: running", code)

    # Foto da config de modelos no momento da execução (não fatal se falhar).
    try:
        r = requests.get(f"{APP_URL}/health", timeout=30)
        r.raise_for_status()
        job["models"] = r.json().get("agents")
    except Exception as e:
        log.warning("job %s: /health falhou (%s), models=None", code, e)
        job["models"] = None
    with _lock:
        store.save(job)

    # Execução do pipeline — chamada síncrona, sem timeout (pode levar horas).
    try:
        r = requests.post(f"{APP_URL}/pipeline/run", json=job["config"], timeout=None)
        r.raise_for_status()
        data = r.json()
        job["output_files"] = data.get("output_files")
        job["num_records"] = len(data.get("records", []) or [])
        job["status"] = "done"
        log.info("job %s: done (%s registros)", code, job.get("num_records"))
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log.exception("job %s: failed", code)
    finally:
        job["finished_at"] = _now()
        with _lock:
            store.save(job)


def _worker() -> None:
    log.info("worker iniciado (APP_URL=%s)", APP_URL)
    while True:
        try:
            job = _next_queued()
        except Exception:
            log.exception("erro ao ler a fila")
            job = None
        if not job:
            time.sleep(POLL_INTERVAL)
            continue
        _run_one(job)


def _recover() -> None:
    """Boot: jobs 'running' interrompidos por restart viram 'failed'."""
    for j in store.load_all():
        if j.get("status") == "running":
            j["status"] = "failed"
            j["error"] = "interrompido por restart"
            j["finished_at"] = _now()
            store.save(j)
            log.info("recuperação: job %s marcado como failed (restart)", j["code"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _recover()
    threading.Thread(target=_worker, name="jobs-worker", daemon=True).start()
    yield


app = FastAPI(title="RecordIndex 2.0 — Jobs", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "app_url": APP_URL}


@app.post("/jobs")
def create_job(cfg: JobConfig):
    with _lock:
        code = store.new_code()
        job = {
            "code": code,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "collection_name": cfg.collection_name,
            "collection_type": cfg.collection_type,
            "config": cfg.model_dump(),
            "models": None,
            "num_images": _count_images(cfg.image_dir),
            "num_records": None,
            "output_files": None,
            "error": None,
        }
        store.save(job)
    log.info("job %s criado (coleção=%s, %d imagens)", code, cfg.collection_name, job["num_images"])
    return job


@app.get("/jobs")
def list_jobs():
    return store.load_all()


@app.get("/jobs/{code}")
def get_job(code: str):
    job = store.load(code)
    if job is None:
        raise HTTPException(status_code=404, detail="job não encontrado")
    return job
