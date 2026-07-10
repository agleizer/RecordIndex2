"""
store — persistência dos jobs em disco.

Um arquivo por job em JOBS_DIR/<code>.json. O disco é o espelho durável do
estado; a lista é reconstruída a partir dele quando necessário. Serviço de
job único e serial, então não há concorrência entre workers (só o lock do
app.py protege leitura/escrita simultânea entre endpoints e o worker).
"""

import json
import os
from datetime import datetime
from pathlib import Path

JOBS_DIR = Path(os.getenv("JOBS_DIR", "/data/jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def path_for(code: str) -> Path:
    return JOBS_DIR / f"{code}.json"


def save(job: dict) -> None:
    """Grava o job inteiro. Escrita atômica via arquivo temporário + replace."""
    dest = path_for(job["code"])
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)


def load(code: str) -> dict | None:
    p = path_for(code)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all() -> list[dict]:
    """Todos os jobs, mais recentes primeiro (por created_at)."""
    jobs = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


def new_code() -> str:
    """Código YYYYMMDD-HHMMSS, com sufixo -N em caso de colisão no mesmo segundo."""
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    code, n = base, 0
    while path_for(code).exists():
        n += 1
        code = f"{base}-{n}"
    return code
