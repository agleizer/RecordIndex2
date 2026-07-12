"""
store — persistência dos jobs no Postgres.

Uma linha por job na tabela `jobs`: colunas soltas de estado (code, status,
created_at) mais o registro inteiro em `doc` (JSONB). As colunas soltas existem
só para as consultas quentes (ordenação por created_at, filtro por status); a
fonte da verdade é o `doc`, que espelha exatamente o dict que o app.py monta.

Serviço de job único e serial. Worker e endpoints compartilham o banco, mas cada
chamada abre e fecha a própria conexão (conexões psycopg não são thread-safe para
compartilhar entre threads). O lock do app.py continua serializando
new_code()+save() na criação, evitando colisão de código.
"""

import logging
import os
import time
from datetime import datetime

import psycopg
from psycopg.types.json import Json

log = logging.getLogger("jobs.store")

DSN = os.getenv(
    "JOBS_DB_DSN",
    "postgresql://recordindex:recordindex@jobsdb:5432/jobs",
)


def _connect():
    return psycopg.connect(DSN)


def init(retries: int = 30, delay: float = 1.0) -> None:
    """Espera o Postgres subir e garante o schema. Idempotente, chamado no boot."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            with _connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobs (
                            code       TEXT PRIMARY KEY,
                            status     TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            doc        JSONB NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status)"
                    )
                    # Registros extraídos (etapa 2). fields é JSONB porque o
                    # conjunto de campos varia por tipo de coleção: batismo tem
                    # pai/mãe, óbito e casamento têm outros. Uma coluna por campo
                    # não serviria; o jsonb acomoda qualquer tipo na mesma tabela.
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS records (
                            pk         BIGSERIAL PRIMARY KEY,
                            job_code   TEXT NOT NULL REFERENCES jobs(code) ON DELETE CASCADE,
                            record_id  INT,
                            page       TEXT,
                            text       TEXT,
                            fields     JSONB NOT NULL DEFAULT '{}'::jsonb,
                            validation JSONB,
                            UNIQUE (job_code, record_id)
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS records_job_idx ON records (job_code)"
                    )
                conn.commit()
            log.info("store: banco pronto (tabela jobs)")
            return
        except Exception as e:  # Postgres ainda inicializando
            last = e
            log.info("store: aguardando banco (%d/%d): %s", attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError(f"store: banco indisponível após {retries} tentativas: {last}")


def save(job: dict) -> None:
    """Grava o job inteiro (upsert por code). As colunas soltas seguem o doc."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (code, status, created_at, doc)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE
                    SET status = EXCLUDED.status,
                        doc    = EXCLUDED.doc
                """,
                (job["code"], job.get("status", ""), job.get("created_at", ""), Json(job)),
            )
        conn.commit()


def load(code: str) -> dict | None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc FROM jobs WHERE code = %s", (code,))
            row = cur.fetchone()
    return row[0] if row else None


def load_all() -> list[dict]:
    """Todos os jobs, mais recentes primeiro (created_at ISO ordena lexicograficamente)."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc FROM jobs ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [r[0] for r in rows]


def save_records(job_code: str, records: list[dict]) -> None:
    """
    Persiste os registros extraídos de um job. Substitui os anteriores (idempotente
    em re-execução). Cada `rec` é o dict de Record.to_dict() do back-end; guardamos
    o essencial para consulta e export, com structured_output em `fields` (jsonb).
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM records WHERE job_code = %s", (job_code,))
            for rec in records:
                validation = rec.get("validation")
                cur.execute(
                    """
                    INSERT INTO records (job_code, record_id, page, text, fields, validation)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_code,
                        rec.get("id"),
                        rec.get("page_filename"),
                        rec.get("corrected_text") or rec.get("text"),
                        Json(rec.get("structured_output") or {}),
                        Json(validation) if validation is not None else None,
                    ),
                )
        conn.commit()


def load_records(job_code: str) -> list[dict]:
    """Registros de um job, na ordem de extração (record_id crescente)."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT record_id, page, text, fields, validation
                FROM records WHERE job_code = %s ORDER BY record_id
                """,
                (job_code,),
            )
            cols = [c.name for c in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def new_code() -> str:
    """Código YYYYMMDD-HHMMSS, com sufixo -N em caso de colisão no mesmo segundo."""
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    code, n = base, 0
    with _connect() as conn:
        with conn.cursor() as cur:
            while True:
                cur.execute("SELECT 1 FROM jobs WHERE code = %s", (code,))
                if cur.fetchone() is None:
                    return code
                n += 1
                code = f"{base}-{n}"
