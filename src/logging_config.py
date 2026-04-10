"""
Configuração centralizada de logging do RecordIndex 2.0.

Dois handlers:
  - Console (stdout): INFO e acima — visível no `docker compose logs`
  - Arquivo rotativo: DEBUG e acima — persistido em /data/output/logs/pipeline.log

O arquivo sobrevive a restarts do container (volume bind-mount).
Rotação: 10MB por arquivo, 3 backups (pipeline.log, pipeline.log.1, pipeline.log.2).

Uso:
    from src.logging_config import setup_logging
    log_path = setup_logging(output_dir="/data/output")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(output_dir: str = "/data/output") -> Path:
    """
    Configura logging com handler de console + arquivo rotativo.
    Retorna o caminho do arquivo de log.
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Handler de arquivo: DEBUG+, rotativo ---
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    # --- Handler de console: INFO+ ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    # --- Logger raiz do RecordIndex ---
    ri_logger = logging.getLogger("recordindex")
    ri_logger.setLevel(logging.DEBUG)
    # Evitar duplicação se setup_logging for chamado mais de uma vez
    if not ri_logger.handlers:
        ri_logger.addHandler(file_handler)
        ri_logger.addHandler(console_handler)

    # --- Capturar logs do uvicorn/httpx no arquivo também ---
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "httpx"]:
        lg = logging.getLogger(name)
        if not any(isinstance(h, RotatingFileHandler) for h in lg.handlers):
            lg.addHandler(file_handler)

    logging.getLogger("recordindex").info(
        "=== RecordIndex logging iniciado — arquivo: %s ===", log_file
    )
    return log_file
