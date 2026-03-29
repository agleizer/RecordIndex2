"""
RecordIndex 2.0 — Ponto de entrada.

Uso dentro do container:
    python -m src.main

Uso local (dev, sem Docker):
    OLLAMA_BASE_URL=http://localhost:11434 SAMPLES_DIR=volumes/samples python -m src.main
"""

import json
import sys
from pathlib import Path
from src.config import Config
from src import pipeline


def main():
    config = Config.from_env()

    samples_dir = Path(config.samples_dir)
    if not samples_dir.exists():
        print(f"Erro: diretório de samples não encontrado: {samples_dir}")
        sys.exit(1)

    print("RecordIndex 2.0 — Pipeline Mínimo (Semana 1)")
    print(f"  Input   : {samples_dir}")
    print(f"  Ollama  : {config.ollama_base_url}  modelo_a2={config.a2_model}")
    print(f"  Output  : {config.output_dir}")
    print("-" * 60)

    collection = pipeline.run(str(samples_dir), config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection.to_dict(), f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"Concluído. {collection.total_lines} linhas transcritas.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
