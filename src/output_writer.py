"""
output_writer — grava os resultados do pipeline em um ou mais formatos.

Formatos suportados:
  json  — collection.to_dict() completo, indentado
  csv   — tabela plana (um registro por linha): metadados + structured_output
  txt   — texto corrido, um registro por parágrafo, separados por "---"

Uso:
  files = write_outputs(collection, output_dir, formats=["json", "csv"], base_name="batismos_2026-04-10_14-30")
  # → {"json": Path(...), "csv": Path(...)}
"""

import csv
import json
import re
from pathlib import Path

from src.models.collection import Collection


def _sanitize(name: str) -> str:
    """Converte nome de coleção em string segura para nome de arquivo."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)   # remove caracteres especiais
    name = re.sub(r"[\s]+", "_", name)      # espaços → _
    return name or "collection"


def make_base_name(collection_name: str, timestamp: str) -> str:
    """Ex: 'Porto da Cruz Batismos 1860' + '2026-04-10_14-30' → 'porto_da_cruz_batismos_1860_2026-04-10_14-30'"""
    return f"{_sanitize(collection_name)}_{timestamp}"


def write_outputs(
    collection: Collection,
    output_dir: Path,
    formats: list[str],
    base_name: str,
) -> dict[str, Path]:
    """
    Grava os formatos solicitados e retorna {formato: Path} para os arquivos criados.
    Ignora formatos desconhecidos silenciosamente.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "json":
            written["json"] = _write_json(collection, output_dir, base_name)
        elif fmt == "csv":
            written["csv"] = _write_csv(collection, output_dir, base_name)
        elif fmt == "txt":
            written["txt"] = _write_txt(collection, output_dir, base_name)

    return written


# ------------------------------------------------------------------
# Escritores por formato
# ------------------------------------------------------------------

def _write_json(collection: Collection, output_dir: Path, base_name: str) -> Path:
    path = output_dir / f"{base_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def _write_csv(collection: Collection, output_dir: Path, base_name: str) -> Path:
    """
    Tabela plana — uma linha por registro.

    Colunas fixas: record_id, page, num_lines, text, validation_score, validation_verdict
    Colunas dinâmicas: todos os campos em structured_output (union de todos os registros)
    """
    records = list(collection.records.values())

    # Descobrir todos os campos de structured_output presentes na coleção
    output_fields: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec.structured_output:
            if key not in seen:
                output_fields.append(key)
                seen.add(key)

    fixed_cols = ["record_id", "page", "num_lines", "htr_scope", "text", "validation_score", "validation_verdict"]
    fieldnames = fixed_cols + output_fields

    path = output_dir / f"{base_name}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            val = rec.validation
            row = {
                "record_id":          rec.id,
                "page":               rec.page_filename,
                "num_lines":          len(rec.lines) if rec.lines else None,
                "htr_scope":          "page" if rec.page_text and not rec.lines else "line",
                "text":               rec.get_concatenated_text(),
                "validation_score":   val.score if val else "",
                "validation_verdict": val.verdict if val else "",
            }
            row.update({field: rec.structured_output.get(field, "") for field in output_fields})
            writer.writerow(row)

    return path


def _write_txt(collection: Collection, output_dir: Path, base_name: str) -> Path:
    """Texto corrido — um registro por parágrafo, separados por '---'."""
    path = output_dir / f"{base_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Coleção: {collection.name}\n")
        f.write(f"Total de registros: {len(collection.records)}\n\n")
        for rec in collection.records.values():
            f.write(f"[Registro {rec.id} — {rec.page_filename}]\n")
            f.write(rec.get_concatenated_text())
            f.write("\n---\n\n")
    return path
