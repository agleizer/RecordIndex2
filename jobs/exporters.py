"""
exporters — serializa os registros de um job nos formatos de download.

Registro único de formatos: adicionar um formato novo é escrever a função de
serialização e incluir uma linha em EXPORTERS. O endpoint /export e o
/export/formats leem daqui, e o front descobre os formatos por /export/formats
(nada hardcoded do lado do front). Cada serializador recebe (job, records) e
devolve bytes prontos para download.
"""

import csv
import io
import json
from collections import namedtuple

Exporter = namedtuple("Exporter", "serialize media_type ext label")


def _to_json(job: dict, records: list[dict]) -> bytes:
    return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")


def _to_csv(job: dict, records: list[dict]) -> bytes:
    """Tabela plana. Colunas de campo = união das chaves de `fields` (varia por tipo)."""
    field_keys: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for k in rec.get("fields") or {}:
            if k not in seen:
                seen.add(k)
                field_keys.append(k)
    fixed = ["record_id", "page", "text"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fixed + field_keys, extrasaction="ignore")
    w.writeheader()
    for rec in records:
        row = {"record_id": rec.get("record_id"), "page": rec.get("page"), "text": rec.get("text")}
        row.update(rec.get("fields") or {})
        w.writerow(row)
    return buf.getvalue().encode("utf-8")


def _to_txt(job: dict, records: list[dict]) -> bytes:
    """Texto corrido — um registro por parágrafo, separados por '---'."""
    out = [f"Coleção: {job.get('collection_name', '')}",
           f"Total de registros: {len(records)}", ""]
    for rec in records:
        out.append(f"[Registro {rec.get('record_id')} — {rec.get('page')}]")
        out.append(rec.get("text") or "")
        out.append("---")
        out.append("")
    return "\n".join(out).encode("utf-8")


# Fonte única da verdade dos formatos de export. Adicionar um formato = uma linha.
EXPORTERS: dict[str, Exporter] = {
    "csv":  Exporter(_to_csv,  "text/csv",                  "csv",  "CSV (planilha)"),
    "txt":  Exporter(_to_txt,  "text/plain; charset=utf-8", "txt",  "Texto corrido"),
    "json": Exporter(_to_json, "application/json",          "json", "JSON"),
}
