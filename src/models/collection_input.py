"""
CollectionInput — entrada externa ao sistema RecordIndex 2.0.

É o que o usuário (via API ou CLI) fornece ao A0.
Diferente de CollectionConfig (que A0 PRODUZ internamente após classificar),
CollectionInput contém apenas o que o usuário sabe/quer informar.

Campos obrigatórios:
  image_dir:        diretório com imagens de página (.jpg/.png/.tif)
  collection_name:  nome legível ("Porto da Cruz Batismos 1860")

Campos opcionais (hints — A0 infere se ausentes):
  year:             ano dos registros ("1860")
  location:         localidade ("Porto da Cruz, Madeira")
  collection_type:  "batismo" | "casamento" | "obito" — se fornecido, A0 usa diretamente
  record_start_hint: expressão típica de início de registro ("Aos")
                     se vazio, A0 usa o padrão do tipo detectado ou infere via LLM

Nota sobre record_start_hint:
  Pode ser simples ("Aos") ou um template com placeholders
  ("Aos [DIA] dias do mês de [MES] do anno de [ANO]...").
  O template longo é mais útil para A4 (correção estrutural) — Semana 4.
  Para A3 (segmentação), basta a palavra inicial.
"""

from dataclasses import dataclass


@dataclass
class CollectionInput:
    image_dir: str
    collection_name: str
    year: str = ""
    location: str = ""
    collection_type: str = ""        # hint opcional — A0 infere se vazio
    record_start_hint: str = ""      # hint opcional — A0 usa padrão se vazio
