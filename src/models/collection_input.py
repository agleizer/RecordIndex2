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
  record_template:  molde de texto com placeholders (<CAMPO>) para A4 (correção estrutural)
                    se vazio, A4 é pulado no pipeline
                    ver README.md — seção "A4 — Template de correção"
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
    record_template: str = ""        # molde com placeholders para A4 — se vazio, A4 é pulado
    htr_scope: str = "line"          # "line": A2 transcreve linha a linha | "page": A2 transcreve página inteira
