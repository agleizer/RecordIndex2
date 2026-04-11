from __future__ import annotations
from dataclasses import dataclass, field
from src.models.line import Line


@dataclass
class Record:
    """
    Registro genealógico: agrupamento de linhas que formam um único ato
    (ex: batismo, casamento, óbito).

    lines é um dict {line.id: Line} — os mesmos objetos Line que estão em
    Page.lines. A dupla-referência permite navegar pela hierarquia em ambas
    as direções (Coleção→Página→Linha e Coleção→Registro→Linha).

    page_filename: página onde o registro começa (primeira linha do registro).
    Para registros que cruzam páginas, cada Line.page_filename indica sua página.
    """

    id: int
    page_filename: str              # página da primeira linha do registro
    lines: dict = field(default_factory=dict)             # {line.id: Line}
    corrected_text: str = ""                              # A4: texto corrigido pelo template
    structured_output: dict = field(default_factory=dict) # A5: {nome, pai, mãe, data}
    validation: object = None                             # A6: ValidationResult (None se não rodou)

    def add_line(self, line: Line):
        self.lines[line.id] = line

    def get_concatenated_text(self) -> str:
        """Retorna corrected_text (A4) se disponível, senão concatena line.best_text.
        Linhas marcadas como inválidas (is_valid=False) são excluídas da concatenação."""
        if self.corrected_text:
            return self.corrected_text
        return " ".join(
            line.best_text
            for line in self.lines.values()
            if line.best_text and line.is_valid
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "page_filename": self.page_filename,
            "text": self.get_concatenated_text(),
            "corrected_text": self.corrected_text,
            "lines": [line.to_dict() for line in self.lines.values()],
            "structured_output": self.structured_output,
            "validation": self.validation.to_dict() if self.validation else None,
        }
