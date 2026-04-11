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
    page_filename: str              # página onde o registro começa
    lines: dict = field(default_factory=dict)             # {line.id: Line}
    corrected_text: str = ""                              # A4: texto corrigido pelo template
    page_text: str = ""                                   # A3 modo page: texto acumulado (pode cruzar páginas)
    structured_output: dict = field(default_factory=dict) # A5: {nome, pai, mãe, data}
    validation: object = None                             # A6: ValidationResult (None se não rodou)

    def add_line(self, line: Line):
        self.lines[line.id] = line

    def get_concatenated_text(self) -> str:
        """
        Retorna o melhor texto disponível para o registro, em ordem de prioridade:
          1. corrected_text  — preenchido por A4 (template)
          2. page_text       — preenchido por A3 modo page (bloco de texto da página)
          3. concatenação de line.best_text — modo line padrão
        Linhas inválidas (is_valid=False) são excluídas da concatenação (modo line).
        """
        if self.corrected_text:
            return self.corrected_text
        if self.page_text:
            return self.page_text
        return " ".join(
            line.best_text
            for line in self.lines.values()
            if line.best_text and line.is_valid
        )

    def last_page_filename(self) -> str:
        """Retorna a página da última linha do registro (pode diferir de page_filename)."""
        if self.lines:
            return list(self.lines.values())[-1].page_filename
        return self.page_filename

    def to_dict(self) -> dict:
        last_page = self.last_page_filename()
        d = {
            "id": self.id,
            "page_filename": self.page_filename,
            "text": self.get_concatenated_text(),
            "corrected_text": self.corrected_text,
            "page_text": self.page_text,
            "lines": [line.to_dict() for line in self.lines.values()],
            "structured_output": self.structured_output,
            "validation": self.validation.to_dict() if self.validation else None,
        }
        if last_page != self.page_filename:
            d["last_page_filename"] = last_page
        return d
