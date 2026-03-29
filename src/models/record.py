from dataclasses import dataclass, field
from src.models.line import Line


@dataclass
class Record:
    """
    Registro genealógico: agrupamento de linhas que formam um único ato
    (ex: batismo, casamento, óbito).

    Linhas agrupadas por A3 (segmentação de registros).
    Saída estruturada gerada por A5 (NER).
    """

    id: int
    page_filename: str              # página onde o registro começa
    lines: list = field(default_factory=list)
    structured_output: dict = field(default_factory=dict)  # A5: {nome, pai, mãe, data}

    def add_line(self, line: Line):
        self.lines.append(line)

    def get_concatenated_text(self) -> str:
        return " ".join(
            line.best_text for line in self.lines if line.best_text
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "page_filename": self.page_filename,
            "text": self.get_concatenated_text(),
            "lines": [line.to_dict() for line in self.lines],
            "structured_output": self.structured_output,
        }
