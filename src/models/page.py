from dataclasses import dataclass, field
from src.models.line import Line


@dataclass
class Page:
    """
    Representa uma página do manuscrito.
    Contém uma lista ordenada de linhas (ordem espacial: de cima para baixo).
    """

    filename: str
    image_path: str
    lines: list = field(default_factory=list)

    def add_line(self, line: Line):
        self.lines.append(line)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "image_path": self.image_path,
            "lines": [line.to_dict() for line in self.lines],
        }
