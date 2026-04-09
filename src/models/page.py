from dataclasses import dataclass, field
from src.models.line import Line


@dataclass
class Page:
    """
    Representa uma página do manuscrito.

    lines é um dict ordenado por line.id (inserção = ordem espacial top→bottom,
    garantida por A1). Usa dict em vez de list para permitir lookup por ID e
    manter o mesmo padrão de dupla-referência da v1.0 (Collection.pages e
    Collection.records apontam para os mesmos objetos Line).
    """

    filename: str
    image_path: str
    lines: dict = field(default_factory=dict)   # {line.id: Line}

    def add_line(self, line: Line):
        line.page_filename = self.filename       # proveniência: linha sabe sua página
        self.lines[line.id] = line

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "image_path": self.image_path,
            "lines": [line.to_dict() for line in self.lines.values()],
        }
