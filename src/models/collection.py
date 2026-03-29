from dataclasses import dataclass, field
from src.models.page import Page
from src.models.record import Record


@dataclass
class Collection:
    """
    Coleção de documentos: conjunto de páginas e registros extraídos.

    Em v1.0: usava AVLTree por filename para acesso ordenado.
    Em v2: listas simples — ordem de inserção = ordem de processamento = ordem espacial.
    A ordenação é garantida upstream (A1 retorna linhas top-to-bottom).
    """

    name: str
    pages: list = field(default_factory=list)
    records: list = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return sum(len(page.lines) for page in self.pages)

    def add_page(self, page: Page):
        self.pages.append(page)

    def add_record(self, record: Record):
        self.records.append(record)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_pages": len(self.pages),
            "total_lines": self.total_lines,
            "total_records": len(self.records),
            "pages": [page.to_dict() for page in self.pages],
            "records": [record.to_dict() for record in self.records],
        }
