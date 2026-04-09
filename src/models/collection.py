from dataclasses import dataclass, field
from src.models.page import Page
from src.models.record import Record


@dataclass
class Collection:
    """
    Coleção de documentos: conjunto de páginas e registros extraídos.

    pages e records são dicts ordenados (Python 3.7+ garante inserção-ordenada),
    seguindo o padrão da v1.0 que usava AVLTree. Permite lookup por chave em O(1)
    e mantém a invariante de dupla-referência: Page.lines e Record.lines
    apontam para os mesmos objetos Line.

    pages:   {page.filename: Page}
    records: {record.id: Record}  — IDs são globais, atribuídos por add_record()

    Navegação possível em ambas as direções:
      - Coleção → Páginas → Linhas  (ordem espacial do documento)
      - Coleção → Registros → Linhas (agrupamento lógico, com line.page_filename
                                       para rastrear a página de cada linha)
    """

    name: str
    pages: dict = field(default_factory=dict)    # {page.filename: Page}
    records: dict = field(default_factory=dict)  # {record.id: Record}
    _next_record_id: int = field(default=0, init=False, repr=False)

    @property
    def total_lines(self) -> int:
        return sum(len(page.lines) for page in self.pages.values())

    def add_page(self, page: Page):
        self.pages[page.filename] = page

    def add_record(self, record: Record):
        """
        Adiciona um registro atribuindo um ID global único.
        IDs locais de A3 (0, 1, 2... por página) são substituídos aqui.
        """
        record.id = self._next_record_id
        self.records[record.id] = record
        self._next_record_id += 1

    def get_page(self, filename: str) -> Page | None:
        """Retorna página por filename, ou None se não existir."""
        return self.pages.get(filename)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_pages": len(self.pages),
            "total_lines": self.total_lines,
            "total_records": len(self.records),
            "pages": [page.to_dict() for page in self.pages.values()],
            "records": [record.to_dict() for record in self.records.values()],
        }
