"""
Pipeline mínimo — Semana 1.

Fluxo: imagens de linhas → A2 (HTR) → Collection com transcrições.

Não inclui A1 (segmentação de linhas): assume que o input já são
imagens de linhas pré-recortadas. A1 será integrado na Semana 2.
"""

from pathlib import Path
from src.config import Config
from src.llm_client import get_chat_model
from src.models.collection import Collection
from src.models.page import Page
from src.models.line import Line
from src.agents.a2_htr import A2HTRAgent


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def run(image_dir: str, config: Config) -> Collection:
    """
    Executa o pipeline mínimo sobre um diretório de imagens de linha.

    Cada arquivo de imagem em image_dir é tratado como uma linha individual.
    Retorna uma Collection com uma Page contendo todas as linhas transcritas.
    """
    model_a2 = get_chat_model(config.a2_provider, config.a2_model, config.ollama_base_url)
    agent_a2 = A2HTRAgent(model_a2)
    collection = Collection(name=Path(image_dir).name)

    image_files = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"Nenhuma imagem encontrada em: {image_dir}")
        return collection

    page = Page(filename=Path(image_dir).name, image_path=image_dir)

    for img_path in image_files:
        print(f"  [{img_path.name}] transcrevendo...", end=" ", flush=True)
        line = Line(id=img_path.stem, image_path=str(img_path))
        agent_a2.transcribe_line(line)
        print(f'"{line.htr_text}"')
        page.add_line(line)

    collection.add_page(page)
    return collection
