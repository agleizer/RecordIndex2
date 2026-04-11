"""
Pipeline — Semana 3.

Funções de entrada para execução do pipeline via API ou scripts.

run():
  Pipeline mínimo (legado — Semana 1). Imagens de linha pré-recortadas → A2.
  Mantido para testes isolados de A2. Não usa A0.

run_pipeline():  [principal — Semana 3]
  Pipeline completo via A0. Recebe CollectionInput (diretório + metadados) e
  delega tudo ao orquestrador: classificação → A1 → A2 → A3 → A5.
  A0 decide o CollectionConfig internamente.

run_with_orchestrator():
  Mantido para compatibilidade com código de teste existente.
  Aceita CollectionConfig já construída (tipo conhecido externamente).
  Usa A0 mas pula a etapa de classificação.
"""

from pathlib import Path

from src.config import Config
from src.llm_client import get_chat_model
from src.models.collection import Collection
from src.models.collection_config import CollectionConfig
from src.models.collection_input import CollectionInput
from src.models.page import Page
from src.models.line import Line
from src.agents.a2_htr import A2HTRAgent
from src.agents.a0_orchestrator import A0Orchestrator

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def run_pipeline(config: Config, collection_input: CollectionInput) -> Collection:
    """
    Executa o pipeline completo via A0Orchestrator.

    A0 classifica o tipo da coleção, chama A1 para segmentar páginas,
    e coordena A2 → A3 → A5. O diretório de imagens vem de collection_input.image_dir.
    """
    a0 = A0Orchestrator(config)
    return a0.run(collection_input)


def run_with_orchestrator(image_dir: str, config: Config, collection_config: CollectionConfig) -> Collection:
    """
    Executa o pipeline com CollectionConfig já conhecida (tipo pré-definido).

    Útil para testes onde o tipo de coleção é conhecido e não se quer
    a etapa de classificação do A0. Pula A1 — assume imagens de linha pré-recortadas.
    """
    image_files = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"Nenhuma imagem encontrada em: {image_dir}")
        return Collection(name=collection_config.collection_name)

    page = Page(filename=Path(image_dir).name, image_path=image_dir)
    for img_path in image_files:
        line = Line(id=img_path.stem, image_path=str(img_path))
        page.add_line(line)

    # Usar A0 internamente mas injetar CollectionConfig já pronta (pula classificação)
    a0 = A0Orchestrator(config)
    a0._record_counter = 0
    collection = Collection(name=collection_config.collection_name)
    collection.add_page(page)
    _pre, records = a0._transcribe_and_segment(page, collection_config, htr_scope="line")
    for record in records:
        a0._finalize_record(record, collection_config)
        collection.add_record(record)
    return collection


def run(image_dir: str, config: Config) -> Collection:
    """
    Pipeline mínimo (legado — Semana 1): apenas A2 (HTR), sem segmentação nem extração.
    Cada imagem no diretório é tratada como uma linha individual.
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
