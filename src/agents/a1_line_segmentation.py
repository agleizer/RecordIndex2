"""
A1 — Segmentação de Linhas (doc-UFCN).

Tipo de agente: Simple Reflex.
Percepção: imagem de página de manuscrito.
Ação: lista de imagens de linha recortadas (uma por bounding box detectado).

Não usa LLM. Usa doc-UFCN — CNN pré-treinada para segmentação de documentos
históricos manuscritos. Modelo: 'generic-historical-line' (HuggingFace).

O modelo é cacheado em HF_HOME (/models/huggingface) via huggingface_hub,
configurado no docker-compose. O serviço 'docufcn-init' baixa o modelo na
primeira inicialização; chamadas seguintes são instantâneas.

Output: Page com Lines ordenadas de cima para baixo. Cada Line tem:
  - image_path: caminho para o recorte salvo em output_dir
  - bbox: (x1, y1, x2, y2) em pixels na imagem original
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from src.models.line import Line
from src.models.page import Page

logger = logging.getLogger("recordindex.a1")


class A1LineSegmentationAgent:
    """
    Detecta linhas de texto em imagens de página de manuscritos históricos.

    Lazy-load: o modelo doc-UFCN é carregado na primeira chamada a segment_page().
    Isso evita atrasos no startup da API quando A1 não é usado naquele run.
    """

    MODEL_NAME = "generic-historical-line"

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        self._parameters = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from doc_ufcn import models as doc_ufcn_models
            from doc_ufcn.main import DocUFCN
        except ImportError as exc:
            raise RuntimeError(
                "doc-ufcn não está instalado. Verifique requirements.txt."
            ) from exc

        logger.info("A1: carregando modelo '%s' (device=%s)...", self.MODEL_NAME, self._device)
        model_path, parameters = doc_ufcn_models.download_model(self.MODEL_NAME)

        model = DocUFCN(len(parameters["classes"]), parameters["input_size"], self._device)
        model.load(model_path, parameters["mean"], parameters["std"])

        self._model = model
        self._parameters = parameters
        logger.info("A1: modelo pronto. Classes: %s", parameters["classes"])

    def segment_page(self, image_path: str, output_dir: str) -> Page:
        """
        Segmenta uma imagem de página em linhas de texto.

        Cada linha detectada é recortada e salva como .jpg em output_dir.
        Retorna uma Page com Lines ordenadas de cima para baixo.

        Args:
            image_path: caminho para a imagem de página (jpg/png/tif).
            output_dir: diretório onde os recortes de linha serão salvos.

        Returns:
            Page com lines populadas (image_path + bbox por linha).
        """
        self._load_model()

        page_stem = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"A1: não foi possível ler imagem: {image_path}")

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = bgr.shape[:2]
        logger.info("A1: segmentando '%s' (%dx%d px)...", page_stem, w_img, h_img)

        detected_polygons, _, _, _ = self._model.predict(
            image_rgb, raw_output=True, mask_output=True, overlap_output=True
        )

        # Dimensão mínima exigida pelo Qwen3VL (SmartResize faz panic abaixo disso)
        MIN_DIM = 32

        # Coletar linhas de texto — ignorar class_id 0 (background)
        candidates: list[tuple] = []
        for class_id, objects in detected_polygons.items():
            if class_id == 0:
                continue
            for obj in objects:
                raw_pts = obj["polygon"]
                pts = np.array([(int(p[0]), int(p[1])) for p in raw_pts], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                # Garantir que bbox está dentro dos limites da imagem
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                if w <= 0 or h <= 0:
                    continue
                # Ignorar recortes menores que o mínimo do VLM
                if w < MIN_DIM or h < MIN_DIM:
                    logger.warning("A1: recorte ignorado — muito pequeno (%dx%d px)", w, h)
                    continue
                centroid_y = y + h / 2
                candidates.append((centroid_y, x, y, w, h))

        # Ordenar de cima para baixo
        candidates.sort(key=lambda c: c[0])
        logger.info("A1: %d linhas detectadas em '%s'", len(candidates), page_stem)

        page = Page(filename=page_stem, image_path=image_path)

        for idx, (_, x, y, w, h) in enumerate(candidates):
            crop = bgr[y : y + h, x : x + w]
            line_id = f"{page_stem}_line_{idx + 1:04d}"
            line_path = os.path.join(output_dir, f"{line_id}.jpg")
            cv2.imwrite(line_path, crop)

            line = Line(
                id=line_id,
                image_path=line_path,
                bbox=(x, y, x + w, y + h),
            )
            page.add_line(line)

        logger.info("A1: %d recortes salvos em '%s'", len(page.lines), output_dir)
        return page
