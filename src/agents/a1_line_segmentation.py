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

# --- Filtros de qualidade de recorte ---
#
# Piso absoluto de largura (A1_MIN_WIDTH): descarta fragmentos irrecuperáveis antes
# de qualquer cálculo estatístico. Não há equivalente para altura/aspect porque a
# detecção IQR é mais robusta que um threshold fixo para essas dimensões.
_MIN_WIDTH = int(os.getenv("A1_MIN_WIDTH", "200"))
#
# Detecção estatística de outliers (IQR por página):
# A1_OUTLIER_SENSITIVITY (k): multiplica o IQR para definir a cerca inferior.
#   Q1 - k * IQR  → abaixo dessa cerca = outlier.
#   k=1.5 padrão estatístico; k=1.0 mais agressivo; k=2.5 mais permissivo.
# A1_OUTLIER_MIN_LINES: mínimo de linhas na página para aplicar o cálculo.
#   Páginas com poucas linhas têm IQR instável; abaixo desse limite, apenas o
#   piso absoluto de largura se aplica.
_OUTLIER_K         = float(os.getenv("A1_OUTLIER_SENSITIVITY",  "1.5"))
_OUTLIER_MIN_LINES = int(os.getenv("A1_OUTLIER_MIN_LINES",      "6"))


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

        # Coletar linhas de texto — ignorar class_id 0 (background)
        # Aplica apenas o piso absoluto de largura; filtragem estatística vem depois.
        candidates: list[tuple] = []
        for class_id, objects in detected_polygons.items():
            if class_id == 0:
                continue
            for obj in objects:
                raw_pts = obj["polygon"]
                pts = np.array([(int(p[0]), int(p[1])) for p in raw_pts], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                if w <= 0 or h <= 0:
                    continue
                if w < _MIN_WIDTH:
                    logger.debug("A1: recorte descartado — largura %dpx < mínimo %dpx", w, _MIN_WIDTH)
                    continue
                centroid_y = y + h / 2
                candidates.append((centroid_y, x, y, w, h))

        # Ordenar de cima para baixo
        candidates.sort(key=lambda c: c[0])
        logger.info("A1: %d linhas candidatas em '%s'", len(candidates), page_stem)

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

        # Detecção estatística de outliers — marca line.is_valid = False
        n_invalid = self._mark_outliers(page)
        n_valid = len(page.lines) - n_invalid
        logger.info(
            "A1: %d linhas em '%s' (%d válidas, %d outliers marcados)",
            len(page.lines), page_stem, n_valid, n_invalid,
        )
        return page

    def _mark_outliers(self, page: Page) -> int:
        """
        Detecta linhas outlier usando IQR por página e marca line.is_valid = False.

        Dimensões analisadas:
          - altura do bbox (h): filtra separadores horizontais e rabiscos finos
          - aspect ratio (w/h): filtra blobs quadrados (carimbos, ornamentos, selos)

        Largura não entra no IQR — a distribuição por página costuma ser bimodal
        (linhas cheias vs. fins de parágrafo), o que tornaria o IQR instável.
        A largura mínima absoluta (_MIN_WIDTH) já cobre os fragmentos mais estreitos.

        Retorna o número de linhas marcadas como inválidas.
        """
        lines = list(page.lines.values())
        if len(lines) < _OUTLIER_MIN_LINES:
            logger.debug(
                "A1: página '%s' tem %d linhas (< %d) — detecção IQR ignorada",
                page.filename, len(lines), _OUTLIER_MIN_LINES,
            )
            return 0

        heights = np.array([l.bbox[3] - l.bbox[1] for l in lines], dtype=float)
        aspects = np.array(
            [(l.bbox[2] - l.bbox[0]) / max(l.bbox[3] - l.bbox[1], 1) for l in lines],
            dtype=float,
        )

        def lower_fence(values: np.ndarray) -> float:
            q1, q3 = np.percentile(values, [25, 75])
            return float(q1 - _OUTLIER_K * (q3 - q1))

        h_fence   = lower_fence(heights)
        asp_fence = lower_fence(aspects)

        logger.debug(
            "A1: '%s' — cercas IQR (k=%.1f): h>%.1fpx  asp>%.2f",
            page.filename, _OUTLIER_K, h_fence, asp_fence,
        )

        n_invalid = 0
        for line, h, asp in zip(lines, heights, aspects):
            reasons = []
            if h < h_fence:
                reasons.append(f"h={h:.0f}px < cerca {h_fence:.1f}")
            if asp < asp_fence:
                reasons.append(f"asp={asp:.2f} < cerca {asp_fence:.2f}")
            if reasons:
                line.is_valid = False
                n_invalid += 1
                logger.debug("A1: outlier '%s' — %s", line.id, ", ".join(reasons))

        return n_invalid
