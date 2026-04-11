import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.llm_client import disable_think
from src.models.line import Line
from src.prompts import get_prompt

if TYPE_CHECKING:
    from src.models.page import Page

logger = logging.getLogger("recordindex.a2")


_MEDIA_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".tif": "jpeg", ".tiff": "jpeg",
}

# Substrings do próprio prompt — se aparecerem na saída, o modelo regurgitou o prompt (DT-15)
_PROMPT_LEAK_MARKERS = [
    "especialista em transcrição",
    "manuscritos históricos",
    "Retorne APENAS o texto transcrito",
]

# Labels de degradação de imagem que modelos multimodais produzem em vez de transcrever
_DEGRADATION_LABELS = [
    "Tinta Repassada",
    "Ink Bleed Through",
    "Bleed Through",
]


class A2HTRAgent:
    """
    A2 — Agente HTR Multimodal (Simple Reflex).

    Agnóstico ao provider: recebe qualquer BaseChatModel (Ollama, Claude, GPT).
    NÃO usa structured output — Qwen3.5 com input de imagem ignora format="json"
    e retorna plain text de qualquer forma. O conteúdo da transcrição é lido
    diretamente de result.content.

    Interface compatível com nós LangGraph via __call__ (para A0, Semana 5).
    """

    def __init__(self, model: BaseChatModel):
        self._model = disable_think(model)
        self._prompt = get_prompt("a2", "transcribe").strip()
        self._page_prompt_template = get_prompt("a2", "transcribe_page").strip()

    def _build_message(self, image_path: str) -> HumanMessage:
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self._prompt},
        ])

    def _sanitize(self, text: str, image_name: str) -> str:
        """
        Detecta dois modos de falha do A2 e substitui por [ILEGÍVEL]:

        1. Regurgitação de prompt (DT-15): modelo retorna o próprio prompt quando
           a imagem é ilegível. Detectado por substrings características do prompt.
        2. Labels de degradação: modelos multimodais rotulam imagens degradadas
           com expressões como "Tinta Repassada | Ink Bleed Through" em vez de
           transcrever o texto. São ruído para A3 e A5.
        """
        tl = text.lower()
        for marker in _PROMPT_LEAK_MARKERS:
            if marker.lower() in tl:
                logger.warning("A2: prompt leak detectado em '%s' — substituindo por [ILEGÍVEL]", image_name)
                return "[ILEGÍVEL]"
        for label in _DEGRADATION_LABELS:
            if label.lower() in tl:
                logger.debug("A2: label de degradação em '%s' ('%s') — substituindo por [ILEGÍVEL]", image_name, text[:40])
                return "[ILEGÍVEL]"
        return text

    def transcribe(self, image_path: str) -> str:
        logger.debug("A2: transcrevendo '%s'", Path(image_path).name)
        result = self._model.invoke([self._build_message(image_path)])
        text = self._sanitize(result.content.strip(), Path(image_path).name)
        logger.debug("A2: → '%s'", text[:80] + ("..." if len(text) > 80 else ""))
        return text

    def transcribe_debug(self, image_path: str):
        """Retorna (result_raw, text) para debug (ativado via DEBUG=true no .env)."""
        result = self._model.invoke([self._build_message(image_path)])
        return repr(result), result.content.strip()

    def transcribe_line(self, line: Line) -> Line:
        line.htr_text = self.transcribe(line.image_path)
        return line

    # ------------------------------------------------------------------
    # Modo PAGE — uma chamada por página (htr_scope="page")
    # ------------------------------------------------------------------

    def transcribe_page(self, page: "Page") -> list[str]:
        """
        Transcreve uma página inteira com uma única chamada ao VLM.

        Claude decide quantas linhas há na imagem autonomamente — sem depender
        do count do A1, que é menos preciso que a detecção visual do modelo.

        Retorna a lista de strings transcritas (uma por linha identificada).
        Lista vazia se a página não contiver texto legível.

        Modo PAGE — alternativa mais barata e contextualmente mais rica a
        chamadas individuais transcribe_line(). NÃO altera transcribe_line().
        """
        suffix = Path(page.image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(page.image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self._page_prompt_template},
        ])

        logger.info("A2 [page]: transcrevendo página '%s'", page.filename)
        result = self._model.invoke([message])
        output_lines = [
            self._sanitize(l.strip(), page.filename)
            for l in result.content.strip().splitlines()
            if l.strip()
        ]
        logger.info("A2 [page]: '%s' concluído — %d linhas transcritas", page.filename, len(output_lines))
        return output_lines

    def __call__(self, state: dict) -> dict:
        """
        Interface LangGraph — usado pelo A0 (Semana 5).
        Se 'a2_model_override' estiver no estado, usa esse model em vez do padrão.
        """
        override = state.get("a2_model_override")
        agent = A2HTRAgent(override) if override else self
        line = state["current_line"]
        agent.transcribe_line(line)
        return {**state, "current_line": line}
