import base64
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.llm_client import disable_think
from src.models.line import Line
from src.prompts import get_prompt


_MEDIA_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".tif": "jpeg", ".tiff": "jpeg",
}


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

    def _build_message(self, image_path: str) -> HumanMessage:
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self._prompt},
        ])

    def transcribe(self, image_path: str) -> str:
        result = self._model.invoke([self._build_message(image_path)])
        return result.content.strip()

    def transcribe_debug(self, image_path: str):
        """Retorna (result_raw, text) para debug (ativado via DEBUG=true no .env)."""
        result = self._model.invoke([self._build_message(image_path)])
        return repr(result), result.content.strip()

    def transcribe_line(self, line: Line) -> Line:
        line.htr_text = self.transcribe(line.image_path)
        return line

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
