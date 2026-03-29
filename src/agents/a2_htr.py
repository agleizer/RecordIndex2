import base64
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.line import Line


_MEDIA_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".tif": "jpeg", ".tiff": "jpeg",
}


class A2HTRAgent:
    """
    A2 — Agente HTR Multimodal (Simple Reflex).

    Agnóstico ao provider: recebe qualquer BaseChatModel (Ollama, Claude, GPT).
    Interface compatível com nós LangGraph via __call__ (para A0, Semana 5).

    O A0 pode escalar para outro provider injetando 'a2_model_override'
    no estado do LangGraph.
    """

    PROMPT = (
        "Você é um especialista em transcrição de manuscritos históricos em português brasileiro. "
        "Transcreva exatamente o texto manuscrito presente nesta imagem de linha. "
        "Preserve a ortografia original, mesmo que arcaica. "
        "Se não conseguir ler alguma palavra, use [?] no lugar. "
        "Responda apenas com o texto transcrito, sem explicações ou comentários adicionais."
    )

    def __init__(self, model: BaseChatModel):
        self._model = model

    def transcribe(self, image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"},
            },
            {"type": "text", "text": self.PROMPT},
        ])

        result = self._model.invoke([message])
        return result.content.strip()

    def transcribe_debug(self, image_path: str):
        """Retorna (result_raw, text) para debug."""
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self.PROMPT},
        ])
        result = self._model.invoke([message])
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
