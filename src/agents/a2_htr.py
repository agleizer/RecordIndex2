import base64
from pathlib import Path

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.models.line import Line


class HTROutput(BaseModel):
    text: str


_MEDIA_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".png": "png",
    ".tif": "jpeg", ".tiff": "jpeg",
}


class A2HTRAgent:
    """
    A2 — Agente HTR Multimodal (Simple Reflex).

    Agnóstico ao provider: recebe qualquer BaseChatModel (Ollama, Claude, GPT).
    Structured output via json_schema — Ollama enforça o schema na geração.
    Interface compatível com nós LangGraph via __call__ (para A0, Semana 5).

    O A0 pode escalar para outro provider injetando 'a2_model_override'
    no estado do LangGraph.
    """

    PROMPT = (
        "Você é um especialista em transcrição de manuscritos históricos em português brasileiro. "
        "Transcreva exatamente o texto manuscrito presente nesta imagem de linha. "
        "Preserve a ortografia original, mesmo que arcaica. "
        "Se não conseguir ler alguma palavra, use [?] no lugar."
    )

    def __init__(self, model: BaseChatModel):
        # .bind(think=False) trava o parâmetro antes do with_structured_output.
        # Qwen3.5 em thinking mode não suporta structured output — o bind garante
        # que think=False está em toda invocação mesmo quando wrappado pelo parser.
        self._model = model.bind(think=False).with_structured_output(HTROutput, method="json_schema")

    def _build_message(self, image_path: str) -> HumanMessage:
        suffix = Path(image_path).suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/{media_type};base64,{image_b64}"}},
            {"type": "text", "text": self.PROMPT},
        ])

    def transcribe(self, image_path: str) -> str:
        result: HTROutput = self._model.invoke([self._build_message(image_path)])
        return result.text.strip()

    def transcribe_debug(self, image_path: str):
        """Retorna (result_raw, text) para debug (ativado via DEBUG=true no .env)."""
        result: HTROutput = self._model.invoke([self._build_message(image_path)])
        return repr(result), result.text.strip()

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
