from dataclasses import dataclass, field


@dataclass
class Line:
    """
    Unidade básica do pipeline: uma linha de texto manuscrito.

    Evolui ao longo do pipeline:
      - image_path: preenchido por A1 (segmentação)
      - htr_text: preenchido por A2 (HTR multimodal)
      - corrected_text: preenchido por A4 (correção estrutural)
      - entities: preenchido por A5 (NER/extração)

    Em v1.0: transcriptionTKBS, transcriptionPYLAIA, transcriptionAICORRECTION.
    Em v2: uma única transcrição HTR via LLM multimodal, mais correção e extração separadas.
    """

    id: str
    image_path: str
    htr_text: str = ""
    corrected_text: str = ""
    bbox: tuple = field(default_factory=tuple)   # (x1, y1, x2, y2) — A1
    entities: dict = field(default_factory=dict) # A5 output

    @property
    def best_text(self) -> str:
        return self.corrected_text or self.htr_text

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "htr_text": self.htr_text,
            "corrected_text": self.corrected_text,
            "bbox": list(self.bbox) if self.bbox else [],
        }
