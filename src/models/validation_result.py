from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Resultado da validação de um registro pelo A6.

    score:        0.0 (falhou) a 1.0 (perfeito) — baseado em regras + LLM
    verdict:      "ok" | "needs_review" | "failed"
    field_errors: {campo: [lista de erros/avisos detectados]}
    notes:        raciocínio do LLM (vazio se regras bastaram)

    Thresholds:
      score >= 0.8 → ok
      score >= 0.5 → needs_review
      score <  0.5 → failed

    Em futuras versões com LangGraph, A0 pode usar verdict == "failed"
    para acionar re-processamento com model_override (ex: VLM cloud).
    """

    score: float
    verdict: str
    field_errors: dict = field(default_factory=dict)  # {campo: [str]}
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "field_errors": self.field_errors,
            "notes": self.notes,
        }
