"""
A6 — Validação de Registros.

Tipo de agente: Goal-Based.
Objetivo: verificar se os campos extraídos pelo A5 são coerentes e completos.

Input:  Record com structured_output preenchido + CollectionConfig
Output: ValidationResult anexado ao record (score, verdict, field_errors, notes)

Estratégia em duas camadas:
  1. Regras determinísticas (sempre): campos vazios, ruído HTR, valores suspeitos.
     Rápido, gratuito, auditável.
  2. LLM (lazy): só acionado quando score < LLM_THRESHOLD (0.8).
     Analisa o contexto completo e confirma/descarta os erros detectados pelas regras.

Score (0.0–1.0):
  Começa em 1.0 e é decrementado por cada problema encontrado:
  - Campo obrigatório vazio:      −0.25
  - Ruído HTR no campo:          −0.15
  - Conteúdo suspeito no campo:  −0.10 por ocorrência

Verdict:
  score >= 0.8 → "ok"
  score >= 0.5 → "needs_review"
  score <  0.5 → "failed"

Feedback para A0 (implementação futura com LangGraph):
  record.validation.verdict == "failed" pode acionar re-processamento com
  model_override (ex: VLM cloud para HTR) sem alterar os agentes.
"""

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.llm_client import make_structured
from src.models.collection_config import CollectionConfig
from src.models.record import Record
from src.models.validation_result import ValidationResult
from src.prompts import get_prompt

logger = logging.getLogger("recordindex.a6")

# Valores que indicam ausência de informação (ruído HTR ou A5 sem extração)
_NOISE_VALUES = {
    "", "[?]", "[...]", "?", "...", "não encontrado", "não consta",
    "ilegível", "[ilegível]", "ditto", "dito", "idem", "não informado",
}

# Indicadores de conteúdo temporal para validar o campo data
_DATE_INDICATORS = {
    "dia", "dias", "mês", "mes", "anno", "ano",
    "janeiro", "fevereiro", "março", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
}

# Campos cujo conteúdo é esperado como nome próprio
_NAME_FIELDS = {"nome", "pai", "mae", "noivo", "noiva",
                "pai_noivo", "mae_noivo", "pai_noiva", "mae_noiva"}

# Placeholder não preenchido pelo A4: <CAMPO> ou <CAMPO_COMPOSTO>
_PLACEHOLDER_RE = re.compile(r'<[A-Z][A-Z_]*>')


class _LLMValidationOutput(BaseModel):
    analysis: str          # raciocínio sobre os erros
    confirmed_errors: list[str]  # erros confirmados (descrição curta por erro)


class A6ValidationAgent:

    SCORE_EMPTY = 0.25        # penalidade por campo obrigatório vazio
    SCORE_NOISE = 0.15        # penalidade por ruído HTR no campo
    SCORE_SUSPICIOUS = 0.10   # penalidade por conteúdo suspeito
    SCORE_UNGROUNDED = 0.20   # penalidade por valor não encontrado no texto fonte
    LLM_THRESHOLD = 0.8       # aciona LLM apenas se score < threshold
    _GROUNDING_MIN_WORD = 3   # tamanho mínimo de palavra para verificar grounding

    def __init__(self, model: BaseChatModel):
        # Limitar tokens gerados pelo LLM — análise de validação deve ser concisa
        try:
            from langchain_ollama import ChatOllama
            if isinstance(model, ChatOllama):
                model = model.model_copy(update={"num_predict": 200})
        except ImportError:
            pass
        self._chain = make_structured(model, _LLMValidationOutput)
        self._prompt_template = get_prompt("a6", "validate")

    # ------------------------------------------------------------------
    # Camada 1 — Regras
    # ------------------------------------------------------------------

    def _is_noise(self, value: str) -> bool:
        return value.strip().lower() in _NOISE_VALUES

    def _check_name_field(self, value: str) -> list[str]:
        issues = []
        v = value.strip()
        if len(v) < 3:
            issues.append("muito curto para um nome")
        if v and v[0].islower():
            issues.append("começa com minúscula — possível erro HTR")
        if any(c.isdigit() for c in v):
            issues.append("contém dígitos — provável erro HTR")
        if len(v.split()) > 7:
            issues.append("palavras demais para um nome — provável HTR ruidoso")
        # Nome não deve conter indicadores de data
        v_lower = v.lower()
        if any(ind in v_lower for ind in _DATE_INDICATORS if len(ind) > 2):
            issues.append("contém expressão de data — possível confusão de campo")
        return issues

    def _check_date_field(self, value: str) -> list[str]:
        issues = []
        v = value.lower()
        if not any(ind in v for ind in _DATE_INDICATORS):
            issues.append("sem referência temporal reconhecível")
        if len(value) > 60:
            issues.append("texto longo demais para uma data — provável HTR ruidoso incluído")
        return issues

    def _check_grounded(self, value: str, record_text: str) -> list[str]:
        """
        Verifica se a maioria das palavras significativas do valor aparece no texto fonte.
        Exige >= 50% das palavras encontradas (não apenas uma) para evitar falso grounding
        com HTR ruidoso onde qualquer fragmento pode aparecer por acaso.
        """
        words = [w for w in value.split() if len(w) > self._GROUNDING_MIN_WORD]
        if not words:
            return []  # valor muito curto — não há como verificar
        text_lower = record_text.lower()
        found = sum(1 for w in words if w.lower() in text_lower)
        if found / len(words) < 0.5:
            return ["valor não encontrado no texto fonte — possível alucinação"]
        return []

    def _run_rules(
        self, record: Record, config: CollectionConfig
    ) -> tuple[float, dict[str, list[str]]]:
        score = 1.0
        field_errors: dict[str, list[str]] = {}
        record_text = record.get_concatenated_text()

        for field_name in config.extraction_fields:
            value = record.structured_output.get(field_name, "")
            errors: list[str] = []

            if _PLACEHOLDER_RE.search(value):
                errors.append("placeholder não preenchido — A4 não completou o template")
                score -= self.SCORE_EMPTY
            elif self._is_noise(value):
                errors.append("campo vazio ou sem informação")
                score -= self.SCORE_EMPTY
            else:
                # Verificações específicas por tipo de campo
                if field_name in _NAME_FIELDS:
                    suspicious = self._check_name_field(value)
                    errors.extend(suspicious)
                    score -= self.SCORE_SUSPICIOUS * len(suspicious)
                elif field_name == "data":
                    suspicious = self._check_date_field(value)
                    errors.extend(suspicious)
                    score -= self.SCORE_SUSPICIOUS * len(suspicious)

                # Verificação de grounding: valor deve estar no texto fonte
                ungrounded = self._check_grounded(value, record_text)
                errors.extend(ungrounded)
                score -= self.SCORE_UNGROUNDED * len(ungrounded)

            if errors:
                field_errors[field_name] = errors

        # Cross-field: pai e mae não podem ser iguais ou quase iguais
        pai = record.structured_output.get("pai", "").strip().lower()
        mae = record.structured_output.get("mae", "").strip().lower()
        if pai and mae and pai == mae:
            msg = "pai e mãe com valor idêntico — provável erro de extração"
            field_errors.setdefault("pai", []).append(msg)
            field_errors.setdefault("mae", []).append(msg)
            score -= self.SCORE_SUSPICIOUS * 2

        # Cross-field: nome não deve ser subconjunto significativo de pai ou mae
        # Cobre o caso onde A5 extrai o nome do pai/mãe em vez do batizado.
        nome = record.structured_output.get("nome", "").strip().lower()
        if nome and (pai or mae):
            nome_words = {w for w in nome.split() if len(w) > 3}
            for other_field, other_val in (("pai", pai), ("mae", mae)):
                if not other_val or not nome_words:
                    continue
                other_words = {w for w in other_val.split() if len(w) > 3}
                overlap = nome_words & other_words
                if overlap and len(overlap) / len(nome_words) >= 0.5:
                    msg = f"nome coincide com {other_field} — possível confusão de entidade pelo A5"
                    field_errors.setdefault("nome", []).append(msg)
                    score -= self.SCORE_SUSPICIOUS
                    break

        return max(0.0, round(score, 2)), field_errors

    # ------------------------------------------------------------------
    # Camada 2 — LLM (lazy)
    # ------------------------------------------------------------------

    def _run_llm(
        self,
        record: Record,
        config: CollectionConfig,
        field_errors: dict[str, list[str]],
    ) -> str:
        try:
            prompt = self._prompt_template.format(
                collection_type=config.collection_type,
                record_text=record.get_concatenated_text(),
                structured_output=json.dumps(
                    record.structured_output, ensure_ascii=False, indent=2
                ),
                field_errors=json.dumps(field_errors, ensure_ascii=False, indent=2),
            )
            result: _LLMValidationOutput = self._chain.invoke(
                [HumanMessage(content=prompt)]
            )
            return result.analysis
        except Exception as e:
            logger.warning("A6: LLM validation falhou — %s", e)
            return ""

    # ------------------------------------------------------------------
    # Validação principal
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.8:
            return "ok"
        elif score >= 0.5:
            return "needs_review"
        return "failed"

    def validate(self, record: Record, config: CollectionConfig) -> ValidationResult:
        """
        Valida o structured_output de um registro.

        Sempre roda as regras. Aciona o LLM apenas se score < LLM_THRESHOLD.
        Preenche record.validation in-place e retorna o ValidationResult.
        """
        score, field_errors = self._run_rules(record, config)
        notes = ""

        if score < self.LLM_THRESHOLD and field_errors:
            logger.info(
                "A6: registro %d score=%.2f — acionando LLM para análise",
                record.id, score,
            )
            notes = self._run_llm(record, config, field_errors)

        result = ValidationResult(
            score=score,
            verdict=self._verdict(score),
            field_errors=field_errors,
            notes=notes,
        )
        record.validation = result
        logger.info(
            "A6: registro %d → score=%.2f verdict=%s errors=%s",
            record.id, score, result.verdict, list(field_errors.keys()),
        )
        return result

    def __call__(self, state: dict) -> dict:
        record = state["current_record"]
        collection_config = state["collection_config"]
        self.validate(record, collection_config)
        return {**state, "current_record": record}
