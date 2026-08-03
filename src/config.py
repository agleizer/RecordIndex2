import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Ollama
    ollama_base_url: str

    # Provider e modelo por agente.
    # Provider: "ollama" | "ollama_cloud" | "anthropic" | "openai"
    # O A0 pode escalar para outro provider em runtime via estado do LangGraph.
    a0_provider: str
    a0_model: str
    a2_provider: str
    a2_model: str
    a3_provider: str
    a3_model: str
    a4_provider: str
    a4_model: str
    a5_provider: str
    a5_model: str
    a6_provider: str
    a6_model: str

    # Fallback do A2: acionado quando A6 detecta score < a2_fallback_threshold
    # Se a2_fallback_model for vazio, o fallback está desabilitado
    a2_fallback_provider: str
    a2_fallback_model: str
    a2_fallback_threshold: float

    # API keys para providers externos (opcionais — só necessários se provider != "ollama")
    anthropic_api_key: str
    openai_api_key: str
    ollama_api_key: str  # requerida apenas se algum provider for "ollama_cloud"

    # Diretórios
    input_dir: str
    output_dir: str
    samples_dir: str

    # Filtros de qualidade A3
    # a3_min_htr_chars: descarta linhas com transcrição A2 mais curta que N chars antes do A3.
    #   Evita que ruídos ("A", "†", "") confundam a detecção de limites de registro.
    a3_min_htr_chars: int
    # a3_outlier_multiplier: registros com mais que avg_linhas * k linhas são re-segmentados.
    #   k=0 desabilita. Padrão 2.0 — apenas merges graves são tratados.
    a3_outlier_multiplier: float

    # A6 — thresholds de validação
    # a6_llm_threshold: aciona LLM de validação apenas quando score < threshold.
    #   Aumentar para acionar com mais frequência; diminuir para economizar.
    a6_llm_threshold: float
    # a6_num_predict: limite de tokens gerados pelo LLM no A6.
    #   200 é suficiente para análise concisa; aumentar se notas ficarem truncadas.
    a6_num_predict: int

    # A0 — parâmetros de classificação e segmentação
    # a0_classify_sample_pages: número de páginas amostradas para classificar a coleção via LLM.
    #   Aumentar para coleções mais heterogêneas.
    a0_classify_sample_pages: int
    # a0_avg_lines_window: tamanho da janela de histórico para calcular avg_lines_hint.
    #   Janela menor reage mais rápido a variações; janela maior é mais estável.
    a0_avg_lines_window: int
    # a0_avg_lines_min_history: mínimo de registros fechados antes de usar avg_lines_hint.
    #   Abaixo desse número o hint não é enviado ao A3.
    a0_avg_lines_min_history: int

    # Debug
    debug: bool

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            a0_provider=os.getenv("A0_PROVIDER", "ollama"),
            a0_model=os.getenv("A0_MODEL", "llama3.2"),
            a2_provider=os.getenv("A2_PROVIDER", "ollama"),
            a2_model=os.getenv("A2_MODEL", "qwen3.5:9b"),
            a3_provider=os.getenv("A3_PROVIDER", "ollama"),
            a3_model=os.getenv("A3_MODEL", "llama3.2"),
            a4_provider=os.getenv("A4_PROVIDER", "ollama"),
            a4_model=os.getenv("A4_MODEL", "llama3.2"),
            a5_provider=os.getenv("A5_PROVIDER", "ollama"),
            a5_model=os.getenv("A5_MODEL", "llama3.2"),
            a6_provider=os.getenv("A6_PROVIDER", "ollama"),
            a6_model=os.getenv("A6_MODEL", "llama3.2"),
            a2_fallback_provider=os.getenv("A2_FALLBACK_PROVIDER", ""),
            a2_fallback_model=os.getenv("A2_FALLBACK_MODEL", ""),
            a2_fallback_threshold=float(os.getenv("A2_FALLBACK_THRESHOLD", "0.5")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
            input_dir=os.getenv("INPUT_DIR", "/data/input"),
            output_dir=os.getenv("OUTPUT_DIR", "/data/output"),
            samples_dir=os.getenv("SAMPLES_DIR", "/data/samples"),
            a3_min_htr_chars=int(os.getenv("A3_MIN_HTR_CHARS", "8")),
            a3_outlier_multiplier=float(os.getenv("A3_OUTLIER_MULTIPLIER", "2.0")),
            a6_llm_threshold=float(os.getenv("A6_LLM_THRESHOLD", "0.8")),
            a6_num_predict=int(os.getenv("A6_NUM_PREDICT", "200")),
            a0_classify_sample_pages=int(os.getenv("A0_CLASSIFY_SAMPLE_PAGES", "3")),
            a0_avg_lines_window=int(os.getenv("A0_AVG_LINES_WINDOW", "20")),
            a0_avg_lines_min_history=int(os.getenv("A0_AVG_LINES_MIN_HISTORY", "3")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
