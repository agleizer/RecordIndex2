import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Ollama
    ollama_base_url: str

    # Provider e modelo por agente.
    # Provider: "ollama" | "anthropic" | "openai"
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

    # API keys para providers externos (opcionais — só necessários se provider != "ollama")
    anthropic_api_key: str
    openai_api_key: str

    # Diretórios
    input_dir: str
    output_dir: str
    samples_dir: str

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
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            input_dir=os.getenv("INPUT_DIR", "/data/input"),
            output_dir=os.getenv("OUTPUT_DIR", "/data/output"),
            samples_dir=os.getenv("SAMPLES_DIR", "/data/samples"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
