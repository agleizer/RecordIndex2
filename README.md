# RecordIndex 2.0

Sistema multi-agente para transcrição automática e indexação de manuscritos históricos genealógicos (registros civis e eclesiásticos em português).

Projeto TCC — Universidade Presbiteriana Mackenzie.
Continuação acadêmica do RecordIndex v1.0 (Iniciação Científica).

---

## Visão geral

O RecordIndex 2.0 substitui a arquitetura CNN-centrada do v1.0 (PyLaia + doc-UFCN + Ollama como corretor) por um pipeline multi-agente baseado em **LLMs multimodais (VLMs)**, onde o Ollama realiza tanto a transcrição HTR quanto a extração estruturada de entidades.

A segmentação de linhas (doc-UFCN) permanece como biblioteca Python dentro do container da aplicação — não é um serviço separado.

---

## Arquitetura — Agentes (A0–A6)

```
Imagem de página
      │
      ▼
  A1 ─ Segmentação de Linhas (doc-UFCN, Simple Reflex)
      │  imagem de página → lista de imagens de linha + bboxes
      ▼
  A2 ─ HTR Multimodal (Ollama VLM, Simple Reflex)
      │  imagem de linha → texto transcrito
      ▼
  A3 ─ Segmentação de Registros (Goal-Based)
      │  linhas → grupos de linhas por registro genealógico
      ▼
  A4 ─ Correção Estrutural (Model-Based Reflex)
      │  texto bruto → texto corrigido (preserva nomes próprios)
      ▼
  A5 ─ NER / Extração (Goal-Based)
      │  texto corrigido → JSON estruturado {nome, pai, mãe, data}
      ▼
  A6 ─ Validação (Goal-Based)
      │  JSON → validação por regras → feedback para A0
      ▼
  A0 ─ Orquestrador (Goal-Based)
         configura pipeline por coleção, recebe feedback de A3/A6
```

**Status de implementação:**

| Agente | Status | Semana |
|--------|--------|--------|
| A2 HTR multimodal | ✅ Implementado (básico) | 1 |
| A1 Segmentação | 🔲 Pendente | 2 |
| A3 Segmentação de registros | 🔲 Pendente | 3 |
| A4 Correção estrutural | 🔲 Pendente | 3 |
| A5 NER/Extração | 🔲 Pendente | 4 |
| A6 Validação | 🔲 Pendente | 4 |
| A0 Orquestrador | 🔲 Pendente | 5 |

---

## Framework de coordenação — LangGraph

Os agentes A1–A6 são implementados como **classes Python puras** (sem dependência de framework). A coordenação e o feedback loop do A0 são implementados com **LangGraph**.

### Por que LangGraph e não Strands Agents

Strands Agents (AWS, mai/2025) foi avaliado e descartado pelos seguintes motivos:

- **A1 não usa LLM**: é chamada de lib Python (doc-UFCN). Strands pressupõe um modelo de linguagem no centro de cada agente. LangGraph trata nós como funções Python puras, sem essa restrição.
- **Auditabilidade acadêmica**: A0 roteia entre A3/A4 com base no feedback de A6. Com LangGraph isso é `add_conditional_edges` — o grafo é o diagrama, o código é a documentação. Com Strands, o roteamento seria decisão do LLM, não-determinístico e difícil de justificar para a banca.
- **Maturidade**: LangGraph tem 80k+ stars, comunidade grande, documentação extensa, empresas em produção. Strands tem < 1 ano de existência e comunidade pequena.
- **Integração Ollama**: ambos têm suporte nativo, sem vantagem decisiva de um sobre o outro.

### Quando LangGraph entra

LangGraph é adicionado na **Semana 5** (implementação do A0). Até lá, os agentes A1–A6 são classes Python chamadas diretamente pelo `pipeline.py`. A interface de cada agente é projetada para ser compatível com nós LangGraph sem refatoração.

Dependências declaradas desde já em `requirements.txt`:
```
langgraph
langchain-core
langchain-ollama
langchain-anthropic
langchain-openai
```

### Abstração de providers — `src/llm_client.py`

Todos os agentes recebem um `BaseChatModel` (LangChain) em vez de URL/modelo hardcoded. A factory `get_chat_model(provider, model, ollama_base_url)` resolve o provider certo:

```python
# mesmo código nos agentes, qualquer provider:
model = get_chat_model("ollama",     "qwen3.5:9b",         ollama_url)
model = get_chat_model("anthropic",  "claude-sonnet-4-6",  ...)
model = get_chat_model("openai",     "gpt-4o",             ...)
```

O A0 pode escalar para um provider externo em runtime injetando `a2_model_override` (ou `aN_model_override`) no estado do LangGraph — sem refatorar os agentes.

> **Qwen3.5 e thinking mode:** modelos da família Qwen3 têm modo *thinking* ativado por padrão. Thinking mode e structured output são **incompatíveis** (documentado pela Alibaba Cloud). Para agentes que usam `with_structured_output`, o parâmetro deve ser travado via `.bind(think=False)` antes do wrapper:
> ```python
> model.bind(think=False).with_structured_output(Schema, method="json_schema")
> ```
> Passar `think=False` só no construtor do `ChatOllama` não é suficiente — é dropado pelo wrapper.

---

## Estrutura do repositório

```
RecordIndex2/
├── docker-compose.yml          # Serviços: app, ollama, ollama-init
├── .env                        # Configuração local (gitignored)
├── .env.example                # Template de configuração (commitado)
├── .gitignore
├── README.md                   # Este arquivo
├── requirements.txt
│
├── docker/
│   └── app/
│       └── Dockerfile          # Container da aplicação Python
│
├── src/
│   ├── api.py                  # API HTTP (FastAPI) — porta 8000
│   ├── main.py                 # Ponto de entrada CLI
│   ├── pipeline.py             # Coordenação do pipeline
│   ├── config.py               # Configuração via variáveis de ambiente
│   ├── llm_client.py           # Factory de providers (Ollama, Anthropic, OpenAI)
│   ├── schemas.py              # Schemas Pydantic compartilhados (API responses)
│   │
│   ├── models/                 # Hierarquia de dados
│   │   ├── line.py             # Linha de texto (unidade básica)
│   │   ├── page.py             # Página do manuscrito
│   │   ├── record.py           # Registro genealógico (grupo de linhas)
│   │   └── collection.py       # Coleção de documentos
│   │
│   └── agents/
│       └── a2_htr.py           # A2: transcrição via Ollama VLM
│
├── evaluation/                 # Módulo de avaliação offline (≠ pipeline)
│   └── __init__.py
│
└── volumes/                    # Dados de execução (bind mounts, gitignored)
    ├── input/                  # Imagens de entrada
    ├── output/                 # Resultados do pipeline (JSON)
    └── samples/                # Imagens de amostra para testes
```

---

## Modelo de dados

```
Collection
  └── pages: list[Page]
        └── lines: list[Line]
              ├── id: str
              ├── image_path: str
              ├── htr_text: str          (A2)
              ├── corrected_text: str    (A4)
              ├── bbox: tuple            (A1)
              └── entities: dict         (A5)
  └── records: list[Record]
        ├── id: int
        ├── page_filename: str
        ├── lines: list[Line]
        └── structured_output: dict    (A5) {nome, pai, mãe, data}
```

**Diferença em relação ao v1.0:** o v1.0 usava `AVLTree` (bintrees) para acesso ordenado por filename. O v2 usa listas Python simples — a ordem é garantida pela ordem de processamento (top-to-bottom na página, que é a ordem natural de doc-UFCN).

---

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- Docker Compose v2 (`docker compose`, não `docker-compose`)
- GPU NVIDIA com drivers atualizados (necessário para inferência dos modelos)

> **Windows:** o repositório precisa estar em uma unidade local (ex: `C:\`). Docker Desktop não consegue fazer bind mount de caminhos de rede UNC (`\\servidor\...`).

---

## Como rodar e testar o A2

### Passo 1 — Configurar o ambiente

```bash
cp .env.example .env
```

Edite o `.env` se necessário. Os valores padrão já apontam para os containers corretos.

### Passo 2 — Subir os serviços

```bash
docker compose up --build
```

O Compose vai executar na seguinte ordem:
1. Build do container `app`
2. Sobe `ollama` e aguarda até ficar saudável
3. `ollama-init` baixa todos os modelos de `OLLAMA_MODELS_PULL` (na primeira execução pode demorar — `qwen3.5:9b` tem ~6.6 GB)
4. Somente após o download completo, o `app` sobe

Acompanhe os logs do download:

```bash
docker compose logs -f ollama-init
```

O `app` está pronto quando aparecer nos logs:

```
app  | INFO:     Application startup complete.
```

Os modelos ficam no named volume `ollama_data` e persistem entre execuções. Nas próximas vezes, o `ollama-init` detecta que os modelos já existem e completa instantaneamente.

### Passo 3 — Verificar que está tudo ok

```
GET http://localhost:8000/health
```

Resposta esperada:
```json
{
  "status": "ok",
  "ollama_url": "http://ollama:11434",
  "agents": {
    "a0": {"provider": "ollama", "model": "llama3.2"},
    "a2": {"provider": "ollama", "model": "qwen3.5:9b"},
    "a3": {"provider": "ollama", "model": "llama3.2"},
    "a4": {"provider": "ollama", "model": "llama3.2"},
    "a5": {"provider": "ollama", "model": "llama3.2"}
  }
}
```

### Passo 4 — Colocar uma imagem de linha manuscrita

Copie uma imagem de linha individual (crop de uma linha do manuscrito) para:

```
volumes/samples/
```

Formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### Passo 5 — Transcrever via Postman

```
POST http://localhost:8000/a2/transcribe
```

- Body: `form-data`
- Campo: `file` | Tipo: `File` | Valor: selecionar a imagem

Resposta esperada:
```json
{
  "filename": "linha_001.jpg",
  "htr_text": "aos vinte dias do mez de janeiro de mil oitocentos"
}
```

> **Nota:** a primeira inferência após subir o container demora mais (~30s para carregar o modelo na GPU). As seguintes são mais rápidas.

---

## API — referência completa

Documentação interativa (Swagger): `http://localhost:8000/docs`

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Liveness check — retorna modelos configurados por agente |
| POST | `/a2/transcribe` | Transcreve uma imagem de linha (testa A2 isolado) |
| POST | `/pipeline/run` | Roda pipeline completo sobre `volumes/samples/` |
| GET | `/pipeline/last-output` | Retorna o último `output.json` gerado |

---

## Rodar sem Docker (desenvolvimento local)

Requer Python 3.11+ e Ollama instalado localmente.

```bash
pip install -r requirements.txt

OLLAMA_BASE_URL=http://localhost:11434 \
SAMPLES_DIR=volumes/samples \
OUTPUT_DIR=volumes/output \
uvicorn src.api:app --reload --port 8000
```

Os demais valores são lidos do `.env` via `python-dotenv`.

---

## Configuração (.env)

Toda a configuração é feita via `.env` na raiz do repositório. Esse arquivo não é commitado — copie o template e ajuste:

```bash
cp .env.example .env
```

| Variável | Valor padrão | Descrição |
|----------|--------------|-----------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL do serviço Ollama |
| `OLLAMA_MODELS_PULL` | `qwen3.5:9b,llama3.2` | Modelos a baixar no startup (comma-separated) |
| `A0_PROVIDER` | `ollama` | Provider do orquestrador (`ollama` \| `anthropic` \| `openai`) |
| `A0_MODEL` | `llama3.2` | Modelo do orquestrador |
| `A2_PROVIDER` | `ollama` | Provider do HTR (deve ser VLM com suporte a imagem) |
| `A2_MODEL` | `qwen3.5:9b` | Modelo do HTR |
| `A3_PROVIDER` | `ollama` | Provider da segmentação de registros |
| `A3_MODEL` | `llama3.2` | Modelo da segmentação de registros |
| `A4_PROVIDER` | `ollama` | Provider da correção estrutural |
| `A4_MODEL` | `llama3.2` | Modelo da correção estrutural |
| `A5_PROVIDER` | `ollama` | Provider do NER/extração |
| `A5_MODEL` | `llama3.2` | Modelo do NER/extração |
| `ANTHROPIC_API_KEY` | _(vazio)_ | Necessário se qualquer provider for `anthropic` |
| `OPENAI_API_KEY` | _(vazio)_ | Necessário se qualquer provider for `openai` |
| `INPUT_DIR` | `/data/input` | Diretório de imagens de entrada (dentro do container) |
| `OUTPUT_DIR` | `/data/output` | Diretório de saída (dentro do container) |
| `SAMPLES_DIR` | `/data/samples` | Imagens de amostra para teste (dentro do container) |

---

## Comandos úteis

```bash
# Subir em background
docker compose up -d

# Ver logs
docker compose logs -f app
docker compose logs -f ollama
docker compose logs -f ollama-init

# Derrubar tudo
docker compose down

# Derrubar e remover volume do Ollama (apaga modelos baixados)
docker compose down -v

# Shell dentro do container app
docker compose exec app bash

# Verificar GPU no container Ollama
docker compose exec ollama nvidia-smi

# Verificar modelos disponíveis
docker compose exec ollama ollama list

# Testar Ollama diretamente (texto)
# Porta 11435 no host — evita conflito com Ollama local instalado no Windows
curl http://localhost:11435/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Olá, tudo bem?",
  "stream": false
}'
```

---

## GPU

O `docker-compose.yml` já configura o Ollama para usar todas as GPUs NVIDIA disponíveis.

**Pré-requisitos no host Windows:**
1. Driver NVIDIA atualizado
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalado no WSL2
3. Docker Desktop com integração WSL2 habilitada

---

## Módulo de avaliação (offline)

O módulo `evaluation/` **não faz parte do pipeline** (A0–A6). É executado separadamente após o pipeline para comparar o output JSON com o ground truth CSV.

Ground truth: arquivos `output.csv` por coleção, com campos `nome, pai, mãe, data`.
Métricas: Precision, Recall, F1 por campo; Jaro-Winkler para nomes.

Documentação completa: `02_desenvolvimento/2026_03_23_protocolo_avaliacao/protocolo_avaliacao.md`

---

## Contexto acadêmico

**RecordIndex v1.0** (IC/IT): pipeline Transkribus → PyLaia HTR → doc-UFCN segmentação → similaridade cosseno para agrupamento → Ollama para correção → export.

**RecordIndex 2.0** (TCC): arquitetura multi-agente onde LLMs multimodais assumem HTR, segmentação de registros e extração de entidades. doc-UFCN permanece para segmentação de linhas (tarefa estrutural onde CNNs ainda são competitivas). A transição é justificada pela hipótese de que VLMs generalizam melhor para scripts históricos sem fine-tuning específico.

Hipótese central: LLMs melhoram a transcrição de texto padrão mas podem degradar nomes próprios (alucinação). O protocolo de avaliação mede isso explicitamente por campo.
