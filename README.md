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
| A2 HTR multimodal | ✅ Implementado e testado | 1 |
| A0 Orquestrador (Python puro) | ✅ Implementado (LangGraph na Semana 5) | 2 |
| A3 Segmentação de registros | ✅ Implementado | 2 |
| A5 NER/Extração | ✅ Implementado | 2 |
| A1 Segmentação de linhas | ✅ Implementado e testado | 3 |
| A4 Correção estrutural | ✅ Implementado (template opcional) | 3 |
| A6 Validação | ✅ Implementado | 3 |

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

> **Qwen3.5 e thinking mode:** modelos da família Qwen3 têm modo *thinking* ativado por padrão. Thinking mode e structured output são **incompatíveis** (documentado pela Alibaba Cloud). O parâmetro `think`/`reasoning` deve ser travado via `model_copy()` na instância antes de construir a chain — `.bind(think=False)` é silenciosamente descartado por `with_structured_output`. Use sempre `make_structured(model, Schema)` de `src/llm_client.py`:
> ```python
> from src.llm_client import make_structured
> self._chain = make_structured(model, MySchema)
> ```
> Em langchain_ollama 1.x o campo se chama `reasoning` (não `think`). `make_structured()` detecta o nome correto em runtime.

---

## Estrutura do repositório

```
RecordIndex2/
├── docker-compose.yml          # Serviços: app, ollama, ollama-init, docufcn-init
├── prompts.yaml                # Prompts de todos os agentes (montado como volume)
├── .env                        # Configuração local (gitignored)
├── .env.example                # Template de configuração (commitado)
├── .gitignore
├── README.md                   # Este arquivo
├── requirements.txt
│
├── docker/
│   ├── app/
│   │   └── Dockerfile          # Container da aplicação Python
│   └── docufcn_init.py         # Script de download do modelo doc-UFCN (roda uma vez)
│
├── src/
│   ├── api.py                  # API HTTP (FastAPI) — porta 8000
│   ├── main.py                 # Ponto de entrada CLI
│   ├── pipeline.py             # run_pipeline(config, collection_input) → Collection
│   ├── config.py               # Configuração via variáveis de ambiente
│   ├── llm_client.py           # Factory de providers + make_structured() + disable_think()
│   ├── schemas.py              # Schemas Pydantic compartilhados (API responses)
│   ├── prompts.py              # Loader singleton de prompts.yaml (get_prompt("a3","segment"))
│   ├── logging_config.py       # setup_logging() — RotatingFileHandler + console
│   │
│   ├── models/                 # Hierarquia de dados
│   │   ├── line.py             # Linha de texto (unidade básica, dual-referenciada)
│   │   ├── page.py             # Página do manuscrito
│   │   ├── record.py           # Registro genealógico (grupo de linhas)
│   │   ├── collection.py       # Coleção de documentos
│   │   ├── collection_config.py  # Contrato A0↔agentes: campos, hints, template
│   │   ├── collection_input.py   # Input bruto do usuário para A0
│   │   └── validation_result.py  # ValidationResult(score, verdict, field_errors, notes)
│   │
│   └── agents/
│       ├── a0_orchestrator.py  # A0: classifica coleção, orquestra A1→A6
│       ├── a1_line_segmentation.py  # A1: doc-UFCN, lazy load, filtro MIN_DIM=32
│       ├── a2_htr.py           # A2: HTR multimodal via VLM (sem structured output)
│       ├── a3_segmentation.py  # A3: segmentação de linhas em registros genealógicos
│       ├── a4_correction.py    # A4: correção estrutural via template (ativação condicional)
│       ├── a5_ner.py           # A5: NER via schema Pydantic dinâmico
│       └── a6_validation.py    # A6: validação rule-based + lazy LLM + grounding check
│
├── evaluation/                 # Módulo de avaliação offline (≠ pipeline)
│   └── __init__.py
│
└── volumes/                    # Dados de execução (bind mounts, gitignored)
    ├── input/                  # Imagens de páginas para processar
    ├── output/                 # Resultados (output.json) e logs (logs/pipeline.log)
    └── samples/                # Imagens de amostra para testes rápidos
```

---

## Modelo de dados

```
Collection
  └── pages: dict[filename → Page]
        └── lines: dict[line.id → Line]   ← mesmo objeto referenciado em Record.lines
              ├── id: str
              ├── image_path: str
              ├── htr_text: str            (A2)
              ├── corrected_text: str      (vazio — A4 opera no nível do Record, não da Line)
              ├── page_filename: str       (proveniência — setado por Page.add_line())
              └── bbox: tuple              (A1)
  └── records: dict[record.id → Record]
        ├── id: int
        ├── page_filename: str
        ├── lines: dict[line.id → Line]   ← mesmos objetos de Page.lines
        ├── corrected_text: str           (A4 — texto corrigido pelo template; vazio se A4 não rodou)
        ├── structured_output: dict       (A5) {nome, pai, mãe, data, ...}
        └── validation: dict              (A6) {score, verdict, field_errors, notes}
```

**Dual-referência:** os mesmos objetos `Line` aparecem tanto em `Page.lines` quanto em `Record.lines`. Isso espelha o padrão AVLTree do v1.0 — navegação possível em ambas as direções sem duplicação de dados.

`Record.get_concatenated_text()` retorna `corrected_text` (A4) se disponível, senão concatena `line.best_text` (que por sua vez prefere `line.corrected_text` sobre `line.htr_text`).

---

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- Docker Compose v2 (`docker compose`, não `docker-compose`)
- GPU NVIDIA com drivers atualizados (necessário para inferência dos modelos)

> **Windows:** o repositório precisa estar em uma unidade local (ex: `C:\`). Docker Desktop não consegue fazer bind mount de caminhos de rede UNC (`\\servidor\...`).

---

## Como rodar

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

---

## Testando com Postman

Documentação interativa (Swagger): `http://localhost:8000/docs`

> **Nota:** a primeira inferência após subir o container demora mais (~30–60s para carregar o modelo na GPU). As seguintes são mais rápidas.

---

### GET /health — verificar que está tudo ok

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
    "a5": {"provider": "ollama", "model": "llama3.2"},
    "a6": {"provider": "ollama", "model": "llama3.2"}
  }
}
```

---

### POST /a2/transcribe — transcrever uma linha manuscrita (testa A2)

Transcreve uma imagem de linha individual (crop de uma linha do manuscrito).

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a2/transcribe`
- Body: `form-data`
- Campo: `file` | Tipo: `File` | Valor: selecionar a imagem

Formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

Coloque a imagem de linha em `volumes/samples/` ou faça upload direto pelo Postman.

**Resposta esperada:**
```json
{
  "filename": "linha_001.jpg",
  "htr_text": "Aos vinte dias do mez de janeiro de mil oitocentos e cincoenta"
}
```

**Com DEBUG=true no .env**, a resposta inclui `debug_raw` com o objeto bruto retornado pelo modelo.

---

### POST /a3/segment — segmentar linhas em registros (testa A3)

Recebe uma lista de textos transcritos e retorna quais índices iniciam novos registros.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a3/segment`
- Body: `raw` → `JSON`

**Body de exemplo (batismo):**
```json
{
  "lines": [
    "Aos quatro dias do mês de abril de mil oitocentos e cincoenta,",
    "baptizei solenemente a Maria, filha legítima de José Ferreira",
    "e de Ana dos Santos. Padrinho: Francisco Alves. Madrina: Rosa.",
    "Aos doze dias do mês de abril de mil oitocentos e cincoenta,",
    "baptizei a João, filho natural de Joaquina de tal.",
    "Nada mais constava. O vigário: Padre Manuel.",
    "Aos vinte dias do mês de abril de mil oitocentos e cincoenta,"
  ],
  "collection_type": "batismo"
}
```

**Resposta esperada:**
```json
{
  "record_start_indices": [0, 3, 6],
  "reasoning": "Cada registro começa com 'Aos X dias do mês...'",
  "num_records": 3
}
```

`collection_type` aceita: `batismo` | `casamento` | `obito`

---

### POST /a4/correct — corrigir texto HTR com template (testa A4)

Recebe o texto bruto de um registro (saída do A2, com erros) e um molde de referência com placeholders. Retorna o texto corrigido.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a4/correct`
- Body: `raw` → `JSON`

**Body de exemplo:**
```json
{
  "record_text": "AAos dezesete dias do mes dAbril do anno de mil eitocentos edessenta, pelo meio dia nesta Igreja Parochial de Nosa Senhora de Quadelupe, baptisei criança do sexo femeneno, a que dei o nome de Maria, filha lisima de Eyidio Francisco Teixeira e de sua mulher Carhota Augusta.",
  "record_template": "Aos <DIA> dias do mês de <MES>, do anno de mil oitocentos e <ANO>, pelo <HORARIO>, n'esta Igreja Parochial Nossa Senhora de Guadelupe, baptisei a uma criança do sexo <SEXO> a que dei o nome de <PRIMEIRO_NOME>, filha legítima de <NOME_PAI> e de sua mulher <NOME_MAE>."
}
```

**Resposta esperada:**
```json
{
  "corrected_text": "Aos dezessete dias do mês de Abril, do anno de mil oitocentos e sessenta, pelo meio dia, n'esta Igreja Parochial Nossa Senhora de Guadelupe, baptisei a uma criança do sexo feminino a que dei o nome de Maria, filha legítima de Eyidio Francisco Teixeira e de sua mulher Carhota Augusta."
}
```

Note que os nomes próprios (`Eyidio Francisco Teixeira`, `Carhota Augusta`) são preservados exatamente como aparecem no texto — A4 não corrige grafias de nomes.

---

### POST /a5/extract — extrair campos de um registro (testa A5)

Recebe o texto de um registro completo e extrai os campos estruturados.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a5/extract`
- Body: `raw` → `JSON`

**Body de exemplo (batismo):**
```json
{
  "record_text": "Aos vinte dias do mez de janeiro de mil oitocentos e cinquenta, na matriz de São Paulo, baptizei a Pedro, filho legítimo de João da Silva e de Maria Antonia.",
  "collection_type": "batismo"
}
```

**Resposta esperada:**
```json
{
  "fields": {
    "nome": "Pedro",
    "pai": "João da Silva",
    "mae": "Maria Antonia",
    "data": "vinte dias do mez de janeiro de mil oitocentos e cinquenta"
  },
  "collection_type": "batismo"
}
```

**Body de exemplo (casamento):**
```json
{
  "record_text": "Aos dois dias do mez de fevereiro de mil oitocentos e sessenta, depois de proclamas, casei Antonio Pereira, filho de Manoel Pereira e de Joanna Maria, com Francisca Gomes, filha de Pedro Gomes e de Clara da Silva.",
  "collection_type": "casamento"
}
```

**Resposta esperada:**
```json
{
  "fields": {
    "noivo": "Antonio Pereira",
    "noiva": "Francisca Gomes",
    "pai_noivo": "Manoel Pereira",
    "mae_noivo": "Joanna Maria",
    "pai_noiva": "Pedro Gomes",
    "mae_noiva": "Clara da Silva",
    "data": "dois dias do mez de fevereiro de mil oitocentos e sessenta"
  },
  "collection_type": "casamento"
}
```

---

### POST /pipeline/run — pipeline completo A1→A2→A3→(A4)→A5→A6

Processa todas as imagens de **página** no diretório especificado. A1 segmenta cada página em linhas, A2 transcreve cada linha, A3 agrupa em registros, A4 corrige (se template fornecido), A5 extrai campos, A6 valida. Salva o resultado em `volumes/output/output.json`.

**Pré-requisito:** colocar imagens de **página** em `volumes/input/` (ou `volumes/samples/`). Não precisa pré-segmentar em linhas — A1 faz isso.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/pipeline/run`
- Body: `raw` → `JSON`

**Body completo:**
```json
{
  "collection_name": "Porto da Cruz Batismos 1866",
  "year": "1866",
  "location": "Porto da Cruz, Madeira",
  "collection_type": "batismo",
  "image_dir": "/data/input",
  "record_template": "",
  "record_start_hint": ""
}
```

Todos os campos são opcionais:
- `collection_type`: `"batismo"` | `"casamento"` | `"obito"` — se vazio, A0 infere automaticamente
- `image_dir`: diretório de imagens dentro do container (padrão: `SAMPLES_DIR` do `.env`)
- `record_template`: molde com placeholders `<CAMPO>` para ativar A4 (se vazio, A4 é pulado)
- `record_start_hint`: expressão de início de registro (ex: `"Aos"`) — se vazio, usa padrão do tipo
- `year` e `location`: metadados que auxiliam A0 na classificação

**Resposta:** JSON completo da Collection com todos os Records, campos extraídos e resultado de validação (score A6). O mesmo JSON é salvo em `volumes/output/output.json`.

> **Atenção:** o pipeline é síncrono — a request bloqueia até terminar. Com qwen3.5:9b, cada linha demora ~40s. Uma página com 30 linhas leva ~20 minutos. Acompanhe o progresso em tempo real via `GET /logs/tail`.

---

### GET /pipeline/last-output — rever o último resultado

```
GET http://localhost:8000/pipeline/last-output
```

Retorna o `output.json` da última execução do pipeline. Útil para inspecionar resultados sem re-executar.

Retorna 404 se nenhum pipeline tiver rodado ainda.

---

### POST /a6/validate — validar campos extraídos (testa A6)

Valida os campos extraídos por A5 contra o texto original do registro.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a6/validate`
- Body: `raw` → `JSON`

**Body de exemplo:**
```json
{
  "record_text": "Aos vinte dias do mez de janeiro de mil oitocentos e cinquenta, na matriz de São Paulo, baptizei a Pedro, filho legítimo de João da Silva e de Maria Antonia.",
  "structured_output": {
    "nome": "Pedro",
    "pai": "João da Silva",
    "mae": "Maria Antonia",
    "data": "vinte dias do mez de janeiro de mil oitocentos e cinquenta"
  },
  "collection_type": "batismo"
}
```

**Resposta esperada:**
```json
{
  "score": 1.0,
  "verdict": "ok",
  "field_errors": {},
  "notes": ""
}
```

`verdict` retorna: `"ok"` (score ≥ 0.8) | `"needs_review"` (0.5–0.8) | `"failed"` (< 0.5)

`notes` é preenchido com análise do LLM apenas quando `score < 0.8`.

---

### GET /logs/tail — acompanhar progresso do pipeline

```
GET http://localhost:8000/logs/tail?lines=100
```

Retorna as últimas N linhas do arquivo de log persistido (`volumes/output/logs/pipeline.log`). Útil para monitorar pipelines longos sem acessar o container.

Os logs incluem marcadores de progresso por agente:
```
── PÁGINA 1/3 ──
[A1] 36 linhas em 2.34s
[A2] linha_0001.jpg → 'Aos oito dias do mes de...' (41.2s)
[A3] 8 registros, starts=[0,4,8,...] (12.1s)
[A4] pulado (sem template)
[A5] record 0 → {nome: Pedro, pai: João...} (8.3s)
[A6] score=0.92 (ok) (1.1s)
══ PIPELINE CONCLUÍDO ══ 3 páginas | 108 linhas | 22 registros | 1847.3s total
```

---

## API — referência rápida

Documentação interativa (Swagger): `http://localhost:8000/docs`

| Método | Endpoint | Body | Descrição |
|--------|----------|------|-----------|
| GET | `/health` | — | Liveness check — retorna modelos configurados por agente (A0–A6) |
| POST | `/a1/segment` | `form-data: file` | Segmenta uma imagem de página em linhas (testa A1 isolado) |
| POST | `/a2/transcribe` | `form-data: file` | Transcreve uma imagem de linha (testa A2 isolado) |
| POST | `/a3/segment` | `{"lines": [...], "collection_type": "batismo"}` | Segmenta textos em registros (testa A3 isolado) |
| POST | `/a4/correct` | `{"record_text": "...", "record_template": "Aos <DIA>..."}` | Corrige texto HTR usando template (testa A4 isolado) |
| POST | `/a5/extract` | `{"record_text": "...", "collection_type": "batismo"}` | Extrai campos de um registro (testa A5 isolado) |
| POST | `/a6/validate` | `{"record_text": "...", "structured_output": {...}, "collection_type": "batismo"}` | Valida campos extraídos (score 0–1, verdict, field_errors) |
| POST | `/pipeline/run` | `{"collection_name": "...", "collection_type": "batismo", "image_dir": "/data/input", ...}` | Roda pipeline completo (A1→A2→A3→A4→A5→A6) |
| GET | `/pipeline/last-output` | — | Retorna o último `output.json` gerado |
| GET | `/logs/tail` | `?lines=100` | Retorna as últimas N linhas do log persistido |

`collection_type` aceita: `batismo` \| `casamento` \| `obito`

Ver seção **Testando com Postman** acima para exemplos completos de body e resposta por endpoint.

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
| `A6_PROVIDER` | `ollama` | Provider da validação |
| `A6_MODEL` | `llama3.2` | Modelo da validação (lazy — só chamado quando score < 0.8) |
| `ANTHROPIC_API_KEY` | _(vazio)_ | Necessário se qualquer provider for `anthropic` |
| `OPENAI_API_KEY` | _(vazio)_ | Necessário se qualquer provider for `openai` |
| `INPUT_DIR` | `/data/input` | Diretório de imagens de entrada (dentro do container) |
| `OUTPUT_DIR` | `/data/output` | Diretório de saída (dentro do container) |
| `SAMPLES_DIR` | `/data/samples` | Imagens de amostra para teste (dentro do container) |
| `DEBUG` | `false` | `true` ativa `debug_raw` nas respostas do A2 e logging httpx verboso |

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

## Escopo: TCC vs Produto

Esta seção documenta decisões de escopo explícitas — o que foi descartado intencionalmente e por quê, para não reabrir discussões resolvidas.

### API síncrona é suficiente para o TCC

O `/pipeline/run` atual é **síncrono**: bloqueia até o pipeline terminar e retorna o resultado. Para um produto real com usuários submetendo documentos, isso seria um problema — o pipeline pode levar minutos para uma página completa.

A arquitetura de produto correta seria:
1. `POST /pipeline/run` retorna imediatamente um `job_id`
2. `GET /pipeline/status/{job_id}` para polling
3. Armazenamento de jobs (SQLite/Redis)
4. Frontend consumindo essa API

**Por que não implementar agora:** o protocolo de avaliação do TCC roda offline — o avaliador lê o `output.json` produzido pelo pipeline, não há usuário em tempo real. Para demonstrar o pipeline e gerar os resultados da avaliação, a API síncrona + Postman é suficiente.

**Quando implementar:** se o projeto evoluir para produto após a entrega. Isso pertence à seção de Trabalhos Futuros do relatório.

### Teste via Postman é intencional

Os agentes são classes Python puras — A3 chama A2 diretamente em Python, sem HTTP. A API HTTP existe para:
- Testes manuais durante o desenvolvimento
- Endpoint `/pipeline/run` para acionar o pipeline de fora
- Exposição futura como serviço

O fluxo interno do pipeline em produção é:
```
pipeline.py → A1 → A2 → A3 → A4 → A5 → A6
```
Nenhuma chamada HTTP entre agentes — tudo em processo.

---

## A4 — Template de correção estrutural

A4 é **opcional**. Só é executado quando o campo `record_template` é fornecido no body do `/pipeline/run`. Sem ele, o pipeline segue diretamente de A3 para A5.

### O que A4 faz

A4 recebe o texto HTR bruto de cada registro (saída do A2, potencialmente com erros de transcrição) e usa um **molde de referência** para produzir um texto corrigido. O LLM preenche os placeholders do molde com as informações extraídas do texto — o texto fixo do molde permanece intacto.

Isso serve para dois propósitos:
1. Normalizar a grafia de termos fixos do documento (datas por extenso, fórmulas repetitivas)
2. Produzir texto mais limpo para o A5 extrair os campos estruturados

### Formato do template

Placeholders são marcados com `<NOME_DO_CAMPO>` (letras maiúsculas, sem espaço). Trechos opcionais usam `<OPT>...</OPT>` — se o texto fonte não contiver aquele trecho, o LLM o remove.

**Exemplo de template para batismo (Porto da Cruz, 1860):**

```
Aos <DIA> dias do mês de <MES>, do anno de mil oitocentos e <ANO>, pelo <HORARIO>,
n'esta Igreja Parochial Nossa Senhora de Guadelupe, freguesia do Porto da Cruz,
Concelho de Machico, Districto Ecclesiastico, e Diocese de Funchal,
eu o Presbytero José Augusto de Freitas Vigário da mesma freguesia,
baptisei solenemente e pus os Santos Oleos a uma criança do sexo <SEXO>
a que dei o nome de <PRIMEIRO_NOME>, filha legítima de <NOME_PAI> e de sua mulher <NOME_MAE>.
<OPT>Foi padrinho <NOME_PADRINHO> e madrinha <NOME_MADRINHA>.</OPT>
Era ut supra.
```

**Placeholders — cada coleção define os seus.** O template acima usa `<DIA>`, `<MES>`, `<ANO>` etc., mas você pode nomear como quiser. O que importa é que o LLM consiga mapear o conteúdo do texto HTR para cada placeholder.

Boas práticas:
- Use nomes descritivos (`<NOME_PAI>` é melhor que `<X2>`)
- Inclua só o texto que realmente é fixo na coleção — variações entre registros devem virar placeholders
- Se um campo é obrigatório, não use `<OPT>`. Se é raro ou ausente em alguns registros, use `<OPT>...<CAMPO>...</OPT>`

### Como usar via API

```json
POST /pipeline/run
{
  "collection_name": "Porto da Cruz Batismos 1860",
  "collection_type": "batismo",
  "record_template": "Aos <DIA> dias do mês de <MES>, do anno de mil oitocentos e <ANO>..."
}
```

Se `record_template` for omitido ou vazio, A4 é pulado — o pipeline roda A1→A2→A3→A5 normalmente.

---

## CollectionConfig — configuração de coleção

`CollectionConfig` é o contrato que permite ao pipeline funcionar com qualquer tipo de coleção sem alteração de código nos agentes.

```python
from src.models.collection_config import CollectionConfig

# Tipos pré-definidos
cfg = CollectionConfig.batismo("Batismos São Paulo 1850–1870")
cfg = CollectionConfig.casamento()
cfg = CollectionConfig.obito()

# Tipo customizado
cfg = CollectionConfig(
    collection_type="inventario",
    collection_name="Inventários 1880",
    record_start_hint="Inventário de",
    extraction_fields={"testador": str, "data": str, "herdeiros": str},
    field_descriptions={
        "testador": "nome do testador (pessoa que fez o inventário)",
        "data": "data do inventário (dia, mês e ano)",
        "herdeiros": "nomes dos herdeiros listados",
    },
)
```

`field_descriptions` é obrigatório para A5 funcionar — sem ele o modelo não sabe o que extrair para cada campo e tende a jogar o texto inteiro no primeiro campo disponível.

Campos de extração por tipo padrão:

| Tipo | Campos |
|------|--------|
| `batismo` | nome (pessoa batizada), pai, mae, data |
| `casamento` | noivo, noiva, pai_noivo, mae_noivo, pai_noiva, mae_noiva, data |
| `obito` | nome (pessoa falecida), pai, mae, data, idade |

### Como A0 classifica a coleção

A0 determina o tipo da coleção (`batismo` | `casamento` | `obito`) em 4 prioridades em cascata, parando na primeira que funcionar:

| Prioridade | Condição | Custo | Método |
|---|---|---|---|
| 1 | `collection_type` explícito no body | Zero | Aceita direto |
| 2 | Keyword no `collection_name` (ex: "Batismos") | Zero | Match em dicionário interno |
| 3 | `record_template` presente | LLM (sem HTR) | Analisa o molde |
| 4 | Nenhuma das anteriores | A1 + A2 + LLM | HTR em páginas aleatórias |

**Prioridade 3** é a mais importante na prática: se o usuário forneceu um template, o LLM consegue inferir o tipo lendo os placeholders e o texto fixo do molde — sem precisar segmentar ou transcrever nenhuma imagem.

**Prioridade 4** usa `random.sample()` para selecionar até 3 páginas aleatórias (não a primeira, que pode ser capa ou índice) e transcreve 2 linhas do meio de cada página (evita cabeçalho e rodapé).

Após classificar, A0 constrói o `CollectionConfig` adequado e orquestra o pipeline completo.

---

### Como A3 usa CollectionConfig

A3 recebe todas as linhas transcritas de uma página e chama o LLM **uma vez** com o contexto da coleção (tipo + `record_start_hint`). O modelo retorna os índices de início de cada registro (`RecordBoundaries`) — não agrupa, apenas detecta fronteiras. O agente faz o corte determinístico.

Uma chamada por página (não por linha) minimiza latência e dá ao modelo contexto completo para detectar padrões inter-linha.

### Como A5 usa CollectionConfig

A5 constrói o schema Pydantic dinamicamente via `pydantic.create_model()`:

```python
NEROutput = create_model(f"NEROutput_{collection_type}", nome=(str, ""), pai=(str, ""), ...)
chain = make_structured(self._base_model, NEROutput)
```

O schema é cacheado por `collection_type`. A instrução de preservar nomes exatamente como aparecem no texto está explícita no prompt — isso é importante para a hipótese acadêmica sobre alucinação em nomes próprios.

---

## Débitos Técnicos

Lista centralizada de decisões adiadas, limitações conhecidas e trabalho futuro. Atualizar a cada sessão de desenvolvimento.

### A2 — sem structured output (Ollama 0.19.0-rc0 + Qwen3.5 + imagem)

**Problema:** Ollama 0.19.0-rc0 ignora o parâmetro `format` quando o input contém imagem. `format=schema_dict` (json_schema), `format="json"` (json_mode) e `method="function_calling"` foram testados — todos falham com input multimodal para Qwen3.5:9b. O modelo transcreve corretamente, mas o parser explode por receber plain text em vez de JSON.

**Decisão:** A2 chama o modelo diretamente sem structured output e retorna `result.content.strip()`. Funciona porque A2 sempre extrai um único campo de texto — não há perda funcional para o TCC.

**Quando retomar:** atualização do Ollama saindo do RC, ou substituição do modelo de HTR por um que suporte tool calling com imagem.

**Status:** workaround em produção desde 29/03/2026.

### Performance — thinking mode do qwen3.5:9b

**Problema:** qwen3.5:9b executa raciocínio interno mesmo com `reasoning=False`. Latência atual: ~25–39s por linha de manuscrito.

**Impacto:** pipeline com 20 linhas por página ≈ 8–13 minutos. Viável para avaliação offline do TCC, mas lento.

**Mitigações a avaliar (Semana 2+):**
- Testar modelo sem thinking nativo (ex: `qwen2.5vl:7b`) — provavelmente mais rápido
- Medir se o thinking melhora a qualidade da transcrição o suficiente para justificar o custo
- Ajustar `num_predict` para limitar geração

**Status:** pendente — avaliar após integrar A1 e medir latência ponta-a-ponta em página real.

### A3 — estratégia de segmentação não escala

**Problema:** A3 recebe todas as linhas de uma página em uma única chamada LLM. Para páginas longas, isso pode exceder o context window do modelo. Para processar registros que cruzam fronteiras de página, a abordagem não funciona.

**Alternativa identificada:** sliding window linha a linha — para cada linha `i`, enviar `[i-1, i, i+1]` e perguntar se inicia novo registro. Mais chamadas LLM, mas chamadas menores e paralelizáveis. Para registros genealógicos com marcadores de início explícitos ("Aos X dias", "Em nome de Deus"), contexto local de 2–3 linhas é suficiente.

**Decisão:** usar abordagem atual (batch por página) para o TCC — o corpus de avaliação tem páginas de tamanho razoável. Documentar como limitação na seção de Discussão.

**Quando implementar:** se houver tempo na Semana 4+, implementar sliding window e comparar resultados. Candidato natural a Trabalhos Futuros.

**Status:** limitação documentada, workaround aceito para o TCC.

### Módulo de avaliação — `evaluation/` (Semana 4)

**Problema:** o pipeline gera `output.json` mas não há nada que compare o resultado com o ground truth e calcule métricas objetivas. Sem isso, a avaliação do TCC é apenas qualitativa.

**O que implementar:**

```
evaluation/
├── evaluator.py        # Comparação output.json × ground_truth.csv → métricas
├── metrics.py          # Precision, Recall, F1, exact match, Jaro-Winkler, segmentação
├── matcher.py          # Estratégias de alinhamento output ↔ GT (ver abaixo)
└── run_evaluation.py   # Entry point CLI: python -m evaluation.run_evaluation
```

**Inputs:**
- `output.json` — saída do pipeline (gerado por `/pipeline/run`)
- `ground_truth.csv` — CSV por coleção com campos `nome, pai, mae, data` (já existente em `_iniciacao/`)

**Outputs:**
- Métricas de segmentação (A3): over-segmentation rate, under-segmentation rate
- Métricas de extração (A5): Precision, Recall, F1 por campo — condicional a alinhamento confiável
- Exact match rate por campo
- Jaro-Winkler para campos de nome (tolerância a erros HTR)
- Relatório em JSON e CSV, com e sem registros flagados

---

#### Problema central: métricas posicionais quebram com hipersegmentação

O matching ingênuo (registro N do output = linha N do GT) colapsa quando A3 hipersegmenta. Se o GT tem N registros e o output tem 2N, o registro correto N+1 fica alinhado com o GT N+1 errado — e a partir daí tudo desalinha em cascata. Isso penaliza A5 por erros de A3, misturando dois problemas distintos:

1. **Qualidade de segmentação** (A3): A3 acertou os boundaries?
2. **Qualidade de extração** (A5): dado um registro, A5 extraiu os campos corretos?

Medir os dois com a mesma métrica posicional produz números impossíveis de interpretar academicamente.

**Abordagens a avaliar (em ordem de complexidade):**

**Opção A — Posicional com filtro de confiança (mais simples):**
Matching posicional padrão, mas antes de calcular métricas de extração, filtra registros flagados (`too_short`, `no_start_hint`, `a6_failed`). Mede dois cenários: "todos os registros" e "apenas registros com alta confiança". Não resolve o desalinhamento, mas isola o efeito.

**Opção B — Best-match por campo-âncora (intermediário):**
Para cada registro do GT, percorre todos os records do output e encontra o que maximiza Jaro-Winkler no campo `nome`. Se o score for acima de um threshold (ex: 0.85), alinha esse par e compara os demais campos. Desacopla extração de posição. Pressuposto: `nome` foi extraído razoavelmente pelo A5 — pode falhar quando A2 alucionou gravemente.

**Opção C — Métricas de segmentação separadas (recomendado para o TCC):**
Medir A3 e A5 de forma independente:
- *Segmentation recall*: que fração dos registros do GT tem ao menos um output record cujo texto concatenado contém o `nome` do GT? (detectou o registro, mesmo que dividido)
- *Segmentation precision*: que fração dos output records corresponde a um registro real e não é fragmento?
- *Over-segmentation rate*: `len(output_records) / len(gt_records)` — ideal = 1.0
- *Extraction F1*: calculado apenas sobre pares que conseguiram ser alinhados com Opção B

Essa separação permite dizer "A3 tem recall de 80% mas over-segmentation de 1.8× — A5 tem F1 de 0.72 nos registros corretamente segmentados" — o que é muito mais útil academicamente do que um único número que mistura tudo.

**Decisão:** implementar Opção A como baseline (rápido, suficiente para uma primeira leitura) e Opção C para a versão final do TCC. Opção B como fallback se C for muito complexa de implementar.

---

#### Guardrails de segmentação — flags no output

Para suportar as métricas acima, adicionar campo `flags: list[str]` em `Record` (não no pipeline de produção, mas como metadado opcional do avaliador ou do A0 pós-processamento):

| Flag | Condição | Significado |
|---|---|---|
| `too_short` | `len(record.lines) < 4` | Provável fragmento |
| `no_start_hint` | Nenhuma linha contém `record_start_hint` | Boundary suspeito |
| `a6_failed` | `validation.verdict == "failed"` | A6 identificou problema grave |
| `possible_merge` | `len(record.lines) > 15` | Possível fusão de dois registros |

O avaliador pode filtrar por flags e comparar métricas com/sem — dá uma leitura de "melhor caso" vs. "caso real".

**Por que não faz parte do pipeline:** é ferramenta acadêmica de avaliação offline, não parte do sistema. Não deve interferir nos agentes A0–A6.

**Status:** pendente — Semana 4. É o entregável acadêmico mais importante para a defesa. Design das métricas (Opção A vs. C) a decidir no início da semana.

---

### Output nomeado por coleção e timestamp

**Problema:** o pipeline sempre salva em `output.json`, sobrescrevendo o resultado anterior. Impossível comparar execuções diferentes ou manter histórico de experimentos.

**O que deveria ser:** `{collection_name}_{YYYY-MM-DD_HH-MM}.json`

Exemplo: `PortoDaCruz_Batismos_1866_2026-04-09_14-32.json`

**Onde mudar:** `src/api.py` → `pipeline_run()` — linha que define `output_path`. Também deveria retornar o nome do arquivo na resposta JSON para o cliente saber onde buscar.

**Decisão:** pequena mudança, alto valor. Implementar junto com o módulo de avaliação na Semana 4 — o avaliador precisa localizar o arquivo de output por nome.

**Status:** pendente — Semana 4.

---

### Frontend Streamlit (Trabalho Futuro)

**Problema:** interação atual via Postman/API é adequada para desenvolvimento mas inacessível para usuários não-técnicos (pesquisadores de genealogia, arquivistas).

**Proposta:** frontend em Streamlit — escolha pragmática (Python puro, sem JavaScript, fácil de subir como container adicional).

**Fluxo mínimo:**
1. Upload de imagens de página (múltiplos arquivos)
2. Configurar coleção (nome, tipo, template opcional)
3. Botão "Processar" → aciona `/pipeline/run` via HTTP
4. Progress bar lendo `/logs/tail` via polling
5. Exibir resultado estruturado em tabela (registros × campos)
6. Download do `output.json` e do relatório de avaliação

**Complicadores:**
- `/pipeline/run` é síncrono — Streamlit vai bloquear na chamada. Precisaria de job assíncrono (ver DT de pipeline assíncrono) ou `st.spinner` com timeout generoso
- Upload de imagens precisa de endpoint dedicado que aceite múltiplos arquivos e os salve em `volumes/input/` antes de rodar o pipeline
- Se rodar em container separado: adicionar serviço `frontend` ao `docker-compose.yml`, porta 8501

**Decisão:** fora do escopo do TCC. Mencionável em Trabalhos Futuros como direção natural de produto. Não implementar antes de ter avaliação e escrita completas.

**Status:** ideia registrada. Retomar após entrega do TCC.

---

### Pipeline assíncrono (jobs + polling)

**Problema:** `/pipeline/run` é síncrono. Para uso interativo com UI, seria necessário job_id + polling.

**Decisão:** fora do escopo do TCC. Pré-requisito para o frontend Streamlit funcionar bem. Implementar como Trabalho Futuro.

### Experimento: granularidade de input do A2 (página vs linha vs bloco)

**Problema:** A2 recebe crops de linha individualmente (saída do A1). Não foi avaliado se passar blocos de texto ou a página inteira melhoraria a qualidade da transcrição — o contexto visual maior pode ajudar o modelo a interpretar palavras ambíguas.

**Status:** pendente — experimento possível após pipeline E2E funcionar.

### CollectionConfig hardcoded — deveria ser dinâmico via A0

**Problema:** os tipos de coleção (`batismo`, `casamento`, `obito`), seus campos de extração e as descrições de cada campo estão hardcoded como factories estáticas em `src/models/collection_config.py`. Para suportar um tipo novo é necessário alterar o código.

**O que deveria acontecer:** o orquestrador A0 deveria receber o tipo de coleção como parâmetro e montar o `CollectionConfig` dinamicamente — via arquivo de configuração externo (YAML/JSON) ou banco de dados. Assim nenhum agente precisaria ser alterado para suportar uma nova coleção.

**Decisão:** manter hardcoded para o TCC — o corpus de avaliação é fixo (batismo, casamento, óbito). Avaliar na Semana 5 ao implementar A0. Candidato a Trabalhos Futuros.

**Status:** limitação documentada, aceita para o TCC.

### A0 auto-configuração completa — problema do ciclo de dependência

**Problema:** há um ciclo de dependência difícil de quebrar.

- Para classificar corretamente e extrair os campos certos, o agente precisa entender o texto.
- Para entender o texto bem (via A4), precisa do template — que é conhecimento humano.
- Para gerar o template automaticamente, precisaria já ter transcrito e compreendido o texto.
- A transcrição (A2) é mais confiável quando sabe o que está procurando.

Em outras palavras: classificar ↔ transcrever são tarefas mutuamente dependentes. A solução ideal seria um processo iterativo (transcreve com qualidade básica → infere estrutura → refina transcrição), mas isso aumentaria significativamente a complexidade do A0 e do pipeline.

**Decisão:** para o TCC, o usuário fornece `collection_type` (ou o nome deixa claro) e opcionalmente `record_template`. A0 usa essas informações como ponto de partida. A auto-configuração completa é Trabalho Futuro — e representa uma contribuição de pesquisa por si só.

**Status:** registrado como débito conceitual / direção de pesquisa futura.

---

## Débitos Técnicos — Pós-avaliação do 2º E2E (09/04/2026)

> **Contexto:** segundo teste E2E completo, 3 páginas reais da coleção PortoDaCruz Batismos 1866 (arquivos `_0002`, `_0003`, `_0004`). 108 linhas segmentadas, 22 registros gerados, score médio A6 = **0.42** (ok: 1 / needs_review: 8 / failed: 13). Pipeline estava com todos os agentes ativos (A1→A2→A3→A5→A6, A4 inativo por ausência de template). Data do teste: 09/04/2026. Ponto do desenvolvimento: fim da Semana 3, todos os agentes A1–A6 implementados.

---

### DT-06 — Qualidade HTR (A2): qwen3.5:9b alucina em manuscrito cursivo histórico

**Descoberto em:** 09/04/2026 — segundo E2E com 3 páginas reais.

**Problema:** qwen3.5:9b produz alucinações graves quando a imagem de linha é ambígua ou de baixa qualidade:
- Gerou texto em **cirílico** (russo) para uma linha de manuscrito português: `"Дода деловодовъ и въ дѣлѣ ревизии"`
- Gerou linguagem **coloquial moderna**: `"acho que é bem provável que a gente vá ter"`
- Inventou **anos e meses inexistentes**: `"18 de Lymasbo de 1831"`, `"18?? (infelizmente não sei qual ano)"`
- Copiou estruturas de frases modernas sem relação com o conteúdo do manuscrito

**Impacto:** quando A2 alucina, tudo downstream é comprometido — A3 segmenta errado, A5 extrai campos errados, A6 dá score alto a campos inventados (o valor existe no texto alucinado, passa no grounding check). Efeito cascata total.

**Causa provável:** qwen3.5:9b não foi treinado em manuscritos históricos cursivos do século XIX em português. Para imagens de baixa legibilidade, o modelo "completa" o texto com o que faz sentido para ele estatisticamente — que não é o conteúdo do manuscrito.

**Sugestões de correção:**
1. **Trocar o modelo base (prioritário):** testar `minicpm-v:8b`, `llava:13b`, `qwen2.5vl:7b` — comparar CER em amostra manual de 10 linhas com ground truth. A hipótese é que o problema é específico do qwen3.5:9b e outros VLMs podem ser mais conservadores.
2. **Prompt de contenção:** adicionar instrução explícita no prompt A2 para retornar `[ilegível]` quando não conseguir transcrever com confiança, em vez de inventar. Avaliar se o modelo respeita essa instrução.
3. **Filtro de alucinação pós-HTR:** detectar sinais de alucinação no resultado do A2 — presença de caracteres não-latinos, proporção de palavras do dicionário português, comprimento anômalo em relação ao bbox. Rejeitar e marcar como `[ilegível]`.
4. **Fine-tuning (Trabalho Futuro):** fine-tune de VLM em corpus anotado de manuscritos luso-brasileiros. Fora do escopo do TCC mas é a solução definitiva.

**Status:** limitação crítica — impacta diretamente a qualidade do output. A ser atacada na Semana 4 com experimento de modelos alternativos.

---

### DT-07 — A3: hipersegmentação — registros de 2 linhas são quase sempre fragmentos

**Descoberto em:** 09/04/2026 — segundo E2E.

**Problema:** das 22 registros gerados, vários têm apenas 2 linhas. Um batismo de 1866 em formato completo tem 6–10 linhas (data, nome, filiação, padrinhos, celebrante). Registros de 2 linhas quase invariavelmente são:
- Fragmento do final de um registro anterior (DT-01 — quebra de página)
- Linha marginal/número de registro isolado (ex: `"N. 1."`, `"=4.º"`)
- Texto que o A2 aluciou, criando dois fragmentos sem coerência

A3 está sendo conservador demais — prefere criar muitos registros pequenos a arriscar fundir dois registros reais.

**Impacto:** A5 e A6 trabalham em vão em registros espúrios. Score médio cai por registros inviáveis.

**Sugestões de correção:**
1. **Tamanho mínimo de registro:** pós-processamento em A0 — descartar registros com menos de N linhas (ex: < 4) e registrar como `fragmento` no output. Simples de implementar, sem custo LLM.
2. **Prompt do A3 com restrição explícita:** adicionar ao prompt que um registro de batismo tem tipicamente 5–8 linhas, e que fragmentos de 1–2 linhas devem ser agrupados ao registro anterior se não houver marcador claro de início.
3. **Sliding window (ver DT-A3 existente):** abordagem mais robusta, mas mais cara.

**Status:** limitação documentada. Correção simples (tamanho mínimo) pode ser implementada em Semana 4 como pós-processamento.

---

### DT-08 — A0/A3: primeira página pode ser frontispício ou termo de abertura

**Descoberto em:** 09/04/2026 — segundo E2E. Página `_0002` é o **termo de abertura do livro** ("Será este Livro para o registo paroquial dos Baptismos..."), não registros reais. A3 tentou segmentá-la como batismos, gerando 5 registros espúrios (Rec 0–4).

**Problema:** A0 envia todas as páginas indiscriminadamente para A1→A2→A3. Não há filtro para páginas introdutórias, índices, termos de encerramento ou páginas em branco.

**Impacto:** registros espúrios inflam o total, derrubam o score médio, poluem o output.

**Sugestões de correção:**
1. **Filtro de página via LLM:** após A2, antes de A3, pedir ao LLM se a página contém "registros genealógicos individuais" ou "texto administrativo/introdutório". Se for administrativo, marcar como `skip_record_segmentation=True` e não chamar A3. Custo: 1 chamada LLM por página.
2. **Heurística por marcador de início:** se nenhuma linha da página contém o `record_start_hint` da coleção ("Aos X dias", "Em nome de Deus"), a página provavelmente é administrativa. Zero custo LLM.
3. **Campo `page_type` no modelo de dados:** `Page` poderia ter `page_type: "records" | "cover" | "administrative" | "unknown"`. A0 seta isso antes de chamar A3.

**Status:** limitação identificada. Heurística por marcador de início (opção 2) é a mais simples e pode ser implementada em Semana 4 sem custo adicional.

---

### DT-09 — A6: score=1.0 em campos semanticamente errados (grounding insuficiente)

**Descoberto em:** 09/04/2026 — Rec 17: `nome="Malta"`, `pai="Malhabita"`, `data="18 de Lymasbo de 1831"`. Score A6 = **1.0 (ok)**.

**Problema:** o grounding check atual verifica apenas se o valor extraído **aparece no texto fonte**. Isso é necessário mas não suficiente. "Malta" aparece no texto → grounding passa. Mas "Malta" é um adjetivo do texto, não um nome de pessoa. "Lymasbo" é um mês inexistente inventado pelo A2 → aparece no texto (alucinado) → grounding passa.

**Raiz do problema:** o A6 não distingue entre:
- Valor extraído corretamente de texto correto
- Valor extraído corretamente de texto alucinado
- Valor extraído incorretamente de texto correto (ex: extraiu adjetivo como nome)

**Sugestões de correção:**
1. **Validação de padrão por campo:** nomes devem ser palavras capitalizadas (regex), datas devem conter padrão temporal reconhecível (mês em português, ano com 4 dígitos). Valores que não batem com o padrão esperado do campo recebem penalidade independentemente do grounding.
2. **Verificação de nomes contra léxico:** palavras de nome extraído que não são nomes próprios comuns em português do século XIX (lista de ~200 nomes comuns) recebem flag de suspeito.
3. **Grounding estrito:** verificar não só se o valor aparece no texto, mas se aparece em posição coerente com o campo. "Malta" no contexto de "como malhabita" não é uma posição de nome de batizado.
4. **Score composto:** combinar score de regras com score do LLM quando LLM foi chamado — agora são usados separadamente (regras determinam se LLM é chamado, LLM produz notas mas não altera score).

**Status:** limitação de design do A6. Validação de padrão por campo (opção 1) é implementável em Semana 4 sem custo LLM. Léxico e grounding estrito são Trabalho Futuro.

---

## Contexto acadêmico

**RecordIndex v1.0** (IC/IT): pipeline Transkribus → PyLaia HTR → doc-UFCN segmentação → similaridade cosseno para agrupamento → Ollama para correção → export.

**RecordIndex 2.0** (TCC): arquitetura multi-agente onde LLMs multimodais assumem HTR, segmentação de registros e extração de entidades. doc-UFCN permanece para segmentação de linhas (tarefa estrutural onde CNNs ainda são competitivas). A transição é justificada pela hipótese de que VLMs generalizam melhor para scripts históricos sem fine-tuning específico.

Hipótese central: LLMs melhoram a transcrição de texto padrão mas podem degradar nomes próprios (alucinação). O protocolo de avaliação mede isso explicitamente por campo.
