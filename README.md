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
| A6 Validação | 🔲 Pendente | 4 |

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
│   ├── pipeline.py             # run() e run_with_orchestrator()
│   ├── config.py               # Configuração via variáveis de ambiente
│   ├── llm_client.py           # Factory de providers (Ollama, Anthropic, OpenAI)
│   ├── schemas.py              # Schemas Pydantic compartilhados (API responses)
│   │
│   ├── models/                 # Hierarquia de dados
│   │   ├── line.py             # Linha de texto (unidade básica)
│   │   ├── page.py             # Página do manuscrito
│   │   ├── record.py           # Registro genealógico (grupo de linhas)
│   │   ├── collection.py       # Coleção de documentos
│   │   └── collection_config.py  # Tipo de coleção + campos de extração (batismo/casamento/obito)
│   │
│   └── agents/
│       ├── a2_htr.py           # A2: transcrição via Ollama VLM
│       ├── a3_segmentation.py  # A3: segmentação de linhas em registros
│       ├── a5_ner.py           # A5: extração de campos via schema Pydantic dinâmico
│       └── a0_orchestrator.py  # A0: coordena A2 → A3 → A5 (LangGraph na Semana 5)
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
  └── pages: dict[filename → Page]
        └── lines: dict[line.id → Line]   ← mesmo objeto referenciado em Record.lines
              ├── id: str
              ├── image_path: str
              ├── htr_text: str            (A2)
              ├── corrected_text: str      (A4 — linha; vazio se A4 não rodou)
              ├── page_filename: str       (proveniência — setado por Page.add_line())
              └── bbox: tuple              (A1)
  └── records: dict[record.id → Record]
        ├── id: int
        ├── page_filename: str
        ├── lines: dict[line.id → Line]   ← mesmos objetos de Page.lines
        ├── corrected_text: str           (A4 — registro completo corrigido pelo template)
        └── structured_output: dict       (A5) {nome, pai, mãe, data}
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
    "a5": {"provider": "ollama", "model": "llama3.2"}
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

### POST /pipeline/run — pipeline completo A2→A3→A5

Processa todas as imagens em `volumes/samples/`, executa A2 (HTR) → A3 (segmentação) → A5 (extração) e salva o resultado em `volumes/output/output.json`.

**Pré-requisito:** colocar imagens de linha em `volumes/samples/` (um arquivo por linha, em ordem).

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/pipeline/run`
- Body: `raw` → `JSON` (opcional — se omitido usa `batismo`)

**Body (opcional):**
```json
{
  "collection_type": "batismo",
  "collection_name": "Batismos São Paulo 1850"
}
```

**Resposta:** JSON completo da Collection com todos os Records e campos extraídos. O mesmo JSON é salvo em `volumes/output/output.json`.

> **Atenção:** o pipeline é síncrono — a request bloqueia até terminar. Para 10 imagens de linha com qwen3.5:9b, espere ~5–7 minutos.

---

### GET /pipeline/last-output — rever o último resultado

```
GET http://localhost:8000/pipeline/last-output
```

Retorna o `output.json` da última execução do pipeline. Útil para inspecionar resultados sem re-executar.

Retorna 404 se nenhum pipeline tiver rodado ainda.

---

## API — referência rápida

Documentação interativa (Swagger): `http://localhost:8000/docs`

| Método | Endpoint | Body | Descrição |
|--------|----------|------|-----------|
| GET | `/health` | — | Liveness check — retorna modelos configurados por agente |
| POST | `/a2/transcribe` | `form-data: file` | Transcreve uma imagem de linha (testa A2 isolado) |
| POST | `/a3/segment` | `{"lines": [...], "collection_type": "batismo"}` | Segmenta textos em registros (testa A3 isolado) |
| POST | `/a5/extract` | `{"record_text": "...", "collection_type": "batismo"}` | Extrai campos de um registro (testa A5 isolado) |
| POST | `/pipeline/run` | `{"collection_type": "batismo", "collection_name": "..."}` (opcional) | Roda pipeline completo (A2→A3→A5) sobre `volumes/samples/` |
| GET | `/pipeline/last-output` | — | Retorna o último `output.json` gerado |

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

### Pipeline assíncrono (jobs + polling)

**Problema:** `/pipeline/run` é síncrono. Para uso interativo com UI, seria necessário job_id + polling.

**Decisão:** fora do escopo do TCC. Implementar como Trabalho Futuro se o projeto evoluir para produto.

### Frontend

**Problema:** não há interface de usuário. Interação atual via Postman/API.

**Decisão:** fora do escopo do TCC. A avaliação é offline. Mencionar em Trabalhos Futuros.

### Experimento: granularidade de input do A2 (página vs linha vs bloco)

**Problema:** A2 recebe crops de linha individualmente (saída do A1). Não foi avaliado se passar blocos de texto ou a página inteira melhoraria a qualidade da transcrição — o contexto visual maior pode ajudar o modelo a interpretar palavras ambíguas.

**Status:** pendente — experimento possível após pipeline E2E funcionar.

### CollectionConfig hardcoded — deveria ser dinâmico via A0

**Problema:** os tipos de coleção (`batismo`, `casamento`, `obito`), seus campos de extração e as descrições de cada campo estão hardcoded como factories estáticas em `src/models/collection_config.py`. Para suportar um tipo novo é necessário alterar o código.

**O que deveria acontecer:** o orquestrador A0 deveria receber o tipo de coleção como parâmetro e montar o `CollectionConfig` dinamicamente — via arquivo de configuração externo (YAML/JSON) ou banco de dados. Assim nenhum agente precisaria ser alterado para suportar uma nova coleção.

**Decisão:** manter hardcoded para o TCC — o corpus de avaliação é fixo (batismo, casamento, óbito). Avaliar na Semana 5 ao implementar A0. Candidato a Trabalhos Futuros.

**Status:** limitação documentada, aceita para o TCC.

---

## Contexto acadêmico

**RecordIndex v1.0** (IC/IT): pipeline Transkribus → PyLaia HTR → doc-UFCN segmentação → similaridade cosseno para agrupamento → Ollama para correção → export.

**RecordIndex 2.0** (TCC): arquitetura multi-agente onde LLMs multimodais assumem HTR, segmentação de registros e extração de entidades. doc-UFCN permanece para segmentação de linhas (tarefa estrutural onde CNNs ainda são competitivas). A transição é justificada pela hipótese de que VLMs generalizam melhor para scripts históricos sem fine-tuning específico.

Hipótese central: LLMs melhoram a transcrição de texto padrão mas podem degradar nomes próprios (alucinação). O protocolo de avaliação mede isso explicitamente por campo.
