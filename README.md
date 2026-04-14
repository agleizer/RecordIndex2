# RecordIndex 2.0

Sistema multi-agente para transcrição automática e indexação de manuscritos históricos genealógicos (registros civis e eclesiásticos em português).

Projeto TCC — Universidade Presbiteriana Mackenzie.
Continuação acadêmica do RecordIndex v1.0 (Iniciação Científica).

**TCC 1 — prazos:** relatório + pôster: 06–11/05/2026 · Mostra de TCC I: 10/06/2026

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
| A0 Orquestrador (Python puro) | ✅ Implementado — LangGraph Trabalho Futuro | 2 |
| A3 Segmentação de registros | ✅ Implementado | 2 |
| A5 NER/Extração | ✅ Implementado | 2 |
| A1 Segmentação de linhas | ✅ Implementado e testado (IQR filtering) | 3 |
| A4 Correção estrutural | ✅ Implementado (template opcional) | 3 |
| A6 Validação | ✅ Implementado | 3 |
| A0 Multi-page open_record | ✅ Implementado | 5 |

---

## Mecanismo de fallback A6→A0→A2

O A6 atribui um score 0–1 a cada registro extraído. Quando o score fica abaixo de `A2_FALLBACK_THRESHOLD` (padrão: 0.5), o A0 interpreta isso como sinal de transcrição HTR insuficiente e retranscreve as linhas do registro usando o modelo fallback configurado em `A2_FALLBACK_MODEL`.

```
A2 (gemma4:e4b) → A3 → A4 → A5 → A6 → score < 0.5?
                                           │
                                    sim ───┘
                                           ↓
                              A2 (claude-sonnet-4-6) → A4 → A5 → A6
```

O fallback é uma instância separada de A2 inicializada no construtor do A0. Se `A2_FALLBACK_MODEL` estiver vazio, o mecanismo é desabilitado e o pipeline segue normalmente.

**Por que isso importa academicamente:** é o comportamento que caracteriza A0 como agente **goal-based** — o objetivo é maximizar qualidade dos registros extraídos, e A0 ajusta seus atuadores (escolha de modelo A2) com base na medida de performance (score A6). Sem isso, A0 seria apenas um orquestrador sequencial.

**Custo:** o fallback só aciona quando necessário — a maioria dos registros passa com o modelo local. Claude Sonnet é chamado apenas para os registros genuinamente difíceis (manuscritos degradados, caligrafia atípica).

---

## Framework de coordenação

Os agentes A1–A6 são implementados como **classes Python puras** (sem dependência de framework). O A0 coordena o pipeline e o feedback loop em Python puro — sem LangGraph.

### Por que não LangGraph

LangGraph foi avaliado e descartado do caminho crítico. O feedback loop A6→A0→A2 já funciona via `_retry_failed_records()` em Python puro. Adicionar LangGraph seria um wrapper de framework sobre código funcional, sem impacto nos resultados ou na análise acadêmica.

Cada agente implementa `__call__(state: dict) → dict`, interface compatível com nós LangGraph. Se o projeto evoluir para produto, a migração é uma refatoração de estrutura, não uma reescrita. **LangGraph está documentado como Trabalho Futuro no TCC.**

Strands Agents (AWS) também foi avaliado e descartado: pressupõe LLM no centro de cada agente (incompatível com A1 que é CNN puro), e o roteamento seria decisão do LLM — não-determinístico, difícil de justificar para a banca.

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
│   ├── output_writer.py        # write_outputs() — json/csv/txt com nome {colecao}_{timestamp}
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
├── evaluation/                 # Serviço de avaliação offline — porta 8001
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py              # FastAPI: POST /evaluate
│   ├── csv_parser.py       # Parse do CSV arquivístico (batismo)
│   ├── matcher.py          # Alinhamento por score composto nome+pai+mae (JW)
│   └── metrics.py          # segmentation_metrics(), extraction_metrics(), compare_field()
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
              ├── bbox: tuple              (A1)
              └── is_valid: bool           (A1 — False se outlier IQR; A2 pula linhas inválidas)
  └── records: dict[record.id → Record]
        ├── id: int
        ├── page_filename: str             (página onde o registro começa)
        ├── last_page_filename: str        (página onde termina — diferente se multi-página)
        ├── lines: dict[line.id → Line]   ← mesmos objetos de Page.lines (pode span múltiplas páginas)
        ├── page_text: str                (modo page — texto completo sem granularidade de linha)
        ├── corrected_text: str           (A4 — texto corrigido pelo template; vazio se A4 não rodou)
        ├── structured_output: dict       (A5) {nome, pai, mãe, data, ...}
        └── validation: dict              (A6) {score, verdict, field_errors, notes}
```

**Dual-referência:** os mesmos objetos `Line` aparecem tanto em `Page.lines` quanto em `Record.lines`. Isso espelha o padrão AVLTree do v1.0 — navegação possível em ambas as direções sem duplicação de dados. Em registros multi-página, `Record.lines` contém linhas de múltiplas páginas.

`Record.get_concatenated_text()` retorna `corrected_text` (A4) se disponível, senão concatena `line.best_text` (que por sua vez prefere `line.corrected_text` sobre `line.htr_text`).

**Registros multi-página:** o A0 mantém um `open_record` entre páginas. Linhas que aparecem antes do primeiro início de registro em uma página são adicionadas ao registro aberto da página anterior. Um registro só é finalizado (A4/A5/A6) quando um novo início é confirmado na página seguinte ou quando a coleção termina.

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
3. `ollama-init` baixa todos os modelos de `OLLAMA_MODELS_PULL` (na primeira execução pode demorar — `gemma4:e4b` tem ~9 GB, `llama3.2` tem ~2 GB)
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
    "a2": {"provider": "ollama", "model": "gemma4:e4b"},
    "a2_fallback": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "a3": {"provider": "ollama", "model": "llama3.2"},
    "a4": {"provider": "ollama", "model": "llama3.2"},
    "a5": {"provider": "ollama", "model": "llama3.2"},
    "a6": {"provider": "ollama", "model": "llama3.2"}
  }
}
```

---

### POST /a2/transcribe — transcrever uma imagem (testa A2)

Transcreve uma imagem de linha individual (crop de uma linha do manuscrito) ou uma página completa.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a2/transcribe`
- Body: `form-data`
- Campo: `file` | Tipo: `File` | Valor: selecionar a imagem
- Campo: `htr_scope` | Tipo: `Text` | Valor: `line` (padrão) ou `page`

Formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

Com `htr_scope=page`, A2 transcreve todas as linhas visíveis da página e retorna uma lista de strings (uma por linha). Com `htr_scope=line` (padrão), retorna o texto da linha como string.

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

Recebe uma lista de textos transcritos (modo line) ou o texto completo de uma página (modo page) e retorna os registros identificados.

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8000/a3/segment`
- Body: `raw` → `JSON`

**Body de exemplo — modo line (batismo):**
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
  "collection_type": "batismo",
  "htr_scope": "line"
}
```

**Resposta esperada (modo line):**
```json
{
  "record_start_indices": [0, 3, 6],
  "reasoning": "Cada registro começa com 'Aos X dias do mês...'",
  "num_records": 3
}
```

**Body de exemplo — modo page:**
```json
{
  "page_text": "Aos quatro dias do mês de abril...\nbaptizei a Maria...\nAos doze dias...",
  "collection_type": "batismo",
  "htr_scope": "page"
}
```

**Resposta esperada (modo page):**
```json
{
  "record_texts": ["Aos quatro dias...baptizei a Maria...", "Aos doze dias..."],
  "reasoning": "Dois registros identificados pela expressão de data",
  "num_records": 2
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
  "record_start_hint": "",
  "output_formats": ["json", "csv"],
  "htr_scope": "line"
}
```

Todos os campos são opcionais:
- `collection_type`: `"batismo"` | `"casamento"` | `"obito"` — se vazio, A0 infere automaticamente
- `image_dir`: diretório de imagens dentro do container (padrão: `SAMPLES_DIR` do `.env`)
- `record_template`: molde com placeholders `<CAMPO>` para ativar A4 (se vazio, A4 é pulado). As primeiras 12 palavras do template também são usadas como `record_start_hint` automaticamente
- `record_start_hint`: expressão de início de registro — se vazio e `record_template` preenchido, extraído automaticamente do template
- `output_formats`: lista de formatos de saída — `"json"` | `"csv"` | `"txt"` (padrão: `["json"]`)
- `htr_scope`: `"line"` (padrão) | `"page"` — modo de transcrição. Em `"line"` A2 processa cada linha individualmente e A3 recebe lista de textos. Em `"page"` A2 transcreve a página inteira e A3 recebe o texto completo.
- `year` e `location`: metadados que auxiliam A0 na classificação

**Resposta:** JSON completo da Collection com todos os Records, campos extraídos e resultado de validação (score A6). Os arquivos de saída são salvos em `volumes/output/` com nome `{colecao}_{YYYY-MM-DD_HH-MM}.{ext}`.

```json
{
  "total_pages": 3,
  "total_lines": 108,
  "total_records": 14,
  "output_files": {
    "json": "porto_da_cruz_batismos_1866_2026-04-10_14-23.json",
    "csv":  "porto_da_cruz_batismos_1866_2026-04-10_14-23.csv"
  }
}
```

> **Atenção:** o pipeline é síncrono — a request bloqueia até terminar. Com gemma4:e4b, cada linha demora ~15–25s. Uma página com 30 linhas leva ~10–15 minutos. Registros com score A6 < 0.5 são automaticamente retranscritos com o modelo fallback (Claude Sonnet). Acompanhe o progresso em tempo real via `GET /logs/tail`.

---

### GET /pipeline/last-output — rever o último resultado

```
GET http://localhost:8000/pipeline/last-output
```

Retorna o JSON mais recente gerado pelo pipeline (determinado por data de modificação do arquivo). Útil para inspecionar resultados sem re-executar.

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

### POST /evaluate — avaliar output do pipeline contra ground truth

Serviço separado na porta 8001. Compara o output do pipeline com o CSV arquivístico de referência. Retorna métricas de segmentação (A3) e extração (A5) em duas camadas independentes.

**Pré-requisito:** serviço `eval` rodando (`docker compose up eval`).

**Configuração no Postman:**
- Method: `POST`
- URL: `http://localhost:8001/evaluate`
- Body: `form-data`

| Campo | Tipo | Valor |
|-------|------|-------|
| `pipeline_json` | File | output JSON gerado pelo `/pipeline/run` |
| `reference_csv` | File | CSV arquivístico (ex: `PortoDaCruz_Batismos_1866_4752/output.csv`) |
| `collection_type` | Text | `batismo` |

**Resposta esperada:**
```json
{
  "collection_type": "batismo",
  "segmentation": {
    "total_gt": 128,
    "total_output": 112,
    "matched": 97,
    "unmatched_gt": 31,
    "unmatched_output": 15,
    "precision": 0.866,
    "recall": 0.758,
    "f1": 0.808,
    "segmentation_ratio": 0.875
  },
  "extraction": {
    "_coverage": 0.758,
    "nome": {"tp": 88, "fp": 5, "fn": 4, "precision": 0.946, "recall": 0.957, "f1": 0.951, "exact_match_rate": 0.72, "effective_recall": 0.726, "effective_f1": 0.821},
    "pai":  {"tp": 74, "fp": 12, "fn": 11, "precision": 0.860, "recall": 0.871, "f1": 0.865, "exact_match_rate": 0.41, "effective_recall": 0.660, "effective_f1": 0.746},
    "mae":  {"tp": 71, "fp": 15, "fn": 11, "precision": 0.826, "recall": 0.866, "f1": 0.845, "exact_match_rate": 0.39, "effective_recall": 0.656, "effective_f1": 0.733},
    "data": {"tp": 82, "fp": 9, "fn": 6, "precision": 0.901, "recall": 0.932, "f1": 0.916, "exact_match_rate": 0.28, "effective_recall": 0.707, "effective_f1": 0.793}
  },
  "record_comparisons": [...],
  "unmatched_output_ids": [3, 17, 42]
}
```

O matching usa score composto ponderado: nome (0.40) + pai (0.35) + mãe (0.25), com Jaro-Winkler e normalização de acentos. Pré-filtro: nome JW ≥ 0.65. Threshold final: 0.80.

`record_comparisons` contém, por registro alinhado: GT, output do pipeline, e comparação campo-a-campo (`exact`, `fuzzy`, `jaro_winkler`). Útil para análise qualitativa.

`unmatched_output_ids` — registros do pipeline sem correspondência no GT com score ≥ threshold. Indica hipersegmentação ou alucinação.

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
[A2] linha_0001.jpg → 'Aos oito dias do mes de...' (18.4s)
[A3] 8 registros, starts=[0,4,8,...] (12.1s)
[A4] pulado (sem template)
[A5] record 0 → {nome: Pedro, pai: João...} (8.3s)
[A6] score=0.92 (ok) (1.1s)
[A6] score=0.38 (failed) (1.2s)
[RETRY] registro 3 score=0.38 < 0.50 — retranscrevendo com fallback (anthropic/claude-sonnet-4-6)
[A2] linha_0021.jpg → 'Aos vinte e dois dias...' (3.1s)
[A6] score=0.81 (ok) (1.0s)
══ PIPELINE CONCLUÍDO ══ 3 páginas | 108 linhas | 14 registros | 612.4s total
```

O marcador `[RETRY]` indica ativação do mecanismo de fallback — A0 detectou score A6 abaixo do threshold e retranscreveu as linhas do registro usando o modelo alternativo.

---

## API — referência rápida

Documentação interativa (Swagger): `http://localhost:8000/docs`

| Método | Endpoint | Body | Descrição |
|--------|----------|------|-----------|
| GET | `/health` | — | Liveness check — retorna modelos configurados por agente (A0–A6) |
| POST | `/a1/segment` | `form-data: file` | Segmenta uma imagem de página em linhas (testa A1 isolado; aplica filtro IQR) |
| POST | `/a2/transcribe` | `form-data: file, htr_scope=line\|page` | Transcreve uma imagem de linha ou página (testa A2 isolado) |
| POST | `/a3/segment` | `{"lines": [...], "collection_type": "batismo", "htr_scope": "line"}` | Segmenta textos em registros (testa A3 isolado; aceita `page_text` em modo page) |
| POST | `/a4/correct` | `{"record_text": "...", "record_template": "Aos <DIA>..."}` | Corrige texto HTR usando template (testa A4 isolado) |
| POST | `/a5/extract` | `{"record_text": "...", "collection_type": "batismo"}` | Extrai campos de um registro (testa A5 isolado) |
| POST | `/a6/validate` | `{"record_text": "...", "structured_output": {...}, "collection_type": "batismo"}` | Valida campos extraídos (score 0–1, verdict, field_errors) |
| POST | `/pipeline/run` | `{"collection_name": "...", "collection_type": "batismo", "image_dir": "/data/input", "output_formats": ["json"], "htr_scope": "line", ...}` | Roda pipeline completo (A1→A2→A3→A4→A5→A6), com suporte multi-página e fallback automático para registros com score < threshold |
| GET | `/pipeline/last-output` | — | Retorna o último `output.json` gerado |
| GET | `/logs/tail` | `?lines=100` | Retorna as últimas N linhas do log persistido |

**Serviço de avaliação — porta 8001:**

| Método | Endpoint | Body | Descrição |
|--------|----------|------|-----------|
| GET | `/health` | — | Liveness check do serviço de avaliação |
| POST | `/evaluate` | `form-data: pipeline_json, reference_csv, collection_type` | Compara output do pipeline com CSV arquivístico — métricas P/R/F1 segmentação + extração |

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
| `OLLAMA_MODELS_PULL` | `gemma4:e4b,llama3.2` | Modelos a baixar no startup (comma-separated) |
| `A0_PROVIDER` | `ollama` | Provider do orquestrador (`ollama` \| `anthropic` \| `openai`) |
| `A0_MODEL` | `llama3.2` | Modelo do orquestrador |
| `A2_PROVIDER` | `ollama` | Provider do HTR (deve ser VLM com suporte a imagem) |
| `A2_MODEL` | `gemma4:e4b` | Modelo do HTR |
| `A2_FALLBACK_PROVIDER` | _(vazio)_ | Provider do fallback HTR — ativado quando score A6 < threshold |
| `A2_FALLBACK_MODEL` | _(vazio)_ | Modelo fallback HTR (ex: `claude-sonnet-4-6`). Vazio = fallback desabilitado |
| `A2_FALLBACK_THRESHOLD` | `0.5` | Score A6 abaixo do qual o fallback é acionado (0.0–1.0) |
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

## Módulo de avaliação (serviço separado)

O módulo `evaluation/` é um **serviço Docker independente** (porta 8001) que compara o output do pipeline com o índice arquivístico da coleção. Não faz parte do pipeline A0–A6 e não usa Ollama nem GPU.

Swagger: `http://localhost:8001/docs`

### POST /evaluate — comparar pipeline com índice arquivístico

```
POST http://localhost:8001/evaluate
Content-Type: multipart/form-data
  pipeline_json : arquivo JSON gerado pelo /pipeline/run
  reference_csv : CSV de referência da coleção (índice arquivístico)
  collection_type : "batismo" | "casamento" | "obito" (default: batismo)
```

**Formato do CSV de referência (batismo):**
```
PT/ABM/PMCH04/001/00025/000001; Registo de batismo n.º 1: Maria. Pai: Romano de Freitas Silva; Mãe: Constantina Narcisa de Nóbrega; 1866-01-01
```

**Estrutura da resposta:**
```json
{
  "segmentation": {
    "total_gt": 128,
    "total_output": 14,
    "matched": 12,
    "unmatched_gt": 116,
    "unmatched_output": 2,
    "precision": 0.857,
    "recall": 0.094,
    "f1": 0.170,
    "segmentation_ratio": 0.109
  },
  "extraction": {
    "_coverage": 0.857,
    "nome": {"tp": 10, "fp": 1, "fn": 1, "precision": 0.91, "recall": 0.91, "f1": 0.91, "exact_match_rate": 0.7, "effective_recall": 0.78, "effective_f1": 0.84},
    "pai":  {"...", "effective_recall": "...", "effective_f1": "..."},
    "mae":  {"...", "effective_recall": "...", "effective_f1": "..."},
    "data": {"...", "effective_recall": "...", "effective_f1": "..."}
  },
  "record_comparisons": [
    {
      "gt_seq": 1,
      "gt": {"nome": "Maria", "pai": "Romano de Freitas Silva", ...},
      "matched": true,
      "match_score_nome": 1.0,
      "output_record_id": 0,
      "output": {"nome": "Maria", "pai": "Romano de Freitas", ...},
      "field_comparison": {
        "nome": {"present": true, "exact": true, "fuzzy": true, "jaro_winkler": 1.0},
        "mae":  {"present": true, "exact": false, "fuzzy": true, "jaro_winkler": 0.965},
        "data": {"present": true, "exact": false, "fuzzy": true, "date_detail": {"month_match": true, ...}}
      }
    }
  ]
}
```

### Métricas em duas camadas

**Camada 1 — Segmentação (mede A3):** calculada sobre `total_gt` vs `total_output`, independente de extração. `recall` baixo quando processamos apenas parte das páginas da coleção — normal. `segmentation_ratio` = `total_output / total_gt` (1.0 = ideal; >1 = hipersegmentação; <1 = fusão).

**Camada 2 — Extração (mede A5):** calculada **apenas** sobre pares alinhados. `_coverage` = fração do GT com alinhamento confiável. Para cada campo são reportadas duas visões:

| Métrica | O que mede | Denominador |
|---|---|---|
| `precision` | Dos valores extraídos nos registros casados, quantos estão corretos | tp + fp |
| `recall` | Dos campos presentes nos registros casados, quantos foram extraídos | tp + fn |
| `f1` | Harmônica de precision e recall (condicional à cobertura) | — |
| `effective_recall` | Dos campos presentes em **todos** os GT, quantos foram extraídos | tp + fn + unmatched_gt |
| `effective_f1` | Harmônica de precision e effective_recall (visão end-to-end) | — |

> **Por que duas visões?** `f1` isola a qualidade de A5 (dado que A3 encontrou o registro, A5 extraiu corretamente?). `effective_f1` combina A3 + A5 numa única métrica de sistema. Para reportar o desempenho geral do pipeline, usar `effective_f1`. Para diagnosticar onde está o gargalo, comparar `f1` com `_coverage`.

> **`effective_precision` = `precision`** — registros GT não-casados não afetam o que foi extraído nos registros casados. Os FP existentes não mudam. Não há campo separado para evitar redundância.

**Alinhamento:** score composto ponderado nome (0.40) + pai (0.35) + mãe (0.25), com Jaro-Winkler e normalização de acentos. Pré-filtro: nome JW ≥ 0.65. Threshold final: 0.80.

**Data:** comparação por mês (extrai mês do texto português extraído vs. mês do ISO GT). `exact_match_rate` reporta casos onde A5 extraiu data em formato ISO diretamente.

**Semântica de FP/FN (não-padrão vs NER clássico):** FP = campo presente no output mas incorreto (JW < 0.85). FN = campo ausente no output. GT sempre tem todos os campos preenchidos, portanto FN real = extração vazia.

Documentação do protocolo completo: `02_desenvolvimento/2026_03_23_protocolo_avaliacao/protocolo_avaliacao.md`

---

## Escopo: TCC 1 vs TCC 2 vs Produto

Esta seção documenta as fronteiras de escopo entre as fases do projeto — o que pertence a cada etapa, para não reabrir discussões resolvidas.

### TCC 1 (entrega atual) — pipeline funcional com métricas

**Objetivo:** ter o pipeline A0–A6 funcionando de ponta a ponta, com métricas de avaliação aceitáveis, e usar os resultados para escrever o relatório.

O relatório do TCC 1 descreve:
- A arquitetura multi-agente (PEAS, tipos de agente, contratos A0↔A3/A5/A6)
- As decisões de implementação e os trade-offs técnicos
- Os resultados quantitativos do protocolo de avaliação (P/R/F1 por campo)
- A comparação com o RecordIndex v1.0 como linha de base histórica

O que **não** entra no TCC 1: comparação exaustiva de modelos, app com usuários, infraestrutura de produção, testes em escala com cloud.

### TCC 2 — experimentação em escala e engenharia de software

O TCC 2 usa o sistema construído no TCC 1 como base para dois eixos:

**Eixo 1 — Experimentação em escala:**
- Rodar o mesmo pipeline em infraestrutura cloud (AWS ou similar)
- Usar modelos Ollama de grande porte (200–300B parâmetros) — o "melhor dos dois mundos": privacidade/custo do Ollama local com qualidade de modelos frontier
- Comparação sistemática entre configurações de modelo por agente
- Novos tipos de coleção (casamentos, óbitos, outras dioceses)

**Eixo 2 — Engenharia de software e produto:**
- API assíncrona com jobs (`job_id`, polling, armazenamento)
- Mensageria (filas, workers)
- Frontend para usuários (Streamlit ou similar)
- Autenticação, multi-tenancy, histórico de coleções

### API síncrona é suficiente para o TCC 1

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

### Módulo de avaliação — `evaluation/` ✅ Implementado (Semana 4)

Ver seção **"Módulo de avaliação (serviço separado)"** acima para documentação completa de uso.

O módulo resolve o problema de métricas posicionais: alinhamento por score composto nome+pai+mae (Jaro-Winkler ponderado) desacopla extração de posição. Métricas em duas camadas independentes (segmentação / extração). Serviço Docker na porta 8001.

### Output nomeado por coleção e timestamp ✅ Implementado (Semana 4)

`src/output_writer.py` — `write_outputs()`. Formato: `{colecao}_{YYYY-MM-DD_HH-MM}.{ext}`. Ativado via `output_formats` no body de `/pipeline/run`.

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

### DT-14 — A3/A5: gemma4:e4b ignora `json_schema` e retorna markdown

**Descoberto em:** 11/04/2026 — primeiro teste E2E com gemma4:e4b, modo line.

**Problema:** o `make_structured()` usa `method="json_schema"`, que passa o schema Pydantic completo via `format: {type: "object", properties: {...}}` para o Ollama. O gemma4:e4b ignora o schema e responde em texto/markdown livre. Isso não é bug do nosso código — o modelo recebe o schema mas não aplica constrained decoding.

**Raiz do problema:** o Ollama suporta grammar-based constrained decoding (que força o JSON estruturalmente), mas nem todos os modelos têm suporte ativo a essa feature. O gemma4:e4b aceita `format: "json"` (json_mode), mas não o schema completo. Comportamento similar ao bug Qwen3.5+imagem documentado em DT-A2 (Semana 2).

**Solução implementada:** fallback em A3 (`_fallback_boundaries`) e A5 (`_fallback_extract`) — quando o structured output lança `OutputParserException`, o agente reinvoca o modelo com instrução explícita de JSON no prompt e extrai o resultado por regex. Custo: ~2× a latência dos agentes afetados.

**Fix correto:** usar qwen2.5:7b para A3/A4/A5 (já decidido — ver `project_model_selection.md`). O qwen2.5:7b respeita json_schema via Ollama. O gemma4:e4b continua como opção viável para A2 (HTR), onde structured output não é usado.

**Status:** mitigado com fallback. Fix definitivo = trocar modelo A3/A4/A5 para qwen2.5:7b.

---

### DT-15 — A2: regurgita prompt em imagens ilegíveis

**Descoberto em:** 11/04/2026 — teste gemma4 by-line.

**Problema:** quando a imagem é ilegível (ink bleed through), o gemma4 retorna o texto do próprio prompt do A2 em vez de tentar transcrever. O texto do prompt (~400 chars) passava pelo filtro `A3_MIN_HTR_CHARS=8` e contaminava registros.

**Solução implementada:** `_sanitize(text, image_name)` em `a2_htr.py`. Verifica substrings `_PROMPT_LEAK_MARKERS` (ex: "especialista em transcrição") e `_DEGRADATION_LABELS` (ex: "Tinta Repassada"). Se detectado, retorna `[ILEGÍVEL]` + WARNING no log. `[ILEGÍVEL]` propagado como token de ruído em A3 pre-filter, A5 `_NOISE_TOKENS`, A6 `_NOISE_VALUES`.

**Status:** mitigado. Funcional com gemma4. Claude Sonnet raramente produz esse comportamento.

---

### DT-16 — A0: log hardcoded "Claude" no modo page

**Descoberto em:** 11/04/2026 — revisão de código.

**Problema:** mensagem de log em `a0_orchestrator.py` usava string literal "Claude transcreveu N linhas" em vez de `self._config.a2_model`.

**Solução:** substituído por `"%s transcreveu %d linhas", self._config.a2_model, ...`

**Status:** corrigido.

---

### DT-17 — A4/A6: tag `<VAR>` detectada como placeholder não preenchido

**Descoberto em:** 11/04/2026 — análise do template de batismo do Porto da Cruz.

**Problema:** `_PLACEHOLDER_RE = re.compile(r'<[A-Z][A-Z_]*>')` foi projetado para detectar placeholders de dados (`<NOME>`, `<DIA>`) não preenchidos. Mas o template de batismo usa `<VAR>...</VAR>` como tag estrutural de seção variável (igual ao `<OPT>`). A regex detectava `<VAR>` como placeholder não preenchido, fazendo com que A4 descartasse o resultado mesmo quando todos os campos de dados haviam sido preenchidos corretamente. O mesmo regex existe no A6 para detectar outputs do A4 com placeholders restantes.

**Solução:** negative lookahead na regex excluindo tags estruturais conhecidas:
```python
_STRUCTURAL_TAGS = {"OPT", "VAR"}
_PLACEHOLDER_RE = re.compile(
    r'<(?!' + '|'.join(t + r'\b' for t in _STRUCTURAL_TAGS) + r')[A-Z][A-Z_]*>'
)
```
Aplicado em `a4_correction.py` e `a6_validation.py`. Prompts do A4 atualizados para explicar comportamento de `<VAR>`.

**Status:** corrigido.

---

### DT-18 — A4: gemma4 não preenche placeholders do template

**Descoberto em:** 11/04/2026 — análise dos 3 testes E2E (100% de falha com gemma4).

**Problema:** o A4 enviava template + texto ao gemma4 e o modelo devolvia o template intacto (sem substituir os placeholders) ou o texto bruto sem estrutura. A tarefa requer extrair ~15 valores do texto HTR ruidoso e injetá-los em posições específicas do template — complexidade alta para um modelo 4B.

**Solução implementada:** estratégia em duas tentativas em `a4_correction.py`:
1. Prompt padrão (`a4/correct`) — mesmo comportamento anterior
2. Se placeholders permanecerem, fallback com `a4/correct_fallback` — prompt com exemplo concreto completo (MOLDE → TEXTO → RESULTADO CORRETO) e marcadores explícitos

**Fix correto:** usar modelo com maior capacidade de instruction-following para A4. Com Claude Sonnet no A2 gerando texto limpo, Claude ou qwen2.5:7b no A4 deve resolver.

**Status:** mitigado com fallback. Fix definitivo = modelo mais capaz no A4.

---

### DT-19 — `make_structured()` passava `method="json_schema"` para Anthropic

**Descoberto em:** 13/04/2026 — testes 4/1b e 4/2 com Claude Sonnet em todos os agentes.

**Problema:** `make_structured()` em `llm_client.py` passava `method="json_schema"` para todos os providers, incluindo Anthropic. Claude ignora o schema e responde em prosa/markdown quando recebe esse parâmetro, causando `OutputParserException` em 100% das chamadas de A3/A5/A6.

**Solução:** remover `method` para providers não-Ollama. Para Anthropic e OpenAI, `with_structured_output()` sem `method` usa function calling nativo (tool_use), que funciona corretamente.

**Status:** corrigido.

---

### DT-20 — qwen3.5 A3/A5 structured output falha sistematicamente (100%)

**Descoberto em:** 14/04/2026 — experimento com 132 registros (108 páginas).

**Problema:** qwen3.5 retorna raciocínio em prosa ou markdown antes do JSON, causando `OutputParserException` no parser LangChain em 100% das chamadas de A3 e A5. O fallback regex ativa e recupera os dados corretamente em 100% dos casos — sem perda de dados.

Exemplos do log:
```
Invalid json output: Com base na análise semântica das linhas...
Invalid json output: ### Análise dos Dados e Raciocínio
```

**Impacto:** leve subfusão de registros (segmentation_ratio=0.9015 vs ideal 1.0) em páginas com 2 registros, onde o regex de fallback é menos confiável. Tempo de execução ~200–400s mais alto por retries.

**O que foi tentado (14/04/2026):**

1. `update["format"] = schema.model_json_schema()` no `model_copy` antes de `with_structured_output` — não funcionou. `with_structured_output(method="json_schema")` chama `self.bind(format=schema_dict)` internamente, que sobrescreve o valor bakeado via kwargs no invoke.

2. `update["format"] = "json"` no `model_copy` — mesmo resultado. O bind interno de `with_structured_output` continua enviando `format=schema_dict` ao Ollama, que qwen3.5 ignora.

**Causa raiz:** `with_structured_output(method="json_schema")` sempre envia o schema completo como `format` ao Ollama via `bind()`. Para qwen3.5 nesta versão do Ollama, o constrained decoding por schema não está sendo aplicado — o modelo produz markdown/prosa livremente.

**Fix correto (não implementado):** substituir `with_structured_output` no path Ollama de `make_structured()` por uma chain manual que usa `format="json"` bakeado via `model_copy` (sem `with_structured_output`), assim o format não é sobrescrito pelo bind interno:

```python
# Em make_structured(), path Ollama — substituir:
#   return configured.with_structured_output(schema, method="json_schema")
# Por:
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
parser = PydanticOutputParser(pydantic_object=schema)
return configured | RunnableLambda(lambda msg: msg.content) | parser
# (requer que os agentes incluam format instructions no prompt manualmente)
```

**Status:** aberto. Fix requer refatoração não trivial de `make_structured()` e verificação dos agentes. Resultados do experimento com 132 registros são válidos e apresentáveis — fallback opera corretamente. Não corrigir antes da entrega do TCC 1. Candidato para TCC 2.

---

## Contexto acadêmico

**RecordIndex v1.0** (IC/IT): pipeline Transkribus → PyLaia HTR → doc-UFCN segmentação → similaridade cosseno para agrupamento → Ollama para correção → export.

**RecordIndex 2.0 — TCC 1** (entrega 06/2026): arquitetura multi-agente onde LLMs multimodais assumem HTR, segmentação de registros e extração de entidades. doc-UFCN permanece para segmentação de linhas (tarefa estrutural onde CNNs ainda são competitivas). Objetivo do TCC 1: pipeline funcional com métricas de avaliação aceitáveis + relatório baseado nos resultados reais do sistema.

**RecordIndex 2.0 — TCC 2** (fase futura): usar o pipeline do TCC 1 para experimentação em escala (cloud + modelos 200–300B parâmetros) e evolução para produto (API assíncrona, mensageria, usuários, frontend). O TCC 2 não recomeça do zero — constrói sobre a fundação do TCC 1.

Hipótese central (TCC 1): LLMs melhoram a transcrição de texto padrão mas podem degradar nomes próprios (alucinação). O protocolo de avaliação mede isso explicitamente por campo.
