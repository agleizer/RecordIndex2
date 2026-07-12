"""
RecordIndex 2.0 — Front-end (Streamlit).

Três abas:
  Executar  — formulário do pipeline + ingestão de imagens; dispara um job.
  Jobs      — lista as execuções, status, log ao vivo e download dos resultados.
  Avaliar   — compara um output do pipeline com a ground truth (serviço eval).

Fala com os serviços por HTTP (jobs:8002, app:8000, eval:8001) e lê/escreve
no volume compartilhado /data. Não toca no back-end.
"""

import hmac
import io
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

JOBS_URL = os.getenv("JOBS_URL", "http://jobs:8002")
APP_URL = os.getenv("APP_URL", "http://app:8000")
EVAL_URL = os.getenv("EVAL_URL", "http://eval:8001")
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/data/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))

COLLECTION_TYPES = ["batismo", "casamento", "obito"]
LOGO_PATH = Path(__file__).parent / "assets" / "RI_logo.png"

st.set_page_config(
    page_title="RecordIndex 2.0",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)

# Esconde a barra superior do Streamlit (deploy, menu, faixa decorativa) e o rodapé.
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden; height: 0;}
      [data-testid="stDecoration"] {display: none;}
      [data-testid="stStatusWidget"] {visibility: hidden;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      .stDeployButton {display: none;}
      [data-testid="stHeader"] {background: transparent;}
      .stTabs [aria-selected="true"] {color: #F0531C;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Espelha src/output_writer._sanitize."""
    name = re.sub(r"[^\w\s-]", "", name.lower().strip())
    name = re.sub(r"\s+", "_", name)
    return name or "colecao"


def _models_summary(models: dict | None) -> str:
    if not models:
        return "—"
    parts = []
    for agent in ("a2", "a3", "a5"):
        cfg = models.get(agent)
        if cfg:
            parts.append(f"{agent}={cfg.get('model', '?')}")
    return ", ".join(parts) if parts else "—"


def _get(url, **kw):
    r = requests.get(url, timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def _export_formats():
    """Formatos de export anunciados pelo serviço jobs (/export/formats).
    Cacheado: mudam raramente. Novos formatos aparecem sem tocar no front."""
    return _get(f"{JOBS_URL}/export/formats", timeout=10)


@st.cache_data(ttl=60)
def _job_records(code):
    """Registros do job vindos do banco. Usado para saber se há o que exportar."""
    return _get(f"{JOBS_URL}/jobs/{code}/records", timeout=30)


@st.cache_data(ttl=60)
def _export_bytes(code, fmt):
    """Bytes do export no formato pedido, gerados pelo banco (não lê disco)."""
    r = requests.get(f"{JOBS_URL}/jobs/{code}/export", params={"fmt": fmt}, timeout=120)
    r.raise_for_status()
    return r.content


def _short_page(p) -> str:
    """Encurta o nome longo da imagem (ex: ..._downloaded_image_0002) para a tabela."""
    p = p or ""
    return p if len(p) <= 28 else "…" + p[-26:]


def _norm(s) -> str:
    """Normaliza para busca: remove acentos e caixa (gouvea casa com Gouvêa)."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _md_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    """Tabela Markdown (sem pandas/Arrow — evita segfault do pyarrow no container)."""
    def cell(v):
        return "" if v is None else str(v).replace("|", "\\|")
    head = "| " + " | ".join(label for _, label in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell(r.get(k)) for k, _ in headers) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def _check_login():
    """Gate de acesso ao front. Credencial única em FRONT_USER/FRONT_PASSWORD (.env).
    Comparação server-side com compare_digest. Protege apenas o Streamlit: os
    serviços jobs/app/eval seguem acessíveis por suas portas, por decisão (permite
    bater nos endpoints durante o desenvolvimento). FRONT_PASSWORD vazio desliga o
    gate. Ver nota no README."""
    if st.session_state.get("auth_ok"):
        return

    exp_user = os.getenv("FRONT_USER", "")
    exp_pass = os.getenv("FRONT_PASSWORD", "")
    if not exp_pass:  # gate desligado quando não há senha configurada
        return

    with st.form("login"):
        st.markdown("#### Entrar")
        user = st.text_input("Usuário")
        pwd = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        ok_user = hmac.compare_digest(user, exp_user)
        ok_pass = hmac.compare_digest(pwd, exp_pass)
        if ok_user and ok_pass:
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Usuário ou senha inválidos.")
    st.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_h1, _h2 = st.columns([1, 6], vertical_alignment="center")
if LOGO_PATH.exists():
    _h1.image(str(LOGO_PATH), width=110)
_h2.title("RecordIndex 2.0")
_h2.caption("Indexação de registros históricos por decomposição multiagente")

_check_login()

tab_run, tab_jobs, tab_eval, tab_index = st.tabs(["Executar", "Jobs", "Avaliar", "Índice"])


# --- Aba Executar ----------------------------------------------------------
with tab_run:
    st.subheader("Disparar uma execução do pipeline")

    c1, c2 = st.columns(2)
    name = c1.text_input("Nome da coleção", "PortoDaCruz Batismos 1863")
    ctype_label = c2.selectbox("Tipo de coleção", ["(inferir automaticamente)"] + COLLECTION_TYPES)

    c3, c4, c5 = st.columns(3)
    year = c3.text_input("Ano", "1863")
    location = c4.text_input("Localidade", "Porto da Cruz, Madeira")
    htr_scope = c5.selectbox("HTR scope", ["line", "page"])

    fmts = st.multiselect("Formatos de saída", ["json", "csv", "txt"], default=["json", "csv"])
    hint = st.text_input("record_start_hint (opcional)", "")
    template = st.text_area("record_template (opcional, molde com placeholders)", "", height=180)

    st.markdown("**Imagens da coleção**")
    origem = st.radio("Origem", ["Diretório existente", "Upload"], horizontal=True)
    image_dir = ""
    uploads = None
    if origem == "Diretório existente":
        image_dir = st.text_input(
            "Caminho dentro de /data/input",
            "/data/input/PortoDaCruz_Batismos_1863",
        )
        p = Path(image_dir) if image_dir else None
        if p and p.exists():
            n = sum(1 for f in p.iterdir() if f.is_file())
            st.caption(f"{n} arquivo(s) encontrados em {image_dir}")
        elif image_dir:
            st.warning("Diretório não encontrado (será validado no disparo).")
    else:
        uploads = st.file_uploader(
            "Selecione as imagens", accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp"],
        )
        if uploads:
            st.caption(f"{len(uploads)} imagem(ns) selecionada(s).")

    if st.button("Disparar", type="primary"):
        try:
            if origem == "Upload":
                if not uploads:
                    st.error("Selecione ao menos uma imagem.")
                    st.stop()
                dest = INPUT_DIR / f"{_sanitize(name)}_{datetime.now():%Y%m%d_%H%M%S}"
                dest.mkdir(parents=True, exist_ok=True)
                for f in uploads:
                    (dest / f.name).write_bytes(f.getbuffer())
                image_dir = str(dest)
            elif not image_dir:
                st.error("Informe o diretório das imagens.")
                st.stop()

            cfg = {
                "htr_scope": htr_scope,
                "collection_name": name,
                "year": year,
                "location": location,
                "collection_type": "" if ctype_label.startswith("(") else ctype_label,
                "record_start_hint": hint,
                "record_template": template,
                "image_dir": image_dir,
                "output_formats": fmts or ["json"],
            }
            with st.spinner("Criando job..."):
                r = requests.post(f"{JOBS_URL}/jobs", json=cfg, timeout=30)
                r.raise_for_status()
                job = r.json()
            st.success(
                f"Job **{job['code']}** criado ({job.get('num_images', '?')} imagens). "
                "Acompanhe na aba **Jobs**."
            )
        except Exception as e:
            st.error(f"Falha ao criar o job: {e}")


# --- Aba Jobs --------------------------------------------------------------
# Isolada num st.fragment: quando há job rodando, só este pedaço se atualiza
# (run_every), sem rerodar o app inteiro nem congelar as outras abas. Timeouts
# curtos garantem que um `app` ocupado no pipeline não trave a sessão.
with tab_jobs:
    st.subheader("Execuções")

    # Sonda curta só para decidir se o painel precisa se auto-atualizar.
    try:
        _probe = _get(f"{JOBS_URL}/jobs", timeout=5)
        _running = any(j.get("status") == "running" for j in _probe)
    except Exception:
        _running = False

    top = st.columns([1, 5])
    if top[0].button("Atualizar agora"):
        st.rerun()
    top[1].caption(
        "Atualizando sozinho a cada 5s (há job rodando)." if _running
        else "Sem job rodando. Use 'Atualizar agora' para recarregar."
    )

    @st.fragment(run_every="5s" if _running else None)
    def _jobs_panel():
        try:
            jobs = _get(f"{JOBS_URL}/jobs", timeout=5)
        except Exception as e:
            st.warning(f"Serviço de jobs indisponível no momento: {e}")
            return

        if not jobs:
            st.info("Nenhum job ainda. Dispare um na aba Executar.")
            return

        table = [{
            "código": j["code"],
            "coleção": j.get("collection_name", ""),
            "status": j.get("status", ""),
            "criado": j.get("created_at", ""),
            "imgs": j.get("num_images"),
            "registros": j.get("num_records"),
            "modelos": _models_summary(j.get("models")),
        } for j in jobs]
        st.markdown(_md_table(table, [
            ("código", "código"), ("coleção", "coleção"), ("status", "status"),
            ("criado", "criado"), ("imgs", "imgs"), ("registros", "registros"),
            ("modelos", "modelos"),
        ]))

        code = st.selectbox("Ver detalhe do job", [j["code"] for j in jobs], key="job_detail_sel")
        job = next((j for j in jobs if j["code"] == code), None)
        if not job:
            return

        status = job.get("status")
        badge = {"queued": "🟡", "running": "🔵", "done": "🟢", "failed": "🔴"}.get(status, "")
        st.markdown(f"### {badge} {code} — `{status}`")

        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Imagens", job.get("num_images") or 0)
        meta2.metric("Registros", job.get("num_records") or 0)
        meta3.metric("Modelos", _models_summary(job.get("models")))

        with st.expander("Config enviada ao pipeline"):
            st.json(job.get("config", {}))

        if status == "running":
            st.markdown("**Log ao vivo** (últimas linhas)")
            try:
                tail = _get(f"{APP_URL}/logs/tail", params={"lines": 150}, timeout=5)
                st.code("".join(tail.get("tail", [])) or "(sem linhas ainda)")
            except Exception as e:
                st.caption(f"log indisponível agora: {e}")

        elif status == "done":
            st.markdown("**Resultados**")
            # Export vem do banco (fonte da verdade). Os arquivos que o back grava
            # em /data/output continuam existindo (ver nota de duplicação no README),
            # mas o download normal não os usa mais: só o fallback abaixo, para jobs
            # antigos sem registros no banco.
            try:
                has_db = len(_job_records(code)) > 0
            except Exception as e:
                has_db = False
                st.caption(f"registros indisponíveis agora: {e}")

            if has_db:
                fmts = _export_formats()
                fmt = st.selectbox(
                    "Formato",
                    [f["fmt"] for f in fmts],
                    format_func=lambda k: next(f["label"] for f in fmts if f["fmt"] == k),
                    key=f"fmt_{code}",
                )
                ext = next(f["ext"] for f in fmts if f["fmt"] == fmt)
                try:
                    st.download_button(
                        "Baixar",
                        data=_export_bytes(code, fmt),
                        file_name=f"{code}.{ext}",
                        key=f"dl_{code}",
                    )
                except Exception as e:
                    st.caption(f"export indisponível agora: {e}")
            else:
                # Fallback: jobs anteriores à persistência no banco (ou save_records
                # que falhou). Baixa os arquivos gravados pelo back em /data/output.
                st.caption("Sem registros no banco para este job; baixando os arquivos gerados.")
                files = job.get("output_files") or {}
                if not files:
                    st.caption("Nenhum arquivo de saída registrado.")
                for fmt, fname in files.items():
                    fpath = OUTPUT_DIR / fname
                    if fpath.exists():
                        st.download_button(
                            f"Baixar {fmt.upper()} ({fname})",
                            data=fpath.read_bytes(),
                            file_name=fname,
                            key=f"dl_{code}_{fmt}",
                        )
                    else:
                        st.caption(f"{fmt}: arquivo {fname} não encontrado em /data/output")

        elif status == "failed":
            st.error(job.get("error") or "Falhou sem mensagem.")

    _jobs_panel()


# --- Aba Avaliar -----------------------------------------------------------
with tab_eval:
    st.subheader("Avaliar output contra a ground truth")
    st.caption("Passo desacoplado do pipeline. Compara um JSON de saída com o CSV de referência.")

    outs = sorted(
        [p.name for p in OUTPUT_DIR.glob("*.json") if not p.name.startswith("eval_")],
        reverse=True,
    )
    sel = st.selectbox("pipeline_json (de /data/output)", ["(fazer upload)"] + outs)
    up_json = None
    if sel == "(fazer upload)":
        up_json = st.file_uploader("Upload do pipeline_json", type=["json"])

    ref = st.file_uploader("reference_csv (ground truth)", type=["csv"])
    etype = st.selectbox("Tipo de coleção", COLLECTION_TYPES)

    if st.button("Avaliar", type="primary"):
        if sel == "(fazer upload)" and not up_json:
            st.error("Envie o pipeline_json ou escolha um da lista.")
            st.stop()
        if not ref:
            st.error("Envie o reference_csv.")
            st.stop()
        try:
            pj_bytes = up_json.getvalue() if up_json else (OUTPUT_DIR / sel).read_bytes()
            files = {
                "pipeline_json": ("pipeline.json", pj_bytes, "application/json"),
                "reference_csv": (ref.name, ref.getvalue(), "text/csv"),
            }
            with st.spinner("Avaliando..."):
                r = requests.post(
                    f"{EVAL_URL}/evaluate", files=files,
                    data={"collection_type": etype}, timeout=180,
                )
                r.raise_for_status()
                res = r.json()

            if "error" in res:
                st.error(res["error"])
                st.stop()

            seg = res.get("segmentation", {})
            st.markdown("### Segmentação (A3)")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("F1", seg.get("f1"))
            s2.metric("Precisão", seg.get("precision"))
            s3.metric("Recall", seg.get("recall"))
            s4.metric("Ratio out/GT", seg.get("segmentation_ratio"))
            st.caption(
                f"GT={seg.get('total_gt')} · output={seg.get('total_output')} · "
                f"casados={seg.get('matched')} · GT sem par={seg.get('unmatched_gt')}"
            )

            ext = res.get("extraction", {})
            st.markdown("### Extração (A5)")
            st.metric("Cobertura (pares casados / GT)", ext.get("_coverage"))
            rows = []
            for field, m in ext.items():
                if field == "_coverage" or not isinstance(m, dict):
                    continue
                rows.append({
                    "campo": field,
                    "slot_acc": m.get("slot_accuracy"),
                    "eff_slot_acc": m.get("effective_slot_accuracy"),
                    "eff_f1": m.get("effective_f1"),
                    "exact_rate": m.get("exact_match_rate"),
                    "tp": m.get("tp"), "fp": m.get("fp"), "fn": m.get("fn"),
                })
            if rows:
                st.markdown(_md_table(rows, [
                    ("campo", "campo"), ("slot_acc", "slot_acc"),
                    ("eff_slot_acc", "eff_slot_acc"), ("eff_f1", "eff_f1"),
                    ("exact_rate", "exact_rate"), ("tp", "tp"), ("fp", "fp"), ("fn", "fn"),
                ]))

            st.download_button(
                "Baixar resultado completo (JSON)",
                data=json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="eval_resultado.json",
            )
            with st.expander("Resposta completa"):
                st.json(res)
        except Exception as e:
            st.error(f"Falha na avaliação: {e}")


# --- Aba Índice ------------------------------------------------------------
with tab_index:
    st.subheader("Índice de registros")
    st.caption("Selecione um job concluído para ver a tabela completa dos registros extraídos do banco.")

    try:
        _all_jobs = _get(f"{JOBS_URL}/jobs")
    except Exception as e:
        _all_jobs = []
        st.error(f"Não foi possível listar os jobs: {e}")

    _done = [j for j in _all_jobs if j.get("status") == "done" and (j.get("num_records") or 0) > 0]
    if not _done:
        st.info("Nenhum job concluído com registros ainda.")
    else:
        _opts = {
            f"{j['code']} — {j.get('collection_name', '?')} "
            f"({j.get('collection_type') or '?'}, {j.get('num_records')} reg.)": j["code"]
            for j in _done
        }
        _label = st.selectbox("Job", list(_opts.keys()), key="idx_job")
        _code = _opts[_label]

        try:
            _records = _job_records(_code)
        except Exception as e:
            _records = []
            st.error(f"Falha ao carregar registros: {e}")

        # Filtro opcional, client-side sobre os registros já carregados deste job.
        # Não dispara query nova (evita seq scan cross-job). Casa só contra os
        # campos extraídos (o índice), não contra o texto do assento: o boilerplate
        # do texto (padre, paróquia) casaria com quase tudo. Insensível a acento.
        _termo = _norm(st.text_input(
            "Filtrar pelos campos (nome, pai/mãe, noivo, data...)", key="idx_q"
        ).strip())
        if _termo:
            def _match(r):
                blob = " ".join(str(v) for v in (r.get("fields") or {}).values())
                return _termo in _norm(blob)
            _shown = [r for r in _records if _match(r)]
        else:
            _shown = _records

        st.caption(f"{len(_shown)} de {len(_records)} registro(s)")

        if _shown:
            # Colunas = fixas + união das chaves de fields (um job = um tipo,
            # então as colunas são consistentes).
            _keys, _seen = [], set()
            for r in _shown:
                for k in r.get("fields") or {}:
                    if k not in _seen:
                        _seen.add(k)
                        _keys.append(k)
            _headers = [("record_id", "Reg"), ("page", "Página")] + [(k, k) for k in _keys]
            _rows = []
            for r in _shown:
                row = {"record_id": r.get("record_id"), "page": _short_page(r.get("page"))}
                row.update(r.get("fields") or {})
                _rows.append(row)
            st.markdown(_md_table(_rows, _headers))

            with st.expander("Ver texto completo dos registros"):
                for r in _shown:
                    st.markdown(f"**[{r.get('record_id')}] {_short_page(r.get('page'))}**")
                    st.write(r.get("text") or "")
