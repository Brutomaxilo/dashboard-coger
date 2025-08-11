import io
import os
import re
import unicodedata
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============ CONFIG INICIAL ============
st.set_page_config(page_title="PCI/SC – Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 Dashboard PCI/SC – Produção & Pendências")
st.markdown("---")

# Utilitário: limpar cache rápido durante ajustes
if st.sidebar.button("🧹 Limpar cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache limpo. Recarregue a página (Ctrl+R).")

# ============ CACHE / IO ============
@st.cache_data
def read_csv_optimized(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    seps = [";", ",", "\t", "|"]
    encs = ["utf-8", "latin-1", "cp1252"]
    for enc in encs:
        for sep in seps:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(bio, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    df.columns = [c.strip('"').strip() for c in df.columns]
                    for c in df.columns:
                        if df[c].dtype == "object":
                            df[c] = df[c].astype(str).str.strip('"').str.strip()
                    return df
            except Exception:
                continue
    try:
        bio = io.BytesIO(file_content)
        df = pd.read_csv(bio, sep=None, engine="python", encoding="utf-8")
        if df.shape[1] > 1:
            df.columns = [c.strip('"').strip() for c in df.columns]
            return df
    except Exception:
        pass
    return None

@st.cache_data
def process_datetime_column(series: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    if series is None or series.empty:
        return None
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
    if dt.isna().sum() > len(dt) * 0.5:
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                dt = pd.to_datetime(series, format=fmt, errors="coerce")
                if dt.notna().sum() > len(series) * 0.5:
                    break
            except Exception:
                continue
    return dt if dt.notna().any() else None

# ============ HELPERS ============
QUANTITY_PATTERNS = re.compile(r"(quant|qtd|qtde|total|volume|contagem|qte|qtda)", re.I)
COMPETENCIA_CANDIDATES = ["txcompetencia", "competencia", "ano_mes", "anomes", "ano_mes_competencia"]
TIPO_CANDIDATES        = ["txtipopericia", "tipopericia", "tipo_pericia", "tipo"]
UNIDADE_CANDIDATES     = ["unidade_emissao", "unidade", "unidade_atendimento"]
ID_CANDIDATES          = ["iddocumento", "id_documento", "idatendimento", "id_atendimento", "n_laudo", "numero_laudo", "caso_sirsaelp"]
DATE_CANDIDATES        = ["data_interesse", "data", "dia", "data_base", "dhemitido", "dhatendimento", "dhsolicitacao", "data_emissao", "data_solicitacao"]

def infer_quantity_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if QUANTITY_PATTERNS.search(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    if "quantidade" in df.columns:
        return "quantidade"
    return None

def pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    if explicit and explicit in df.columns:
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    return None

def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))

def normalize_name(s: str) -> str:
    s = strip_accents(s.lower())
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")

def safe_period_str(df: pd.DataFrame) -> str:
    """
    Retorna 'YYYY-MM a YYYY-MM' de forma segura.
    Prioriza 'anomês_dt'. Se não houver, tenta usar strings 'anomês' válidas (YYYY-MM).
    """
    try:
        if "anomês_dt" in df.columns and df["anomês_dt"].notna().any():
            mi = pd.to_datetime(df["anomês_dt"]).min()
            ma = pd.to_datetime(df["anomês_dt"]).max()
            if pd.notna(mi) and pd.notna(ma):
                return f"{mi.strftime('%Y-%m')} a {ma.strftime('%Y-%m')}"
        if "anomês" in df.columns:
            s = df["anomês"].astype(str)
            s = s[s.str.match(r"^\d{4}-\d{2}$", na=False)]
            if not s.empty:
                # ordenar lexicalmente funciona para YYYY-MM
                return f"{s.min()} a {s.max()}"
    except Exception:
        pass
    return "Sem dados temporais"

def format_number(value: float, decimal_places: int = 0) -> str:
    if pd.isna(value):
        return "—"
    try:
        if decimal_places == 0:
            return f"{int(round(value)):,}".replace(",", ".")
        else:
            return f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "—"

# ============ ARQUIVOS / UPLOAD ============
@st.cache_data
def detect_data_sources():
    return os.path.exists("data") and any(p.lower().endswith(".csv") for p in os.listdir("data"))

has_data_dir = detect_data_sources()
st.sidebar.header("📁 Configuração de Dados")
if not has_data_dir:
    st.sidebar.info("💡 Envie os arquivos CSV disponíveis. O dashboard se adapta automaticamente.")

file_configs = {
    "Atendimentos_todos_Mensal": {
        "label": "Atendimentos Todos (Mensal)",
        "description": "Dados gerais de atendimentos por mês - agregados por competência",
        "pattern": ["atendimentos_todos", "atendimentos todos"]
    },
    "Laudos_todos_Mensal": {
        "label": "Laudos Todos (Mensal)",
        "description": "Dados gerais de laudos por mês - agregados por competência",
        "pattern": ["laudos_todos", "laudos todos"]
    },
    "Atendimentos_especifico_Mensal": {
        "label": "Atendimentos Específicos (Mensal)",
        "description": "Atendimentos detalhados por competência e tipo",
        "pattern": ["atendimentos_especifico", "atendimentos especifico"]
    },
    "Laudos_especifico_Mensal": {
        "label": "Laudos Específicos (Mensal)",
        "description": "Laudos detalhados por competência e tipo",
        "pattern": ["laudos_especifico", "laudos especifico"]
    },
    "laudos_realizados": {
        "label": "Laudos Realizados",
        "description": "Histórico detalhado de laudos concluídos com TME",
        "pattern": ["laudos_realizados", "laudos realizados"]
    },
    "detalhes_laudospendentes": {
        "label": "Laudos Pendentes",
        "description": "Laudos aguardando conclusão com aging",
        "pattern": ["laudospendentes", "laudos_pendentes", "detalhes_laudospendentes"]
    },
    "detalhes_examespendentes": {
        "label": "Exames Pendentes",
        "description": "Exames aguardando realização com aging",
        "pattern": ["examespendentes", "exames_pendentes", "detalhes_examespendentes"]
    },
    "laudos_Diário": {
        "label": "Laudos (Diário)",
        "description": "Registros diários de laudos emitidos",
        "pattern": ["laudos_diario", "laudos_diário", "laudos_di_rio"]
    },
    "atendimentos_Diário": {
        "label": "Atendimentos (Diário)",
        "description": "Registros diários de atendimentos",
        "pattern": ["atendimentos_diario", "atendimentos_diário", "atendimentos_di_rio"]
    },
}

uploads = {}
for key, config in file_configs.items():
    if not has_data_dir:
        uploads[key] = st.sidebar.file_uploader(
            f"{config['label']} (.csv)", help=config["description"], key=f"upload_{key}"
        )
    else:
        uploads[key] = None

def resolve_file_path(name: str) -> Optional[str]:
    if not os.path.exists("data"):
        return None
    config = file_configs.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    patterns.append(name.lower().replace(" ", "_"))
    patterns_norm = [normalize_name(p) for p in patterns]
    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        base = os.path.splitext(filename)[0]
        norm = normalize_name(base)
        for p in patterns_norm:
            if p in norm or norm.startswith(p):
                return os.path.join("data", filename)
    return None

def create_sample_laudos_realizados() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tipos = ["Química Forense", "Balística", "Informática Forense", "Traumatologia Forense"]
    unidades = ["Joinville", "Florianópolis", "Blumenau", "Chapecó", "Criciúma"]
    diretorias = ["Diretoria Criminal", "Diretoria Cível", "Diretoria Administrativa"]
    peritos = ["Alcides Ogliardi Junior", "Dr. Silva Santos", "Dra. Maria Oliveira", "Dr. João Pereira", "Dra. Ana Costa"]
    start = pd.Timestamp("2023-01-01"); end = pd.Timestamp("2024-12-31")
    rows = []
    for i in range(500):
        sol = start + pd.Timedelta(days=int(rng.integers(0, (end - start).days)))
        ate = sol + pd.Timedelta(days=int(rng.integers(1, 30)))
        emi = ate + pd.Timedelta(days=int(rng.integers(1, 120)))
        rows.append({
            "dhsolicitacao": sol.strftime("%d/%m/%Y"),
            "dhatendimento": ate.strftime("%d/%m/%Y"),
            "dhemitido": emi.strftime("%d/%m/%Y"),
            "n_laudo": f"L{2000+i}",
            "ano_emissao": emi.year,
            "mes_emissao": emi.month,
            "unidade_emissao": rng.choice(unidades),
            "diretoria": rng.choice(diretorias),
            "txcompetencia": f"{emi.year}-{emi.month:02d}",
            "txtipopericia": rng.choice(tipos),
            "perito": rng.choice(peritos)
        })
    return pd.DataFrame(rows)

@st.cache_data
def load_all_data(file_sources: Dict) -> Dict[str, pd.DataFrame]:
    loaded = {}
    for name, upload in file_sources.items():
        df = None
        if has_data_dir:
            path = resolve_file_path(name)
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        df = read_csv_optimized(f.read(), name)
                    if df is not None:
                        st.sidebar.success(f"✅ {name}: {len(df)} registros")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao carregar {name}: {e}")
        else:
            if upload is not None:
                try:
                    df = read_csv_optimized(upload.read(), name)
                    if df is not None:
                        st.sidebar.success(f"✅ {name}: {len(df)} registros")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao processar {name}: {e}")
        if df is not None:
            df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]
            loaded[name] = df

    if "laudos_realizados" not in loaded:
        st.sidebar.info("📊 Usando dados simulados para Laudos Realizados (demo)")
        loaded["laudos_realizados"] = create_sample_laudos_realizados()
    return loaded

raw_dataframes = load_all_data(uploads)
if not raw_dataframes:
    st.warning("⚠️ Nenhum arquivo foi carregado. Envie os CSVs na barra lateral ou coloque-os em `data/`.")
    st.info("📝 **Arquivos esperados:** " + ", ".join(file_configs.keys()))
    st.stop()

# ============ MAPEAMENTO ============
COLUMN_MAPPINGS = {
    "detalhes_laudospendentes": {
        "date": "data_solicitacao", "ano": "ano_sol", "id": "caso_sirsaelp",
        "unidade": "unidade", "superintendencia": "superintendencia",
        "diretoria": "diretoria", "competencia": "competencia",
        "tipo": "tipopericia", "perito": "perito"
    },
    "detalhes_examespendentes": {
        "date": "data_solicitacao", "ano": "ano_sol", "id": "caso_sirsaelp",
        "unidade": "unidade", "superintendencia": "superintendencia",
        "diretoria": "diretoria", "competencia": "competencia",
        "tipo": "tipopericia"
    },
    "Atendimentos_todos_Mensal": {"date": "data_interesse", "id": "idatendimento"},
    "Atendimentos_especifico_Mensal": {"date": "data_interesse", "competencia": "txcompetencia", "id": "idatendimento", "tipo": "txcompetencia"},
    "Laudos_todos_Mensal": {"date": "data_interesse", "id": "iddocumento"},
    "Laudos_especifico_Mensal": {"date": "data_interesse", "competencia": "txcompetencia", "id": "iddocumento", "tipo": "txcompetencia"},
    "laudos_realizados": {
        "solicitacao": "dhsolicitacao", "atendimento": "dhatendimento", "emissao": "dhemitido",
        "n_laudo": "n_laudo", "ano": "ano_emissao", "mes": "mes_emissao",
        "unidade": "unidade_emissao", "diretoria": "diretoria",
        "competencia": "txcompetencia", "tipo": "txtipopericia", "perito": "perito"
    },
    "laudos_Diário": {},
    "atendimentos_Diário": {},
}

# ============ PADRONIZAÇÃO ============
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapping = COLUMN_MAPPINGS.get(name, {})
    res = df.copy()

    # quantidade
    qcol = infer_quantity_col(res)
    if qcol:
        res["quantidade"] = pd.to_numeric(res[qcol], errors="coerce").fillna(0)
    else:
        res["quantidade"] = 1

    # dimensões
    tipo_col = pick_col(res, mapping.get("tipo"), TIPO_CANDIDATES)
    und_col  = pick_col(res, mapping.get("unidade"), UNIDADE_CANDIDATES)
    id_col   = pick_col(res, mapping.get("id"), ID_CANDIDATES)
    if tipo_col: res["tipo"] = res[tipo_col]
    if und_col:  res["unidade"] = res[und_col]
    if id_col:   res["id"] = res[id_col]
    if "perito" in res.columns: res["perito"] = res["perito"]
    if "diretoria" in res.columns: res["diretoria"] = res["diretoria"]
    if "superintendencia" in res.columns: res["superintendencia"] = res["superintendencia"]

    # competência mensal
    anomes_dt = None
    comp_col = pick_col(res, mapping.get("competencia"), COMPETENCIA_CANDIDATES)
    if comp_col and comp_col in res.columns:
        raw = res[comp_col].astype(str).str.replace(r"[^\d/.\-]", "", regex=True)
        for fmt in ("%Y-%m", "%Y/%m", "%Y%m", "%m/%Y"):
            anomes_dt = pd.to_datetime(raw, errors="coerce", format=fmt)
            if anomes_dt.notna().any():
                break
        if anomes_dt is None or anomes_dt.isna().all():
            anomes_dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if anomes_dt.notna().any():
            anomes_dt = anomes_dt.dt.to_period("M").dt.to_timestamp()

    # data diária
    day_col = pick_col(res, mapping.get("date"), DATE_CANDIDATES)
    data_base = pd.to_datetime(res[day_col], errors="coerce", dayfirst=True) if (day_col and day_col in res.columns) else None
    if (anomes_dt is None or anomes_dt.isna().all()) and (data_base is not None) and data_base.notna().any():
        anomes_dt = data_base.dt.to_period("M").dt.to_timestamp()

    if anomes_dt is not None and anomes_dt.notna().any():
        res["anomês_dt"] = anomes_dt
        res["anomês"] = anomes_dt.dt.strftime("%Y-%m")
        res["ano"] = anomes_dt.dt.year
        res["mes"] = anomes_dt.dt.month

    if data_base is not None and data_base.notna().any():
        res["data_base"] = data_base
        res["dia"] = data_base.dt.normalize()

    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            col = mapping.get(field)
            if col and col in res.columns:
                res[f"dh{field}"] = pd.to_datetime(res[col], errors="coerce", dayfirst=True)
        if "dhemissao" in res.columns:
            base = res.get("dhatendimento") if "dhatendimento" in res.columns else res.get("dhsolicitacao")
            if base is not None:
                res["tme_dias"] = (res["dhemissao"] - base).dt.days
                res["sla_30_ok"] = res["tme_dias"] <= 30
                res["sla_60_ok"] = res["tme_dias"] <= 60

    for c in ["diretoria", "superintendencia", "unidade", "tipo", "id", "perito", "anomês"]:
        if c in res.columns:
            res[c] = res[c].astype(str).str.strip().str.title().replace({"Nan": None, "": None, "None": None})
    return res

# Padroniza todos
standardized_dfs: Dict[str, pd.DataFrame] = {}
processing_info = []
for name, df in raw_dataframes.items():
    s = standardize_dataframe(name, df)
    standardized_dfs[name] = s
    processing_info.append({
        "Arquivo": name,
        "Linhas": len(s),
        "Período": safe_period_str(s)
    })

with st.sidebar.expander("📊 Resumo dos Dados", expanded=False):
    st.dataframe(pd.DataFrame(processing_info), use_container_width=True)

# ============ FILTROS ============
def extract_filter_values(column: str) -> List[str]:
    vals = set()
    for df in standardized_dfs.values():
        if column in df.columns:
            uv = df[column].dropna().astype(str).unique()
            vals.update(v for v in uv if v and v.lower() != "nan")
    return sorted(list(vals))

st.sidebar.subheader("🔍 Filtros")
filter_diretoria = st.sidebar.multiselect("Diretoria", extract_filter_values("diretoria"))
filter_superintendencia = st.sidebar.multiselect("Superintendência", extract_filter_values("superintendencia"))
filter_unidade = st.sidebar.multiselect("Unidade", extract_filter_values("unidade"))
filter_tipo = st.sidebar.multiselect("Tipo de Perícia", extract_filter_values("tipo"))
period_options = ["Todo o período", "Últimos 6 meses", "Últimos 3 meses", "Ano atual"]
filter_periodo = st.sidebar.selectbox("Período de análise", period_options)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = df.copy()
    for col, sel in [("diretoria", filter_diretoria), ("superintendencia", filter_superintendencia),
                     ("unidade", filter_unidade), ("tipo", filter_tipo)]:
        if col in f.columns and sel:
            f = f[f[col].astype(str).isin(sel)]
    if "anomês_dt" in f.columns and filter_periodo != "Todo o período":
        maxd = f["anomês_dt"].max()
        if pd.notna(maxd):
            if filter_periodo == "Últimos 3 meses":
                cutoff = maxd - pd.DateOffset(months=3)
            elif filter_periodo == "Últimos 6 meses":
                cutoff = maxd - pd.DateOffset(months=6)
            elif filter_periodo == "Ano atual":
                cutoff = pd.Timestamp(maxd.year, 1, 1)
            f = f[f["anomês_dt"] >= cutoff]
    return f

filtered_dfs = {n: apply_filters(df) for n, df in standardized_dfs.items()}

# Aliases
df_atend_todos = filtered_dfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dfs.get("Laudos_todos_Mensal")
df_atend_esp   = filtered_dfs.get("Atendimentos_especifico_Mensal")
df_laudos_esp  = filtered_dfs.get("Laudos_especifico_Mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("detalhes_laudospendentes")
df_pend_exames = filtered_dfs.get("detalhes_examespendentes")
df_laudos_di   = filtered_dfs.get("laudos_Diário")
df_atend_di    = filtered_dfs.get("atendimentos_Diário")

# ============ KPIs ============
def calculate_total(df: pd.DataFrame) -> int:
    if df is None or df.empty or "quantidade" not in df.columns:
        return 0
    return int(pd.to_numeric(df["quantidade"], errors="coerce").fillna(0).sum())

def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    m = df.groupby("anomês_dt")["quantidade"].sum()
    return m.mean() if len(m) > 0 else None

def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    s = df.groupby("anomês_dt")["quantidade"].sum().sort_index().tail(periods * 2)
    if len(s) < 2:
        return None
    mid = len(s) // 2
    a = s.iloc[:mid].mean()
    b = s.iloc[mid:].mean()
    return ((b - a) / a) * 100 if a > 0 else None

def calculate_productivity_metrics(df_atend: pd.DataFrame, df_laudos: pd.DataFrame) -> Dict:
    out = {}
    if df_atend is not None and df_laudos is not None:
        ta = calculate_total(df_atend)
        tl = calculate_total(df_laudos)
        if ta > 0:
            out["taxa_conversao"] = (tl / ta) * 100
        if "anomês_dt" in df_atend.columns and "anomês_dt" in df_laudos.columns:
            a = df_atend.groupby("anomês_dt")["quantidade"].sum()
            l = df_laudos.groupby("anomês_dt")["quantidade"].sum()
            common = a.index.intersection(l.index)
            if len(common) > 3:
                out["correlacao_atend_laudos"] = a.loc[common].corr(l.loc[common])
    return out

total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None and not df_pend_laudos.empty else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None and not df_pend_exames.empty else 0

media_mensal_laudos = calculate_monthly_average(df_laudos_todos)
backlog_meses = (total_pend_laudos / media_mensal_laudos) if media_mensal_laudos and media_mensal_laudos > 0 else None

prod = calculate_productivity_metrics(df_atend_todos, df_laudos_todos)
taxa_atendimento = prod.get("taxa_conversao")
correlacao_atend_laudos = prod.get("correlacao_atend_laudos")

crescimento_laudos = calculate_growth_rate(df_laudos_todos)
crescimento_atendimentos = calculate_growth_rate(df_atend_todos)

tme_mediano = tme_medio = sla_30_percent = sla_60_percent = None
if df_laudos_real is not None and not df_laudos_real.empty:
    if "tme_dias" in df_laudos_real.columns:
        t = pd.to_numeric(df_laudos_real["tme_dias"], errors="coerce").dropna()
        if not t.empty:
            tme_mediano = t.median(); tme_medio = t.mean()
    if "sla_30_ok" in df_laudos_real.columns:
        sla_30_percent = df_laudos_real["sla_30_ok"].mean() * 100
    if "sla_60_ok" in df_laudos_real.columns:
        sla_60_percent = df_laudos_real["sla_60_ok"].mean() * 100

aging_laudos_medio = aging_exames_medio = None
if df_pend_laudos is not None and not df_pend_laudos.empty and "data_base" in df_pend_laudos.columns:
    d = pd.to_datetime(df_pend_laudos["data_base"], errors="coerce")
    if d.notna().any():
        hoje = pd.Timestamp.now().normalize()
        aging_laudos_medio = (hoje - d).dt.days.mean()
if df_pend_exames is not None and not df_pend_exames.empty and "data_base" in df_pend_exames.columns:
    d = pd.to_datetime(df_pend_exames["data_base"], errors="coerce")
    if d.notna().any():
        hoje = pd.Timestamp.now().normalize()
        aging_exames_medio = (hoje - d).dt.days.mean()

# ============ KPIs UI ============
st.subheader("📈 Indicadores Principais")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Atendimentos Totais", format_number(total_atendimentos),
              delta=(f"+{format_number(crescimento_atendimentos,1)}%" if crescimento_atendimentos else None))
with c2:
    st.metric("Laudos Emitidos", format_number(total_laudos),
              delta=(f"+{format_number(crescimento_laudos,1)}%" if crescimento_laudos else None))
with c3:
    st.metric("Taxa de Conversão", f"{format_number(taxa_atendimento,1)}%" if taxa_atendimento else "—")
with c4:
    st.metric("Produtividade Mensal", format_number(media_mensal_laudos,1) if media_mensal_laudos else "—")

st.markdown("#### ⏰ Gestão de Pendências")
c5, c6, c7, c8 = st.columns(4)
with c5: st.metric("Laudos Pendentes", format_number(total_pend_laudos))
with c6: st.metric("Exames Pendentes", format_number(total_pend_exames))
with c7: st.metric("Backlog (meses)", format_number(backlog_meses,1) if backlog_meses else "—")
with c8:
    aging_m = aging_laudos_medio or aging_exames_medio
    st.metric("Aging Médio (dias)", format_number(aging_m,0) if aging_m else "—")

if tme_mediano is not None or sla_30_percent is not None:
    st.markdown("#### 🎯 Indicadores de Performance")
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.metric("TME Mediano (dias)", format_number(tme_mediano,1) if tme_mediano else "—")
    with d2: st.metric("TME Médio (dias)", format_number(tme_medio,1) if tme_medio else "—")
    with d3: st.metric("SLA 30 dias", f"{format_number(sla_30_percent,1)}%" if sla_30_percent else "—")
    with d4: st.metric("SLA 60 dias", f"{format_number(sla_60_percent,1)}%" if sla_60_percent else "—")

st.markdown("#### 🚨 Alertas e Insights")
alerts = []
if backlog_meses and backlog_meses > 6:
    alerts.append("🔴 **Backlog crítico**: Mais de 6 meses para liquidar pendências")
elif backlog_meses and backlog_meses > 3:
    alerts.append("🟡 **Atenção**: Backlog de pendências acima de 3 meses")
if sla_30_percent and sla_30_percent < 70:
    alerts.append("🔴 **SLA 30 dias baixo**: Menos de 70% dos laudos emitidos no prazo")
if taxa_atendimento and taxa_atendimento < 50:
    alerts.append("🟡 **Taxa de conversão baixa**: Menos de 50% dos atendimentos resultam em laudos")
if crescimento_laudos and crescimento_laudos < -10:
    alerts.append("🔴 **Queda na produção**: Redução de mais de 10% nos laudos emitidos")
if correlacao_atend_laudos and correlacao_atend_laudos < 0.5:
    alerts.append("🟡 **Descorrelação**: Atendimentos e laudos não estão alinhados temporalmente")
if alerts:
    for a in alerts: st.markdown(a)
else:
    st.success("✅ **Indicadores saudáveis**: Todos os KPIs estão dentro dos parâmetros esperados")
st.markdown("---")

# ============ ABAS ============
tab1, tabD, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral", "📆 Diário", "📈 Tendências", "🏆 Rankings", "⏰ Pendências", "📋 Dados", "📑 Relatórios"
])

# -------- Visão Geral --------
with tab1:
    st.subheader("📊 Resumo Executivo")
    if df_laudos_todos is not None and not df_laudos_todos.empty:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🏢 Performance por Unidade")
            if "unidade" in df_laudos_todos.columns:
                un = (df_laudos_todos.groupby("unidade", as_index=False)["quantidade"].sum()
                      .sort_values("quantidade", ascending=False).head(15))
                fig = px.bar(un, x="quantidade", y="unidade", orientation="h",
                             title="Top 15 Unidades - Laudos Emitidos",
                             color="quantidade", color_continuous_scale="Blues", text="quantidade")
                fig.update_traces(texttemplate="%{text}", textposition="outside")
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### 🔍 Distribuição por Tipo")
            if "tipo" in df_laudos_todos.columns:
                tp = (df_laudos_todos.groupby("tipo", as_index=False)["quantidade"].sum()
                      .sort_values("quantidade", ascending=False).head(10))
                pie = px.pie(tp, values="quantidade", names="tipo", title="Top 10 Tipos de Perícia")
                pie.update_traces(textposition="inside", textinfo="percent+label")
                pie.update_layout(height=500)
                st.plotly_chart(pie, use_container_width=True)

    if (df_atend_todos is not None and df_laudos_todos is not None and
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        st.markdown("#### 📅 Evolução Temporal Consolidada")
        a_m = df_atend_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        l_m = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        a_m["Tipo"] = "Atendimentos"; l_m["Tipo"] = "Laudos"
        a_m.rename(columns={"quantidade": "Total"}, inplace=True)
        l_m.rename(columns={"quantidade": "Total"}, inplace=True)
        comb = pd.concat([a_m, l_m]); comb["Mês"] = comb["anomês_dt"].dt.strftime("%Y-%m")
        line = px.line(comb, x="Mês", y="Total", color="Tipo", markers=True,
                       title="Evolução Mensal: Atendimentos vs Laudos", line_shape="spline")
        line.update_layout(height=400, hovermode="x unified", xaxis_title="Período", yaxis_title="Quantidade")
        st.plotly_chart(line, use_container_width=True)

        merged = pd.merge(
            a_m.rename(columns={"Total": "Atendimentos"}),
            l_m.rename(columns={"Total": "Laudos"}),
            on="anomês_dt", how="inner"
        )
        if not merged.empty:
            merged["Taxa_Conversao"] = merged["Laudos"] / merged["Atendimentos"] * 100
            merged["Mês"] = merged["anomês_dt"].dt.strftime("%Y-%m")
            lc = px.line(merged, x="Mês", y="Taxa_Conversao", markers=True,
                         title="Taxa de Conversão Mensal (%)", line_shape="spline")
            lc.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Meta: 70%")
            lc.update_layout(height=350, yaxis_title="Taxa de Conversão (%)", xaxis_title="Período")
            st.plotly_chart(lc, use_container_width=True)

# -------- Diário --------
with tabD:
    st.subheader("📆 Análise Diária (Laudos & Atendimentos)")
    def daily_summary(df: Optional[pd.DataFrame], label: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if "dia" not in df.columns:
            base = None
            if "data_base" in df.columns:
                base = pd.to_datetime(df["data_base"], errors="coerce", dayfirst=True)
            else:
                for c in DATE_CANDIDATES:
                    if c in df.columns:
                        base = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                        break
            if base is None:
                return None
            df = df.copy(); df["dia"] = base.dt.normalize()
        if df["dia"].isna().all():
            return None
        q = pd.to_numeric(df.get("quantidade", 1), errors="coerce").fillna(1)
        return df.assign(_q=q).groupby("dia", as_index=False)["_q"].sum().rename(columns={"_q": label}).sort_values("dia")

    daily_l = daily_summary(df_laudos_di, "Laudos")
    daily_a = daily_summary(df_atend_di, "Atendimentos")
    if daily_l is None and daily_a is None:
        st.info("Não há dados diários disponíveis nos arquivos enviados.")
    else:
        if daily_l is not None and daily_a is not None:
            daily = pd.merge(daily_a, daily_l, on="dia", how="outer").sort_values("dia")
        elif daily_l is not None:
            daily = daily_l.copy(); daily["Atendimentos"] = 0
        else:
            daily = daily_a.copy(); daily["Laudos"] = 0

        daily[["Atendimentos", "Laudos"]] = daily[["Atendimentos", "Laudos"]].fillna(0).astype(int)
        cutoff = daily["dia"].max() - pd.Timedelta(days=29) if not daily["dia"].isna().all() else None
        last30 = daily[daily["dia"] >= cutoff] if cutoff is not None else daily.copy()

        e1, e2, e3, e4 = st.columns(4)
        with e1: st.metric("Laudos (30d)", format_number(last30["Laudos"].sum()))
        with e2: st.metric("Atendimentos (30d)", format_number(last30["Atendimentos"].sum()))
        with e3: st.metric("Média diária Laudos (30d)", format_number(last30["Laudos"].mean(), 1) if not last30.empty else "—")
        with e4: st.metric("Média diária Atend. (30d)", format_number(last30["Atendimentos"].mean(), 1) if not last30.empty else "—")

        dplot = daily.copy()
        dplot["MM7 Laudos"] = dplot["Laudos"].rolling(7).mean()
        dplot["MM7 Atendimentos"] = dplot["Atendimentos"].rolling(7).mean()
        dplot["Dia"] = dplot["dia"].dt.strftime("%Y-%m-%d")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dplot["Dia"], y=dplot["Atendimentos"], mode="lines", name="Atendimentos"))
        fig.add_trace(go.Scatter(x=dplot["Dia"], y=dplot["Laudos"], mode="lines", name="Laudos"))
        fig.add_trace(go.Scatter(x=dplot["Dia"], y=dplot["MM7 Atendimentos"], mode="lines", name="MM7 Atendimentos", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=dplot["Dia"], y=dplot["MM7 Laudos"], mode="lines", name="MM7 Laudos", line=dict(dash="dash")))
        fig.update_layout(title="Série Diária (MM7)", height=420, hovermode="x unified", xaxis_title="Dia", yaxis_title="Quantidade")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Tabela: Laudos e Atendimentos por dia")
        last_n = st.slider("Dias a exibir", min_value=30, max_value=365, value=90, step=30)
        cutoff_tbl = daily["dia"].max() - pd.Timedelta(days=last_n-1) if not daily["dia"].isna().all() else None
        tbl = daily[daily["dia"] >= cutoff_tbl].copy() if cutoff_tbl is not None else daily.copy()
        tbl = tbl.sort_values("dia", ascending=False)
        tbl["dia"] = tbl["dia"].dt.strftime("%d/%m/%Y")
        st.dataframe(tbl, use_container_width=True, height=360)
        st.download_button("📥 Baixar tabela diária (CSV)", data=tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"diario_laudos_atendimentos_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# -------- Tendências --------
with tab2:
    st.subheader("📈 Análise de Tendências")
    def create_enhanced_time_series(df: pd.DataFrame, title: str, color: str = "blue") -> None:
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        m = (df.groupby("anomês_dt", as_index=False)["quantidade"].sum().sort_values("anomês_dt"))
        if m.empty:
            st.info(f"Sem dados temporais para {title}")
            return
        m["Mês"] = m["anomês_dt"].dt.strftime("%Y-%m")
        fig = make_subplots(rows=2, cols=1, subplot_titles=(title, "Variação Percentual Mensal"),
                            vertical_spacing=0.15, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=m["Mês"], y=m["quantidade"], mode="lines+markers", name="Valores",
                                 line=dict(color=color, width=2)), row=1, col=1)
        if len(m) >= 3:
            m["mm3"] = m["quantidade"].rolling(window=3, center=True).mean()
            fig.add_trace(go.Scatter(x=m["Mês"], y=m["mm3"], mode="lines", name="Média Móvel (3m)",
                                     line=dict(dash="dash", color="red", width=2)), row=1, col=1)
        m["var%"] = m["quantidade"].pct_change() * 100
        cols = ["red" if x < 0 else "green" for x in m["var%"].fillna(0)]
        fig.add_trace(go.Bar(x=m["Mês"], y=m["var%"], name="Variação %", marker_color=cols, showlegend=False), row=2, col=1)
        fig.update_layout(height=600, hovermode="x unified", showlegend=True)
        fig.update_xaxes(title_text="Período", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        create_enhanced_time_series(df_atend_todos, "🏥 Atendimentos - Análise Temporal", "blue")
    with colB:
        create_enhanced_time_series(df_laudos_todos, "📄 Laudos - Análise Temporal", "green")

    if (df_atend_todos is not None and df_laudos_todos is not None and
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        st.markdown("#### 🔗 Correlação Atendimentos x Laudos")
        a = df_atend_todos.groupby("anomês_dt")["quantidade"].sum()
        l = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum()
        common = a.index.intersection(l.index)
        if len(common) > 3:
            data = pd.DataFrame({"anomês_dt": common, "Atendimentos": a.loc[common].values, "Laudos": l.loc[common].values})
            data["Período"] = data["anomês_dt"].dt.strftime("%Y-%m")
            sc = px.scatter(data, x="Atendimentos", y="Laudos", hover_data=["Período"],
                            title="Correlação: Atendimentos vs Laudos", trendline="ols")
            coef = data["Atendimentos"].corr(data["Laudos"])
            sc.add_annotation(text=f"Correlação: {coef:.3f}", xref="paper", yref="paper", x=0.02, y=0.98,
                              showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            sc.update_layout(height=400)
            st.plotly_chart(sc, use_container_width=True)

# -------- Rankings --------
with tab3:
    st.subheader("🏆 Rankings e Comparativos")
    def create_enhanced_ranking(df: pd.DataFrame, dim: str, title: str, top_n: int = 20) -> None:
        if df is None or df.empty or dim not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        g = df.groupby(dim).agg({"quantidade": ["sum", "count", "mean"]}).round(2)
        g.columns = ["Total", "Registros", "Média"]
        g = g.sort_values("Total", ascending=False).head(top_n).reset_index()
        if g.empty:
            st.info(f"Sem dados para {title}")
            return
        fig = px.bar(g, x="Total", y=dim, orientation="h", title=title,
                     color="Total", color_continuous_scale="Viridis", hover_data=["Registros", "Média"])
        fig.update_layout(height=max(400, len(g) * 30), showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"📊 Detalhes - {title}"):
            st.dataframe(g, use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["Por Diretoria", "Por Unidade", "Por Tipo", "Comparativo"])
    with t1:
        a, b = st.columns(2)
        with a: create_enhanced_ranking(df_atend_todos, "diretoria", "🏥 Atendimentos por Diretoria")
        with b: create_enhanced_ranking(df_laudos_todos, "diretoria", "📄 Laudos por Diretoria")
    with t2:
        a, b = st.columns(2)
        with a: create_enhanced_ranking(df_atend_todos, "unidade", "🏥 Atendimentos por Unidade", 25)
        with b: create_enhanced_ranking(df_laudos_todos, "unidade", "📄 Laudos por Unidade", 25)
    with t3:
        a, b = st.columns(2)
        with a: create_enhanced_ranking(df_atend_esp, "tipo", "🏥 Atendimentos por Tipo", 20)
        with b: create_enhanced_ranking(df_laudos_esp, "tipo", "📄 Laudos por Tipo", 20)
    with t4:
        if (df_atend_todos is not None and df_laudos_todos is not None and
            "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns):
            au = df_atend_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Atendimentos"})
            lu = df_laudos_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Laudos"})
            ef = pd.merge(au, lu, on="unidade", how="inner")
            if not ef.empty:
                ef["Taxa_Conversao"] = ef["Laudos"] / ef["Atendimentos"] * 100
                ef = ef.sort_values("Taxa_Conversao", ascending=False)
                fig = px.scatter(ef.head(20), x="Atendimentos", y="Laudos", size="Taxa_Conversao",
                                 hover_name="unidade", title="Eficiência por Unidade",
                                 color="Taxa_Conversao", color_continuous_scale="RdYlGn")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**🥇 Top 10 Unidades Mais Eficientes:**")
                st.dataframe(ef.head(10)[["unidade", "Taxa_Conversao", "Atendimentos", "Laudos"]],
                             use_container_width=True)

# -------- Pendências --------
with tab4:
    st.subheader("⏰ Gestão de Pendências")
    def calculate_aging_analysis(df: pd.DataFrame, date_column: str = "data_base"):
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype="int64"), {}
        dcols = [c for c in df.columns if "data" in c.lower()]
        if date_column not in df.columns and dcols:
            date_column = dcols[0]
        if date_column not in df.columns:
            return df, pd.Series(dtype="int64"), {}
        res = df.copy()
        dates = pd.to_datetime(res[date_column], errors="coerce")
        if dates.isna().all():
            return df, pd.Series(dtype="int64"), {}
        hoje = pd.Timestamp.now().normalize()
        dias = (hoje - dates).dt.days
        fx = pd.cut(dias, bins=[-1,15,30,60,90,180,365,float("inf")],
                    labels=["0-15 dias","16-30 dias","31-60 dias","61-90 dias","91-180 dias","181-365 dias","> 365 dias"])
        res["dias_pendentes"] = dias; res["faixa_aging"] = fx
        res["prioridade"] = pd.cut(dias, bins=[-1,30,90,180,float("inf")], labels=["Normal","Atenção","Urgente","Crítico"])
        dist = fx.value_counts().sort_index()
        stats = {"total": len(res), "media_dias": dias.mean(), "mediana_dias": dias.median(),
                 "max_dias": dias.max(), "criticos": (res["prioridade"]=="Crítico").sum(),
                 "urgentes": (res["prioridade"]=="Urgente").sum()}
        return res, dist, stats

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("#### 📄 Laudos Pendentes")
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            la, dl, sl = calculate_aging_analysis(df_pend_laudos)
            a,b,c = st.columns(3)
            with a: st.metric("Total", format_number(sl.get("total",0)))
            with b: st.metric("Críticos", sl.get("criticos",0))
            with c: st.metric("Média (dias)", format_number(sl.get("media_dias",0),1))
            if not dl.empty:
                fig = px.bar(x=dl.index, y=dl.values, title="Distribuição por Tempo de Pendência",
                             color=dl.values, color_continuous_scale="Reds", text=dl.values)
                fig.update_traces(texttemplate="%{text}", textposition="outside")
                fig.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(fig, use_container_width=True)
            if "prioridade" in la.columns:
                p = la["prioridade"].value_counts()
                pie = px.pie(values=p.values, names=p.index, title="Distribuição por Prioridade",
                             color_discrete_map={"Normal":"green","Atenção":"yellow","Urgente":"orange","Crítico":"red"})
                pie.update_layout(height=300)
                st.plotly_chart(pie, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in la.columns:
                cols = [c for c in ["id","unidade","tipo","dias_pendentes","prioridade"] if c in la.columns]
                top = la.nlargest(10, "dias_pendentes")[cols] if cols else la.nlargest(10, "dias_pendentes")
                st.dataframe(top, use_container_width=True, height=250)
        else:
            st.info("Sem dados de laudos pendentes disponíveis.")
    with rcol:
        st.markdown("#### 🔬 Exames Pendentes")
        if df_pend_exames is not None and not df_pend_exames.empty:
            ex, de, se = calculate_aging_analysis(df_pend_exames)
            a,b,c = st.columns(3)
            with a: st.metric("Total", format_number(se.get("total",0)))
            with b: st.metric("Críticos", se.get("criticos",0))
            with c: st.metric("Média (dias)", format_number(se.get("media_dias",0),1))
            if not de.empty:
                fig = px.bar(x=de.index, y=de.values, title="Distribuição por Tempo de Pendência",
                             color=de.values, color_continuous_scale="Oranges", text=de.values)
                fig.update_traces(texttemplate="%{text}", textposition="outside")
                fig.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(fig, use_container_width=True)
            if "prioridade" in ex.columns:
                p = ex["prioridade"].value_counts()
                pie = px.pie(values=p.values, names=p.index, title="Distribuição por Prioridade",
                             color_discrete_map={"Normal":"green","Atenção":"yellow","Urgente":"orange","Crítico":"red"})
                pie.update_layout(height=300)
                st.plotly_chart(pie, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in ex.columns:
                cols = [c for c in ["id","unidade","tipo","dias_pendentes","prioridade"] if c in ex.columns]
                top = ex.nlargest(10, "dias_pendentes")[cols] if cols else ex.nlargest(10, "dias_pendentes")
                st.dataframe(top, use_container_width=True, height=250)
        else:
            st.info("Sem dados de exames pendentes disponíveis.")

# -------- Dados --------
with tab5:
    st.subheader("📋 Exploração dos Dados")
    st.markdown("#### 📊 Resumo dos Datasets Carregados")
    summary = []
    for name, df in standardized_dfs.items():
        if df is not None and not df.empty:
            summary.append({
                "Dataset": name.replace("_"," ").title(),
                "Registros": f"{len(df):,}".replace(",", "."),
                "Colunas": len(df.columns),
                "Período": safe_period_str(df),
                "Tamanho (MB)": round(df.memory_usage(deep=True).sum()/1024/1024, 2),
                "Status": "✅ Carregado"
            })
    if summary:
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

    st.markdown("#### 🔍 Exploração Detalhada")
    options = [n for n, df in standardized_dfs.items() if df is not None]
    if options:
        sel = st.selectbox("Selecione o dataset para explorar:", options, format_func=lambda x: x.replace("_"," ").title())
        if sel:
            dsel = standardized_dfs[sel]
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Registros", f"{len(dsel):,}".replace(",", "."))
            with c2: st.metric("Colunas", len(dsel.columns))
            with c3: st.metric("Valores Nulos", f"{dsel.isnull().sum().sum():,}".replace(",", "."))
            with c4:
                if "anomês_dt" in dsel.columns:
                    st.metric("Meses Únicos", dsel["anomês_dt"].nunique())
                else:
                    st.metric("Período", "N/A")

            with st.expander("🔍 Análise de Qualidade dos Dados", expanded=False):
                qinfo = []
                for col in dsel.columns:
                    dtype = str(dsel[col].dtype)
                    null_count = dsel[col].isnull().sum()
                    null_percent = (null_count / len(dsel)) * 100
                    unique_count = dsel[col].nunique()
                    quality = "🟢 Excelente" if null_percent == 0 else ("🟡 Boa" if null_percent < 5 else ("🟠 Regular" if null_percent < 20 else "🔴 Ruim"))
                    qinfo.append({
                        "Coluna": col, "Tipo": dtype, "Nulos": f"{null_count:,}".replace(",", "."),
                        "% Nulos": f"{null_percent:.1f}%", "Únicos": f"{unique_count:,}".replace(",", "."),
                        "Qualidade": quality
                    })
                st.dataframe(pd.DataFrame(qinfo), use_container_width=True)

            st.markdown("**🎛️ Controles de Visualização:**")
            v1, v2, v3 = st.columns(3)
            with v1:
                max_rows = st.number_input("Máximo de linhas:", min_value=10, max_value=5000, value=500, step=50)
            with v2:
                if "anomês" in dsel.columns:
                    meses = sorted(dsel["anomês"].dropna().astype(str).unique(), reverse=True)
                    meses_sel = st.multiselect("Filtrar por período:", meses, default=meses[:6] if len(meses) > 6 else meses)
                else:
                    meses_sel = []
            with v3:
                cols = list(dsel.columns)
                cols_sel = st.multiselect("Colunas a exibir:", cols, default=cols[:10] if len(cols) > 10 else cols)

            dshow = dsel.copy()
            if meses_sel and "anomês" in dshow.columns:
                dshow = dshow[dshow["anomês"].astype(str).isin(meses_sel)]
            if cols_sel:
                dshow = dshow[cols_sel]
            dshow = dshow.head(max_rows)

            if not dshow.empty:
                st.markdown("**📈 Estatísticas Descritivas:**")
                num_cols = dshow.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    st.dataframe(dshow[num_cols].describe().round(2), use_container_width=True)
                else:
                    st.info("Nenhuma coluna numérica encontrada para estatísticas.")

            st.markdown(f"**📋 Dados Filtrados ({len(dshow):,} de {len(dsel):,} registros):**".replace(",", "."))
            st.dataframe(dshow, use_container_width=True, height=400)

            cdl, cdc = st.columns(2)
            with cdl:
                st.download_button("📥 Download Dados Filtrados (CSV)",
                                   data=dshow.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{sel}_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv")
            with cdc:
                st.download_button("📥 Download Dataset Completo (CSV)",
                                   data=dsel.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{sel}_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv")

# -------- Relatórios --------
with tab6:
    st.subheader("📑 Relatórios Executivos")
    tipo = st.selectbox("Tipo de Relatório:",
                        ["Relatório Executivo Completo","Relatório de Produção","Relatório de Pendências","Relatório de Performance","Relatório Comparativo"])

    def gerar_relatorio_executivo() -> str:
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        rel = f"""
# RELATÓRIO EXECUTIVO PCI/SC
**Data de Geração:** {ts}
**Período de Análise:** {filter_periodo}

## 📊 RESUMO EXECUTIVO
- **Atendimentos Totais:** {format_number(total_atendimentos)}
- **Laudos Emitidos:** {format_number(total_laudos)}
- **Taxa de Conversão:** {format_number(taxa_atendimento, 1) if taxa_atendimento else 'N/A'}%
- **Produtividade Mensal:** {format_number(media_mensal_laudos, 1) if media_mensal_laudos else 'N/A'} laudos/mês

## ⏰ PENDÊNCIAS
- **Laudos Pendentes:** {format_number(total_pend_laudos)}
- **Exames Pendentes:** {format_number(total_pend_exames)}
- **Backlog Estimado:** {format_number(backlog_meses, 1) if backlog_meses else 'N/A'} meses

## 🎯 PERFORMANCE
- **TME Mediano:** {format_number(tme_mediano, 1) if tme_mediano else 'N/A'} dias
- **SLA 30 dias:** {format_number(sla_30_percent, 1) if sla_30_percent else 'N/A'}%
- **SLA 60 dias:** {format_number(sla_60_percent, 1) if sla_60_percent else 'N/A'}%
"""
        return rel.strip()

    if tipo == "Relatório Executivo Completo":
        txt = gerar_relatorio_executivo()
        st.markdown("#### 📄 Visualização do Relatório")
        st.markdown(txt)
        st.download_button("📥 Download Relatório Executivo",
                           data=txt.encode("utf-8"),
                           file_name=f"relatorio_executivo_pci_sc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                           mime="text/markdown")
    elif tipo == "Relatório de Produção":
        st.markdown("#### 📊 Relatório de Produção")
        if df_laudos_todos is not None and "anomês" in df_laudos_todos.columns:
            prod = df_laudos_todos.groupby("anomês")["quantidade"].sum().reset_index().sort_values("anomês")
            st.line_chart(prod.set_index("anomês")["quantidade"], height=300)
    else:
        st.info(f"Relatório '{tipo}' em desenvolvimento.")

# ============ Rodapé ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.1</strong> - Sistema Avançado de Monitoramento</p>
    <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
    <p>Para suporte técnico ou sugestões: <strong>equipe-ti@pci.sc.gov.br</strong></p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)
