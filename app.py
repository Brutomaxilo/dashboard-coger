import io
import os
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============ CONFIGURAÇÃO INICIAL ============
st.set_page_config(
    page_title="PCI/SC – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Dashboard PCI/SC – Produção & Pendências")
st.markdown("---")

# ----------------- BOTÃO LIMPAR CACHE (útil durante ajustes) -----------------
if st.sidebar.button("🧹 Limpar cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache limpo. Recarregue a página (Ctrl+R).")

# ============ CACHE E PERFORMANCE ============
@st.cache_data
def read_csv_optimized(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Lê CSV com detecção automática de separador e encoding otimizada."""
    separators = [";", ",", "\t", "|"]
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(bio, sep=sep, encoding=encoding, engine="python")
                if df.shape[1] > 1:
                    df.columns = [col.strip('"').strip() for col in df.columns]
                    for col in df.columns:
                        if df[col].dtype == "object":
                            df[col] = df[col].astype(str).str.strip('"').str.strip()
                    return df
            except Exception:
                continue

    try:
        bio = io.BytesIO(file_content)
        df = pd.read_csv(bio, sep=None, engine="python", encoding="utf-8")
        if df.shape[1] > 1:
            df.columns = [col.strip('"').strip() for col in df.columns]
            return df
    except Exception:
        pass

    return None

@st.cache_data
def process_datetime_column(series: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    """Processa coluna de data/hora com múltiplos formatos."""
    if series is None or series.empty:
        return None

    dt_series = pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=dayfirst,
        infer_datetime_format=True
    )

    if dt_series.isna().sum() > len(dt_series) * 0.5:
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                dt_series = pd.to_datetime(series, format=fmt, errors="coerce")
                if dt_series.notna().sum() > len(dt_series) * 0.5:
                    break
            except Exception:
                continue

    return dt_series if dt_series.notna().any() else None

# ============ HELPERS (devem vir ANTES da padronização) ============
QUANTITY_PATTERNS = re.compile(r"(quant|qtd|qtde|total|volume|contagem|qte|qtda)", re.I)
COMPETENCIA_CANDIDATES = ["txcompetencia", "competencia", "ano_mes", "anomes", "ano_mes_competencia"]
TIPO_CANDIDATES        = ["txtipopericia", "tipopericia", "tipo_pericia", "tipo"]
UNIDADE_CANDIDATES     = ["unidade_emissao", "unidade", "unidade_atendimento"]
ID_CANDIDATES          = ["iddocumento", "id_documento", "idatendimento", "id_atendimento", "n_laudo", "numero_laudo", "caso_sirsaelp"]
DATE_CANDIDATES        = ["data_interesse", "data", "dia", "data_base", "dhemitido", "dhatendimento", "dhsolicitacao", "data_emissao", "data_solicitacao"]

def infer_quantity_col(df: pd.DataFrame) -> Optional[str]:
    """Tenta achar coluna numérica que represente quantidade."""
    for c in df.columns:
        if QUANTITY_PATTERNS.search(str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback: se houver uma coluna chamada 'quantidade' mas está texto, ainda tentamos
    if "quantidade" in df.columns:
        return "quantidade"
    return None

def pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    """Escolhe coluna: preferir mapeamento explícito; senão tentar candidatos."""
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

# ============ UTILITÁRIOS ============
def format_number(value: float, decimal_places: int = 0) -> str:
    """Formata números com separadores brasileiros."""
    if pd.isna(value):
        return "—"
    try:
        if decimal_places == 0:
            return f"{int(round(value)):,}".replace(",", ".")
        else:
            return f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "—"

def calculate_percentage(numerator: float, denominator: float) -> Optional[float]:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return (numerator / denominator) * 100

def get_period_filter_options(df: pd.DataFrame) -> List[str]:
    if df is None or "anomês_dt" not in df.columns:
        return []
    dates = df["anomês_dt"].dropna()
    if dates.empty:
        return []
    periods = ["Últimos 3 meses", "Últimos 6 meses", "Último ano", "Ano atual", "Todo o período"]
    return periods

# ============ DETECÇÃO DE ARQUIVOS ============
@st.cache_data
def detect_data_sources():
    """Detecta se existem arquivos na pasta data/."""
    return os.path.exists("data") and any(p.lower().endswith(".csv") for p in os.listdir("data"))

has_data_dir = detect_data_sources()

# ============ INTERFACE DE UPLOAD ============
st.sidebar.header("📁 Configuração de Dados")

if not has_data_dir:
    st.sidebar.info("💡 Envie os arquivos CSV disponíveis. O dashboard se adapta automaticamente.")

# Definição dos arquivos esperados (inclui DIÁRIO)
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
    # --- Novos diários ---
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
            f"{config['label']} (.csv)",
            help=config['description'],
            key=f"upload_{key}"
        )
    else:
        uploads[key] = None

# ============ RESOLUÇÃO DE ARQUIVOS ============
def resolve_file_path(name: str) -> Optional[str]:
    """Resolve caminho do arquivo com tolerância a variações de nome e acentos."""
    if not os.path.exists("data"):
        return None

    config = file_configs.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    patterns.append(name.lower().replace(" ", "_"))
    patterns_norm = [normalize_name(p) for p in patterns]

    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        base_name = os.path.splitext(filename)[0]
        normalized_name = normalize_name(base_name)
        for pattern in patterns_norm:
            if pattern in normalized_name or normalized_name.startswith(pattern):
                return os.path.join("data", filename)
    return None

# ============ DADOS SIMULADOS PARA DEMO ============
def create_sample_laudos_realizados() -> pd.DataFrame:
    sample_data = []
    tipos_pericia = [
        "Química Forense", "Criminal Local de crime contra o patrimônio",
        "Criminal Local de crime contra a vida", "Criminal Engenharia Forense",
        "Criminal Identificação de veículos", "Criminal Identificação",
        "Informática Forense", "Balística", "Traumatologia Forense"
    ]
    unidades = ["Joinville", "Florianópolis", "Blumenau", "Chapecó", "Criciúma"]
    diretorias = ["Diretoria Criminal", "Diretoria Cível", "Diretoria Administrativa"]
    peritos = ["Alcides Ogliardi Junior", "Dr. Silva Santos", "Dra. Maria Oliveira", "Dr. João Pereira", "Dra. Ana Costa"]

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2024-12-31")

    rng = np.random.default_rng(42)
    total = 500
    for i in range(total):
        solicitacao = start_date + pd.Timedelta(days=int(rng.integers(0, (end_date - start_date).days)))
        atendimento = solicitacao + pd.Timedelta(days=int(rng.integers(1, 30)))
        emissao = atendimento + pd.Timedelta(days=int(rng.integers(1, 120)))
        sample_data.append({
            "dhsolicitacao": solicitacao.strftime("%d/%m/%Y"),
            "dhatendimento": atendimento.strftime("%d/%m/%Y"),
            "dhemitido": emissao.strftime("%d/%m/%Y"),
            "n_laudo": f"L{2000 + i}",
            "ano_emissao": emissao.year,
            "mes_emissao": emissao.month,
            "unidade_emissao": rng.choice(unidades),
            "diretoria": rng.choice(diretorias),
            "txcompetencia": f"{emissao.year}-{emissao.month:02d}",
            "txtipopericia": rng.choice(tipos_pericia),
            "perito": rng.choice(peritos)
        })
    return pd.DataFrame(sample_data)

# ============ CARREGAMENTO DE DADOS ============
@st.cache_data
def load_all_data(file_sources: Dict) -> Dict[str, pd.DataFrame]:
    """Carrega todos os dados disponíveis."""
    loaded_data = {}

    for name, upload_file in file_sources.items():
        df = None

        if has_data_dir:
            file_path = resolve_file_path(name)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                    df = read_csv_optimized(content, name)
                    if df is not None:
                        st.sidebar.success(f"✅ {name}: {len(df)} registros")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao carregar {name}: {str(e)}")
        else:
            if upload_file is not None:
                try:
                    content = upload_file.read()
                    df = read_csv_optimized(content, name)
                    if df is not None:
                        st.sidebar.success(f"✅ {name}: {len(df)} registros")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao processar {name}: {str(e)}")

        if df is not None:
            df.columns = [re.sub(r"\s+", " ", col.strip().lower()) for col in df.columns]
            loaded_data[name] = df

    if "laudos_realizados" not in loaded_data:
        st.sidebar.info("📊 Usando dados simulados para Laudos Realizados (demo)")
        loaded_data["laudos_realizados"] = create_sample_laudos_realizados()

    return loaded_data

# Carrega os dados
raw_dataframes = load_all_data(uploads)

if not raw_dataframes:
    st.warning("⚠️ Nenhum arquivo foi carregado. Por favor, envie os arquivos CSV pela barra lateral ou coloque-os na pasta `data/`.")
    st.info("📝 **Arquivos esperados:** " + ", ".join(file_configs.keys()))
    st.stop()

# ============ MAPEAMENTO DE COLUNAS ============
COLUMN_MAPPINGS = {
    "detalhes_laudospendentes": {
        "date": "data_solicitacao",
        "ano": "ano_sol",
        "id": "caso_sirsaelp",
        "unidade": "unidade",
        "superintendencia": "superintendencia",
        "diretoria": "diretoria",
        "competencia": "competencia",
        "tipo": "tipopericia",
        "perito": "perito"
    },
    "detalhes_examespendentes": {
        "date": "data_solicitacao",
        "ano": "ano_sol",
        "id": "caso_sirsaelp",
        "unidade": "unidade",
        "superintendencia": "superintendencia",
        "diretoria": "diretoria",
        "competencia": "competencia",
        "tipo": "tipopericia"
    },
    "Atendimentos_todos_Mensal": {
        "date": "data_interesse",
        "id": "idatendimento",
        # se existir coluna de quantidade nos arquivos reais, o helper detecta
    },
    "Atendimentos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "idatendimento",
        "tipo": "txcompetencia"
    },
    "Laudos_todos_Mensal": {
        "date": "data_interesse",
        "id": "iddocumento",
    },
    "Laudos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "iddocumento",
        "tipo": "txcompetencia"
    },
    "laudos_realizados": {
        "solicitacao": "dhsolicitacao",
        "atendimento": "dhatendimento",
        "emissao": "dhemitido",
        "n_laudo": "n_laudo",
        "ano": "ano_emissao",
        "mes": "mes_emissao",
        "unidade": "unidade_emissao",
        "diretoria": "diretoria",
        "competencia": "txcompetencia",
        "tipo": "txtipopericia",
        "perito": "perito"
    },
    # Diários: usamos inferência automática para data/quantidade
    "laudos_Diário": {},
    "atendimentos_Diário": {},
}

# ============ PADRONIZAÇÃO DE DADOS ============
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza estrutura do DataFrame para análise unificada (mensal e diário)."""
    if df is None or df.empty:
        return pd.DataFrame()

    mapping = COLUMN_MAPPINGS.get(name, {})
    result = df.copy()

    # ---------- QUANTIDADE ----------
    qcol = infer_quantity_col(result)
    if qcol:
        result["quantidade"] = pd.to_numeric(result[qcol], errors="coerce").fillna(0)
    else:
        result["quantidade"] = 1

    # ---------- DIMENSÕES ----------
    tipo_col    = pick_col(result, mapping.get("tipo"), TIPO_CANDIDATES)
    unidade_col = pick_col(result, mapping.get("unidade"), UNIDADE_CANDIDATES)
    id_col      = pick_col(result, mapping.get("id"), ID_CANDIDATES)

    if tipo_col:    result["tipo"] = result[tipo_col]
    if unidade_col: result["unidade"] = result[unidade_col]
    if id_col:      result["id"] = result[id_col]
    if "perito" in result.columns:            result["perito"] = result["perito"]
    if "diretoria" in result.columns:         result["diretoria"] = result["diretoria"]
    if "superintendencia" in result.columns:  result["superintendencia"] = result["superintendencia"]

    # ---------- TEMPO: competência (mensal) ----------
    anomes_dt = None
    comp_col = pick_col(result, mapping.get("competencia"), COMPETENCIA_CANDIDATES)
    if comp_col and comp_col in result.columns:
        raw = result[comp_col].astype(str).str.replace(r"[^\d/.\-]", "", regex=True)
        for fmt in ("%Y-%m", "%Y/%m", "%Y%m", "%m/%Y"):
            anomes_dt = pd.to_datetime(raw, errors="coerce", format=fmt)
            if anomes_dt.notna().any():
                break
        if anomes_dt is None or anomes_dt.isna().all():
            anomes_dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if anomes_dt.notna().any():
            anomes_dt = anomes_dt.dt.to_period("M").dt.to_timestamp()

    # ---------- TEMPO: data diária ----------
    day_col = pick_col(result, mapping.get("date"), DATE_CANDIDATES)
    data_base = pd.to_datetime(result[day_col], errors="coerce", dayfirst=True) if (day_col and day_col in result.columns) else None

    # se não veio competência mas temos data diária, preenche o mês
    if (anomes_dt is None or anomes_dt.isna().all()) and (data_base is not None) and data_base.notna().any():
        anomes_dt = data_base.dt.to_period("M").dt.to_timestamp()

    if anomes_dt is not None and anomes_dt.notna().any():
        result["anomês_dt"] = anomes_dt
        result["anomês"]    = anomes_dt.dt.strftime("%Y-%m")
        result["ano"]       = anomes_dt.dt.year
        result["mes"]       = anomes_dt.dt.month

    if data_base is not None and data_base.notna().any():
        result["data_base"] = data_base
        result["dia"] = data_base.dt.normalize()

    # ---------- Específico: laudos_realizados ----------
    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            col_name = mapping.get(field)
            if col_name and col_name in result.columns:
                result[f"dh{field}"] = pd.to_datetime(result[col_name], errors="coerce", dayfirst=True)
        if "dhemissao" in result.columns:
            base_date = result.get("dhatendimento") if "dhatendimento" in result.columns else result.get("dhsolicitacao")
            if base_date is not None:
                result["tme_dias"]  = (result["dhemissao"] - base_date).dt.days
                result["sla_30_ok"] = result["tme_dias"] <= 30
                result["sla_60_ok"] = result["tme_dias"] <= 60

    # ---------- Limpeza texto ----------
    for col in ["diretoria", "superintendencia", "unidade", "tipo", "id", "perito", "anomês"]:
        if col in result.columns:
            result[col] = (
                result[col].astype(str).str.strip().str.title()
                .replace({"Nan": None, "": None, "None": None})
            )

    return result

# Padroniza todos os DataFrames
standardized_dfs = {}
processing_info = []
for name, df in raw_dataframes.items():
    standardized_df = standardize_dataframe(name, df)
    standardized_dfs[name] = standardized_df

    processing_info.append({
        "Arquivo": name,
        "Linhas": len(standardized_df),
        "Período": f"{standardized_df['anomês'].min()} a {standardized_df['anomês'].max()}"
                   if "anomês" in standardized_df.columns and not standardized_df["anomês"].isna().all()
                   else "Sem dados temporais"
    })

# Exibe informações de processamento
with st.sidebar.expander("📊 Resumo dos Dados", expanded=False):
    info_df = pd.DataFrame(processing_info)
    st.dataframe(info_df, use_container_width=True)

# ============ FILTROS ============
def extract_filter_values(column: str) -> List[str]:
    values = set()
    for df in standardized_dfs.values():
        if column in df.columns:
            unique_vals = df[column].dropna().astype(str).unique()
            values.update(v for v in unique_vals if v and v.lower() != "nan")
    return sorted(list(values))

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
    filtered = df.copy()

    for column, filter_values in [("diretoria", filter_diretoria),
                                  ("superintendencia", filter_superintendencia),
                                  ("unidade", filter_unidade),
                                  ("tipo", filter_tipo)]:
        if column in filtered.columns and filter_values:
            filtered = filtered[filtered[column].astype(str).isin(filter_values)]

    if "anomês_dt" in filtered.columns and filter_periodo != "Todo o período":
        max_date = filtered["anomês_dt"].max()
        if pd.notna(max_date):
            if filter_periodo == "Últimos 3 meses":
                cutoff_date = max_date - pd.DateOffset(months=3)
            elif filter_periodo == "Últimos 6 meses":
                cutoff_date = max_date - pd.DateOffset(months=6)
            elif filter_periodo == "Ano atual":
                cutoff_date = pd.Timestamp(max_date.year, 1, 1)
            filtered = filtered[filtered["anomês_dt"] >= cutoff_date]

    return filtered

filtered_dfs = {name: apply_filters(df) for name, df in standardized_dfs.items()}

# Extrai DataFrames filtrados para uso
df_atend_todos = filtered_dfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dfs.get("Laudos_todos_Mensal")
df_atend_esp   = filtered_dfs.get("Atendimentos_especifico_Mensal")
df_laudos_esp  = filtered_dfs.get("Laudos_especifico_Mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("detalhes_laudospendentes")
df_pend_exames = filtered_dfs.get("detalhes_examespendentes")
df_laudos_di   = filtered_dfs.get("laudos_Diário")
df_atend_di    = filtered_dfs.get("atendimentos_Diário")

# ============ CÁLCULO DE KPIs ============
def calculate_total(df: pd.DataFrame) -> int:
    if df is None or df.empty or "quantidade" not in df.columns:
        return 0
    return int(pd.to_numeric(df["quantidade"], errors="coerce").fillna(0).sum())

def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    monthly_totals = df.groupby("anomês_dt")["quantidade"].sum()
    return monthly_totals.mean() if len(monthly_totals) > 0 else None

def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    monthly_data = (
        df.groupby("anomês_dt")["quantidade"]
        .sum()
        .sort_index()
        .tail(periods * 2)
    )
    if len(monthly_data) < 2:
        return None
    mid_point = len(monthly_data) // 2
    first_half = monthly_data.iloc[:mid_point].mean()
    second_half = monthly_data.iloc[mid_point:].mean()
    if first_half > 0:
        return ((second_half - first_half) / first_half) * 100
    return None

def calculate_productivity_metrics(df_atend: pd.DataFrame, df_laudos: pd.DataFrame) -> Dict:
    metrics = {}
    if df_atend is not None and df_laudos is not None:
        total_atend = calculate_total(df_atend)
        total_laudos = calculate_total(df_laudos)
        if total_atend > 0:
            metrics["taxa_conversao"] = (total_laudos / total_atend) * 100
        if ("anomês_dt" in df_atend.columns and "anomês_dt" in df_laudos.columns):
            atend_monthly = df_atend.groupby("anomês_dt")["quantidade"].sum()
            laudos_monthly = df_laudos.groupby("anomês_dt")["quantidade"].sum()
            common_months = atend_monthly.index.intersection(laudos_monthly.index)
            if len(common_months) > 3:
                correlation = atend_monthly.loc[common_months].corr(laudos_monthly.loc[common_months])
                metrics["correlacao_atend_laudos"] = correlation
    return metrics

total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None and not df_pend_laudos.empty else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None and not df_pend_exames.empty else 0

media_mensal_laudos = calculate_monthly_average(df_laudos_todos)
backlog_meses = (total_pend_laudos / media_mensal_laudos) if media_mensal_laudos and media_mensal_laudos > 0 else None

produtividade_metrics = calculate_productivity_metrics(df_atend_todos, df_laudos_todos)
taxa_atendimento = produtividade_metrics.get("taxa_conversao")
correlacao_atend_laudos = produtividade_metrics.get("correlacao_atend_laudos")

crescimento_laudos = calculate_growth_rate(df_laudos_todos)
crescimento_atendimentos = calculate_growth_rate(df_atend_todos)

# KPIs de performance (laudos realizados)
tme_mediano = None
tme_medio = None
sla_30_percent = None
sla_60_percent = None

if df_laudos_real is not None and not df_laudos_real.empty:
    if "tme_dias" in df_laudos_real.columns:
        tme_values = pd.to_numeric(df_laudos_real["tme_dias"], errors="coerce").dropna()
        if not tme_values.empty:
            tme_mediano = tme_values.median()
            tme_medio = tme_values.mean()
    if "sla_30_ok" in df_laudos_real.columns:
        sla_30_percent = df_laudos_real["sla_30_ok"].mean() * 100
    if "sla_60_ok" in df_laudos_real.columns:
        sla_60_percent = df_laudos_real["sla_60_ok"].mean() * 100

# KPIs de aging (pendências)
aging_laudos_medio = None
aging_exames_medio = None

if df_pend_laudos is not None and not df_pend_laudos.empty and "data_base" in df_pend_laudos.columns:
    dates = pd.to_datetime(df_pend_laudos["data_base"], errors="coerce")
    if dates.notna().any():
        hoje = pd.Timestamp.now().normalize()
        dias_pendentes = (hoje - dates).dt.days
        aging_laudos_medio = dias_pendentes.mean()

if df_pend_exames is not None and not df_pend_exames.empty and "data_base" in df_pend_exames.columns:
    dates = pd.to_datetime(df_pend_exames["data_base"], errors="coerce")
    if dates.notna().any():
        hoje = pd.Timestamp.now().normalize()
        dias_pendentes = (hoje - dates).dt.days
        aging_exames_medio = dias_pendentes.mean()

# ============ EXIBIÇÃO DE KPIS ============
st.subheader("📈 Indicadores Principais")
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_atend = f"+{format_number(crescimento_atendimentos, 1)}%" if crescimento_atendimentos else None
    st.metric("Atendimentos Totais", format_number(total_atendimentos), delta=delta_atend)

with col2:
    delta_laudos = f"+{format_number(crescimento_laudos, 1)}%" if crescimento_laudos else None
    st.metric("Laudos Emitidos", format_number(total_laudos), delta=delta_laudos)

with col3:
    st.metric("Taxa de Conversão", f"{format_number(taxa_atendimento, 1)}%" if taxa_atendimento else "—")

with col4:
    st.metric("Produtividade Mensal", format_number(media_mensal_laudos, 1) if media_mensal_laudos else "—")

st.markdown("#### ⏰ Gestão de Pendências")
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Laudos Pendentes", format_number(total_pend_laudos))

with col6:
    st.metric("Exames Pendentes", format_number(total_pend_exames))

with col7:
    st.metric("Backlog (meses)", format_number(backlog_meses, 1) if backlog_meses else "—")

with col8:
    aging_medio = aging_laudos_medio or aging_exames_medio
    st.metric("Aging Médio (dias)", format_number(aging_medio, 0) if aging_medio else "—")

if tme_mediano is not None or sla_30_percent is not None:
    st.markdown("#### 🎯 Indicadores de Performance")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric("TME Mediano (dias)", format_number(tme_mediano, 1) if tme_mediano else "—")
    with col10:
        st.metric("TME Médio (dias)", format_number(tme_medio, 1) if tme_medio else "—")
    with col11:
        st.metric("SLA 30 dias", f"{format_number(sla_30_percent, 1)}%" if sla_30_percent else "—")
    with col12:
        st.metric("SLA 60 dias", f"{format_number(sla_60_percent, 1)}%" if sla_60_percent else "—")

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
    for alert in alerts:
        st.markdown(alert)
else:
    st.success("✅ **Indicadores saudáveis**: Todos os KPIs estão dentro dos parâmetros esperados")
st.markdown("---")

# ============ ABAS DO DASHBOARD ============
tab1, tabD, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "📆 Diário",
    "📈 Tendências",
    "🏆 Rankings",
    "⏰ Pendências",
    "📋 Dados",
    "📑 Relatórios"
])

# ============ ABA 1: VISÃO GERAL ============
with tab1:
    st.subheader("📊 Resumo Executivo")

    if df_laudos_todos is not None and not df_laudos_todos.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🏢 Performance por Unidade")
            if "unidade" in df_laudos_todos.columns:
                unidade_summary = (
                    df_laudos_todos.groupby("unidade", as_index=False)["quantidade"]
                    .sum()
                    .sort_values("quantidade", ascending=False)
                    .head(15)
                )
                fig_unidades = px.bar(
                    unidade_summary,
                    x="quantidade",
                    y="unidade",
                    orientation="h",
                    title="Top 15 Unidades - Laudos Emitidos",
                    color="quantidade",
                    color_continuous_scale="Blues",
                    text="quantidade"
                )
                fig_unidades.update_traces(texttemplate="%{text}", textposition="outside")
                fig_unidades.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_unidades, use_container_width=True)

        with col_right:
            st.markdown("#### 🔍 Distribuição por Tipo")
            if "tipo" in df_laudos_todos.columns:
                tipo_summary = (
                    df_laudos_todos.groupby("tipo", as_index=False)["quantidade"]
                    .sum()
                    .sort_values("quantidade", ascending=False)
                    .head(10)
                )
                fig_tipos = px.pie(
                    tipo_summary,
                    values="quantidade",
                    names="tipo",
                    title="Top 10 Tipos de Perícia"
                )
                fig_tipos.update_traces(textposition="inside", textinfo="percent+label")
                fig_tipos.update_layout(height=500)
                st.plotly_chart(fig_tipos, use_container_width=True)

    if (df_atend_todos is not None and df_laudos_todos is not None and
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):

        st.markdown("#### 📅 Evolução Temporal Consolidada")
        atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        atend_monthly["Tipo"] = "Atendimentos"
        atend_monthly.rename(columns={"quantidade": "Total"}, inplace=True)

        laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        laudos_monthly["Tipo"] = "Laudos"
        laudos_monthly.rename(columns={"quantidade": "Total"}, inplace=True)

        combined_data = pd.concat([atend_monthly, laudos_monthly])
        combined_data["Mês"] = combined_data["anomês_dt"].dt.strftime("%Y-%m")

        fig_temporal = px.line(
            combined_data,
            x="Mês",
            y="Total",
            color="Tipo",
            markers=True,
            title="Evolução Mensal: Atendimentos vs Laudos",
            line_shape="spline"
        )
        fig_temporal.update_layout(height=400, hovermode="x unified",
                                   xaxis_title="Período", yaxis_title="Quantidade")
        st.plotly_chart(fig_temporal, use_container_width=True)

        merged_monthly = pd.merge(
            atend_monthly.rename(columns={"Total": "Atendimentos"}),
            laudos_monthly.rename(columns={"Total": "Laudos"}),
            on="anomês_dt",
            how="inner"
        )
        if not merged_monthly.empty:
            merged_monthly["Taxa_Conversao"] = (merged_monthly["Laudos"] / merged_monthly["Atendimentos"] * 100)
            merged_monthly["Mês"] = merged_monthly["anomês_dt"].dt.strftime("%Y-%m")
            fig_conversao = px.line(
                merged_monthly,
                x="Mês",
                y="Taxa_Conversao",
                markers=True,
                title="Taxa de Conversão Mensal (%)",
                line_shape="spline"
            )
            fig_conversao.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Meta: 70%")
            fig_conversao.update_layout(height=350, yaxis_title="Taxa de Conversão (%)", xaxis_title="Período")
            st.plotly_chart(fig_conversao, use_container_width=True)

# ============ ABA 2: DIÁRIO ============
with tabD:
    st.subheader("📆 Análise Diária (Laudos & Atendimentos)")

    def daily_summary(df: Optional[pd.DataFrame], label: str) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        # garante coluna de dia
        if "dia" not in df.columns:
            if "data_base" in df.columns:
                df = df.copy()
                df["dia"] = pd.to_datetime(df["data_base"], errors="coerce", dayfirst=True).dt.normalize()
            else:
                # tenta criar a partir de qualquer coluna de data
                for c in DATE_CANDIDATES:
                    if c in df.columns:
                        df = df.copy()
                        df["dia"] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.normalize()
                        break
        if "dia" not in df.columns or df["dia"].isna().all():
            return None

        q = pd.to_numeric(df.get("quantidade", 1), errors="coerce").fillna(1)
        base = df.assign(_q=q).groupby("dia", as_index=False)["_q"].sum().rename(columns={"_q": label})
        base = base.sort_values("dia")
        return base

    daily_laudos = daily_summary(df_laudos_di, "Laudos")
    daily_atends = daily_summary(df_atend_di, "Atendimentos")

    if daily_laudos is None and daily_atends is None:
        st.info("Não há dados diários disponíveis nos arquivos enviados.")
    else:
        # Merge das duas séries
        if daily_laudos is not None and daily_atends is not None:
            daily = pd.merge(daily_atends, daily_laudos, on="dia", how="outer").sort_values("dia")
        elif daily_laudos is not None:
            daily = daily_laudos.copy()
            daily["Atendimentos"] = 0
        else:
            daily = daily_atends.copy()
            daily["Laudos"] = 0

        daily[["Atendimentos", "Laudos"]] = daily[["Atendimentos", "Laudos"]].fillna(0).astype(int)
        daily["Diferença (Laudos-Atends)"] = daily["Laudos"] - daily["Atendimentos"]

        # KPIs (últimos 30 dias)
        cutoff = (daily["dia"].max() - pd.Timedelta(days=29)) if not daily["dia"].isna().all() else None
        last30 = daily[daily["dia"] >= cutoff] if cutoff is not None else daily.copy()

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Laudos (30d)", format_number(last30["Laudos"].sum()))
        with colB:
            st.metric("Atendimentos (30d)", format_number(last30["Atendimentos"].sum()))
        with colC:
            st.metric("Média diária Laudos (30d)", format_number(last30["Laudos"].mean(), 1) if not last30.empty else "—")
        with colD:
            st.metric("Média diária Atend. (30d)", format_number(last30["Atendimentos"].mean(), 1) if not last30.empty else "—")

        # Série temporal com MM7
        daily_plot = daily.copy()
        daily_plot["MM7 Laudos"] = daily_plot["Laudos"].rolling(7).mean()
        daily_plot["MM7 Atendimentos"] = daily_plot["Atendimentos"].rolling(7).mean()
        daily_plot["Dia"] = daily_plot["dia"].dt.strftime("%Y-%m-%d")

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(x=daily_plot["Dia"], y=daily_plot["Atendimentos"], mode="lines", name="Atendimentos"))
        fig_daily.add_trace(go.Scatter(x=daily_plot["Dia"], y=daily_plot["Laudos"], mode="lines", name="Laudos"))
        fig_daily.add_trace(go.Scatter(x=daily_plot["Dia"], y=daily_plot["MM7 Atendimentos"], mode="lines", name="MM7 Atendimentos", line=dict(dash="dash")))
        fig_daily.add_trace(go.Scatter(x=daily_plot["Dia"], y=daily_plot["MM7 Laudos"], mode="lines", name="MM7 Laudos", line=dict(dash="dash")))
        fig_daily.update_layout(title="Série Diária (com Média Móvel 7 dias)", height=420, hovermode="x unified",
                                xaxis_title="Dia", yaxis_title="Quantidade")
        st.plotly_chart(fig_daily, use_container_width=True)

        # Tabela consolidada (últimos 90 dias)
        st.markdown("#### 📋 Tabela: Laudos e Atendimentos por dia")
        last_n = st.slider("Dias a exibir", min_value=30, max_value=365, value=90, step=30)
        cutoff_table = daily["dia"].max() - pd.Timedelta(days=last_n-1) if not daily["dia"].isna().all() else None
        table_df = daily[daily["dia"] >= cutoff_table].copy() if cutoff_table is not None else daily.copy()
        table_df = table_df.sort_values("dia", ascending=False)
        table_df["dia"] = table_df["dia"].dt.strftime("%d/%m/%Y")
        st.dataframe(table_df, use_container_width=True, height=360)

        csv_daily = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar tabela diária (CSV)", data=csv_daily,
                           file_name=f"diario_laudos_atendimentos_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        # Quebra por dia da semana
        st.markdown("#### 🗓️ Distribuição por Dia da Semana (Agregado)")
        agg = daily.copy()
        agg["dow"] = agg["dia"].dt.dayofweek  # 0=Seg
        map_dow = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
        dist = agg.groupby("dow")[["Atendimentos", "Laudos"]].mean().round(2).reset_index()
        dist["Dia"] = dist["dow"].map(map_dow)
        dist = dist.sort_values("dow")
        fig_dow = px.bar(dist, x="Dia", y=["Atendimentos", "Laudos"], barmode="group",
                         title="Média diária por dia da semana")
        fig_dow.update_layout(height=360, xaxis_title="", yaxis_title="Média diária")
        st.plotly_chart(fig_dow, use_container_width=True)

# ============ ABA 3: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise de Tendências")

    def create_enhanced_time_series(df: pd.DataFrame, title: str, color: str = "blue") -> None:
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        monthly_data = (
            df.groupby("anomês_dt", as_index=False)["quantidade"]
            .sum()
            .sort_values("anomês_dt")
        )
        if monthly_data.empty:
            st.info(f"Sem dados temporais para {title}")
            return

        monthly_data["Mês"] = monthly_data["anomês_dt"].dt.strftime("%Y-%m")
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(title, "Variação Percentual Mensal"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(
            go.Scatter(x=monthly_data["Mês"], y=monthly_data["quantidade"], mode="lines+markers",
                       name="Valores", line=dict(color=color, width=2)),
            row=1, col=1
        )
        if len(monthly_data) >= 3:
            monthly_data["media_movel"] = monthly_data["quantidade"].rolling(window=3, center=True).mean()
            fig.add_trace(
                go.Scatter(x=monthly_data["Mês"], y=monthly_data["media_movel"], mode="lines",
                           name="Média Móvel (3m)", line=dict(dash="dash", color="red", width=2)),
                row=1, col=1
            )
        monthly_data["variacao_pct"] = monthly_data["quantidade"].pct_change() * 100
        colors = ["red" if x < 0 else "green" for x in monthly_data["variacao_pct"].fillna(0)]
        fig.add_trace(
            go.Bar(x=monthly_data["Mês"], y=monthly_data["variacao_pct"], name="Variação %", marker_color=colors, showlegend=False),
            row=2, col=1
        )
        fig.update_layout(height=600, hovermode="x unified", showlegend=True)
        fig.update_xaxes(title_text="Período", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        create_enhanced_time_series(df_atend_todos, "🏥 Atendimentos - Análise Temporal", "blue")
        if df_atend_todos is not None and "anomês_dt" in df_atend_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Atendimentos")
            seasonal_data = df_atend_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            monthly_totals = (
                seasonal_data.groupby(["mes_num", "mes_nome"])["quantidade"].sum().reset_index().sort_values("mes_num")
            )
            fig_sazonal = px.bar(monthly_totals, x="mes_nome", y="quantidade",
                                 title="Distribuição Sazonal", color="quantidade", color_continuous_scale="Blues")
            fig_sazonal.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sazonal, use_container_width=True)

    with col2:
        create_enhanced_time_series(df_laudos_todos, "📄 Laudos - Análise Temporal", "green")
        if df_laudos_todos is not None and "anomês_dt" in df_laudos_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Laudos")
            seasonal_data = df_laudos_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            monthly_totals = (
                seasonal_data.groupby(["mes_num", "mes_nome"])["quantidade"].sum().reset_index().sort_values("mes_num")
            )
            fig_sazonal = px.bar(monthly_totals, x="mes_nome", y="quantidade",
                                 title="Distribuição Sazonal", color="quantidade", color_continuous_scale="Greens")
            fig_sazonal.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sazonal, use_container_width=True)

    if (df_atend_todos is not None and df_laudos_todos is not None and
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        st.markdown("#### 🔗 Análise de Correlação")
        atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum()
        laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum()
        common_periods = atend_monthly.index.intersection(laudos_monthly.index)
        if len(common_periods) > 3:
            correlation_data = pd.DataFrame({
                "anomês_dt": common_periods,
                "Atendimentos": atend_monthly.loc[common_periods].values,
                "Laudos": laudos_monthly.loc[common_periods].values
            })
            correlation_data["Período"] = correlation_data["anomês_dt"].dt.strftime("%Y-%m")
            fig_scatter = px.scatter(
                correlation_data, x="Atendimentos", y="Laudos", hover_data=["Período"],
                title="Correlação: Atendimentos vs Laudos", trendline="ols"
            )
            correlation_coef = correlation_data["Atendimentos"].corr(correlation_data["Laudos"])
            fig_scatter.add_annotation(text=f"Correlação: {correlation_coef:.3f}",
                                       xref="paper", yref="paper", x=0.02, y=0.98,
                                       showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ============ ABA 4: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Comparativos")

    def create_enhanced_ranking(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20) -> None:
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        ranking_data = df.groupby(dimension).agg({"quantidade": ["sum", "count", "mean"]}).round(2)
        ranking_data.columns = ["Total", "Registros", "Média"]
        ranking_data = ranking_data.sort_values("Total", ascending=False).head(top_n).reset_index()
        if ranking_data.empty:
            st.info(f"Sem dados para {title}")
            return
        fig = px.bar(ranking_data, x="Total", y=dimension, orientation="h",
                     title=title, color="Total", color_continuous_scale="Viridis",
                     hover_data=["Registros", "Média"])
        fig.update_layout(height=max(400, len(ranking_data) * 30), showlegend=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"📊 Detalhes - {title}"):
            st.dataframe(ranking_data, use_container_width=True)

    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs(["Por Diretoria", "Por Unidade", "Por Tipo", "Comparativo"])

    with rank_tab1:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_todos, "diretoria", "🏥 Atendimentos por Diretoria")
        with col2:
            create_enhanced_ranking(df_laudos_todos, "diretoria", "📄 Laudos por Diretoria")

    with rank_tab2:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_todos, "unidade", "🏥 Atendimentos por Unidade", 25)
        with col2:
            create_enhanced_ranking(df_laudos_todos, "unidade", "📄 Laudos por Unidade", 25)

    with rank_tab3:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_esp, "tipo", "🏥 Atendimentos por Tipo", 20)
        with col2:
            create_enhanced_ranking(df_laudos_esp, "tipo", "📄 Laudos por Tipo", 20)

    with rank_tab4:
        st.markdown("#### 📊 Análise Comparativa de Eficiência")
        if (df_atend_todos is not None and df_laudos_todos is not None and
            "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns):
            atend_por_unidade = df_atend_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Atendimentos"})
            laudos_por_unidade = df_laudos_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Laudos"})
            eficiencia_data = pd.merge(atend_por_unidade, laudos_por_unidade, on="unidade", how="inner")
            if not eficiencia_data.empty:
                eficiencia_data["Taxa_Conversao"] = (eficiencia_data["Laudos"] / eficiencia_data["Atendimentos"] * 100)
                eficiencia_data = eficiencia_data.sort_values("Taxa_Conversao", ascending=False)
                fig_eficiencia = px.scatter(
                    eficiencia_data.head(20),
                    x="Atendimentos", y="Laudos", size="Taxa_Conversao",
                    hover_name="unidade",
                    title="Eficiência por Unidade (Atendimentos vs Laudos)",
                    color="Taxa_Conversao", color_continuous_scale="RdYlGn"
                )
                fig_eficiencia.update_layout(height=500)
                st.plotly_chart(fig_eficiencia, use_container_width=True)
                st.markdown("**🥇 Top 10 Unidades Mais Eficientes:**")
                top_eficientes = eficiencia_data.head(10)[["unidade", "Taxa_Conversao", "Atendimentos", "Laudos"]]
                st.dataframe(top_eficientes, use_container_width=True)

# ============ ABA 5: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Gestão de Pendências")

    def calculate_aging_analysis(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype="int64"), {}
        available_date_columns = [col for col in df.columns if "data" in col.lower()]
        if date_column not in df.columns and available_date_columns:
            date_column = available_date_columns[0]
        if date_column not in df.columns:
            return df, pd.Series(dtype="int64"), {}

        result = df.copy()
        dates = pd.to_datetime(result[date_column], errors="coerce")
        if dates.isna().all():
            return df, pd.Series(dtype="int64"), {}

        hoje = pd.Timestamp.now().normalize()
        dias_pendentes = (hoje - dates).dt.days
        faixas_aging = pd.cut(
            dias_pendentes,
            bins=[-1, 15, 30, 60, 90, 180, 365, float("inf")],
            labels=["0-15 dias", "16-30 dias", "31-60 dias", "61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"]
        )
        result["dias_pendentes"] = dias_pendentes
        result["faixa_aging"] = faixas_aging
        result["prioridade"] = pd.cut(
            dias_pendentes,
            bins=[-1, 30, 90, 180, float("inf")],
            labels=["Normal", "Atenção", "Urgente", "Crítico"]
        )
        distribuicao = faixas_aging.value_counts().sort_index()
        stats = {
            "total": len(result),
            "media_dias": dias_pendentes.mean(),
            "mediana_dias": dias_pendentes.median(),
            "max_dias": dias_pendentes.max(),
            "criticos": len(result[result["prioridade"] == "Crítico"]),
            "urgentes": len(result[result["prioridade"] == "Urgente"])
        }
        return result, distribuicao, stats

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Laudos Pendentes")
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            laudos_aged, dist_laudos, stats_laudos = calculate_aging_analysis(df_pend_laudos)
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("Total", format_number(stats_laudos.get("total", 0)))
            with col_b: st.metric("Críticos", stats_laudos.get("criticos", 0))
            with col_c: st.metric("Média (dias)", format_number(stats_laudos.get("media_dias", 0), 1))
            if not dist_laudos.empty:
                fig_aging_laudos = px.bar(x=dist_laudos.index, y=dist_laudos.values,
                                          title="Distribuição por Tempo de Pendência",
                                          color=dist_laudos.values, color_continuous_scale="Reds", text=dist_laudos.values)
                fig_aging_laudos.update_traces(texttemplate="%{text}", textposition="outside")
                fig_aging_laudos.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(fig_aging_laudos, use_container_width=True)
            if "prioridade" in laudos_aged.columns:
                prioridade_dist = laudos_aged["prioridade"].value_counts()
                fig_prioridade = px.pie(values=prioridade_dist.values, names=prioridade_dist.index,
                                        title="Distribuição por Prioridade",
                                        color_discrete_map={"Normal": "green", "Atenção": "yellow", "Urgente": "orange", "Crítico": "red"})
                fig_prioridade.update_layout(height=300)
                st.plotly_chart(fig_prioridade, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in laudos_aged.columns:
                cols_show = [c for c in ["id", "unidade", "tipo", "dias_pendentes", "prioridade"] if c in laudos_aged.columns]
                oldest = laudos_aged.nlargest(10, "dias_pendentes")[cols_show] if cols_show else laudos_aged.nlargest(10, "dias_pendentes")
                st.dataframe(oldest, use_container_width=True, height=250)
        else:
            st.info("Sem dados de laudos pendentes disponíveis.")

    with col2:
        st.markdown("#### 🔬 Exames Pendentes")
        if df_pend_exames is not None and not df_pend_exames.empty:
            exames_aged, dist_exames, stats_exames = calculate_aging_analysis(df_pend_exames)
            col_a, col_b, col_c = st.columns(3)
            with col_a: st.metric("Total", format_number(stats_exames.get("total", 0)))
            with col_b: st.metric("Críticos", stats_exames.get("criticos", 0))
            with col_c: st.metric("Média (dias)", format_number(stats_exames.get("media_dias", 0), 1))
            if not dist_exames.empty:
                fig_aging_exames = px.bar(x=dist_exames.index, y=dist_exames.values,
                                          title="Distribuição por Tempo de Pendência",
                                          color=dist_exames.values, color_continuous_scale="Oranges", text=dist_exames.values)
                fig_aging_exames.update_traces(texttemplate="%{text}", textposition="outside")
                fig_aging_exames.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(fig_aging_exames, use_container_width=True)
            if "prioridade" in exames_aged.columns:
                prioridade_dist = exames_aged["prioridade"].value_counts()
                fig_prioridade = px.pie(values=prioridade_dist.values, names=prioridade_dist.index,
                                        title="Distribuição por Prioridade",
                                        color_discrete_map={"Normal": "green", "Atenção": "yellow", "Urgente": "orange", "Crítico": "red"})
                fig_prioridade.update_layout(height=300)
                st.plotly_chart(fig_prioridade, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in exames_aged.columns:
                cols_show = [c for c in ["id", "unidade", "tipo", "dias_pendentes", "prioridade"] if c in exames_aged.columns]
                oldest = exames_aged.nlargest(10, "dias_pendentes")[cols_show] if cols_show else exames_aged.nlargest(10, "dias_pendentes")
                st.dataframe(oldest, use_container_width=True, height=250)
        else:
            st.info("Sem dados de exames pendentes disponíveis.")

    st.markdown("#### 🏢 Análise de Pendências por Unidade")
    pendencias_por_unidade = []
    if df_pend_laudos is not None and "unidade" in df_pend_laudos.columns:
        laudos_unidade = df_pend_laudos.groupby("unidade").size().reset_index(name="Laudos_Pendentes")
        pendencias_por_unidade.append(laudos_unidade)
    if df_pend_exames is not None and "unidade" in df_pend_exames.columns:
        exames_unidade = df_pend_exames.groupby("unidade").size().reset_index(name="Exames_Pendentes")
        pendencias_por_unidade.append(exames_unidade)

    if pendencias_por_unidade:
        from functools import reduce
        pendencias_consolidadas = reduce(lambda l, r: pd.merge(l, r, on="unidade", how="outer"),
                                         pendencias_por_unidade).fillna(0)
        pendencias_consolidadas["Total_Pendencias"] = pendencias_consolidadas.get("Laudos_Pendentes", 0) + pendencias_consolidadas.get("Exames_Pendentes", 0)
        pendencias_consolidadas = pendencias_consolidadas.sort_values("Total_Pendencias", ascending=False)
        fig_pendencias = go.Figure()
        if "Laudos_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(name="Laudos Pendentes",
                                            y=pendencias_consolidadas["unidade"].head(15),
                                            x=pendencias_consolidadas["Laudos_Pendentes"].head(15),
                                            orientation="h", marker_color="lightcoral"))
        if "Exames_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(name="Exames Pendentes",
                                            y=pendencias_consolidadas["unidade"].head(15),
                                            x=pendencias_consolidadas["Exames_Pendentes"].head(15),
                                            orientation="h", marker_color="lightsalmon"))
        fig_pendencias.update_layout(title="Top 15 Unidades com Mais Pendências",
                                     barmode="stack", height=500,
                                     xaxis_title="Quantidade de Pendências",
                                     yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_pendencias, use_container_width=True)
        st.markdown("**📊 Detalhamento por Unidade:**")
        st.dataframe(pendencias_consolidadas.head(20), use_container_width=True, height=300)

# ============ ABA 6: DADOS ============
with tab5:
    st.subheader("📋 Exploração dos Dados")

    st.markdown("#### 📊 Resumo dos Datasets Carregados")
    data_summary = []
    for name, df in standardized_dfs.items():
        if df is not None and not df.empty:
            periodo_info = "Sem dados temporais"
            if "anomês" in df.columns and not df["anomês"].isna().all():
                min_periodo = df["anomês"].min()
                max_periodo = df["anomês"].max()
                periodo_info = f"{min_periodo} a {max_periodo}"
            data_summary.append({
                "Dataset": name.replace("_", " ").title(),
                "Registros": f"{len(df):,}".replace(",", "."),
                "Colunas": len(df.columns),
                "Período": periodo_info,
                "Tamanho (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "Status": "✅ Carregado"
            })

    if data_summary:
        summary_df = pd.DataFrame(data_summary)
        st.dataframe(summary_df, use_container_width=True)
        total_registros = sum(int(row["Registros"].replace(".", "")) for row in data_summary)
        total_tamanho = sum(row["Tamanho (MB)"] for row in data_summary)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total de Registros", f"{total_registros:,}".replace(",", "."))
        with col2: st.metric("Datasets Carregados", len(data_summary))
        with col3: st.metric("Tamanho Total (MB)", f"{total_tamanho:.1f}")
        with col4:
            avg_size = total_tamanho / len(data_summary) if data_summary else 0
            st.metric("Tamanho Médio (MB)", f"{avg_size:.1f}")

    st.markdown("#### 🔍 Exploração Detalhada")
    available_datasets = [name for name, df in standardized_dfs.items() if df is not None]
    if available_datasets:
        selected_dataset = st.selectbox(
            "Selecione o dataset para explorar:",
            available_datasets,
            format_func=lambda x: x.replace("_", " ").title()
        )
        if selected_dataset:
            df_selected = standardized_dfs[selected_dataset]
            st.markdown(f"#### 📄 {selected_dataset.replace('_', ' ').title()}")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Registros", f"{len(df_selected):,}".replace(",", "."))
            with col2: st.metric("Colunas", len(df_selected.columns))
            with col3:
                valores_nulos = df_selected.isnull().sum().sum()
                st.metric("Valores Nulos", f"{valores_nulos:,}".replace(",", "."))
            with col4:
                if "anomês_dt" in df_selected.columns:
                    unique_months = df_selected["anomês_dt"].nunique()
                    st.metric("Meses Únicos", unique_months)
                else:
                    st.metric("Período", "N/A")

            with st.expander("🔍 Análise de Qualidade dos Dados", expanded=False):
                quality_info = []
                for col in df_selected.columns:
                    dtype = str(df_selected[col].dtype)
                    null_count = df_selected[col].isnull().sum()
                    null_percent = (null_count / len(df_selected)) * 100
                    unique_count = df_selected[col].nunique()
                    if null_percent == 0: quality = "🟢 Excelente"
                    elif null_percent < 5: quality = "🟡 Boa"
                    elif null_percent < 20: quality = "🟠 Regular"
                    else: quality = "🔴 Ruim"
                    quality_info.append({
                        "Coluna": col, "Tipo": dtype, "Nulos": f"{null_count:,}".replace(",", "."),
                        "% Nulos": f"{null_percent:.1f}%", "Únicos": f"{unique_count:,}".replace(",", "."),
                        "Qualidade": quality
                    })
                quality_df = pd.DataFrame(quality_info)
                st.dataframe(quality_df, use_container_width=True)

            st.markdown("**🎛️ Controles de Visualização:**")
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            with viz_col1:
                max_rows = st.number_input("Máximo de linhas:", min_value=10, max_value=5000, value=500, step=50)
            with viz_col2:
                if "anomês" in df_selected.columns:
                    available_months = sorted(df_selected["anomês"].dropna().unique(), reverse=True)
                    selected_months = st.multiselect("Filtrar por período:", available_months,
                                                     default=available_months[:6] if len(available_months) > 6 else available_months)
                else:
                    selected_months = []
            with viz_col3:
                all_columns = list(df_selected.columns)
                selected_columns = st.multiselect("Colunas a exibir:", all_columns,
                                                  default=all_columns[:10] if len(all_columns) > 10 else all_columns)

            df_display = df_selected.copy()
            if selected_months and "anomês" in df_display.columns:
                df_display = df_display[df_display["anomês"].isin(selected_months)]
            if selected_columns:
                df_display = df_display[selected_columns]
            df_display = df_display.head(max_rows)

            if not df_display.empty:
                st.markdown("**📈 Estatísticas Descritivas:**")
                numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats = df_display[numeric_cols].describe().round(2)
                    st.dataframe(stats, use_container_width=True)
                else:
                    st.info("Nenhuma coluna numérica encontrada para estatísticas.")

            st.markdown(f"**📋 Dados Filtrados ({len(df_display):,} de {len(df_selected):,} registros):**".replace(",", "."))
            st.dataframe(df_display, use_container_width=True, height=400)

            col_down1, col_down2 = st.columns(2)
            with col_down1:
                csv_data = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Dados Filtrados (CSV)",
                    data=csv_data,
                    file_name=f"{selected_dataset}_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_down2:
                csv_complete = df_selected.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Dataset Completo (CSV)",
                    data=csv_complete,
                    file_name=f"{selected_dataset}_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ============ ABA 7: RELATÓRIOS ============
with tab6:
    st.subheader("📑 Relatórios Executivos")

    tipo_relatorio = st.selectbox(
        "Tipo de Relatório:",
        ["Relatório Executivo Completo", "Relatório de Produção", "Relatório de Pendências", "Relatório de Performance", "Relatório Comparativo"]
    )

    def gerar_relatorio_executivo() -> str:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        relatorio = f"""
# RELATÓRIO EXECUTIVO PCI/SC
**Data de Geração:** {timestamp}
**Período de Análise:** {filter_periodo}

## 📊 RESUMO EXECUTIVO

### Indicadores Principais
- **Atendimentos Totais:** {format_number(total_atendimentos)}
- **Laudos Emitidos:** {format_number(total_laudos)}
- **Taxa de Conversão:** {format_number(taxa_atendimento, 1) if taxa_atendimento else 'N/A'}%
- **Produtividade Mensal:** {format_number(media_mensal_laudos, 1) if media_mensal_laudos else 'N/A'} laudos/mês

### Gestão de Pendências
- **Laudos Pendentes:** {format_number(total_pend_laudos)}
- **Exames Pendentes:** {format_number(total_pend_exames)}
- **Backlog Estimado:** {format_number(backlog_meses, 1) if backlog_meses else 'N/A'} meses
- **Aging Médio:** {format_number(aging_laudos_medio or aging_exames_medio, 0) if (aging_laudos_medio or aging_exames_medio) else 'N/A'} dias

### Performance Operacional
- **TME Mediano:** {format_number(tme_mediano, 1) if tme_mediano else 'N/A'} dias
- **SLA 30 dias:** {format_number(sla_30_percent, 1) if sla_30_percent else 'N/A'}%
- **SLA 60 dias:** {format_number(sla_60_percent, 1) if sla_60_percent else 'N/A'}%

## 📈 ANÁLISE DE TENDÊNCIAS
"""
        if crescimento_laudos is not None:
            if crescimento_laudos > 5:
                relatorio += f"- **Crescimento Positivo:** Laudos cresceram {format_number(crescimento_laudos, 1)}% no período\n"
            elif crescimento_laudos < -5:
                relatorio += f"- **Alerta:** Laudos decresceram {format_number(abs(crescimento_laudos), 1)}% no período\n"
            else:
                relatorio += f"- **Estabilidade:** Variação de {format_number(crescimento_laudos, 1)}% nos laudos\n"

        relatorio += "\n## 🚨 ALERTAS E RECOMENDAÇÕES\n"
        alertas_relatorio = []
        if backlog_meses and backlog_meses > 6:
            alertas_relatorio.append("🔴 **CRÍTICO:** Backlog superior a 6 meses - necessário plano de ação imediato")
        elif backlog_meses and backlog_meses > 3:
            alertas_relatorio.append("🟡 **ATENÇÃO:** Backlog entre 3-6 meses - monitorar tendência")
        if sla_30_percent and sla_30_percent < 70:
            alertas_relatorio.append("🔴 **CRÍTICO:** SLA 30 dias abaixo de 70% - revisar processos")
        if taxa_atendimento and taxa_atendimento < 50:
            alertas_relatorio.append("🟡 **ATENÇÃO:** Taxa de conversão baixa - analisar gargalos")
        if alertas_relatorio:
            relatorio += "\n".join(alertas_relatorio)
        else:
            relatorio += "✅ **Situação Normal:** Todos os indicadores dentro dos parâmetros esperados"

        relatorio += "\n\n## 📋 DATASETS UTILIZADOS\n"
        for name, df in standardized_dfs.items():
            if df is not None and not df.empty:
                relatorio += f"- **{name.replace('_', ' ').title()}:** {len(df):,} registros\n"

        relatorio += "\n---\n*Relatório gerado automaticamente pelo Dashboard PCI/SC*\n*Sistema de Monitoramento de Produção e Pendências*\n"
        return relatorio.strip()

    if tipo_relatorio == "Relatório Executivo Completo":
        relatorio_texto = gerar_relatorio_executivo()
        st.markdown("#### 📄 Visualização do Relatório")
        st.markdown(relatorio_texto)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Relatório Executivo",
            data=relatorio_texto.encode("utf-8"),
            file_name=f"relatorio_executivo_pci_sc_{timestamp}.md",
            mime="text/markdown"
        )

    elif tipo_relatorio == "Relatório de Produção":
        st.markdown("#### 📊 Relatório de Produção")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Métricas de Produção:**")
            if df_laudos_todos is not None and "anomês" in df_laudos_todos.columns:
                prod_mensal = df_laudos_todos.groupby("anomês")["quantidade"].sum().reset_index().sort_values("anomês")
                st.line_chart(prod_mensal.set_index("anomês")["quantidade"], height=300)
        with col2:
            st.markdown("**Top Produtores:**")
            if df_laudos_todos is not None and "unidade" in df_laudos_todos.columns:
                top_unidades = df_laudos_todos.groupby("unidade")["quantidade"].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_unidades, height=300)
    else:
        st.info(f"Relatório '{tipo_relatorio}' em desenvolvimento.")

# ============ RODAPÉ ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.1</strong> - Sistema Avançado de Monitoramento</p>
    <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
    <p>Para suporte técnico ou sugestões: <strong>equipe-ti@pci.sc.gov.br</strong></p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)
