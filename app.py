import io
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="PCI/SC – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CONFIGURAÇÃO INICIAL ============
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = 400

CUSTOM_CSS = """
<style>
.kpi-card{
  background: linear-gradient(180deg,#ffffff,#f7f9fc);
  border:1px solid #e6eaf2; border-radius:12px; padding:14px 16px; height:100%;
  box-shadow: 0 1px 2px rgba(16,24,40,.06);
}
.kpi-title{font-size:13px;color:#667085;margin:0;}
.kpi-value{font-size:24px;font-weight:700;color:#0f172a;margin:2px 0 0 0;}
.kpi-delta{font-size:12px;color:#475467;margin-top:6px;}
.section-title{ margin:18px 0 8px 0; }
hr{ margin:6px 0 18px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Compat: segmented_control (fallback para radio)
def segment(label, options, default=None, key=None):
    try:
        return st.segmented_control(label, options, default=default, key=key)
    except Exception:
        idx = options.index(default) if (default in options) else 0
        return st.radio(label, options, index=idx, horizontal=True, key=key)

# === PRO UI: Header estilizado com badges ===
colh1, colh2 = st.columns([0.75, 0.25])
with colh1:
    st.markdown("<h2 style='margin-bottom:6px'>🏥 Dashboard PCI/SC – Produção & Pendências</h2>", unsafe_allow_html=True)
    st.caption("Monitoramento executivo • Produção mensal e diária • Pendências e SLA • Rankings e Tendências")
with colh2:
    st.markdown(f"""
    <div style="display:flex; gap:8px; justify-content:flex-end;">
      <div class="kpi-card" style="padding:8px 10px;"><span class="kpi-title">Versão</span><div class="kpi-value" style="font-size:16px;">2.2</div></div>
      <div class="kpi-card" style="padding:8px 10px;"><span class="kpi-title">Atualizado</span><div class="kpi-value" style="font-size:16px;">{datetime.now().strftime("%d/%m/%Y %H:%M")}</div></div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

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
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip('"').str.strip()
                    return df
            except Exception:
                continue

    # Fallback para detecção automática
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
    if series is None or len(series) == 0:
        return None

    dt_series = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)

    if dt_series.isna().sum() > len(dt_series) * 0.5:
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                dt_series = pd.to_datetime(series, format=fmt, errors="coerce")
                if dt_series.notna().sum() > len(dt_series) * 0.5:
                    break
            except Exception:
                continue

    return dt_series if dt_series.notna().any() else None

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

# ============ DETECÇÃO DE ARQUIVOS ============
@st.cache_data
def detect_data_sources():
    """Detecta se existem arquivos na pasta data/."""
    return os.path.exists("data") and any(p.endswith(".csv") for p in os.listdir("data"))

has_data_dir = detect_data_sources()

# ============ INTERFACE DE UPLOAD ============
st.sidebar.header("📁 Configuração de Dados")
if not has_data_dir:
    st.sidebar.info("💡 Envie os arquivos CSV disponíveis. O dashboard se adapta automaticamente.")

# Definição dos arquivos esperados
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
    "Atendimentos_diario": {
        "label": "Atendimentos (Diário)",
        "description": "Registros de atendimentos em granularidade diária",
        "pattern": ["atendimentos_diario", "atendimentos_diário", "atendimentos diário"]
    },
    "Laudos_diario": {
        "label": "Laudos (Diário)",
        "description": "Registros de laudos em granularidade diária",
        "pattern": ["laudos_diario", "laudos_diário", "laudos diário"]
    }
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
    """Resolve caminho do arquivo com tolerância a variações de nome."""
    if not os.path.exists("data"):
        return None

    config = file_configs.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    patterns.append(name.lower().replace(" ", "_"))

    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        base_name = os.path.splitext(filename)[0].lower()
        normalized_name = re.sub(r"[^\w]", "_", base_name)
        for pattern in patterns:
            if pattern in normalized_name or normalized_name.startswith(pattern):
                return os.path.join("data", filename)

    return None

# ============ DADOS SIMULADOS PARA DEMO ============
def create_sample_laudos_realizados() -> pd.DataFrame:
    """Cria dados simulados de laudos realizados baseados no screenshot."""
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

    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2024-12-31')

    np.random.seed(42)
    for i in range(500):
        solicitacao = start_date + pd.Timedelta(days=np.random.randint(0, (end_date - start_date).days))
        atendimento = solicitacao + pd.Timedelta(days=np.random.randint(1, 30))
        emissao = atendimento + pd.Timedelta(days=np.random.randint(1, 120))

        sample_data.append({
            'dhsolicitacao': solicitacao.strftime('%d/%m/%Y'),
            'dhatendimento': atendimento.strftime('%d/%m/%Y'),
            'dhemitido': emissao.strftime('%d/%m/%Y'),
            'n_laudo': f"L{2000 + i}",
            'ano_emissao': emissao.year,
            'mes_emissao': emissao.month,
            'unidade_emissao': np.random.choice(unidades),
            'diretoria': np.random.choice(diretorias),
            'txcompetencia': f"{emissao.year}-{emissao.month:02d}",
            'txtipopericia': np.random.choice(tipos_pericia),
            'perito': np.random.choice(peritos)
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
                    with open(file_path, 'rb') as f:
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
            df.columns = [col.strip() for col in df.columns]
            loaded_data[name] = df

    if "laudos_realizados" not in loaded_data:
        st.sidebar.info("📊 Usando dados simulados para Laudos Realizados (demo)")
        loaded_data["laudos_realizados"] = create_sample_laudos_realizados()

    return loaded_data

# Carrega os dados
with st.spinner("Carregando e padronizando dados..."):
    raw_dataframes = load_all_data(uploads)

if not raw_dataframes:
    st.warning("⚠️ Nenhum arquivo foi carregado. Por favor, envie os arquivos CSV pela barra lateral ou coloque-os na pasta `data/`.")
    st.info("📝 **Arquivos esperados:** " + ", ".join(file_configs.keys()))
    st.stop()

# ============ PADRONIZAÇÃO DE DADOS ============
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza estrutura do DataFrame para análise unificada - CORRIGIDO."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    
    # ===== CORREÇÃO PRINCIPAL: Tratar os valores como TOTAIS, não IDs =====
    
    # Para datasets mensais e diários, os valores nas colunas idatendimento/iddocumento
    # representam TOTAIS AGREGADOS, não IDs individuais
    if name in ["Atendimentos_todos_Mensal", "Laudos_todos_Mensal",
                "Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal",
                "Atendimentos_diario", "Laudos_diario"]:
        
        # Identificar a coluna de quantidade correta
        if "idatendimento" in result.columns:
            result["quantidade"] = pd.to_numeric(result["idatendimento"], errors="coerce").fillna(0)
        elif "iddocumento" in result.columns:
            result["quantidade"] = pd.to_numeric(result["iddocumento"], errors="coerce").fillna(0)
        else:
            result["quantidade"] = 1
            
        # Para datasets específicos, manter também a informação do tipo
        if "txcompetencia" in result.columns:
            result["tipo"] = result["txcompetencia"]
            
    else:
        # Para outros datasets (pendências, laudos realizados), cada linha é um registro
        result["quantidade"] = 1

    # ===== PROCESSAMENTO DE DATAS =====
    
    # Identificar coluna de data principal
    date_columns = ["data_interesse", "data_solicitacao", "dhemitido", "dhatendimento", "dhsolicitacao"]
    chosen_date_col = None
    
    for col in date_columns:
        if col in result.columns:
            chosen_date_col = col
            break
    
    if chosen_date_col:
        result["data_base"] = process_datetime_column(result[chosen_date_col])
        
        # Para dados mensais, usar o primeiro dia do mês
        if "mensal" in name.lower() or chosen_date_col == "data_interesse":
            if result["data_base"].notna().any():
                result["anomês_dt"] = result["data_base"].dt.to_period("M").dt.to_timestamp()
        else:
            # Para dados diários, manter a data original
            result["dia"] = result["data_base"].dt.normalize()
            if result["data_base"].notna().any():
                result["anomês_dt"] = result["data_base"].dt.to_period("M").dt.to_timestamp()

    # ===== DIMENSÕES PADRÃO =====
    
    # Mapear colunas de dimensões
    dimension_mapping = {
        "caso_sirsaelp": "id",
        "unidade": "unidade", 
        "superintendencia": "superintendencia",
        "diretoria": "diretoria",
        "tipopericia": "tipo",
        "competencia": "tipo",
        "perito": "perito"
    }
    
    for original_col, standard_col in dimension_mapping.items():
        if original_col in result.columns and standard_col not in result.columns:
            result[standard_col] = result[original_col]

    # ===== PROCESSAMENTO ESPECÍFICO LAUDOS REALIZADOS =====
    
    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            col_name = f"dh{field}"
            if col_name in result.columns:
                result[f"dh{field}"] = process_datetime_column(result[col_name])

        # Calcular TME se possível
        if "dhemissao" in result.columns and "dhatendimento" in result.columns:
            result["tme_dias"] = (result["dhemissao"] - result["dhatendimento"]).dt.days
        elif "dhemissao" in result.columns and "dhsolicitacao" in result.columns:
            result["tme_dias"] = (result["dhemissao"] - result["dhsolicitacao"]).dt.days
            
        if "tme_dias" in result.columns:
            result["sla_30_ok"] = result["tme_dias"] <= 30
            result["sla_60_ok"] = result["tme_dias"] <= 60

    # ===== CAMPOS DERIVADOS =====
    
    if "anomês_dt" in result.columns:
        result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
        result["ano"] = result["anomês_dt"].dt.year
        result["mes"] = result["anomês_dt"].dt.month

    # ===== LIMPEZA DE TEXTO =====
    
    text_cols = ["diretoria", "superintendencia", "unidade", "tipo", "id", "perito", "anomês"]
    for col in text_cols:
        if col in result.columns:
            result[col] = (
                result[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": None, "": None, "None": None})
            )

    return result

# === PADRONIZAÇÃO COM PERÍODO SEGURO ===
standardized_dfs = {}
processing_info = []

for name, df in raw_dataframes.items():
    standardized_df = standardize_dataframe(name, df)
    standardized_dfs[name] = standardized_df

    if "anomês" in standardized_df.columns and standardized_df["anomês"].notna().any():
        anomes_drop = standardized_df["anomês"].dropna()
        periodo_txt = f"{anomes_drop.min()} a {anomes_drop.max()}"
    else:
        periodo_txt = "Sem dados temporais"

    processing_info.append({
        "Arquivo": name,
        "Linhas": len(standardized_df),
        "Período": periodo_txt
    })

# Resumo na barra lateral
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

    for column, filter_values in [
        ("diretoria", filter_diretoria),
        ("superintendencia", filter_superintendencia),
        ("unidade", filter_unidade),
        ("tipo", filter_tipo),
    ]:
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
            else:
                cutoff_date = None
            if cutoff_date is not None:
                filtered = filtered[filtered["anomês_dt"] >= cutoff_date]

    return filtered

# === FILTROS RÁPIDOS ===
st.markdown("<h4 class='section-title'>🎛️ Filtros Rápidos</h4>", unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([0.45, 0.35, 0.20])
with fc1:
    quick_period = segment("Período", ["Ano atual","Últimos 6 meses","Últimos 3 meses","Todo o período"],
                           default=filter_periodo, key="quick_period")
    filter_periodo = quick_period
with fc2:
    foco = segment("Foco", ["Geral","Mensal","Diário"], default="Geral", key="quick_foco")
with fc3:
    show_bench = st.toggle("Metas", value=True, help="Exibir linhas de meta/benchmark nos gráficos")
    
filtered_dfs = {name: apply_filters(df) for name, df in standardized_dfs.items()}

# Atalhos
df_atend_todos = filtered_dfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dfs.get("Laudos_todos_Mensal")
df_atend_esp = filtered_dfs.get("Atendimentos_especifico_Mensal")
df_laudos_esp = filtered_dfs.get("Laudos_especifico_Mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("detalhes_laudospendentes")
df_pend_exames = filtered_dfs.get("detalhes_examespendentes")
df_atend_diario = filtered_dfs.get("Atendimentos_diario")
df_laudos_diario = filtered_dfs.get("Laudos_diario")

# ============ CÁLCULOS DE KPIs ============
def calculate_total(df: pd.DataFrame) -> int:
    if df is None or df.empty or "quantidade" not in df.columns:
        return 0
    return int(df["quantidade"].sum())

def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    monthly_totals = df.groupby("anomês_dt")["quantidade"].sum()
    return monthly_totals.mean() if len(monthly_totals) > 0 else None

def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    monthly_data = df.groupby("anomês_dt")["quantidade"].sum().sort_index().tail(periods * 2)
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
                metrics["correlacao_atend_laudos"] = float(correlation) if not pd.isna(correlation) else None
    return metrics

# CÁLCULOS PRINCIPAIS
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

tme_mediano = tme_medio = sla_30_percent = sla_60_percent = None
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

aging_laudos_medio = aging_exames_medio = None
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
def kpi_card(title, value, delta=None, help_text=None):
    html = f"""
    <div class="kpi-card">
      <p class="kpi-title">{title}</p>
      <p class="kpi-value">{value}</p>
      {f'<p class="kpi-delta">{delta}</p>' if delta else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.markdown("<h4 class='section-title'>📈 Indicadores Principais</h4>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    delta_atend = f"{format_number(crescimento_atendimentos,1)}% vs período anterior" if crescimento_atendimentos is not None else None
    kpi_card("Atendimentos Totais", format_number(total_atendimentos), delta_atend)
with c2:
    delta_laudos = f"{format_number(crescimento_laudos,1)}% vs período anterior" if crescimento_laudos is not None else None
    kpi_card("Laudos Emitidos", format_number(total_laudos), delta_laudos)
with c3:
    kpi_card("Taxa de Conversão", f"{format_number(taxa_atendimento,1)}%" if taxa_atendimento else "—")
with c4:
    kpi_card("Produtividade Mensal", f"{format_number(media_mensal_laudos,1)}" if media_mensal_laudos else "—")

st.markdown("<h4 class='section-title'>⏰ Gestão de Pendências</h4>", unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
with c5: kpi_card("Laudos Pendentes", format_number(total_pend_laudos))
with c6: kpi_card("Exames Pendentes", format_number(total_pend_exames))
with c7: kpi_card("Backlog (meses)", format_number(backlog_meses,1) if backlog_meses else "—")
with c8:
    aging_medio = aging_laudos_medio or aging_exames_medio
    kpi_card("Aging Médio (dias)", format_number(aging_medio,0) if aging_medio else "—")

if tme_mediano is not None or sla_30_percent is not None:
    st.markdown("<h4 class='section-title'>🎯 Indicadores de Performance</h4>", unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)
    with c9: kpi_card("TME Mediano (dias)", format_number(tme_mediano,1) if tme_mediano else "—")
    with c10: kpi_card("TME Médio (dias)", format_number(tme_medio,1) if tme_medio else "—")
    with c11: kpi_card("SLA 30 dias", f"{format_number(sla_30_percent,1)}%" if sla_30_percent else "—")
    with c12: kpi_card("SLA 60 dias", f"{format_number(sla_60_percent,1)}%" if sla_60_percent else "—")

# Alertas e insights
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

# ============ ABAS ============
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Visão Geral",
    "📈 Tendências", 
    "🏆 Rankings",
    "⏰ Pendências",
    "📋 Dados",
    "📑 Relatórios",
    "📅 Diário"
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
                    df_laudos_todos.groupby("unidade", as_index=False)["quantidade"].sum()
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
                    text="quantidade",
                )
                fig_unidades.update_traces(texttemplate='%{text}', textposition='outside')
                fig_unidades.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_unidades, use_container_width=True)

        with col_right:
            st.markdown("#### 🔍 Distribuição por Tipo (Pareto)")
            if df_laudos_esp is not None and "tipo" in df_laudos_esp.columns:
                tipo_summary = (
                    df_laudos_esp.groupby("tipo", as_index=False)["quantidade"].sum()
                    .sort_values("quantidade", ascending=False)
                )
                tipo_summary["pct"] = 100 * tipo_summary["quantidade"] / tipo_summary["quantidade"].sum()
                tipo_summary["pct_acum"] = tipo_summary["pct"].cumsum()

                fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pareto.add_trace(
                    go.Bar(x=tipo_summary["tipo"], y=tipo_summary["quantidade"], name="Total")
                )
                fig_pareto.add_trace(
                    go.Scatter(
                        x=tipo_summary["tipo"],
                        y=tipo_summary["pct_acum"],
                        mode="lines+markers",
                        name="% Acumulado",
                    ),
                    secondary_y=True,
                )
                if show_bench:
                    fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True)

                fig_pareto.update_layout(
                    title="Pareto – Tipos de Perícia",
                    hovermode="x unified",
                    xaxis={'categoryorder': 'array', 'categoryarray': tipo_summary["tipo"]},
                )
                fig_pareto.update_yaxes(title_text="Quantidade", secondary_y=False)
                fig_pareto.update_yaxes(title_text="% Acumulado", range=[0, 100], secondary_y=True)
                st.plotly_chart(fig_pareto, use_container_width=True)

    # Evolução Mensal
    if (
        df_atend_todos is not None and df_laudos_todos is not None
        and "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns
    ):
        st.markdown("#### 📅 Evolução Mensal: Atendimentos vs Laudos")

        atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        atend_monthly["Tipo"] = "Atendimentos"
        atend_monthly = atend_monthly.rename(columns={"quantidade": "Total"})

        laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        laudos_monthly["Tipo"] = "Laudos"
        laudos_monthly = laudos_monthly.rename(columns={"quantidade": "Total"})

        combined_data = pd.concat([atend_monthly, laudos_monthly])
        combined_data["Mês"] = combined_data["anomês_dt"].dt.strftime("%Y-%m")

        fig_temporal = px.line(
            combined_data,
            x="Mês",
            y="Total",
            color="Tipo",
            markers=True,
            title="Evolução Mensal: Atendimentos vs Laudos",
            line_shape="spline",
        )
        fig_temporal.update_layout(height=400, hovermode="x unified", xaxis_title="Período", yaxis_title="Quantidade")
        st.plotly_chart(fig_temporal, use_container_width=True)

        # Taxa de conversão
        merged_monthly = pd.merge(
            atend_monthly.rename(columns={"Total": "Atendimentos"}),
            laudos_monthly.rename(columns={"Total": "Laudos"}),
            on="anomês_dt",
            how="inner",
        )
        if not merged_monthly.empty:
            merged_monthly["Taxa_Conversao"] = (merged_monthly["Laudos"] / merged_monthly["Atendimentos"]) * 100
            merged_monthly["Mês"] = merged_monthly["anomês_dt"].dt.strftime("%Y-%m")
            fig_conversao = px.line(
                merged_monthly,
                x="Mês",
                y="Taxa_Conversao",
                markers=True,
                title="Taxa de Conversão Mensal (%)",
                line_shape="spline",
            )
            if show_bench:
                fig_conversao.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Meta: 70%")
            st.plotly_chart(fig_conversao, use_container_width=True)

        # Funil de conversão
        st.markdown("#### 🧯 Funil de Conversão (Atendimento → Laudo)")
        total_at = calculate_total(df_atend_todos)
        total_la = calculate_total(df_laudos_todos)
        funil = pd.DataFrame({"Etapa": ["Atendimentos", "Laudos"], "Total": [total_at, total_la]})
        fig_funnel = px.funnel(funil, x="Total", y="Etapa")
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Heatmap
    if df_laudos_todos is not None and "anomês_dt" in df_laudos_todos.columns:
        st.markdown("#### 🔥 Heatmap de Produção (Ano × Mês) – Laudos")
        tmp = df_laudos_todos.copy()
        tmp["Ano"] = tmp["anomês_dt"].dt.year
        tmp["Mês"] = tmp["anomês_dt"].dt.strftime("%b")

        pivot = (
            tmp.groupby(["Ano", "Mês"])["quantidade"].sum().reset_index()
        )

        meses_ordem = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot["Mês"] = pd.Categorical(pivot["Mês"], categories=meses_ordem, ordered=True)

        pivot_mat = pivot.pivot(index="Ano", columns="Mês", values="quantidade").fillna(0)

        fig_heat = px.imshow(
            pivot_mat,
            aspect="auto",
            text_auto=True,
            title="Heatmap Ano×Mês – Laudos"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ============ RODAPÉ ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.2</strong> - Sistema Avançado de Monitoramento (CORRIGIDO)</p>
    <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
    <p>Para suporte técnico ou sugestões: <strong>equipe-ti@pci.sc.gov.br</strong></p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
    <div style='margin-top: 15px; padding: 10px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;'>
        <p style='margin: 0; font-weight: bold; color: #1e40af;'>✨ CORREÇÕES APLICADAS:</p>
        <p style='margin: 5px 0 0 0; font-size: 12px; color: #1e40af;'>
            • Valores tratados como totais agregados (não como IDs)<br>
            • Cálculos de KPIs corrigidos<br>
            • Mapeamento de colunas ajustado<br>
            • Processamento de datas otimizado
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
