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
# === PRO UI: Tema Plotly, CSS e utilitários ===
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
      <div class="kpi-card" style="padding:8px 10px;"><span class="kpi-title">Versão</span><div class="kpi-value" style="font-size:16px;">2.1</div></div>
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
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
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

def get_period_filter_options(df: pd.DataFrame) -> List[str]:
    if df is None or "anomês_dt" not in df.columns:
        return []
    dates = df["anomês_dt"].dropna()
    if dates.empty:
        return []
    return ["Últimos 3 meses", "Últimos 6 meses", "Último ano", "Ano atual", "Todo o período"]

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
    }
}

# +++ NOVOS DATASETS DIÁRIOS +++
file_configs.update({
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
})

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
            df.columns = [re.sub(r"\s+", " ", col.strip().lower()) for col in df.columns]
            loaded_data[name] = df

    if "laudos_realizados" not in loaded_data:
        st.sidebar.info("📊 Usando dados simulados para Laudos Realizados (demo)")
        loaded_data["laudos_realizados"] = create_sample_laudos_realizados()

    return loaded_data

# Carrega os dados
# Spinner de carregamento
with st.spinner("Carregando e padronizando dados..."):
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
        "quantidade": "idatendimento"
    },
    "Atendimentos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "idatendimento",
        "quantidade": "idatendimento",
        "tipo": "txcompetencia"
    },
    "Laudos_todos_Mensal": {
        "date": "data_interesse",
        "id": "iddocumento",
        "quantidade": "iddocumento"
    },
    "Laudos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "iddocumento",
        "quantidade": "iddocumento",
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
    }
}

# +++ NOVOS MAPEAMENTOS DIÁRIOS +++
COLUMN_MAPPINGS.update({
    "Atendimentos_diario": {
        "date": "data_interesse",
        "id": "idatendimento",
        "quantidade": "idatendimento"
    },
    "Laudos_diario": {
        "date": "data_interesse",
        "id": "iddocumento",
        "quantidade": "iddocumento"
    }
})

# ============ PADRONIZAÇÃO DE DADOS ============
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza estrutura do DataFrame para análise unificada."""
    if df is None or df.empty:
        return pd.DataFrame()

    mapping = COLUMN_MAPPINGS.get(name, {})
    result = df.copy()

    # Quantidade
    if name in ["Atendimentos_todos_Mensal", "Laudos_todos_Mensal",
                "Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal",
                "Atendimentos_diario", "Laudos_diario"]:
        quantity_col = mapping.get("quantidade", mapping.get("id"))
        if quantity_col and quantity_col in result.columns:
            result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(1)
        else:
            result["quantidade"] = 1
    else:
        result["quantidade"] = 1

    # Dimensões
    for dim_col in ["diretoria", "superintendencia", "unidade", "tipo", "perito", "id"]:
        if dim_col in mapping and mapping[dim_col] in result.columns:
            result[dim_col] = result[mapping[dim_col]]

    # Fallbacks inteligentes de data-base (nível diário)
    fallback_date_candidates = [
        "dhemitido", "dhatendimento", "dhsolicitacao", "data_emissao",
        "data_interesse", "data", "dia", "data_base"
    ]
    mapped_date_col = mapping.get("date")
    chosen_date_col = None
    if mapped_date_col and mapped_date_col in result.columns:
        chosen_date_col = mapped_date_col
    else:
        for c in fallback_date_candidates:
            if c in result.columns:
                chosen_date_col = c
                break
    if chosen_date_col:
        result["data_base"] = process_datetime_column(result[chosen_date_col])

    # Competência / mês
    anomes_dt = None
    if "competencia" in mapping and mapping["competencia"] in result.columns:
        if mapping["competencia"] == "txcompetencia":
            date_col = mapping.get("date")
            if date_col and date_col in result.columns:
                date_series = process_datetime_column(result[date_col])
                if date_series is not None:
                    anomes_dt = date_series.dt.to_period("M").dt.to_timestamp()
        else:
            anomes_dt = process_datetime_column(result[mapping["competencia"]])
            if anomes_dt is not None:
                anomes_dt = anomes_dt.dt.to_period("M").dt.to_timestamp()

    if anomes_dt is None and "date" in mapping and mapping["date"] in result.columns:
        date_col = process_datetime_column(result[mapping["date"]])
        if date_col is not None:
            anomes_dt = date_col.dt.to_period("M").dt.to_timestamp()

    # Para laudos_realizados usar ano/mes se existir
    if anomes_dt is None and name == "laudos_realizados":
        ano_col = mapping.get("ano")
        mes_col = mapping.get("mes")
        if ano_col in result.columns and mes_col in result.columns:
            try:
                anos = pd.to_numeric(result[ano_col], errors="coerce")
                meses = pd.to_numeric(result[mes_col], errors="coerce")
                valid_mask = (~anos.isna()) & (~meses.isna()) & (meses >= 1) & (meses <= 12)
                if valid_mask.any():
                    dates = pd.to_datetime({'year': anos, 'month': meses, 'day': 1}, errors="coerce")
                    anomes_dt = dates.dt.to_period("M").dt.to_timestamp()
            except Exception:
                pass

    if anomes_dt is not None:
        result["anomês_dt"] = anomes_dt
        result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
        result["ano"] = result["anomês_dt"].dt.year
        result["mes"] = result["anomês_dt"].dt.month

    # Campo 'dia'
    if "data_base" in result.columns and result["data_base"].notna().any():
        result["dia"] = pd.to_datetime(result["data_base"]).dt.normalize()
    elif "anomês_dt" in result.columns:
        result["dia"] = pd.to_datetime(result["anomês_dt"]).dt.normalize()

    # Processamento específico laudos_realizados
    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            col_name = mapping.get(field)
            if col_name and col_name in result.columns:
                result[f"dh{field}"] = process_datetime_column(result[col_name])

        if "dhemissao" in result.columns:
            base_date = result.get("dhatendimento") if "dhatendimento" in result.columns else result.get("dhsolicitacao")
            if base_date is not None:
                result["tme_dias"] = (result["dhemissao"] - base_date).dt.days
                result["sla_30_ok"] = result["tme_dias"] <= 30
                result["sla_60_ok"] = result["tme_dias"] <= 60

    # Limpeza texto
    for col in ["diretoria", "superintendencia", "unidade", "tipo", "id", "perito", "anomês"]:
        if col in result.columns:
            result[col] = (
                result[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": None, "": None, "None": None})
            )

    return result

# === PRO: Padronização com período seguro ===
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

# === PRO UI: Filtros rápidos ===
st.markdown("<h4 class='section-title'>🎛️ Filtros Rápidos</h4>", unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([0.45, 0.35, 0.20])
with fc1:
    quick_period = segment("Período", ["Ano atual","Últimos 6 meses","Últimos 3 meses","Todo o período"],
                           default=filter_periodo, key="quick_period")
    filter_periodo = quick_period  # sincroniza com sua função apply_filters
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

# === DEBUG TEMPORÁRIO ===
st.sidebar.markdown("### 🔧 Debug Final")
if df_pend_laudos is not None:
    st.sidebar.write(f"📊 Laudos pendentes carregados: {len(df_pend_laudos)}")
    st.sidebar.write(f"📅 Colunas: {list(df_pend_laudos.columns)}")
    
    # Verificar coluna data_solicitacao especificamente
    if 'data_solicitacao' in df_pend_laudos.columns:
        valores_nao_nulos = df_pend_laudos['data_solicitacao'].notna().sum()
        st.sidebar.write(f"🗓️ data_solicitacao: {valores_nao_nulos} valores não nulos")
        if valores_nao_nulos > 0:
            amostra = df_pend_laudos['data_solicitacao'].dropna().head(3).tolist()
            st.sidebar.write(f"📝 Amostra: {amostra}")
    
    # Testar conversão de data
    if 'data_solicitacao' in df_pend_laudos.columns:
        try:
            datas_convertidas = pd.to_datetime(df_pend_laudos['data_solicitacao'], errors='coerce', dayfirst=True)
            validas = datas_convertidas.notna().sum()
            st.sidebar.write(f"✅ Datas convertidas com sucesso: {validas}")
        except Exception as e:
            st.sidebar.error(f"❌ Erro na conversão: {str(e)}")
else:
    st.sidebar.error("❌ df_pend_laudos é None")
# === FIM DEBUG ===

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

total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None and not df_pend_laudos.empty else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None and not df_pend_exames.empty else 0

# Debug para verificar
print(f"DEBUG Visão Geral - Laudos pendentes: {total_pend_laudos}")
print(f"DEBUG Visão Geral - Exames pendentes: {total_pend_exames}")
if df_pend_laudos is not None:
    print(f"DEBUG Visão Geral - df_pend_laudos shape: {df_pend_laudos.shape}")
if df_pend_exames is not None:
    print(f"DEBUG Visão Geral - df_pend_exames shape: {df_pend_exames.shape}")

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
# === PRO UI: Cards KPI ===
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
            if "tipo" in df_laudos_todos.columns:
                tipo_summary = (
                    df_laudos_todos.groupby("tipo", as_index=False)["quantidade"].sum()
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

                # mantém a ordem do eixo x conforme o ranking
                fig_pareto.update_layout(
                    title="Pareto – Tipos de Perícia",
                    hovermode="x unified",
                    xaxis={'categoryorder': 'array', 'categoryarray': tipo_summary["tipo"]},
                )
                fig_pareto.update_yaxes(title_text="Quantidade", secondary_y=False)
                fig_pareto.update_yaxes(title_text="% Acumulado", range=[0, 100], secondary_y=True)
                st.plotly_chart(fig_pareto, use_container_width=True)

    # --- Evolução Mensal: depende de atendimentos E laudos ---
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

        # Funil depende dos dois datasets (ok ficar aqui)
        st.markdown("#### 🧯 Funil de Conversão (Atendimento → Laudo)")
        total_at = calculate_total(df_atend_todos)
        total_la = calculate_total(df_laudos_todos)
        funil = pd.DataFrame({"Etapa": ["Atendimentos", "Laudos"], "Total": [total_at, total_la]})
        fig_funnel = px.funnel(funil, x="Total", y="Etapa")
        st.plotly_chart(fig_funnel, use_container_width=True)

    # --- Heatmap: depende só de laudos (fica FORA do if acima) ---
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


# ============ ABA 2: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise de Tendências")

    def create_enhanced_time_series(df: pd.DataFrame, title: str, line_color: str = "blue") -> None:
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        monthly_data = df.groupby("anomês_dt", as_index=False)["quantidade"].sum().sort_values("anomês_dt")
        if monthly_data.empty:
            st.info(f"Sem dados temporais para {title}")
            return
        monthly_data["Mês"] = monthly_data["anomês_dt"].dt.strftime("%Y-%m")

        fig = make_subplots(rows=2, cols=1, subplot_titles=(title, "Variação Percentual Mensal"),
                            vertical_spacing=0.15, row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=monthly_data["Mês"], y=monthly_data["quantidade"], mode="lines+markers",
                                 name="Valores", line=dict(color=line_color, width=2)), row=1, col=1)

        if len(monthly_data) >= 3:
            monthly_data["media_movel"] = monthly_data["quantidade"].rolling(window=3, center=True).mean()
            fig.add_trace(go.Scatter(x=monthly_data["Mês"], y=monthly_data["media_movel"], mode="lines",
                                     name="Média Móvel (3m)", line=dict(dash="dash", color="red", width=2)), row=1, col=1)

        monthly_data["variacao_pct"] = monthly_data["quantidade"].pct_change() * 100
        colors = ['red' if x < 0 else 'green' for x in monthly_data["variacao_pct"].fillna(0)]
        fig.add_trace(go.Bar(x=monthly_data["Mês"], y=monthly_data["variacao_pct"], name="Variação %",
                             marker_color=colors, showlegend=False), row=2, col=1)

        fig.update_layout(height=600, hovermode="x unified", showlegend=True)
        fig.update_xaxes(title_text="Período", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        create_enhanced_time_series(df_atend_todos, "🏥 Atendimentos - Análise Temporal", "blue")
        if df_atend_todos is not None and "anomês_dt" in df_atend_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Atendimentos")
            seasonal_data = df_atend_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            monthly_totals = seasonal_data.groupby(["mes_num", "mes_nome"])["quantidade"].sum().reset_index().sort_values("mes_num")
            fig_sazonal = px.bar(monthly_totals, x="mes_nome", y="quantidade", title="Distribuição Sazonal",
                                 color="quantidade", color_continuous_scale="Blues")
            fig_sazonal.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sazonal, use_container_width=True)

    with colB:
        create_enhanced_time_series(df_laudos_todos, "📄 Laudos - Análise Temporal", "green")
        if df_laudos_todos is not None and "anomês_dt" in df_laudos_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Laudos")
            seasonal_data = df_laudos_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            monthly_totals = seasonal_data.groupby(["mes_num", "mes_nome"])["quantidade"].sum().reset_index().sort_values("mes_num")
            fig_sazonal = px.bar(monthly_totals, x="mes_nome", y="quantidade", title="Distribuição Sazonal",
                                 color="quantidade", color_continuous_scale="Greens")
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
                "Atendimentos": atend_monthly.loc[common_periods],
                "Laudos": laudos_monthly.loc[common_periods]
            }).reset_index()
            correlation_data["Período"] = correlation_data["anomês_dt"].dt.strftime("%Y-%m")
            fig_scatter = px.scatter(correlation_data, x="Atendimentos", y="Laudos", hover_data=["Período"],
                                     title="Correlação: Atendimentos vs Laudos", trendline="ols")
            correlation_coef = correlation_data["Atendimentos"].corr(correlation_data["Laudos"])
            fig_scatter.add_annotation(text=f"Correlação: {correlation_coef:.3f}", xref="paper", yref="paper",
                                       x=0.02, y=0.98, showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

# === PRO Chart: Boxplot de TME por Unidade / Tipo ===
    if df_laudos_real is not None and "tme_dias" in df_laudos_real.columns:
        st.markdown("#### 📦 Distribuição de TME por Unidade / Tipo")
        bx1, bx2 = st.columns(2)
        with bx1:
            if "unidade" in df_laudos_real.columns:
                fig_box_u = px.box(df_laudos_real.dropna(subset=["tme_dias","unidade"]),
                    x="unidade", y="tme_dias", points="outliers",
                    title="TME (dias) por Unidade")
                fig_box_u.update_layout(height=450)
                st.plotly_chart(fig_box_u, use_container_width=True)
        with bx2:
            if "tipo" in df_laudos_real.columns:
                top_tipos = df_laudos_real["tipo"].value_counts().head(15).index
                df_top = df_laudos_real[df_laudos_real["tipo"].isin(top_tipos)]
                fig_box_t = px.box(df_top.dropna(subset=["tme_dias","tipo"]),
                    x="tipo", y="tme_dias", points="outliers",
                    title="TME (dias) por Tipo (Top 15)")
                fig_box_t.update_layout(height=450)
                st.plotly_chart(fig_box_t, use_container_width=True)


# ============ ABA 3: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Comparativos")

    def create_enhanced_ranking(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20) -> None:
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        # Verificar se tem dados válidos na dimensão
        valid_data = df[df[dimension].notna() & (df[dimension] != '') & (df[dimension] != 'nan')]
        if valid_data.empty:
            st.info(f"Sem dados válidos para {title}")
            return
            
        ranking_data = (valid_data.groupby(dimension).agg({"quantidade": ["sum", "count", "mean"]}).round(2))
        ranking_data.columns = ["Total", "Registros", "Média"]
        ranking_data = ranking_data.sort_values("Total", ascending=False).head(top_n).reset_index()
        
        if ranking_data.empty or ranking_data["Total"].sum() == 0:
            st.info(f"Sem dados numéricos para {title}")
            return
            
        fig = px.bar(
            ranking_data, x="Total", y=dimension, orientation="h", title=title,
            color="Total", color_continuous_scale="Viridis", hover_data=["Registros", "Média"]
        )
        fig.update_layout(height=max(400, len(ranking_data) * 30), showlegend=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"📊 Detalhes - {title}"):
            st.dataframe(ranking_data, use_container_width=True)

    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs(["Por Diretoria", "Por Unidade", "Por Tipo", "Comparativo"])
    
    with rank_tab1:
        st.markdown("#### 🏢 Rankings por Diretoria")
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_todos, "diretoria", "🏥 Atendimentos por Diretoria")
        with col2:
            create_enhanced_ranking(df_laudos_todos, "diretoria", "📄 Laudos por Diretoria")

    with rank_tab2:
        st.markdown("#### 🏢 Rankings por Unidade")
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_todos, "unidade", "🏥 Atendimentos por Unidade", 25)
        with col2:
            create_enhanced_ranking(df_laudos_todos, "unidade", "📄 Laudos por Unidade", 25)

    with rank_tab3:
        st.markdown("#### 🔍 Rankings por Tipo de Perícia")
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(df_atend_esp, "tipo", "🏥 Atendimentos por Tipo", 20)
        with col2:
            create_enhanced_ranking(df_laudos_esp, "tipo", "📄 Laudos por Tipo", 20)

    with rank_tab4:
        st.markdown("#### 📊 Análise Comparativa de Eficiência")
        if (df_atend_todos is not None and df_laudos_todos is not None and
                "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns):
            
            # Filtrar dados válidos
            atend_validos = df_atend_todos[df_atend_todos["unidade"].notna() & (df_atend_todos["unidade"] != '')]
            laudos_validos = df_laudos_todos[df_laudos_todos["unidade"].notna() & (df_laudos_todos["unidade"] != '')]
            
            if not atend_validos.empty and not laudos_validos.empty:
                atend_por_unidade = atend_validos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Atendimentos"})
                laudos_por_unidade = laudos_validos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade": "Laudos"})
                eficiencia_data = pd.merge(atend_por_unidade, laudos_por_unidade, on="unidade", how="inner")
                
                if not eficiencia_data.empty and len(eficiencia_data) > 0:
                    eficiencia_data["Taxa_Conversao"] = (eficiencia_data["Laudos"] / eficiencia_data["Atendimentos"]) * 100
                    eficiencia_data = eficiencia_data.sort_values("Taxa_Conversao", ascending=False)
                    
                    fig_eficiencia = px.scatter(
                        eficiencia_data.head(20), x="Atendimentos", y="Laudos", size="Taxa_Conversao",
                        hover_name="unidade", title="Eficiência por Unidade (Atendimentos vs Laudos)",
                        color="Taxa_Conversao", color_continuous_scale="RdYlGn"
                    )
                    fig_eficiencia.update_layout(height=500)
                    st.plotly_chart(fig_eficiencia, use_container_width=True)
                    
                    st.markdown("**🥇 Top 10 Unidades Mais Eficientes:**")
                    top_eficientes = eficiencia_data.head(10)[["unidade", "Taxa_Conversao", "Atendimentos", "Laudos"]]
                    st.dataframe(top_eficientes, use_container_width=True)
                else:
                    st.info("Sem dados suficientes para análise comparativa.")
            else:
                st.info("Dados de atendimentos ou laudos não disponíveis para comparação.")
        else:
            st.info("Datasets necessários não carregados para análise comparativa.")

    # === SEÇÃO ADICIONAL: Rankings de Pendências ===
    if df_pend_laudos is not None or df_pend_exames is not None:
        st.markdown("---")
        st.markdown("#### ⏰ Rankings de Pendências")
        
        pend_col1, pend_col2 = st.columns(2)
        
        with pend_col1:
            if df_pend_laudos is not None and not df_pend_laudos.empty:
                st.markdown("**📄 Laudos Pendentes por Unidade**")
                pend_laudos_ranking = df_pend_laudos.groupby("unidade").size().reset_index(name="Pendencias")
                pend_laudos_ranking = pend_laudos_ranking.sort_values("Pendencias", ascending=False).head(15)
                
                if not pend_laudos_ranking.empty:
                    fig_pend_laudos = px.bar(
                        pend_laudos_ranking, x="Pendencias", y="unidade", orientation="h",
                        title="Top 15 - Laudos Pendentes", color="Pendencias", color_continuous_scale="Reds"
                    )
                    fig_pend_laudos.update_layout(height=400, showlegend=False, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_pend_laudos, use_container_width=True)
        
        with pend_col2:
            if df_pend_exames is not None and not df_pend_exames.empty:
                st.markdown("**🔬 Exames Pendentes por Unidade**")
                pend_exames_ranking = df_pend_exames.groupby("unidade").size().reset_index(name="Pendencias")
                pend_exames_ranking = pend_exames_ranking.sort_values("Pendencias", ascending=False).head(15)
                
                if not pend_exames_ranking.empty:
                    fig_pend_exames = px.bar(
                        pend_exames_ranking, x="Pendencias", y="unidade", orientation="h",
                        title="Top 15 - Exames Pendentes", color="Pendencias", color_continuous_scale="Oranges"
                    )
                    fig_pend_exames.update_layout(height=400, showlegend=False, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_pend_exames, use_container_width=True)

# ============ ABA 4: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Gestão de Pendências")

def calculate_aging_analysis(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Calcula análise de aging com fallbacks robustos para colunas de data."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype="int64"), {}
    
    result = df.copy()
    
    # FORÇAR uso da data_solicitacao se disponível
    if 'data_solicitacao' in result.columns:
        try:
            # Limpar a coluna primeiro (remover espaços, aspas, etc.)
            data_col = result['data_solicitacao'].astype(str).str.strip().str.replace('"', '').str.replace("'", "")
            
            # Tentar múltiplos formatos de data
            dates = None
            formats_to_try = [
                None,  # Auto-detect
                "%d/%m/%Y",
                "%Y-%m-%d", 
                "%d-%m-%Y",
                "%Y/%m/%d",
                "%m/%d/%Y",
                "%d.%m.%Y",
                "%Y.%m.%d"
            ]
            
            for fmt in formats_to_try:
                try:
                    if fmt is None:
                        # Auto-detect com dayfirst=True (formato brasileiro)
                        test_dates = pd.to_datetime(data_col, errors="coerce", dayfirst=True, infer_datetime_format=True)
                    else:
                        # Formato específico
                        test_dates = pd.to_datetime(data_col, format=fmt, errors="coerce")
                    
                    valid_count = test_dates.notna().sum()
                    if valid_count > 0:  # Se pelo menos 1 data é válida
                        dates = test_dates
                        print(f"DEBUG: Formato funcionou: {fmt or 'auto-detect'}, {valid_count} datas válidas")
                        break
                        
                except Exception as e:
                    print(f"DEBUG: Erro no formato {fmt}: {e}")
                    continue
            
            if dates is None or dates.notna().sum() == 0:
                # Último recurso: tentar interpretar manualmente alguns formatos comuns
                try:
                    # Verificar se são números (timestamp)
                    numeric_data = pd.to_numeric(data_col, errors='coerce')
                    if numeric_data.notna().sum() > 0:
                        # Tentar como timestamp
                        dates = pd.to_datetime(numeric_data, unit='s', errors='coerce')
                        if dates.notna().sum() == 0:
                            dates = pd.to_datetime(numeric_data, unit='ms', errors='coerce')
                except:
                    pass
                    
        except Exception as e:
            print(f"DEBUG: Erro geral na conversão: {e}")
            dates = None
    else:
        dates = None
    
    # Se ainda não conseguiu converter, retornar erro informativo
    if dates is None or dates.notna().sum() == 0:
        # Mostrar amostras dos dados para debug
        if 'data_solicitacao' in result.columns:
            sample_data = result['data_solicitacao'].head(10).tolist()
            error_msg = f"Não foi possível converter as datas. Amostras: {sample_data}"
        else:
            error_msg = "Coluna data_solicitacao não encontrada"
            
        return result, pd.Series(dtype="int64"), {
            "total": len(result),
            "total_com_data_valida": 0,
            "media_dias": 0,
            "mediana_dias": 0,
            "max_dias": 0,
            "criticos": 0,
            "urgentes": 0,
            "erro": error_msg
        }
    
    # Calcular dias pendentes
    hoje = pd.Timestamp.now().normalize()
    dias_pendentes = (hoje - dates).dt.days
    
    # Filtrar valores válidos (0 a 15000 dias - mais permissivo)
    mask_valido = (dias_pendentes >= 0) & (dias_pendentes <= 15000) & dias_pendentes.notna()
    dias_pendentes_validos = dias_pendentes.where(mask_valido)
    
    # Criar faixas de aging
    faixas_aging = pd.cut(
        dias_pendentes_validos,
        bins=[-1, 15, 30, 60, 90, 180, 365, float('inf')],
        labels=["0-15 dias", "16-30 dias", "31-60 dias", "61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"],
        include_lowest=True
    )
    
    # Criar prioridades
    prioridade = pd.cut(
        dias_pendentes_validos,
        bins=[-1, 30, 90, 180, float('inf')],
        labels=["Normal", "Atenção", "Urgente", "Crítico"],
        include_lowest=True
    )
    
    # Adicionar colunas ao resultado
    result["dias_pendentes"] = dias_pendentes_validos
    result["faixa_aging"] = faixas_aging
    result["prioridade"] = prioridade
    result["data_convertida"] = dates  # Para debug
    
    # Distribuição por faixa (apenas valores válidos)
    distribuicao = faixas_aging.value_counts().sort_index()
    
    # Estatísticas
    dias_validos = dias_pendentes_validos.dropna()
    stats = {
        "total": len(result),
        "total_com_data_valida": len(dias_validos),
        "media_dias": float(dias_validos.mean()) if len(dias_validos) > 0 else 0,
        "mediana_dias": float(dias_validos.median()) if len(dias_validos) > 0 else 0,
        "max_dias": int(dias_validos.max()) if len(dias_validos) > 0 else 0,
        "criticos": int((prioridade == "Crítico").sum()),
        "urgentes": int((prioridade == "Urgente").sum()),
        "datas_originais_validas": dates.notna().sum() if dates is not None else 0
    }
    
    return result, distribuicao, stats

    col1, col2 = st.columns(2)
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📄 Laudos Pendentes")
    if df_pend_laudos is not None and not df_pend_laudos.empty:
        # Forçar processamento correto das datas
        df_laudos_temp = df_pend_laudos.copy()
        
        # Se data_base não existe ou está vazia, tentar criar
        if 'data_base' not in df_laudos_temp.columns or df_laudos_temp['data_base'].isna().all():
            if 'data_solicitacao' in df_laudos_temp.columns:
                df_laudos_temp['data_base'] = pd.to_datetime(df_laudos_temp['data_solicitacao'], errors='coerce', dayfirst=True)
        
        laudos_aged, dist_laudos, stats_laudos = calculate_aging_analysis(df_laudos_temp)
        
        # Mostrar métricas
        col_a, col_b, col_c = st.columns(3)
        with col_a: 
            total_laudos = stats_laudos.get("total_com_data_valida", stats_laudos.get("total", 0))
            st.metric("Total", format_number(total_laudos))
        with col_b: 
            criticos_laudos = stats_laudos.get("criticos", 0)
            st.metric("Críticos", criticos_laudos)
        with col_c: 
            media_laudos = stats_laudos.get("media_dias", 0)
            st.metric("Média (dias)", format_number(media_laudos, 1))
        
        # Verificar se há dados válidos para gráficos
        if stats_laudos.get("total_com_data_valida", 0) > 0 and not dist_laudos.empty:
            # Gráfico de distribuição
            fig_aging_laudos = px.bar(
                x=dist_laudos.index, y=dist_laudos.values, 
                title="Distribuição por Tempo de Pendência",
                color=dist_laudos.values, color_continuous_scale="Reds", 
                text=dist_laudos.values
            )
            fig_aging_laudos.update_traces(texttemplate='%{text}', textposition='outside')
            fig_aging_laudos.update_layout(height=350, showlegend=False, 
                                         xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
            st.plotly_chart(fig_aging_laudos, use_container_width=True)
            
            # Gráfico de prioridade
            if "prioridade" in laudos_aged.columns:
                prioridade_dist = laudos_aged["prioridade"].value_counts()
                if not prioridade_dist.empty:
                    fig_prioridade = px.pie(values=prioridade_dist.values, names=prioridade_dist.index,
                                          title="Distribuição por Prioridade",
                                          color_discrete_map={"Normal": "green", "Atenção": "yellow", 
                                                            "Urgente": "orange", "Crítico": "red"})
                    fig_prioridade.update_layout(height=300)
                    st.plotly_chart(fig_prioridade, use_container_width=True)
            
            # Top 10 mais antigas
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in laudos_aged.columns:
                # Selecionar colunas para exibição
                display_cols = []
                for col in ["id", "caso_sirsaelp", "unidade", "tipo", "tipopericia", "dias_pendentes", "prioridade"]:
                    if col in laudos_aged.columns:
                        display_cols.append(col)
                
                if display_cols and laudos_aged["dias_pendentes"].notna().sum() > 0:
                    oldest = laudos_aged.nlargest(10, "dias_pendentes")[display_cols]
                    st.dataframe(oldest, use_container_width=True, height=250)
                else:
                    st.info("Não há dados de aging válidos para exibir.")
        else:
            st.warning(f"⚠️ Problema no processamento das datas dos laudos pendentes.")
            if "erro" in stats_laudos:
                st.error(stats_laudos["erro"])
            st.info(f"Registros carregados: {stats_laudos.get('total', 0)}")
            st.info(f"Registros com data válida: {stats_laudos.get('total_com_data_valida', 0)}")
            
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
                fig_aging_exames = px.bar(
                    x=dist_exames.index, y=dist_exames.values, title="Distribuição por Tempo de Pendência",
                    color=dist_exames.values, color_continuous_scale="Oranges", text=dist_exames.values
                )
                fig_aging_exames.update_traces(texttemplate='%{text}', textposition='outside')
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
                display_cols = [c for c in ["id", "unidade", "tipo", "dias_pendentes", "prioridade"] if c in exames_aged.columns]
                oldest = exames_aged.nlargest(10, "dias_pendentes")[display_cols] if display_cols else exames_aged.nlargest(10, "dias_pendentes")
                st.dataframe(oldest, use_container_width=True, height=250)
        else:
            st.info("Sem dados de exames pendentes disponíveis.")

    st.markdown("#### 🏢 Análise de Pendências por Unidade")
    from functools import reduce
    pendencias_por_unidade = []
    if df_pend_laudos is not None and "unidade" in df_pend_laudos.columns:
        laudos_unidade = df_pend_laudos.groupby("unidade").size().reset_index(name="Laudos_Pendentes")
        pendencias_por_unidade.append(laudos_unidade)
    if df_pend_exames is not None and "unidade" in df_pend_exames.columns:
        exames_unidade = df_pend_exames.groupby("unidade").size().reset_index(name="Exames_Pendentes")
        pendencias_por_unidade.append(exames_unidade)

    if pendencias_por_unidade:
        pendencias_consolidadas = reduce(lambda left, right: pd.merge(left, right, on="unidade", how="outer"),
                                         pendencias_por_unidade).fillna(0)
        pendencias_consolidadas["Total_Pendencias"] = pendencias_consolidadas.get("Laudos_Pendentes", 0) + pendencias_consolidadas.get("Exames_Pendentes", 0)
        pendencias_consolidadas = pendencias_consolidadas.sort_values("Total_Pendencias", ascending=False)

        fig_pendencias = go.Figure()
        if "Laudos_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(name='Laudos Pendentes',
                                            y=pendencias_consolidadas["unidade"].head(15),
                                            x=pendencias_consolidadas["Laudos_Pendentes"].head(15),
                                            orientation='h', marker_color='lightcoral'))
        if "Exames_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(name='Exames Pendentes',
                                            y=pendencias_consolidadas["unidade"].head(15),
                                            x=pendencias_consolidadas["Exames_Pendentes"].head(15),
                                            orientation='h', marker_color='lightsalmon'))
        fig_pendencias.update_layout(title="Top 15 Unidades com Mais Pendências", barmode='stack',
                                     height=500, xaxis_title="Quantidade de Pendências",
                                     yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pendencias, use_container_width=True)

        st.markdown("**📊 Detalhamento por Unidade:**")
        st.dataframe(pendencias_consolidadas.head(20), use_container_width=True, height=300)
        # === PRO Chart: Stacked – Faixa de Aging x Diretoria ===
    st.markdown("#### 🧱 Pendências por Faixa de Aging × Diretoria (Stacked)")
    def stacked_aging(df, titulo):
        if df is None or df.empty or "diretoria" not in df.columns:
            st.info(f"Sem dados para {titulo}")
            return
        aged, _, _ = calculate_aging_analysis(df)
        if "faixa_aging" not in aged.columns:
            st.info(f"Sem dados para {titulo}")
            return
        g = aged.groupby(["diretoria","faixa_aging"]).size().reset_index(name="Total")
        g = g.pivot(index="diretoria", columns="faixa_aging", values="Total").fillna(0)
        fig = px.bar(g, barmode="stack", title=titulo)
        fig.update_layout(height=420, xaxis_title="Diretoria", yaxis_title="Pendências")
        st.plotly_chart(fig, use_container_width=True)

    stacked_aging(df_pend_laudos, "Laudos Pendentes – Faixa de Aging × Diretoria")
    stacked_aging(df_pend_exames, "Exames Pendentes – Faixa de Aging × Diretoria")


# ============ ABA 5: DADOS ============
with tab5:
    st.subheader("📋 Exploração dos Dados")

    st.markdown("#### 📊 Resumo dos Datasets Carregados")
    data_summary = []
    for name, df in standardized_dfs.items():
        if df is not None and not df.empty:
            periodo_info = "Sem dados temporais"
            if 'anomês' in df.columns and not df['anomês'].isna().all():
                periodo_info = f"{df['anomês'].min()} a {df['anomês'].max()}"
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
                if 'anomês_dt' in df_selected.columns:
                    unique_months = df_selected['anomês_dt'].nunique()
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
                    if null_percent == 0:
                        quality = "🟢 Excelente"
                    elif null_percent < 5:
                        quality = "🟡 Boa"
                    elif null_percent < 20:
                        quality = "🟠 Regular"
                    else:
                        quality = "🔴 Ruim"
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
                if 'anomês' in df_selected.columns:
                    available_months = sorted(df_selected['anomês'].dropna().unique(), reverse=True)
                    selected_months = st.multiselect("Filtrar por período:", available_months,
                                                     default=available_months[:6] if len(available_months) > 6 else available_months)
                else:
                    selected_months = []
            with viz_col3:
                all_columns = list(df_selected.columns)
                selected_columns = st.multiselect("Colunas a exibir:", all_columns,
                                                  default=all_columns[:10] if len(all_columns) > 10 else all_columns)

            df_display = df_selected.copy()
            if selected_months and 'anomês' in df_display.columns:
                df_display = df_display[df_display['anomês'].isin(selected_months)]
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
                csv_data = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dados Filtrados (CSV)",
                    data=csv_data,
                    file_name=f"{selected_dataset}_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col_down2:
                csv_complete = df_selected.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset Completo (CSV)",
                    data=csv_complete,
                    file_name=f"{selected_dataset}_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ============ ABA 6: RELATÓRIOS ============
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
- **Atendimentos Totais:** {format_number(total_atendimentos)}
- **Laudos Emitidos:** {format_number(total_laudos)}
- **Taxa de Conversão:** {format_number(taxa_atendimento, 1) if taxa_atendimento else 'N/A'}%
- **Produtividade Mensal:** {format_number(media_mensal_laudos, 1) if media_mensal_laudos else 'N/A'} laudos/mês

## ⏰ GESTÃO DE PENDÊNCIAS
- **Laudos Pendentes:** {format_number(total_pend_laudos)}
- **Exames Pendentes:** {format_number(total_pend_exames)}
- **Backlog Estimado:** {format_number(backlog_meses, 1) if backlog_meses else 'N/A'} meses
- **Aging Médio:** {format_number(aging_laudos_medio or aging_exames_medio, 0) if (aging_laudos_medio or aging_exames_medio) else 'N/A'} dias

## 🎯 PERFORMANCE OPERACIONAL
- **TME Mediano:** {format_number(tme_mediano, 1) if tme_mediano else 'N/A'} dias
- **SLA 30 dias:** {format_number(sla_30_percent, 1) if sla_30_percent else 'N/A'}%
- **SLA 60 dias:** {format_number(sla_60_percent, 1) if sla_60_percent else 'N/A'}%

## 📈 TENDÊNCIAS
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
        relatorio += "\n".join(alertas_relatorio) if alertas_relatorio else "✅ **Situação Normal:** Todos os indicadores dentro dos parâmetros esperados"

        relatorio += "\n\n## 📋 DATASETS UTILIZADOS\n"
        for name, df in standardized_dfs.items():
            if df is not None and not df.empty:
                relatorio += f"- **{name.replace('_', ' ').title()}:** {len(df):,} registros\n"

        relatorio += "\n---\n*Relatório gerado automaticamente pelo Dashboard PCI/SC*\n*Sistema de Monitoramento de Produção e Pendências*"
        return relatorio.strip()

    if tipo_relatorio == "Relatório Executivo Completo":
        relatorio_texto = gerar_relatorio_executivo()
        st.markdown("#### 📄 Visualização do Relatório")
        st.markdown(relatorio_texto)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Relatório Executivo",
            data=relatorio_texto.encode('utf-8'),
            file_name=f"relatorio_executivo_pci_sc_{timestamp}.md",
            mime="text/markdown"
        )
    elif tipo_relatorio == "Relatório de Produção":
        st.markdown("#### 📊 Relatório de Produção")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Métricas de Produção:**")
            if df_laudos_todos is not None and "anomês" in df_laudos_todos.columns:
                prod_mensal = (df_laudos_todos.groupby("anomês")["quantidade"].sum().reset_index().sort_values("anomês"))
                st.line_chart(prod_mensal.set_index("anomês")["quantidade"], height=300)
        with col2:
            st.markdown("**Top Produtores (Unidades):**")
            if df_laudos_todos is not None and "unidade" in df_laudos_todos.columns:
                top_unidades = (df_laudos_todos.groupby("unidade")["quantidade"].sum().sort_values(ascending=False).head(10))
                st.bar_chart(top_unidades, height=300)
    else:
        st.info(f"Relatório '{tipo_relatorio}' em desenvolvimento.")

# ============ ABA 7: DIÁRIO ============
with tab7:
    st.subheader("📅 Análise Diária – Atendimentos e Laudos")

    def daily_counts(df: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
        if df is None or df.empty or "dia" not in df.columns:
            return pd.DataFrame(columns=["dia", label])
        tmp = (df.dropna(subset=["dia"]).groupby("dia", as_index=False)["quantidade"].sum()
               .rename(columns={"quantidade": label}).sort_values("dia"))
        return tmp

    atend_d = daily_counts(df_atend_diario, "Atendimentos")
    laudos_d = daily_counts(df_laudos_diario, "Laudos")

    if atend_d.empty and laudos_d.empty:
        st.info("Sem dados diários carregados. Envie **Atendimentos (Diário)** e/ou **Laudos (Diário)**.")
    else:
        diario = pd.merge(atend_d, laudos_d, on="dia", how="outer").fillna(0)
        diario["Atendimentos"] = pd.to_numeric(diario["Atendimentos"], errors="coerce").fillna(0)
        diario["Laudos"] = pd.to_numeric(diario["Laudos"], errors="coerce").fillna(0)
        diario = diario.sort_values("dia").reset_index(drop=True)

        def mm7(s: pd.Series) -> pd.Series:
            return s.rolling(7).mean()

        diario["MA7_Atend"] = mm7(diario["Atendimentos"])
        diario["MA7_Laudos"] = mm7(diario["Laudos"])
        diario["Taxa_Conversao_%"] = np.where(
            diario["Atendimentos"] > 0, (diario["Laudos"] / diario["Atendimentos"]) * 100, np.nan
        )
        diario["MA7_Taxa_%"] = mm7(diario["Taxa_Conversao_%"])

        ultima_data = diario["dia"].max() if not diario.empty else None
        ult_reg = diario[diario["dia"] == ultima_data].iloc[0] if ultima_data is not None else None

        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric("Último dia", ultima_data.strftime("%d/%m/%Y") if ultima_data is not None else "—")
        with colB:
            st.metric("Atendimentos (último dia)", f"{int(ult_reg['Atendimentos']):,}".replace(",", ".") if ult_reg is not None else "—")
        with colC:
            st.metric("Laudos (último dia)", f"{int(ult_reg['Laudos']):,}".replace(",", ".") if ult_reg is not None else "—")
        with colD:
            taxa = ult_reg["Taxa_Conversao_%"] if (ult_reg is not None and not pd.isna(ult_reg["Taxa_Conversao_%"])) else None
            st.metric("Taxa de Conversão (últ. dia)", f"{taxa:.1f}%" if taxa is not None else "—")

        st.markdown("#### 📈 Evolução Diária")
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=diario["dia"], y=diario["Atendimentos"], mode="lines", name="Atendimentos"))
        fig_d.add_trace(go.Scatter(x=diario["dia"], y=diario["Laudos"], mode="lines", name="Laudos"))
        if diario["MA7_Atend"].notna().any():
            fig_d.add_trace(go.Scatter(x=diario["dia"], y=diario["MA7_Atend"], mode="lines", name="Atend MM7", line=dict(dash="dash")))
        if diario["MA7_Laudos"].notna().any():
            fig_d.add_trace(go.Scatter(x=diario["dia"], y=diario["MA7_Laudos"], mode="lines", name="Laudos MM7", line=dict(dash="dash")))
        fig_d.update_layout(height=420, hovermode="x unified", xaxis_title="Dia", yaxis_title="Quantidade")
        st.plotly_chart(fig_d, use_container_width=True)

        if diario["Taxa_Conversao_%"].notna().any():
            st.markdown("#### 🎯 Taxa de Conversão Diária (%)")
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(x=diario["dia"], y=diario["Taxa_Conversao_%"], mode="lines", name="Taxa Conversão (%)"))
            if diario["MA7_Taxa_%"].notna().any():
                fig_tc.add_trace(go.Scatter(x=diario["dia"], y=diario["MA7_Taxa_%"], mode="lines", name="Taxa MM7 (%)", line=dict(dash="dash")))
            if show_bench:
                fig_tc.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Meta 70%")
            fig_tc.update_layout(height=320, hovermode="x unified", xaxis_title="Dia", yaxis_title="%")
            st.plotly_chart(fig_tc, use_container_width=True)

        st.markdown("#### 📋 Tabela Diária – Atendimentos e Laudos")
        tabela = diario.copy()
        tabela["dia"] = tabela["dia"].dt.strftime("%d/%m/%Y")
        cols = ["dia", "Atendimentos", "Laudos", "Taxa_Conversao_%", "MA7_Atend", "MA7_Laudos", "MA7_Taxa_%"]
        cols = [c for c in cols if c in tabela.columns]
        st.dataframe(tabela[cols].tail(120), use_container_width=True, height=420)

        csv_daily = diario.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar tabela diária (CSV)",
            data=csv_daily,
            file_name=f"diario_atendimentos_laudos_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============ RODAPÉ ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.1</strong> - Sistema Avançado de Monitoramento</p>
    <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
    <p>Para suporte técnico ou sugestões: <strong>victor.poubel@policiacientifica.sc.gov.br</strong></p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)
