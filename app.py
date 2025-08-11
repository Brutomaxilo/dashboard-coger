import io
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Union

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============ CONFIGURAÇÃO INICIAL ============
st.set_page_config(
    page_title="PCI/SC – Dashboard Executivo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# === CONFIGURAÇÕES DE TEMA E ESTILO ===
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = 400

CUSTOM_CSS = """
<style>
/* Cards KPI */
.kpi-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 16px 20px;
  height: 100%;
  box-shadow: 0 2px 4px rgba(16,24,40,0.08);
  transition: all 0.2s ease;
}
.kpi-card:hover {
  box-shadow: 0 4px 8px rgba(16,24,40,0.12);
  transform: translateY(-1px);
}
.kpi-title {
  font-size: 13px;
  color: #64748b;
  font-weight: 500;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.kpi-value {
  font-size: 28px;
  font-weight: 700;
  color: #1e293b;
  margin: 4px 0 2px 0;
  line-height: 1.1;
}
.kpi-delta {
  font-size: 12px;
  color: #475569;
  margin-top: 4px;
  font-weight: 500;
}
.section-title {
  margin: 24px 0 12px 0;
  color: #1e293b;
  font-weight: 600;
}
.alert-success { 
  background: #dcfce7; 
  border-left: 4px solid #22c55e; 
  padding: 12px 16px; 
  border-radius: 6px;
  margin: 8px 0;
}
.alert-warning { 
  background: #fef3c7; 
  border-left: 4px solid #f59e0b; 
  padding: 12px 16px; 
  border-radius: 6px;
  margin: 8px 0;
}
.alert-danger { 
  background: #fee2e2; 
  border-left: 4px solid #ef4444; 
  padding: 12px 16px; 
  border-radius: 6px;
  margin: 8px 0;
}
hr { 
  margin: 12px 0 24px 0; 
  border: none; 
  height: 1px; 
  background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
}
/* Melhoria nos gráficos */
.js-plotly-plot {
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# === FUNÇÕES UTILITÁRIAS ===
def segment(label, options, default=None, key=None):
    """Wrapper para segmented_control com fallback para radio"""
    try:
        return st.segmented_control(label, options, default=default, key=key)
    except Exception:
        idx = options.index(default) if (default in options) else 0
        return st.radio(label, options, index=idx, horizontal=True, key=key)

def format_number(value: Union[float, int], decimal_places: int = 0) -> str:
    """Formata números com separadores brasileiros"""
    if pd.isna(value) or value is None:
        return "—"
    try:
        if decimal_places == 0:
            return f"{int(round(value)):,}".replace(",", ".")
        else:
            return f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "—"

def calculate_percentage(numerator: float, denominator: float) -> Optional[float]:
    """Calcula percentual com verificação de divisão por zero"""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return (numerator / denominator) * 100

def kpi_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Cria card KPI estilizado"""
    delta_html = f'<p class="kpi-delta">{delta}</p>' if delta else ''
    html = f"""
    <div class="kpi-card">
      <p class="kpi-title">{title}</p>
      <p class="kpi-value">{value}</p>
      {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# === HEADER ESTILIZADO ===
colh1, colh2 = st.columns([0.7, 0.3])
with colh1:
    st.markdown("<h1 style='margin-bottom:8px'>🏥 Dashboard PCI/SC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:16px; margin:0;'>Sistema Executivo de Monitoramento • Produção • Pendências • Performance</p>", unsafe_allow_html=True)
with colh2:
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.markdown(f"""
    <div style="display:flex; gap:8px; justify-content:flex-end; align-items:center;">
      <div class="kpi-card" style="padding:8px 12px; text-align:center;">
        <span class="kpi-title">Versão</span>
        <div class="kpi-value" style="font-size:18px;">3.0</div>
      </div>
      <div class="kpi-card" style="padding:8px 12px; text-align:center;">
        <span class="kpi-title">Atualizado</span>
        <div class="kpi-value" style="font-size:14px;">{current_time}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ============ CACHE E PERFORMANCE ============
@st.cache_data(ttl=3600, show_spinner="Processando dados...")
def read_csv_enhanced(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Lê CSV com detecção automática melhorada de separador e encoding"""
    separators = [";", ",", "\t", "|"]
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    for encoding in encodings:
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                # Primeira tentativa com configurações básicas
                df = pd.read_csv(bio, sep=sep, encoding=encoding, engine="python")
                
                # Verificação de qualidade do parsing
                if df.shape[1] > 1 and len(df) > 0:
                    # Limpeza de aspas e espaços
                    df.columns = [col.strip().strip('"').strip() for col in df.columns]
                    
                    # Limpeza dos dados
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip().str.strip('"')
                            # Conversão de valores numéricos mascarados como string
                            if col in ['idatendimento', 'iddocumento', 'quantidade']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
            except Exception as e:
                continue
    
    # Fallback para detecção automática mais agressiva
    try:
        bio = io.BytesIO(file_content)
        df = pd.read_csv(bio, sep=None, engine="python", encoding="utf-8", 
                        quotechar='"', skipinitialspace=True)
        if df.shape[1] > 1:
            df.columns = [col.strip().strip('"') for col in df.columns]
            return df
    except Exception:
        pass
    
    st.error(f"❌ Não foi possível processar o arquivo {filename}")
    return None

@st.cache_data(ttl=3600)
def process_datetime_enhanced(series: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    """Processamento aprimorado de datas com múltiplos formatos"""
    if series is None or len(series) == 0:
        return None
    
    # Lista de formatos comuns para tentar
    date_formats = [
        "%Y-%m-%d",      # ISO format
        "%d/%m/%Y",      # Brazilian format
        "%m/%d/%Y",      # US format
        "%d-%m-%Y",      # European format
        "%Y/%m/%d",      # Alternative ISO
        "%d.%m.%Y",      # German format
    ]
    
    # Primeira tentativa com inferência automática
    dt_series = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
    
    # Se muitos valores falharam, tentar formatos específicos
    if dt_series.isna().sum() > len(dt_series) * 0.3:
        for fmt in date_formats:
            try:
                dt_test = pd.to_datetime(series, format=fmt, errors="coerce")
                if dt_test.notna().sum() > dt_series.notna().sum():
                    dt_series = dt_test
                    break
            except Exception:
                continue
    
    return dt_series if dt_series.notna().any() else None

# ============ MAPEAMENTO DE COLUNAS MELHORADO ============
ENHANCED_COLUMN_MAPPINGS = {
    "detalhes_laudospendentes": {
        "date_columns": ["data_solicitacao"],
        "id_column": "caso_sirsaelp",
        "dimensions": {
            "unidade": "unidade",
            "superintendencia": "superintendencia", 
            "diretoria": "diretoria",
            "tipo": "tipopericia",
            "perito": "perito",
            "competencia": "competencia"
        }
    },
    "detalhes_examespendentes": {
        "date_columns": ["data_solicitacao"],
        "id_column": "caso_sirsaelp",
        "dimensions": {
            "unidade": "unidade",
            "superintendencia": "superintendencia",
            "diretoria": "diretoria", 
            "tipo": "tipopericia",
            "competencia": "competencia"
        }
    },
    "Atendimentos_todos_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "aggregation_level": "monthly"
    },
    "Laudos_todos_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento", 
        "quantity_column": "iddocumento",
        "aggregation_level": "monthly"
    },
    "Atendimentos_especifico_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "dimensions": {"tipo": "txcompetencia"},
        "aggregation_level": "monthly"
    },
    "Laudos_especifico_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento",
        "quantity_column": "iddocumento", 
        "dimensions": {"tipo": "txcompetencia"},
        "aggregation_level": "monthly"
    },
    "Atendimentos_diario": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "aggregation_level": "daily"
    },
    "Laudos_diario": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento",
        "quantity_column": "iddocumento", 
        "aggregation_level": "daily"
    }
}

# ============ PADRONIZAÇÃO MELHORADA ============
@st.cache_data(ttl=3600)
def standardize_dataframe_enhanced(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padronização aprimorada com mapeamento flexível"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    result = df.copy()
    mapping = ENHANCED_COLUMN_MAPPINGS.get(name, {})
    
    # Normalização de nomes de colunas
    result.columns = [col.lower().strip().replace(' ', '_') for col in result.columns]
    
    # Processamento de quantidade
    quantity_col = mapping.get("quantity_column")
    if quantity_col and quantity_col in result.columns:
        result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(1)
    else:
        result["quantidade"] = 1
    
    # Processamento de dimensões
    dimensions = mapping.get("dimensions", {})
    for target_col, source_col in dimensions.items():
        if source_col in result.columns:
            result[target_col] = result[source_col].astype(str).str.strip().str.title()
    
    # Processamento de datas
    date_columns = mapping.get("date_columns", [])
    for date_col in date_columns:
        if date_col in result.columns:
            processed_date = process_datetime_enhanced(result[date_col])
            if processed_date is not None:
                result["data_base"] = processed_date
                
                # Criar campos derivados de data
                result["anomês_dt"] = processed_date.dt.to_period("M").dt.to_timestamp()
                result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
                result["ano"] = result["anomês_dt"].dt.year
                result["mes"] = result["anomês_dt"].dt.month
                result["dia"] = processed_date.dt.normalize()
                break
    
    # Processamento de ID único
    id_col = mapping.get("id_column")
    if id_col and id_col in result.columns:
        result["id"] = result[id_col].astype(str)
    
    # Limpeza final de dados categóricos
    categorical_cols = ["diretoria", "superintendencia", "unidade", "tipo", "perito", "competencia"]
    for col in categorical_cols:
        if col in result.columns:
            result[col] = (result[col].astype(str)
                          .str.strip()
                          .str.title()
                          .replace({"Nan": None, "": None, "None": None}))
    
    return result

# ============ DETECÇÃO E CARREGAMENTO DE ARQUIVOS ============
def detect_data_sources() -> bool:
    """Detecta se existem arquivos na pasta data/"""
    return os.path.exists("data") and any(f.endswith(".csv") for f in os.listdir("data"))

def get_file_configs() -> Dict[str, Dict]:
    """Configurações dos arquivos esperados"""
    return {
        "Atendimentos_todos_Mensal": {
            "label": "Atendimentos Todos (Mensal)",
            "description": "Dados agregados de atendimentos por mês",
            "pattern": ["atendimentos_todos", "atendimentos todos"]
        },
        "Laudos_todos_Mensal": {
            "label": "Laudos Todos (Mensal)", 
            "description": "Dados agregados de laudos por mês",
            "pattern": ["laudos_todos", "laudos todos"]
        },
        "Atendimentos_especifico_Mensal": {
            "label": "Atendimentos Específicos (Mensal)",
            "description": "Atendimentos detalhados por tipo e competência",
            "pattern": ["atendimentos_especifico", "atendimentos especifico"]
        },
        "Laudos_especifico_Mensal": {
            "label": "Laudos Específicos (Mensal)",
            "description": "Laudos detalhados por tipo e competência", 
            "pattern": ["laudos_especifico", "laudos especifico"]
        },
        "Atendimentos_diario": {
            "label": "Atendimentos (Diário)",
            "description": "Registros diários de atendimentos",
            "pattern": ["atendimentos_diario", "atendimentos_diário", "atendimentos diário"]
        },
        "Laudos_diario": {
            "label": "Laudos (Diário)",
            "description": "Registros diários de laudos", 
            "pattern": ["laudos_diario", "laudos_diário", "laudos diário"]
        },
        "detalhes_laudospendentes": {
            "label": "Laudos Pendentes",
            "description": "Detalhes de laudos aguardando conclusão",
            "pattern": ["laudospendentes", "laudos_pendentes", "detalhes_laudospendentes"]
        },
        "detalhes_examespendentes": {
            "label": "Exames Pendentes", 
            "description": "Detalhes de exames aguardando realização",
            "pattern": ["examespendentes", "exames_pendentes", "detalhes_examespendentes"]
        }
    }

def resolve_file_path(name: str, file_configs: Dict) -> Optional[str]:
    """Resolve caminho do arquivo com tolerância a variações de nome"""
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
            pattern_normalized = re.sub(r"[^\w]", "_", pattern)
            if pattern_normalized in normalized_name or normalized_name.startswith(pattern_normalized):
                return os.path.join("data", filename)
    
    return None

# ============ INTERFACE DE UPLOAD E CARREGAMENTO ============
st.sidebar.header("📁 Dados do Sistema")
has_data_dir = detect_data_sources()
file_configs = get_file_configs()

if not has_data_dir:
    st.sidebar.info("💡 Upload dos arquivos CSV para análise")

# Upload de arquivos
uploads = {}
for key, config in file_configs.items():
    if not has_data_dir:
        uploads[key] = st.sidebar.file_uploader(
            f"{config['label']} (.csv)",
            help=config['description'],
            key=f"upload_{key}",
            type=['csv']
        )
    else:
        uploads[key] = None

@st.cache_data(ttl=3600, show_spinner="Carregando dados...")
def load_all_data_enhanced(file_sources: Dict) -> Dict[str, pd.DataFrame]:
    """Carregamento otimizado de todos os dados"""
    loaded_data = {}
    loading_stats = []
    
    for name, upload_file in file_sources.items():
        df = None
        
        # Carregamento de pasta local ou upload
        if has_data_dir:
            file_path = resolve_file_path(name, file_configs)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    df = read_csv_enhanced(content, name)
                    
                    if df is not None:
                        loading_stats.append(f"✅ {name}: {len(df):,} registros".replace(",", "."))
                except Exception as e:
                    loading_stats.append(f"❌ {name}: Erro - {str(e)}")
        else:
            if upload_file is not None:
                try:
                    content = upload_file.read()
                    df = read_csv_enhanced(content, name)
                    
                    if df is not None:
                        loading_stats.append(f"✅ {name}: {len(df):,} registros".replace(",", "."))
                except Exception as e:
                    loading_stats.append(f"❌ {name}: Erro - {str(e)}")
        
        # Padronização dos dados carregados
        if df is not None:
            standardized_df = standardize_dataframe_enhanced(name, df)
            if not standardized_df.empty:
                loaded_data[name] = standardized_df
    
    # Exibir estatísticas de carregamento
    for stat in loading_stats:
        if "✅" in stat:
            st.sidebar.success(stat)
        else:
            st.sidebar.error(stat)
    
    return loaded_data

# Carregamento dos dados
with st.spinner("Processando e padronizando dados..."):
    dataframes = load_all_data_enhanced(uploads)

if not dataframes:
    st.warning("⚠️ Nenhum arquivo foi carregado com sucesso.")
    st.info("📝 **Arquivos esperados:** " + ", ".join(file_configs.keys()))
    st.info("🔧 **Formatos suportados:** CSV com separadores `;`, `,`, `|` ou tab")
    st.stop()

# ============ FILTROS APRIMORADOS ============
def extract_filter_values_enhanced(column: str) -> List[str]:
    """Extração aprimorada de valores únicos para filtros"""
    values = set()
    for df in dataframes.values():
        if column in df.columns:
            unique_vals = df[column].dropna().astype(str).unique()
            values.update(v for v in unique_vals if v and v.lower() not in ["nan", "none", ""])
    return sorted(list(values))

def apply_filters_enhanced(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Aplicação otimizada de filtros"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    # Filtros dimensionais
    for column, filter_values in filters.get("dimensions", {}).items():
        if column in filtered.columns and filter_values:
            filtered = filtered[filtered[column].astype(str).isin(filter_values)]
    
    # Filtro temporal
    period_filter = filters.get("period")
    if "anomês_dt" in filtered.columns and period_filter != "Todo o período":
        max_date = filtered["anomês_dt"].max()
        if pd.notna(max_date):
            cutoff_map = {
                "Últimos 3 meses": pd.DateOffset(months=3),
                "Últimos 6 meses": pd.DateOffset(months=6), 
                "Último ano": pd.DateOffset(years=1),
                "Ano atual": None
            }
            
            if period_filter == "Ano atual":
                cutoff_date = pd.Timestamp(max_date.year, 1, 1)
            else:
                offset = cutoff_map.get(period_filter)
                cutoff_date = max_date - offset if offset else None
            
            if cutoff_date is not None:
                filtered = filtered[filtered["anomês_dt"] >= cutoff_date]
    
    return filtered

# Configuração de filtros na sidebar
st.sidebar.subheader("🔍 Filtros de Análise")

# Filtros dimensionais
dimensional_filters = {}
dimensional_filters["diretoria"] = st.sidebar.multiselect(
    "🏢 Diretoria", 
    extract_filter_values_enhanced("diretoria"),
    help="Filtrar por diretoria específica"
)
dimensional_filters["superintendencia"] = st.sidebar.multiselect(
    "🏛️ Superintendência", 
    extract_filter_values_enhanced("superintendencia"),
    help="Filtrar por superintendência"
)
dimensional_filters["unidade"] = st.sidebar.multiselect(
    "🏪 Unidade", 
    extract_filter_values_enhanced("unidade"),
    help="Filtrar por unidade operacional"
)
dimensional_filters["tipo"] = st.sidebar.multiselect(
    "🔬 Tipo de Perícia", 
    extract_filter_values_enhanced("tipo"),
    help="Filtrar por tipo de perícia"
)

# Filtro temporal
period_options = ["Todo o período", "Ano atual", "Últimos 6 meses", "Últimos 3 meses"]
period_filter = st.sidebar.selectbox(
    "📅 Período de Análise", 
    period_options,
    help="Selecionar período temporal para análise"
)

# Configurações de visualização
st.sidebar.subheader("⚙️ Configurações")
show_benchmarks = st.sidebar.toggle(
    "📊 Exibir Metas", 
    value=True,
    help="Mostrar linhas de referência e metas nos gráficos"
)
chart_height = st.sidebar.slider(
    "📏 Altura dos Gráficos", 
    min_value=300, 
    max_value=600, 
    value=400,
    help="Ajustar altura padrão dos gráficos"
)

# Consolidação dos filtros
all_filters = {
    "dimensions": dimensional_filters,
    "period": period_filter
}

# Aplicação dos filtros
filtered_dataframes = {
    name: apply_filters_enhanced(df, all_filters) 
    for name, df in dataframes.items()
}

# Atalhos para datasets principais
df_atend_todos = filtered_dataframes.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dataframes.get("Laudos_todos_Mensal")
df_atend_esp = filtered_dataframes.get("Atendimentos_especifico_Mensal")
df_laudos_esp = filtered_dataframes.get("Laudos_especifico_Mensal")
df_atend_diario = filtered_dataframes.get("Atendimentos_diario")
df_laudos_diario = filtered_dataframes.get("Laudos_diario")
df_pend_laudos = filtered_dataframes.get("detalhes_laudospendentes")
df_pend_exames = filtered_dataframes.get("detalhes_examespendentes")

# ============ CÁLCULOS DE KPIS APRIMORADOS ============
class KPICalculator:
    """Classe para cálculos padronizados de KPIs"""
    
    @staticmethod
    def calculate_total(df: pd.DataFrame) -> int:
        """Calcula total de registros/quantidade"""
        if df is None or df.empty or "quantidade" not in df.columns:
            return 0
        return int(df["quantidade"].sum())
    
    @staticmethod
    def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
        """Calcula média mensal"""
        if df is None or df.empty or "anomês_dt" not in df.columns:
            return None
        monthly_totals = df.groupby("anomês_dt")["quantidade"].sum()
        return monthly_totals.mean() if len(monthly_totals) > 0 else None
    
    @staticmethod
    def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
        """Calcula taxa de crescimento entre períodos"""
        if df is None or df.empty or "anomês_dt" not in df.columns:
            return None
        
        monthly_data = df.groupby("anomês_dt")["quantidade"].sum().sort_index()
        if len(monthly_data) < periods * 2:
            return None
        
        recent_data = monthly_data.tail(periods * 2)
        mid_point = len(recent_data) // 2
        first_half = recent_data.iloc[:mid_point].mean()
        second_half = recent_data.iloc[mid_point:].mean()
        
        if first_half > 0:
            return ((second_half - first_half) / first_half) * 100
        return None
    
    @staticmethod
    def calculate_conversion_rate(df_input: pd.DataFrame, df_output: pd.DataFrame) -> Optional[float]:
        """Calcula taxa de conversão entre dois datasets"""
        if df_input is None or df_output is None:
            return None
        
        total_input = KPICalculator.calculate_total(df_input)
        total_output = KPICalculator.calculate_total(df_output)
        
        return calculate_percentage(total_output, total_input)
    
    @staticmethod
    def calculate_aging_stats(df: pd.DataFrame, date_column: str = "data_base") -> Dict:
        """Calcula estatísticas de aging para pendências"""
        if df is None or df.empty or date_column not in df.columns:
            return {}
        
        dates = pd.to_datetime(df[date_column], errors="coerce")
        if dates.isna().all():
            return {}
        
        hoje = pd.Timestamp.now().normalize()
        aging_days = (hoje - dates).dt.days
        
        return {
            "total": len(df),
            "media_dias": float(aging_days.mean()),
            "mediana_dias": float(aging_days.median()),
            "max_dias": int(aging_days.max()),
            "p75_dias": float(aging_days.quantile(0.75)),
            "p90_dias": float(aging_days.quantile(0.90))
        }

# Cálculo dos KPIs principais
calc = KPICalculator()

# KPIs de Produção
total_atendimentos = calc.calculate_total(df_atend_todos)
total_laudos = calc.calculate_total(df_laudos_todos)
media_mensal_atendimentos = calc.calculate_monthly_average(df_atend_todos)
media_mensal_laudos = calc.calculate_monthly_average(df_laudos_todos)
taxa_conversao = calc.calculate_conversion_rate(df_atend_todos, df_laudos_todos)
crescimento_atendimentos = calc.calculate_growth_rate(df_atend_todos)
crescimento_laudos = calc.calculate_growth_rate(df_laudos_todos)

# KPIs de Pendências
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None else 0
aging_laudos = calc.calculate_aging_stats(df_pend_laudos)
aging_exames = calc.calculate_aging_stats(df_pend_exames)

# Estimativa de backlog
backlog_meses = None
if media_mensal_laudos and media_mensal_laudos > 0:
    backlog_meses = total_pend_laudos / media_mensal_laudos

# ============ FILTROS RÁPIDOS ============
st.markdown("<h4 class='section-title'>🎛️ Filtros Rápidos</h4>", unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns([0.3, 0.25, 0.25, 0.2])

with fc1:
    quick_period = segment(
        "📅 Período", 
        ["Todo o período", "Ano atual", "Últimos 6 meses", "Últimos 3 meses"],
        default=period_filter, 
        key="quick_period"
    )

with fc2:
    view_mode = segment(
        "👁️ Visualização", 
        ["Resumo", "Detalhado", "Comparativo"],
        default="Resumo", 
        key="view_mode"
    )

with fc3:
    analysis_focus = segment(
        "🎯 Foco", 
        ["Produção", "Pendências", "Performance"],
        default="Produção", 
        key="analysis_focus"
    )

with fc4:
    export_format = st.selectbox(
        "📥 Exportar",
        ["Nenhum", "PDF", "Excel", "CSV"],
        key="export_format"
    )

# ============ DASHBOARD PRINCIPAL - KPIS ============
st.markdown("<h4 class='section-title'>📈 Indicadores Principais</h4>", unsafe_allow_html=True)

# Linha 1 - Produção
c1, c2, c3, c4 = st.columns(4)
with c1:
    delta_atend = f"↗️ {format_number(crescimento_atendimentos,1)}%" if crescimento_atendimentos and crescimento_atendimentos > 0 else f"↘️ {format_number(abs(crescimento_atendimentos or 0),1)}%" if crescimento_atendimentos else None
    kpi_card("Atendimentos Totais", format_number(total_atendimentos), delta_atend)

with c2:
    delta_laudos = f"↗️ {format_number(crescimento_laudos,1)}%" if crescimento_laudos and crescimento_laudos > 0 else f"↘️ {format_number(abs(crescimento_laudos or 0),1)}%" if crescimento_laudos else None
    kpi_card("Laudos Emitidos", format_number(total_laudos), delta_laudos)

with c3:
    taxa_str = f"{format_number(taxa_conversao,1)}%" if taxa_conversao else "—"
    taxa_color = "🟢" if taxa_conversao and taxa_conversao >= 70 else "🟡" if taxa_conversao and taxa_conversao >= 50 else "🔴"
    kpi_card("Taxa de Conversão", f"{taxa_color} {taxa_str}")

with c4:
    prod_str = f"{format_number(media_mensal_laudos,0)}" if media_mensal_laudos else "—"
    kpi_card("Produtividade Mensal", f"{prod_str} laudos/mês")

# Linha 2 - Pendências e Performance
st.markdown("<h4 class='section-title'>⏰ Gestão de Pendências</h4>", unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)

with c5:
    pend_color = "🔴" if total_pend_laudos > 1000 else "🟡" if total_pend_laudos > 500 else "🟢"
    kpi_card("Laudos Pendentes", f"{pend_color} {format_number(total_pend_laudos)}")

with c6:
    exam_color = "🔴" if total_pend_exames > 2000 else "🟡" if total_pend_exames > 1000 else "🟢"
    kpi_card("Exames Pendentes", f"{exam_color} {format_number(total_pend_exames)}")

with c7:
    backlog_str = f"{format_number(backlog_meses,1)} meses" if backlog_meses else "—"
    backlog_color = "🔴" if backlog_meses and backlog_meses > 6 else "🟡" if backlog_meses and backlog_meses > 3 else "🟢"
    kpi_card("Backlog Estimado", f"{backlog_color} {backlog_str}")

with c8:
    aging_medio = aging_laudos.get("media_dias") or aging_exames.get("media_dias")
    aging_str = f"{format_number(aging_medio,0)} dias" if aging_medio else "—"
    aging_color = "🔴" if aging_medio and aging_medio > 90 else "🟡" if aging_medio and aging_medio > 60 else "🟢"
    kpi_card("Aging Médio", f"{aging_color} {aging_str}")

# ============ ALERTAS INTELIGENTES ============
st.markdown("<h4 class='section-title'>🚨 Alertas e Insights</h4>", unsafe_allow_html=True)

def generate_smart_alerts() -> List[Dict]:
    """Gera alertas inteligentes baseados nos KPIs"""
    alerts = []
    
    # Alertas críticos
    if backlog_meses and backlog_meses > 6:
        alerts.append({
            "type": "danger",
            "icon": "🔴",
            "title": "BACKLOG CRÍTICO",
            "message": f"Backlog de {format_number(backlog_meses,1)} meses excede limite seguro (6 meses)"
        })
    
    if taxa_conversao and taxa_conversao < 50:
        alerts.append({
            "type": "danger", 
            "icon": "🔴",
            "title": "EFICIÊNCIA BAIXA",
            "message": f"Taxa de conversão de {format_number(taxa_conversao,1)}% abaixo do mínimo (50%)"
        })
    
    # Alertas de atenção
    if crescimento_laudos and crescimento_laudos < -10:
        alerts.append({
            "type": "warning",
            "icon": "🟡", 
            "title": "QUEDA NA PRODUÇÃO",
            "message": f"Redução de {format_number(abs(crescimento_laudos),1)}% na emissão de laudos"
        })
    
    if backlog_meses and 3 < backlog_meses <= 6:
        alerts.append({
            "type": "warning",
            "icon": "🟡",
            "title": "BACKLOG ELEVADO", 
            "message": f"Backlog de {format_number(backlog_meses,1)} meses requer atenção"
        })
    
    # Alertas positivos
    if crescimento_laudos and crescimento_laudos > 10:
        alerts.append({
            "type": "success",
            "icon": "🟢",
            "title": "CRESCIMENTO POSITIVO",
            "message": f"Aumento de {format_number(crescimento_laudos,1)}% na produção de laudos"
        })
    
    if taxa_conversao and taxa_conversao >= 80:
        alerts.append({
            "type": "success",
            "icon": "🟢", 
            "title": "ALTA EFICIÊNCIA",
            "message": f"Taxa de conversão de {format_number(taxa_conversao,1)}% acima da meta"
        })
    
    return alerts

alerts = generate_smart_alerts()

if alerts:
    # Organizar alertas por tipo
    alert_cols = st.columns(len(alerts))
    for i, alert in enumerate(alerts):
        with alert_cols[i]:
            css_class = f"alert-{alert['type']}"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{alert['icon']} {alert['title']}</strong><br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-success">
        <strong>✅ SITUAÇÃO NORMAL</strong><br>
        Todos os indicadores estão dentro dos parâmetros esperados
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============ NAVEGAÇÃO POR ABAS ============
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Visão Geral",
    "📈 Tendências",
    "🏆 Rankings", 
    "⏰ Pendências",
    "📅 Análise Diária",
    "📋 Dados Brutos",
    "📑 Relatórios"
])

# ============ ABA 1: VISÃO GERAL ============
with tab1:
    st.subheader("📊 Panorama Executivo")
    
    # Gráfico principal - Evolução temporal
    if df_atend_todos is not None and df_laudos_todos is not None:
        col_chart1, col_chart2 = st.columns([0.7, 0.3])
        
        with col_chart1:
            st.markdown("#### 📈 Evolução Mensal: Atendimentos vs Laudos")
            
            # Preparação dos dados
            atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
            laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
            
            # Gráfico combinado
            fig_evolution = go.Figure()
            
            fig_evolution.add_trace(go.Scatter(
                x=atend_monthly["anomês_dt"],
                y=atend_monthly["quantidade"], 
                mode='lines+markers',
                name='Atendimentos',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ))
            
            fig_evolution.add_trace(go.Scatter(
                x=laudos_monthly["anomês_dt"],
                y=laudos_monthly["quantidade"],
                mode='lines+markers', 
                name='Laudos',
                line=dict(color='#10b981', width=3),
                marker=dict(size=6)
            ))
            
            # Linhas de tendência se solicitado
            if show_benchmarks and len(atend_monthly) > 3:
                # Média móvel simples
                atend_monthly['ma3'] = atend_monthly['quantidade'].rolling(3).mean()
                laudos_monthly['ma3'] = laudos_monthly['quantidade'].rolling(3).mean()
                
                fig_evolution.add_trace(go.Scatter(
                    x=atend_monthly["anomês_dt"],
                    y=atend_monthly["ma3"],
                    mode='lines',
                    name='Tend. Atendimentos',
                    line=dict(color='#3b82f6', width=2, dash='dash'),
                    showlegend=False
                ))
                
                fig_evolution.add_trace(go.Scatter(
                    x=laudos_monthly["anomês_dt"], 
                    y=laudos_monthly["ma3"],
                    mode='lines',
                    name='Tend. Laudos',
                    line=dict(color='#10b981', width=2, dash='dash'),
                    showlegend=False
                ))
            
            fig_evolution.update_layout(
                height=chart_height,
                hovermode='x unified',
                xaxis_title="Período",
                yaxis_title="Quantidade",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### 🎯 Taxa de Conversão")
            
            # Cálculo da taxa de conversão mensal
            merged_monthly = pd.merge(
                atend_monthly.rename(columns={"quantidade": "Atendimentos"}),
                laudos_monthly.rename(columns={"quantidade": "Laudos"}),
                on="anomês_dt",
                how="inner"
            )
            
            if not merged_monthly.empty:
                merged_monthly["Taxa_Conversao"] = (merged_monthly["Laudos"] / merged_monthly["Atendimentos"]) * 100
                
                fig_conversion = go.Figure()
                fig_conversion.add_trace(go.Scatter(
                    x=merged_monthly["anomês_dt"],
                    y=merged_monthly["Taxa_Conversao"],
                    mode='lines+markers',
                    line=dict(color='#f59e0b', width=3),
                    marker=dict(size=8),
                    name='Taxa de Conversão'
                ))
                
                if show_benchmarks:
                    fig_conversion.add_hline(
                        y=70, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Meta: 70%"
                    )
                
                fig_conversion.update_layout(
                    height=chart_height,
                    xaxis_title="Período",
                    yaxis_title="Taxa (%)",
                    yaxis=dict(range=[0, 100]),
                    showlegend=False
                )
                
                st.plotly_chart(fig_conversion, use_container_width=True)
    
    # Análises por dimensão
    col_dim1, col_dim2 = st.columns(2)
    
    with col_dim1:
        st.markdown("#### 🏢 Performance por Unidade")
        if df_laudos_todos is not None and "unidade" in df_laudos_todos.columns:
            unidade_summary = (
                df_laudos_todos.groupby("unidade")["quantidade"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            
            fig_unidades = px.bar(
                unidade_summary,
                x="quantidade", 
                y="unidade",
                orientation="h",
                title="Top 15 Unidades - Laudos Emitidos",
                color="quantidade",
                color_continuous_scale="Blues"
            )
            
            fig_unidades.update_layout(
                height=500,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_unidades, use_container_width=True)
    
    with col_dim2:
        st.markdown("#### 🔍 Análise Pareto - Tipos de Perícia")
        if df_laudos_esp is not None and "tipo" in df_laudos_esp.columns:
            tipo_summary = (
                df_laudos_esp.groupby("tipo")["quantidade"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            
            tipo_summary["pct"] = 100 * tipo_summary["quantidade"] / tipo_summary["quantidade"].sum()
            tipo_summary["pct_acum"] = tipo_summary["pct"].cumsum()
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_pareto.add_trace(
                go.Bar(
                    x=tipo_summary["tipo"].head(10),
                    y=tipo_summary["quantidade"].head(10),
                    name="Quantidade",
                    marker_color='lightblue'
                )
            )
            
            fig_pareto.add_trace(
                go.Scatter(
                    x=tipo_summary["tipo"].head(10),
                    y=tipo_summary["pct_acum"].head(10),
                    mode="lines+markers",
                    name="% Acumulado",
                    line=dict(color='red', width=3),
                    marker=dict(size=6)
                ),
                secondary_y=True,
            )
            
            if show_benchmarks:
                fig_pareto.add_hline(
                    y=80, 
                    line_dash="dash", 
                    line_color="red",
                    secondary_y=True,
                    annotation_text="80%"
                )
            
            fig_pareto.update_layout(
                title="Top 10 Tipos de Perícia",
                height=500,
                hovermode="x unified"
            )
            fig_pareto.update_yaxes(title_text="Quantidade", secondary_y=False)
            fig_pareto.update_yaxes(title_text="% Acumulado", range=[0, 100], secondary_y=True)
            
            st.plotly_chart(fig_pareto, use_container_width=True)

# ============ ABA 2: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise de Tendências Avançada")
    
    def create_advanced_time_series(df: pd.DataFrame, title: str, color: str = "blue"):
        """Cria série temporal avançada com decomposição"""
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        monthly_data = df.groupby("anomês_dt")["quantidade"].sum().sort_index()
        if len(monthly_data) < 3:
            st.info(f"Período insuficiente para análise de tendência: {title}")
            return
        
        # Preparação dos dados
        dates = monthly_data.index
        values = monthly_data.values
        
        # Cálculos estatísticos
        ma3 = monthly_data.rolling(window=3, center=True).mean()
        pct_change = monthly_data.pct_change() * 100
        
        # Detecção de tendência (regressão linear simples)
        try:
            from scipy import stats
            x_numeric = np.arange(len(monthly_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, values)
            trend_line = slope * x_numeric + intercept
        except ImportError:
            # Fallback se scipy não estiver disponível
            slope = np.polyfit(range(len(values)), values, 1)[0]
            intercept = values.mean() - slope * np.mean(range(len(values)))
            trend_line = slope * np.arange(len(values)) + intercept
            r_value = np.corrcoef(range(len(values)), values)[0, 1]
        
        # Criação do gráfico
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"{title} - Série Temporal",
                "Variação Percentual Mensal", 
                "Tendência e Sazonalidade"
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Série principal
        fig.add_trace(
            go.Scatter(
                x=dates, y=values,
                mode="lines+markers",
                name="Valores Observados",
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Média móvel
        fig.add_trace(
            go.Scatter(
                x=dates, y=ma3,
                mode="lines",
                name="Média Móvel (3m)",
                line=dict(color="red", width=2, dash="dash")
            ),
            row=1, col=1
        )
        
        # Linha de tendência
        fig.add_trace(
            go.Scatter(
                x=dates, y=trend_line,
                mode="lines",
                name=f"Tendência (R²={r_value**2:.3f})",
                line=dict(color="orange", width=2, dash="dot")
            ),
            row=1, col=1
        )
        
        # Variação percentual
        colors = ['red' if x < 0 else 'green' for x in pct_change.fillna(0)]
        fig.add_trace(
            go.Bar(
                x=dates, y=pct_change,
                name="Variação %",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Análise sazonal (se houver dados suficientes)
        if len(monthly_data) >= 12:
            seasonal_pattern = monthly_data.groupby(monthly_data.index.month).mean()
            months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                     'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            
            fig.add_trace(
                go.Bar(
                    x=months[:len(seasonal_pattern)], 
                    y=seasonal_pattern.values,
                    name="Padrão Sazonal",
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Layout e configurações
        fig.update_layout(
            height=600,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Títulos dos eixos
        fig.update_xaxes(title_text="Período", row=3, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        fig.update_yaxes(title_text="Média Sazonal", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de tendência
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            trend_direction = "↗️ Crescente" if slope > 0 else "↘️ Decrescente" if slope < 0 else "→ Estável"
            st.metric("Tendência", trend_direction)
        with col2:
            st.metric("Correlação", f"{r_value:.3f}")
        with col3:
            volatility = pct_change.std()
            st.metric("Volatilidade", f"{volatility:.1f}%")
        with col4:
            last_change = pct_change.iloc[-1] if not pct_change.empty else 0
            st.metric("Última Variação", f"{last_change:.1f}%")
    
    # Análises de tendência por dataset
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        create_advanced_time_series(df_atend_todos, "Atendimentos", "#3b82f6")
    
    with trend_col2:
        create_advanced_time_series(df_laudos_todos, "Laudos", "#10b981")
    
    # Análise de correlação cruzada
    if df_atend_todos is not None and df_laudos_todos is not None:
        st.markdown("#### 🔗 Análise de Correlação Cruzada")
        
        atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum()
        laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum()
        common_periods = atend_monthly.index.intersection(laudos_monthly.index)
        
        if len(common_periods) > 3:
            correlation_data = pd.DataFrame({
                "Atendimentos": atend_monthly.loc[common_periods],
                "Laudos": laudos_monthly.loc[common_periods]
            }).reset_index()
            
            correlation_coef = correlation_data["Atendimentos"].corr(correlation_data["Laudos"])
            
            fig_correlation = px.scatter(
                correlation_data,
                x="Atendimentos", 
                y="Laudos",
                trendline="ols",
                title=f"Correlação: Atendimentos vs Laudos (r = {correlation_coef:.3f})",
                hover_data=["anomês_dt"]
            )
            
            fig_correlation.update_layout(height=400)
            st.plotly_chart(fig_correlation, use_container_width=True)
            
            # Interpretação da correlação
            if correlation_coef > 0.8:
                st.success(f"🟢 **Correlação Forte** ({correlation_coef:.3f}): Atendimentos e laudos estão bem alinhados")
            elif correlation_coef > 0.5:
                st.warning(f"🟡 **Correlação Moderada** ({correlation_coef:.3f}): Algum desalinhamento entre atendimentos e laudos")
            else:
                st.error(f"🔴 **Correlação Fraca** ({correlation_coef:.3f}): Atendimentos e laudos não estão alinhados")

# ============ ABA 3: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Análises Comparativas")
    
    def create_comprehensive_ranking(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20):
        """Cria ranking abrangente com múltiplas métricas"""
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        # Agregação de dados
        ranking_data = df.groupby(dimension).agg({
            "quantidade": ["sum", "count", "mean", "std"]
        }).round(2)
        
        ranking_data.columns = ["Total", "Registros", "Média", "Desvio"]
        ranking_data = ranking_data.fillna(0)
        
        # Cálculos adicionais
        ranking_data["Coef_Variacao"] = (ranking_data["Desvio"] / ranking_data["Média"]).replace([np.inf, -np.inf], 0)
        ranking_data["Percentual"] = (ranking_data["Total"] / ranking_data["Total"].sum()) * 100
        ranking_data["Percentual_Acum"] = ranking_data.sort_values("Total", ascending=False)["Percentual"].cumsum()
        
        # Top N
        top_ranking = ranking_data.sort_values("Total", ascending=False).head(top_n).reset_index()
        
        if top_ranking.empty:
            st.info(f"Sem dados para {title}")
            return
        
        # Gráfico principal
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Ranking por Volume", "Distribuição Percentual"),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.1
        )
        
        # Gráfico de barras
        fig.add_trace(
            go.Bar(
                y=top_ranking[dimension],
                x=top_ranking["Total"],
                orientation="h",
                name="Total",
                marker=dict(
                    color=top_ranking["Total"],
                    colorscale="Viridis",
                    showscale=True
                ),
                text=top_ranking["Total"],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        # Gráfico de pizza (top 10)
        fig.add_trace(
            go.Pie(
                labels=top_ranking[dimension].head(10),
                values=top_ranking["Percentual"].head(10),
                name="Distribuição"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text=title
        )
        
        fig.update_yaxes(categoryorder="total ascending", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela detalhada
        with st.expander(f"📊 Dados Detalhados - {title}", expanded=False):
            # Formatação da tabela
            display_df = top_ranking.copy()
            display_df["Total"] = display_df["Total"].apply(lambda x: format_number(x))
            display_df["Média"] = display_df["Média"].apply(lambda x: format_number(x, 1))
            display_df["Percentual"] = display_df["Percentual"].apply(lambda x: f"{x:.1f}%")
            display_df["Coef_Variacao"] = display_df["Coef_Variacao"].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
    
    # Tabs de rankings
    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "🏢 Por Diretoria", 
        "🏪 Por Unidade", 
        "🔬 Por Tipo", 
        "📊 Análise Comparativa"
    ])
    
    with rank_tab1:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(df_atend_todos, "diretoria", "Atendimentos por Diretoria")
        with col2:
            create_comprehensive_ranking(df_laudos_todos, "diretoria", "Laudos por Diretoria")
    
    with rank_tab2:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(df_atend_todos, "unidade", "Atendimentos por Unidade", 25)
        with col2:
            create_comprehensive_ranking(df_laudos_todos, "unidade", "Laudos por Unidade", 25)
    
    with rank_tab3:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(df_atend_esp, "tipo", "Atendimentos por Tipo", 20)
        with col2:
            create_comprehensive_ranking(df_laudos_esp, "tipo", "Laudos por Tipo", 20)
    
    with rank_tab4:
        st.markdown("#### 📊 Matriz de Eficiência: Atendimentos vs Laudos")
        
        if df_atend_todos is not None and df_laudos_todos is not None:
            # Análise por unidade
            if "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns:
                atend_unidade = df_atend_todos.groupby("unidade")["quantidade"].sum()
                laudos_unidade = df_laudos_todos.groupby("unidade")["quantidade"].sum()
                
                efficiency_data = pd.DataFrame({
                    "Atendimentos": atend_unidade,
                    "Laudos": laudos_unidade
                }).fillna(0)
                
                efficiency_data["Taxa_Conversao"] = (
                    efficiency_data["Laudos"] / efficiency_data["Atendimentos"] * 100
                ).replace([np.inf, -np.inf], 0)
                
                efficiency_data["Eficiencia_Score"] = (
                    efficiency_data["Taxa_Conversao"] * 0.7 + 
                    (efficiency_data["Laudos"] / efficiency_data["Laudos"].max()) * 30
                )
                
                # Classificação em quadrantes
                mediana_atend = efficiency_data["Atendimentos"].median()
                mediana_laudos = efficiency_data["Laudos"].median()
                
                def classify_quadrant(row):
                    if row["Atendimentos"] >= mediana_atend and row["Laudos"] >= mediana_laudos:
                        return "⭐ Alto Volume/Alta Produção"
                    elif row["Atendimentos"] >= mediana_atend and row["Laudos"] < mediana_laudos:
                        return "🔄 Alto Volume/Baixa Produção"
                    elif row["Atendimentos"] < mediana_atend and row["Laudos"] >= mediana_laudos:
                        return "🎯 Baixo Volume/Alta Eficiência"
                    else:
                        return "📉 Baixo Volume/Baixa Produção"
                
                efficiency_data["Quadrante"] = efficiency_data.apply(classify_quadrant, axis=1)
                
                # Gráfico de dispersão
                fig_efficiency = px.scatter(
                    efficiency_data.reset_index(),
                    x="Atendimentos",
                    y="Laudos",
                    size="Taxa_Conversao",
                    color="Quadrante",
                    hover_name="unidade",
                    title="Matriz de Eficiência por Unidade",
                    size_max=20
                )
                
                # Linhas de referência
                fig_efficiency.add_vline(
                    x=mediana_atend, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Mediana Atendimentos"
                )
                fig_efficiency.add_hline(
                    y=mediana_laudos, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Mediana Laudos"
                )
                
                fig_efficiency.update_layout(height=500)
                st.plotly_chart(fig_efficiency, use_container_width=True)
                
                # Top performers
                st.markdown("**🏆 Top 10 Unidades Mais Eficientes:**")
                top_efficient = efficiency_data.sort_values("Eficiencia_Score", ascending=False).head(10)
                top_efficient_display = top_efficient.reset_index()
                top_efficient_display["Taxa_Conversao"] = top_efficient_display["Taxa_Conversao"].apply(lambda x: f"{x:.1f}%")
                top_efficient_display["Eficiencia_Score"] = top_efficient_display["Eficiencia_Score"].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(
                    top_efficient_display[["unidade", "Atendimentos", "Laudos", "Taxa_Conversao", "Quadrante"]],
                    use_container_width=True
                )

# ============ ABA 4: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Gestão Avançada de Pendências")
    
    def analyze_aging_comprehensive(df: pd.DataFrame, title: str, date_column: str = "data_base"):
        """Análise abrangente de aging"""
        if df is None or df.empty:
            st.info(f"Sem dados de {title}")
            return None, {}
        
        # Buscar coluna de data disponível
        date_cols = [col for col in df.columns if "data" in col.lower()]
        if date_column not in df.columns and date_cols:
            date_column = date_cols[0]
        
        if date_column not in df.columns:
            st.warning(f"Coluna de data não encontrada para {title}")
            return None, {}
        
        # Processamento de aging
        dates = pd.to_datetime(df[date_column], errors="coerce")
        if dates.isna().all():
            st.warning(f"Datas inválidas em {title}")
            return None, {}
        
        hoje = pd.Timestamp.now().normalize()
        aging_days = (hoje - dates).dt.days
        
        # Classificação de aging
        aging_ranges = [
            (0, 15, "0-15 dias", "🟢"),
            (16, 30, "16-30 dias", "🟡"),
            (31, 60, "31-60 dias", "🟠"),
            (61, 90, "61-90 dias", "🔴"),
            (91, 180, "91-180 dias", "🔴"),
            (181, 365, "181-365 dias", "⚫"),
            (366, float('inf'), "> 365 dias", "⚫")
        ]
        
        def classify_aging(days):
            for min_days, max_days, label, color in aging_ranges:
                if min_days <= days <= max_days:
                    return label, color
            return "Indefinido", "⚪"
        
        aging_classifications = aging_days.apply(lambda x: classify_aging(x) if pd.notna(x) else ("Indefinido", "⚪"))
        df_analysis = df.copy()
        df_analysis["dias_pendentes"] = aging_days
        df_analysis["faixa_aging"] = [item[0] for item in aging_classifications]
        df_analysis["cor_aging"] = [item[1] for item in aging_classifications]
        
        # Estatísticas
        stats = {
            "total": len(df_analysis),
            "media_dias": float(aging_days.mean()),
            "mediana_dias": float(aging_days.median()),
            "max_dias": int(aging_days.max()) if not aging_days.empty else 0,
            "p90_dias": float(aging_days.quantile(0.9)),
            "criticos": int((aging_days > 90).sum()),
            "urgentes": int((aging_days > 60).sum())
        }
        
        # Layout em colunas
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.markdown(f"#### 📊 {title} - Distribuição de Aging")
            
            # Distribuição por faixa
            aging_dist = df_analysis["faixa_aging"].value_counts()
            aging_dist = aging_dist.reindex([label for _, _, label, _ in aging_ranges if label in aging_dist.index])
            
            fig_aging = px.bar(
                x=aging_dist.index,
                y=aging_dist.values,
                title=f"Distribuição de {title}",
                color=aging_dist.values,
                color_continuous_scale="Reds"
            )
            
            fig_aging.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Faixa de Aging",
                yaxis_title="Quantidade"
            )
            
            st.plotly_chart(fig_aging, use_container_width=True)
        
        with col2:
            st.markdown(f"#### 📈 Estatísticas de {title}")
            
            # Cards de estatísticas
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Total", format_number(stats["total"]))
                st.metric("Críticos (>90d)", format_number(stats["criticos"]))
                st.metric("Média (dias)", format_number(stats["media_dias"], 1))
            
            with stat_col2:
                st.metric("Máximo (dias)", format_number(stats["max_dias"]))
                st.metric("P90 (dias)", format_number(stats["p90_dias"], 1))
                st.metric("Mediana (dias)", format_number(stats["mediana_dias"], 1))
            
            # Gráfico de pizza - prioridades
            prioridade_map = {
                "Normal": aging_days <= 30,
                "Atenção": (aging_days > 30) & (aging_days <= 60),
                "Urgente": (aging_days > 60) & (aging_days <= 90),
                "Crítico": aging_days > 90
            }
            
            prioridade_counts = {k: v.sum() for k, v in prioridade_map.items()}
            
            fig_priority = px.pie(
                values=list(prioridade_counts.values()),
                names=list(prioridade_counts.keys()),
                title="Distribuição por Prioridade",
                color_discrete_map={
                    "Normal": "green",
                    "Atenção": "yellow", 
                    "Urgente": "orange",
                    "Crítico": "red"
                }
            )
            
            fig_priority.update_layout(height=300)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Análise por dimensões
        if "unidade" in df_analysis.columns:
            st.markdown(f"#### 🏢 {title} por Unidade")
            
            unidade_aging = df_analysis.groupby("unidade").agg({
                "dias_pendentes": ["count", "mean", "max"],
                "faixa_aging": lambda x: (x.isin(["61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"])).sum()
            }).round(1)
            
            unidade_aging.columns = ["Total", "Média_Dias", "Max_Dias", "Críticos"]
            unidade_aging = unidade_aging.sort_values("Críticos", ascending=False).head(15)
            
            fig_unidade = px.bar(
                unidade_aging.reset_index(),
                x="unidade",
                y="Críticos",
                title=f"Top 15 Unidades - {title} Críticos",
                color="Média_Dias",
                color_continuous_scale="Reds"
            )
            
            fig_unidade.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_unidade, use_container_width=True)
        
        # Top casos mais antigos
        st.markdown(f"**🔴 Top 20 {title} Mais Antigos:**")
        oldest_cases = df_analysis.nlargest(20, "dias_pendentes")
        
        display_cols = []
        if "id" in oldest_cases.columns:
            display_cols.append("id")
        if "unidade" in oldest_cases.columns:
            display_cols.append("unidade")
        if "tipo" in oldest_cases.columns:
            display_cols.append("tipo")
        display_cols.extend(["dias_pendentes", "faixa_aging"])
        
        available_cols = [col for col in display_cols if col in oldest_cases.columns]
        if available_cols:
            st.dataframe(oldest_cases[available_cols], use_container_width=True, height=300)
        
        return df_analysis, stats
    
    # Análise de laudos pendentes
    laudos_analysis, laudos_stats = analyze_aging_comprehensive(df_pend_laudos, "Laudos Pendentes")
    
    # Análise de exames pendentes  
    exames_analysis, exames_stats = analyze_aging_comprehensive(df_pend_exames, "Exames Pendentes")
    
    # Análise comparativa consolidada
    if laudos_stats and exames_stats:
        st.markdown("#### 📊 Análise Comparativa de Pendências")
        
        comparison_data = pd.DataFrame({
            "Tipo": ["Laudos", "Exames"],
            "Total": [laudos_stats["total"], exames_stats["total"]],
            "Média_Dias": [laudos_stats["media_dias"], exames_stats["media_dias"]],
            "Críticos": [laudos_stats["criticos"], exames_stats["criticos"]],
            "P90_Dias": [laudos_stats["p90_dias"], exames_stats["p90_dias"]]
        })
        
        fig_comparison = px.bar(
            comparison_data,
            x="Tipo",
            y=["Total", "Críticos"],
            title="Comparativo: Laudos vs Exames Pendentes",
            barmode="group"
        )
        
        fig_comparison.update_layout(height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Tabela comparativa
        st.dataframe(comparison_data, use_container_width=True)

# ============ ABA 5: ANÁLISE DIÁRIA ============
with tab5:
    st.subheader("📅 Análise Operacional Diária")
    
    def process_daily_data(df_atend: pd.DataFrame, df_laudos: pd.DataFrame):
        """Processamento de dados diários"""
        def extract_daily_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
            if df is None or df.empty or "dia" not in df.columns:
                return pd.DataFrame(columns=["dia", label])
            
            daily_data = (
                df.dropna(subset=["dia"])
                .groupby("dia")["quantidade"]
                .sum()
                .reset_index()
                .rename(columns={"quantidade": label})
                .sort_values("dia")
            )
            return daily_data
        
        atend_daily = extract_daily_counts(df_atend, "Atendimentos")
        laudos_daily = extract_daily_counts(df_laudos, "Laudos")
        
        if atend_daily.empty and laudos_daily.empty:
            return None
        
        # Merge dos dados
        daily_combined = pd.merge(atend_daily, laudos_daily, on="dia", how="outer").fillna(0)
        daily_combined["Atendimentos"] = pd.to_numeric(daily_combined["Atendimentos"], errors="coerce").fillna(0)
        daily_combined["Laudos"] = pd.to_numeric(daily_combined["Laudos"], errors="coerce").fillna(0)
        daily_combined = daily_combined.sort_values("dia")
        
        # Cálculos adicionais
        daily_combined["Taxa_Conversao"] = np.where(
            daily_combined["Atendimentos"] > 0,
            (daily_combined["Laudos"] / daily_combined["Atendimentos"]) * 100,
            0
        )
        
        # Médias móveis
        for period in [7, 14, 30]:
            daily_combined[f"MA{period}_Atendimentos"] = daily_combined["Atendimentos"].rolling(period).mean()
            daily_combined[f"MA{period}_Laudos"] = daily_combined["Laudos"].rolling(period).mean()
            daily_combined[f"MA{period}_Taxa"] = daily_combined["Taxa_Conversao"].rolling(period).mean()
        
        # Detecção de tendências
        if len(daily_combined) >= 30:
            recent_30 = daily_combined.tail(30)
            trend_atend = np.polyfit(range(30), recent_30["Atendimentos"], 1)[0]
            trend_laudos = np.polyfit(range(30), recent_30["Laudos"], 1)[0]
            
            daily_combined["Trend_Atendimentos"] = trend_atend
            daily_combined["Trend_Laudos"] = trend_laudos
        
        # Análise de sazonalidade semanal
        daily_combined["Dia_Semana"] = pd.to_datetime(daily_combined["dia"]).dt.day_name()
        daily_combined["Numero_Semana"] = pd.to_datetime(daily_combined["dia"]).dt.isocalendar().week
        
        return daily_combined
    
    daily_data = process_daily_data(df_atend_diario, df_laudos_diario)
    
    if daily_data is None or daily_data.empty:
        st.info("📝 Sem dados diários disponíveis. Carregue os arquivos 'Atendimentos (Diário)' e 'Laudos (Diário)'")
    else:
        # Métricas principais
        ultima_data = daily_data["dia"].max()
        ultimo_registro = daily_data[daily_data["dia"] == ultima_data].iloc[0]
        
        col_metrics = st.columns(5)
        with col_metrics[0]:
            st.metric("Último Dia", ultima_data.strftime("%d/%m/%Y"))
        with col_metrics[1]:
            st.metric("Atendimentos", f"{int(ultimo_registro['Atendimentos']):,}".replace(",", "."))
        with col_metrics[2]:
            st.metric("Laudos", f"{int(ultimo_registro['Laudos']):,}".replace(",", "."))
        with col_metrics[3]:
            taxa_ultima = ultimo_registro["Taxa_Conversao"]
            st.metric("Taxa Conversão", f"{taxa_ultima:.1f}%")
        with col_metrics[4]:
            dias_analisados = len(daily_data)
            st.metric("Dias Analisados", f"{dias_analisados:,}".replace(",", "."))
        
        # Gráfico principal - Série temporal
        st.markdown("#### 📈 Evolução Diária Completa")
        
        fig_daily = go.Figure()
        
        # Série principal
        fig_daily.add_trace(go.Scatter(
            x=daily_data["dia"],
            y=daily_data["Atendimentos"],
            mode="lines",
            name="Atendimentos",
            line=dict(color="#3b82f6", width=2)
        ))
        
        fig_daily.add_trace(go.Scatter(
            x=daily_data["dia"],
            y=daily_data["Laudos"],
            mode="lines",
            name="Laudos",
            line=dict(color="#10b981", width=2)
        ))
        
        # Médias móveis
        if "MA7_Atendimentos" in daily_data.columns:
            fig_daily.add_trace(go.Scatter(
                x=daily_data["dia"],
                y=daily_data["MA7_Atendimentos"],
                mode="lines",
                name="MM7 Atendimentos",
                line=dict(color="#3b82f6", width=2, dash="dash"),
                opacity=0.7
            ))
            
            fig_daily.add_trace(go.Scatter(
                x=daily_data["dia"],
                y=daily_data["MA7_Laudos"],
                mode="lines",
                name="MM7 Laudos",
                line=dict(color="#10b981", width=2, dash="dash"),
                opacity=0.7
            ))
        
        fig_daily.update_layout(
            height=chart_height,
            hovermode="x unified",
            xaxis_title="Data",
            yaxis_title="Quantidade",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Análise de taxa de conversão diária
        col_conv1, col_conv2 = st.columns([0.7, 0.3])
        
        with col_conv1:
            st.markdown("#### 🎯 Taxa de Conversão Diária")
            
            fig_conversion = go.Figure()
            
            fig_conversion.add_trace(go.Scatter(
                x=daily_data["dia"],
                y=daily_data["Taxa_Conversao"],
                mode="lines+markers",
                name="Taxa Diária",
                line=dict(color="#f59e0b", width=2),
                marker=dict(size=4)
            ))
            
            if "MA7_Taxa" in daily_data.columns:
                fig_conversion.add_trace(go.Scatter(
                    x=daily_data["dia"],
                    y=daily_data["MA7_Taxa"],
                    mode="lines",
                    name="Média Móvel 7 dias",
                    line=dict(color="#ef4444", width=3, dash="dash")
                ))
            
            if show_benchmarks:
                fig_conversion.add_hline(
                    y=70, 
                    line_dash="dot", 
                    line_color="red",
                    annotation_text="Meta: 70%"
                )
                fig_conversion.add_hline(
                    y=50, 
                    line_dash="dot", 
                    line_color="orange",
                    annotation_text="Mínimo: 50%"
                )
            
            fig_conversion.update_layout(
                height=400,
                hovermode="x unified",
                xaxis_title="Data",
                yaxis_title="Taxa (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_conversion, use_container_width=True)
        
        with col_conv2:
            st.markdown("#### 📊 Distribuição Semanal")
            
            # Análise por dia da semana
            weekly_pattern = daily_data.groupby("Dia_Semana").agg({
                "Atendimentos": "mean",
                "Laudos": "mean",
                "Taxa_Conversao": "mean"
            }).round(1)
            
            # Reordenar dias da semana
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_names_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            
            weekly_pattern = weekly_pattern.reindex([day for day in day_order if day in weekly_pattern.index])
            weekly_pattern.index = [day_names_pt[day_order.index(day)] for day in weekly_pattern.index]
            
            fig_weekly = px.bar(
                weekly_pattern.reset_index(),
                x="Dia_Semana",
                y=["Atendimentos", "Laudos"],
                title="Média por Dia da Semana",
                barmode="group"
            )
            
            fig_weekly.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Estatísticas semanais
            st.markdown("**📈 Estatísticas:**")
            melhor_dia = weekly_pattern["Taxa_Conversao"].idxmax()
            pior_dia = weekly_pattern["Taxa_Conversao"].idxmin()
            
            st.write(f"🏆 **Melhor dia:** {melhor_dia} ({weekly_pattern.loc[melhor_dia, 'Taxa_Conversao']:.1f}%)")
            st.write(f"📉 **Pior dia:** {pior_dia} ({weekly_pattern.loc[pior_dia, 'Taxa_Conversao']:.1f}%)")

# ============ ABA 6: DADOS BRUTOS ============
with tab6:
    st.subheader("📋 Exploração e Qualidade dos Dados")
    
    # Resumo geral dos datasets
    st.markdown("#### 📊 Resumo dos Datasets Carregados")
    
    data_summary = []
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            # Informações básicas
            periodo_info = "Sem dados temporais"
            if 'anomês' in df.columns and not df['anomês'].isna().all():
                periodo_info = f"{df['anomês'].min()} a {df['anomês'].max()}"
            elif 'dia' in df.columns and not df['dia'].isna().all():
                min_date = df['dia'].min().strftime("%d/%m/%Y") if pd.notna(df['dia'].min()) else "N/A"
                max_date = df['dia'].max().strftime("%d/%m/%Y") if pd.notna(df['dia'].max()) else "N/A"
                periodo_info = f"{min_date} a {max_date}"
            
            # Cálculo de qualidade
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            quality_score = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
            
            quality_status = "🟢 Excelente" if quality_score >= 95 else "🟡 Boa" if quality_score >= 85 else "🟠 Regular" if quality_score >= 70 else "🔴 Ruim"
            
            data_summary.append({
                "Dataset": name.replace("_", " ").title(),
                "Registros": f"{len(df):,}".replace(",", "."),
                "Colunas": len(df.columns),
                "Período": periodo_info,
                "Qualidade": quality_status,
                "Tamanho (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "Status": "✅ Ativo" if name in filtered_dataframes and not filtered_dataframes[name].empty else "⚠️ Filtrado"
            })
    
    if data_summary:
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.metric("Total de Registros", f"{total_registros:,}".replace(",", "."))
        with summary_col2:
            st.metric("Datasets Carregados", len(data_summary))
        with summary_col3:
            st.metric("Datasets Ativos", datasets_ativos)
        with summary_col4:
            st.metric("Tamanho Total", f"{total_tamanho:.1f} MB")

# ============ ABA 7: RELATÓRIOS ============
with tab7:
    st.subheader("📑 Relatórios Executivos e Exportações")
    
    # Seleção do tipo de relatório
    report_col1, report_col2 = st.columns([0.7, 0.3])
    
    with report_col1:
        report_type = st.selectbox(
            "🎯 Tipo de Relatório:",
            [
                "Relatório Executivo Completo",
                "Relatório de Produção",
                "Relatório de Pendências",
                "Relatório de Performance",
                "Relatório de Tendências"
            ]
        )
    
    with report_col2:
        report_format = st.selectbox(
            "📄 Formato de Exportação:",
            ["Markdown", "HTML", "JSON"]
        )
    
    def generate_executive_report() -> str:
        """Gera relatório executivo completo"""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Cálculo de insights adicionais
        insights = []
        
        if crescimento_laudos:
            if crescimento_laudos > 10:
                insights.append(f"📈 **Crescimento Forte**: Laudos cresceram {format_number(crescimento_laudos,1)}% no período")
            elif crescimento_laudos < -10:
                insights.append(f"📉 **Alerta de Queda**: Laudos decresceram {format_number(abs(crescimento_laudos),1)}% no período")
        
        if taxa_conversao:
            if taxa_conversao > 80:
                insights.append(f"🎯 **Alta Eficiência**: Taxa de conversão de {format_number(taxa_conversao,1)}% acima da meta")
            elif taxa_conversao < 50:
                insights.append(f"⚠️ **Baixa Eficiência**: Taxa de conversão de {format_number(taxa_conversao,1)}% abaixo do aceitável")
        
        # Recomendações baseadas em dados
        recommendations = []
        
        if backlog_meses and backlog_meses > 6:
            recommendations.append("🔴 **URGENTE**: Implementar plano de redução de backlog com metas semanais")
        
        if taxa_conversao and taxa_conversao < 60:
            recommendations.append("🟡 **MELHORIA**: Revisar processos de conversão de atendimentos em laudos")
        
        if total_pend_laudos > total_pend_exames * 2:
            recommendations.append("🟡 **PROCESSO**: Investigar gargalos na finalização de laudos")
        
        report = f"""# 📊 RELATÓRIO EXECUTIVO PCI/SC

**Data de Geração:** {timestamp}  
**Período de Análise:** {period_filter}  
**Filtros Aplicados:** {len([f for f in dimensional_filters.values() if f])} filtros ativos

---

## 🎯 RESUMO EXECUTIVO

### Indicadores Principais
| Métrica | Valor | Status |
|---------|-------|--------|
| **Atendimentos Totais** | {format_number(total_atendimentos)} | {("🟢" if crescimento_atendimentos and crescimento_atendimentos > 0 else "🔴")} |
| **Laudos Emitidos** | {format_number(total_laudos)} | {("🟢" if crescimento_laudos and crescimento_laudos > 0 else "🔴")} |
| **Taxa de Conversão** | {format_number(taxa_conversao, 1) if taxa_conversao else 'N/A'}% | {("🟢" if taxa_conversao and taxa_conversao >= 70 else "🟡" if taxa_conversao and taxa_conversao >= 50 else "🔴")} |
| **Produtividade Mensal** | {format_number(media_mensal_laudos, 1) if media_mensal_laudos else 'N/A'} laudos | - |

---

## ⏰ SITUAÇÃO DE PENDÊNCIAS

### Backlog Atual
- **Laudos Pendentes:** {format_number(total_pend_laudos)} casos
- **Exames Pendentes:** {format_number(total_pend_exames)} casos
- **Backlog Estimado:** {format_number(backlog_meses, 1) if backlog_meses else 'N/A'} meses
- **Aging Médio:** {format_number(aging_laudos.get("media_dias") or aging_exames.get("media_dias"), 0) if (aging_laudos.get("media_dias") or aging_exames.get("media_dias")) else 'N/A'} dias

### Casos Críticos (>90 dias)
- **Laudos Críticos:** {aging_laudos.get("criticos", 0)} casos
- **Exames Críticos:** {aging_exames.get("criticos", 0)} casos

---

## 📈 ANÁLISE DE PERFORMANCE

### Tendências Identificadas
{chr(10).join(insights) if insights else "- Sem tendências significativas identificadas no período"}

### Crescimento Período
- **Atendimentos:** {format_number(crescimento_atendimentos, 1) if crescimento_atendimentos else 'N/A'}%
- **Laudos:** {format_number(crescimento_laudos, 1) if crescimento_laudos else 'N/A'}%

---

## 🚨 ALERTAS E RECOMENDAÇÕES

### Recomendações Prioritárias
{chr(10).join(recommendations) if recommendations else "✅ **Situação Normal**: Todos os indicadores dentro dos parâmetros esperados"}

### Plano de Ação Sugerido
1. **Curto Prazo (30 dias):**
   - Monitorar diariamente casos com aging > 90 dias
   - Implementar reuniões semanais de acompanhamento de backlog

2. **Médio Prazo (90 dias):**
   - Otimizar processos de conversão de atendimentos
   - Estabelecer metas de produtividade por unidade

3. **Longo Prazo (180 dias):**
   - Implementar sistema de alertas automáticos
   - Desenvolver painéis de monitoramento em tempo real

---

## 📊 DADOS UTILIZADOS

### Datasets Processados
{chr(10).join([f"- **{name.replace('_', ' ').title()}**: {len(df):,} registros".replace(",", ".") for name, df in dataframes.items() if df is not None and not df.empty])}

### Período de Dados
- **Dados Mais Antigos:** {min([df['anomês'].min() for df in dataframes.values() if df is not None and 'anomês' in df.columns and not df['anomês'].isna().all()], default='N/A')}
- **Dados Mais Recentes:** {max([df['anomês'].max() for df in dataframes.values() if df is not None and 'anomês' in df.columns and not df['anomês'].isna().all()], default='N/A')}

---

## 📝 METODOLOGIA

### Cálculos Realizados
- **Taxa de Conversão:** (Total Laudos / Total Atendimentos) × 100
- **Crescimento:** Comparação entre primeiros e últimos 3 meses do período
- **Backlog:** Total Pendências / Produtividade Mensal Média
- **Aging:** Dias corridos desde a data de solicitação

### Critérios de Alerta
- 🟢 **Normal:** Taxa conversão > 70%, Backlog < 3 meses
- 🟡 **Atenção:** Taxa conversão 50-70%, Backlog 3-6 meses  
- 🔴 **Crítico:** Taxa conversão < 50%, Backlog > 6 meses

---

*Relatório gerado automaticamente pelo Dashboard PCI/SC v3.0*  
*Sistema de Monitoramento Executivo - Desenvolvido para otimização operacional*
"""
        
        return report.strip()
    
    # Interface de geração de relatórios
    if st.button("📊 Gerar Relatório", type="primary"):
        with st.spinner("Gerando relatório..."):
            # Seleção do conteúdo baseado no tipo
            if report_type == "Relatório Executivo Completo":
                report_content = generate_executive_report()
            else:
                report_content = f"# {report_type}\n\n*Relatório em desenvolvimento*\n\nEste tipo de relatório será implementado em versões futuras do dashboard."
            
            # Exibição do relatório
            st.markdown("#### 📄 Visualização do Relatório")
            st.markdown(report_content)
            
            # Preparação para download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{report_type.lower().replace(' ', '_')}_{timestamp}"
            
            if report_format == "Markdown":
                st.download_button(
                    label="📥 Download Relatório (Markdown)",
                    data=report_content.encode('utf-8'),
                    file_name=f"{filename_base}.md",
                    mime="text/markdown"
                )
            elif report_format == "HTML":
                # Conversão básica para HTML
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report_type} - PCI/SC</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        .metric {{ background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .alert {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
{report_content.replace(chr(10), '<br>')}
</body>
</html>
"""
                st.download_button(
                    label="📥 Download Relatório (HTML)",
                    data=html_content.encode('utf-8'),
                    file_name=f"{filename_base}.html",
                    mime="text/html"
                )

# ============ RESUMO NA SIDEBAR ============
with st.sidebar.expander("📊 Resumo da Sessão", expanded=False):
    # Datasets carregados
    st.markdown("**📁 Datasets Ativos:**")
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            filtered_df = filtered_dataframes.get(name, df)
            icon = "🟢" if not filtered_df.empty else "🟡"
            st.write(f"{icon} {name.replace('_', ' ').title()}: {len(filtered_df):,}".replace(",", "."))
    
    # Filtros aplicados
    active_filters = sum(1 for filters in dimensional_filters.values() if filters)
    st.markdown(f"**🔍 Filtros Ativos:** {active_filters}")
    
    # Período de análise
    st.markdown(f"**📅 Período:** {period_filter}")
    
    # Status geral
    if alerts:
        critical_count = len([a for a in alerts if a["type"] == "danger"])
        warning_count = len([a for a in alerts if a["type"] == "warning"])
        st.markdown(f"**🚨 Alertas:** {critical_count} críticos, {warning_count} atenção")
    else:
        st.markdown("**✅ Status:** Normal")

# ============ RODAPÉ ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #64748b; padding: 30px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin-top: 20px;'>
    <h4 style='color: #1e293b; margin-bottom: 16px;'>🏥 Dashboard PCI/SC v3.0</h4>
    <p style='margin: 8px 0; font-size: 16px;'><strong>Sistema Avançado de Monitoramento Executivo</strong></p>
    <p style='margin: 8px 0;'>📊 Análise de Produção • ⏰ Gestão de Pendências • 📈 Indicadores de Performance • 📋 Controle Operacional</p>
    <div style='margin: 16px 0; padding: 12px; background: rgba(255,255,255,0.7); border-radius: 8px; display: inline-block;'>
        <p style='margin: 4px 0; font-size: 14px;'><strong>📧 Suporte:</strong> equipe-ti@pci.sc.gov.br</p>
        <p style='margin: 4px 0; font-size: 14px;'><strong>🔧 Versão:</strong> 3.0.0 - Melhorias em Performance e UX</p>
        <p style='margin: 4px 0; font-size: 12px; color: #7f8c8d;'><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
    </div>
    <p style='margin-top: 16px; font-size: 12px; color: #9ca3af;'>Desenvolvido para otimização operacional e tomada de decisão baseada em dados</p>
</div>
""", unsafe_allow_html=True)df = pd.DataFrame(data_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Métricas consolidadas
        total_registros = sum(int(row["Registros"].replace(".", "")) for row in data_summary)
        total_tamanho = sum(row["Tamanho (MB)"] for row in data_summary)
        datasets_ativos = sum(1 for row in data_summary if row["Status"] == "✅ Ativo")
        
        summary_
