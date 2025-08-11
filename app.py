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

# ============ CONFIGURAÇÃO INICIAL ============
st.set_page_config(
    page_title="PCI/SC – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Dashboard PCI/SC – Produção & Pendências")
st.markdown("---")

# ============ CACHE E PERFORMANCE ============
@st.cache_data
def read_csv_optimized(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Lê CSV com detecção automática de separador e encoding otimizada."""
    separators = [",", ";", "\t", "|"]
    encodings = ["utf-8", "latin-1", "cp1252"]
    
    for encoding in encodings:
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(bio, sep=sep, encoding=encoding, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    
    # Fallback para detecção automática
    try:
        bio = io.BytesIO(file_content)
        return pd.read_csv(bio, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return None

@st.cache_data
def process_datetime_column(series: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    """Processa coluna de data/hora com múltiplos formatos."""
    if series is None or series.empty:
        return None
    
    # Primeiro tenta conversão direta
    dt_series = pd.to_datetime(
        series, 
        errors="coerce", 
        dayfirst=dayfirst, 
        infer_datetime_format=True
    )
    
    # Se muitas datas inválidas, tenta outros formatos
    if dt_series.isna().sum() > len(dt_series) * 0.5:
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                dt_series = pd.to_datetime(series, format=fmt, errors="coerce")
                if dt_series.notna().sum() > len(dt_series) * 0.5:
                    break
            except Exception:
                continue
    
    return dt_series if dt_series.notna().any() else None

# ============ UTILITÁRIOS MELHORADOS ============
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
    """Calcula percentual com validação."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return (numerator / denominator) * 100

def get_period_filter_options(df: pd.DataFrame) -> List[str]:
    """Gera opções de filtro de período baseado nos dados."""
    if df is None or "anomês_dt" not in df.columns:
        return []
    
    dates = df["anomês_dt"].dropna()
    if dates.empty:
        return []
    
    periods = []
    min_date = dates.min()
    max_date = dates.max()
    
    # Últimos períodos
    now = pd.Timestamp.now()
    periods.extend([
        "Últimos 3 meses",
        "Últimos 6 meses", 
        "Último ano",
        "Ano atual",
        "Todo o período"
    ])
    
    return periods

# ============ DETECÇÃO DE ARQUIVOS ============
@st.cache_data
def detect_data_sources():
    """Detecta se existem arquivos na pasta data/."""
    return os.path.exists("data") and any(
        p.endswith(".csv") for p in os.listdir("data")
    )

has_data_dir = detect_data_sources()

# ============ INTERFACE DE UPLOAD ============
st.sidebar.header("📁 Configuração de Dados")

if not has_data_dir:
    st.sidebar.info("💡 Envie os arquivos CSV disponíveis. O dashboard se adapta automaticamente.")

# Definição dos arquivos esperados
file_configs = {
    "Atendimentos_todos_Mensal": {
        "label": "Atendimentos Todos (Mensal)",
        "description": "Dados gerais de atendimentos por mês"
    },
    "Laudos_todos_Mensal": {
        "label": "Laudos Todos (Mensal)", 
        "description": "Dados gerais de laudos por mês"
    },
    "Atendimentos_especifico_Mensal": {
        "label": "Atendimentos Específicos (Mensal)",
        "description": "Atendimentos detalhados por competência"
    },
    "Laudos_especifico_Mensal": {
        "label": "Laudos Específicos (Mensal)",
        "description": "Laudos detalhados por competência"
    },
    "laudos_realizados": {
        "label": "Laudos Realizados",
        "description": "Histórico detalhado de laudos concluídos"
    },
    "detalhes_laudospendentes": {
        "label": "Laudos Pendentes",
        "description": "Laudos aguardando conclusão"
    },
    "detalhes_examespendentes": {
        "label": "Exames Pendentes", 
        "description": "Exames aguardando realização"
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
    
    target_name = name.lower().replace(" ", "_")
    
    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        
        base_name = os.path.splitext(filename)[0].lower()
        normalized_name = re.sub(r"[^\w]", "_", base_name)
        
        if normalized_name.startswith(target_name) or target_name in normalized_name:
            return os.path.join("data", filename)
    
    return None

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
                except Exception as e:
                    st.sidebar.error(f"Erro ao carregar {name}: {str(e)}")
        else:
            if upload_file is not None:
                try:
                    content = upload_file.read()
                    df = read_csv_optimized(content, name)
                except Exception as e:
                    st.sidebar.error(f"Erro ao processar {name}: {str(e)}")
        
        if df is not None:
            # Normaliza nomes das colunas
            df.columns = [
                re.sub(r"\s+", " ", col.strip().lower()) 
                for col in df.columns
            ]
            loaded_data[name] = df
    
    return loaded_data

# Carrega os dados
raw_dataframes = load_all_data(uploads)

if not raw_dataframes:
    st.warning("⚠️ Nenhum arquivo foi carregado. Por favor, envie os arquivos CSV pela barra lateral ou coloque-os na pasta `data/`.")
    st.info("📝 **Arquivos esperados:** " + ", ".join(file_configs.keys()))
    st.stop()

# ============ MAPEAMENTO DE COLUNAS APRIMORADO ============
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
        "id": "idatendimento"
    },
    "Atendimentos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "idatendimento"
    },
    "Laudos_todos_Mensal": {
        "date": "data_interesse", 
        "id": "iddocumento"
    },
    "Laudos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "iddocumento"
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

# ============ PADRONIZAÇÃO DE DADOS ============
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza estrutura do DataFrame para análise unificada."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    mapping = COLUMN_MAPPINGS.get(name, {})
    result = df.copy()
    
    # Adiciona coluna de quantidade padrão
    result["quantidade"] = 1
    
    # Mapeia colunas dimensionais
    dimension_columns = [
        "diretoria", "superintendencia", "unidade", 
        "tipo", "perito", "id"
    ]
    
    for dim_col in dimension_columns:
        if dim_col in mapping and mapping[dim_col] in result.columns:
            result[dim_col] = result[mapping[dim_col]]
    
    # Processa datas e competências
    anomês_dt = None
    
    # Prioridade: competencia -> date -> ano/mes
    if "competencia" in mapping and mapping["competencia"] in result.columns:
        anomês_dt = process_datetime_column(result[mapping["competencia"]])
        if anomês_dt is not None:
            anomês_dt = anomês_dt.dt.to_period("M").dt.to_timestamp()
    
    if anomês_dt is None and "date" in mapping and mapping["date"] in result.columns:
        date_col = process_datetime_column(result[mapping["date"]])
        if date_col is not None:
            anomês_dt = date_col.dt.to_period("M").dt.to_timestamp()
    
    # Para laudos_realizados: usar ano/mes se disponível
    if anomês_dt is None and name == "laudos_realizados":
        ano_col = mapping.get("ano")
        mes_col = mapping.get("mes")
        
        if ano_col in result.columns and mes_col in result.columns:
            try:
                anos = pd.to_numeric(result[ano_col], errors="coerce")
                meses = pd.to_numeric(result[mes_col], errors="coerce")
                
                valid_mask = (~anos.isna()) & (~meses.isna()) & (meses >= 1) & (meses <= 12)
                if valid_mask.any():
                    dates = pd.to_datetime({
                        'year': anos,
                        'month': meses, 
                        'day': 1
                    }, errors="coerce")
                    anomês_dt = dates.dt.to_period("M").dt.to_timestamp()
            except Exception:
                pass
    
    # Adiciona colunas de tempo padronizadas
    if anomês_dt is not None:
        result["anomês_dt"] = anomês_dt
        result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
        result["ano"] = result["anomês_dt"].dt.year
        result["mes"] = result["anomês_dt"].dt.month
    
    # Adiciona data base para cálculos de aging
    if "date" in mapping and mapping["date"] in result.columns:
        result["data_base"] = process_datetime_column(result[mapping["date"]])
    
    # Processamento específico para laudos realizados
    if name == "laudos_realizados":
        date_fields = ["solicitacao", "atendimento", "emissao"]
        
        for field in date_fields:
            col_name = mapping.get(field)
            if col_name and col_name in result.columns:
                result[f"dh{field}"] = process_datetime_column(result[col_name])
        
        # Calcula TME (Tempo Médio de Execução)
        if "dhemitido" in result.columns:
            base_date = (
                result.get("dhatendimento") 
                if "dhatendimento" in result.columns 
                else result.get("dhsolicitacao")
            )
            
            if base_date is not None:
                result["tme_dias"] = (result["dhemitido"] - base_date).dt.days
                result["sla_30_ok"] = result["tme_dias"] <= 30
                result["sla_60_ok"] = result["tme_dias"] <= 60
    
    # Limpeza e padronização de texto
    text_columns = [
        "diretoria", "superintendencia", "unidade", 
        "tipo", "id", "perito", "anomês"
    ]
    
    for col in text_columns:
        if col in result.columns:
            result[col] = (
                result[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": None, "": None})
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
                   if 'anomês' in standardized_df.columns and not standardized_df['anomês'].isna().all()
                   else "Sem dados temporais"
    })

# Exibe informações de processamento
with st.sidebar.expander("📊 Resumo dos Dados", expanded=False):
    info_df = pd.DataFrame(processing_info)
    st.dataframe(info_df, use_container_width=True)

# ============ FILTROS AVANÇADOS ============
def extract_filter_values(column: str) -> List[str]:
    """Extrai valores únicos de uma coluna em todos os DataFrames."""
    values = set()
    
    for df in standardized_dfs.values():
        if column in df.columns:
            unique_vals = df[column].dropna().astype(str).unique()
            values.update(v for v in unique_vals if v and v.lower() != "nan")
    
    return sorted(list(values))

st.sidebar.subheader("🔍 Filtros")

# Filtros dimensionais
filter_diretoria = st.sidebar.multiselect(
    "Diretoria", 
    extract_filter_values("diretoria"),
    help="Selecione uma ou mais diretorias"
)

filter_superintendencia = st.sidebar.multiselect(
    "Superintendência", 
    extract_filter_values("superintendencia"),
    help="Selecione uma ou mais superintendências"
)

filter_unidade = st.sidebar.multiselect(
    "Unidade", 
    extract_filter_values("unidade"),
    help="Selecione uma ou mais unidades"
)

filter_tipo = st.sidebar.multiselect(
    "Tipo de Perícia", 
    extract_filter_values("tipo"),
    help="Selecione um ou mais tipos"
)

# Filtro de período
period_options = ["Todo o período", "Últimos 6 meses", "Últimos 3 meses", "Ano atual"]
filter_periodo = st.sidebar.selectbox(
    "Período de análise",
    period_options,
    help="Selecione o período para análise"
)

# ============ APLICAÇÃO DE FILTROS ============
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos os filtros selecionados ao DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    # Filtros dimensionais
    filter_mappings = [
        ("diretoria", filter_diretoria),
        ("superintendencia", filter_superintendencia), 
        ("unidade", filter_unidade),
        ("tipo", filter_tipo)
    ]
    
    for column, filter_values in filter_mappings:
        if column in filtered.columns and filter_values:
            filtered = filtered[
                filtered[column].astype(str).isin(filter_values)
            ]
    
    # Filtro de período
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

# Aplica filtros a todos os DataFrames
filtered_dfs = {
    name: apply_filters(df) 
    for name, df in standardized_dfs.items()
}

# Extrai DataFrames filtrados para uso
df_atend_todos = filtered_dfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dfs.get("Laudos_todos_Mensal") 
df_atend_esp = filtered_dfs.get("Atendimentos_especifico_Mensal")
df_laudos_esp = filtered_dfs.get("Laudos_especifico_Mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("detalhes_laudospendentes")
df_pend_exames = filtered_dfs.get("detalhes_examespendentes")

# ============ CÁLCULO DE KPIS APRIMORADOS ============
def calculate_total(df: pd.DataFrame) -> int:
    """Calcula total de registros."""
    return len(df) if df is not None and not df.empty else 0

def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
    """Calcula média mensal."""
    if df is None or df.empty or "anomês_dt" not in df.columns:
        return None
    
    monthly_totals = df.groupby("anomês_dt")["quantidade"].sum()
    return monthly_totals.mean() if len(monthly_totals) > 0 else None

def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
    """Calcula taxa de crescimento dos últimos períodos."""
    if df is None or df.empty or "anomês_dt" not in df.columns:
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

# Calcula KPIs principais
total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = calculate_total(df_pend_laudos)
total_pend_exames = calculate_total(df_pend_exames)

# KPIs derivados
media_mensal_laudos = calculate_monthly_average(df_laudos_todos)
backlog_meses = (
    total_pend_laudos / media_mensal_laudos 
    if media_mensal_laudos and media_mensal_laudos > 0 
    else None
)

taxa_atendimento = calculate_percentage(total_laudos, total_atendimentos)
crescimento_laudos = calculate_growth_rate(df_laudos_todos)

# KPIs de performance (laudos realizados)
tme_mediano = None
sla_30_percent = None
sla_60_percent = None

if df_laudos_real is not None and not df_laudos_real.empty:
    if "tme_dias" in df_laudos_real.columns:
        tme_values = pd.to_numeric(df_laudos_real["tme_dias"], errors="coerce").dropna()
        tme_mediano = tme_values.median() if not tme_values.empty else None
    
    if "sla_30_ok" in df_laudos_real.columns:
        sla_30_percent = df_laudos_real["sla_30_ok"].mean() * 100
    
    if "sla_60_ok" in df_laudos_real.columns:
        sla_60_percent = df_laudos_real["sla_60_ok"].mean() * 100

# ============ EXIBIÇÃO DE KPIS ============
st.subheader("📈 Indicadores Principais")

# Primeira linha de KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Atendimentos Totais",
        format_number(total_atendimentos),
        help="Total de atendimentos no período filtrado"
    )

with col2:
    delta_laudos = f"+{format_number(crescimento_laudos, 1)}%" if crescimento_laudos else None
    st.metric(
        "Laudos Emitidos", 
        format_number(total_laudos),
        delta=delta_laudos,
        help="Total de laudos emitidos no período"
    )

with col3:
    st.metric(
        "Taxa de Atendimento",
        f"{format_number(taxa_atendimento, 1)}%" if taxa_atendimento else "—",
        help="Percentual de atendimentos que resultaram em laudos"
    )

with col4:
    st.metric(
        "Produtividade Mensal",
        format_number(media_mensal_laudos, 1) if media_mensal_laudos else "—",
        help="Média de laudos emitidos por mês"
    )

# Segunda linha de KPIs
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Laudos Pendentes",
        format_number(total_pend_laudos),
        help="Laudos aguardando emissão"
    )

with col6:
    st.metric(
        "Exames Pendentes", 
        format_number(total_pend_exames),
        help="Exames aguardando realização"
    )

with col7:
    backlog_color = "normal"
    if backlog_meses:
        if backlog_meses > 6:
            backlog_color = "inverse"
        elif backlog_meses > 3:
            backlog_color = "off"
    
    st.metric(
        "Backlog (meses)",
        format_number(backlog_meses, 1) if backlog_meses else "—",
        help="Tempo estimado para liquidar pendências atuais"
    )

with col8:
    st.metric(
        "TME Mediano (dias)",
        format_number(tme_mediano, 1) if tme_mediano else "—", 
        help="Tempo mediano de execução dos laudos"
    )

# Terceira linha - SLAs
if sla_30_percent is not None or sla_60_percent is not None:
    st.markdown("#### 🎯 Indicadores de SLA")
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        sla_30_delta = "normal" if sla_30_percent and sla_30_percent >= 80 else "inverse"
        st.metric(
            "SLA 30 dias",
            f"{format_number(sla_30_percent, 1)}%" if sla_30_percent else "—",
            help="Percentual de laudos emitidos em até 30 dias"
        )
    
    with col10:
        st.metric(
            "SLA 60 dias", 
            f"{format_number(sla_60_percent, 1)}%" if sla_60_percent else "—",
            help="Percentual de laudos emitidos em até 60 dias"
        )

st.markdown("---")

# ============ ABAS DO DASHBOARD ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral", 
    "📈 Tendências", 
    "🏆 Rankings", 
    "⏰ Pendências", 
    "📋 Dados",
    "📑 Relatórios"
])

# ============ ABA 1: VISÃO GERAL ============
with tab1:
    st.subheader("Resumo Executivo")
    
    # Gráficos principais lado a lado
    col_left, col_right = st.columns(2)
    
    with col_left:
        if df_laudos_todos is not None and "unidade" in df_laudos_todos.columns:
            unidade_summary = (
                df_laudos_todos
                .groupby("unidade", as_index=False)["quantidade"]
                .sum()
                .sort_values("quantidade", ascending=False)
                .head(15)
            )
            
            fig_unidades = px.bar(
                unidade_summary,
                x="quantidade", 
                y="unidade",
                orientation="h",
                title="🏢 Top 15 Unidades - Laudos Emitidos",
                color="quantidade",
                color_continuous_scale="Blues"
            )
            fig_unidades.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_unidades, use_container_width=True)
    
    with col_right:
        if df_laudos_todos is not None and "tipo" in df_laudos_todos.columns:
            tipo_summary = (
                df_laudos_todos
                .groupby("tipo", as_index=False)["quantidade"]
                .sum()
                .sort_values("quantidade", ascending=False)
                .head(12)
            )
            
            fig_tipos = px.pie(
                tipo_summary,
                values="quantidade",
                names="tipo", 
                title="🔍 Distribuição por Tipo de Perícia"
            )
            fig_tipos.update_traces(textposition='inside', textinfo='percent+label')
            fig_tipos.update_layout(height=500)
            st.plotly_chart(fig_tipos, use_container_width=True)
    
    # Estatísticas por diretoria
    if df_laudos_todos is not None and "diretoria" in df_laudos_todos.columns:
        st.markdown("#### 📊 Performance por Diretoria")
        
        diretoria_stats = (
            df_laudos_todos
            .groupby("diretoria")
            .agg({
                "quantidade": ["sum", "count"],
                "anomês_dt": ["min", "max"]
            })
            .round(2)
        )
        
        diretoria_stats.columns = ["Total Laudos", "Registros", "Início", "Fim"]
        diretoria_stats = diretoria_stats.sort_values("Total Laudos", ascending=False)
        
        st.dataframe(diretoria_stats, use_container_width=True)

# ============ ABA 2: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise Temporal")
    
    def create_time_series_chart(df: pd.DataFrame, title: str, color: str = "blue") -> None:
        """Cria gráfico de série temporal."""
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
        
        monthly_data["anomês"] = monthly_data["anomês_dt"].dt.strftime("%Y-%m")
        
        # Adiciona linha de tendência
        fig = px.line(
            monthly_data,
            x="anomês",
            y="quantidade", 
            markers=True,
            title=title,
            line_shape="spline"
        )
        
        # Adiciona média móvel
        if len(monthly_data) >= 3:
            monthly_data["media_movel"] = (
                monthly_data["quantidade"]
                .rolling(window=3, center=True)
                .mean()
            )
            
            fig.add_scatter(
                x=monthly_data["anomês"],
                y=monthly_data["media_movel"],
                mode="lines",
                name="Média Móvel (3m)",
                line=dict(dash="dash", color="red")
            )
        
        fig.update_layout(
            height=400,
            xaxis_title="Período",
            yaxis_title="Quantidade",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos de tendência em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        create_time_series_chart(
            df_atend_todos,
            "🏥 Atendimentos - Evolução Mensal",
            "blue"
        )
        
        create_time_series_chart(
            df_atend_esp,
            "🏥 Atendimentos Específicos - Evolução",
            "lightblue"
        )
    
    with col2:
        create_time_series_chart(
            df_laudos_todos,
            "📄 Laudos - Evolução Mensal", 
            "green"
        )
        
        create_time_series_chart(
            df_laudos_esp,
            "📄 Laudos Específicos - Evolução",
            "lightgreen"
        )
    
    # Gráfico comparativo
    if (df_atend_todos is not None and df_laudos_todos is not None and 
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        
        st.markdown("#### 📊 Comparativo: Atendimentos vs Laudos")
        
        atend_monthly = (
            df_atend_todos
            .groupby("anomês_dt")["quantidade"]
            .sum()
            .reset_index()
        )
        atend_monthly["tipo"] = "Atendimentos"
        
        laudos_monthly = (
            df_laudos_todos
            .groupby("anomês_dt")["quantidade"] 
            .sum()
            .reset_index()
        )
        laudos_monthly["tipo"] = "Laudos"
        
        combined_data = pd.concat([atend_monthly, laudos_monthly])
        combined_data["anomês"] = combined_data["anomês_dt"].dt.strftime("%Y-%m")
        
        fig_combined = px.line(
            combined_data,
            x="anomês",
            y="quantidade",
            color="tipo",
            markers=True,
            title="Comparativo Mensal: Atendimentos vs Laudos"
        )
        
        fig_combined.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_combined, use_container_width=True)

# ============ ABA 3: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Comparativos")
    
    def create_ranking_chart(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20) -> None:
        """Cria gráfico de ranking."""
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        ranking_data = (
            df.groupby(dimension, as_index=False)["quantidade"]
            .sum()
            .sort_values("quantidade", ascending=False)
            .head(top_n)
        )
        
        if ranking_data.empty:
            st.info(f"Sem dados para {title}")
            return
        
        fig = px.bar(
            ranking_data,
            x="quantidade",
            y=dimension,
            orientation="h",
            title=title,
            color="quantidade",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            height=max(400, len(ranking_data) * 25),
            showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Rankings em abas
    rank_tab1, rank_tab2, rank_tab3 = st.tabs(["Por Diretoria", "Por Unidade", "Por Tipo"])
    
    with rank_tab1:
        col1, col2 = st.columns(2)
        with col1:
            create_ranking_chart(
                df_atend_todos,
                "diretoria", 
                "🏥 Atendimentos por Diretoria"
            )
        with col2:
            create_ranking_chart(
                df_laudos_todos,
                "diretoria",
                "📄 Laudos por Diretoria"
            )
    
    with rank_tab2:
        col1, col2 = st.columns(2)
        with col1:
            create_ranking_chart(
                df_atend_todos,
                "unidade",
                "🏥 Atendimentos por Unidade",
                25
            )
        with col2:
            create_ranking_chart(
                df_laudos_todos,
                "unidade", 
                "📄 Laudos por Unidade",
                25
            )
    
    with rank_tab3:
        create_ranking_chart(
            df_laudos_todos,
            "tipo",
            "📄 Laudos por Tipo de Perícia",
            30
        )

# ============ ABA 4: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Análise de Pendências")
    
    def calculate_aging(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, pd.Series]:
        """Calcula aging das pendências."""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype="int64")
        
        # Usa data_base se disponível, senão procura outras colunas de data
        if date_column not in df.columns:
            date_columns = [col for col in df.columns if "data" in col.lower()]
            if date_columns:
                date_column = date_columns[0]
            else:
                return df, pd.Series(dtype="int64")
        
        result = df.copy()
        dates = pd.to_datetime(result[date_column], errors="coerce")
        
        if dates.isna().all():
            return df, pd.Series(dtype="int64")
        
        hoje = pd.Timestamp.now().normalize()
        dias_pendentes = (hoje - dates).dt.days
        
        # Categorias de aging
        faixas_aging = pd.cut(
            dias_pendentes,
            bins=[-1, 30, 60, 90, 180, 365, float('inf')],
            labels=["0-30 dias", "31-60 dias", "61-90 dias", 
                   "91-180 dias", "181-365 dias", "> 365 dias"]
        )
        
        result["dias_pendentes"] = dias_pendentes
        result["faixa_aging"] = faixas_aging
        
        distribuicao = faixas_aging.value_counts().sort_index()
        
        return result, distribuicao
    
    col1, col2 = st.columns(2)
    
    # Análise de Laudos Pendentes
    with col1:
        st.markdown("#### 📄 Laudos Pendentes")
        
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            laudos_aged, dist_laudos = calculate_aging(df_pend_laudos)
            
            # Gráfico de distribuição de aging
            if not dist_laudos.empty:
                fig_aging_laudos = px.bar(
                    x=dist_laudos.index,
                    y=dist_laudos.values,
                    title="Distribuição por Tempo de Pendência",
                    color=dist_laudos.values,
                    color_continuous_scale="Reds"
                )
                fig_aging_laudos.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="Faixa de Dias",
                    yaxis_title="Quantidade"
                )
                st.plotly_chart(fig_aging_laudos, use_container_width=True)
            
            # Tabela resumo
            if "dias_pendentes" in laudos_aged.columns:
                resumo_laudos = {
                    "Total": len(laudos_aged),
                    "Média de dias": laudos_aged["dias_pendentes"].mean(),
                    "Mediana": laudos_aged["dias_pendentes"].median(),
                    "Máximo": laudos_aged["dias_pendentes"].max()
                }
                
                for metric, value in resumo_laudos.items():
                    if metric == "Total":
                        st.metric(metric, format_number(value))
                    else:
                        st.metric(f"{metric} (dias)", format_number(value, 1))
            
            # Amostra dos dados
            st.markdown("**Amostra dos Dados:**")
            display_columns = [
                col for col in ["id", "unidade", "diretoria", "tipo", "dias_pendentes", "faixa_aging"]
                if col in laudos_aged.columns
            ]
            st.dataframe(
                laudos_aged[display_columns].head(100),
                use_container_width=True,
                height=250
            )
        else:
            st.info("Sem dados de laudos pendentes disponíveis.")
    
    # Análise de Exames Pendentes
    with col2:
        st.markdown("#### 🔬 Exames Pendentes")
        
        if df_pend_exames is not None and not df_pend_exames.empty:
            exames_aged, dist_exames = calculate_aging(df_pend_exames)
            
            # Gráfico de distribuição de aging
            if not dist_exames.empty:
                fig_aging_exames = px.bar(
                    x=dist_exames.index,
                    y=dist_exames.values,
                    title="Distribuição por Tempo de Pendência",
                    color=dist_exames.values,
                    color_continuous_scale="Oranges"
                )
                fig_aging_exames.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="Faixa de Dias",
                    yaxis_title="Quantidade"
                )
                st.plotly_chart(fig_aging_exames, use_container_width=True)
            
            # Tabela resumo
            if "dias_pendentes" in exames_aged.columns:
                resumo_exames = {
                    "Total": len(exames_aged),
                    "Média de dias": exames_aged["dias_pendentes"].mean(),
                    "Mediana": exames_aged["dias_pendentes"].median(),
                    "Máximo": exames_aged["dias_pendentes"].max()
                }
                
                for metric, value in resumo_exames.items():
                    if metric == "Total":
                        st.metric(metric, format_number(value))
                    else:
                        st.metric(f"{metric} (dias)", format_number(value, 1))
            
            # Amostra dos dados
            st.markdown("**Amostra dos Dados:**")
            display_columns = [
                col for col in ["id", "unidade", "diretoria", "tipo", "dias_pendentes", "faixa_aging"] 
                if col in exames_aged.columns
            ]
            st.dataframe(
                exames_aged[display_columns].head(100),
                use_container_width=True,
                height=250
            )
        else:
            st.info("Sem dados de exames pendentes disponíveis.")
    
    # Análise consolidada de pendências por unidade
    st.markdown("#### 🏢 Pendências por Unidade")
    
    pendencias_consolidadas = []
    
    if df_pend_laudos is not None and "unidade" in df_pend_laudos.columns:
        laudos_por_unidade = (
            df_pend_laudos.groupby("unidade")
            .size()
            .reset_index(name="Laudos_Pendentes")
        )
        pendencias_consolidadas.append(laudos_por_unidade)
    
    if df_pend_exames is not None and "unidade" in df_pend_exames.columns:
        exames_por_unidade = (
            df_pend_exames.groupby("unidade")
            .size()
            .reset_index(name="Exames_Pendentes")
        )
        pendencias_consolidadas.append(exames_por_unidade)
    
    if pendencias_consolidadas:
        from functools import reduce
        
        pendencias_df = reduce(
            lambda left, right: pd.merge(left, right, on="unidade", how="outer"),
            pendencias_consolidadas
        ).fillna(0)
        
        pendencias_df["Total_Pendencias"] = (
            pendencias_df.get("Laudos_Pendentes", 0) + 
            pendencias_df.get("Exames_Pendentes", 0)
        )
        
        pendencias_df = pendencias_df.sort_values("Total_Pendencias", ascending=False)
        
        fig_pendencias = px.bar(
            pendencias_df.head(20),
            x="Total_Pendencias",
            y="unidade",
            orientation="h",
            title="Top 20 Unidades com Mais Pendências",
            color="Total_Pendencias",
            color_continuous_scale="Reds"
        )
        
        fig_pendencias.update_layout(
            height=600,
            showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )
        
        st.plotly_chart(fig_pendencias, use_container_width=True)

# ============ ABA 5: DADOS ============
with tab5:
    st.subheader("📋 Exploração dos Dados")
    
    st.markdown("#### 📊 Resumo dos DataFrames Carregados")
    
    # Tabela resumo dos dados
    data_summary = []
    for name, df in standardized_dfs.items():
        if df is not None and not df.empty:
            data_summary.append({
                "Dataset": name,
                "Registros": len(df),
                "Colunas": len(df.columns),
                "Período": (
                    f"{df['anomês'].min()} a {df['anomês'].max()}"
                    if 'anomês' in df.columns and not df['anomês'].isna().all()
                    else "Sem dados temporais"
                ),
                "Memória (MB)": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            })
    
    if data_summary:
        summary_df = pd.DataFrame(data_summary)
        st.dataframe(summary_df, use_container_width=True)
    
    st.markdown("#### 🔍 Visualização Detalhada")
    
    # Seletor de dataset
    available_datasets = [name for name, df in standardized_dfs.items() if df is not None]
    
    if available_datasets:
        selected_dataset = st.selectbox(
            "Selecione o dataset para visualizar:",
            available_datasets,
            help="Escolha qual conjunto de dados deseja explorar"
        )
        
        if selected_dataset:
            df_selected = standardized_dfs[selected_dataset]
            
            # Informações básicas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registros", format_number(len(df_selected)))
            with col2:
                st.metric("Colunas", len(df_selected.columns))
            with col3:
                valores_nulos = df_selected.isnull().sum().sum()
                st.metric("Valores Nulos", format_number(valores_nulos))
            with col4:
                if 'anomês_dt' in df_selected.columns:
                    unique_months = df_selected['anomês_dt'].nunique()
                    st.metric("Meses Únicos", format_number(unique_months))
            
            # Filtros para visualização
            st.markdown("**Filtros de Visualização:**")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                max_rows = st.number_input(
                    "Máximo de linhas a exibir:",
                    min_value=10,
                    max_value=10000,
                    value=500,
                    step=50
                )
            
            with viz_col2:
                if 'anomês' in df_selected.columns:
                    available_months = sorted(df_selected['anomês'].dropna().unique(), reverse=True)
                    selected_months = st.multiselect(
                        "Filtrar por mês:",
                        available_months,
                        default=available_months[:6] if len(available_months) > 6 else available_months
                    )
                else:
                    selected_months = []
            
            # Aplica filtros de visualização
            df_display = df_selected.copy()
            
            if selected_months and 'anomês' in df_display.columns:
                df_display = df_display[df_display['anomês'].isin(selected_months)]
            
            # Limita número de linhas
            df_display = df_display.head(max_rows)
            
            # Exibe informações das colunas
            with st.expander("📋 Informações das Colunas", expanded=False):
                column_info = []
                for col in df_selected.columns:
                    dtype = str(df_selected[col].dtype)
                    null_count = df_selected[col].isnull().sum()
                    null_percent = (null_count / len(df_selected)) * 100
                    unique_count = df_selected[col].nunique()
                    
                    column_info.append({
                        "Coluna": col,
                        "Tipo": dtype,
                        "Nulos": null_count,
                        "% Nulos": round(null_percent, 2),
                        "Únicos": unique_count
                    })
                
                column_df = pd.DataFrame(column_info)
                st.dataframe(column_df, use_container_width=True)
            
            # Exibe os dados
            st.markdown(f"**Dados ({len(df_display)} de {len(df_selected)} registros):**")
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
            
            # Opção de download
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download dos dados filtrados (CSV)",
                data=csv_data,
                file_name=f"{selected_dataset}_filtrado.csv",
                mime="text/csv"
            )

# ============ ABA 6: RELATÓRIOS ============
with tab6:
    st.subheader("📑 Relatórios Executivos")
    
    st.markdown("#### 📈 Resumo Executivo")
    
    # Gera relatório executivo automático
    relatorio_sections = []
    
    # Seção 1: Produção
    if total_laudos > 0:
        relatorio_sections.append(f"""
        **📊 PRODUÇÃO**
        - Total de laudos emitidos: {format_number(total_laudos)}
        - Média mensal de produção: {format_number(media_mensal_laudos, 1) if media_mensal_laudos else 'N/A'} laudos
        - Taxa de atendimento: {format_number(taxa_atendimento, 1) if taxa_atendimento else 'N/A'}%
        - Crescimento recente: {format_number(crescimento_laudos, 1) if crescimento_laudos else 'N/A'}%
        """)
    
    # Seção 2: Pendências
    if total_pend_laudos > 0 or total_pend_exames > 0:
        relatorio_sections.append(f"""
        **⏰ PENDÊNCIAS**
        - Laudos pendentes: {format_number(total_pend_laudos)}
        - Exames pendentes: {format_number(total_pend_exames)}
        - Backlog estimado: {format_number(backlog_meses, 1) if backlog_meses else 'N/A'} meses
        """)
    
    # Seção 3: Performance
    if tme_mediano is not None or sla_30_percent is not None:
        relatorio_sections.append(f"""
        **🎯 PERFORMANCE**
        - TME mediano: {format_number(tme_mediano, 1) if tme_mediano else 'N/A'} dias
        - SLA 30 dias: {format_number(sla_30_percent, 1) if sla_30_percent else 'N/A'}%
        - SLA 60 dias: {format_number(sla_60_percent, 1) if sla_60_percent else 'N/A'}%
        """)
    
    # Exibe o relatório
    if relatorio_sections:
        relatorio_completo = "\n".join(relatorio_sections)
        st.markdown(relatorio_completo)
        
        # Botão para download do relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relatorio_txt = f"""
RELATÓRIO EXECUTIVO PCI/SC
Data de Geração: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
Período de Análise: {filter_periodo}

{relatorio_completo}

---
Dashboard PCI/SC - Sistema de Monitoramento de Produção e Pendências
        """.strip()
        
        st.download_button(
            label="📥 Download Relatório Executivo",
            data=relatorio_txt.encode('utf-8'),
            file_name=f"relatorio_executivo_pci_sc_{timestamp}.txt",
            mime="text/plain"
        )
    else:
        st.info("Dados insuficientes para gerar relatório executivo.")
    
    st.markdown("#### 📊 Análises Adicionais")
    
    # Análise de sazonalidade
    if df_laudos_todos is not None and 'anomês_dt' in df_laudos_todos.columns:
        st.markdown("**📅 Análise de Sazonalidade**")
        
        monthly_production = (
            df_laudos_todos
            .groupby(df_laudos_todos['anomês_dt'].dt.month)['quantidade']
            .sum()
            .reset_index()
        )
        monthly_production['mes_nome'] = monthly_production['anomês_dt'].map({
            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
        })
        
        fig_sazonalidade = px.bar(
            monthly_production,
            x='mes_nome',
            y='quantidade',
            title='Produção de Laudos por Mês (Agregado)',
            color='quantidade',
            color_continuous_scale='Blues'
        )
        
        fig_sazonalidade.update_layout(
            height=400,
            xaxis_title='Mês',
            yaxis_title='Total de Laudos'
        )
        
        st.plotly_chart(fig_sazonalidade, use_container_width=True)

# ============ RODAPÉ ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>Dashboard PCI/SC - Sistema de Monitoramento de Produção e Pendências</p>
    <p>📧 Para sugestões e melhorias, entre em contato com a equipe de TI</p>
</div>
""", unsafe_allow_html=True)
