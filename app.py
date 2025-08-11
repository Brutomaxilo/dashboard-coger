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
    separators = [";", ",", "\t", "|"]
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(bio, sep=sep, encoding=encoding, engine="python", quotechar='"', skipinitialspace=True)
                if df.shape[1] > 1:
                    df.columns = [str(col).strip('"').strip() for col in df.columns]
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip('"').str.strip()
                    return df
            except Exception:
                continue
    return None

@st.cache_data
def process_datetime_column(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Processa coluna de data/hora com múltiplos formatos de forma robusta."""
    if series is None or len(series) == 0:
        return pd.Series(dtype='datetime64[ns]')

    # Tenta o formato inferido primeiro, que é mais rápido
    dt_series = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)

    # Se a primeira tentativa falhar para uma grande parte dos dados, tenta formatos explícitos
    if dt_series.isna().sum() > len(series) * 0.5:
        # Converte para string para garantir que os formatos funcionem
        str_series = series.astype(str)
        formats_to_try = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
        for fmt in formats_to_try:
            try:
                temp_series = pd.to_datetime(str_series, format=fmt, errors="coerce")
                # Se o novo formato for melhor, usa ele
                if temp_series.notna().sum() > dt_series.notna().sum():
                    dt_series = temp_series
            except (ValueError, TypeError):
                continue

    return dt_series

# ============ UTILITÁRIOS ============
def format_number(value, decimal_places: int = 0) -> str:
    """Formata números com separadores brasileiros e lida com Nones/NaNs."""
    if value is None or pd.isna(value):
        return "—"
    try:
        if decimal_places == 0:
            return f"{int(round(value)):,}".replace(",", ".")
        else:
            return f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "—"
        
def calculate_aging_analysis(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Calcula o aging (tempo de pendência) de um DataFrame."""
    if df is None or df.empty or date_column not in df.columns or df[date_column].isna().all():
        return pd.DataFrame(), pd.Series(dtype="int64"), {}
    
    result = df.copy()
    hoje = pd.Timestamp.now().normalize()
    dias_pendentes = (hoje - result[date_column]).dt.days
    
    # Define as faixas de aging
    bins = [-1, 15, 30, 60, 90, 180, 365, float('inf')]
    labels = ["0-15 dias", "16-30 dias", "31-60 dias", "61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"]
    result["faixa_aging"] = pd.cut(dias_pendentes, bins=bins, labels=labels)
    
    # Define as prioridades
    priority_bins = [-1, 30, 90, 180, float('inf')]
    priority_labels = ["Normal", "Atenção", "Urgente", "Crítico"]
    result["prioridade"] = pd.cut(dias_pendentes, bins=priority_bins, labels=priority_labels)
    
    distribuicao = result["faixa_aging"].value_counts().sort_index()
    stats = {
        "total": len(result),
        "media_dias": dias_pendentes.mean(),
        "mediana_dias": dias_pendentes.median(),
        "max_dias": dias_pendentes.max(),
        "criticos": (result["prioridade"] == "Crítico").sum(),
        "urgentes": (result["prioridade"] == "Urgente").sum()
    }
    return result, distribuicao, stats


# ============ MAPEAMENTO E RESOLUÇÃO DE ARQUIVOS ============
FILE_CONFIGS = {
    "Atendimentos_todos_Mensal": {"label": "Atendimentos Consolidados (Mensal)", "pattern": ["atendimentos_todos"]},
    "Laudos_todos_Mensal": {"label": "Laudos Consolidados (Mensal)", "pattern": ["laudos_todos"]},
    "Atendimentos_especifico_Mensal": {"label": "Atendimentos por Tipo (Mensal)", "pattern": ["atendimentos_especifico"]},
    "Laudos_especifico_Mensal": {"label": "Laudos por Tipo (Mensal)", "pattern": ["laudos_especifico"]},
    "laudos_realizados": {"label": "Histórico de Laudos Realizados (TME)", "pattern": ["laudos_realizados"]},
    "detalhes_laudospendentes": {"label": "Detalhes de Laudos Pendentes", "pattern": ["laudospendentes", "laudos_pendentes"]},
    "detalhes_examespendentes": {"label": "Detalhes de Exames Pendentes", "pattern": ["examespendentes", "exames_pendentes"]},
    "Atendimentos_diario": {"label": "Atendimentos (Diário)", "pattern": ["atendimentos_diario", "atendimentos_diário"]},
    "Laudos_diario": {"label": "Laudos (Diário)", "pattern": ["laudos_diario", "laudos_diário"]},
}

COLUMN_MAPPINGS = {
    "laudos_realizados": {
        "solicitacao": "dhsolicitacao", "atendimento": "dhatendimento", "emissao": "dhemitido",
        "n_laudo": "n_laudo", "ano": "ano_emissao", "mes": "mes_emissao",
        "unidade": "unidade_emissao", "diretoria": "diretoria", "competencia": "txcompetencia",
        "tipo": "txtipopericia", "perito": "perito"
    },
    "detalhes_laudospendentes": {"date": "data_solicitacao", "id": "caso_sirsaelp", "unidade": "unidade", "diretoria": "diretoria", "tipo": "tipopericia", "perito": "perito"},
    "detalhes_examespendentes": {"date": "data_solicitacao", "id": "caso_sirsaelp", "unidade": "unidade", "diretoria": "diretoria", "tipo": "tipopericia"},
    "Atendimentos_todos_Mensal": {"date": "data_interesse", "quantidade": "idatendimento"},
    "Laudos_todos_Mensal": {"date": "data_interesse", "quantidade": "iddocumento"},
    "Atendimentos_especifico_Mensal": {"date": "data_interesse", "quantidade": "idatendimento", "tipo": "txcompetencia"},
    "Laudos_especifico_Mensal": {"date": "data_interesse", "quantidade": "iddocumento", "tipo": "txcompetencia"},
    "Atendimentos_diario": {"date": "data_interesse", "quantidade": "idatendimento"},
    "Laudos_diario": {"date": "data_interesse", "quantidade": "iddocumento"},
}

@st.cache_data
def resolve_file_path(name: str) -> Optional[str]:
    """Resolve caminho do arquivo com tolerância a variações de nome."""
    if not os.path.exists("data"):
        return None
    config = FILE_CONFIGS.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        base_name = os.path.splitext(filename)[0].lower()
        normalized_name = re.sub(r"[^\w]", "_", base_name)
        for pattern in patterns:
            if pattern in normalized_name:
                return os.path.join("data", filename)
    return None

# ============ DADOS SIMULADOS (FALLBACK) ============
def create_sample_laudos_realizados() -> pd.DataFrame:
    """Cria dados simulados de laudos realizados para demonstração."""
    return pd.DataFrame() # Retorna vazio para forçar o uso dos arquivos reais, se existirem

# ============ CARREGAMENTO E PADRONIZAÇÃO DE DADOS ============
@st.cache_data(show_spinner="Padronizando dados...")
def load_and_standardize_all_data(file_sources: Dict, has_data_dir: bool) -> Dict[str, pd.DataFrame]:
    """Carrega e padroniza todos os dados disponíveis."""
    loaded_data = {}
    
    # 1. Carregar dados
    for name, upload_file in file_sources.items():
        df = None
        content = None
        source_msg = ""
        if has_data_dir:
            file_path = resolve_file_path(name)
            if file_path and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                source_msg = f"✅ {FILE_CONFIGS[name]['label']}"
        elif upload_file:
            content = upload_file.read()
            source_msg = f"✅ {FILE_CONFIGS[name]['label']}"

        if content:
            df = read_csv_optimized(content, name)
            if df is not None:
                st.sidebar.success(f"{source_msg}: {len(df)} registros")
                df.columns = [re.sub(r"\s+", "_", col.strip().lower()) for col in df.columns]
                loaded_data[name] = df
            else:
                st.sidebar.warning(f"⚠️ Falha ao ler {name}")
    
    if "laudos_realizados" not in loaded_data:
        st.sidebar.info("📊 Usando dados simulados para Laudos Realizados (demo)")
        loaded_data["laudos_realizados"] = create_sample_laudos_realizados()

    # 2. Padronizar dados
    standardized_dfs = {}
    for name, df in loaded_data.items():
        standardized_dfs[name] = standardize_dataframe(name, df.copy())
        
    return standardized_dfs

def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza a estrutura do DataFrame para análise unificada."""
    if df.empty:
        return pd.DataFrame()

    mapping = COLUMN_MAPPINGS.get(name, {})
    
    # Renomeia colunas com base no mapping
    rename_dict = {v: k for k, v in mapping.items() if v in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    # Garante coluna 'quantidade'
    if 'quantidade' not in df.columns:
        df['quantidade'] = 1
    else:
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(1)
        
    # Processamento de Datas para 'dia' e 'anomês_dt'
    date_col_options = ['date', 'emissao', 'solicitacao']
    chosen_date_col = next((col for col in date_col_options if col in df.columns), None)

    if chosen_date_col:
        df['data_base'] = process_datetime_column(df[chosen_date_col])
        df['dia'] = df['data_base'].dt.normalize()
        df['anomês_dt'] = df['data_base'].dt.to_period('M').dt.to_timestamp()

    if 'anomês_dt' in df.columns:
        df['anomês'] = df['anomês_dt'].dt.strftime('%Y-%m')
        df['ano'] = df['anomês_dt'].dt.year
        df['mes'] = df['anomês_dt'].dt.month

    # Processamento específico para TME e SLA em laudos_realizados
    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            if field in df.columns:
                df[f"dh_{field}"] = process_datetime_column(df[field])
        
        if "dh_emissao" in df.columns:
            base_date = df.get("dh_atendimento", df.get("dh_solicitacao"))
            if base_date is not None:
                valid_dates = df['dh_emissao'].notna() & base_date.notna()
                df.loc[valid_dates, 'tme_dias'] = (df.loc[valid_dates, 'dh_emissao'] - base_date.loc[valid_dates]).dt.days
                df['tme_dias'] = df['tme_dias'].apply(lambda x: x if x >= 0 else np.nan)
                df['sla_30_ok'] = df['tme_dias'] <= 30
                df['sla_60_ok'] = df['tme_dias'] <= 60

    # Limpeza de colunas de texto (dimensões)
    for col in ["diretoria", "superintendencia", "unidade", "tipo", "perito"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.title()
                       .replace({"Nan": None, "": None, "None": None, "nan": None}))

    return df

# ============ INTERFACE E CARREGAMENTO ============
st.sidebar.header("📁 Configuração de Dados")
HAS_DATA_DIR = os.path.exists("data") and any(p.endswith(".csv") for p in os.listdir("data"))

uploads = {}
if not HAS_DATA_DIR:
    st.sidebar.info("💡 Envie os arquivos CSV disponíveis. O dashboard se adapta automaticamente.")
    for key, config in FILE_CONFIGS.items():
        uploads[key] = st.sidebar.file_uploader(
            f"{config['label']} (.csv)",
            key=f"upload_{key}"
        )
else:
    st.sidebar.info("🔍 Lendo arquivos da pasta `data/`.")

# Carrega e padroniza todos os dados
standardized_dfs = load_and_standardize_all_data(uploads, HAS_DATA_DIR)

if not any(not df.empty for df in standardized_dfs.values()):
    st.warning("⚠️ Nenhum arquivo foi carregado ou os arquivos estão vazios. Por favor, envie os arquivos CSV pela barra lateral ou coloque-os na pasta `data/`.")
    st.stop()

# ============ FILTROS GLOBAIS ============
def extract_filter_values(column: str) -> List[str]:
    values = set()
    for df in standardized_dfs.values():
        if column in df.columns:
            unique_vals = df[column].dropna().astype(str).unique()
            values.update(v for v in unique_vals if v and v not in ["Nan", "None"])
    return sorted(list(values))

st.sidebar.subheader("🔍 Filtros")
filter_diretoria = st.sidebar.multiselect("Diretoria", extract_filter_values("diretoria"))
filter_unidade = st.sidebar.multiselect("Unidade", extract_filter_values("unidade"))
filter_tipo = st.sidebar.multiselect("Tipo de Perícia", extract_filter_values("tipo"))
filter_perito = st.sidebar.multiselect("Perito", extract_filter_values("perito"))

all_months = sorted(extract_filter_values("anomês"), reverse=True)
if all_months:
    start_month, end_month = st.sidebar.select_slider(
        "Selecione o Período (Ano-Mês)",
        options=all_months,
        value=(all_months[-1] if len(all_months) > 1 else all_months[0], all_months[0])
    )
else:
    start_month, end_month = None, None

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    filtered = df.copy()
    for column, filter_values in [("diretoria", filter_diretoria), ("unidade", filter_unidade), ("tipo", filter_tipo), ("perito", filter_perito)]:
        if column in filtered.columns and filter_values:
            filtered = filtered[filtered[column].isin(filter_values)]
    if start_month and end_month and "anomês" in filtered.columns:
        filtered = filtered[(filtered["anomês"] >= start_month) & (filtered["anomês"] <= end_month)]
    return filtered

filtered_dfs = {name: apply_filters(df) for name, df in standardized_dfs.items()}

df_atend_todos = filtered_dfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = filtered_dfs.get("Laudos_todos_Mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("detalhes_laudospendentes")
df_pend_exames = filtered_dfs.get("detalhes_examespendentes")
df_atend_diario = filtered_dfs.get("Atendimentos_diario")
df_laudos_diario = filtered_dfs.get("Laudos_diario")

# ============ CÁLCULOS DE KPIs ============
def calculate_total(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty: return 0
    return int(df["quantidade"].sum())

total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None else 0

media_mensal_laudos = 0
if df_laudos_todos is not None and not df_laudos_todos.empty and "anomês_dt" in df_laudos_todos.columns:
    monthly_totals = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum()
    if not monthly_totals.empty:
        media_mensal_laudos = monthly_totals.mean()

backlog_meses = (total_pend_laudos / media_mensal_laudos) if media_mensal_laudos > 0 else 0
taxa_conversao = (total_laudos / total_atendimentos) * 100 if total_atendimentos > 0 else 0

tme_mediano = tme_medio = sla_30_percent = sla_60_percent = None
if df_laudos_real is not None and not df_laudos_real.empty and "tme_dias" in df_laudos_real.columns:
    tme_values = df_laudos_real["tme_dias"].dropna()
    if not tme_values.empty:
        tme_mediano = tme_values.median()
        tme_medio = tme_values.mean()
        sla_30_percent = df_laudos_real["sla_30_ok"].mean() * 100
        sla_60_percent = df_laudos_real["sla_60_ok"].mean() * 100

aging_laudos_medio = aging_exames_medio = None
hoje = pd.Timestamp.now().normalize()
if df_pend_laudos is not None and not df_pend_laudos.empty and "data_base" in df_pend_laudos.columns:
    aging_laudos_medio = (hoje - df_pend_laudos["data_base"].dropna()).dt.days.mean()
if df_pend_exames is not None and not df_pend_exames.empty and "data_base" in df_pend_exames.columns:
    aging_exames_medio = (hoje - df_pend_exames["data_base"].dropna()).dt.days.mean()

# ============ LAYOUT DO DASHBOARD ============
st.subheader("📈 Indicadores Principais")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Atendimentos Totais", format_number(total_atendimentos))
with col2: st.metric("Laudos Emitidos", format_number(total_laudos))
with col3: st.metric("Taxa de Conversão", f"{format_number(taxa_conversao, 1)}%")
with col4: st.metric("Produtividade Mensal", format_number(media_mensal_laudos, 1))

st.markdown("#### ⏰ Gestão de Pendências")
col5, col6, col7, col8 = st.columns(4)
with col5: st.metric("Laudos Pendentes", format_number(total_pend_laudos))
with col6: st.metric("Exames Pendentes", format_number(total_pend_exames))
with col7: st.metric("Backlog (meses)", format_number(backlog_meses, 1))
with col8: st.metric("Aging Médio (dias)", format_number(aging_laudos_medio or aging_exames_medio, 0))

if tme_mediano is not None:
    st.markdown("#### 🎯 Indicadores de Performance (TME)")
    col9, col10, col11, col12 = st.columns(4)
    with col9: st.metric("TME Mediano (dias)", format_number(tme_mediano, 1))
    with col10: st.metric("TME Médio (dias)", format_number(tme_medio, 1))
    with col11: st.metric("SLA 30 dias", f"{format_number(sla_30_percent, 1)}%")
    with col12: st.metric("SLA 60 dias", f"{format_number(sla_60_percent, 1)}%")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Visão Geral", "📈 Tendências", "🏆 Rankings", "⏰ Pendências", "🎯 Performance", "📑 Relatórios", "📋 Dados"
])

# ============ ABA 1: VISÃO GERAL ============
with tab1:
    st.subheader("📊 Resumo Executivo")
    if df_laudos_todos is not None and not df_laudos_todos.empty:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 🏢 Performance por Unidade")
            if "unidade" in df_laudos_todos.columns:
                unidade_summary = df_laudos_todos.groupby("unidade")["quantidade"].sum().nlargest(15).reset_index()
                fig_unidades = px.bar(unidade_summary, x="quantidade", y="unidade", orientation="h", title="Top 15 Unidades - Laudos Emitidos", text="quantidade")
                st.plotly_chart(fig_unidades, use_container_width=True)
        with col_right:
            st.markdown("#### 🔍 Distribuição por Tipo")
            if "tipo" in df_laudos_todos.columns:
                tipo_summary = df_laudos_todos.groupby("tipo")["quantidade"].sum().nlargest(10).reset_index()
                fig_tipos = px.pie(tipo_summary, values="quantidade", names="tipo", title="Top 10 Tipos de Perícia")
                st.plotly_chart(fig_tipos, use_container_width=True)

# ============ ABA 2: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise de Tendências")
    df_plot = df_laudos_todos
    if df_plot is not None and not df_plot.empty and 'anomês' in df_plot.columns:
        monthly_data = df_plot.groupby("anomês")["quantidade"].sum().reset_index()
        fig = px.line(monthly_data, x="anomês", y="quantidade", title="Evolução Mensal de Laudos", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ============ ABA 3: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Comparativos")
    if df_laudos_real is not None and not df_laudos_real.empty:
        st.markdown("#### Performance por Perito")
        perito_perf = df_laudos_real.groupby('perito').agg(
            laudos_emitidos=('n_laudo', 'count'),
            tme_medio=('tme_dias', 'mean'),
            sla_30_aderencia=('sla_30_ok', lambda x: x.mean() * 100)
        ).dropna().sort_values('laudos_emitidos', ascending=False).reset_index()
        st.dataframe(perito_perf.head(25), use_container_width=True)

# ============ ABA 4: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Gestão de Pendências")
    if df_pend_laudos is not None and not df_pend_laudos.empty:
        laudos_aged, dist_laudos, stats_laudos = calculate_aging_analysis(df_pend_laudos)
        st.markdown(f"#### Análise de {stats_laudos['total']} Laudos Pendentes")
        fig = px.bar(dist_laudos, x=dist_laudos.index, y=dist_laudos.values, title="Distribuição por Tempo de Pendência", text=dist_laudos.values)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**🔴 Top 10 Mais Antigas:**")
        st.dataframe(laudos_aged.nlargest(10, "dias_pendentes")[['id', 'unidade', 'tipo', 'dias_pendentes', 'prioridade']], use_container_width=True)

# ============ ABA 5: PERFORMANCE ============
with tab5:
    st.subheader("🎯 Análise de Performance (TME & SLA)")
    if df_laudos_real is not None and not df_laudos_real.empty:
        st.markdown("#### TME Médio por Unidade")
        tme_unidade = df_laudos_real.groupby('unidade')['tme_dias'].mean().dropna().sort_values().reset_index()
        tme_unidade = tme_unidade[tme_unidade['tme_dias'] > 0]
        if not tme_unidade.empty:
            fig = px.bar(tme_unidade.head(20), x='tme_dias', y='unidade', orientation='h', title="Top 20 Unidades com Menor TME Médio", text='tme_dias')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

# ============ ABA 6: RELATÓRIOS ============
with tab6:
    st.subheader("📑 Relatórios Executivos")
    # Funções de geração de relatório
    def gerar_relatorio_executivo():
        return f"""
# RELATÓRIO EXECUTIVO
- **Atendimentos:** {format_number(total_atendimentos)}
- **Laudos:** {format_number(total_laudos)}
- **Pendências:** {format_number(total_pend_laudos)}
"""
    def gerar_relatorio_producao():
        if df_laudos_todos is None or df_laudos_todos.empty: return "Dados insuficientes."
        top_unidades = df_laudos_todos.groupby('unidade')['quantidade'].sum().nlargest(5).index.tolist()
        return f"# RELATÓRIO DE PRODUÇÃO\n**Top 5 Unidades:**\n- " + "\n- ".join(top_unidades)

    def gerar_relatorio_pendencias():
        if df_pend_laudos is None or df_pend_laudos.empty: return "Dados insuficientes."
        _, _, stats = calculate_aging_analysis(df_pend_laudos)
        return f"# RELATÓRIO DE PENDÊNCIAS\n- **Total:** {stats['total']}\n- **Média Dias:** {format_number(stats['media_dias'], 1)}"

    report_map = {
        "Relatório Executivo Completo": gerar_relatorio_executivo,
        "Relatório de Produção": gerar_relatorio_producao,
        "Relatório de Pendências": gerar_relatorio_pendencias,
    }
    tipo_relatorio = st.selectbox("Tipo de Relatório:", list(report_map.keys()))
    if tipo_relatorio:
        relatorio_texto = report_map[tipo_relatorio]()
        st.markdown(relatorio_texto)
        st.download_button("📥 Download", relatorio_texto, f"{tipo_relatorio.lower().replace(' ', '_')}.md")

# ============ ABA 7: DADOS ============
with tab7:
    st.subheader("📋 Exploração dos Dados")
    dataset_name = st.selectbox("Selecione o dataset para explorar:", list(standardized_dfs.keys()))
    if dataset_name and standardized_dfs[dataset_name] is not None:
        st.dataframe(standardized_dfs[dataset_name], use_container_width=True)
        csv = standardized_dfs[dataset_name].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download (CSV)", csv, f"{dataset_name}.csv", "text/csv")

# ============ RODAPÉ ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.1</strong> - Sistema Avançado de Monitoramento</p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)
