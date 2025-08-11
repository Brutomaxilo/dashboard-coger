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
    # Para arquivos PCI/SC, o separador principal é ponto e vírgula
    separators = [";", ",", "\t", "|"]
    encodings = ["utf-8", "latin-1", "cp1252"]
    
    for encoding in encodings:
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(bio, sep=sep, encoding=encoding, engine="python")
                # Verifica se realmente separou as colunas (mais de 1 coluna)
                if df.shape[1] > 1:
                    # Remove aspas extras se existirem
                    df.columns = [col.strip('"').strip() for col in df.columns]
                    
                    # Limpa dados das células também
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
    
    # Usar padrões específicos se disponíveis
    config = file_configs.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    
    # Adiciona o nome original também
    patterns.append(name.lower().replace(" ", "_"))
    
    for filename in os.listdir("data"):
        if not filename.lower().endswith(".csv"):
            continue
        
        base_name = os.path.splitext(filename)[0].lower()
        normalized_name = re.sub(r"[^\w]", "_", base_name)
        
        # Testa todos os padrões
        for pattern in patterns:
            if pattern in normalized_name or normalized_name.startswith(pattern):
                return os.path.join("data", filename)
    
    return None

# ============ DADOS SIMULADOS PARA DEMO ============
def create_sample_laudos_realizados() -> pd.DataFrame:
    """Cria dados simulados de laudos realizados baseados no screenshot."""
    
    # Dados de exemplo baseados no screenshot fornecido
    sample_data = []
    
    # Tipos de perícia do screenshot
    tipos_pericia = [
        "Química Forense", "Criminal Local de crime contra o patrimônio",
        "Criminal Local de crime contra a vida", "Criminal Engenharia Forense",
        "Criminal Identificação de veículos", "Criminal Identificação",
        "Informática Forense", "Balística", "Traumatologia Forense"
    ]
    
    # Unidades
    unidades = ["Joinville", "Florianópolis", "Blumenau", "Chapecó", "Criciúma"]
    
    # Diretorias
    diretorias = ["Diretoria Criminal", "Diretoria Cível", "Diretoria Administrativa"]
    
    # Peritos
    peritos = [
        "Alcides Ogliardi Junior", "Dr. Silva Santos", "Dra. Maria Oliveira",
        "Dr. João Pereira", "Dra. Ana Costa"
    ]
    
    # Gera dados simulados para os últimos 24 meses
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2024-01-01')
    
    np.random.seed(42)  # Para reprodutibilidade
    
    for i in range(500):  # 500 laudos simulados
        # Datas
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

# ============ CARREGAMENTO DE DADOS MELHORADO ============
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
            # Normaliza nomes das colunas
            df.columns = [
                re.sub(r"\s+", " ", col.strip().lower()) 
                for col in df.columns
            ]
            loaded_data[name] = df
    
    # Se não há dados de laudos realizados, cria dados simulados para demo
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

# ============ MAPEAMENTO DE COLUNAS CORRIGIDO ============
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
        "quantidade": "idatendimento"  # O ID representa a quantidade de atendimentos
    },
    "Atendimentos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "idatendimento",
        "quantidade": "idatendimento",  # O ID representa a quantidade de atendimentos
        "tipo": "txcompetencia"  # txcompetencia é o tipo de perícia
    },
    "Laudos_todos_Mensal": {
        "date": "data_interesse", 
        "id": "iddocumento",
        "quantidade": "iddocumento"  # O ID representa a quantidade de documentos
    },
    "Laudos_especifico_Mensal": {
        "date": "data_interesse",
        "competencia": "txcompetencia",
        "id": "iddocumento",
        "quantidade": "iddocumento",  # O ID representa a quantidade de documentos
        "tipo": "txcompetencia"  # txcompetencia é o tipo de perícia
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
    
    # Para arquivos mensais, o valor na coluna ID já representa a quantidade
    if name in ["Atendimentos_todos_Mensal", "Laudos_todos_Mensal", 
                "Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal"]:
        
        quantity_col = mapping.get("quantidade", mapping.get("id"))
        if quantity_col and quantity_col in result.columns:
            # Converte para numérico
            result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(0)
        else:
            result["quantidade"] = 1
    else:
        # Para outros arquivos, cada linha é uma unidade
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
        # Para txcompetencia, precisamos agrupar por data + tipo
        if mapping["competencia"] == "txcompetencia":
            date_col = mapping.get("date")
            if date_col and date_col in result.columns:
                date_series = process_datetime_column(result[date_col])
                if date_series is not None:
                    anomês_dt = date_series.dt.to_period("M").dt.to_timestamp()
        else:
            # Para outras competências, tenta converter diretamente
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
        if "dhemissao" in result.columns:
            base_date = (
                result.get("dhatendimento") 
                if "dhatendimento" in result.columns 
                else result.get("dhsolicitacao")
            )
            
            if base_date is not None:
                result["tme_dias"] = (result["dhemissao"] - base_date).dt.days
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
                .replace({"Nan": None, "": None, "None": None})
            )
    
    # Para arquivos específicos, agrupa por mês + tipo
    if name in ["Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal"]:
        if "anomês_dt" in result.columns and "tipo" in result.columns:
            # Já está no formato correto, apenas certifica que quantidade está correta
            pass
    
    return result
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
        if "dhemissao" in result.columns:
            base_date = (
                result.get("dhatendimento") 
                if "dhatendimento" in result.columns 
                else result.get("dhsolicitacao")
            )
            
            if base_date is not None:
                result["tme_dias"] = (result["dhemissao"] - base_date).dt.days
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
                .replace({"Nan": None, "": None, "None": None})
            )
    
    # Para arquivos específicos, agrupa por mês + tipo
    if name in ["Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal"]:
        if "anomês_dt" in result.columns and "tipo" in result.columns:
            # Já está no formato correto, apenas certifica que quantidade está correta
            pass
    
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

# ============ CÁLCULO DE KPIS CORRIGIDOS ============
def calculate_total(df: pd.DataFrame) -> int:
    """Calcula total considerando a coluna quantidade."""
    if df is None or df.empty or "quantidade" not in df.columns:
        return 0
    return int(df["quantidade"].sum())

def calculate_monthly_average(df: pd.DataFrame) -> Optional[float]:
    """Calcula média mensal considerando quantidade."""
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns:
        return None
    
    monthly_totals = df.groupby("anomês_dt")["quantidade"].sum()
    return monthly_totals.mean() if len(monthly_totals) > 0 else None

def calculate_growth_rate(df: pd.DataFrame, periods: int = 3) -> Optional[float]:
    """Calcula taxa de crescimento dos últimos períodos."""
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
    """Calcula métricas de produtividade comparativas."""
    metrics = {}
    
    if df_atend is not None and df_laudos is not None:
        # Taxa de conversão atendimento -> laudo
        total_atend = calculate_total(df_atend)
        total_laudos = calculate_total(df_laudos)
        
        if total_atend > 0:
            metrics["taxa_conversao"] = (total_laudos / total_atend) * 100
        
        # Eficiência temporal (se há dados temporais)
        if ("anomês_dt" in df_atend.columns and "anomês_dt" in df_laudos.columns):
            atend_monthly = df_atend.groupby("anomês_dt")["quantidade"].sum()
            laudos_monthly = df_laudos.groupby("anomês_dt")["quantidade"].sum()
            
            # Correlação entre atendimentos e laudos
            common_months = atend_monthly.index.intersection(laudos_monthly.index)
            if len(common_months) > 3:
                correlation = atend_monthly.loc[common_months].corr(laudos_monthly.loc[common_months])
                metrics["correlacao_atend_laudos"] = correlation
    
    return metrics

# Calcula KPIs principais
total_atendimentos = calculate_total(df_atend_todos)
total_laudos = calculate_total(df_laudos_todos)
total_pend_laudos = len(df_pend_laudos) if df_pend_laudos is not None and not df_pend_laudos.empty else 0
total_pend_exames = len(df_pend_exames) if df_pend_exames is not None and not df_pend_exames.empty else 0

# KPIs derivados
media_mensal_laudos = calculate_monthly_average(df_laudos_todos)
backlog_meses = (
    total_pend_laudos / media_mensal_laudos 
    if media_mensal_laudos and media_mensal_laudos > 0 
    else None
)

# Métricas de produtividade
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

# ============ EXIBIÇÃO DE KPIS MELHORADOS ============
st.subheader("📈 Indicadores Principais")

# Primeira linha de KPIs - Produção
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_atend = f"+{format_number(crescimento_atendimentos, 1)}%" if crescimento_atendimentos else None
    st.metric(
        "Atendimentos Totais",
        format_number(total_atendimentos),
        delta=delta_atend,
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
        "Taxa de Conversão",
        f"{format_number(taxa_atendimento, 1)}%" if taxa_atendimento else "—",
        help="Percentual de atendimentos que resultaram em laudos"
    )

with col4:
    st.metric(
        "Produtividade Mensal",
        format_number(media_mensal_laudos, 1) if media_mensal_laudos else "—",
        help="Média de laudos emitidos por mês"
    )

# Segunda linha de KPIs - Pendências
st.markdown("#### ⏰ Gestão de Pendências")
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
        help="Tempo estimado para liquidar pendências atuais baseado na produção média"
    )

with col8:
    aging_medio = aging_laudos_medio or aging_exames_medio
    st.metric(
        "Aging Médio (dias)",
        format_number(aging_medio, 0) if aging_medio else "—",
        help="Tempo médio de pendência dos processos em aberto"
    )

# Terceira linha - Performance e SLAs
if tme_mediano is not None or sla_30_percent is not None:
    st.markdown("#### 🎯 Indicadores de Performance")
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.metric(
            "TME Mediano (dias)",
            format_number(tme_mediano, 1) if tme_mediano else "—", 
            help="Tempo mediano de execução dos laudos (mais robusto que a média)"
        )
    
    with col10:
        st.metric(
            "TME Médio (dias)",
            format_number(tme_medio, 1) if tme_medio else "—",
            help="Tempo médio de execução dos laudos"
        )
    
    with col11:
        sla_30_delta = "normal" if sla_30_percent and sla_30_percent >= 80 else "inverse"
        st.metric(
            "SLA 30 dias",
            f"{format_number(sla_30_percent, 1)}%" if sla_30_percent else "—",
            help="Percentual de laudos emitidos em até 30 dias"
        )
    
    with col12:
        st.metric(
            "SLA 60 dias", 
            f"{format_number(sla_60_percent, 1)}%" if sla_60_percent else "—",
            help="Percentual de laudos emitidos em até 60 dias"
        )

# Alertas e insights automáticos
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
    st.subheader("📊 Resumo Executivo")
    
    # Métricas de eficiência
    if df_laudos_todos is not None and not df_laudos_todos.empty:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🏢 Performance por Unidade")
            
            if "unidade" in df_laudos_todos.columns:
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
                    title="Top 15 Unidades - Laudos Emitidos",
                    color="quantidade",
                    color_continuous_scale="Blues",
                    text="quantidade"
                )
                fig_unidades.update_traces(texttemplate='%{text}', textposition='outside')
                fig_unidades.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_unidades, use_container_width=True)
        
        with col_right:
            st.markdown("#### 🔍 Distribuição por Tipo")
            
            if "tipo" in df_laudos_todos.columns:
                tipo_summary = (
                    df_laudos_todos
                    .groupby("tipo", as_index=False)["quantidade"]
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
                fig_tipos.update_traces(textposition='inside', textinfo='percent+label')
                fig_tipos.update_layout(height=500)
                st.plotly_chart(fig_tipos, use_container_width=True)
    
    # Análise temporal consolidada
    if (df_atend_todos is not None and df_laudos_todos is not None and 
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        
        st.markdown("#### 📅 Evolução Temporal Consolidada")
        
        # Combina dados mensais
        atend_monthly = (
            df_atend_todos
            .groupby("anomês_dt")["quantidade"]
            .sum()
            .reset_index()
        )
        atend_monthly["Tipo"] = "Atendimentos"
        atend_monthly.rename(columns={"quantidade": "Total"}, inplace=True)
        
        laudos_monthly = (
            df_laudos_todos
            .groupby("anomês_dt")["quantidade"] 
            .sum()
            .reset_index()
        )
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
        
        fig_temporal.update_layout(
            height=400,
            hovermode="x unified",
            xaxis_title="Período",
            yaxis_title="Quantidade"
        )
        
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Taxa de conversão ao longo do tempo
        merged_monthly = pd.merge(
            atend_monthly.rename(columns={"Total": "Atendimentos"}),
            laudos_monthly.rename(columns={"Total": "Laudos"}),
            on="anomês_dt",
            how="inner"
        )
        
        if not merged_monthly.empty:
            merged_monthly["Taxa_Conversao"] = (
                merged_monthly["Laudos"] / merged_monthly["Atendimentos"] * 100
            )
            merged_monthly["Mês"] = merged_monthly["anomês_dt"].dt.strftime("%Y-%m")
            
            fig_conversao = px.line(
                merged_monthly,
                x="Mês",
                y="Taxa_Conversao",
                markers=True,
                title="Taxa de Conversão Mensal (%)",
                line_shape="spline"
            )
            
            # Adiciona linha de meta (70%)
            fig_conversao.add_hline(
                y=70, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Meta: 70%"
            )
            
            fig_conversao.update_layout(
                height=350,
                yaxis_title="Taxa de Conversão (%)",
                xaxis_title="Período"
            )
            
            st.plotly_chart(fig_conversao, use_container_width=True)

# ============ ABA 2: TENDÊNCIAS ============
with tab2:
    st.subheader("📈 Análise de Tendências")
    
    def create_enhanced_time_series(df: pd.DataFrame, title: str, color: str = "blue") -> None:
        """Cria gráfico de série temporal com análises avançadas."""
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
        
        # Cria subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(title, "Variação Percentual Mensal"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Gráfico principal com linha e média móvel
        fig.add_trace(
            go.Scatter(
                x=monthly_data["Mês"],
                y=monthly_data["quantidade"],
                mode="lines+markers",
                name="Valores",
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )
        
        # Média móvel (3 meses)
        if len(monthly_data) >= 3:
            monthly_data["media_movel"] = (
                monthly_data["quantidade"]
                .rolling(window=3, center=True)
                .mean()
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data["Mês"],
                    y=monthly_data["media_movel"],
                    mode="lines",
                    name="Média Móvel (3m)",
                    line=dict(dash="dash", color="red", width=2)
                ),
                row=1, col=1
            )
        
        # Variação percentual
        monthly_data["variacao_pct"] = monthly_data["quantidade"].pct_change() * 100
        
        colors = ['red' if x < 0 else 'green' for x in monthly_data["variacao_pct"].fillna(0)]
        
        fig.add_trace(
            go.Bar(
                x=monthly_data["Mês"],
                y=monthly_data["variacao_pct"],
                name="Variação %",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            hovermode="x unified",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Período", row=2, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos de tendência em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        create_enhanced_time_series(
            df_atend_todos,
            "🏥 Atendimentos - Análise Temporal",
            "blue"
        )
        
        # Análise sazonal para atendimentos
        if df_atend_todos is not None and "anomês_dt" in df_atend_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Atendimentos")
            
            seasonal_data = df_atend_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            
            monthly_totals = (
                seasonal_data
                .groupby(["mes_num", "mes_nome"])["quantidade"]
                .sum()
                .reset_index()
                .sort_values("mes_num")
            )
            
            fig_sazonal = px.bar(
                monthly_totals,
                x="mes_nome",
                y="quantidade",
                title="Distribuição Sazonal",
                color="quantidade",
                color_continuous_scale="Blues"
            )
            
            fig_sazonal.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sazonal, use_container_width=True)
    
    with col2:
        create_enhanced_time_series(
            df_laudos_todos,
            "📄 Laudos - Análise Temporal", 
            "green"
        )
        
        # Análise sazonal para laudos
        if df_laudos_todos is not None and "anomês_dt" in df_laudos_todos.columns:
            st.markdown("#### 📅 Sazonalidade - Laudos")
            
            seasonal_data = df_laudos_todos.copy()
            seasonal_data["mes_nome"] = seasonal_data["anomês_dt"].dt.month_name()
            seasonal_data["mes_num"] = seasonal_data["anomês_dt"].dt.month
            
            monthly_totals = (
                seasonal_data
                .groupby(["mes_num", "mes_nome"])["quantidade"]
                .sum()
                .reset_index()
                .sort_values("mes_num")
            )
            
            fig_sazonal = px.bar(
                monthly_totals,
                x="mes_nome",
                y="quantidade",
                title="Distribuição Sazonal",
                color="quantidade",
                color_continuous_scale="Greens"
            )
            
            fig_sazonal.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sazonal, use_container_width=True)
    
    # Análise de correlação
    if (df_atend_todos is not None and df_laudos_todos is not None and 
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        
        st.markdown("#### 🔗 Análise de Correlação")
        
        atend_monthly = df_atend_todos.groupby("anomês_dt")["quantidade"].sum()
        laudos_monthly = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum()
        
        # Períodos em comum
        common_periods = atend_monthly.index.intersection(laudos_monthly.index)
        
        if len(common_periods) > 3:
            correlation_data = pd.DataFrame({
                "Atendimentos": atend_monthly.loc[common_periods],
                "Laudos": laudos_monthly.loc[common_periods]
            }).reset_index()
            
            correlation_data["Período"] = correlation_data["anomês_dt"].dt.strftime("%Y-%m")
            
            fig_scatter = px.scatter(
                correlation_data,
                x="Atendimentos",
                y="Laudos",
                hover_data=["Período"],
                title="Correlação: Atendimentos vs Laudos",
                trendline="ols"
            )
            
            correlation_coef = correlation_data["Atendimentos"].corr(correlation_data["Laudos"])
            fig_scatter.add_annotation(
                text=f"Correlação: {correlation_coef:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)"
            )
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ============ ABA 3: RANKINGS ============
with tab3:
    st.subheader("🏆 Rankings e Comparativos")
    
    def create_enhanced_ranking(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20) -> None:
        """Cria gráfico de ranking com informações adicionais."""
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        ranking_data = (
            df.groupby(dimension)
            .agg({
                "quantidade": ["sum", "count", "mean"]
            })
            .round(2)
        )
        
        ranking_data.columns = ["Total", "Registros", "Média"]
        ranking_data = ranking_data.sort_values("Total", ascending=False).head(top_n)
        ranking_data.reset_index(inplace=True)
        
        if ranking_data.empty:
            st.info(f"Sem dados para {title}")
            return
        
        # Gráfico principal
        fig = px.bar(
            ranking_data,
            x="Total",
            y=dimension,
            orientation="h",
            title=title,
            color="Total",
            color_continuous_scale="Viridis",
            hover_data=["Registros", "Média"]
        )
        
        fig.update_layout(
            height=max(400, len(ranking_data) * 30),
            showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de detalhes
        with st.expander(f"📊 Detalhes - {title}"):
            st.dataframe(ranking_data, use_container_width=True)
    
    # Rankings em abas
    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "Por Diretoria", "Por Unidade", "Por Tipo", "Comparativo"
    ])
    
    with rank_tab1:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(
                df_atend_todos,
                "diretoria", 
                "🏥 Atendimentos por Diretoria"
            )
        with col2:
            create_enhanced_ranking(
                df_laudos_todos,
                "diretoria",
                "📄 Laudos por Diretoria"
            )
    
    with rank_tab2:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(
                df_atend_todos,
                "unidade",
                "🏥 Atendimentos por Unidade",
                25
            )
        with col2:
            create_enhanced_ranking(
                df_laudos_todos,
                "unidade", 
                "📄 Laudos por Unidade",
                25
            )
    
    with rank_tab3:
        col1, col2 = st.columns(2)
        with col1:
            create_enhanced_ranking(
                df_atend_esp,
                "tipo",
                "🏥 Atendimentos por Tipo",
                20
            )
        with col2:
            create_enhanced_ranking(
                df_laudos_esp,
                "tipo",
                "📄 Laudos por Tipo",
                20
            )
    
    with rank_tab4:
        st.markdown("#### 📊 Análise Comparativa de Eficiência")
        
        # Compara eficiência por unidade
        if (df_atend_todos is not None and df_laudos_todos is not None and
            "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns):
            
            atend_por_unidade = (
                df_atend_todos
                .groupby("unidade")["quantidade"]
                .sum()
                .reset_index()
                .rename(columns={"quantidade": "Atendimentos"})
            )
            
            laudos_por_unidade = (
                df_laudos_todos
                .groupby("unidade")["quantidade"]
                .sum()
                .reset_index()
                .rename(columns={"quantidade": "Laudos"})
            )
            
            eficiencia_data = pd.merge(
                atend_por_unidade,
                laudos_por_unidade,
                on="unidade",
                how="inner"
            )
            
            if not eficiencia_data.empty:
                eficiencia_data["Taxa_Conversao"] = (
                    eficiencia_data["Laudos"] / eficiencia_data["Atendimentos"] * 100
                )
                eficiencia_data = eficiencia_data.sort_values("Taxa_Conversao", ascending=False)
                
                fig_eficiencia = px.scatter(
                    eficiencia_data.head(20),
                    x="Atendimentos",
                    y="Laudos",
                    size="Taxa_Conversao",
                    hover_name="unidade",
                    title="Eficiência por Unidade (Atendimentos vs Laudos)",
                    color="Taxa_Conversao",
                    color_continuous_scale="RdYlGn"
                )
                
                fig_eficiencia.update_layout(height=500)
                st.plotly_chart(fig_eficiencia, use_container_width=True)
                
                # Top 10 mais eficientes
                st.markdown("**🥇 Top 10 Unidades Mais Eficientes:**")
                top_eficientes = eficiencia_data.head(10)[["unidade", "Taxa_Conversao", "Atendimentos", "Laudos"]]
                st.dataframe(top_eficientes, use_container_width=True)

# ============ ABA 4: PENDÊNCIAS ============
with tab4:
    st.subheader("⏰ Gestão de Pendências")
    
    def calculate_aging_analysis(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Calcula aging das pendências com análises detalhadas."""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series(dtype="int64"), {}
        
        # Busca coluna de data válida
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
        
        # Categorias de aging mais detalhadas
        faixas_aging = pd.cut(
            dias_pendentes,
            bins=[-1, 15, 30, 60, 90, 180, 365, float('inf')],
            labels=["0-15 dias", "16-30 dias", "31-60 dias", 
                   "61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"]
        )
        
        result["dias_pendentes"] = dias_pendentes
        result["faixa_aging"] = faixas_aging
        
        # Cria categoria de prioridade
        result["prioridade"] = pd.cut(
            dias_pendentes,
            bins=[-1, 30, 90, 180, float('inf')],
            labels=["Normal", "Atenção", "Urgente", "Crítico"]
        )
        
        distribuicao = faixas_aging.value_counts().sort_index()
        
        # Estatísticas adicionais
        stats = {
            "total": len(result),
            "media_dias": dias_pendentes.mean(),
            "mediana_dias": dias_pendentes.median(),
            "max_dias": dias_pendentes.max(),
            "criticos": len(result[result["prioridade"] == "Crítico"]),
            "urgentes": len(result[result["prioridade"] == "Urgente"])
        }
        
        return result, distribuicao, stats
    
    # Análise de Laudos Pendentes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Laudos Pendentes")
        
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            laudos_aged, dist_laudos, stats_laudos = calculate_aging_analysis(df_pend_laudos)
            
            # Métricas principais
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total", format_number(stats_laudos.get("total", 0)))
            with col_b:
                st.metric("Críticos", stats_laudos.get("criticos", 0))
            with col_c:
                st.metric("Média (dias)", format_number(stats_laudos.get("media_dias", 0), 1))
            
            # Gráfico de distribuição por aging
            if not dist_laudos.empty:
                fig_aging_laudos = px.bar(
                    x=dist_laudos.index,
                    y=dist_laudos.values,
                    title="Distribuição por Tempo de Pendência",
                    color=dist_laudos.values,
                    color_continuous_scale="Reds",
                    text=dist_laudos.values
                )
                fig_aging_laudos.update_traces(texttemplate='%{text}', textposition='outside')
                fig_aging_laudos.update_layout(
                    height=350,
                    showlegend=False,
                    xaxis_title="Faixa de Dias",
                    yaxis_title="Quantidade"
                )
                st.plotly_chart(fig_aging_laudos, use_container_width=True)
            
            # Gráfico de prioridades
            if "prioridade" in laudos_aged.columns:
                prioridade_dist = laudos_aged["prioridade"].value_counts()
                
                fig_prioridade = px.pie(
                    values=prioridade_dist.values,
                    names=prioridade_dist.index,
                    title="Distribuição por Prioridade",
                    color_discrete_map={
                        "Normal": "green",
                        "Atenção": "yellow", 
                        "Urgente": "orange",
                        "Crítico": "red"
                    }
                )
                fig_prioridade.update_layout(height=300)
                st.plotly_chart(fig_prioridade, use_container_width=True)
            
            # Top pendências mais antigas
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in laudos_aged.columns:
                oldest = (
                    laudos_aged
                    .nlargest(10, "dias_pendentes")
                    [["id", "unidade", "tipo", "dias_pendentes", "prioridade"]]
                    if all(col in laudos_aged.columns for col in ["id", "unidade", "tipo"])
                    else laudos_aged.nlargest(10, "dias_pendentes")
                )
                st.dataframe(oldest, use_container_width=True, height=250)
        else:
            st.info("Sem dados de laudos pendentes disponíveis.")
    
    with col2:
        st.markdown("#### 🔬 Exames Pendentes")
        
        if df_pend_exames is not None and not df_pend_exames.empty:
            exames_aged, dist_exames, stats_exames = calculate_aging_analysis(df_pend_exames)
            
            # Métricas principais
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total", format_number(stats_exames.get("total", 0)))
            with col_b:
                st.metric("Críticos", stats_exames.get("criticos", 0))
            with col_c:
                st.metric("Média (dias)", format_number(stats_exames.get("media_dias", 0), 1))
            
            # Gráfico de distribuição por aging
            if not dist_exames.empty:
                fig_aging_exames = px.bar(
                    x=dist_exames.index,
                    y=dist_exames.values,
                    title="Distribuição por Tempo de Pendência",
                    color=dist_exames.values,
                    color_continuous_scale="Oranges",
                    text=dist_exames.values
                )
                fig_aging_exames.update_traces(texttemplate='%{text}', textposition='outside')
                fig_aging_exames.update_layout(
                    height=350,
                    showlegend=False,
                    xaxis_title="Faixa de Dias",
                    yaxis_title="Quantidade"
                )
                st.plotly_chart(fig_aging_exames, use_container_width=True)
            
            # Gráfico de prioridades
            if "prioridade" in exames_aged.columns:
                prioridade_dist = exames_aged["prioridade"].value_counts()
                
                fig_prioridade = px.pie(
                    values=prioridade_dist.values,
                    names=prioridade_dist.index,
                    title="Distribuição por Prioridade",
                    color_discrete_map={
                        "Normal": "green",
                        "Atenção": "yellow",
                        "Urgente": "orange", 
                        "Crítico": "red"
                    }
                )
                fig_prioridade.update_layout(height=300)
                st.plotly_chart(fig_prioridade, use_container_width=True)
            
            # Top pendências mais antigas
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in exames_aged.columns:
                oldest = (
                    exames_aged
                    .nlargest(10, "dias_pendentes")
                    [["id", "unidade", "tipo", "dias_pendentes", "prioridade"]]
                    if all(col in exames_aged.columns for col in ["id", "unidade", "tipo"])
                    else exames_aged.nlargest(10, "dias_pendentes")
                )
                st.dataframe(oldest, use_container_width=True, height=250)
        else:
            st.info("Sem dados de exames pendentes disponíveis.")
    
    # Análise consolidada por unidade
    st.markdown("#### 🏢 Análise de Pendências por Unidade")
    
    pendencias_por_unidade = []
    
    # Processa laudos pendentes
    if df_pend_laudos is not None and "unidade" in df_pend_laudos.columns:
        laudos_unidade = (
            df_pend_laudos.groupby("unidade")
            .agg({
                "quantidade": "count" if "quantidade" in df_pend_laudos.columns else lambda x: len(x)
            })
            .rename(columns={"quantidade": "Laudos_Pendentes"})
            .reset_index()
        )
        if "Laudos_Pendentes" not in laudos_unidade.columns:
            laudos_unidade["Laudos_Pendentes"] = df_pend_laudos.groupby("unidade").size().values
        pendencias_por_unidade.append(laudos_unidade)
    
    # Processa exames pendentes
    if df_pend_exames is not None and "unidade" in df_pend_exames.columns:
        exames_unidade = (
            df_pend_exames.groupby("unidade")
            .agg({
                "quantidade": "count" if "quantidade" in df_pend_exames.columns else lambda x: len(x)
            })
            .rename(columns={"quantidade": "Exames_Pendentes"})
            .reset_index()
        )
        if "Exames_Pendentes" not in exames_unidade.columns:
            exames_unidade["Exames_Pendentes"] = df_pend_exames.groupby("unidade").size().values
        pendencias_por_unidade.append(exames_unidade)
    
    if pendencias_por_unidade:
        from functools import reduce
        
        pendencias_consolidadas = reduce(
            lambda left, right: pd.merge(left, right, on="unidade", how="outer"),
            pendencias_por_unidade
        ).fillna(0)
        
        pendencias_consolidadas["Total_Pendencias"] = (
            pendencias_consolidadas.get("Laudos_Pendentes", 0) + 
            pendencias_consolidadas.get("Exames_Pendentes", 0)
        )
        
        pendencias_consolidadas = pendencias_consolidadas.sort_values("Total_Pendencias", ascending=False)
        
        # Gráfico de barras empilhadas
        fig_pendencias = go.Figure()
        
        if "Laudos_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(
                name='Laudos Pendentes',
                y=pendencias_consolidadas["unidade"].head(15),
                x=pendencias_consolidadas["Laudos_Pendentes"].head(15),
                orientation='h',
                marker_color='lightcoral'
            ))
        
        if "Exames_Pendentes" in pendencias_consolidadas.columns:
            fig_pendencias.add_trace(go.Bar(
                name='Exames Pendentes',
                y=pendencias_consolidadas["unidade"].head(15),
                x=pendencias_consolidadas["Exames_Pendentes"].head(15),
                orientation='h',
                marker_color='lightsalmon'
            ))
        
        fig_pendencias.update_layout(
            title="Top 15 Unidades com Mais Pendências",
            barmode='stack',
            height=500,
            xaxis_title="Quantidade de Pendências",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_pendencias, use_container_width=True)
        
        # Tabela de detalhes
        st.markdown("**📊 Detalhamento por Unidade:**")
        st.dataframe(
            pendencias_consolidadas.head(20),
            use_container_width=True,
            height=300
        )

# ============ ABA 5: DADOS ============
with tab5:
    st.subheader("📋 Exploração dos Dados")
    
    # Resumo geral dos datasets
    st.markdown("#### 📊 Resumo dos Datasets Carregados")
    
    data_summary = []
    for name, df in standardized_dfs.items():
        if df is not None and not df.empty:
            periodo_info = "Sem dados temporais"
            if 'anomês' in df.columns and not df['anomês'].isna().all():
                min_periodo = df['anomês'].min()
                max_periodo = df['anomês'].max()
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
        
        # Estatísticas gerais
        total_registros = sum(int(row["Registros"].replace(".", "")) for row in data_summary)
        total_tamanho = sum(row["Tamanho (MB)"] for row in data_summary)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", f"{total_registros:,}".replace(",", "."))
        with col2:
            st.metric("Datasets Carregados", len(data_summary))
        with col3:
            st.metric("Tamanho Total (MB)", f"{total_tamanho:.1f}")
        with col4:
            avg_size = total_tamanho / len(data_summary) if data_summary else 0
            st.metric("Tamanho Médio (MB)", f"{avg_size:.1f}")
    
    st.markdown("#### 🔍 Exploração Detalhada")
    
    # Seletor de dataset
    available_datasets = [name for name, df in standardized_dfs.items() if df is not None]
    
    if available_datasets:
        selected_dataset = st.selectbox(
            "Selecione o dataset para explorar:",
            available_datasets,
            format_func=lambda x: x.replace("_", " ").title(),
            help="Escolha qual conjunto de dados deseja analisar em detalhes"
        )
        
        if selected_dataset:
            df_selected = standardized_dfs[selected_dataset]
            
            # Informações básicas do dataset
            st.markdown(f"#### 📄 {selected_dataset.replace('_', ' ').title()}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registros", f"{len(df_selected):,}".replace(",", "."))
            with col2:
                st.metric("Colunas", len(df_selected.columns))
            with col3:
                valores_nulos = df_selected.isnull().sum().sum()
                st.metric("Valores Nulos", f"{valores_nulos:,}".replace(",", "."))
            with col4:
                if 'anomês_dt' in df_selected.columns:
                    unique_months = df_selected['anomês_dt'].nunique()
                    st.metric("Meses Únicos", unique_months)
                else:
                    st.metric("Período", "N/A")
            
            # Análise de qualidade dos dados
            with st.expander("🔍 Análise de Qualidade dos Dados", expanded=False):
                quality_info = []
                
                for col in df_selected.columns:
                    dtype = str(df_selected[col].dtype)
                    null_count = df_selected[col].isnull().sum()
                    null_percent = (null_count / len(df_selected)) * 100
                    unique_count = df_selected[col].nunique()
                    
                    # Determina qualidade da coluna
                    if null_percent == 0:
                        quality = "🟢 Excelente"
                    elif null_percent < 5:
                        quality = "🟡 Boa"
                    elif null_percent < 20:
                        quality = "🟠 Regular"
                    else:
                        quality = "🔴 Ruim"
                    
                    quality_info.append({
                        "Coluna": col,
                        "Tipo": dtype,
                        "Nulos": f"{null_count:,}".replace(",", "."),
                        "% Nulos": f"{null_percent:.1f}%",
                        "Únicos": f"{unique_count:,}".replace(",", "."),
                        "Qualidade": quality
                    })
                
                quality_df = pd.DataFrame(quality_info)
                st.dataframe(quality_df, use_container_width=True)
            
            # Filtros para visualização
            st.markdown("**🎛️ Controles de Visualização:**")
            
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                max_rows = st.number_input(
                    "Máximo de linhas:",
                    min_value=10,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Número máximo de linhas a exibir"
                )
            
            with viz_col2:
                if 'anomês' in df_selected.columns:
                    available_months = sorted(df_selected['anomês'].dropna().unique(), reverse=True)
                    selected_months = st.multiselect(
                        "Filtrar por período:",
                        available_months,
                        default=available_months[:6] if len(available_months) > 6 else available_months,
                        help="Selecione os meses para análise"
                    )
                else:
                    selected_months = []
            
            with viz_col3:
                # Filtro por colunas
                all_columns = list(df_selected.columns)
                selected_columns = st.multiselect(
                    "Colunas a exibir:",
                    all_columns,
                    default=all_columns[:10] if len(all_columns) > 10 else all_columns,
                    help="Selecione as colunas para visualização"
                )
            
            # Aplica filtros
            df_display = df_selected.copy()
            
            if selected_months and 'anomês' in df_display.columns:
                df_display = df_display[df_display['anomês'].isin(selected_months)]
            
            if selected_columns:
                df_display = df_display[selected_columns]
            
            df_display = df_display.head(max_rows)
            
            # Estatísticas descritivas
            if not df_display.empty:
                st.markdown("**📈 Estatísticas Descritivas:**")
                
                # Apenas colunas numéricas
                numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    stats = df_display[numeric_cols].describe().round(2)
                    st.dataframe(stats, use_container_width=True)
                else:
                    st.info("Nenhuma coluna numérica encontrada para estatísticas.")
            
            # Exibe os dados
            st.markdown(f"**📋 Dados Filtrados ({len(df_display):,} de {len(df_selected):,} registros):**".replace(",", "."))
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )
            
            # Opções de download
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                # Download dos dados filtrados
                csv_data = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dados Filtrados (CSV)",
                    data=csv_data,
                    file_name=f"{selected_dataset}_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Baixa apenas os dados atualmente visíveis"
                )
            
            with col_down2:
                # Download do dataset completo
                csv_complete = df_selected.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset Completo (CSV)",
                    data=csv_complete,
                    file_name=f"{selected_dataset}_completo_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Baixa o dataset completo sem filtros"
                )

# ============ ABA 6: RELATÓRIOS ============
with tab6:
    st.subheader("📑 Relatórios Executivos")
    
    # Seletor de tipo de relatório
    tipo_relatorio = st.selectbox(
        "Tipo de Relatório:",
        [
            "Relatório Executivo Completo",
            "Relatório de Produção",
            "Relatório de Pendências", 
            "Relatório de Performance",
            "Relatório Comparativo"
        ],
        help="Escolha o tipo de relatório que deseja gerar"
    )
    
    # Função para gerar relatórios
    def gerar_relatorio_executivo() -> str:
        """Gera relatório executivo completo."""
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
        
        # Adiciona análise de crescimento
        if crescimento_laudos is not None:
            if crescimento_laudos > 5:
                relatorio += f"- **Crescimento Positivo:** Laudos cresceram {format_number(crescimento_laudos, 1)}% no período\n"
            elif crescimento_laudos < -5:
                relatorio += f"- **Alerta:** Laudos decresceram {format_number(abs(crescimento_laudos), 1)}% no período\n"
            else:
                relatorio += f"- **Estabilidade:** Variação de {format_number(crescimento_laudos, 1)}% nos laudos\n"
        
        # Adiciona alertas
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
        
        relatorio += f"""

## 📋 DATASETS UTILIZADOS
"""
        
        for name, df in standardized_dfs.items():
            if df is not None and not df.empty:
                relatorio += f"- **{name.replace('_', ' ').title()}:** {len(df):,} registros\n"
        
        relatorio += f"""

---
*Relatório gerado automaticamente pelo Dashboard PCI/SC*
*Sistema de Monitoramento de Produção e Pendências*
        """
        
        return relatorio.strip()
    
    # Gera relatório baseado na seleção
    if tipo_relatorio == "Relatório Executivo Completo":
        relatorio_texto = gerar_relatorio_executivo()
        
        st.markdown("#### 📄 Visualização do Relatório")
        st.markdown(relatorio_texto)
        
        # Download do relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Relatório Executivo",
            data=relatorio_texto.encode('utf-8'),
            file_name=f"relatorio_executivo_pci_sc_{timestamp}.md",
            mime="text/markdown",
            help="Baixa o relatório em formato Markdown"
        )
    
    elif tipo_relatorio == "Relatório de Produção":
        st.markdown("#### 📊 Relatório de Produção")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Métricas de Produção:**")
            if df_laudos_todos is not None:
                prod_mensal = (
                    df_laudos_todos
                    .groupby("anomês")["quantidade"]
                    .sum()
                    .reset_index()
                    .sort_values("anomês")
                )
                
                st.line_chart(
                    prod_mensal.set_index("anomês")["quantidade"],
                    height=300
                )
        
        with col2:
            st.markdown("**Top Produtores:**")
            if df_laudos_todos is not None and "unidade" in df_laudos_todos.columns:
                top_unidades = (
                    df_laudos_todos
                    .groupby("unidade")["quantidade"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                
                st.bar_chart(top_unidades, height=300)
    
    # Continua com outros tipos de relatório...
    else:
        st.info(f"Relatório '{tipo_relatorio}' em desenvolvimento.")

# ============ RODAPÉ ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
    <p><strong>Dashboard PCI/SC v2.0</strong> - Sistema Avançado de Monitoramento</p>
    <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
    <p>Para suporte técnico ou sugestões: <strong>equipe-ti@pci.sc.gov.br</strong></p>
    <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)

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
