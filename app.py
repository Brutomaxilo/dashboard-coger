import io
import os
import re
import warnings
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurações de warnings e performance
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ============ CONFIGURAÇÃO INICIAL ============
st.set_page_config(
    page_title="PCI/SC – Dashboard Executivo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥",
    menu_items={
        'Get Help': 'mailto:equipe-ti@pci.sc.gov.br',
        'Report a bug': 'mailto:equipe-ti@pci.sc.gov.br',
        'About': "Dashboard Executivo PCI/SC v4.0 - Sistema Avançado de Monitoramento"
    }
)

# ============ CONFIGURAÇÕES GLOBAIS ============
@dataclass
class DashboardConfig:
    """Configurações centralizadas do dashboard"""
    VERSION = "4.0.0"
    COMPANY = "PCI/SC"
    CACHE_TTL = 3600
    DEFAULT_CHART_HEIGHT = 450
    MAX_UPLOAD_SIZE = 100  # MB
    
    # Cores do tema
    COLORS = {
        'primary': '#1f2937',
        'secondary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#06b6d4',
        'light': '#f8fafc',
        'dark': '#111827'
    }
    
    # Metas e benchmarks
    BENCHMARKS = {
        'taxa_conversao_excelente': 80,
        'taxa_conversao_boa': 70,
        'taxa_conversao_minima': 50,
        'backlog_critico': 6,  # meses
        'backlog_atencao': 3,  # meses
        'aging_critico': 90,   # dias
        'aging_atencao': 60    # dias
    }

config = DashboardConfig()

# === ESTILOS CSS MODERNOS ===
MODERN_CSS = f"""
<style>
/* Reset e base */
.main {{
    padding-top: 1rem;
}}

/* Design System - Cards */
.metric-card {{
    background: linear-gradient(135deg, #ffffff 0%, {config.COLORS['light']} 100%);
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, {config.COLORS['secondary']}, {config.COLORS['success']});
}}

.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}}

.metric-title {{
    font-size: 0.875rem;
    color: #6b7280;
    font-weight: 600;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.metric-value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: {config.COLORS['primary']};
    margin: 8px 0 4px 0;
    line-height: 1;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

.metric-delta {{
    font-size: 0.875rem;
    font-weight: 600;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.metric-delta.positive {{ color: {config.COLORS['success']}; }}
.metric-delta.negative {{ color: {config.COLORS['danger']}; }}
.metric-delta.neutral {{ color: #6b7280; }}

/* Alertas modernos */
.alert {{
    padding: 16px 20px;
    border-radius: 12px;
    margin: 16px 0;
    border-left: 4px solid;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}}

.alert::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    pointer-events: none;
}}

.alert-success {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
    border-left-color: {config.COLORS['success']};
    color: #065f46;
}}

.alert-warning {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
    border-left-color: {config.COLORS['warning']};
    color: #92400e;
}}

.alert-danger {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
    border-left-color: {config.COLORS['danger']};
    color: #991b1b;
}}

.alert-info {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(8, 145, 178, 0.05) 100%);
    border-left-color: {config.COLORS['info']};
    color: #0c4a6e;
}}

/* Títulos de seção */
.section-header {{
    margin: 32px 0 20px 0;
    padding: 0 0 12px 0;
    border-bottom: 2px solid #e5e7eb;
    color: {config.COLORS['primary']};
    font-weight: 700;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}}

/* Header principal */
.main-header {{
    background: linear-gradient(135deg, {config.COLORS['primary']} 0%, #374151 100%);
    color: white;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 32px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}}

.main-header h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}}

.main-header p {{
    font-size: 1.125rem;
    opacity: 0.9;
    margin: 0;
}}

/* Status badges */
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    backdrop-filter: blur(10px);
}}

.status-excellent {{ background: rgba(16, 185, 129, 0.2); color: {config.COLORS['success']}; }}
.status-good {{ background: rgba(245, 158, 11, 0.2); color: {config.COLORS['warning']}; }}
.status-poor {{ background: rgba(239, 68, 68, 0.2); color: {config.COLORS['danger']}; }}

/* Responsividade */
@media (max-width: 768px) {{
    .metric-card {{
        padding: 16px;
    }}
    
    .metric-value {{
        font-size: 1.875rem;
    }}
    
    .main-header {{
        padding: 24px;
    }}
    
    .main-header h1 {{
        font-size: 2rem;
    }}
}}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ============ UTILITÁRIOS AVANÇADOS ============
class DataProcessor:
    """Classe para processamento avançado de dados"""
    
    @staticmethod
    def detect_encoding(file_content: bytes) -> str:
        """Detecta encoding do arquivo com tratamento para BOM"""
        # Verificar BOM (Byte Order Mark)
        if file_content.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        elif file_content.startswith(b'\xff\xfe'):
            return 'utf-16-le'
        elif file_content.startswith(b'\xfe\xff'):
            return 'utf-16-be'
        
        try:
            import chardet
            result = chardet.detect(file_content)
            detected_encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Se a confiança é baixa, usar utf-8-sig como fallback
            if confidence < 0.7:
                return 'utf-8-sig'
            
            return detected_encoding
        except ImportError:
            # Fallback sem chardet
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8-sig'
    
    @staticmethod  
    def smart_csv_reader(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Leitor otimizado especificamente para os arquivos PCI/SC"""
        
        st.write(f"🔍 Processando: {filename} ({len(file_content)} bytes)")
        
        # Configurações específicas para os dados do PCI/SC
        try:
            # Primeiro, tentar a configuração mais provável baseada na prévia
            df = pd.read_csv(
                io.BytesIO(file_content),
                sep=';',
                encoding='utf-8-sig',
                engine='python',
                quotechar='"',
                doublequote=True,
                skipinitialspace=True,
                skip_blank_lines=True,
                on_bad_lines='skip',
                low_memory=False
            )
            
            # Verificar se o resultado faz sentido
            if df.shape[1] >= 2 and len(df) > 0:
                # Limpeza das colunas
                df.columns = [str(col).strip().strip('"').strip() for col in df.columns]
                
                # Remover linhas completamente vazias
                df = df.dropna(how='all')
                
                # Limpeza específica dos dados
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Limpar strings
                        df[col] = df[col].astype(str).str.strip().str.strip('"')
                        
                        # Tratar valores especiais
                        df[col] = df[col].replace(['nan', 'NaN', 'None', ''], None)
                        
                        # Conversão numérica para IDs e quantidades
                        if any(keyword in col.lower() for keyword in ['id', 'quantidade', 'numero']):
                            # Remover .0 desnecessário e converter para numérico
                            df[col] = df[col].str.replace('.0', '', regex=False)
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                st.success(f"✅ {filename} processado com sucesso!")
                st.success(f"   📊 {len(df)} registros, {len(df.columns)} colunas")
                st.write(f"   📋 Colunas: {', '.join(df.columns)}")
                
                # Mostrar amostra dos dados para validação
                if len(df) > 0:
                    st.write("📊 Amostra dos primeiros registros:")
                    sample_df = df.head(3).copy()
                    
                    # Limitar o tamanho das strings para exibição
                    for col in sample_df.columns:
                        if sample_df[col].dtype == 'object':
                            sample_df[col] = sample_df[col].astype(str).apply(
                                lambda x: x[:50] + '...' if len(str(x)) > 50 else x
                            )
                    
                    st.dataframe(sample_df, use_container_width=True)
                
                return df
            
        except Exception as e:
            st.warning(f"⚠️ Erro na primeira tentativa: {str(e)}")
        
        # Se falhou, tentar outras configurações
        configurations = [
            {'sep': ';', 'encoding': 'utf-8'},
            {'sep': ';', 'encoding': 'latin-1'},
            {'sep': ',', 'encoding': 'utf-8-sig'},
            {'sep': ',', 'encoding': 'utf-8'},
            {'sep': '\t', 'encoding': 'utf-8-sig'},
        ]
        
        for config in configurations:
            try:
                st.write(f"🔄 Tentando: separador='{config['sep']}', encoding='{config['encoding']}'")
                
                df = pd.read_csv(
                    io.BytesIO(file_content),
                    sep=config['sep'],
                    encoding=config['encoding'],
                    engine='python',
                    quotechar='"',
                    doublequote=True,
                    skipinitialspace=True,
                    skip_blank_lines=True,
                    on_bad_lines='skip',
                    low_memory=False
                )
                
                if df.shape[1] >= 2 and len(df) > 0:
                    # Aplicar a mesma limpeza
                    df.columns = [str(col).strip().strip('"').strip() for col in df.columns]
                    df = df.dropna(how='all')
                    
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip().str.strip('"')
                            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], None)
                            
                            if any(keyword in col.lower() for keyword in ['id', 'quantidade', 'numero']):
                                df[col] = df[col].str.replace('.0', '', regex=False)
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    st.success(f"✅ Sucesso com separador='{config['sep']}', encoding='{config['encoding']}'")
                    st.success(f"   📊 {len(df)} registros, {len(df.columns)} colunas")
                    
                    return df
                    
            except Exception as e:
                continue
        
        # Se ainda não funcionou, fornecer diagnóstico
        st.error(f"❌ Não foi possível processar {filename}")
        
        # Análise detalhada do problema
        try:
            # Tentar diferentes encodings para mostrar conteúdo
            for encoding in ['utf-8-sig', 'utf-8', 'latin-1']:
                try:
                    content_str = file_content.decode(encoding)
                    lines = content_str.split('\n')
                    
                    st.write(f"📝 Análise com encoding {encoding}:")
                    st.write(f"   - Total de linhas: {len(lines)}")
                    st.write(f"   - Header: {lines[0][:100] if lines else 'Vazio'}")
                    st.write(f"   - Primeira linha de dados: {lines[1][:100] if len(lines) > 1 else 'Não existe'}")
                    
                    # Verificar separadores na primeira linha
                    if lines:
                        header = lines[0]
                        sep_counts = {
                            ';': header.count(';'),
                            ',': header.count(','),
                            '\t': header.count('\t'),
                            '|': header.count('|')
                        }
                        st.write(f"   - Contagem de separadores: {sep_counts}")
                    
                    break
                    
                except UnicodeDecodeError:
                    continue
                    
        except Exception as e:
            st.write(f"🔍 Erro na análise detalhada: {str(e)}")
        
        return None

class MetricsCalculator:
    """Calculadora avançada de métricas"""
    
    @staticmethod
    def calculate_growth_rate(series: pd.Series, periods: int = 3) -> Optional[float]:
        """Calcula taxa de crescimento com análise de tendência"""
        if len(series) < periods * 2:
            return None
        
        series = series.dropna().sort_index()
        if len(series) < periods * 2:
            return None
        
        # Divide em dois períodos
        mid_point = len(series) // 2
        first_half = series.iloc[:mid_point].mean()
        second_half = series.iloc[mid_point:].mean()
        
        if first_half > 0:
            return ((second_half - first_half) / first_half) * 100
        return None
    
    @staticmethod
    def calculate_volatility(series: pd.Series) -> Optional[float]:
        """Calcula volatilidade da série"""
        if len(series) < 3:
            return None
        
        pct_change = series.pct_change().dropna()
        return pct_change.std() * 100 if len(pct_change) > 0 else None
    
    @staticmethod
    def calculate_efficiency_score(atendimentos: float, laudos: float, taxa_conversao: float) -> float:
        """Calcula score de eficiência ponderado"""
        if atendimentos == 0:
            return 0
        
        # Ponderação: 50% taxa conversão, 30% volume laudos, 20% volume atendimentos
        volume_score = min(laudos / 100, 1) * 30  # Normalizado para max 100 laudos
        conversion_score = min(taxa_conversao / 100, 1) * 50
        activity_score = min(atendimentos / 200, 1) * 20  # Normalizado para max 200 atendimentos
        
        return volume_score + conversion_score + activity_score

def format_number(value: Union[float, int], decimal_places: int = 0, 
                 suffix: str = "") -> str:
    """Formatação avançada de números"""
    if pd.isna(value) or value is None:
        return "—"
    
    try:
        if abs(value) >= 1_000_000:
            formatted = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value/1_000:.1f}K"
        else:
            if decimal_places == 0:
                formatted = f"{int(round(value)):,}".replace(",", ".")
            else:
                formatted = f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        return f"{formatted}{suffix}"
    except (ValueError, TypeError):
        return "—"

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      icon: str = "📊", delta_type: str = "neutral") -> str:
    """Cria card de métrica moderno"""
    delta_class = f"metric-delta {delta_type}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ============ HEADER PRINCIPAL ============
def render_main_header():
    """Renderiza header principal"""
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>🏥 Dashboard Executivo {config.COMPANY}</h1>
                <p>Sistema Avançado de Monitoramento e Análise Operacional</p>
            </div>
            <div style="text-align: right;">
                <div class="status-badge status-excellent">
                    <span>🚀</span> v{config.VERSION}
                </div>
                <div style="margin-top: 8px; opacity: 0.8; font-size: 0.9rem;">
                    Atualizado: {current_time}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_main_header()

# ============ CONFIGURAÇÃO DE DADOS ============
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

@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padronização inteligente de DataFrames com debug"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    st.write(f"🔧 Padronizando dataset: {name}")
    
    result = df.copy()
    mapping = ENHANCED_COLUMN_MAPPINGS.get(name, {})
    
    # Debug: Mostrar colunas originais
    st.write(f"📋 Colunas originais: {list(result.columns)}")
    
    # Normalização de colunas (mais tolerante)
    original_columns = result.columns.tolist()
    result.columns = [str(col).lower().strip().replace(' ', '_').replace('-', '_') for col in result.columns]
    
    # Mapeamento de colunas comuns (mais flexível)
    column_aliases = {
        'data_interesse': ['data', 'data_atendimento', 'dt_interesse', 'data_solicitacao'],
        'quantidade': ['qtd', 'qtde', 'total', 'count', 'numero'],
        'idatendimento': ['id_atendimento', 'atendimento_id', 'cod_atendimento'],
        'iddocumento': ['id_documento', 'documento_id', 'cod_documento', 'id_laudo'],
        'unidade': ['unidade_origem', 'local', 'origem'],
        'diretoria': ['dir', 'diret'],
        'superintendencia': ['super', 'superintend'],
        'tipo': ['tipopericia', 'tipo_pericia', 'competencia', 'txcompetencia']
    }
    
    # Aplicar aliases
    for target_col, aliases in column_aliases.items():
        if target_col not in result.columns:
            for alias in aliases:
                if alias in result.columns:
                    result = result.rename(columns={alias: target_col})
                    st.write(f"🔄 Renomeado: {alias} → {target_col}")
                    break
    
    # Processamento de quantidade (mais flexível)
    quantity_col = mapping.get("quantity_column", "quantidade")
    if quantity_col in result.columns:
        result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(1)
    elif "quantidade" not in result.columns:
        # Se não tem coluna quantidade, criar baseada no número de linhas
        result["quantidade"] = 1
        st.write("ℹ️ Coluna 'quantidade' criada com valor padrão 1")
    
    # Processamento de dimensões
    dimensions = mapping.get("dimensions", {})
    for target_col, source_col in dimensions.items():
        if source_col in result.columns:
            result[target_col] = (result[source_col]
                                 .astype(str)
                                 .str.strip()
                                 .str.title()
                                 .replace({"Nan": None, "": None, "None": None}))
    
    # Processamento de datas (mais robusto)
    date_columns = mapping.get("date_columns", [])
    date_processed = False
    
    # Tentar colunas de data do mapeamento
    for date_col in date_columns:
        if date_col in result.columns:
            processed_date = pd.to_datetime(result[date_col], 
                                          errors="coerce", 
                                          dayfirst=True,
                                          infer_datetime_format=True)
            if processed_date.notna().any():
                result["data_base"] = processed_date
                date_processed = True
                st.write(f"📅 Data processada da coluna: {date_col}")
                break
    
    # Se não encontrou, tentar colunas que parecem data
    if not date_processed:
        potential_date_cols = [col for col in result.columns if 'data' in col.lower()]
        for date_col in potential_date_cols:
            try:
                processed_date = pd.to_datetime(result[date_col], 
                                              errors="coerce", 
                                              dayfirst=True,
                                              infer_datetime_format=True)
                if processed_date.notna().sum() > len(result) * 0.5:  # Se pelo menos 50% são datas válidas
                    result["data_base"] = processed_date
                    date_processed = True
                    st.write(f"📅 Data inferida da coluna: {date_col}")
                    break
            except:
                continue
    
    # Criar campos derivados de data se foi processada
    if date_processed and "data_base" in result.columns:
        result["anomês_dt"] = result["data_base"].dt.to_period("M").dt.to_timestamp()
        result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
        result["ano"] = result["anomês_dt"].dt.year
        result["mes"] = result["anomês_dt"].dt.month
        result["dia"] = result["data_base"].dt.normalize()
        result["dia_semana"] = result["data_base"].dt.day_name()
        st.write("📊 Campos temporais derivados criados")
    else:
        st.warning("⚠️ Nenhuma coluna de data válida encontrada")
    
    # ID único
    id_col = mapping.get("id_column")
    if id_col and id_col in result.columns:
        result["id"] = result[id_col].astype(str)
    elif not any("id" in col.lower() for col in result.columns):
        # Criar ID sequencial se não existir
        result["id"] = range(len(result))
        st.write("🆔 ID sequencial criado")
    
    st.write(f"✅ Padronização concluída: {len(result)} registros, {len(result.columns)} colunas")
    return result

# ============ SIDEBAR AVANÇADA ============
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                color: white; padding: 20px; border-radius: 16px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: white;">🎛️ Controle Central</h3>
        <p style="margin: 8px 0 0 0; opacity: 0.9;">Configurações e filtros avançados</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload de arquivos com interface melhorada
    st.subheader("📁 Gestão de Dados")
    
    with st.expander("📤 Upload de Arquivos", expanded=True):
        uploaded_files = st.file_uploader(
            "Selecione os arquivos CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="Arraste e solte arquivos CSV ou clique para selecionar"
        )
    
    # Configurações de exibição
    st.subheader("⚙️ Configurações")
    
    chart_height = st.slider(
        "📏 Altura dos Gráficos",
        min_value=300,
        max_value=700,
        value=config.DEFAULT_CHART_HEIGHT,
        step=50,
        help="Ajuste a altura dos gráficos para melhor visualização"
    )
    
    show_benchmarks = st.toggle(
        "📊 Exibir Metas e Benchmarks",
        value=True,
        help="Mostra linhas de referência nos gráficos"
    )
    
    auto_refresh = st.toggle(
        "🔄 Atualização Automática",
        value=False,
        help="Atualiza dados automaticamente a cada 5 minutos"
    )

# ============ PROCESSAMENTO DE DADOS ============
@st.cache_data(ttl=config.CACHE_TTL, show_spinner="Processando dados...")
def load_and_process_data(files: List) -> Dict[str, pd.DataFrame]:
    """Carrega e processa todos os dados com debug melhorado"""
    dataframes = {}
    
    if not files:
        st.info("📁 Nenhum arquivo enviado via upload")
        # Tentar carregar da pasta data/ se existir
        if os.path.exists("data"):
            st.info("🔍 Verificando pasta data/ local...")
            csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
            if csv_files:
                st.info(f"📂 Encontrados {len(csv_files)} arquivos CSV na pasta data/")
                for filename in csv_files:
                    filepath = os.path.join("data", filename)
                    try:
                        with open(filepath, 'rb') as f:
                            content = f.read()
                        
                        df = DataProcessor.smart_csv_reader(content, filename)
                        if df is not None:
                            dataset_name = detect_dataset_type(filename)
                            standardized_df = standardize_dataframe(dataset_name, df)
                            if not standardized_df.empty:
                                dataframes[dataset_name] = standardized_df
                                st.success(f"✅ Carregado: {filename} → {dataset_name}")
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar {filename}: {str(e)}")
            else:
                st.warning("📂 Pasta data/ existe mas não contém arquivos CSV")
        return dataframes
    
    # Processar uploads
    st.info(f"📤 Processando {len(files)} arquivo(s) enviado(s)...")
    
    for i, uploaded_file in enumerate(files):
        if uploaded_file is not None:
            try:
                st.write(f"📁 Processando arquivo {i+1}/{len(files)}: {uploaded_file.name}")
                
                # Ler conteúdo
                content = uploaded_file.read()
                
                # Processar CSV
                df = DataProcessor.smart_csv_reader(content, uploaded_file.name)
                
                if df is not None:
                    # Detectar tipo e padronizar
                    dataset_name = detect_dataset_type(uploaded_file.name)
                    st.write(f"🔍 Detectado como: {dataset_name}")
                    
                    standardized_df = standardize_dataframe(dataset_name, df)
                    
                    if not standardized_df.empty:
                        dataframes[dataset_name] = standardized_df
                        st.success(f"✅ Processado com sucesso: {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ Arquivo processado mas ficou vazio após padronização: {uploaded_file.name}")
                else:
                    st.error(f"❌ Falha no processamento: {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"❌ Erro inesperado ao processar {uploaded_file.name}: {str(e)}")
                # Mostrar traceback para debug
                import traceback
                st.code(traceback.format_exc())
    
    st.info(f"📊 Total de datasets válidos carregados: {len(dataframes)}")
    return dataframes

def detect_dataset_type(filename: str) -> str:
    """Detecta o tipo de dataset pelo nome do arquivo com melhor tolerância"""
    filename_clean = filename.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    
    # Remover números entre parênteses que podem estar no nome
    import re
    filename_clean = re.sub(r'\(\d+\)', '', filename_clean)
    
    patterns = {
        'atendimentos_diario': 'Atendimentos_diario',
        'atendimentos_todos': 'Atendimentos_todos_Mensal', 
        'atendimentos_especifico': 'Atendimentos_especifico_Mensal',
        'laudos_diario': 'Laudos_diario',
        'laudos_todos': 'Laudos_todos_Mensal',
        'laudos_especifico': 'Laudos_especifico_Mensal',
        'laudos_realizados': 'Laudos_todos_Mensal',  # Alias
        'laudospendentes': 'detalhes_laudospendentes',
        'examespendentes': 'detalhes_examespendentes',
        'detalhes_laudospendentes': 'detalhes_laudospendentes',
        'detalhes_examespendentes': 'detalhes_examespendentes'
    }
    
    # Buscar correspondência mais flexível
    for pattern, dataset_type in patterns.items():
        if pattern in filename_clean:
            return dataset_type
    
    # Fallback baseado em palavras-chave
    if 'atendimento' in filename_clean:
        if 'diario' in filename_clean:
            return 'Atendimentos_diario'
        elif 'especifico' in filename_clean:
            return 'Atendimentos_especifico_Mensal'
        else:
            return 'Atendimentos_todos_Mensal'
    
    if 'laudo' in filename_clean:
        if 'diario' in filename_clean:
            return 'Laudos_diario'
        elif 'especifico' in filename_clean:
            return 'Laudos_especifico_Mensal'
        elif 'pendente' in filename_clean:
            return 'detalhes_laudospendentes'
        else:
            return 'Laudos_todos_Mensal'
    
    if 'exame' in filename_clean and 'pendente' in filename_clean:
        return 'detalhes_examespendentes'
    
    # Usar nome original se não encontrou padrão
    return filename_clean

# Carregar dados
with st.spinner("🔄 Carregando e processando dados..."):
    dataframes = load_and_process_data(uploaded_files if uploaded_files else [])

# Debug: Mostrar informações dos dataframes carregados
if dataframes:
    st.sidebar.success(f"✅ {len(dataframes)} datasets processados")
    with st.sidebar.expander("📊 Detalhes dos Datasets", expanded=False):
        for name, df in dataframes.items():
            if df is not None and not df.empty:
                st.write(f"**{name}:**")
                st.write(f"- Registros: {len(df):,}")
                st.write(f"- Colunas: {len(df.columns)}")
                st.write(f"- Colunas principais: {list(df.columns[:3])}")
                st.write("---")

# Validação e feedback melhorado
if not dataframes:
    st.warning("⚠️ Nenhum arquivo de dados foi carregado com sucesso")
    st.info("""
    📝 **Para começar:**
    1. Faça upload dos arquivos CSV usando a sidebar  
    2. Verifique se os arquivos estão no formato correto
    3. Os arquivos devem ter pelo menos 2 colunas e dados válidos
    
    **Formatos suportados:** CSV com separadores `;`, `,`, `|` ou tab
    **Encoding:** UTF-8, Latin-1, CP1252, ISO-8859-1
    """)
    
    # Mostrar exemplo de estrutura esperada
    with st.expander("📋 Estrutura esperada dos arquivos"):
        st.markdown("""
        **Atendimentos:**
        - Deve conter: data_interesse, idatendimento, quantidade
        
        **Laudos:**  
        - Deve conter: data_interesse, iddocumento, quantidade
        
        **Pendências:**
        - Deve conter: data_solicitacao, caso_sirsaelp, unidade
        """)
    
    st.stop()

# Só continua se tiver dados válidos
valid_dataframes = {k: v for k, v in dataframes.items() if v is not None and not v.empty}

if not valid_dataframes:
    st.error("❌ Nenhum dataset válido foi carregado")
    st.info("Verifique se os arquivos contêm dados válidos e estrutura correta")
    st.stop()

# Atualizar dataframes para usar apenas os válidos
dataframes = valid_dataframes

# ============ FILTROS INTELIGENTES ============
class FilterEngine:
    """Sistema avançado de filtros"""
    
    @staticmethod
    def extract_unique_values(dataframes: Dict[str, pd.DataFrame], column: str) -> List[str]:
        """Extrai valores únicos de uma coluna em todos os dataframes"""
        values = set()
        for df in dataframes.values():
            if df is not None and column in df.columns:
                unique_vals = df[column].dropna().astype(str).unique()
                values.update(v for v in unique_vals if v and v.lower() not in ["nan", "none", ""])
        return sorted(list(values))
    
    @staticmethod
    def get_date_range(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Obtém o range de datas disponível"""
        min_date, max_date = None, None
        
        for df in dataframes.values():
            if df is not None and 'data_base' in df.columns:
                df_min = df['data_base'].min()
                df_max = df['data_base'].max()
                
                if pd.notna(df_min):
                    min_date = df_min if min_date is None else min(min_date, df_min)
                if pd.notna(df_max):
                    max_date = df_max if max_date is None else max(max_date, df_max)
        
        return min_date, max_date

filter_engine = FilterEngine()

# Configuração de filtros na sidebar
with st.sidebar:
    st.markdown("### 🔍 Filtros Avançados")
    
    # Filtros dimensionais com interface melhorada
    col1, col2 = st.columns(2)
    
    with col1:
        diretorias = st.multiselect(
            "🏢 Diretoria",
            filter_engine.extract_unique_values(dataframes, "diretoria"),
            help="Filtrar por diretoria específica"
        )
        
        unidades = st.multiselect(
            "🏪 Unidade",
            filter_engine.extract_unique_values(dataframes, "unidade"),
            help="Filtrar por unidade operacional"
        )
    
    with col2:
        superintendencias = st.multiselect(
            "🏛️ Superintendência",
            filter_engine.extract_unique_values(dataframes, "superintendencia"),
            help="Filtrar por superintendência"
        )
        
        tipos = st.multiselect(
            "🔬 Tipo",
            filter_engine.extract_unique_values(dataframes, "tipo"),
            help="Filtrar por tipo de perícia"
        )
    
    # Filtro temporal avançado
    st.markdown("#### 📅 Período de Análise")
    
    min_date, max_date = filter_engine.get_date_range(dataframes)
    
    if min_date and max_date:
        period_type = st.radio(
            "Tipo de período:",
            ["Predefinido", "Personalizado"],
            horizontal=True
        )
        
        if period_type == "Predefinido":
            period_options = {
                "Todo o período": None,
                "Último ano": 365,
                "Últimos 6 meses": 180,
                "Últimos 3 meses": 90,
                "Último mês": 30
            }
            
            selected_period = st.selectbox(
                "Período:",
                list(period_options.keys())
            )
            
            if period_options[selected_period]:
                start_date = max_date - timedelta(days=period_options[selected_period])
                end_date = max_date
            else:
                start_date, end_date = min_date, max_date
        else:
            date_range = st.date_input(
                "Selecione o período:",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = pd.Timestamp(date_range[0])
                end_date = pd.Timestamp(date_range[1])
            else:
                start_date, end_date = min_date, max_date
    else:
        start_date = end_date = None

# Aplicar filtros
@st.cache_data(ttl=config.CACHE_TTL)
def apply_filters(dataframes: Dict[str, pd.DataFrame], filters: Dict) -> Dict[str, pd.DataFrame]:
    """Aplica filtros avançados aos dataframes"""
    filtered_dfs = {}
    
    for name, df in dataframes.items():
        if df is None or df.empty:
            filtered_dfs[name] = df
            continue
        
        filtered = df.copy()
        
        # Filtros dimensionais
        for filter_name, values in filters.get('dimensions', {}).items():
            if values and filter_name in filtered.columns:
                filtered = filtered[filtered[filter_name].isin(values)]
        
        # Filtro temporal
        if 'data_base' in filtered.columns and filters.get('start_date') and filters.get('end_date'):
            filtered = filtered[
                (filtered['data_base'] >= filters['start_date']) &
                (filtered['data_base'] <= filters['end_date'])
            ]
        
        filtered_dfs[name] = filtered
    
    return filtered_dfs

# Consolidar filtros
filters = {
    'dimensions': {
        'diretoria': diretorias,
        'superintendencia': superintendencias,
        'unidade': unidades,
        'tipo': tipos
    },
    'start_date': start_date,
    'end_date': end_date
}

# Aplicar filtros
filtered_dataframes = apply_filters(dataframes, filters)

# ============ CÁLCULO DE MÉTRICAS PRINCIPAIS ============
class KPIEngine:
    """Motor de cálculo de KPIs avançados"""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dfs = dataframes
        self.calc = MetricsCalculator()
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de produção"""
        df_atend = self.dfs.get("Atendimentos_todos_Mensal")
        df_laudos = self.dfs.get("Laudos_todos_Mensal")
        
        metrics = {}
        
        if df_atend is not None and not df_atend.empty:
            metrics['total_atendimentos'] = df_atend['quantidade'].sum()
            metrics['media_mensal_atendimentos'] = df_atend.groupby('anomês_dt')['quantidade'].sum().mean()
            
            # Tendência
            monthly_atend = df_atend.groupby('anomês_dt')['quantidade'].sum().sort_index()
            metrics['crescimento_atendimentos'] = self.calc.calculate_growth_rate(monthly_atend)
            metrics['volatilidade_atendimentos'] = self.calc.calculate_volatility(monthly_atend)
        
        if df_laudos is not None and not df_laudos.empty:
            metrics['total_laudos'] = df_laudos['quantidade'].sum()
            metrics['media_mensal_laudos'] = df_laudos.groupby('anomês_dt')['quantidade'].sum().mean()
            
            # Tendência
            monthly_laudos = df_laudos.groupby('anomês_dt')['quantidade'].sum().sort_index()
            metrics['crescimento_laudos'] = self.calc.calculate_growth_rate(monthly_laudos)
            metrics['volatilidade_laudos'] = self.calc.calculate_volatility(monthly_laudos)
        
        # Taxa de conversão
        if metrics.get('total_atendimentos', 0) > 0:
            metrics['taxa_conversao'] = (metrics.get('total_laudos', 0) / metrics['total_atendimentos']) * 100
        
        return metrics
    
    def get_pendency_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de pendências"""
        df_pend_laudos = self.dfs.get("detalhes_laudospendentes")
        df_pend_exames = self.dfs.get("detalhes_examespendentes")
        
        metrics = {}
        
        def calculate_aging_stats(df: pd.DataFrame) -> Dict:
            if df is None or df.empty or 'data_base' not in df.columns:
                return {}
            
            hoje = pd.Timestamp.now().normalize()
            aging_days = (hoje - df['data_base']).dt.days
            
            return {
                'total': len(df),
                'media_dias': aging_days.mean(),
                'mediana_dias': aging_days.median(),
                'max_dias': aging_days.max(),
                'p90_dias': aging_days.quantile(0.9),
                'criticos': (aging_days > config.BENCHMARKS['aging_critico']).sum(),
                'urgentes': (aging_days > config.BENCHMARKS['aging_atencao']).sum()
            }
        
        metrics['laudos_pendentes'] = calculate_aging_stats(df_pend_laudos)
        metrics['exames_pendentes'] = calculate_aging_stats(df_pend_exames)
        
        # Backlog estimado
        media_laudos = self.get_production_metrics().get('media_mensal_laudos', 0)
        total_pend_laudos = metrics['laudos_pendentes'].get('total', 0)
        
        if media_laudos > 0:
            metrics['backlog_meses'] = total_pend_laudos / media_laudos
        
        return metrics
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de eficiência"""
        production = self.get_production_metrics()
        pendency = self.get_pendency_metrics()
        
        metrics = {}
        
        # Score de eficiência global
        atend = production.get('total_atendimentos', 0)
        laudos = production.get('total_laudos', 0)
        taxa_conv = production.get('taxa_conversao', 0)
        
        metrics['efficiency_score'] = self.calc.calculate_efficiency_score(atend, laudos, taxa_conv)
        
        # Status baseado em benchmarks
        if taxa_conv >= config.BENCHMARKS['taxa_conversao_excelente']:
            metrics['conversion_status'] = 'excellent'
        elif taxa_conv >= config.BENCHMARKS['taxa_conversao_boa']:
            metrics['conversion_status'] = 'good'
        elif taxa_conv >= config.BENCHMARKS['taxa_conversao_minima']:
            metrics['conversion_status'] = 'fair'
        else:
            metrics['conversion_status'] = 'poor'
        
        # Status do backlog
        backlog = pendency.get('backlog_meses', 0)
        if backlog <= config.BENCHMARKS['backlog_atencao']:
            metrics['backlog_status'] = 'excellent'
        elif backlog <= config.BENCHMARKS['backlog_critico']:
            metrics['backlog_status'] = 'good'
        else:
            metrics['backlog_status'] = 'poor'
        
        return metrics

# Calcular métricas
kpi_engine = KPIEngine(filtered_dataframes)
production_metrics = kpi_engine.get_production_metrics()
pendency_metrics = kpi_engine.get_pendency_metrics()
efficiency_metrics = kpi_engine.get_efficiency_metrics()

# ============ DASHBOARD PRINCIPAL ============
st.markdown('<h2 class="section-header">📊 Indicadores Principais de Performance</h2>', unsafe_allow_html=True)

# Linha 1: Métricas de Produção
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_atend = production_metrics.get('total_atendimentos', 0)
    cresc_atend = production_metrics.get('crescimento_atendimentos')
    
    delta_text = None
    delta_type = "neutral"
    if cresc_atend is not None:
        delta_text = f"↗️ {format_number(cresc_atend, 1)}%" if cresc_atend > 0 else f"↘️ {format_number(abs(cresc_atend), 1)}%"
        delta_type = "positive" if cresc_atend > 0 else "negative"
    
    card_html = create_metric_card(
        "Atendimentos Totais",
        format_number(total_atend),
        delta_text,
        "👥",
        delta_type
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col2:
    total_laudos = production_metrics.get('total_laudos', 0)
    cresc_laudos = production_metrics.get('crescimento_laudos')
    
    delta_text = None
    delta_type = "neutral"
    if cresc_laudos is not None:
        delta_text = f"↗️ {format_number(cresc_laudos, 1)}%" if cresc_laudos > 0 else f"↘️ {format_number(abs(cresc_laudos), 1)}%"
        delta_type = "positive" if cresc_laudos > 0 else "negative"
    
    card_html = create_metric_card(
        "Laudos Emitidos",
        format_number(total_laudos),
        delta_text,
        "📋",
        delta_type
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col3:
    taxa_conv = production_metrics.get('taxa_conversao', 0)
    conv_status = efficiency_metrics.get('conversion_status', 'poor')
    
    status_icons = {
        'excellent': '🟢',
        'good': '🟡',
        'fair': '🟠',
        'poor': '🔴'
    }
    
    card_html = create_metric_card(
        "Taxa de Conversão",
        f"{status_icons[conv_status]} {format_number(taxa_conv, 1)}%",
        f"Meta: {config.BENCHMARKS['taxa_conversao_boa']}%",
        "🎯",
        "positive" if conv_status in ['excellent', 'good'] else "negative"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col4:
    media_laudos = production_metrics.get('media_mensal_laudos', 0)
    efficiency_score = efficiency_metrics.get('efficiency_score', 0)
    
    card_html = create_metric_card(
        "Produtividade Mensal",
        f"{format_number(media_laudos)} laudos",
        f"Score: {format_number(efficiency_score, 1)}/100",
        "⚡",
        "positive" if efficiency_score > 70 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

# Linha 2: Métricas de Pendências
st.markdown('<h2 class="section-header">⏰ Gestão de Pendências e Backlog</h2>', unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    total_pend_laudos = pendency_metrics['laudos_pendentes'].get('total', 0)
    criticos_laudos = pendency_metrics['laudos_pendentes'].get('criticos', 0)
    
    pct_criticos = (criticos_laudos / total_pend_laudos * 100) if total_pend_laudos > 0 else 0
    status_icon = "🔴" if pct_criticos > 20 else "🟡" if pct_criticos > 10 else "🟢"
    
    card_html = create_metric_card(
        "Laudos Pendentes",
        f"{status_icon} {format_number(total_pend_laudos)}",
        f"Críticos: {format_number(criticos_laudos)} ({format_number(pct_criticos, 1)}%)",
        "📋",
        "negative" if pct_criticos > 20 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col6:
    total_pend_exames = pendency_metrics['exames_pendentes'].get('total', 0)
    criticos_exames = pendency_metrics['exames_pendentes'].get('criticos', 0)
    
    pct_criticos_ex = (criticos_exames / total_pend_exames * 100) if total_pend_exames > 0 else 0
    status_icon = "🔴" if pct_criticos_ex > 20 else "🟡" if pct_criticos_ex > 10 else "🟢"
    
    card_html = create_metric_card(
        "Exames Pendentes",
        f"{status_icon} {format_number(total_pend_exames)}",
        f"Críticos: {format_number(criticos_exames)} ({format_number(pct_criticos_ex, 1)}%)",
        "🔬",
        "negative" if pct_criticos_ex > 20 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col7:
    backlog_meses = pendency_metrics.get('backlog_meses', 0)
    backlog_status = efficiency_metrics.get('backlog_status', 'poor')
    
    status_icons = {
        'excellent': '🟢',
        'good': '🟡',
        'poor': '🔴'
    }
    
    card_html = create_metric_card(
        "Backlog Estimado",
        f"{status_icons[backlog_status]} {format_number(backlog_meses, 1)} meses",
        f"Meta: < {config.BENCHMARKS['backlog_atencao']} meses",
        "📈",
        "negative" if backlog_status == 'poor' else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col8:
    media_aging_laudos = pendency_metrics['laudos_pendentes'].get('media_dias', 0)
    media_aging_exames = pendency_metrics['exames_pendentes'].get('media_dias', 0)
    aging_medio = max(media_aging_laudos, media_aging_exames)
    
    status_icon = ("🔴" if aging_medio > config.BENCHMARKS['aging_critico'] 
                  else "🟡" if aging_medio > config.BENCHMARKS['aging_atencao'] 
                  else "🟢")
    
    card_html = create_metric_card(
        "Aging Médio",
        f"{status_icon} {format_number(aging_medio)} dias",
        f"P90: {format_number(max(pendency_metrics['laudos_pendentes'].get('p90_dias', 0), pendency_metrics['exames_pendentes'].get('p90_dias', 0)))} dias",
        "⏱️",
        "negative" if aging_medio > config.BENCHMARKS['aging_critico'] else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

# ============ SISTEMA DE ALERTAS INTELIGENTES ============
class AlertSystem:
    """Sistema inteligente de alertas"""
    
    @staticmethod
    def generate_alerts(production: Dict, pendency: Dict, efficiency: Dict) -> List[Dict]:
        """Gera alertas baseados em regras de negócio"""
        alerts = []
        
        # Alertas críticos
        backlog = pendency.get('backlog_meses', 0)
        if backlog > config.BENCHMARKS['backlog_critico']:
            alerts.append({
                'type': 'danger',
                'title': 'BACKLOG CRÍTICO',
                'message': f'Backlog de {format_number(backlog, 1)} meses excede limite crítico ({config.BENCHMARKS["backlog_critico"]} meses)',
                'priority': 1
            })
        
        taxa_conv = production.get('taxa_conversao', 0)
        if taxa_conv < config.BENCHMARKS['taxa_conversao_minima']:
            alerts.append({
                'type': 'danger',
                'title': 'EFICIÊNCIA CRÍTICA',
                'message': f'Taxa de conversão de {format_number(taxa_conv, 1)}% abaixo do mínimo aceitável ({config.BENCHMARKS["taxa_conversao_minima"]}%)',
                'priority': 1
            })
        
        # Alertas de atenção
        cresc_laudos = production.get('crescimento_laudos', 0)
        if cresc_laudos and cresc_laudos < -15:
            alerts.append({
                'type': 'warning',
                'title': 'QUEDA NA PRODUÇÃO',
                'message': f'Redução significativa de {format_number(abs(cresc_laudos), 1)}% na emissão de laudos',
                'priority': 2
            })
        
        # Alertas informativos
        if taxa_conv >= config.BENCHMARKS['taxa_conversao_excelente']:
            alerts.append({
                'type': 'info',
                'title': 'PERFORMANCE EXCELENTE',
                'message': f'Taxa de conversão de {format_number(taxa_conv, 1)}% acima da meta de excelência',
                'priority': 3
            })
        
        return sorted(alerts, key=lambda x: x['priority'])

alert_system = AlertSystem()
alerts = alert_system.generate_alerts(production_metrics, pendency_metrics, efficiency_metrics)

# Exibir alertas
if alerts:
    st.markdown('<h2 class="section-header">🚨 Central de Alertas e Insights</h2>', unsafe_allow_html=True)
    
    # Organizar alertas por tipo
    critical_alerts = [a for a in alerts if a['type'] == 'danger']
    warning_alerts = [a for a in alerts if a['type'] == 'warning']
    info_alerts = [a for a in alerts if a['type'] == 'info']
    
    # Exibir alertas críticos primeiro
    for alert in critical_alerts[:3]:  # Máximo 3 críticos
        st.markdown(f"""
        <div class="alert alert-{alert['type']}">
            <strong>🔴 {alert['title']}</strong><br>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)
    
    # Alertas de atenção em colunas
    if warning_alerts:
        cols = st.columns(min(len(warning_alerts), 2))
        for i, alert in enumerate(warning_alerts[:2]):
            with cols[i]:
                st.markdown(f"""
                <div class="alert alert-{alert['type']}">
                    <strong>🟡 {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
    
    # Alertas informativos
    if info_alerts and not critical_alerts:  # Só mostrar se não há críticos
        st.markdown(f"""
        <div class="alert alert-{info_alerts[0]['type']}">
            <strong>ℹ️ {info_alerts[0]['title']}</strong><br>
            {info_alerts[0]['message']}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert alert-success">
        <strong>✅ SITUAÇÃO OPERACIONAL NORMAL</strong><br>
        Todos os indicadores estão dentro dos parâmetros esperados. Sistema operando com eficiência.
    </div>
    """, unsafe_allow_html=True)

# ============ SISTEMA DE ABAS SIMPLES ============
tab1, tab2, tab3 = st.tabs([
    "📊 **Visão Geral**",
    "📈 **Análise Detalhada**", 
    "📑 **Relatórios**"
])

# ============ ABA 1: VISÃO GERAL ============
with tab1:
    st.markdown('<h3 class="section-header">📊 Panorama Executivo</h3>', unsafe_allow_html=True)
    
    df_atend = filtered_dataframes.get("Atendimentos_todos_Mensal")
    df_laudos = filtered_dataframes.get("Laudos_todos_Mensal")
    
    if df_atend is not None and df_laudos is not None and not df_atend.empty and not df_laudos.empty:
        
        # Preparação dos dados
        atend_monthly = df_atend.groupby("anomês_dt")["quantidade"].sum().reset_index()
        laudos_monthly = df_laudos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        
        # Gráfico de evolução principal
        col_chart1, col_chart2 = st.columns([0.7, 0.3])
        
        with col_chart1:
            st.markdown("#### 📈 Evolução Temporal: Atendimentos vs Laudos")
            
            fig_evolution = go.Figure()
            
            fig_evolution.add_trace(go.Scatter(
                x=atend_monthly["anomês_dt"],
                y=atend_monthly["quantidade"], 
                mode='lines+markers',
                name='Atendimentos',
                line=dict(color=config.COLORS['secondary'], width=3),
                marker=dict(size=6)
            ))
            
            fig_evolution.add_trace(go.Scatter(
                x=laudos_monthly["anomês_dt"],
                y=laudos_monthly["quantidade"],
                mode='lines+markers', 
                name='Laudos',
                line=dict(color=config.COLORS['success'], width=3),
                marker=dict(size=6)
            ))
            
            fig_evolution.update_layout(
                height=chart_height,
                hovermode='x unified',
                xaxis_title="Período",
                yaxis_title="Quantidade",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(0,0,0,0)'
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
                    line=dict(color=config.COLORS['warning'], width=3),
                    marker=dict(size=8),
                    name='Taxa de Conversão'
                ))
                
                if show_benchmarks:
                    fig_conversion.add_hline(
                        y=config.BENCHMARKS['taxa_conversao_boa'],
                        line_dash="dash",
                        line_color=config.COLORS['success'],
                        annotation_text=f"Meta: {config.BENCHMARKS['taxa_conversao_boa']}%"
                    )
                
                fig_conversion.update_layout(
                    height=chart_height,
                    xaxis_title="Período",
                    yaxis_title="Taxa (%)",
                    yaxis=dict(range=[0, 100]),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_conversion, use_container_width=True)

# ============ ABA 2: ANÁLISE DETALHADA ============
with tab2:
    st.markdown('<h3 class="section-header">📈 Análise Detalhada</h3>', unsafe_allow_html=True)
    
    # Rankings por unidade
    st.markdown("#### 🏢 Performance por Unidade")
    
    if df_laudos is not None and "unidade" in df_laudos.columns:
        unidade_summary = (
            df_laudos.groupby("unidade")["quantidade"]
            .sum()
            .sort_values(ascending=True)
            .tail(15)
            .reset_index()
        )
        
        fig_unidades = px.bar(
            unidade_summary,
            x="quantidade",
            y="unidade",
            orientation="h",
            color="quantidade",
            color_continuous_scale="Blues",
            title="Top 15 Unidades - Laudos Emitidos"
        )
        
        fig_unidades.update_layout(
            height=500,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_unidades, use_container_width=True)

# ============ ABA 3: RELATÓRIOS ============
with tab3:
    st.markdown('<h3 class="section-header">📑 Centro de Relatórios</h3>', unsafe_allow_html=True)
    
    def generate_simple_report() -> str:
        """Gera relatório executivo simplificado"""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        report = f"""# 📊 RELATÓRIO EXECUTIVO - {config.COMPANY}

**Data:** {timestamp}  
**Versão:** {config.VERSION}

## 📈 INDICADORES PRINCIPAIS

- **Atendimentos Totais:** {format_number(production_metrics.get('total_atendimentos', 0))}
- **Laudos Emitidos:** {format_number(production_metrics.get('total_laudos', 0))}
- **Taxa de Conversão:** {format_number(production_metrics.get('taxa_conversao', 0), 1)}%
- **Backlog:** {format_number(pendency_metrics.get('backlog_meses', 0), 1)} meses

## ⏰ PENDÊNCIAS

- **Laudos Pendentes:** {format_number(pendency_metrics['laudos_pendentes'].get('total', 0))}
- **Exames Pendentes:** {format_number(pendency_metrics['exames_pendentes'].get('total', 0))}
- **Casos Críticos:** {format_number(pendency_metrics['laudos_pendentes'].get('criticos', 0) + pendency_metrics['exames_pendentes'].get('criticos', 0))}

## 🎯 STATUS GERAL

{efficiency_metrics.get('conversion_status', 'poor').replace('excellent', '🟢 Excelente').replace('good', '🟡 Boa').replace('poor', '🔴 Necessita Atenção')}

---
*Relatório gerado automaticamente*
"""
        return report
    
    col_report1, col_report2 = st.columns([0.7, 0.3])
    
    with col_report1:
        if st.button("📊 Gerar Relatório", type="primary"):
            report_content = generate_simple_report()
            st.markdown("### 📄 Relatório Gerado")
            st.markdown(report_content)
            
            # Download
            st.download_button(
                label="📥 Download Relatório",
                data=report_content.encode('utf-8'),
                file_name=f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col_report2:
        st.markdown("#### 📊 Estatísticas")
        total_registros = sum(len(df) for df in dataframes.values() if df is not None and not df.empty)
        st.metric("Total Registros", f"{total_registros:,}".replace(",", "."))
        st.metric("Datasets", len(dataframes))

# ============ SIDEBAR RESUMO ============
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Resumo da Sessão")
    
    # Status dos datasets
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            filtered_df = filtered_dataframes.get(name, df)
            status_icon = "🟢" if not filtered_df.empty else "🟡"
            count_text = f"{len(filtered_df):,}".replace(",", ".")
            st.write(f"{status_icon} {count_text}")
    
    # Status geral
    critical_alerts = len([a for a in alerts if a.get('type') == 'danger'])
    if critical_alerts > 0:
        st.markdown(f"**🚨 Status:** {critical_alerts} alertas críticos")
    else:
        st.markdown("**✅ Status:** Normal")
    
    st.markdown("---")
    st.markdown("**🛠️ Suporte**")
    st.markdown("📧 equipe-ti@pci.sc.gov.br")

# Auto-refresh
if auto_refresh:
    time.sleep(300)  # 5 minutos
    st.rerun()

# ============ RODAPÉ FINAL ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 40px; background: linear-gradient(135deg, {config.COLORS['primary']} 0%, #374151 100%); 
           border-radius: 20px; margin-top: 40px; color: white;'>
    <h2 style='color: white; margin-bottom: 20px;'>🏥 Dashboard Executivo {config.COMPANY}</h2>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0;'>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>📊 Análise Inteligente</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Monitoramento em tempo real • KPIs avançados • Alertas automáticos
            </p>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>🎯 Gestão Estratégica</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Pendências • Backlog • Performance • Eficiência operacional
            </p>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>📈 Insights Acionáveis</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Relatórios executivos • Tendências • Recomendações estratégicas
            </p>
        </div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin-top: 20px;'>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.9);'><strong>📧 Suporte Técnico:</strong> equipe-ti@pci.sc.gov.br</p>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.9);'><strong>🔧 Versão do Sistema:</strong> {config.VERSION} - Dashboard Profissional</p>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.8); font-size: 0.85em;'>
            <em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em>
        </p>
    </div>
    <div style='margin-top: 30px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;'>
        <p style='margin: 0; font-size: 0.9em; color: rgba(255,255,255,0.8);'>
            <strong>🚀 Funcionalidades:</strong> Upload inteligente de CSV • Filtros avançados • Alertas contextuais • 
            Relatórios automáticos • Análise de tendências • Gestão de pendências
        </p>
    </div>
    <p style='margin-top: 25px; font-size: 0.85em; color: rgba(255,255,255,0.6);'>
        Sistema desenvolvido para excelência operacional e tomada de decisão estratégica baseada em dados
    </p>
</div>
""", unsafe_allow_html=True)
