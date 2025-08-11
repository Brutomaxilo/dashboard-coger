# Quarta linha - Análise Diária (se dados disponíveis)
    if produtividade_diaria is not None:
        st.markdown("#### 📅 Indicadores de Análise Diária")
        col13, col14, col15, col16 = st.columns(4)
        
        with col13:
            st.metric(
                "Produtividade Diária",
                format_number(produtividade_diaria, 1),
                help="Média de laudos emitidos por dia (dados históricos)"
            )
        
        with col14:
            # Delta para tendência recente
            delta_tendencia = f"+{format_number(tendencia_recente, 1)}%" if tendencia_recente and tendencia_recente > 0 else f"{format_number(tendencia_recente, 1)}%" if tendencia_recente else None
            st.metric(
                "Tendência (30d)",
                "📈" if tendencia_recente and tendencia_recente > 0 else "📉" if tendencia_recente and tendencia_recente < 0 else "➡️",
                delta=delta_tendencia,
                help="Tendência dos últimos 30 dias comparado aos 30 anteriores"
            )
        
        with col15:
            # Indicador de variabilidade
            variabilidade_status = "🟢 Baixa" if variabilidade_diaria and variabilidade_diaria < 30 else "🟡 Média" if variabilidade_diaria and variabilidade_diaria < 50 else "🔴 Alta"
            st.metric(
                "Variabilidade",
                variabilidade_status,
                help=f"Coeficiente de variação: {format_number(variabilidade_diaria, 1)}%" if variabilidade_diaria else "Variabilidade da produção diária"
            )
        
        with col16:
            # Dias úteis vs fins de semana
            if df_laudos_diario is not None and "fim_semana" in df_laudos_diario.columns:
                laudos_dias_uteis = df_laudos_diario[~df_laudos_diario["fim_semana"]]["quantidade"].mean()
                laudos_fins_semana = df_laudos_diario[df_laudos_diario["fim_semana"]]["quantidade"].mean()
                if laudos_fins_semana > 0:
                    ratio_fim_semana = (laudos_fins_semana / laudos_dias_uteis) * 100
                    st.metric(
                        "Atividade Fins Semana",
                        f"{format_number(ratio_fim_semana, 1)}%",
                        help=f"Produção de fins de semana como % dos dias úteis"
                    )
                else:
                    st.metric("Atividade Fins Semana", "0%", help="Sem atividade nos fins de semana")
            else:
                st.metric("Dados Diários", "✅", help="Dados diários carregados e disponíveis")

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

if tendencia_recente and tendencia_recente < -15:
    alerts.append("🔴 **Tendência negativa**: Queda de mais de 15% na produção dos últimos 30 dias")

if variabilidade_diaria and variabilidade_diaria > 60:
    alerts.append("🟡 **Alta variabilidade**: Produção diária muito irregular (>60% de variação)")

if alerts:
    for alert in alerts:
        st.markdown(alert)
else:
    st.success("✅ **Indicadores saudáveis**: Todos os KPIs estão dentro dos parâmetros esperados")# ============ ABA 3: ANÁLISE DIÁRIA ============
with tab3:
    st.subheader("📅 Análise Diária Detalhada")
    
    if df_atend_diario is None and df_laudos_diario is None:
        st.info("📊 Dados diários não disponíveis. Esta aba requer os arquivos de dados diários.")
    else:
        # Seletor de período para análise diária
        st.markdown("#### ⚙️ Controles de Análise")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            # Seletor de ano
            anos_disponiveis = []
            if df_atend_diario is not None and "ano" in df_atend_diario.columns:
                anos_disponiveis.extend(df_atend_diario["ano"].dropna().unique())
            if df_laudos_diario is not None and "ano" in df_laudos_diario.columns:
                anos_disponiveis.extend(df_laudos_diario["ano"].dropna().unique())
            
            anos_disponiveis = sorted(list(set(anos_disponiveis)))
            
            if anos_disponiveis:
                ano_selecionado = st.selectbox(
                    "Ano para análise:",
                    anos_disponiveis,
                    index=len(anos_disponiveis)-1,  # Último ano por padrão
                    help="Selecione o ano para análise detalhada"
                )
            else:
                ano_selecionado = 2024
        
        with col_ctrl2:
            tipo_visualizacao = st.selectbox(
                "Tipo de visualização:",
                ["Série Temporal", "Heatmap Mensal", "Análise Semanal", "Comparativo"],
                help="Escolha como visualizar os dados diários"
            )
        
        with col_ctrl3:
        with col_ctrl3:
            incluir_fins_semana = st.checkbox(
                "Incluir fins de semana",
                value=True,
                help="Marque para incluir sábados e domingos na análise"
            )
        
        # Filtra dados por ano
        df_atend_diario_filtrado = None
        df_laudos_diario_filtrado = None
        
        if df_atend_diario is not None and "ano" in df_atend_diario.columns:
            df_atend_diario_filtrado = df_atend_diario[df_atend_diario["ano"] == ano_selecionado].copy()
            if not incluir_fins_semana and "fim_semana" in df_atend_diario_filtrado.columns:
                df_atend_diario_filtrado = df_atend_diario_filtrado[~df_atend_diario_filtrado["fim_semana"]]
        
        if df_laudos_diario is not None and "ano" in df_laudos_diario.columns:
            df_laudos_diario_filtrado = df_laudos_diario[df_laudos_diario["ano"] == ano_selecionado].copy()
            if not incluir_fins_semana and "fim_semana" in df_laudos_diario_filtrado.columns:
                df_laudos_diario_filtrado = df_laudos_diario_filtrado[~df_laudos_diario_filtrado["fim_semana"]]
        
        # KPIs específicos para dados diários
        st.markdown("#### 📈 KPIs Diários")
        
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        
        # Calcula KPIs diários
        if df_atend_diario_filtrado is not None and not df_atend_diario_filtrado.empty:
            media_atend_dia = df_atend_diario_filtrado["quantidade"].mean()
            max_atend_dia = df_atend_diario_filtrado["quantidade"].max()
            
            with col_k1:
                st.metric(
                    "Média Atend./Dia",
                    format_number(media_atend_dia, 1),
                    help=f"Média diária de atendimentos em {ano_selecionado}"
                )
            
            with col_k2:
                st.metric(
                    "Pico Atendimentos",
                    format_number(max_atend_dia),
                    help=f"Maior número de atendimentos em um dia em {ano_selecionado}"
                )
        
        if df_laudos_diario_filtrado is not None and not df_laudos_diario_filtrado.empty:
            media_laudos_dia = df_laudos_diario_filtrado["quantidade"].mean()
            max_laudos_dia = df_laudos_diario_filtrado["quantidade"].max()
            
            with col_k3:
                st.metric(
                    "Média Laudos/Dia",
                    format_number(media_laudos_dia, 1),
                    help=f"Média diária de laudos em {ano_selecionado}"
                )
            
            with col_k4:
                st.metric(
                    "Pico Laudos",
                    format_number(max_laudos_dia),
                    help=f"Maior número de laudos em um dia em {ano_selecionado}"
                )
        
        # Visualizações baseadas no tipo selecionado
        if tipo_visualizacao == "Série Temporal":
            st.markdown("#### 📈 Série Temporal Diária")
            
            # Gráfico de série temporal
            fig_temporal = go.Figure()
            
            if df_atend_diario_filtrado is not None and not df_atend_diario_filtrado.empty:
                fig_temporal.add_trace(go.Scatter(
                    x=df_atend_diario_filtrado["data_completa"],
                    y=df_atend_diario_filtrado["quantidade"],
                    mode='lines',
                    name='Atendimentos',
                    line=dict(color='blue', width=1),
                    hovertemplate='<b>Atendimentos</b><br>Data: %{x}<br>Quantidade: %{y}<extra></extra>'
                ))
            
            if df_laudos_diario_filtrado is not None and not df_laudos_diario_filtrado.empty:
                fig_temporal.add_trace(go.Scatter(
                    x=df_laudos_diario_filtrado["data_completa"],
                    y=df_laudos_diario_filtrado["quantidade"],
                    mode='lines',
                    name='Laudos',
                    line=dict(color='green', width=1),
                    hovertemplate='<b>Laudos</b><br>Data: %{x}<br>Quantidade: %{y}<extra></extra>'
                ))
            
            fig_temporal.update_layout(
                title=f"Evolução Diária - {ano_selecionado}",
                xaxis_title="Data",
                yaxis_title="Quantidade",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Adiciona médias móveis
            st.markdown("#### 📊 Tendências com Médias Móveis")
            
            fig_tendencia = go.Figure()
            
            if df_laudos_diario_filtrado is not None and not df_laudos_diario_filtrado.empty:
                # Média móvel de 7 dias
                df_laudos_diario_filtrado["media_movel_7"] = df_laudos_diario_filtrado["quantidade"].rolling(window=7).mean()
                # Média móvel de 30 dias
                df_laudos_diario_filtrado["media_movel_30"] = df_laudos_diario_filtrado["quantidade"].rolling(window=30).mean()
                
                fig_tendencia.add_trace(go.Scatter(
                    x=df_laudos_diario_filtrado["data_completa"],
                    y=df_laudos_diario_filtrado["quantidade"],
                    mode='lines',
                    name='Laudos (diário)',
                    line=dict(color='lightblue', width=0.5),
                    opacity=0.6
                ))
                
                fig_tendencia.add_trace(go.Scatter(
                    x=df_laudos_diario_filtrado["data_completa"],
                    y=df_laudos_diario_filtrado["media_movel_7"],
                    mode='lines',
                    name='Média Móvel 7 dias',
                    line=dict(color='orange', width=2)
                ))
                
                fig_tendencia.add_trace(go.Scatter(
                    x=df_laudos_diario_filtrado["data_completa"],
                    y=df_laudos_diario_filtrado["media_movel_30"],
                    mode='lines',
                    name='Média Móvel 30 dias',
                    line=dict(color='red', width=2)
                ))
            
            fig_tendencia.update_layout(
                title=f"Tendências de Laudos com Médias Móveis - {ano_selecionado}",
                xaxis_title="Data",
                yaxis_title="Quantidade de Laudos",
                height=400
            )
            
            st.plotly_chart(fig_tendencia, use_container_width=True)
        
        elif tipo_visualizacao == "Heatmap Mensal":
            st.markdown("#### 🔥 Heatmap Mensal")
            
            if df_laudos_diario_filtrado is not None and not df_laudos_diario_filtrado.empty:
                # Prepara dados para heatmap
                df_heatmap = df_laudos_diario_filtrado.copy()
                df_heatmap["dia"] = df_heatmap["data_completa"].dt.day
                df_heatmap["mes"] = df_heatmap["data_completa"].dt.month
                
                # Cria matriz para heatmap
                heatmap_data = df_heatmap.pivot_table(
                    values='quantidade', 
                    index='mes', 
                    columns='dia', 
                    aggfunc='mean'
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Dia do Mês", y="Mês", color="Laudos"),
                    title=f"Heatmap de Produção de Laudos - {ano_selecionado}",
                    color_continuous_scale="RdYlGn"
                )
                
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        elif tipo_visualizacao == "Análise Semanal":
            st.markdown("#### 📅 Padrões por Dia da Semana")
            
            col_sem1, col_sem2 = st.columns(2)
            
            with col_sem1:
                if df_atend_diario_filtrado is not None and "dia_semana" in df_atend_diario_filtrado.columns:
                    atend_por_dia_semana = (
                        df_atend_diario_filtrado
                        .groupby("dia_semana")["quantidade"]
                        .mean()
                        .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    )
                    
                    fig_sem_atend = px.bar(
                        x=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
                        y=atend_por_dia_semana.values,
                        title="Média de Atendimentos por Dia da Semana",
                        color=atend_por_dia_semana.values,
                        color_continuous_scale="Blues"
                    )
                    fig_sem_atend.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_sem_atend, use_container_width=True)
            
            with col_sem2:
                if df_laudos_diario_filtrado is not None and "dia_semana" in df_laudos_diario_filtrado.columns:
                    laudos_por_dia_semana = (
                        df_laudos_diario_filtrado
                        .groupby("dia_semana")["quantidade"]
                        .mean()
                        .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    )
                    
                    fig_sem_laudos = px.bar(
                        x=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
                        y=laudos_por_dia_semana.values,
                        title="Média de Laudos por Dia da Semana",
                        color=laudos_por_dia_semana.values,
                        color_continuous_scale="Greens"
                    )
                    fig_sem_laudos.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_sem_laudos, use_container_width=True)
            
            # Análise de sazonalidade mensal
            st.markdown("#### 📈 Sazonalidade Mensal")
            
            if df_laudos_diario_filtrado is not None and "mes" in df_laudos_diario_filtrado.columns:
                laudos_por_mes = (
                    df_laudos_diario_filtrado
                    .groupby("mes")["quantidade"]
                    .agg(['mean', 'std', 'count'])
                    .reset_index()
                )
                
                meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                
                fig_sazonalidade = go.Figure()
                
                fig_sazonalidade.add_trace(go.Bar(
                    x=[meses_nomes[i-1] for i in laudos_por_mes["mes"]],
                    y=laudos_por_mes["mean"],
                    error_y=dict(type='data', array=laudos_por_mes["std"]),
                    name='Média ± Desvio Padrão',
                    marker_color='lightgreen'
                ))
                
                fig_sazonalidade.update_layout(
                    title=f"Sazonalidade Mensal de Laudos - {ano_selecionado}",
                    xaxis_title="Mês",
                    yaxis_title="Média de Laudos por Dia",
                    height=400
                )
                
                st.plotly_chart(fig_sazonalidade, use_container_width=True)
        
        elif tipo_visualizacao == "Comparativo":
            st.markdown("#### ⚖️ Análise Comparativa")
            
            # Comparação com anos anteriores se disponível
            anos_para_comparar = []
            if df_laudos_diario is not None and "ano" in df_laudos_diario.columns:
                anos_para_comparar = sorted(df_laudos_diario["ano"].unique())
            
            if len(anos_para_comparar) > 1:
                anos_selecionados = st.multiselect(
                    "Selecione anos para comparar:",
                    anos_para_comparar,
                    default=anos_para_comparar[-2:],  # Últimos 2 anos
                    help="Escolha quais anos comparar"
                )
                
                if len(anos_selecionados) >= 2:
                    fig_comparativo = go.Figure()
                    
                    cores = ['blue', 'red', 'green', 'orange', 'purple']
                    
                    for i, ano in enumerate(anos_selecionados):
                        df_ano = df_laudos_diario[df_laudos_diario["ano"] == ano]
                        
                        # Agrupa por mês para comparação
                        df_ano_mensal = (
                            df_ano.groupby("mes")["quantidade"]
                            .mean()
                            .reset_index()
                        )
                        
                        fig_comparativo.add_trace(go.Scatter(
                            x=df_ano_mensal["mes"],
                            y=df_ano_mensal["quantidade"],
                            mode='lines+markers',
                            name=f'Ano {ano}',
                            line=dict(color=cores[i % len(cores)], width=2)
                        ))
                    
                    fig_comparativo.update_layout(
                        title="Comparativo de Produção Mensal entre Anos",
                        xaxis_title="Mês",
                        yaxis_title="Média de Laudos por Dia",
                        height=400,
                        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                                  ticktext=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                                           'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
                    )
                    
                    st.plotly_chart(fig_comparativo, use_container_width=True)
            
            # Correlação entre atendimentos e laudos diários
            if (df_atend_diario_filtrado is not None and df_laudos_diario_filtrado is not None and
                not df_atend_diario_filtrado.empty and not df_laudos_diario_filtrado.empty):
                
                st.markdown("#### 🔗 Correlação Atendimentos vs Laudos")
                
                # Merge dos dados por data
                df_correlacao = pd.merge(
                    df_atend_diario_filtrado[["data_completa", "quantidade"]].rename(columns={"quantidade": "atendimentos"}),
                    df_laudos_diario_filtrado[["data_completa", "quantidade"]].rename(columns={"quantidade": "laudos"}),
                    on="data_completa",
                    how="inner"
                )
                
                if not df_correlacao.empty:
                    correlacao = df_correlacao["atendimentos"].corr(df_correlacao["laudos"])
                    
                    fig_corr = px.scatter(
                        df_correlacao,
                        x="atendimentos",
                        y="laudos",
                        title=f"Correlação Diária: Atendimentos vs Laudos (r = {correlacao:.3f})",
                        trendline="ols"
                    )
                    
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Interpretação da correlação
                    if correlacao > 0.7:
                        st.success(f"🟢 **Correlação Forte** (r = {correlacao:.3f}): Atendimentos e laudos estão bem alinhados")
                    elif correlacao > 0.4:
                        st.warning(f"🟡 **Correlação Moderada** (r = {correlacao:.3f}): Há algum alinhamento entre atendimentos e laudos")
                    else:
                        st.error(f"🔴 **Correlação Fraca** (r = {correlacao:.3f}): Atendimentos e laudos não estão bem correlacionados")

# ============ ABA 4: RANKINGS ============
with tab4:import io
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
    "atendimentos_Diario": {
        "label": "Atendimentos Diários",
        "description": "Dados diários de atendimentos (2019-2025) - análise temporal detalhada",
        "pattern": ["atendimentos_diario", "atendimentos diario", "atendimentos_diário"]
    },
    "laudos_Diario": {
        "label": "Laudos Diários",
        "description": "Dados diários de laudos (2019-2025) - análise temporal detalhada",
        "pattern": ["laudos_diario", "laudos diario", "laudos_diário"]
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
    "atendimentos_Diario": {
        "date": "data_interesse",
        "id": "idatendimento", 
        "quantidade": "idatendimento"  # O ID representa a quantidade diária de atendimentos
    },
    "laudos_Diario": {
        "date": "data_interesse",
        "id": "iddocumento",
        "quantidade": "iddocumento"  # O ID representa a quantidade diária de laudos
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
    
    # Para arquivos mensais e diários, o valor na coluna ID já representa a quantidade
    if name in ["Atendimentos_todos_Mensal", "Laudos_todos_Mensal", 
                "Atendimentos_especifico_Mensal", "Laudos_especifico_Mensal",
                "atendimentos_Diario", "laudos_Diario"]:
        
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
        
        # Para dados diários, adiciona também informações de dia
        if name in ["atendimentos_Diario", "laudos_Diario"]:
            result["data_completa"] = result["anomês_dt"]
            result["dia_semana"] = result["anomês_dt"].dt.day_name()
            result["dia_semana_num"] = result["anomês_dt"].dt.dayofweek
            result["fim_semana"] = result["dia_semana_num"].isin([5, 6])  # Sábado e Domingo
            result["trimestre"] = result["anomês_dt"].dt.quarter
            result["semana_ano"] = result["anomês_dt"].dt.isocalendar().week
    
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
                   else f"{standardized_df['data_completa'].min().strftime('%Y-%m-%d')} a {standardized_df['data_completa'].max().strftime('%Y-%m-%d')}"
                   if 'data_completa' in standardized_df.columns and not standardized_df['data_completa'].isna().all()
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
df_atend_diario = filtered_dfs.get("atendimentos_Diario")
df_laudos_diario = filtered_dfs.get("laudos_Diario")
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
