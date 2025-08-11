import io, os, re, zipfile
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =================== CONFIG ===================
st.set_page_config(page_title="PCI/SC – Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 Dashboard PCI/SC – Produção & Pendências")
st.markdown("---")

# =================== CACHES ===================
@st.cache_data
def read_csv_any(content: bytes) -> Optional[pd.DataFrame]:
    """Lê CSV detectando separador/encoding."""
    seps = [";", ",", "\t", "|"]
    encs = ["utf-8", "latin-1", "cp1252"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, engine="python", encoding=enc)
                if df.shape[1] > 1:
                    df.columns = [c.strip().strip('"').lower() for c in df.columns]
                    for c in df.columns:
                        if df[c].dtype == object:
                            df[c] = df[c].astype(str).str.strip().str.strip('"')
                    return df
            except Exception:
                continue
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        if df.shape[1] > 1:
            df.columns = [c.strip().strip('"').lower() for c in df.columns]
            return df
    except Exception:
        pass
    return None

@st.cache_data
def process_datetime_column(series: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    if series is None or series.empty:
        return None
    s = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
    if s.isna().sum() > len(s) * 0.5:
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
            s = pd.to_datetime(series, format=fmt, errors="coerce")
            if s.notna().sum() > len(series) * 0.5:
                break
    return s if s.notna().any() else None

# =================== HELPERS ===================
def format_number(v: float, d: int = 0) -> str:
    if pd.isna(v): return "—"
    try:
        return (f"{int(round(v)):,}".replace(",", ".")
                if d == 0 else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", "."))
    except Exception:
        return "—"

QUANTITY_PATTERNS = re.compile(r"(quant|qtd|qtde|total|volume|contagem|qte|qtda)", re.I)
COMPETENCIA_CANDIDATES = ["txcompetencia", "competencia", "ano_mes", "anomes", "ano_mes_competencia"]
TIPO_CANDIDATES = ["txtipopericia", "tipopericia", "tipo_pericia", "tipo"]
UNIDADE_CANDIDATES = ["unidade_emissao", "unidade", "unidade_atendimento"]
ID_CANDIDATES = ["iddocumento", "id_documento", "idatendimento", "id_atendimento", "n_laudo", "numero_laudo"]
DATE_CANDIDATES = ["data_interesse", "data", "dia", "data_base", "dhemitido", "dhatendimento", "dhsolicitacao", "data_emissao"]

def infer_quantity_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if QUANTITY_PATTERNS.search(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def pick_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    if explicit and explicit in df.columns: return explicit
    for c in candidates:
        if c in df.columns: return c
    return None

# =================== ENTRADAS ===================
st.sidebar.header("📁 Dados")

# Upload opcional de .zip com a pasta data/
zip_file = st.sidebar.file_uploader("Enviar pasta data compactada (.zip)", type=["zip"])
uploaded_files = {}
st.sidebar.markdown("ou envie os CSVs individualmente:")

expected = {
    "Atendimentos_todos_Mensal": ["atendimentos_todos", "atendimentos todos"],
    "Laudos_todos_Mensal": ["laudos_todos", "laudos todos"],
    "Atendimentos_especifico_Mensal": ["atendimentos_especifico", "atendimentos especifico"],
    "Laudos_especifico_Mensal": ["laudos_especifico", "laudos especifico"],
    "laudos_realizados": ["laudos_realizados", "laudos realizados"],
    "detalhes_laudospendentes": ["laudospendentes", "laudos_pendentes", "detalhes_laudospendentes"],
    "detalhes_examespendentes": ["examespendentes", "exames_pendentes", "detalhes_examespendentes"],
    "Atendimentos_diario": ["atendimentos_diario", "atendimentos diário"],
    "Laudos_diario": ["laudos_diario", "laudos diário"],
}

if not zip_file:
    for key, _ in expected.items():
        uploaded_files[key] = st.sidebar.file_uploader(f"{key} (.csv)", key=f"up_{key}")

@st.cache_data
def load_data_from_zip(zbytes: bytes) -> Dict[str, pd.DataFrame]:
    out = {}
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"): continue
            content = zf.read(name)
            df = read_csv_any(content)
            if df is None: continue
            base = os.path.splitext(os.path.basename(name))[0].lower()
            norm = re.sub(r"[^\w]", "_", base)
            # vincula ao esperado pelo padrão
            for key, pats in expected.items():
                if any(p in norm for p in pats) or norm.startswith(key.lower()):
                    out[key] = df
    return out

@st.cache_data
def load_data_from_uploads(files_dict: Dict[str, any]) -> Dict[str, pd.DataFrame]:
    out = {}
    for key, f in files_dict.items():
        if f is None: continue
        try:
            out[key] = read_csv_any(f.read())
        except Exception:
            pass
    return out

raw = {}
if zip_file:
    raw = load_data_from_zip(zip_file.getvalue())
else:
    raw = load_data_from_uploads(uploaded_files)

if not raw:
    st.warning("Envie os dados (ZIP ou CSVs).")
    st.stop()

# =================== MAPEAMENTOS ===================
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
        "perito": "perito",
    },
    "detalhes_examespendentes": {
        "date": "data_solicitacao",
        "ano": "ano_sol",
        "id": "caso_sirsaelp",
        "unidade": "unidade",
        "superintendencia": "superintendencia",
        "diretoria": "diretoria",
        "competencia": "competencia",
        "tipo": "tipopericia",
    },
    # Mensais (corrigido: têm 'competencia' e NÃO usam id como quantidade)
    "Atendimentos_todos_Mensal": {"date": "data_interesse", "competencia": "txcompetencia"},
    "Laudos_todos_Mensal": {"date": "data_interesse", "competencia": "txcompetencia"},
    # Específicos (corrigido: tipo é txtipopericia; competencia existe)
    "Atendimentos_especifico_Mensal": {"date": "data_interesse", "competencia": "txcompetencia", "tipo": "txtipopericia"},
    "Laudos_especifico_Mensal": {"date": "data_interesse", "competencia": "txcompetencia", "tipo": "txtipopericia"},
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
        "perito": "perito",
    },
    "Atendimentos_diario": {"date": "data_interesse"},
    "Laudos_diario": {"date": "data_interesse"},
}

# =================== PADRONIZAÇÃO ===================
@st.cache_data
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    mapping = COLUMN_MAPPINGS.get(name, {})
    result = df.copy()

    # -------- QUANTIDADE (NUNCA usar ID como quantidade) --------
    qcol = infer_quantity_col(result)
    if qcol:
        result["quantidade"] = pd.to_numeric(result[qcol], errors="coerce").fillna(0)
    else:
        result["quantidade"] = 1

    # -------- DIMENSÕES --------
    for dim, cands in [("tipo", TIPO_CANDIDATES), ("unidade", UNIDADE_CANDIDATES),
                       ("perito", ["perito"]), ("diretoria", ["diretoria"]), ("superintendencia", ["superintendencia"])]:
        chosen = pick_col(result, mapping.get(dim), cands)
        if chosen: result[dim] = result[chosen]

    id_col = pick_col(result, mapping.get("id"), ID_CANDIDATES)
    if id_col: result["id"] = result[id_col]

    # -------- COMPETÊNCIA (mensal) --------
    anomes_dt = None
    comp_col = pick_col(result, mapping.get("competencia"), COMPETENCIA_CANDIDATES)
    if comp_col and comp_col in result.columns:
        raw = result[comp_col].astype(str).str.replace(r"[^\d/.\-]", "", regex=True)
        tries = [
            ("%Y-%m", False),
            ("%Y/%m", False),
            ("%Y%m", False),
            ("%m/%Y", True),
        ]
        for fmt, _ in tries:
            anomes_dt = pd.to_datetime(raw, errors="coerce", format=fmt)
            if anomes_dt.notna().any(): break
        if anomes_dt is None or anomes_dt.isna().all():
            anomes_dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        if anomes_dt.notna().any():
            anomes_dt = anomes_dt.dt.to_period("M").dt.to_timestamp()

    # -------- DATA DIÁRIA --------
    day_col = pick_col(result, mapping.get("date"), DATE_CANDIDATES)
    data_base = process_datetime_column(result[day_col]) if day_col and day_col in result.columns else None

    if anomes_dt is not None and anomes_dt.notna().any():
        result["anomês_dt"] = anomes_dt
        result["anomês"] = anomes_dt.dt.strftime("%Y-%m")
        result["ano"] = anomes_dt.dt.year
        result["mes"] = anomes_dt.dt.month

    if data_base is not None and data_base.notna().any():
        result["data_base"] = data_base
        result["dia"] = data_base.dt.normalize()
    elif "anomês_dt" in result.columns:
        result["dia"] = pd.to_datetime(result["anomês_dt"]).dt.normalize()

    # -------- Regras específicas: laudos_realizados --------
    if name == "laudos_realizados":
        for field in ["solicitacao", "atendimento", "emissao"]:
            col = mapping.get(field)
            if col in result.columns:
                result[f"dh{field}"] = process_datetime_column(result[col])
        if "dhemissao" in result.columns:
            base_date = result.get("dhatendimento") if "dhatendimento" in result.columns else result.get("dhsolicitacao")
            if base_date is not None:
                result["tme_dias"] = (result["dhemissao"] - base_date).dt.days
                result["sla_30_ok"] = result["tme_dias"] <= 30
                result["sla_60_ok"] = result["tme_dias"] <= 60

    # -------- limpeza texto --------
    for col in ["diretoria", "superintendencia", "unidade", "tipo", "id", "perito", "anomês"]:
        if col in result.columns:
            result[col] = (result[col].astype(str).str.strip().str.title().replace({"Nan": None, "": None, "None": None}))

    return result

# Padroniza tudo
standardized = {k: standardize_dataframe(k, v) for k, v in raw.items()}

# =================== FILTROS ===================
def extract_filter_values(column: str) -> List[str]:
    vals = set()
    for df in standardized.values():
        if column in df.columns:
            vals.update(df[column].dropna().astype(str))
    return sorted(vals)

st.sidebar.subheader("🔍 Filtros")
f_diretoria = st.sidebar.multiselect("Diretoria", extract_filter_values("diretoria"))
f_super = st.sidebar.multiselect("Superintendência", extract_filter_values("superintendencia"))
f_unidade = st.sidebar.multiselect("Unidade", extract_filter_values("unidade"))
f_tipo = st.sidebar.multiselect("Tipo de Perícia", extract_filter_values("tipo"))
period_options = ["Todo o período", "Últimos 6 meses", "Últimos 3 meses", "Ano atual"]
f_periodo = st.sidebar.selectbox("Período de análise", period_options)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    for col, flt in [("diretoria", f_diretoria), ("superintendencia", f_super), ("unidade", f_unidade), ("tipo", f_tipo)]:
        if flt and col in out.columns:
            out = out[out[col].astype(str).isin(flt)]
    if "anomês_dt" in out.columns and f_periodo != "Todo o período":
        maxd = out["anomês_dt"].max()
        if pd.notna(maxd):
            if f_periodo == "Últimos 3 meses": cutoff = maxd - pd.DateOffset(months=3)
            elif f_periodo == "Últimos 6 meses": cutoff = maxd - pd.DateOffset(months=6)
            elif f_periodo == "Ano atual": cutoff = pd.Timestamp(maxd.year, 1, 1)
            else: cutoff = None
            if cutoff is not None:
                out = out[out["anomês_dt"] >= cutoff]
    return out

fdfs = {k: apply_filters(v) for k, v in standardized.items()}

# atalhos
df_atend_todos = fdfs.get("Atendimentos_todos_Mensal")
df_laudos_todos = fdfs.get("Laudos_todos_Mensal")
df_atend_esp   = fdfs.get("Atendimentos_especifico_Mensal")
df_laudos_esp  = fdfs.get("Laudos_especifico_Mensal")
df_laudos_real = fdfs.get("laudos_realizados")
df_pend_laudos = fdfs.get("detalhes_laudospendentes")
df_pend_exames = fdfs.get("detalhes_examespendentes")
df_atend_diario= fdfs.get("Atendimentos_diario")
df_laudos_diario= fdfs.get("Laudos_diario")

# =================== KPIs ===================
def total(df): 
    return int(df["quantidade"].sum()) if (df is not None and not df.empty and "quantidade" in df.columns) else 0

def media_mensal(df):
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns: return None
    m = df.groupby("anomês_dt")["quantidade"].sum()
    return float(m.mean()) if not m.empty else None

def crescimento(df, periods=3):
    if df is None or df.empty or "anomês_dt" not in df.columns or "quantidade" not in df.columns: return None
    s = df.groupby("anomês_dt")["quantidade"].sum().sort_index().tail(periods*2)
    if len(s) < 2: return None
    mid = len(s)//2
    a, b = s.iloc[:mid].mean(), s.iloc[mid:].mean()
    return ((b-a)/a)*100 if a>0 else None

def produtividade(dfA, dfL):
    met = {}
    if dfA is not None and dfL is not None:
        A, L = total(dfA), total(dfL)
        if A>0: met["taxa_conversao"] = (L/A)*100
        if ("anomês_dt" in dfA.columns and "anomês_dt" in dfL.columns):
            a = dfA.groupby("anomês_dt")["quantidade"].sum()
            l = dfL.groupby("anomês_dt")["quantidade"].sum()
            idx = a.index.intersection(l.index)
            if len(idx)>3:
                c = a.loc[idx].corr(l.loc[idx])
                met["correlacao"] = float(c) if not pd.isna(c) else None
    return met

tot_at = total(df_atend_todos)
tot_ld = total(df_laudos_todos)
med_ld = media_mensal(df_laudos_todos)
backlog = (len(df_pend_laudos) / med_ld) if (df_pend_laudos is not None and not df_pend_laudos.empty and med_ld and med_ld>0) else None

prod = produtividade(df_atend_todos, df_laudos_todos)
tx_conv = prod.get("taxa_conversao")
corr_al = prod.get("correlacao")
cres_at = crescimento(df_atend_todos)
cres_ld = crescimento(df_laudos_todos)

tme_med = tme_avg = sla30 = sla60 = None
if df_laudos_real is not None and not df_laudos_real.empty:
    if "tme_dias" in df_laudos_real.columns:
        t = pd.to_numeric(df_laudos_real["tme_dias"], errors="coerce").dropna()
        if not t.empty: tme_med, tme_avg = float(t.median()), float(t.mean())
    if "sla_30_ok" in df_laudos_real.columns: sla30 = float(df_laudos_real["sla_30_ok"].mean()*100)
    if "sla_60_ok" in df_laudos_real.columns: sla60 = float(df_laudos_real["sla_60_ok"].mean()*100)

aging_laudos = aging_exames = None
def mean_aging(df):
    if df is None or df.empty: return None
    cands = [c for c in df.columns if "data" in c]
    if not cands: return None
    d = pd.to_datetime(df[cands[0]], errors="coerce")
    if d.notna().any():
        hoje = pd.Timestamp.now().normalize()
        return float(((hoje - d).dt.days).mean())
    return None

aging_laudos = mean_aging(df_pend_laudos)
aging_exames = mean_aging(df_pend_exames)

# =================== KPIs VISUAIS ===================
st.subheader("📈 Indicadores Principais")
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Atendimentos Totais", format_number(tot_at), delta=(f"+{format_number(cres_at,1)}%" if cres_at is not None else None))
with c2: st.metric("Laudos Emitidos", format_number(tot_ld), delta=(f"+{format_number(cres_ld,1)}%" if cres_ld is not None else None))
with c3: st.metric("Taxa de Conversão", f"{format_number(tx_conv,1)}%" if tx_conv else "—")
with c4: st.metric("Produtividade Mensal", format_number(med_ld,1) if med_ld else "—")

st.markdown("#### ⏰ Gestão de Pendências")
c5,c6,c7,c8 = st.columns(4)
with c5: st.metric("Laudos Pendentes", format_number(len(df_pend_laudos) if df_pend_laudos is not None else 0))
with c6: st.metric("Exames Pendentes", format_number(len(df_pend_exames) if df_pend_exames is not None else 0))
with c7: st.metric("Backlog (meses)", format_number(backlog,1) if backlog else "—")
with c8: st.metric("Aging Médio (dias)", format_number(aging_laudos or aging_exames,0) if (aging_laudos or aging_exames) else "—")

if tme_med is not None or sla30 is not None:
    st.markdown("#### 🎯 Indicadores de Performance")
    c9,c10,c11,c12 = st.columns(4)
    with c9: st.metric("TME Mediano (dias)", format_number(tme_med,1) if tme_med else "—")
    with c10: st.metric("TME Médio (dias)", format_number(tme_avg,1) if tme_avg else "—")
    with c11: st.metric("SLA 30 dias", f"{format_number(sla30,1)}%" if sla30 else "—")
    with c12: st.metric("SLA 60 dias", f"{format_number(sla60,1)}%" if sla60 else "—")

st.markdown("#### 🚨 Alertas e Insights")
alerts=[]
if backlog and backlog>6: alerts.append("🔴 **Backlog crítico**: > 6 meses")
elif backlog and backlog>3: alerts.append("🟡 **Atenção**: backlog > 3 meses")
if sla30 and sla30<70: alerts.append("🔴 **SLA 30 baixo**: < 70%")
if tx_conv and tx_conv<50: alerts.append("🟡 **Conversão baixa**: < 50%")
if cres_ld and cres_ld<-10: alerts.append("🔴 **Queda na produção**: laudos -10%+")
if corr_al and corr_al<0.5: alerts.append("🟡 **Descorrelação** entre atendimentos e laudos")
if alerts: [st.markdown(a) for a in alerts]
else: st.success("✅ Indicadores dentro do esperado")

st.markdown("---")

# =================== ABAS ===================
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "📊 Visão Geral","📈 Tendências","🏆 Rankings","⏰ Pendências","📋 Dados","📑 Relatórios","📅 Diário","🧪 Validação"
])

# -------- VISÃO GERAL --------
with tab1:
    st.subheader("📊 Resumo Executivo")
    if df_laudos_todos is not None and not df_laudos_todos.empty and "unidade" in df_laudos_todos.columns:
        colL,colR = st.columns(2)
        with colL:
            st.markdown("#### 🏢 Performance por Unidade")
            u = (df_laudos_todos.groupby("unidade",as_index=False)["quantidade"].sum()
                 .sort_values("quantidade",ascending=False).head(15))
            fig = px.bar(u, x="quantidade", y="unidade", orientation="h",
                         title="Top 15 Unidades - Laudos", color="quantidade", color_continuous_scale="Blues", text="quantidade")
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with colR:
            if "tipo" in df_laudos_todos.columns:
                st.markdown("#### 🔍 Distribuição por Tipo")
                t = (df_laudos_todos.groupby("tipo",as_index=False)["quantidade"].sum()
                     .sort_values("quantidade",ascending=False).head(10))
                pie = px.pie(t, values="quantidade", names="tipo", title="Top 10 Tipos")
                pie.update_traces(textposition='inside', textinfo='percent+label')
                pie.update_layout(height=500)
                st.plotly_chart(pie, use_container_width=True)

    if (df_atend_todos is not None and df_laudos_todos is not None and
        "anomês_dt" in df_atend_todos.columns and "anomês_dt" in df_laudos_todos.columns):
        st.markdown("#### 📅 Evolução Mensal: Atendimentos vs Laudos")
        a = df_atend_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        a["Tipo"]="Atendimentos"; a.rename(columns={"quantidade":"Total"}, inplace=True)
        l = df_laudos_todos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        l["Tipo"]="Laudos"; l.rename(columns={"quantidade":"Total"}, inplace=True)
        comb = pd.concat([a,l]); comb["Mês"]=comb["anomês_dt"].dt.strftime("%Y-%m")
        line = px.line(comb, x="Mês", y="Total", color="Tipo", markers=True, line_shape="spline")
        line.update_layout(height=400, hovermode="x unified", xaxis_title="Período", yaxis_title="Quantidade")
        st.plotly_chart(line, use_container_width=True)

# -------- TENDÊNCIAS --------
with tab2:
    st.subheader("📈 Análise de Tendências")

    def trend(df: pd.DataFrame, title: str, color="blue"):
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}"); return
        m = df.groupby("anomês_dt",as_index=False)["quantidade"].sum().sort_values("anomês_dt")
        if m.empty: st.info(f"Sem dados para {title}"); return
        m["Mês"]=m["anomês_dt"].dt.strftime("%Y-%m")
        fig = make_subplots(rows=2, cols=1, subplot_titles=(title, "Variação Percentual Mensal"),
                            vertical_spacing=0.15, row_heights=[0.7,0.3])
        fig.add_trace(go.Scatter(x=m["Mês"], y=m["quantidade"], mode="lines+markers",
                                 name="Valores", line=dict(color=color, width=2)), row=1,col=1)
        if len(m)>=3:
            m["mm3"]=m["quantidade"].rolling(3,center=True).mean()
            fig.add_trace(go.Scatter(x=m["Mês"], y=m["mm3"], mode="lines", name="Média Móvel (3m)",
                                     line=dict(dash="dash", color="red", width=2)), row=1,col=1)
        m["var%"]=m["quantidade"].pct_change()*100
        colors = ['red' if x<0 else 'green' for x in m["var%"].fillna(0)]
        fig.add_trace(go.Bar(x=m["Mês"], y=m["var%"], marker_color=colors, showlegend=False), row=2,col=1)
        fig.update_layout(height=600, hovermode="x unified")
        fig.update_xaxes(title_text="Período", row=2,col=1)
        fig.update_yaxes(title_text="Quantidade", row=1,col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2,col=1)
        st.plotly_chart(fig, use_container_width=True)

    colA,colB = st.columns(2)
    with colA:
        trend(df_atend_todos, "🏥 Atendimentos - Análise Temporal", "blue")
    with colB:
        trend(df_laudos_todos, "📄 Laudos - Análise Temporal", "green")

# -------- RANKINGS --------
with tab3:
    st.subheader("🏆 Rankings e Comparativos")
    def ranking(df: pd.DataFrame, dim: str, title: str, top_n=20):
        if df is None or df.empty or dim not in df.columns:
            st.info(f"Dados insuficientes para {title}"); return
        r = (df.groupby(dim).agg({"quantidade":["sum","count","mean"]}).round(2))
        r.columns=["Total","Registros","Média"]; r=r.sort_values("Total",ascending=False).head(top_n).reset_index()
        fig = px.bar(r, x="Total", y=dim, orientation="h", title=title,
                     color="Total", color_continuous_scale="Viridis", hover_data=["Registros","Média"])
        fig.update_layout(height=max(400, len(r)*30), showlegend=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"📊 Detalhes - {title}"):
            st.dataframe(r, use_container_width=True)

    t1,t2,t3,t4 = st.tabs(["Por Diretoria","Por Unidade","Por Tipo","Comparativo"])
    with t1:
        c1,c2 = st.columns(2)
        with c1: ranking(df_atend_todos,"diretoria","🏥 Atendimentos por Diretoria")
        with c2: ranking(df_laudos_todos,"diretoria","📄 Laudos por Diretoria")
    with t2:
        c1,c2 = st.columns(2)
        with c1: ranking(df_atend_todos,"unidade","🏥 Atendimentos por Unidade",25)
        with c2: ranking(df_laudos_todos,"unidade","📄 Laudos por Unidade",25)
    with t3:
        c1,c2 = st.columns(2)
        with c1: ranking(df_atend_esp,"tipo","🏥 Atendimentos por Tipo",20)
        with c2: ranking(df_laudos_esp,"tipo","📄 Laudos por Tipo",20)
    with t4:
        st.markdown("#### 📊 Eficiência por Unidade")
        if (df_atend_todos is not None and df_laudos_todos is not None and
            "unidade" in df_atend_todos.columns and "unidade" in df_laudos_todos.columns):
            A = df_atend_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade":"Atendimentos"})
            L = df_laudos_todos.groupby("unidade")["quantidade"].sum().reset_index().rename(columns={"quantidade":"Laudos"})
            e = pd.merge(A,L,on="unidade",how="inner")
            if not e.empty:
                e["Taxa_Conversao"]=(e["Laudos"]/e["Atendimentos"]*100).round(1)
                e = e.sort_values("Taxa_Conversao",ascending=False)
                fig = px.scatter(e.head(20), x="Atendimentos", y="Laudos", size="Taxa_Conversao",
                                 hover_name="unidade", color="Taxa_Conversao", color_continuous_scale="RdYlGn",
                                 title="Atendimentos vs Laudos (bolha = taxa)")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(e.head(10)[["unidade","Taxa_Conversao","Atendimentos","Laudos"]], use_container_width=True)

# -------- PENDÊNCIAS --------
with tab4:
    st.subheader("⏰ Gestão de Pendências")

    def aging_analysis(df: pd.DataFrame, date_col: str="data_base") -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if df is None or df.empty: return pd.DataFrame(), pd.Series(dtype="int64"), {"total": 0}
        cands = [c for c in df.columns if "data" in c.lower()]
        if date_col not in df.columns and cands: date_col = cands[0]
        if date_col not in df.columns: return df, pd.Series(dtype="int64"), {"total": len(df)}
        res = df.copy()
        d = pd.to_datetime(res[date_col], errors="coerce")
        if d.isna().all(): return df, pd.Series(dtype="int64"), {"total": len(df)}
        hoje = pd.Timestamp.now().normalize()
        dias = (hoje - d).dt.days
        faixas = pd.cut(dias, bins=[-1,15,30,60,90,180,365,float('inf')],
                        labels=["0-15 dias","16-30 dias","31-60 dias","61-90 dias","91-180 dias","181-365 dias","> 365 dias"])
        res["dias_pendentes"]=dias; res["faixa_aging"]=faixas
        res["prioridade"]=pd.cut(dias, bins=[-1,30,90,180,float('inf')], labels=["Normal","Atenção","Urgente","Crítico"])
        dist = faixas.value_counts().sort_index()
        stats = {"total": len(res), "media_dias": float(dias.mean()), "mediana_dias": float(dias.median()),
                 "max_dias": int(dias.max()), "criticos": int((res["prioridade"]=="Crítico").sum()),
                 "urgentes": int((res["prioridade"]=="Urgente").sum())}
        return res, dist, stats

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### 📄 Laudos Pendentes")
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            la, dist, stats = aging_analysis(df_pend_laudos)
            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Total", format_number(stats.get("total", len(df_pend_laudos))))
            with m2: st.metric("Críticos", stats.get("criticos", 0))
            with m3: st.metric("Média (dias)", format_number(stats.get("media_dias", 0),1))
            if not dist.empty:
                bar = px.bar(x=dist.index, y=dist.values, title="Distribuição por Tempo de Pendência",
                             color=dist.values, color_continuous_scale="Reds", text=dist.values)
                bar.update_traces(texttemplate='%{text}', textposition='outside')
                bar.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(bar, use_container_width=True)
            if "prioridade" in la.columns:
                pr = la["prioridade"].value_counts()
                pie = px.pie(values=pr.values, names=pr.index, title="Distribuição por Prioridade",
                             color_discrete_map={"Normal":"green","Atenção":"yellow","Urgente":"orange","Crítico":"red"})
                pie.update_layout(height=300); st.plotly_chart(pie, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in la.columns:
                cols=[c for c in ["id","unidade","tipo","dias_pendentes","prioridade"] if c in la.columns]
                st.dataframe(la.nlargest(10,"dias_pendentes")[cols] if cols else la.nlargest(10,"dias_pendentes"), use_container_width=True, height=250)
        else:
            st.info("Sem dados de laudos pendentes.")

    with c2:
        st.markdown("#### 🔬 Exames Pendentes")
        if df_pend_exames is not None and not df_pend_exames.empty:
            ex, dist, stats = aging_analysis(df_pend_exames)
            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Total", format_number(stats.get("total", len(df_pend_exames))))
            with m2: st.metric("Críticos", stats.get("criticos", 0))
            with m3: st.metric("Média (dias)", format_number(stats.get("media_dias", 0),1))
            if not dist.empty:
                bar = px.bar(x=dist.index, y=dist.values, title="Distribuição por Tempo de Pendência",
                             color=dist.values, color_continuous_scale="Oranges", text=dist.values)
                bar.update_traces(texttemplate='%{text}', textposition='outside')
                bar.update_layout(height=350, showlegend=False, xaxis_title="Faixa de Dias", yaxis_title="Quantidade")
                st.plotly_chart(bar, use_container_width=True)
            if "prioridade" in ex.columns:
                pr = ex["prioridade"].value_counts()
                pie = px.pie(values=pr.values, names=pr.index, title="Distribuição por Prioridade",
                             color_discrete_map={"Normal":"green","Atenção":"yellow","Urgente":"orange","Crítico":"red"})
                pie.update_layout(height=300); st.plotly_chart(pie, use_container_width=True)
            st.markdown("**🔴 Top 10 Mais Antigas:**")
            if "dias_pendentes" in ex.columns:
                cols=[c for c in ["id","unidade","tipo","dias_pendentes","prioridade"] if c in ex.columns]
                st.dataframe(ex.nlargest(10,"dias_pendentes")[cols] if cols else ex.nlargest(10,"dias_pendentes"), use_container_width=True, height=250)
        else:
            st.info("Sem dados de exames pendentes.")

# -------- DADOS --------
with tab5:
    st.subheader("📋 Exploração dos Dados")
    data_summary=[]
    for name, df in standardized.items():
        if df is not None and not df.empty:
            periodo="Sem dados temporais"
            if 'anomês' in df.columns and not df['anomês'].isna().all():
                periodo=f"{df['anomês'].min()} a {df['anomês'].max()}"
            data_summary.append({
                "Dataset": name.replace("_"," ").title(),
                "Registros": f"{len(df):,}".replace(",","."),
                "Colunas": len(df.columns),
                "Período": periodo,
                "Tamanho (MB)": round(df.memory_usage(deep=True).sum()/1024/1024,2),
                "Status": "✅ Carregado"
            })
    if data_summary:
        st.dataframe(pd.DataFrame(data_summary), use_container_width=True)

    st.markdown("#### 🔍 Visualização")
    options=[k for k,v in standardized.items() if v is not None]
    if options:
        sel = st.selectbox("Escolha o dataset", options, format_func=lambda x: x.replace("_"," ").title())
        df = standardized[sel]
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Registros", f"{len(df):,}".replace(",","."))
        with c2: st.metric("Colunas", len(df.columns))
        with c3: st.metric("Valores Nulos", f"{df.isnull().sum().sum():,}".replace(",","."))
        with c4:
            m = df['anomês_dt'].nunique() if 'anomês_dt' in df.columns else 0
            st.metric("Meses Únicos", m)
        cols = st.multiselect("Colunas", list(df.columns), default=list(df.columns)[:10])
        n = st.number_input("Máx. linhas", min_value=10, max_value=10000, value=500, step=50)
        show = df[cols].head(n) if cols else df.head(n)
        st.dataframe(show, use_container_width=True, height=400)
        st.download_button("📥 Download (CSV)", data=show.to_csv(index=False).encode("utf-8"),
                           file_name=f"{sel}_amostra_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# -------- RELATÓRIOS --------
with tab6:
    st.subheader("📑 Relatórios Executivos")
    tipo = st.selectbox("Tipo", ["Relatório Executivo Completo","Relatório de Produção","Relatório de Pendências","Relatório de Performance","Relatório Comparativo"])

    def relatorio_exec():
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        txt = f"""
# RELATÓRIO EXECUTIVO PCI/SC
**Data:** {ts}
**Período:** {f_periodo}

## 📊 RESUMO
- **Atendimentos:** {format_number(tot_at)}
- **Laudos:** {format_number(tot_ld)}
- **Conversão:** {format_number(tx_conv,1) if tx_conv else 'N/A'}%
- **Prod. Mensal (Laudos):** {format_number(med_ld,1) if med_ld else 'N/A'}

## ⏰ PENDÊNCIAS
- **Laudos pendentes:** {format_number(len(df_pend_laudos) if df_pend_laudos is not None else 0)}
- **Exames pendentes:** {format_number(len(df_pend_exames) if df_pend_exames is not None else 0)}
- **Backlog:** {format_number(backlog,1) if backlog else 'N/A'} meses
- **Aging médio:** {format_number(aging_laudos or aging_exames,0) if (aging_laudos or aging_exames) else 'N/A'} dias

## 🎯 PERFORMANCE
- **TME Mediano:** {format_number(tme_med,1) if tme_med else 'N/A'} dias
- **SLA 30:** {format_number(sla30,1) if sla30 else 'N/A'}%
- **SLA 60:** {format_number(sla60,1) if sla60 else 'N/A'}%

## 🚨 ALERTAS
{os.linesep.join(alerts) if alerts else 'Nenhum alerta.'}

## 📋 DATASETS
""" 
        for k, df in standardized.items():
            if df is not None and not df.empty:
                txt += f"- **{k.replace('_',' ').title()}**: {len(df):,} registros\n"
        return txt.strip()

    if tipo == "Relatório Executivo Completo":
        txt = relatorio_exec()
        st.markdown(txt)
        st.download_button("📥 Download Relatório (MD)", data=txt.encode("utf-8"),
                           file_name=f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
    else:
        st.info(f"Modelo '{tipo}' em desenvolvimento (posso habilitar já no próximo commit).")

# -------- DIÁRIO --------
with tab7:
    st.subheader("📅 Análise Diária – Atendimentos e Laudos")

    def daily(df: Optional[pd.DataFrame], label: str):
        if df is None or df.empty or "dia" not in df.columns: return pd.DataFrame(columns=["dia", label])
        t = df.dropna(subset=["dia"]).groupby("dia", as_index=False)["quantidade"].sum().rename(columns={"quantidade": label})
        return t.sort_values("dia")

    A = daily(df_atend_diario, "Atendimentos")
    L = daily(df_laudos_diario, "Laudos")
    if A.empty and L.empty:
        st.info("Envie **Atendimentos (Diário)** e/ou **Laudos (Diário)**.")
    else:
        D = pd.merge(A, L, on="dia", how="outer").fillna(0).sort_values("dia")
        for col in ["Atendimentos","Laudos"]:
            D[col] = pd.to_numeric(D[col], errors="coerce").fillna(0)
        D["MA7_Atend"] = D["Atendimentos"].rolling(7).mean()
        D["MA7_Laudos"] = D["Laudos"].rolling(7).mean()
        D["Taxa_%"] = np.where(D["Atendimentos"]>0, (D["Laudos"]/D["Atendimentos"])*100, np.nan)
        D["MA7_Taxa_%"] = D["Taxa_%"].rolling(7).mean()

        ult = D.iloc[-1] if not D.empty else None
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Último dia", D["dia"].max().strftime("%d/%m/%Y") if not D.empty else "—")
        with c2: st.metric("Atendimentos (últ. dia)", format_number(ult["Atendimentos"]) if ult is not None else "—")
        with c3: st.metric("Laudos (últ. dia)", format_number(ult["Laudos"]) if ult is not None else "—")
        with c4: 
            taxa = ult["Taxa_%"] if (ult is not None and not pd.isna(ult["Taxa_%"])) else None
            st.metric("Conversão (últ. dia)", f"{taxa:.1f}%" if taxa is not None else "—")

        st.markdown("#### 📈 Evolução Diária")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D["dia"], y=D["Atendimentos"], mode="lines", name="Atendimentos"))
        fig.add_trace(go.Scatter(x=D["dia"], y=D["Laudos"], mode="lines", name="Laudos"))
        if D["MA7_Atend"].notna().any(): fig.add_trace(go.Scatter(x=D["dia"], y=D["MA7_Atend"], mode="lines", name="Atend MM7", line=dict(dash="dash")))
        if D["MA7_Laudos"].notna().any(): fig.add_trace(go.Scatter(x=D["dia"], y=D["MA7_Laudos"], mode="lines", name="Laudos MM7", line=dict(dash="dash")))
        fig.update_layout(height=420, hovermode="x unified", xaxis_title="Dia", yaxis_title="Quantidade")
        st.plotly_chart(fig, use_container_width=True)

        if D["Taxa_%"].notna().any():
            st.markdown("#### 🎯 Taxa de Conversão Diária (%)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=D["dia"], y=D["Taxa_%"], mode="lines", name="Taxa"))
            if D["MA7_Taxa_%"].notna().any():
                fig2.add_trace(go.Scatter(x=D["dia"], y=D["MA7_Taxa_%"], mode="lines", name="MM7", line=dict(dash="dash")))
            fig2.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Meta 70%")
            fig2.update_layout(height=320, hovermode="x unified", xaxis_title="Dia", yaxis_title="%")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📋 Tabela Diária")
        tab = D.copy()
        tab["dia"] = tab["dia"].dt.strftime("%d/%m/%Y")
        cols = ["dia","Atendimentos","Laudos","Taxa_%","MA7_Atend","MA7_Laudos","MA7_Taxa_%"]
        st.dataframe(tab[cols].tail(180), use_container_width=True, height=420)
        st.download_button("📥 Baixar Tabela (CSV)", data=D.to_csv(index=False).encode("utf-8"),
                           file_name=f"diario_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# -------- VALIDAÇÃO --------
with tab8:
    st.subheader("🧪 Validação de Ingestão")
    for name, df in raw.items():
        st.markdown(f"### {name.replace('_',' ').title()}")
        if df is None or df.empty:
            st.info("Sem dados."); st.divider(); continue
        qcol = infer_quantity_col(df)
        comp_guess = pick_col(df, COLUMN_MAPPINGS.get(name,{}).get("competencia"), COMPETENCIA_CANDIDATES)
        date_guess = pick_col(df, COLUMN_MAPPINGS.get(name,{}).get("date"), DATE_CANDIDATES)
        tipo_guess = pick_col(df, COLUMN_MAPPINGS.get(name,{}).get("tipo"), TIPO_CANDIDATES)
        unidade_guess = pick_col(df, COLUMN_MAPPINGS.get(name,{}).get("unidade"), UNIDADE_CANDIDATES)
        id_guess = pick_col(df, COLUMN_MAPPINGS.get(name,{}).get("id"), ID_CANDIDATES)
        st.write({
            "linhas": len(df), "colunas": len(df.columns),
            "quantidade_col": qcol, "competencia_col": comp_guess, "data_col": date_guess,
            "tipo_col": tipo_guess, "unidade_col": unidade_guess, "id_col": id_guess
        })
        st.caption("Colunas: " + ", ".join(df.columns[:30]) + (" ..." if len(df.columns)>30 else ""))
        st.divider()

# =================== RODAPÉ ===================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
  <p><strong>Dashboard PCI/SC v2.0</strong> - Sistema Avançado de Monitoramento</p>
  <p>📊 Produção • ⏰ Pendências • 📈 Performance • 📋 Gestão</p>
  <p><em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em></p>
</div>
""", unsafe_allow_html=True)
