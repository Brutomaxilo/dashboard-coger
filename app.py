import io, os, re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="PCI/SC – Dashboard", layout="wide")
st.title("Dashboard PCI/SC – Produção & Pendências")

# ============ UTIL ============
def read_csv_any(path_or_file):
    if path_or_file is None: return None
    seps = [",",";","\t","|"]
    if isinstance(path_or_file, str):
        for sep in seps:
            try:
                df = pd.read_csv(path_or_file, sep=sep, engine="python")
                if df.shape[1] > 1: return df
            except: pass
        try:
            return pd.read_csv(path_or_file, sep=None, engine="python")
        except: return None
    else:
        bio = io.BytesIO(path_or_file.read())
        for sep in seps:
            try:
                bio.seek(0)
                df = pd.read_csv(bio, sep=sep, engine="python")
                if df.shape[1] > 1: return df
            except: pass
        try:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python")
        except: return None

def to_month(dt_series):
    if dt_series is None: return None
    d = pd.to_datetime(dt_series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if d.notna().any():
        return d.dt.to_period("M").dt.to_timestamp()
    return None

def fmt_int(x):
    try: return f"{int(round(x)):,}".replace(",", ".")
    except: return "0"

def pct(a,b):
    return (a/b) if b and b!=0 else None

# ============ DETECÇÃO DE FONTE ============
has_data_dir = os.path.exists("data") and any(p.endswith(".csv") for p in os.listdir("data"))

st.sidebar.header("Arquivos CSV")
if not has_data_dir:
    st.sidebar.caption("Envie os arquivos (pode enviar só os que tiver; o app se adapta).")

uploads = {
    "Atendimentos_todos_Mensal": None if has_data_dir else st.sidebar.file_uploader("Atendimentos_todos_Mensal (.csv)"),
    "Laudos_todos_Mensal": None if has_data_dir else st.sidebar.file_uploader("Laudos_todos_Mensal (.csv)"),
    "Atendimentos_especifico_Mensal": None if has_data_dir else st.sidebar.file_uploader("Atendimentos_especifico_Mensal (.csv)"),
    "Laudos_especifico_Mensal": None if has_data_dir else st.sidebar.file_uploader("Laudos_especifico_Mensal (.csv)"),
    "laudos_realizados": None if has_data_dir else st.sidebar.file_uploader("laudos_realizados (.csv)"),
    "detalhes_laudospendentes": None if has_data_dir else st.sidebar.file_uploader("detalhes_laudospendentes (.csv)"),
    "detalhes_examespendentes": None if has_data_dir else st.sidebar.file_uploader("detalhes_examespendentes (.csv)"),
}

def resolve_path(name):
    # aceita variações: espaço/underscore, maiúsculas e sufixos como " (6).csv"
    wanted = name.lower().replace(" ", "_")
    for fname in os.listdir("data"):
        if not fname.lower().endswith(".csv"):
            continue
        base = os.path.splitext(fname)[0].lower()
        base_norm = base.replace(" ", "_")
        # bate se começar com o nome esperado
        if base_norm.startswith(wanted):
            return os.path.join("data", fname)
    return None

dfs_raw = {}
for name, up in uploads.items():
    if has_data_dir:
        p = resolve_path(name)
        if p: dfs_raw[name] = read_csv_any(p)
    else:
        if up: dfs_raw[name] = read_csv_any(up)

if not dfs_raw:
    st.warning("Nenhum arquivo carregado. Suba pela barra lateral ou coloque os CSVs em `data/`.")
    st.stop()

# ============ MAPEAMENTO FIXO POR ARQUIVO ============
# Nomes que você enviou (copiados das prints):
MAPS = {
    # Pendências
    "detalhes_laudospendentes": {
        "date":"data_solicitacao", "ano":"ano_sol", "id":"caso_sirsaelp",
        "unidade":"unidade", "superintendencia":"superintendencia", "diretoria":"diretoria",
        "competencia":"competencia", "tipo":"tipopericia", "perito":"perito"
    },
    "detalhes_examespendentes": {
        "date":"data_solicitacao", "ano":"ano_sol", "id":"caso_sirsaelp",
        "unidade":"unidade", "superintendencia":"superintendencia", "diretoria":"diretoria",
        "competencia":"competencia", "tipo":"tipopericia"
    },
    # Mensal – todos / específico
    "Atendimentos_todos_Mensal": {
        "date":"data_interesse", "id":"idatendimento"
    },
    "Atendimentos_especifico_Mensal": {
        "date":"data_interesse", "competencia":"txcompetencia", "id":"idatendimento"
    },
    "Laudos_todos_Mensal": {
        "date":"data_interesse", "id":"iddocumento"
    },
    "Laudos_especifico_Mensal": {
        "date":"data_interesse", "competencia":"txcompetencia", "id":"iddocumento"
    },
    # Realizados
    "laudos_realizados": {
        "solicitacao":"dhsolicitacao", "atendimento":"dhatendimento", "emissao":"dhemitido",
        "n_laudo":"n_laudo", "ano":"ano_emissao", "mes":"mes_emissao",
        "unidade":"unidade_emissao", "diretoria":"diretoria",
        "competencia":"txcompetencia", "tipo":"txtipopericia", "perito":"perito"
    },
}

def standardize(name, df):
    """Padroniza para colunas: anomês_dt, anomês, quantidade, diretoria, superintendencia, unidade, tipo, id, perito, data_base"""
    m = MAPS.get(name, {})
    out = df.copy()
    # quant
    out["quantidade"] = 1

    # dimensões diretas (se existirem)
    for std, src in [("diretoria","diretoria"), ("superintendencia","superintendencia"), ("unidade","unidade"),
                     ("tipo","tipo"), ("perito","perito"), ("id","id")]:
        if src in m and m[src] in out.columns:
            out[std] = out[m[src]]

    # ANOMÊS: prioridade por competencia -> date
    anomes_dt = None
    if "competencia" in m and m["competencia"] in out.columns:
        anomes_dt = to_month(out[m["competencia"]])
    if anomes_dt is None and "date" in m and m["date"] in out.columns:
        anomes_dt = to_month(out[m["date"]])
    # ano/mes (apenas em laudos_realizados)
    if anomes_dt is None and name == "laudos_realizados":
        if ("ano" in m and m["ano"] in out.columns) and ("mes" in m and m["mes"] in out.columns):
            y = pd.to_numeric(out[m["ano"]], errors="coerce")
            mo = pd.to_numeric(out[m["mes"]], errors="coerce")
            tmp = pd.to_datetime(dict(year=y, month=mo, day=1), errors="coerce")
            if tmp.notna().any():
                anomes_dt = tmp.dt.to_period("M").dt.to_timestamp()

    if anomes_dt is not None:
        out["anomês_dt"] = anomes_dt
        out["anomês"] = out["anomês_dt"].dt.strftime("%Y-%m")

    # datas base brutas úteis
    if "date" in m and m["date"] in out.columns:
        out["data_base"] = pd.to_datetime(out[m["date"]], errors="coerce", dayfirst=True, infer_datetime_format=True)

    # métricas especiais (laudos_realizados)
    if name == "laudos_realizados":
        for k,std in [("solicitacao","dhsolicitacao"),("atendimento","dhatendimento"),("emissao","dhemitido")]:
            if k in m and m[k] in out.columns:
                out[std] = pd.to_datetime(out[m[k]], errors="coerce", dayfirst=True, infer_datetime_format=True)
        # TME: emissão - atendimento (fallback emissão - solicitação)
        if "dhemitido" in out.columns:
            base = out["dhatendimento"] if "dhatendimento" in out.columns else out.get("dhsolicitacao")
            out["tme_dias"] = (out["dhemitido"] - base).dt.days
            out["sla_30_ok"] = out["tme_dias"] <= 30

    # limpeza básica
    for c in ["diretoria","superintendencia","unidade","tipo","id","perito","anomês"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out

dfs = {}
for name, df in dfs_raw.items():
    if df is None: continue
    # normaliza nomes originais (minúsculo, sem espaços duplicados)
    df.columns = [re.sub(r"\s+"," ", c.strip().lower()) for c in df.columns]
    dfs[name] = standardize(name, df)

# ============ FILTROS ============
def collect_values(col):
    vals = set()
    for df in dfs.values():
        if col in df.columns:
            vals |= set(df[col].dropna().astype(str))
    vals = [v for v in vals if v and v.lower()!="nan"]
    return sorted(vals)

st.sidebar.subheader("Filtros")
f_diretoria = st.sidebar.multiselect("Diretoria", collect_values("diretoria"))
f_sr        = st.sidebar.multiselect("Superintendência", collect_values("superintendencia"))
f_unid      = st.sidebar.multiselect("Unidade / Núcleo", collect_values("unidade"))
f_tipo      = st.sidebar.multiselect("Tipo", collect_values("tipo"))
f_comp      = st.sidebar.multiselect("Competência (Ano-Mês)", collect_values("anomês"))

def apply_filters(df):
    if df is None: return None
    m = pd.Series(True, index=df.index)
    def fcol(col, flt):
        nonlocal m
        if col in df.columns and flt: m &= df[col].astype(str).isin(flt)
    fcol("diretoria", f_diretoria)
    fcol("superintendencia", f_sr)
    fcol("unidade", f_unid)
    fcol("tipo", f_tipo)
    fcol("anomês", f_comp)
    return df[m].copy()

# bases filtradas
at_todos = apply_filters(dfs.get("Atendimentos_todos_Mensal"))
la_todos = apply_filters(dfs.get("Laudos_todos_Mensal"))
at_esp   = apply_filters(dfs.get("Atendimentos_especifico_Mensal"))
la_esp   = apply_filters(dfs.get("Laudos_especifico_Mensal"))
la_real  = apply_filters(dfs.get("laudos_realizados"))
pend_l   = apply_filters(dfs.get("detalhes_laudospendentes"))
pend_e   = apply_filters(dfs.get("detalhes_examespendentes"))

# ============ KPIs ============
def soma_q(df): 
    return int(df["quantidade"].sum()) if df is not None and "quantidade" in df.columns else 0

tot_at = soma_q(at_todos)
tot_la = soma_q(la_todos)
tot_pendl = len(pend_l) if pend_l is not None else 0
tot_pende = len(pend_e) if pend_e is not None else 0

# média mensal de laudos (para backlog)
def media_mensal(df):
    if df is None or "anomês_dt" not in df.columns: return None
    g = df.groupby("anomês_dt")["quantidade"].sum()
    return g.mean() if len(g)>0 else None

med_la = media_mensal(la_todos)
backlog = (tot_pendl / med_la) if med_la and med_la>0 else None
taxa_at = pct(tot_la, tot_at)

# TME & SLA 30 (laudos_realizados)
tme_med = None
sla30 = None
if la_real is not None and "tme_dias" in la_real.columns:
    tme_med = float(pd.to_numeric(la_real["tme_dias"], errors="coerce").dropna().median()) if la_real["tme_dias"].notna().any() else None
    if "sla_30_ok" in la_real.columns and la_real["sla_30_ok"].notna().any():
        sla30 = la_real["sla_30_ok"].mean()

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("Atendimentos", fmt_int(tot_at))
k2.metric("Laudos", fmt_int(tot_la))
k3.metric("Laudos pendentes", fmt_int(tot_pendl))
k4.metric("Exames pendentes", fmt_int(tot_pende))
k5.metric("Backlog (meses)", f"{backlog:.1f}" if backlog is not None else "—")
k6.metric("Taxa de atendimento", f"{taxa_at*100:.1f}%" if taxa_at is not None else "—")
k7.metric("TME (dias)", f"{tme_med:.1f}" if tme_med is not None else "—")
k8.metric("SLA 30d", f"{sla30*100:.1f}%" if sla30 is not None else "—")

st.divider()

# ============ ABAS ============
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visão Geral","Tendências","Rankings","Pendências","Dados"])

# Visão Geral
with tab1:
    st.subheader("Resumo")
    if la_todos is not None and "unidade" in la_todos.columns:
        g = la_todos.groupby("unidade", as_index=False)["quantidade"].sum().sort_values("quantidade", ascending=False).head(10)
        st.plotly_chart(px.bar(g, x="quantidade", y="unidade", orientation="h", title="Top 10 Unidades – Laudos"), use_container_width=True)
    if la_todos is not None and "tipo" in la_todos.columns:
        g = la_todos.groupby("tipo", as_index=False)["quantidade"].sum().sort_values("quantidade", ascending=False).head(12)
        st.plotly_chart(px.bar(g, x="tipo", y="quantidade", title="Tipos de Laudo – Top 12"), use_container_width=True)

# Tendências
def line_series(df, title):
    if df is None or "anomês_dt" not in df.columns: 
        st.info(f"Sem dados suficientes para {title}."); return
    g = df.groupby("anomês_dt", as_index=False)["quantidade"].sum().sort_values("anomês_dt")
    g["anomês"] = g["anomês_dt"].dt.strftime("%Y-%m")
    fig = px.line(g, x="anomês", y="quantidade", markers=True, title=title)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Séries temporais")
    c1,c2 = st.columns(2)
    with c1: line_series(at_todos, "Atendimentos – Todos (mensal)")
    with c2: line_series(la_todos, "Laudos – Todos (mensal)")
    c3,c4 = st.columns(2)
    with c3: line_series(at_esp, "Atendimentos – Específico (mensal)")
    with c4: line_series(la_esp, "Laudos – Específico (mensal)")

# Rankings
def bar_dim(df, dim, title, topn=25):
    if df is None or dim not in df.columns:
        st.info(f"Sem {dim} para {title}."); return
    g = df.groupby(dim, as_index=False)["quantidade"].sum().sort_values("quantidade", ascending=False).head(topn)
    fig = px.bar(g, x="quantidade", y=dim, orientation="h", title=title)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=520)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Comparativos")
    c1,c2 = st.columns(2)
    with c1: bar_dim(at_todos, "diretoria", "Atendimentos por Diretoria")
    with c2: bar_dim(la_todos, "diretoria", "Laudos por Diretoria")
    c3,c4 = st.columns(2)
    with c3: bar_dim(la_todos, "unidade", "Laudos por Unidade")
    with c4: bar_dim(at_todos, "unidade", "Atendimentos por Unidade")
    bar_dim(la_todos, "tipo", "Laudos por Tipo")

# Pendências (inclui aging se houver data_solicitacao)
def add_aging(df, colname="data_base"):
    if df is None or colname not in df.columns: return df, None
    d = pd.to_datetime(df[colname], errors="coerce", dayfirst=True, infer_datetime_format=True)
    dias = (pd.Timestamp("now").normalize() - d).dt.days
    buckets = pd.cut(dias, bins=[-1,30,60,90,180,10**5],
                     labels=["0–30","31–60","61–90","91–180","180+"])
    out = df.copy()
    out["dias_pendentes"] = dias
    out["faixa_dias"] = buckets
    dist = out["faixa_dias"].value_counts(dropna=False).sort_index()
    return out, dist

with tab4:
    st.subheader("Pendências")
    c1,c2 = st.columns(2)
    with c1:
        if pend_l is not None:
            p2, dist = add_aging(pend_l, "data_base" if "data_base" in pend_l.columns else "data_solicitacao")
            if dist is not None: st.bar_chart(dist)
            st.dataframe(p2.head(500), use_container_width=True, height=320)
        else:
            st.info("Sem base de laudos pendentes.")
    with c2:
        if pend_e is not None:
            p3, dist2 = add_aging(pend_e, "data_base" if "data_base" in pend_e.columns else "data_solicitacao")
            if dist2 is not None: st.bar_chart(dist2)
            st.dataframe(p3.head(500), use_container_width=True, height=320)
        else:
            st.info("Sem base de exames pendentes.")

# Dados
with tab5:
    st.subheader("Dados brutos")
    for name, df in dfs.items():
        st.markdown(f"**{name}** — {df.shape[0]} linhas / {df.shape[1]} colunas")
        st.dataframe(df.head(50), use_container_width=True, height=250)
