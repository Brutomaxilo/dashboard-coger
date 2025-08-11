import io
import re
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="PCI/SC – Dashboard Rápido", layout="wide")

st.title("Dashboard Rápido – Produção, Pendências e Séries Temporais")

# =========================
# AJUDA / ESQUEMA DE DADOS
# =========================
with st.expander("Ajuda → Esquema de dados esperado (clique para abrir)"):
    st.markdown("""
**Arquivos aceitos** (qualquer subseto):  
- `Atendimentos_todos_Mensal`  
- `Laudos_todos_Mensal`  
- `Atendimentos_especifico_Mensal`  
- `Laudos_especifico_Mensal`  
- `laudos_realizados`  
- `detalhes_laudospendentes`  
- `detalhes_examespendentes`  

**Colunas recomendadas** (o app tenta detectar automaticamente):  
- Dimensões: `Diretoria`, `Superintendência` (ou `SR`), `Unidade` (ou `Núcleo`), `Tipo`  
- Tempo: `Competência`/`AnoMes` **ou** `Data` (o app cria `anomês` = `YYYY-MM`)  
- Métrica: `Quantidade` (se não existir, assume 1 por linha)

**Dica**: Se precisar padronizar, renomeie no CSV ou ajuste os *aliases* no código (seção "PADRONIZAÇÃO").
""")

# =========================
# FUNÇÕES DE I/O
# =========================
def read_csv_any(file_or_path):
    """Lê CSV detectando separador comum; aceita caminho (str) ou UploadedFile."""
    if file_or_path is None:
        return None
    try_list = [",",";","\t","|"]
    bytes_data = None
    if not isinstance(file_or_path, str):
        bytes_data = io.BytesIO(file_or_path.read())
    for sep in try_list:
        try:
            if isinstance(file_or_path, str):
                df = pd.read_csv(file_or_path, sep=sep, engine="python")
            else:
                bytes_data.seek(0)
                df = pd.read_csv(bytes_data, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # último recurso
    try:
        if isinstance(file_or_path, str):
            return pd.read_csv(file_or_path, sep=None, engine="python")
        else:
            bytes_data.seek(0)
            return pd.read_csv(bytes_data, sep=None, engine="python")
    except Exception:
        return None

# =========================
# ENTRADA DE DADOS
# =========================
st.sidebar.header("Arquivos CSV")
st.sidebar.caption("Envie os arquivos (pode enviar só os que tiver; o app se adapta).")
uploads = {
    "Atendimentos_todos_Mensal": st.sidebar.file_uploader("Atendimentos_todos_Mensal (.csv)"),
    "Laudos_todos_Mensal": st.sidebar.file_uploader("Laudos_todos_Mensal (.csv)"),
    "Atendimentos_especifico_Mensal": st.sidebar.file_uploader("Atendimentos_especifico_Mensal (.csv)"),
    "Laudos_especifico_Mensal": st.sidebar.file_uploader("Laudos_especifico_Mensal (.csv)"),
    "laudos_realizados": st.sidebar.file_uploader("laudos_realizados (.csv)"),
    "detalhes_laudospendentes": st.sidebar.file_uploader("detalhes_laudospendentes (.csv)"),
    "detalhes_examespendentes": st.sidebar.file_uploader("detalhes_examespendentes (.csv)"),
}

# Leitura: (1) uploads; (2) fallback para pasta local data/
dfs = {}
for name, file in uploads.items():
    df = read_csv_any(file) if file else None
    if df is None:
        # tenta data/<name>.csv com variações simples
        candidates = [
            f"data/{name}.csv",
            f"data/{name}.CSV",
            f"data/{name.replace(' ', '_')}.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                df = read_csv_any(c)
                if df is not None:
                    break
    if df is not None:
        dfs[name] = df

if not dfs:
    st.warning("Nenhum arquivo carregado ainda. Faça upload pela barra lateral ou coloque seus CSVs em `data/`.")
    st.stop()

# =========================
# PADRONIZAÇÃO
# =========================
def normalize_cols(df):
    df = df.copy()
    # normaliza nomes
    df.columns = [re.sub(r"\s+"," ", c.strip()).lower() for c in df.columns]
    # aliases para mapear colunas comuns
    aliases = {
        "diretoria": ["diretoria","dir"],
        "superintendencia": ["superintendência","superintendencia","sr","superint"],
        "unidade": ["unidade","nucleo","núcleo","unidade/regional","núcleo regional","nucleo regional"],
        "tipo": ["tipo","tipologia","classe","natureza"],
        "quantidade": ["qtd","quantidade","count","total","qtde"],
        "anomês": ["anomes","ano_mes","competencia","competência","mes_ano","mês/ano","ano-mes"],
        "data": ["data","dt","data_atendimento","data_laudo","data_emissao","emissao"]
    }
    # mapeia: primeira correspondência ganha
    used_targets = set()
    for std, cands in aliases.items():
        for c in cands:
            if c in df.columns and std not in used_targets:
                df.rename(columns={c: std}, inplace=True)
                used_targets.add(std)
                break

    # cria anomês a partir de data se necessário
    if "anomês" not in df.columns:
        if "data" in df.columns:
            d = pd.to_datetime(df["data"], errors="coerce", dayfirst=True, infer_datetime_format=True)
            df["anomês"] = d.dt.strftime("%Y-%m")
        else:
            # tenta qualquer coluna que pareça competência
            for c in df.columns:
                if re.search(r"(comp|mes|mês|ano)", c):
                    tmp = df[c].astype(str)
                    parsed = pd.to_datetime(tmp, errors="coerce", dayfirst=True, infer_datetime_format=True)
                    if parsed.notna().mean() > 0.3:
                        df["anomês"] = parsed.dt.strftime("%Y-%m")
                        break

    # garante quantidade
    if "quantidade" not in df.columns:
        df["quantidade"] = 1

    # limpa espaços / padroniza strings
    for c in ["diretoria","superintendencia","unidade","tipo","anomês"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

for k in list(dfs.keys()):
    dfs[k] = normalize_cols(dfs[k])

# =========================
# PREPARAÇÃO E FILTROS
# =========================
def unique_values(cols):
    vals = {}
    for c in ["diretoria","superintendencia","unidade","tipo","anomês"]:
        acc = set()
        for df in dfs.values():
            if df is not None and c in df.columns:
                acc |= set(df[c].dropna().astype(str).unique())
        if acc:
            vals[c] = sorted([v for v in acc if v and v.lower()!="nan"])
    return vals

vals = unique_values(dfs)

st.sidebar.subheader("Filtros")
f_diretoria = st.sidebar.multiselect("Diretoria", vals.get("diretoria", []))
f_sr        = st.sidebar.multiselect("Superintendência", vals.get("superintendencia", []))
f_unid      = st.sidebar.multiselect("Unidade / Núcleo", vals.get("unidade", []))
f_tipo      = st.sidebar.multiselect("Tipo", vals.get("tipo", []))
f_comp      = st.sidebar.multiselect("Competência (Ano-Mês)", vals.get("anomês", []))

def apply_filters(df):
    if df is None: return df
    m = pd.Series([True]*len(df), index=df.index)
    def fcol(col, flt):
        nonlocal m
        if col in df.columns and flt:
            m &= df[col].astype(str).isin(flt)
    fcol("diretoria", f_diretoria)
    fcol("superintendencia", f_sr)
    fcol("unidade", f_unid)
    fcol("tipo", f_tipo)
    fcol("anomês", f_comp)
    return df[m].copy()

# bases principais
at_todos = apply_filters(dfs.get("Atendimentos_todos_Mensal"))
la_todos = apply_filters(dfs.get("Laudos_todos_Mensal"))
at_esp  = apply_filters(dfs.get("Atendimentos_especifico_Mensal"))
la_esp  = apply_filters(dfs.get("Laudos_especifico_Mensal"))
la_real = apply_filters(dfs.get("laudos_realizados"))
pend_l  = apply_filters(dfs.get("detalhes_laudospendentes"))
pend_e  = apply_filters(dfs.get("detalhes_examespendentes"))

# =========================
# KPIs
# =========================
def k(x):
    try:
        return int(x) if pd.notna(x) else 0
    except Exception:
        return 0

tot_at = k(at_todos["quantidade"].sum()) if at_todos is not None else 0
tot_la = k(la_todos["quantidade"].sum()) if la_todos is not None else 0
tot_pendl = int(pend_l.shape[0]) if pend_l is not None else 0
tot_pende = int(pend_e.shape[0]) if pend_e is not None else 0

# cálculo de backlog e taxa de atendimento (se possível)
backlog = None
taxa_atendimento = None
if la_todos is not None and "anomês" in la_todos.columns:
    med_mensal_la = la_todos.groupby("anomês")["quantidade"].sum().mean()
    if (pend_l is not None):
        backlog = tot_pendl / med_mensal_la if med_mensal_la and med_mensal_la > 0 else None
if at_todos is not None and la_todos is not None:
    soma_at = at_todos["quantidade"].sum()
    soma_la = la_todos["quantidade"].sum()
    taxa_atendimento = (soma_la / soma_at) if soma_at > 0 else None

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Atendimentos (período filtrado)", f"{tot_at:,}".replace(",", "."))
col2.metric("Laudos (período filtrado)", f"{tot_la:,}".replace(",", "."))
col3.metric("Laudos pendentes", f"{tot_pendl:,}".replace(",", "."))
col4.metric("Exames pendentes", f"{tot_pende:,}".replace(",", "."))
col5.metric("Backlog (meses)", f"{backlog:.1f}" if backlog is not None else "—")
col6.metric("Taxa de atendimento", f"{taxa_atendimento*100:.1f}%" if taxa_atendimento is not None else "—")

st.divider()

# =========================
# VISÕES
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Séries Temporais", "Comparativos", "Pendências", "Dados Brutos"])

# --- Séries temporais
with tab1:
    st.subheader("Séries temporais por competência (Ano-Mês)")
    def plot_series(df, label):
        if df is None or "anomês" not in df.columns or "quantidade" not in df.columns:
            st.info(f"Sem dados suficientes para {label}.")
            return
        g = df.groupby("anomês", as_index=False)["quantidade"].sum().sort_values("anomês")
        fig = px.line(g, x="anomês", y="quantidade", markers=True, title=label)
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        plot_series(at_todos, "Atendimentos – Todos (mensal)")
    with c2:
        plot_series(la_todos, "Laudos – Todos (mensal)")

    c3, c4 = st.columns(2)
    with c3:
        plot_series(at_esp, "Atendimentos – Específico (mensal)")
    with c4:
        plot_series(la_esp, "Laudos – Específico (mensal)")

# --- Comparativos
with tab2:
    st.subheader("Comparativos por Diretoria / Unidade / Tipo")
    def bar_dim(df, dim, title):
        if df is None or dim not in df.columns:
            st.info(f"Sem {dim} para {title}.")
            return
        g = df.groupby(dim, as_index=False)["quantidade"].sum().sort_values("quantidade", ascending=False).head(25)
        fig = px.bar(g, x="quantidade", y=dim, orientation="h", title=title)
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=500)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        bar_dim(at_todos, "diretoria", "Atendimentos por Diretoria")
    with c2:
        bar_dim(la_todos, "diretoria", "Laudos por Diretoria")

    c3, c4 = st.columns(2)
    with c3:
        bar_dim(la_todos, "unidade", "Laudos por Unidade")
    with c4:
        bar_dim(at_todos, "unidade", "Atendimentos por Unidade")

    bar_dim(la_todos, "tipo", "Laudos por Tipo")

# --- Pendências
with tab3:
    st.subheader("Pendências – Laudos e Exames")
    if pend_l is not None:
        st.markdown("**Laudos Pendentes**")
        st.dataframe(pend_l, use_container_width=True, height=320)
    else:
        st.info("Sem base de laudos pendentes.")
    if pend_e is not None:
        st.markdown("**Exames Pendentes**")
        st.dataframe(pend_e, use_container_width=True, height=320)
    else:
        st.info("Sem base de exames pendentes.")

# --- Dados brutos
with tab4:
    st.subheader("Dados brutos (primeiras linhas)")
    for name, df in dfs.items():
        st.markdown(f"**{name}**")
        st.dataframe(df.head(50), use_container_width=True, height=250)

st.caption("Obs.: O app detecta nomes de colunas (Diretoria, Superintendência, Unidade, Tipo, AnoMês/Data, Quantidade). Se o seu layout divergir, renomeie as colunas nos CSVs ou ajuste o dicionário de aliases na seção PADRONIZAÇÃO.")
