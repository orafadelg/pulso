# -*- coding: utf-8 -*-
"""
Atlas – Bluecompany (Glovis) • Streamlit Prototype v1 (Okiar)
Como rodar:
    streamlit run app.py
Observações:
- Dados 100% simulados (mock). Estrutura pronta p/ plugar dados reais depois.
- BUs e concorrentes baseados na proposta da Glovis (#9794).
- Abas: Overview, Market (Share & Sizing), Concorrência, Tendências, AI.
"""

import math
import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Plotly (auto-install se faltar)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.22.0"])
    import plotly.express as px
    import plotly.graph_objects as go

import streamlit as st

# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================
st.set_page_config(
    page_title="Atlas – Bluecompany (Glovis)",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta e helpers visuais
PRIMARY  = "#2E5BFF"
ACCENT   = "#00C2A8"
WARNING  = "#FFB020"
DANGER   = "#EB5757"
MUTED    = "#9AA0A6"
LIGHT_BG = "#F7F8FA"

CUSTOM_CSS = f"""
/* Cards e métricas */
.metric-card {{
  background: white;
  border: 1px solid #EAECF0;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(16,24,40,.05);
}}
.metric-label {{ color: {MUTED}; font-size: 12px; font-weight: 500; }}
.metric-value {{ font-size: 28px; font-weight: 700; }}
.metric-delta-up {{ color: {ACCENT}; font-weight: 600; }}
.metric-delta-down {{ color: {DANGER}; font-weight: 600; }}

/* Títulos e seções */
.block-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.subtle {{ color: {MUTED}; font-size: 12px; }}
.section {{ padding: 6px 0 2px 0; }}

/* Tabela compacta */
.small-table table {{ font-size: 12px; }}

/* Badges */
.badge {{
  display: inline-block; padding: 4px 8px; font-size: 11px;
  border-radius: 12px; background: {LIGHT_BG}; border: 1px solid #E5E7EB; color: #374151
}}
.badge-green {{ background: #ECFDF5; color: #047857; border-color: #A7F3D0; }}
.badge-yellow {{ background: #FFFBEB; color: #92400E; border-color: #FDE68A; }}
.badge-red {{ background: #FEF2F2; color: #991B1B; border-color: #FCA5A5; }}

/* Alertas */
.alert {{ border-left: 4px solid {ACCENT}; background: white; padding: 10px 12px; border-radius: 8px; border:1px solid #EAECF0; }}
.alert-critico {{ border-left-color: {DANGER}; }}
.alert-alto {{ border-left-color: {WARNING}; }}

/* Expander */
details > summary {{ cursor: pointer; }}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# =============================================================================
# PARÂMETROS DE SIMULAÇÃO
# =============================================================================
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

COMPANY = "Bluecompany"  # codinome p/ Glovis neste app
# Concorrentes relacionados
COMPETITORS = ["Tegma", "JSL", "CEVA", "Autoport", "Transauto"]
PLAYERS = [COMPANY] + COMPETITORS

# Business Units (BUs)
BUS = [
    "Milk Run",
    "Distribuição de Peças",
    "Gerenciamento de Pátios",
    "Transporte de Veículos",
    "PDI",
    "PDS",
]

# Regiões/UFs
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
UF = [
    "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MG","MS","MT",
    "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"
]

# Datas (12 meses)
HOJE = date.today()
MESES = [HOJE - relativedelta(months=i) for i in range(0, 12)][::-1]
TRIS = sorted({(d.year, (d.month - 1)//3 + 1) for d in MESES})

# =============================================================================
# GERADORES DE DADOS MOCK
# =============================================================================
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def gen_market_share_series(players: List[str], months: List[date], bus: List[str]) -> pd.DataFrame:
    """Séries mensais de market share por BU e player."""
    rows = []
    base_by_bu = {b: {p: random.uniform(5, 35) for p in players} for b in bus}
    for b in bus:
        base_by_bu[b][COMPANY] = random.uniform(18, 30)
    for m in months:
        for b in bus:
            noise = {p: np.clip(np.random.normal(0, 1.2), -3, 3) for p in players}
            total = sum(base_by_bu[b][p] + noise[p] for p in players)
            for p in players:
                share = max(0.01, (base_by_bu[b][p] + noise[p]) / max(total, 1e-9))
                rows.append({"mes": m, "bu": b, "player": p, "market_share": share})
    df = pd.DataFrame(rows)
    df["sum_mb"] = df.groupby(["mes", "bu"])["market_share"].transform("sum")
    df["market_share"] = df["market_share"] / df["sum_mb"]
    df.drop(columns=["sum_mb"], inplace=True)
    return df

def gen_market_sizing(bus: List[str], months: List[date]) -> pd.DataFrame:
    """Sizing mensal por BU (receita/volume total do mercado)."""
    rows = []
    base_map = {
        "Milk Run": 120,
        "Distribuição de Peças": 160,
        "Gerenciamento de Pátios": 90,
        "Transporte de Veículos": 220,
        "PDI": 70,
        "PDS": 60,
    }
    for m in months:
        for b in bus:
            base = base_map[b]
            sazonal = 1 + 0.05*np.sin((m.timetuple().tm_yday/365)*2*math.pi)
            total = int(max(10, np.random.normal(base, base*0.12) * sazonal))
            rows.append({"mes": m, "bu": b, "tamanho_mercado": total})
    return pd.DataFrame(rows)

def gen_competitor_pricing(players: List[str], bus: List[str]) -> pd.DataFrame:
    """Preço médio relativo por BU e player (índice Bluecompany=100 +- variações) + SLA/Leadtime."""
    rows = []
    base_preco_bu = {
        "Milk Run": 1000,
        "Distribuição de Peças": 1400,
        "Gerenciamento de Pátios": 1800,
        "Transporte de Veículos": 2200,
        "PDI": 1300,
        "PDS": 1200,
    }
    for b in bus:
        for p in players:
            idx = 1.0 if p==COMPANY else np.clip(np.random.normal(1.02, 0.07), 0.88, 1.22)
            preco_medio = base_preco_bu[b] * idx
            sla = np.clip(np.random.normal(0.93 if p==COMPANY else 0.9, 0.05), 0.7, 0.99)
            lead = np.clip(np.random.normal(4.5 if p==COMPANY else 5.2, 1.1), 2.0, 9.0)
            rows.append({
                "bu": b, "player": p,
                "preco_medio": float(preco_medio),
                "indice_preco": preco_medio / base_preco_bu[b],
                "sla_cumprimento": float(sla),
                "leadtime_dias": float(lead),
            })
    return pd.DataFrame(rows)

def gen_competitor_diff(players: List[str], bus: List[str]) -> pd.DataFrame:
    """Diferenciais competitivos (scores 0-100) por BU, player e pilar."""
    pilares = ["Cobertura Geográfica", "Telemetria/Rastreio", "Certificações/ESG",
               "Qualidade de SLA", "Flexibilidade Operacional", "Capilaridade Hubs"]
    rows = []
    for b in bus:
        for p in players:
            base = 70 if p==COMPANY else 60
            for pil in pilares:
                mu = base + np.random.normal(0, 6)
                rows.append({"bu": b, "player": p, "pilar": pil, "score": int(np.clip(mu, 35, 95))})
    return pd.DataFrame(rows)

def gen_coverage(players: List[str], bus: List[str], ufs: List[str]) -> pd.DataFrame:
    """Cobertura por UF (presença=1) por BU e player."""
    rows = []
    for b in bus:
        for p in players:
            prob = 0.70 if p==COMPANY else 0.55
            for uf in ufs:
                pres = np.random.rand() < np.clip(np.random.normal(prob, 0.08), 0.25, 0.9)
                rows.append({"bu": b, "player": p, "uf": uf, "presenca": int(pres)})
    return pd.DataFrame(rows)

def gen_events(players: List[str], months: List[date], bus: List[str]) -> pd.DataFrame:
    """Eventos competitivos por BU."""
    tipos = ["Contrato", "Hub/Filial", "Parceria", "Aquisição/M&A", "Regulatório", "Campanha"]
    severidades = ["Baixo", "Médio", "Alto", "Crítico"]
    weights = [5, 7, 4, 2]
    rows = []
    for m in months:
        for _ in range(random.randint(2, 7)):
            p = random.choice([pp for pp in players if pp != COMPANY])
            b = random.choice(bus)
            t = random.choice(tipos)
            sev = random.choices(severidades, weights=weights, k=1)[0]
            score = {"Baixo": 10, "Médio": 30, "Alto": 70, "Crítico": 90}[sev]
            desc = f"{t} de {p} em {b} ({m.strftime('%b/%Y')})"
            rows.append({
                "data": m + timedelta(days=random.randint(0, 27)),
                "player": p, "bu": b, "tipo": t, "severidade": sev, "score": score,
                "descricao": desc
            })
    return pd.DataFrame(rows)

def gen_docs_inventory() -> pd.DataFrame:
    """Inventário de estudos/dossiês (metadados simulados)."""
    bases = [
        ("Desk – Concorrência (Tegma/JSL/CEVA)", "Concorrência", "PDF", "2025-07-29"),
        ("Relatório – Sizing por BU (Q2)", "Market", "XLSX", "2025-08-05"),
        ("Quali – Roteiro Stakeholders BU Pátios", "Concorrência", "DOCX", "2025-08-01"),
        ("Proposta #9794 – Glovis (Jun/25)", "Overview", "PDF", "2025-06-30"),
        ("Radar – Tendências Logísticas (Jul)", "Tendências", "PDF", "2025-07-31"),
    ]
    docs = [{"titulo": t, "aba": aba, "tipo": ext, "data": pd.to_datetime(dt)} for t, aba, ext, dt in bases]
    return pd.DataFrame(docs)

# =============================================================================
# CRIAÇÃO DOS DATASETS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data():
    df_ms   = gen_market_share_series(PLAYERS, MESES, BUS)
    df_siz  = gen_market_sizing(BUS, MESES)
    df_prc  = gen_competitor_pricing(PLAYERS, BUS)
    df_diff = gen_competitor_diff(PLAYERS, BUS)
    df_cov  = gen_coverage(PLAYERS, BUS, UF)
    df_evt  = gen_events(PLAYERS, MESES, BUS)
    df_docs = gen_docs_inventory()
    df_evt  = ensure_datetime(df_evt, "data")
    df_docs = ensure_datetime(df_docs, "data")
    return {
        "share": df_ms,
        "sizing": df_siz,
        "pricing": df_prc,
        "diffs": df_diff,
        "coverage": df_cov,
        "events": df_evt,
        "docs": df_docs,
    }

data = load_data()

# =============================================================================
# HELPERS DE UI
# =============================================================================
def metric_card(label: str, value: str, delta: float | None = None):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-up" if delta >= 0 else "metric-delta-down"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div class='{cls}'>{arrow} {delta:.1f}%</div>"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def pct(x: float) -> str:
    return f"{x*100:.1f}%" if x==x else "–"

def line_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.line(df, x=x, y=y, color=color, markers=True)
    fig.update_layout(
        title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if y in ("market_share",):
        fig.update_yaxes(tickformat=",.0%")
    return fig

def area_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.area(df, x=x, y=y, color=color)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickformat=",.0%")
    return fig

def bars_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=".0%")
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def bars_num(df: pd.DataFrame, x: str, y: str, color: str, title: str, orientation="v"):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=True, orientation=orientation)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def alert_box(titulo: str, texto: str, nivel: str = "normal"):
    cls = "alert"
    if nivel == "critico":
        cls += " alert-critico"
    elif nivel == "alto":
        cls += " alert-alto"
    st.markdown(f"<div class='{cls}'><b>{titulo}</b><br/>{texto}</div>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR – FILTROS GLOBAIS
# =============================================================================
with st.sidebar:
    st.write("## ⚙️ Filtros")
    default_players = [COMPANY, "Tegma", "JSL", "CEVA"]
    players_sel = st.multiselect("Players", PLAYERS, default=default_players)
    bus_sel = st.multiselect("BUs", BUS, default=["Transporte de Veículos", "Gerenciamento de Pátios", "Distribuição de Peças"])
    tri_sel = st.selectbox("Trimestre", options=[f"{y}-T{q}" for (y, q) in TRIS], index=len(TRIS)-1)
    ano_tri = tuple(map(int, tri_sel.replace("-T", " ").split()))
    mes_sel = st.slider("Janela de meses (histórico)", min_value=3, max_value=12, value=12)

    st.divider()
    st.caption("Versão: v1 • Dados simulados (mock)")

# Subconjuntos
df_share = data["share"][(data["share"]["player"].isin(players_sel)) & (data["share"]["bu"].isin(bus_sel))].copy()
df_share_hist = df_share[df_share["mes"].isin(MESES[-mes_sel:])].copy()

df_sizing = data["sizing"][data["sizing"]["bu"].isin(bus_sel)].copy()
df_sizing_hist = df_sizing[df_sizing["mes"].isin(MESES[-mes_sel:])].copy()

df_pricing = data["pricing"][(data["pricing"]["player"].isin(players_sel)) & (data["pricing"]["bu"].isin(bus_sel))].copy()
df_diffs = data["diffs"][(data["diffs"]["player"].isin(players_sel)) & (data["diffs"]["bu"].isin(bus_sel))].copy()
df_cov = data["coverage"][(data["coverage"]["player"].isin(players_sel)) & (data["coverage"]["bu"].isin(bus_sel))].copy()

df_events = data["events"][data["events"]["player"].isin([p for p in players_sel if p != COMPANY]) & (data["events"]["bu"].isin(bus_sel))].copy()
df_events = ensure_datetime(df_events, "data")

# =============================================================================
# HEADER
# =============================================================================
st.title("Atlas – Bluecompany (Glovis)")
st.caption("War Room de Inteligência: visão integrada de Share & Sizing, Concorrência, Tendências e AI • Dados simulados")

# =============================================================================
# ABAS
# =============================================================================
tab_overview, tab_market, tab_comp, tab_trends, tab_ai = st.tabs([
    "Overview", "Market (Share & Sizing)", "Concorrência (Diferenciais e Preços)", "Tendências", "AI"
])

# =============================================================================
# OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Resumo Executivo – Bluecompany")

    # KPIs
    if not df_share_hist.empty:
        last_month = df_share_hist["mes"].max()
        prev_months = sorted(df_share_hist["mes"].unique())
        prev_month = prev_months[-2] if len(prev_months) >= 2 else prev_months[-1]
        ms_cur = df_share_hist[(df_share_hist["player"]==COMPANY) & (df_share_hist["mes"]==last_month)]["market_share"].mean()
        ms_prev = df_share_hist[(df_share_hist["player"]==COMPANY) & (df_share_hist["mes"]==prev_month)]["market_share"].mean()
        delta_ms = ((ms_cur - ms_prev)/max(ms_prev, 1e-9))*100 if ms_prev==ms_prev else 0.0
    else:
        ms_cur, delta_ms = np.nan, 0.0

    cov = df_cov[df_cov["player"]==COMPANY].groupby("uf")["presenca"].max().sum() if not df_cov.empty else 0
    idx_preco = df_pricing[df_pricing["player"]==COMPANY]["indice_preco"].mean() if not df_pricing.empty else np.nan
    sla = df_pricing[df_pricing["player"]==COMPANY]["sla_cumprimento"].mean() if not df_pricing.empty else np.nan
    lead = df_pricing[df_pricing["player"]==COMPANY]["leadtime_dias"].mean() if not df_pricing.empty else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Market Share (mês)", pct(ms_cur), delta_ms)
    with c2: metric_card("Cobertura (UFs c/ presença)", f"{int(cov)}")
    with c3: metric_card("Índice de Preço (ref. BU)", f"{(idx_preco*100):.0f}%" if idx_preco==idx_preco else "–")
    with c4: metric_card("SLA médio • Leadtime", f"{pct(sla)} • {lead:.1f}d" if sla==sla and lead==lead else "–")

    st.divider()

    o1, o2 = st.columns((1,1))
    with o1:
        st.markdown("<div class='block-title'>Market share – histórico (BUs selecionadas)</div>", unsafe_allow_html=True)
        fig_ms = line_pct(df_share_hist.groupby(["mes","player"], as_index=False)["market_share"].mean(),
                          x="mes", y="market_share", color="player", title="")
        st.plotly_chart(fig_ms, use_container_width=True)
    with o2:
        st.markdown("<div class='block-title'>Market sizing – histórico (BUs selecionadas)</div>", unsafe_allow_html=True)
        fig_sz = px.area(df_sizing_hist, x="mes", y="tamanho_mercado", color="bu", title="")
        fig_sz.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sz, use_container_width=True)

    st.divider()
    st.markdown("<div class='block-title'>Alertas Recentes</div>", unsafe_allow_html=True)
    evts = df_events.sort_values("data", ascending=False).head(6)
    if evts.empty:
        st.info("Sem eventos recentes para os filtros atuais.")
    else:
        for _, r in evts.iterrows():
            nivel = "critico" if r["severidade"]=="Crítico" else ("alto" if r["severidade"]=="Alto" else "normal")
            alert_box(f"{r['tipo']} – {r['player']} / {r['bu']} ({r['data'].strftime('%d/%m/%Y')})", r["descricao"], nivel=nivel)

    with st.expander("Inventário de estudos (últimos)"):
        st.dataframe(data["docs"].sort_values("data", ascending=False),
                     use_container_width=True, height=200)

# =============================================================================
# MARKET (Share & Sizing)
# =============================================================================
with tab_market:
    st.subheader("Market – Share & Sizing")

    bu_focal = st.selectbox("Escolha uma BU para detalhar", options=bus_sel if bus_sel else BUS, index=0)
    df_share_bu = df_share_hist[df_share_hist["bu"]==bu_focal].copy()
    df_sizing_bu = df_sizing_hist[df_sizing_hist["bu"]==bu_focal].copy()

    m1, m2 = st.columns((1,1))
    with m1:
        st.markdown(f"<div class='block-title'>Share por player – {bu_focal}</div>", unsafe_allow_html=True)
        fig1 = line_pct(df_share_bu.groupby(["mes","player"], as_index=False)["market_share"].mean(),
                        x="mes", y="market_share", color="player", title="")
        st.plotly_chart(fig1, use_container_width=True)
    with m2:
        st.markdown(f"<div class='block-title'>Sizing (volume total/receita) – {bu_focal}</div>", unsafe_allow_html=True)
        fig2 = px.bar(df_sizing_bu, x="mes", y="tamanho_mercado", title="")
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    if not df_share_hist.empty:
        lm = df_share_hist["mes"].max()
        cur_bu_share = (df_share_hist[df_share_hist["mes"]==lm]
                        .groupby(["bu","player"], as_index=False)["market_share"].mean())
        fig3 = bars_pct(cur_bu_share, x="bu", y="market_share", color="player",
                        title="Share atual por BU (mês mais recente)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sem séries de share para os filtros atuais.")

    st.divider()
    cov_blue = df_cov[df_cov["player"]==COMPANY].groupby("uf")["presenca"].max().reset_index(name="blue")
    cov_others = (df_cov[df_cov["player"]!=COMPANY]
                  .groupby(["uf"])["presenca"].mean().reset_index(name="concorrentes_media"))
    cov_merged = pd.merge(cov_blue, cov_others, on="uf", how="outer").fillna(0)
    cov_merged["dif"] = cov_merged["blue"] - cov_merged["concorrentes_media"]

    m3, m4 = st.columns((1,1))
    with m3:
        st.markdown("<div class='block-title'>Cobertura por UF – Bluecompany</div>", unsafe_allow_html=True)
        fig4 = px.bar(cov_blue.sort_values("blue", ascending=True), x="blue", y="uf",
                      orientation="h", title="")
        fig4.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                           yaxis=dict(categoryorder="array", categoryarray=cov_blue.sort_values("blue")["uf"].tolist()))
        st.plotly_chart(fig4, use_container_width=True)
    with m4:
        st.markdown("<div class='block-title'>Δ Cobertura vs concorrentes (média)</div>", unsafe_allow_html=True)
        fig5 = px.bar(cov_merged.sort_values("dif", ascending=True), x="dif", y="uf",
                      orientation="h", title="")
        fig5.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig5, use_container_width=True)

# =============================================================================
# CONCORRÊNCIA (Diferenciais e Preços)
# =============================================================================
with tab_comp:
    st.subheader("Concorrência – Diferenciais e Preços")

    bu_comp = st.selectbox("BU para comparar", options=bus_sel if bus_sel else BUS, index=0, key="comp_bu")
    df_prc_bu = df_pricing[df_pricing["bu"]==bu_comp].copy()
    df_diff_bu = df_diffs[df_diffs["bu"]==bu_comp].copy()

    c1, c2 = st.columns((1,1))
    with c1:
        figp = bars_num(df_prc_bu.sort_values("preco_medio"),
                        x="player", y="preco_medio", color="player",
                        title=f"Preço médio por player – {bu_comp}")
        figp.update_layout(showlegend=False)
        st.plotly_chart(figp, use_container_width=True)
    with c2:
        figsla = px.scatter(df_prc_bu, x="leadtime_dias", y="sla_cumprimento", color="player",
                            size="preco_medio", hover_data=["indice_preco"],
                            title=f"Posicionamento: SLA × Leadtime – {bu_comp}")
        figsla.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        figsla.update_yaxes(tickformat=",.0%")
        st.plotly_chart(figsla, use_container_width=True)

    st.divider()
    concorrente = st.selectbox("Concorrente para comparar no radar", options=[p for p in players_sel if p!=COMPANY], index=0 if len(players_sel)>1 else 0)
    pilares = df_diff_bu["pilar"].unique().tolist()

    def prep_radar(df: pd.DataFrame, player: str) -> pd.DataFrame:
        tmp = df[df["player"]==player][["pilar","score"]].copy()
        tmp = tmp.set_index("pilar").reindex(pilares).reset_index()
        return tmp

    rad_blue = prep_radar(df_diff_bu, COMPANY)
    rad_comp = prep_radar(df_diff_bu, concorrente)

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=rad_blue["score"], theta=rad_blue["pilar"], fill='toself', name=COMPANY))
    radar.add_trace(go.Scatterpolar(r=rad_comp["score"], theta=rad_comp["pilar"], fill='toself', name=concorrente))
    radar.update_layout(
        title=f"Diferenciais competitivos – {bu_comp}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=420, margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(radar, use_container_width=True)

    st.divider()
    with st.expander("Tabela – Preços & SLAs (BU)"):
        st.dataframe(df_prc_bu.sort_values(["player"]), use_container_width=True, height=240)
    with st.expander("Tabela – Diferenciais (BU)"):
        tbl = df_diff_bu.pivot_table(index="pilar", columns="player", values="score").reset_index()
        st.dataframe(tbl, use_container_width=True, height=240)

# =============================================================================
# TENDÊNCIAS
# =============================================================================
with tab_trends:
    st.subheader("Tendências – Impacto & Maturidade")

    temas = [
        "Telemetria & Rastreabilidade",
        "Automação de Pátios",
        "Analytics/Roteirização (AI)",
        "ESG & Certificações",
        "Veículos Elétricos/Autônomos",
        "Integração Sistemas & S&OP",
    ]
    impacto = np.clip(np.random.normal(72, 12, len(temas)), 35, 95)
    matur  = np.clip(np.random.normal(58, 18, len(temas)), 15, 95)
    df_rad = pd.DataFrame({"tema": temas, "impacto": impacto, "maturidade": matur})

    t1, t2 = st.columns((1,1))
    with t1:
        st.markdown("<div class='block-title'>Radar (impacto × maturidade)</div>", unsafe_allow_html=True)
        figt = px.scatter(df_rad, x="maturidade", y="impacto", text="tema", size=[18]*len(temas), title="")
        figt.update_traces(textposition="top center")
        figt.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                           xaxis_title="Maturidade", yaxis_title="Impacto esperado")
        st.plotly_chart(figt, use_container_width=True)
    with t2:
        marcos = [
            ("2025-02-15", "ANTT – tabela mínima frete (revisão)"),
            ("2025-03-30", "ABOL – guia ESG operadores logísticos"),
            ("2025-05-10", "Resolução – rastreabilidade em pátios"),
            ("2025-07-20", "Portos – concessões/novos terminais"),
        ]
        df_reg = pd.DataFrame({"data": pd.to_datetime([d for d,_ in marcos]),
                               "evento": [e for _,e in marcos]})
        df_reg["y"] = 1
        figr = px.scatter(df_reg, x="data", y="y", text="evento", title="Linha do tempo regulatória (exemplos)")
        figr.update_traces(textposition="top center")
        figr.update_layout(yaxis=dict(visible=False), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figr, use_container_width=True)

    st.divider()
    st.markdown("<div class='block-title'>Central de Insights (inventário)</div>", unsafe_allow_html=True)
    filtro_aba = st.selectbox("Filtrar por aba", options=["Todas"] + sorted(data["docs"]["aba"].unique().tolist()))
    docs = data["docs"] if filtro_aba=="Todas" else data["docs"][data["docs"]["aba"]==filtro_aba]
    st.dataframe(docs.sort_values("data", ascending=False), use_container_width=True, height=240)

    with st.expander("Notas de fontes & próximos passos"):
        st.markdown("""
- Estrutura de temas, BUs e concorrentes baseada na proposta da Glovis (#9794).
- Para produção real: integrar fontes como ANFAVEA, ABOL, FENABRAVE, ABLA, IBGE, OICA, relatórios de RI (Tegma/JSL/CEVA), Comprasnet/SICAF e mídia setorial.
- Recomenda-se: Desk + Quali (stakeholders) → Quant (validação) → refresh trimestral no Atlas.
        """)

# =============================================================================
# AI – Q&A SIMULADO (RAG MOCK)
# =============================================================================
with tab_ai:
    st.subheader("AI – Pergunte ao Atlas (mock)")
    st.caption("Gera respostas a partir dos dados atuais deste app (share, sizing, preços/SLAs, diferenciais, cobertura, eventos e inventário).")

    # Filtros locais (opcionais) para contexto das respostas
    ai_c1, ai_c2 = st.columns((1,1))
    with ai_c1:
        bu_ai = st.selectbox("BU foco para responder", options=(bus_sel if bus_sel else BUS), index=0, key="ai_bu")
    with ai_c2:
        players_ai = st.multiselect("Players foco nas respostas", options=players_sel, default=players_sel, key="ai_players")

    # Knowledge base simples a partir dos dataframes filtrados
    def build_kb() -> List[Dict[str, str]]:
        kb = []

        # 1) Market share (mês mais recente) por BU e player
        if not df_share_hist.empty:
            lm = df_share_hist["mes"].max()
            cur = (df_share_hist[(df_share_hist["mes"]==lm) & (df_share_hist["bu"]==bu_ai) & (df_share_hist["player"].isin(players_ai))]
                   .groupby("player", as_index=False)["market_share"].mean())
            for _, r in cur.iterrows():
                kb.append({"tema":"share", "texto": f"No último mês, em {bu_ai}, o market share estimado de {r['player']} foi {r['market_share']:.1%}."})

        # 2) Sizing (mês mais recente) para a BU
        if not df_sizing_hist.empty:
            lm2 = df_sizing_hist["mes"].max()
            siz = df_sizing_hist[(df_sizing_hist["mes"]==lm2) & (df_sizing_hist["bu"]==bu_ai)]["tamanho_mercado"].mean()
            if pd.notnull(siz):
                kb.append({"tema":"sizing", "texto": f"O tamanho de mercado estimado em {bu_ai} no mês mais recente foi {siz:,.0f} (unidade simulada)."})

        # 3) Pricing/SLA/Leadtime por player (BU)
        prc = df_pricing[(df_pricing["bu"]==bu_ai) & (df_pricing["player"].isin(players_ai))]
        for _, r in prc.iterrows():
            kb.append({"tema":"precos", "texto": f"Em {bu_ai}, {r['player']} tem preço médio ~{r['preco_medio']:.0f} (índice {r['indice_preco']*100:.0f}%), SLA {r['sla_cumprimento']:.0%} e leadtime {r['leadtime_dias']:.1f}d."})

        # 4) Diferenciais (média por player) – BU
        dif = df_diffs[(df_diffs["bu"]==bu_ai) & (df_diffs["player"].isin(players_ai))]
        if not dif.empty:
            agg = dif.groupby("player")["score"].mean().reset_index()
            for _, r in agg.iterrows():
                kb.append({"tema":"diferenciais", "texto": f"Em {bu_ai}, a média de diferenciais de {r['player']} está em {r['score']:.0f}/100."})

        # 5) Cobertura – UFs com presença (Bluecompany) na BU
        cov_bu = df_cov[(df_cov["bu"]==bu_ai) & (df_cov["player"]==COMPANY)]
        if not cov_bu.empty:
            ufs = cov_bu.groupby("uf")["presenca"].max()
            kb.append({"tema":"cobertura", "texto": f"Bluecompany tem presença em {int((ufs>0).sum())} UFs na BU {bu_ai}."})

        # 6) Eventos recentes
        ev_last = df_events[df_events["bu"]==bu_ai].sort_values("data", ascending=False).head(8)
        for _, r in ev_last.iterrows():
            kb.append({"tema":"eventos", "texto": f"{r['data'].strftime('%d/%m')}: {r['tipo']} de {r['player']} em {bu_ai} (sev. {r['severidade']})."})

        # 7) Inventário
        docs = data["docs"].sort_values("data", ascending=False).head(5)
        for _, r in docs.iterrows():
            kb.append({"tema":"docs", "texto": f"{r['data'].date()}: {r['titulo']} [{r['aba']}/{r['tipo']}]."})
        return kb

    KB = build_kb()

    # UI de exemplos + busca simples
    ex_cols = st.columns(5)
    examples = [
        f"Qual o share da {COMPANY} em {bu_ai}?",
        f"Como está o preço e SLA por player em {bu_ai}?",
        f"Eventos recentes dos concorrentes em {bu_ai}?",
        f"Tamanho de mercado atual de {bu_ai}?",
        "Quais docs mais recentes do inventário?"
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True):
            st.session_state["atlas_q"] = ex

    q = st.text_input("Pergunte ao Atlas", value=st.session_state.get("atlas_q", ""))
    ask = st.button("Responder", type="primary")

    def simple_search(query: str, kb: List[Dict[str, str]], topk: int = 6) -> List[str]:
        terms = set([t.lower() for t in query.split()])
        scores = []
        for chunk in kb:
            text = chunk["texto"].lower()
            score = sum(1 for t in terms if t in text)
            scores.append((score, chunk["texto"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for score, t in scores[:topk] if score > 0]

    if not q:
        st.info("Dica: clique em um exemplo acima ou digite uma pergunta sobre market share, preços/SLAs, eventos, cobertura ou docs.")
    elif ask:
        resps = simple_search(q, KB)
        if not resps:
            st.warning("Não encontrei uma resposta direta. Tente reformular, ou consulte as abas acima.")
        else:
            st.success("Respostas encontradas:")
            for r in resps:
                st.markdown(f"- {r}")

    st.divider()
    st.markdown("### Central de Insights (inventário)")
    filtro_aba_ai = st.selectbox("Filtrar por aba (docs)", options=["Todas"] + sorted(data["docs"]["aba"].unique().tolist()), key="ai_docs_aba")
    docs_ai = data["docs"] if filtro_aba_ai=="Todas" else data["docs"][data["docs"]["aba"]==filtro_aba_ai]
    st.dataframe(docs_ai.sort_values("data", ascending=False), use_container_width=True, height=220)

# =============================================================================
# RODAPÉ
# =============================================================================
st.divider()
st.write(":grey[Protótipo conceitual • Okiar • Dados simulados para apresentação • Versão v1]")
