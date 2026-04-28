"""
ghosh_app.py — Multi-country Ghosh supply-side shock simulator
================================================================
Reads OECD ICIO single-country "ttl" CSVs (XXX2022ttl.csv) and propagates
a supply-side shock to imports of any chosen sector through the Ghosh
inverse, reporting sectoral supply contraction and GDP loss.

Run:
    streamlit run ghosh_app.py

Expected file layout (configurable in the sidebar):
    ./data/
        DEU2022ttl.csv
        FRA2022ttl.csv
        ...

Units: OECD ICIO is published in USD millions, so all monetary outputs
       are in USD.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# 1. Static lookups
# ----------------------------------------------------------------------------

ISO3_TO_NAME: dict[str, str] = {
    "AGO": "Angola", "ARE": "United Arab Emirates", "ARG": "Argentina",
    "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium",
    "BGD": "Bangladesh", "BGR": "Bulgaria", "BLR": "Belarus",
    "BRA": "Brazil", "BRN": "Brunei", "CAN": "Canada", "CHE": "Switzerland",
    "CHL": "Chile", "CHN": "China", "CIV": "Côte d'Ivoire",
    "CMR": "Cameroon", "COD": "DR Congo", "COL": "Colombia",
    "CRI": "Costa Rica", "CYP": "Cyprus", "CZE": "Czechia",
    "DEU": "Germany", "DNK": "Denmark", "EGY": "Egypt", "ESP": "Spain",
    "EST": "Estonia", "FIN": "Finland", "FRA": "France",
    "GBR": "United Kingdom", "GRC": "Greece", "HKG": "Hong Kong",
    "HRV": "Croatia", "HUN": "Hungary", "IDN": "Indonesia", "IND": "India",
    "IRL": "Ireland", "ISL": "Iceland", "ISR": "Israel", "ITA": "Italy",
    "JOR": "Jordan", "JPN": "Japan", "KAZ": "Kazakhstan", "KHM": "Cambodia",
    "KOR": "Korea", "LAO": "Laos", "LTU": "Lithuania", "LUX": "Luxembourg",
    "LVA": "Latvia", "MAR": "Morocco", "MEX": "Mexico", "MLT": "Malta",
    "MMR": "Myanmar", "MYS": "Malaysia", "NGA": "Nigeria",
    "NLD": "Netherlands", "NOR": "Norway", "NZL": "New Zealand",
    "PAK": "Pakistan", "PER": "Peru", "PHL": "Philippines", "POL": "Poland",
    "PRT": "Portugal", "ROU": "Romania", "RUS": "Russia",
    "SAU": "Saudi Arabia", "SEN": "Senegal", "SGP": "Singapore",
    "STP": "São Tomé and Príncipe", "SVK": "Slovakia", "SVN": "Slovenia",
    "SWE": "Sweden", "THA": "Thailand", "TUN": "Tunisia", "TUR": "Türkiye",
    "TWN": "Chinese Taipei", "UKR": "Ukraine", "USA": "United States",
    "VNM": "Viet Nam", "ZAF": "South Africa",
}

SECTOR_NAMES: dict[str, str] = {
    "A01": "Agriculture", "A02": "Forestry", "A03": "Fishing",
    "B05": "Coal mining", "B06": "Oil & gas extraction",
    "B07": "Metal ores", "B08": "Other mining", "B09": "Mining services",
    "C10T12": "Food & beverages", "C13T15": "Textiles & apparel",
    "C16": "Wood products", "C17_18": "Paper & printing",
    "C19": "Refined petroleum", "C20": "Chemicals",
    "C21": "Pharmaceuticals", "C22": "Rubber & plastics",
    "C23": "Non-metallic minerals", "C24A": "Basic iron & steel",
    "C24B": "Non-ferrous metals", "C25": "Fabricated metals",
    "C26": "Electronics", "C27": "Electrical equipment",
    "C28": "Machinery", "C29": "Motor vehicles",
    "C301": "Ships & boats", "C302T309": "Other transport eq.",
    "C31T33": "Other mfg & repair",
    "D": "Electricity & gas", "E": "Water & waste", "F": "Construction",
    "G": "Wholesale & retail", "H49": "Land transport",
    "H50": "Water transport", "H51": "Air transport",
    "H52": "Warehousing", "H53": "Postal", "I": "Hospitality",
    "J58T60": "Publishing & media", "J61": "Telecoms",
    "J62_63": "IT services", "K": "Finance", "L": "Real estate",
    "M": "Professional svcs", "N": "Admin svcs", "O": "Public admin",
    "P": "Education", "Q": "Health & social", "R": "Arts & rec.",
    "S": "Other svcs", "T": "Households",
}

# ----------------------------------------------------------------------------
# 2. Core model
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_country(csv_path: str) -> dict:
    """Load an OECD ICIO single-country ttl CSV and return its key arrays."""
    df = pd.read_csv(csv_path, header=0, index_col=0)
    sectors = list(df.columns[:50])
    data = df.values.astype(float)
    return dict(
        sectors=sectors,
        Z=data[:50, :50],
        IMPO=data[:50, 58],     # imports (negative)
        OUTPUT=data[:50, 59],   # domestic gross output
        VALU=data[53, :50],     # value added per sector
    )

@st.cache_data(show_spinner=False)
def run_ghosh_shock(csv_path: str, sector_code: str, delta: float) -> dict:
    """
    Apply Ghosh propagation of a Δ-fraction cut to sector_code's imports.

    Model
    -----
    x_i  = OUTPUT_i + IMPORTS_i        (total supply)
    B    = diag(x)^-1 · Z              (allocation matrix)
    G    = (I - B)^-1                  (Ghosh inverse)
    Δv_j = -δ · IMPORTS_{sector}       (primary supply shock)
    Δx'  = Δv' · G
    ΔGDP_j = Δx_j · VALU_j / x_j
    """
    c = load_country(csv_path)
    sectors, Z, IMPO, OUTPUT, VALU = (
        c["sectors"], c["Z"], c["IMPO"], c["OUTPUT"], c["VALU"]
    )
    n = len(sectors)
    x = OUTPUT - IMPO

    # Guard against zero-supply rows
    safe_x = np.where(x > 0, x, 1.0)
    B = Z / safe_x[:, None]
    G = np.linalg.inv(np.eye(n) - B)

    j = sectors.index(sector_code)
    imports_j = -IMPO[j]
    shock = -delta * imports_j

    dv = np.zeros(n)
    dv[j] = shock
    dx = dv @ G

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_supply = np.where(x > 0, 100.0 * dx / x, 0.0)
        dGDP = np.where(x > 0, dx * VALU / x, 0.0)

    return dict(
        sectors=sectors,
        dx=dx, pct_supply=pct_supply, dGDP=dGDP,
        VALU=VALU, x=x,
        shock=shock, imports_j=imports_j,
        total_dx=float(dx.sum()),
        total_dGDP=float(dGDP.sum()),
        share_GDP=float(100.0 * dGDP.sum() / VALU.sum())
                  if VALU.sum() > 0 else 0.0,
    )

# ----------------------------------------------------------------------------
# 3. Plot
# ----------------------------------------------------------------------------

def make_figure(res: dict, country_name: str, sector_code: str,
                delta: float, top_n: int = 12) -> go.Figure:
    sectors    = res["sectors"]
    pct        = res["pct_supply"]
    dGDP       = res["dGDP"]

    labels = [f"{SECTOR_NAMES.get(s, s)} ({s})" for s in sectors]
    order_pct = np.argsort(pct)[:top_n]
    order_gdp = np.argsort(dGDP)[:top_n]

    # gradient colour scales
    reds  = [f"rgba({200-10*k},{50+5*k},{50+5*k},0.9)"   for k in range(top_n)]
    blues = [f"rgba({50+5*k},{100+8*k},{200-10*k},0.9)"  for k in range(top_n)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Supply contraction by sector</b>",
                        "<b>GDP loss by sector</b>"),
        horizontal_spacing=0.32,
    )

    # Left: % supply contraction
    sel = order_pct
    fig.add_trace(go.Bar(
        y=[labels[i] for i in sel][::-1],
        x=pct[sel][::-1],
        orientation="h",
        marker=dict(color=reds[::-1], line=dict(color="black", width=0.4)),
        text=[f"{v:.1f}%" for v in pct[sel][::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#222"),
        hovertemplate="<b>%{y}</b><br>Δ supply: %{x:.2f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Right: GDP loss in USD bn (with % in label)
    sel = order_gdp
    fig.add_trace(go.Bar(
        y=[labels[i] for i in sel][::-1],
        x=(dGDP[sel] / 1000)[::-1],
        orientation="h",
        marker=dict(color=blues[::-1], line=dict(color="black", width=0.4)),
        text=[f"{dGDP[i]/1000:.2f} bn ({pct[i]:.2f}%)" for i in sel][::-1],
        textposition="outside",
        textfont=dict(size=11, color="#222"),
        hovertemplate="<b>%{y}</b><br>Δ GDP: %{x:.3f} bn USD<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    # Headroom for outside labels
    xmin1 = pct[order_pct].min() * 1.30
    xmin2 = (dGDP[order_gdp].min() / 1000) * 1.45
    fig.update_xaxes(range=[xmin1, abs(xmin1)*0.18], row=1, col=1,
                     title=dict(text="Δ supply (% of total sector supply)",
                                font=dict(color="#222")),
                     tickfont=dict(color="#222"),
                     zeroline=True, zerolinecolor="black", zerolinewidth=1,
                     gridcolor="#e5e5e5")
    fig.update_xaxes(range=[xmin2, abs(xmin2)*0.32], row=1, col=2,
                     title=dict(text="Δ GDP (bn USD)",
                                font=dict(color="#222")),
                     tickfont=dict(color="#222"),
                     zeroline=True, zerolinecolor="black", zerolinewidth=1,
                     gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=False, tickfont=dict(color="#222", size=11))

    # Re-style the subplot titles (pushed down to avoid overlap with axes)
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#222")
        ann.y = ann.y + 0.02  # nudge up away from y-axis labels

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=70, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",       # force light bg even under Streamlit dark theme
        font=dict(family="Arial, sans-serif", size=11, color="#222"),
        bargap=0.25,
    )
    return fig

# ----------------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Ghosh ICIO shock simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OECD ICIO — Ghosh supply-side shock simulator")
st.caption(
    "Propagates a supply-side shock to imports of a chosen sector "
    "through the Ghosh inverse $G = (I-B)^{-1}$, where "
    "$B_{ij}=Z_{ij}/x_i$ is the allocation matrix and "
    "$x_i = \\text{OUTPUT}_i + \\text{IMPORTS}_i$ is total supply."
)

# --- sidebar controls ---
with st.sidebar:
    st.header("Configuration")

    data_dir = st.text_input(
        "Data directory (folder containing the CSVs)",
        value="data",
        help="Path to a directory containing files like DEU2022ttl.csv.",
    )

    files = sorted(Path(data_dir).glob("*2022ttl.csv"))
    if not files:
        st.error(
            f"No `*2022ttl.csv` files found in `{data_dir}`. "
            "Put the OECD ICIO single-country CSVs in that folder."
        )
        st.stop()

    countries = [f.stem[:3] for f in files]
    default_country = "DEU" if "DEU" in countries else countries[0]
    country = st.selectbox(
        "Country",
        countries,
        index=countries.index(default_country),
        format_func=lambda c: f"{c} — {ISO3_TO_NAME.get(c, c)}",
    )

    # Determine available sectors from the chosen file
    csv_path = str(Path(data_dir) / f"{country}2022ttl.csv")
    available_sectors = load_country(csv_path)["sectors"]

    default_sector = "B06" if "B06" in available_sectors else available_sectors[0]
    sector = st.selectbox(
        "Sector to shock (cut its imports)",
        available_sectors,
        index=available_sectors.index(default_sector),
        format_func=lambda s: f"{s} — {SECTOR_NAMES.get(s, s)}",
    )

    delta_pct = st.slider(
        "Shock magnitude (% of imports cut)",
        min_value=0, max_value=100, value=30, step=5,
    )
    delta = delta_pct / 100.0

    top_n = st.slider("Sectors shown in chart", 8, 25, 12)

    st.markdown("---")
    st.caption(
        "All monetary values in USD (millions in the data, billions in the chart). "
        "Source: OECD ICIO, 2022 release."
    )

# --- run the model ---
res = run_ghosh_shock(csv_path, sector, delta)
country_name = ISO3_TO_NAME.get(country, country)

# Warn if the chosen sector has no meaningful imports for this country
if res["imports_j"] < 1.0:
    st.warning(
        f"**{country_name}** has essentially no imports of "
        f"{sector} ({SECTOR_NAMES.get(sector, sector)}) — "
        f"the shock has no effect. Try a different sector or country."
    )

# --- headline metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Direct shock",
    f"{res['shock']/1000:,.2f} bn USD",
    delta=f"{-delta_pct}% of imports",
    delta_color="inverse",
)
m2.metric("Total supply loss",   f"{res['total_dx']/1000:,.2f} bn USD")
m3.metric("Total GDP loss",      f"{res['total_dGDP']/1000:,.2f} bn USD")
m4.metric("Share of GDP",        f"{res['share_GDP']:.3f}%")

# --- chart ---
st.subheader(
    f"{country_name} — Ghosh propagation of a {delta_pct}% cut to "
    f"{sector} ({SECTOR_NAMES.get(sector, sector)}) imports"
)
fig = make_figure(res, country_name, sector, delta, top_n=top_n)
st.plotly_chart(fig, use_container_width=True)

# --- detailed table + download ---
with st.expander("Sectoral results table", expanded=False):
    table = pd.DataFrame({
        "code":        res["sectors"],
        "sector":      [SECTOR_NAMES.get(s, s) for s in res["sectors"]],
        "Δ supply (mn USD)":   res["dx"].round(1),
        "Δ supply (%)":        res["pct_supply"].round(3),
        "Δ GDP (mn USD)":      res["dGDP"].round(1),
        "VA share":            (res["VALU"] / res["x"]).round(4),
    }).sort_values("Δ GDP (mn USD)")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name=f"ghosh_{country}_{sector}_d{delta_pct}.csv",
        mime="text/csv",
    )

# --- methodological notes ---
with st.expander("Method and caveats"):
    st.markdown(r"""
**Ghosh supply-side propagation.**
Given the intermediate-flow matrix $Z$, total supply $x_i = \text{OUTPUT}_i + \text{IMPORTS}_i$,
and the allocation matrix $B = \widehat{x}^{-1} Z$, the Ghosh inverse $G=(I-B)^{-1}$
maps a primary-supply shock $\Delta v$ into an output response
$\Delta x' = \Delta v' G$. The diagonal of $G$ exceeds 1 because of feedback —
downstream contraction reduces demand cycling back to the shocked sector.

**Shock specification.**
A fraction $\delta$ of the chosen sector's imports is removed:
$\Delta v_j = -\delta \cdot \text{IMPORTS}_j$. Other primary inputs are held fixed.

**GDP translation.**
$\Delta\text{GDP}_j = \Delta x_j \cdot v_j / x_j$ where $v_j$ is the sector's
value added. Each USD of supply carries $v_j/x_j$ of domestic VA on average;
the rest is intermediates and imported supply.

**Caveats.**
1. Single-country ICIO with competitive imports — the shock cuts the supply
   origin uniformly across all buyers; can't tell you which firm gets less.
2. Pure quantity model. No price channel, no substitution, no SPR releases,
   no behavioural response. For sustained shocks, expect 1.5–3× the GDP hit
   shown here once you add cost-push and demand multipliers.
3. Total imports of the sector are cut, not just the share routed via a
   particular geography. Realistic Hormuz-only, Russia-only, etc. requires
   the multi-country ICIO with origin-disaggregated import flows.
""")
