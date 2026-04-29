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


@st.cache_data(show_spinner=False)
def run_leontief_cascade(csv_path: str, sector_code: str,
                         delta: float, eps: float = 1e-6) -> dict:
    """
    Strict Leontief min-rule cascade.

    Production function
    -------------------
        x_j = min_i ( z_ij / a_ij )

    Each sector can produce only as much as its scarcest input allows.
    Apply the shock as a supply cap on the chosen sector's *imports*:
    available supply of input i drops to

        r_i = (OUTPUT_i / x_i) * s_i + (IMPORTS_i / x_i) * (1 - δ_i)

    where s_i is sector i's output scaling and δ_i is the shock fraction
    on its imports (non-zero only for the shocked sector). Sector j's
    feasible scaling is then

        s_j = min over inputs i (with a_ij > eps) of r_i

    Iterate until fixed point. The eps threshold filters out numerical
    noise (rounding-artefact a_ij values like 1e-9); set to 0 for the
    purely formal Leontief result.

    GDP impact
    ----------
        ΔGDP_j = (s_j - 1) · v_j
        Δx_j   = (s_j - 1) · x_j
    """
    c = load_country(csv_path)
    sectors, Z, IMPO, OUTPUT, VALU = (
        c["sectors"], c["Z"], c["IMPO"], c["OUTPUT"], c["VALU"]
    )
    n = len(sectors)
    x = OUTPUT - IMPO
    safe_x = np.where(x > 0, x, 1.0)
    A = Z / safe_x[None, :]

    j = sectors.index(sector_code)
    imports_j = float(-IMPO[j])
    imp_share = np.where(x > 0, -IMPO / x, 0.0)
    dom_share = np.where(x > 0,  OUTPUT / x, 0.0)

    shock = np.zeros(n)
    shock[j] = delta

    s = np.ones(n)
    converged = False
    for _ in range(500):
        r = dom_share * s + (1 - shock) * imp_share
        s_new = np.ones(n)
        # Vectorized "min over rows where A[i,j] > eps" per column j
        for jj in range(n):
            uses = A[:, jj] > eps
            if uses.any():
                s_new[jj] = min(1.0, r[uses].min())
        # Shocked sector itself capped by its own supply availability
        s_new[j] = min(s_new[j], r[j])
        if np.max(np.abs(s_new - s)) < 1e-10:
            converged = True
            s = s_new
            break
        s = s_new

    dx = (s - 1.0) * x
    dGDP = (s - 1.0) * VALU
    pct_supply = 100.0 * (s - 1.0)

    return dict(
        sectors=sectors,
        dx=dx, pct_supply=pct_supply, dGDP=dGDP,
        VALU=VALU, x=x, s=s,
        n_collapsed=int((s < 0.5).sum()),
        # Same key names as Ghosh so the headline metrics row works
        shock=float(-delta * imports_j),
        imports_j=imports_j,
        total_dx=float(dx.sum()),
        total_dGDP=float(dGDP.sum()),
        share_GDP=float(100.0 * dGDP.sum() / VALU.sum())
                  if VALU.sum() > 0 else 0.0,
        converged=converged,
    )


@st.cache_data(show_spinner=False)
def run_ces_cascade(csv_path: str, sector_code: str, delta: float,
                    sigma: float, eps: float = 1e-12) -> dict:
    """
    CES intermediate-aggregation cascade with elasticity of substitution σ.

    Each sector j produces output via a CES aggregator over its
    intermediate inputs:

        Q_j = (sum_i γ_ij * z_ij^η)^(1/η),   η = (σ-1)/σ

    Calibrated to reproduce the observed I-O flows at baseline (when all
    relative prices equal 1), the weights become γ_ij = m_ij^(1-η) where
    m_ij = a_ij / sum_k a_kj is the cost share of input i in j's
    intermediate basket.

    Under proportional rationing of available supply r_i, each sector's
    feasible output scaling becomes

        s_j = (sum_i m_ij * r_i^η)^(1/η)

    Iterated to fixed point. Limits:
        σ → 0   : Leontief (min rule)         — strict no-substitution
        σ = 1   : Cobb-Douglas (geometric mean) — unit-elastic
        σ → ∞   : linear (arithmetic mean)     — perfect substitution

    Realistic short-run values for energy-economy substitution: σ ∈ [0.1, 0.5].
    Long-run / textbook values: σ ∈ [0.5, 2.0].

    The sigma=0 case dispatches to the strict Leontief code path for
    numerical stability.
    """
    # σ = 0 → use the strict Leontief implementation
    if sigma < 0.01:
        return run_leontief_cascade(csv_path, sector_code, delta)

    c = load_country(csv_path)
    sectors, Z, IMPO, OUTPUT, VALU = (
        c["sectors"], c["Z"], c["IMPO"], c["OUTPUT"], c["VALU"]
    )
    n = len(sectors)
    x = OUTPUT - IMPO
    safe_x = np.where(x > 0, x, 1.0)
    A = Z / safe_x[None, :]                  # technical coefficients

    # Normalised cost shares within each sector's intermediate basket
    col_sum = A.sum(axis=0)
    safe_col = np.where(col_sum > 1e-12, col_sum, 1.0)
    M = np.where(col_sum > 1e-12, A / safe_col, 0.0)

    j = sectors.index(sector_code)
    imports_j = float(-IMPO[j])
    imp_share = np.where(x > 0, -IMPO / x, 0.0)
    dom_share = np.where(x > 0,  OUTPUT / x, 0.0)

    shock = np.zeros(n)
    shock[j] = delta

    eta = (sigma - 1.0) / sigma
    s = np.ones(n)
    converged = False

    for _ in range(500):
        r = dom_share * s + (1.0 - shock) * imp_share
        r_safe = np.maximum(r, 1e-12)
        s_new = np.ones(n)

        for jj in range(n):
            mj = M[:, jj]
            mask = mj > eps
            if not mask.any():
                continue
            mu = mj[mask]
            ru = r_safe[mask]
            if abs(eta) < 1e-6:
                # Cobb-Douglas: s = exp(sum m * log r)
                s_new[jj] = min(1.0, np.exp(np.sum(mu * np.log(ru))))
            else:
                inner = max(np.sum(mu * ru**eta), 1e-300)
                s_new[jj] = min(1.0, inner**(1.0 / eta))

        # Shocked sector capped by its own supply availability
        s_new[j] = min(s_new[j], r[j])

        if np.max(np.abs(s_new - s)) < 1e-9:
            converged = True
            s = s_new
            break
        s = s_new

    dx = (s - 1.0) * x
    dGDP = (s - 1.0) * VALU
    pct_supply = 100.0 * (s - 1.0)

    return dict(
        sectors=sectors,
        dx=dx, pct_supply=pct_supply, dGDP=dGDP,
        VALU=VALU, x=x, s=s,
        n_collapsed=int((s < 0.5).sum()),
        shock=float(-delta * imports_j),
        imports_j=imports_j,
        total_dx=float(dx.sum()),
        total_dGDP=float(dGDP.sum()),
        share_GDP=float(100.0 * dGDP.sum() / VALU.sum())
                  if VALU.sum() > 0 else 0.0,
        converged=converged,
        sigma=sigma,
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

    # Filter out sectors with negligible impact, then take top_n.
    # argsort[:top_n] alone would pad the chart with zeros when fewer than
    # top_n sectors are actually affected.
    threshold = 1e-3                            # mn USD; ignore noise
    affected_pct = np.where(pct < -1e-6)[0]
    affected_gdp = np.where(dGDP < -threshold)[0]
    order_pct = affected_pct[np.argsort(pct[affected_pct])][:top_n]
    order_gdp = affected_gdp[np.argsort(dGDP[affected_gdp])][:top_n]

    # Gradient colour scales — clamped to the [0, 255] range so they never
    # produce illegal rgba() values when top_n is large.
    n1 = max(len(order_pct), 1)
    n2 = max(len(order_gdp), 1)

    def _reds(k, n):
        # k = 0 (darkest) to k = n-1 (lightest)
        t = k / max(n - 1, 1)
        r = int(round(200 - 100*t))             # 200 → 100
        g = int(round(50 + 100*t))              # 50  → 150
        b = int(round(50 + 100*t))              # 50  → 150
        return f"rgba({r},{g},{b},0.9)"

    def _blues(k, n):
        t = k / max(n - 1, 1)
        r = int(round(50 + 100*t))              # 50  → 150
        g = int(round(100 + 80*t))              # 100 → 180
        b = int(round(200 - 50*t))              # 200 → 150
        return f"rgba({r},{g},{b},0.9)"

    reds  = [_reds(k, n1)  for k in range(n1)]
    blues = [_blues(k, n2) for k in range(n2)]

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
    page_title="ICIO shock simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OECD ICIO — supply-shock simulator")
st.caption(
    "Three models on the same data. **Ghosh** — fixed allocation shares, "
    "free divisibility (*lower bound*). **CES** — calibrated CES "
    "intermediate aggregation with tunable elasticity σ (*the realistic "
    "middle*). **Leontief min-rule** — $x_j = \\min_i (z_{ij}/a_{ij})$, "
    "no substitution (*upper bound*). The CES collapses to Ghosh as "
    "σ → ∞ in the *use* sense, and to Leontief as σ → 0."
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
        "Sector to shock",
        available_sectors,
        index=available_sectors.index(default_sector),
        format_func=lambda s: f"{s} — {SECTOR_NAMES.get(s, s)}",
    )

    model = st.radio(
        "Model",
        options=["Ghosh (linear, lower bound)",
                 "CES intermediate aggregation (calibrated, tunable)",
                 "Leontief min-rule (strict, upper bound)"],
        index=1,
        help=(
            "**Ghosh**: forward propagation through fixed allocation "
            "shares, free output divisibility — *lower bound*.\n\n"
            "**CES**: each sector aggregates its intermediates with "
            "elasticity σ. σ→0 recovers Leontief; σ=1 is Cobb-Douglas; "
            "σ→∞ approaches linear. Calibrated to the observed I-O "
            "table at baseline. The right tool for *playing with "
            "substitutability*.\n\n"
            "**Leontief min-rule**: $x_j = \\min_i (z_{ij}/a_{ij})$. "
            "Strictest possible reading — *upper bound*."
        ),
    )
    is_leontief = model.startswith("Leontief")
    is_ces = model.startswith("CES")

    delta_pct = st.slider(
        "Shock magnitude (% of imports cut)",
        min_value=0, max_value=100, value=30, step=5,
    )
    delta = delta_pct / 100.0

    if is_ces:
        sigma = st.slider(
            "Elasticity of substitution σ",
            min_value=0.0, max_value=5.0, value=0.5, step=0.05,
            help=(
                "How easily sectors can substitute one intermediate for "
                "another.\n\n"
                "• σ = 0   — Leontief (no substitution, min rule)\n"
                "• σ = 0.1–0.5 — short-run energy substitution\n"
                "• σ = 1   — Cobb-Douglas (textbook benchmark)\n"
                "• σ = 1–3 — medium-run, common CGE values\n"
                "• σ → ∞   — linear, arithmetic-mean cascade"
            ),
        )
    else:
        sigma = 0.5  # default value, unused outside CES mode

    top_n = st.slider("Sectors shown in chart", 8, 25, 12)

    st.markdown("---")
    st.caption(
        "All monetary values in USD (millions in the data, billions in the chart). "
        "Source: OECD ICIO, 2022 release."
    )

# --- run the model ---
if is_leontief:
    res = run_leontief_cascade(csv_path, sector, delta)
elif is_ces:
    res = run_ces_cascade(csv_path, sector, delta, sigma=sigma)
else:
    res = run_ghosh_shock(csv_path, sector, delta)
country_name = ISO3_TO_NAME.get(country, country)

# Warn if the chosen sector has no meaningful imports
if res["imports_j"] < 1.0:
    st.warning(
        f"**{country_name}** has essentially no imports of "
        f"{sector} ({SECTOR_NAMES.get(sector, sector)}) — "
        f"the shock has no effect."
    )

# --- headline metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Direct shock",
    f"{res['shock']/1000:,.2f} bn USD",
    delta=f"{-delta_pct}% of {sector} imports",
    delta_color="inverse",
)
m2.metric("Total supply loss",   f"{res['total_dx']/1000:,.2f} bn USD")
m3.metric("Total GDP loss",      f"{res['total_dGDP']/1000:,.2f} bn USD")
m4.metric("Share of GDP",        f"{res['share_GDP']:.3f}%")

# --- chart ---
if is_leontief:
    st.subheader(
        f"{country_name} — Leontief min-rule, "
        f"{delta_pct}% cut to {sector} "
        f"({SECTOR_NAMES.get(sector, sector)}) imports"
    )
elif is_ces:
    st.subheader(
        f"{country_name} — CES (σ = {sigma:.2f}), "
        f"{delta_pct}% cut to {sector} "
        f"({SECTOR_NAMES.get(sector, sector)}) imports"
    )
else:
    st.subheader(
        f"{country_name} — Ghosh propagation, "
        f"{delta_pct}% cut to {sector} "
        f"({SECTOR_NAMES.get(sector, sector)}) imports"
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
        "VA share":            np.where(
            res["x"] > 0, res["VALU"] / res["x"], 0
        ).round(4),
    }).sort_values("Δ GDP (mn USD)")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    if is_leontief:
        suffix = f"leontief_d{delta_pct}"
    elif is_ces:
        suffix = f"ces_s{int(sigma*100):03d}_d{delta_pct}"
    else:
        suffix = f"ghosh_d{delta_pct}"
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name=f"{country}_{sector}_{suffix}.csv",
        mime="text/csv",
    )

# --- methodological notes ---
with st.expander("Method and caveats"):
    st.markdown(r"""
### Ghosh supply-side propagation (lower bound)

Given the intermediate-flow matrix $Z$, total supply
$x_i = \text{OUTPUT}_i + \text{IMPORTS}_i$, and the allocation matrix
$B = \widehat{x}^{-1} Z$, the Ghosh inverse $G=(I-B)^{-1}$
maps a primary-supply shock $\Delta v$ into an output response
$\Delta x' = \Delta v' G$. The diagonal of $G$ exceeds 1 because of
feedback — downstream contraction reduces demand cycling back to the
shocked sector.

**Shock specification.** A fraction $\delta$ of the chosen sector's
imports is removed: $\Delta v_j = -\delta \cdot \text{IMPORTS}_j$. Other
primary inputs are held fixed.

**Why Ghosh is a *lower* bound.** Ghosh treats outputs as continuously
divisible, so a partial shock to one input produces a *partial* drop in
output everywhere — never a binding constraint. In the limit of small
shocks this is fine because production is roughly linear. For larger
shocks the model systematically understates damage because it ignores
the non-linearity of fixed-coefficient production.

### Leontief min-rule (upper bound)

The textbook Leontief production function:

$$x_j = \min_i \left( \frac{z_{ij}}{a_{ij}} \right)$$

Each sector can produce only as much as its scarcest input allows. We
implement this as a fixed-point iteration on sectoral output scaling
$s_j$. Available supply of input $i$ is

$$r_i = \frac{\text{OUTPUT}_i}{x_i} \cdot s_i \;+\; \frac{\text{IMPORTS}_i}{x_i} \cdot (1-\delta_i)$$

where $\delta_i$ is the import-cut fraction (non-zero only for the
shocked sector). Each sector's feasible scaling is then

$$s_j = \min_{i \,:\, a_{ij} > 0} \, r_i$$

Iterate to fixed point. Output and GDP changes follow as
$\Delta x_j = (s_j - 1) \cdot x_j$ and $\Delta\text{GDP}_j = (s_j - 1) \cdot v_j$.

**Why Leontief is an *upper* bound.** No substitution is allowed
between inputs at all. If a sector loses 30% of its B06 supply and B06
is binding, output drops by 30% even if that sector could in reality
substitute coal, electricity, or imports from another origin for crude
at some cost. This is the strictest possible reading of the I-O matrix
as a production technology.

**Numerical detail.** A tiny epsilon ($10^{-6}$) filters out rounding-
artefact $a_{ij}$ values from the input requirements list. The pure
mathematical Leontief result with epsilon = 0 typically collapses to
nearly all sectors at $\delta = 100\%$ because every sector uses some
small amount of energy.

### CES intermediate aggregation (calibrated middle)

The Leontief min-rule is the strictest reading of the I-O table. The
realistic middle ground replaces the min with a CES aggregator over
intermediates:

$$Q_j = \left(\sum_i \gamma_{ij}\, z_{ij}^\eta\right)^{1/\eta}, \quad \eta = \frac{\sigma - 1}{\sigma}$$

where $\sigma$ is the elasticity of substitution. Calibrating to the
observed flows at baseline (where all relative prices equal 1) pins down
the weights:

$$\gamma_{ij} = m_{ij}^{\,1-\eta}, \qquad m_{ij} = \frac{a_{ij}}{\sum_k a_{kj}}$$

i.e. the cost share of input $i$ in sector $j$'s intermediate basket.
Under proportional rationing of available supply $r_i$, the resulting
sectoral output scaling is

$$s_j = \left(\sum_i m_{ij}\, r_i^{\eta}\right)^{1/\eta}$$

iterated to fixed point through the network as before.

**Limits.** As $\sigma \to 0$ the CES collapses to the min function and
recovers the strict Leontief result. At $\sigma = 1$ it is Cobb-Douglas
($s_j = \prod_i r_i^{m_{ij}}$ — geometric mean weighted by cost share).
As $\sigma \to \infty$ it approaches the linear (arithmetic-mean)
aggregator — every sector's output scales with the cost-share-weighted
average of its inputs' availability.

**Choosing σ.** This is where the literature lives and where the
results live or die. Estimated short-run elasticities for energy-economy
substitution are typically in the $\sigma \in [0.1, 0.5]$ range. Medium-
run / long-run textbook values used in CGE studies are $\sigma \in
[0.5, 2.0]$. Energy is famously *less* substitutable than other
intermediates over short horizons, so for an oil-shock scenario the
short-run end of this range is appropriate. **Treat σ as the principal
sensitivity dimension** and report a range, not a point.

### Reading the three models together

$$\text{Ghosh GDP loss} \;\le\; \text{CES GDP loss}(\sigma) \;\le\; \text{Leontief GDP loss}$$

The CES result depends on σ: low-σ approaches Leontief, high-σ falls
toward the bottom of the band. A CGE model with calibrated substitution
elasticities is the right tool to pin down a single σ; this app lets
you see the whole curve. Useful comparisons: Ghosh vs CES at σ=2 tells
you how much of the difference is the *use*-vs-*allocation* framing;
CES at σ=0.3 vs Leontief tells you how much the strict no-substitution
assumption matters.

### GDP translation

$\Delta\text{GDP}_j = \Delta x_j \cdot v_j / x_j$ where $v_j$ is the
sector's value added. Each USD of supply carries on average $v_j/x_j$ of
domestic VA; the rest is intermediates and imported supply. Under
Leontief, since $\Delta x_j / x_j = s_j - 1$, this simplifies to
$\Delta\text{GDP}_j = (s_j - 1) \cdot v_j$.

### Other caveats

1. **Single-country ICIO with competitive imports.** The shock cuts the
   supply origin uniformly across all buyers. For Hormuz-only or
   Russia-only scenarios you need the multi-country ICIO with
   origin-disaggregated import flows.
2. **No price channel.** Both models are pure quantity. For a sustained
   oil shock the price/cost-push effect on inflation, real incomes, and
   competitiveness is typically as large as the quantity effect — and
   would need to be added separately (Leontief price model dual).
3. **No behavioural response.** No SPR releases, no rerouting, no
   monetary or fiscal policy. The numbers are *structural* impact
   estimates.
""")
