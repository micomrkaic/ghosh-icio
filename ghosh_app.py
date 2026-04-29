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
def run_hem_extraction(csv_path: str, sector_code: str, threshold: float) -> dict:
    """
    Hypothetical-Extraction cascade with a Leontief criticality threshold.

    Asks: "If sector j is removed entirely, which other sectors must shut
    down because their dependence on j (or on something that itself
    depends on j) exceeds τ of their input cost?"

    Algorithm
    ---------
    1.  Build technical coefficients A = Z / x_j (column-wise share).
    2.  Mark j_shock as dead.
    3.  Repeat until stable: any sector with a dead supplier whose input
        coefficient exceeds τ also dies.
    4.  Dead sectors lose 100% of their output and value added; other
        sectors are assumed to substitute away (no smooth Ghosh-style
        partial loss).

    The threshold τ is the substitutability cushion. τ → 0 means every
    input is essential (catastrophic upper bound). τ → ∞ means only the
    shocked sector itself dies (trivial bound). Realistic energy-economy
    values are 1–10%.

    Returns a dict matching the Ghosh-output schema so the chart and table
    rendering can be reused.
    """
    c = load_country(csv_path)
    sectors, Z, OUTPUT, IMPO, VALU = (
        c["sectors"], c["Z"], c["OUTPUT"], c["IMPO"], c["VALU"]
    )
    n = len(sectors)
    x = OUTPUT - IMPO
    safe_x = np.where(x > 0, x, 1.0)
    A = Z / safe_x[None, :]      # technical coefficients (column-wise)

    j = sectors.index(sector_code)
    dead = np.zeros(n, dtype=bool)
    dead[j] = True

    # Iterative cascade
    changed = True
    while changed:
        changed = False
        for jj in range(n):
            if dead[jj]:
                continue
            # If any dead sector supplies > τ of jj's inputs, jj dies
            if np.any(dead & (A[:, jj] > threshold)):
                dead[jj] = True
                changed = True

    # Dead sectors lose 100% of their supply and VA
    dx = np.where(dead, -x, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_supply = np.where(dead, -100.0, 0.0)
    dGDP = np.where(dead, -VALU, 0.0)

    return dict(
        sectors=sectors,
        dx=dx, pct_supply=pct_supply, dGDP=dGDP,
        VALU=VALU, x=x,
        dead=dead,
        n_killed=int(dead.sum()),
        shock=float(-x[j]),                 # full removal of shocked sector
        imports_j=float(-IMPO[j]) if IMPO[j] < 0 else 0.0,
        total_dx=float(dx.sum()),
        total_dGDP=float(dGDP.sum()),
        share_GDP=float(100.0 * dGDP.sum() / VALU.sum())
                  if VALU.sum() > 0 else 0.0,
    )


@st.cache_data(show_spinner=False)
def run_hem_extraction(csv_path: str, sector_code: str,
                       threshold: float = 0.05) -> dict:
    """
    Hypothetical Extraction via supply-cascade Leontief.

    Why supply-cascade rather than textbook HEM
    -------------------------------------------
    The textbook total-HEM formula (zero out row & column of A, recompute
    Leontief) measures only **backward linkages**: the GDP that vanishes
    if sector j had never bought any inputs from anyone. For an importing
    country, this is small for B06 because Germany's domestic oil VA is
    tiny — almost all is foreign value added passing through.

    The user's actual intuition — "no oil = catastrophic GDP loss" — is a
    **forward-linkage** question: which sectors **structurally depend on
    sector j as an input**? A strict Leontief reading says: any sector
    with a non-zero technical coefficient a[j,i] > 0 cannot operate
    without j. That cascades to nearly the entire economy because every
    sector uses *some* energy, however small.

    Compromise — cascade with criticality threshold
    -----------------------------------------------
    A sector i shuts down if its input share from a dead sector exceeds
    a `threshold` (e.g., 5%). Sectors that use only a trace amount of
    the missing input are assumed to substitute or absorb the loss.

    The threshold is a tunable parameter:
      * threshold ≈ 0.005–0.02 → catastrophic cascade (everything dies)
      * threshold ≈ 0.05      → energy-dependent core (~5% of GDP for oil)
      * threshold ≈ 0.10      → only the most directly dependent sectors

    This is a heuristic upper bound, not a CGE prediction. It captures
    structural dependency under fixed Leontief inputs without claiming
    the result is a literal forecast.
    """
    c = load_country(csv_path)
    sectors, Z, IMPO, OUTPUT, VALU = (
        c["sectors"], c["Z"], c["IMPO"], c["OUTPUT"], c["VALU"]
    )
    n = len(sectors)
    x = OUTPUT - IMPO

    safe_x = np.where(x > 0, x, 1.0)
    # Leontief technical coefficients: a[i,k] = inputs from i per unit output of k
    A = Z / safe_x[None, :]

    j = sectors.index(sector_code)

    # Iterative cascade: a sector dies if any of its critical inputs is dead
    alive = np.ones(n, dtype=bool)
    alive[j] = False
    for _ in range(n + 1):
        new_alive = alive.copy()
        for i in range(n):
            if not new_alive[i]:
                continue
            for k in range(n):
                if not alive[k] and A[k, i] > threshold:
                    new_alive[i] = False
                    break
        if np.array_equal(new_alive, alive):
            break
        alive = new_alive

    # Output of dead sectors goes to zero; alive sectors unchanged
    dx = np.where(alive, 0.0, -x)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_supply = np.where(x > 0, 100.0 * dx / x, 0.0)
        dGDP = np.where(x > 0, dx * VALU / x, 0.0)

    return dict(
        sectors=sectors,
        dx=dx, pct_supply=pct_supply, dGDP=dGDP,
        VALU=VALU, x=x,
        shock=-x[j], imports_j=-IMPO[j],
        n_killed=int((~alive).sum()),
        threshold=threshold,
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
    page_title="ICIO shock simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OECD ICIO — supply-shock simulator")
st.caption(
    "Two complementary models on the same data. "
    "**Ghosh** propagates a partial shock through fixed allocation shares — "
    "*lower bound* on GDP loss. "
    "**HEM cascade** removes the sector entirely under strict Leontief "
    "input requirements — *upper bound*."
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
        options=["Ghosh (partial shock, lower bound)",
                 "HEM cascade (full extraction, upper bound)"],
        index=0,
        help=(
            "**Ghosh**: cuts a fraction of the sector's *imports* and propagates "
            "linearly via fixed allocation shares. Free output divisibility — "
            "smooth losses, no shutdowns. *Lower bound* on GDP loss.\n\n"
            "**HEM cascade**: removes the sector entirely; downstream sectors "
            "shut down if their input dependency exceeds the threshold. Strict "
            "Leontief — catches non-linear cascades. *Upper bound* on GDP loss."
        ),
    )
    is_hem = model.startswith("HEM")

    if is_hem:
        threshold_pct = st.slider(
            "Criticality threshold (% input share)",
            min_value=1.0, max_value=20.0, value=5.0, step=0.5,
            help=(
                "A sector shuts down if it depends on a dead sector for more "
                "than this share of its inputs. Lower threshold → more "
                "cascading shutdowns. Try 1% for catastrophe, 5% for "
                "energy-dependent core, 10% for only the most directly "
                "dependent sectors."
            ),
        )
        threshold = threshold_pct / 100.0
        delta_pct = 100
        delta = 1.0
    else:
        delta_pct = st.slider(
            "Shock magnitude (% of imports cut)",
            min_value=0, max_value=100, value=30, step=5,
        )
        delta = delta_pct / 100.0
        threshold = 0.05  # placeholder, unused

    top_n = st.slider("Sectors shown in chart", 8, 25, 12)

    st.markdown("---")
    st.caption(
        "All monetary values in USD (millions in the data, billions in the chart). "
        "Source: OECD ICIO, 2022 release."
    )

# --- run the model ---
if is_hem:
    res = run_hem_extraction(csv_path, sector, threshold=threshold)
else:
    res = run_ghosh_shock(csv_path, sector, delta)
country_name = ISO3_TO_NAME.get(country, country)

# Warn if Ghosh sector has no meaningful imports
if not is_hem and res["imports_j"] < 1.0:
    st.warning(
        f"**{country_name}** has essentially no imports of "
        f"{sector} ({SECTOR_NAMES.get(sector, sector)}) — "
        f"the shock has no effect. Try a different sector or country."
    )

# --- headline metrics ---
m1, m2, m3, m4 = st.columns(4)
if is_hem:
    m1.metric(
        "Sector extracted",
        f"{res['shock']/1000:,.2f} bn USD",
        delta=f"{res['n_killed']} sectors die",
        delta_color="inverse",
    )
else:
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
if is_hem:
    st.subheader(
        f"{country_name} — HEM cascade extraction of "
        f"{sector} ({SECTOR_NAMES.get(sector, sector)}), "
        f"threshold = {threshold_pct:.1f}%"
    )
else:
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
    suffix = f"hem_thr{int(threshold_pct*10)}" if is_hem else f"ghosh_d{delta_pct}"
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
Given the intermediate-flow matrix $Z$, total supply $x_i = \text{OUTPUT}_i + \text{IMPORTS}_i$,
and the allocation matrix $B = \widehat{x}^{-1} Z$, the Ghosh inverse $G=(I-B)^{-1}$
maps a primary-supply shock $\Delta v$ into an output response
$\Delta x' = \Delta v' G$. A fraction $\delta$ of the chosen sector's imports
is removed: $\Delta v_j = -\delta \cdot \text{IMPORTS}_j$.

Why this is a *lower bound*: Ghosh assumes fixed allocation shares — buyers
absorb whatever supply arrives in fixed proportions and continue producing
scaled-down output. There is no threshold ("no oil → no refining") and no
cascading shutdowns. For small shocks (a few percent) this is a fine local
approximation; for large shocks it understates because real production
functions are sharply non-linear at the boundaries.

### HEM cascade (upper bound)
Build the demand-side technical-coefficient matrix $A = Z\widehat{x}^{-1}$
(column-wise division) where $a_{ij}$ is the input from sector $i$ needed per
unit of sector $j$'s output.

Run an iterative cascade. Sector $j$ is killed (output = 0). At each step,
any *alive* sector $i$ that has $a_{kj} > \tau$ for some *dead* sector $k$
is killed too, where $\tau$ is the criticality threshold. Iterate to fixed
point.

The threshold $\tau$ matters a lot:

- $\tau \to 0$ — every sector with any oil dependency dies. For modern economies
  this kills 95–99% of GDP because every sector uses some energy.
- $\tau \approx 0.05$ — the "structurally oil-dependent core" emerges:
  refining, petrochemicals, electricity, transport. ~5% of GDP for an
  oil-importing economy.
- $\tau \approx 0.10$ — only sectors with $\geq$10% input dependency die.
  Typically a smaller cascade.

This is a *heuristic* upper bound, not a CGE prediction. It captures
structural dependency under fixed Leontief inputs without claiming the
result is a literal forecast.

### Why textbook HEM isn't used here
The standard total-HEM formula (zero out row & column of $A$, recompute
$L$) measures only **backward linkages** — the GDP that vanishes if
sector $j$ had never *demanded* inputs. For an importing country this is
small for B06 because Germany's domestic oil VA is tiny; almost all is
foreign value added. The user's intuition ("no oil = catastrophe") is a
**forward-linkage** question, which the cascade implementation answers.

### GDP translation (both models)
$\Delta\text{GDP}_j = \Delta x_j \cdot v_j / x_j$ where $v_j$ is the sector's
domestic value added. Each USD of supply carries $v_j/x_j$ of domestic VA
on average; the rest is intermediates and imported supply.

### Reading the bounds together
$$\text{Ghosh GDP loss} \;\le\; \text{realised GDP loss} \;\le\; \text{HEM cascade GDP loss}$$

The realistic medium-run estimate for a sustained 100% oil disruption
likely sits closer to HEM than to Ghosh — perhaps half-way — because
adjustment costs, cascading shutdowns, and price spillovers dominate over
quarters. A CGE model with non-zero substitution elasticities is the
right tool to pin down where between the bounds the truth actually sits.

### Other caveats
1. Single-country ICIO with competitive imports — the shock cuts the
   supply origin uniformly across all buyers; it can't tell you whose
   firm gets less.
2. No price channel, no SPR releases, no monetary or fiscal response.
3. Total imports are cut, not just the share routed via a particular
   geography. Hormuz-only / Russia-only requires the multi-country ICIO.
""")
