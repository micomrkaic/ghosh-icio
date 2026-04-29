# OECD ICIO Supply-Shock Simulator

A Streamlit app for exploring how supply-side shocks propagate through OECD inter-country input–output (ICIO) tables. Pick a country, a sector, and a shock magnitude — for example, *"cut 30% of Germany's imports of crude oil & natural gas"* — and the app reports the sectoral and aggregate GDP impact under three complementary models that bracket the realistic range of outcomes.

## What the app does

- Auto-discovers every country CSV in `data/` (filename pattern `XXX2022ttl.csv`, where `XXX` is the ISO-3166 alpha-3 code).
- Lets you choose a country, a sector to shock, a shock magnitude δ ∈ [0, 100%], and a model.
- Three production models on the same data:
  - **Ghosh** — forward propagation through fixed allocation shares (lower bound).
  - **CES intermediate aggregation** — calibrated CES with tunable elasticity σ (the realistic middle).
  - **Leontief min-rule** — strict $x_j = \min_i (z_{ij}/a_{ij})$, no substitution (upper bound).
- Reports sectoral supply contraction, sectoral GDP loss, and aggregate GDP impact as a share of national GDP.
- Interactive Plotly charts with full sector-name labels.
- Downloadable CSV of the detailed sectoral results.

## The models

### Notation

| Symbol | Definition |
|---|---|
| $Z$ | 50×50 intermediate-flow matrix; $Z_{ij}$ = sales from sector $i$ to sector $j$ |
| $x$ | Total supply: $x_i = \text{OUTPUT}_i + \text{IMPORTS}_i$ |
| $A$ | Technical coefficients: $A_{ij} = Z_{ij}/x_j$ (input share, column-wise) |
| $B$ | Allocation matrix: $B_{ij} = Z_{ij}/x_i$ (output share, row-wise) |
| $m_{ij}$ | Normalised cost shares: $m_{ij} = a_{ij} / \sum_k a_{kj}$ |
| $v_j$ | Sectoral value added |

### Ghosh propagation (lower bound)

The Ghosh inverse $G = (I - B)^{-1}$ maps a primary-supply shock into an output response:

$$x' = v'\, G, \qquad \Delta v_j = -\delta \cdot \text{IMPORTS}_j, \qquad \Delta x' = \Delta v'\, G$$

Output is treated as continuously divisible — a 30% cut to one input produces a *partial* drop in output everywhere, never a binding constraint. This is why Ghosh is a lower bound on the GDP impact: it ignores the non-linearity of fixed-coefficient production.

### Leontief min-rule (upper bound)

The textbook Leontief production function:

$$x_j = \min_i \left( \frac{z_{ij}}{a_{ij}} \right)$$

Each sector can produce only as much as its scarcest input allows. Implemented as a fixed-point iteration on output scaling $s_j$, where input availability cascades through the network. No substitution is allowed: if a sector loses 30% of its B06 supply and B06 is binding, output drops 30% even if coal, electricity, or alternative imports could in reality fill the gap. This is the strictest possible reading of the I-O matrix as a production technology.

### CES intermediate aggregation (calibrated middle)

Each sector's output is bounded by a CES aggregator over its intermediate inputs:

$$Q_j = \left(\sum_i \gamma_{ij}\, z_{ij}^{\eta}\right)^{1/\eta}, \qquad \eta = \frac{\sigma - 1}{\sigma}$$

Calibrating to the observed I-O flows at baseline (where all relative prices equal 1) pins down the weights as $\gamma_{ij} = m_{ij}^{1-\eta}$. Under proportional rationing of available supply $r_i$, the resulting output scaling becomes

$$s_j = \left(\sum_i m_{ij}\, r_i^{\eta}\right)^{1/\eta}$$

iterated through the network to fixed point.

**Limits.** As $\sigma \to 0$ this collapses to the Leontief min function. At $\sigma = 1$ it is Cobb-Douglas ($s_j = \prod_i r_i^{m_{ij}}$). As $\sigma \to \infty$ it approaches the linear arithmetic-mean aggregator.

**Choosing σ.** Estimated short-run elasticities for energy-economy substitution are typically $\sigma \in [0.1, 0.5]$. Medium-run / long-run textbook values used in CGE studies are $\sigma \in [0.5, 2.0]$. Energy is famously *less* substitutable than other intermediates over short horizons, so for an oil-shock scenario the short-run end of this range is appropriate. Treat σ as the principal sensitivity dimension and report a range, not a point.

### Reading the three models together

$$\text{Ghosh GDP loss} \;\le\; \text{CES GDP loss}(\sigma) \;\le\; \text{Leontief GDP loss}$$

The CES result depends on σ: low-σ approaches Leontief, high-σ falls toward the bottom of the band. A full CGE model with calibrated substitution elasticities is the right tool to pin down a single σ; this app lets you see the whole curve. Useful comparisons: Ghosh vs CES at σ=2 isolates the *use*-vs-*allocation* framing difference; CES at σ=0.3 vs Leontief shows how much the strict no-substitution assumption matters.

### GDP translation

For all three models, sectoral GDP loss is

$$\Delta\text{GDP}_j = \Delta x_j \cdot \frac{v_j}{x_j}$$

Each USD of supply carries on average $v_j/x_j$ of domestic value added; the rest is intermediates and imported supply. For Leontief and CES, since $\Delta x_j / x_j = s_j - 1$, this simplifies to $\Delta\text{GDP}_j = (s_j - 1) \cdot v_j$.

## Data

The app expects **OECD ICIO single-country "ttl" tables**, one CSV per country, named `XXX2022ttl.csv` where `XXX` is the ISO-3166-1 alpha-3 country code (e.g. `DEU2022ttl.csv`, `JPN2022ttl.csv`).

Each CSV contains a 50×50 intermediate-flow matrix, eight final-demand columns (HFCE, NPISH, GGFC, GFCF, INVNT, DPABR, CONS_NONRES, EXPO), an IMPO column (negative entries = imports), and primary-input rows including VALU (value added) and OUTPUT.

Monetary values are in **USD millions, current prices**. The release is publicly available from the [OECD ICIO data portal](https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html).

## Setup

```bash
git clone https://github.com/<your-username>/ghosh-icio.git
cd ghosh-icio
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
streamlit run ghosh_app.py
```

The app opens at `http://localhost:8501`.

## Repository layout

```
ghosh-icio/
├── ghosh_app.py        # Streamlit app (UI + all three models)
├── requirements.txt    # Python dependencies
├── .gitignore
├── README.md
└── data/
    ├── DEU2022ttl.csv
    ├── FRA2022ttl.csv
    └── ...             # one CSV per country
```

## Deploying

Compatible with Streamlit Community Cloud — push the repo to GitHub, connect it at [share.streamlit.io](https://share.streamlit.io), and point at `ghosh_app.py`. First build takes 2–3 minutes; subsequent pushes redeploy in ~30 seconds.

If your data licensing prevents public redistribution, make the GitHub repo private — Streamlit Community Cloud supports private repos on the free tier.

## Worked example: Germany, 30% cut to oil & gas imports

| Model | σ | ΔGDP (USD bn) | % of GDP |
|---|:-:|---:|---:|
| Ghosh | — | −13.5 | −0.37% |
| CES | 5.0 | −105 | −2.85% |
| CES | 1.0 (Cobb-Douglas) | −116 | −3.14% |
| CES | 0.3 (short-run energy) | −153 | −4.15% |
| CES | 0.1 | −297 | −8.03% |
| Leontief (σ → 0) | 0 | −1,105 | −29.93% |

For a sustained Hormuz-type disruption the σ ∈ [0.2, 0.5] range is the defensible policy answer: **3–5% of German GDP**, or roughly **$110–180 bn**. That's an order of magnitude above Ghosh's frictionless lower bound and an order of magnitude below the Leontief catastrophe — and it's where IMF/IEA scenario reports typically land.

## Caveats

The models are deliberately compact and the results should be read in that light.

1. **Single-country, competitive imports.** The OECD ICIO single-country tables aggregate imported intermediates with domestic flows. The shock cuts the supply origin uniformly across all downstream buyers; the model cannot tell you whose firm gets less. For origin-specific scenarios (e.g. Hormuz-routed imports only), you need the multi-country ICIO with origin-disaggregated flows.

2. **Pure quantity model.** No price channel, no terms-of-trade adjustment, no inventory or strategic-reserve release, no monetary or fiscal response. For sustained shocks the price/cost-push effect on inflation, real wages, and competitiveness is typically as large as the quantity effect — and would need to be added separately (Leontief price model dual or full CGE).

3. **Why GDP loss < import shock for Ghosh.** This often surprises new users. The dollars of "lost imports" are mostly *foreign* value added passing through the domestic supply chain — cutting them removes foreign GDP, not domestic GDP. Domestic GDP is lost via the *downstream* sectors that can no longer operate at full output, weighted by their domestic value-added shares. This is the same logic underlying the Koopman–Wang–Wei trade-in-value-added (TiVA) decomposition: gross trade flows systematically overstate the domestic GDP at stake when those flows are interrupted.

4. **Why Leontief is near-linear for energy shocks.** When the shocked input is universally used (energy, electricity, retail), it is the binding constraint for nearly every sector under strict Leontief. A δ% cut in supply then produces a δ% cut in (almost) all sectoral output — hence in GDP. This is a feature of the min-rule applied to a network with one universally-used input, not a bug.

5. **σ is where the action is.** A CES result for a given country/shock can vary by a factor of 3 depending on whether you use Bayesian-estimated short-run elasticities (~0.2) or long-run textbook values (~2.0). The model architecture is standardised; the elasticities are not. Whatever you report, treat σ as the principal sensitivity dimension.

## References

- Ghosh, A. (1958). *Input–Output Approach in an Allocation System*. Economica, 25(97), 58–64.
- Oosterhaven, J. (1988). *On the plausibility of the supply-driven input–output model*. Journal of Regional Science, 28(2), 203–217.
- Oosterhaven, J. (2012). *Adding supply-driven consumption makes the Ghosh model even more implausible*. Economic Systems Research, 24(1), 101–111.
- Koopman, R., Wang, Z., & Wei, S.-J. (2014). *Tracing value-added and double counting in gross exports*. American Economic Review, 104(2), 459–494.
- Lofgren, H., Harris, R., & Robinson, S. (2002). *A Standard Computable General Equilibrium (CGE) Model in GAMS*. IFPRI.
- Hosoe, N., Gasawa, K., & Hashimoto, H. (2010). *Textbook of Computable General Equilibrium Modelling: Programming and Simulations*. Palgrave Macmillan.
- OECD (2024). *Inter-Country Input-Output (ICIO) Tables*. OECD Publishing, Paris.

## License

Code released under the MIT License. OECD ICIO data is © OECD and redistributed under the OECD terms of use; verify your specific license before publishing the data alongside this app.
