import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import (
    simulate_gbm_paths_antithetic,
    monte_carlo_option_price,
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantOpt Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .metric-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 18px 22px;
            border: 1px solid #333355;
        }
        .greek-label { font-size: 0.75rem; color: #aaaacc; text-transform: uppercase; letter-spacing: 0.08em; }
        .greek-value { font-size: 1.5rem; font-weight: 700; color: #e0e0ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar – Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Parameters")

    st.subheader("Market Inputs")
    S0    = st.number_input("Spot Price (S₀)", min_value=1.0,  max_value=10000.0, value=100.0, step=1.0)
    K     = st.number_input("Strike Price (K)", min_value=1.0,  max_value=10000.0, value=100.0, step=1.0)
    r     = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.001, format="%.3f")
    sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0,  value=0.20, step=0.01, format="%.2f")
    T     = st.slider("Time to Maturity (T, years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01, format="%.2f")

    st.subheader("Monte Carlo Settings")
    n_paths = st.select_slider("Number of Paths", options=[500, 1000, 2000, 5000, 10000], value=5000)
    steps   = st.select_slider("Time Steps", options=[50, 100, 252, 500], value=252)
    mc_paths_to_plot = st.slider("Paths to Display in Chart", 5, 50, 20, step=5)

    run_btn = st.button("Run / Update", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("QuantOpt Engine")
st.caption("Black-Scholes Options Pricing & Greeks Analyzer · Monte Carlo Simulation")

# ─────────────────────────────────────────────
# Compute (cache by inputs)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute(S0, K, r, sigma, T, n_paths, steps):
    t, paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths)

    mc_call = monte_carlo_option_price(paths, K, r, T, option_type="call")
    mc_put  = monte_carlo_option_price(paths, K, r, T, option_type="put")

    bs_call = black_scholes_price(S0, K, r, sigma, T, option_type="call")
    bs_put  = black_scholes_price(S0, K, r, sigma, T, option_type="put")

    greeks = {
        "delta_call": black_scholes_delta(S0, K, r, sigma, T, "call"),
        "delta_put":  black_scholes_delta(S0, K, r, sigma, T, "put"),
        "gamma":      black_scholes_gamma(S0, K, r, sigma, T),
        "vega":       black_scholes_vega(S0, K, r, sigma, T),
        "theta_call": black_scholes_theta(S0, K, r, sigma, T, "call"),
        "theta_put":  black_scholes_theta(S0, K, r, sigma, T, "put"),
        "rho_call":   black_scholes_rho(S0, K, r, sigma, T, "call"),
        "rho_put":    black_scholes_rho(S0, K, r, sigma, T, "put"),
    }

    return t, paths, mc_call, mc_put, bs_call, bs_put, greeks


with st.spinner("Running simulation…"):
    t, paths, mc_call, mc_put, bs_call, bs_put, greeks = compute(
        S0, K, r, sigma, T, n_paths, steps
    )

diff_call = mc_call - bs_call
diff_put  = mc_put  - bs_put
moneyness = "ATM" if abs(S0 - K) / K < 0.02 else ("ITM" if S0 > K else "OTM")

# ─────────────────────────────────────────────
# Pricing Cards
# ─────────────────────────────────────────────
st.subheader("Option Prices")
c1, c2, c3, c4 = st.columns(4)
c1.metric("BS Call Price",   f"${bs_call:.4f}")
c2.metric("BS Put Price",    f"${bs_put:.4f}")
c3.metric("MC Call Price",   f"${mc_call:.4f}", delta=f"{diff_call:+.4f} vs BS")
c4.metric("MC Put Price",    f"${mc_put:.4f}",  delta=f"{diff_put:+.4f} vs BS")

# Moneyness badge
st.markdown(
    f"**Moneyness:** `{moneyness}` &nbsp;|&nbsp; "
    f"**Intrinsic (Call):** `${max(S0-K, 0):.2f}` &nbsp;|&nbsp; "
    f"**Intrinsic (Put):** `${max(K-S0, 0):.2f}`"
)

st.divider()

# ─────────────────────────────────────────────
# Greeks Cards
# ─────────────────────────────────────────────
st.subheader("Greeks (Black-Scholes)")
g1, g2, g3, g4, g5, g6, g7, g8 = st.columns(8)
g1.metric("Δ Call",    f"{greeks['delta_call']:.4f}",  help="Call Delta")
g2.metric("Δ Put",     f"{greeks['delta_put']:.4f}",   help="Put Delta")
g3.metric("Γ",         f"{greeks['gamma']:.4f}",        help="Gamma (shared)")
g4.metric("ν (Vega)",  f"{greeks['vega']:.4f}",         help="Vega per 1% vol move")
g5.metric("Θ Call/day",f"{greeks['theta_call']:.4f}",  help="Theta per calendar day")
g6.metric("Θ Put/day", f"{greeks['theta_put']:.4f}",   help="Theta per calendar day")
g7.metric("ρ Call",    f"{greeks['rho_call']:.4f}",    help="Rho per 1% rate move")
g8.metric("ρ Put",     f"{greeks['rho_put']:.4f}",     help="Rho per 1% rate move")

st.divider()

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["GBM Paths", "Greeks Curves", "Price Comparison", "Sensitivity Table"])

# ── Tab 1: GBM paths ─────────────────────────
with tab1:
    fig_paths = go.Figure()
    display_paths = min(mc_paths_to_plot, paths.shape[0])
    colors = px.colors.sequential.Plasma

    for i in range(display_paths):
        color = colors[i % len(colors)]
        fig_paths.add_trace(go.Scatter(
            x=t, y=paths[i],
            mode="lines",
            line=dict(width=0.9, color=color),
            opacity=0.5,
            showlegend=False,
            hovertemplate=f"Path {i+1}<br>S(t)=%{{y:.2f}}<extra></extra>",
        ))

    # Strike line
    fig_paths.add_hline(y=K, line_dash="dash", line_color="red",
                        annotation_text=f"Strike K={K}", annotation_position="top right")

    fig_paths.update_layout(
        title=f"GBM Stock Price Paths (Antithetic Variates) — {display_paths} of {n_paths} shown",
        xaxis_title="Time (years)", yaxis_title="Stock Price",
        template="plotly_dark", height=440,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    st.plotly_chart(fig_paths, use_container_width=True)

# ── Tab 2: Greeks curves ─────────────────────
with tab2:
    stock_range = np.linspace(max(1, S0 * 0.5), S0 * 1.5, 200)
    eps = 1e-9

    delta_call_curve = [black_scholes_delta(s, K, r, sigma, max(T, eps), "call") for s in stock_range]
    delta_put_curve  = [black_scholes_delta(s, K, r, sigma, max(T, eps), "put")  for s in stock_range]
    gamma_curve      = [black_scholes_gamma(s, K, r, sigma, max(T, eps))          for s in stock_range]
    vega_curve       = [black_scholes_vega(s, K, r, sigma, max(T, eps))            for s in stock_range]
    theta_call_curve = [black_scholes_theta(s, K, r, sigma, max(T, eps), "call") for s in stock_range]
    theta_put_curve  = [black_scholes_theta(s, K, r, sigma, max(T, eps), "put")  for s in stock_range]

    fig_greeks = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Delta (Call & Put)", "Gamma", "Vega", "Theta (Call)", "Theta (Put)", "BS Price vs Spot"),
    )

    # Delta
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=delta_call_curve, name="Delta Call", line=dict(color="cyan")), row=1, col=1)
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=delta_put_curve,  name="Delta Put",  line=dict(color="magenta")), row=1, col=1)

    # Gamma
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=gamma_curve, name="Gamma", line=dict(color="yellow")), row=1, col=2)

    # Vega
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=vega_curve, name="Vega", line=dict(color="lime")), row=1, col=3)

    # Theta Call
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=theta_call_curve, name="Theta Call", line=dict(color="orange")), row=2, col=1)

    # Theta Put
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=theta_put_curve, name="Theta Put", line=dict(color="salmon")), row=2, col=2)

    # BS Price vs Spot
    bs_call_curve = [black_scholes_price(s, K, r, sigma, max(T, eps), "call") for s in stock_range]
    bs_put_curve  = [black_scholes_price(s, K, r, sigma, max(T, eps), "put")  for s in stock_range]
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=bs_call_curve, name="BS Call", line=dict(color="deepskyblue")), row=2, col=3)
    fig_greeks.add_trace(go.Scatter(x=stock_range, y=bs_put_curve,  name="BS Put",  line=dict(color="violet")), row=2, col=3)

    # Current spot line in all subplots
    for row in range(1, 3):
        for col in range(1, 4):
            fig_greeks.add_vline(x=S0, line_dash="dot", line_color="white", opacity=0.4, row=row, col=col)

    fig_greeks.update_layout(template="plotly_dark", height=580, showlegend=True,
                              margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig_greeks, use_container_width=True)

# ── Tab 3: Price comparison ───────────────────
with tab3:
    methods = ["Monte Carlo (Antithetic)", "Black-Scholes"]
    call_prices = [mc_call, bs_call]
    put_prices  = [mc_put,  bs_put]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Call Price", x=methods, y=call_prices,
                             marker_color=["#00bcd4", "#0288d1"],
                             text=[f"${v:.4f}" for v in call_prices], textposition="outside"))
    fig_bar.add_trace(go.Bar(name="Put Price",  x=methods, y=put_prices,
                             marker_color=["#e91e8c", "#880e4f"],
                             text=[f"${v:.4f}" for v in put_prices],  textposition="outside"))
    fig_bar.update_layout(
        barmode="group", template="plotly_dark", height=400,
        title="Option Pricing Comparison: Monte Carlo vs Black-Scholes",
        yaxis_title="Price ($)", margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Pricing Error")
        err_df = pd.DataFrame({
            "Type":        ["Call", "Put"],
            "MC Price":    [f"${mc_call:.6f}", f"${mc_put:.6f}"],
            "BS Price":    [f"${bs_call:.6f}", f"${bs_put:.6f}"],
            "Abs Error":   [f"{abs(diff_call):.6f}", f"{abs(diff_put):.6f}"],
            "Rel Error %": [f"{100*diff_call/bs_call:.3f}%", f"{100*diff_put/bs_put:.3f}%"],
        })
        st.dataframe(err_df, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("#### Hedging Interpretation")
        st.info(
            f"**Delta (Call):** {greeks['delta_call']:.4f}  \n"
            f"Hedge 100 calls → short **{100*greeks['delta_call']:.0f} shares**\n\n"
            f"**Delta (Put):** {greeks['delta_put']:.4f}  \n"
            f"Hedge 100 puts → long **{100*abs(greeks['delta_put']):.0f} shares**\n\n"
            f"**Gamma:** {greeks['gamma']:.6f}  \n"
            f"Delta shifts by **{greeks['gamma']:.6f}** per $1 stock move"
        )

# ── Tab 4: Sensitivity table ──────────────────
with tab4:
    st.markdown("### Sensitivity to Spot Price")
    spot_range = np.linspace(max(1, S0 * 0.7), S0 * 1.3, 13)
    rows = []
    for s in spot_range:
        rows.append({
            "Spot (S)":    f"${s:.2f}",
            "BS Call":     f"${black_scholes_price(s, K, r, sigma, T, 'call'):.4f}",
            "BS Put":      f"${black_scholes_price(s, K, r, sigma, T, 'put'):.4f}",
            "Δ Call":      f"{black_scholes_delta(s, K, r, sigma, T, 'call'):.4f}",
            "Δ Put":       f"{black_scholes_delta(s, K, r, sigma, T, 'put'):.4f}",
            "Γ":           f"{black_scholes_gamma(s, K, r, sigma, T):.6f}",
            "Vega":        f"{black_scholes_vega(s, K, r, sigma, T):.4f}",
            "Θ Call/day":  f"{black_scholes_theta(s, K, r, sigma, T, 'call'):.4f}",
        })
    sens_df = pd.DataFrame(rows)
    st.dataframe(sens_df, hide_index=True, use_container_width=True)

    st.markdown("### Sensitivity to Volatility")
    vol_range = np.arange(0.05, 0.81, 0.05)
    vol_rows = []
    for v in vol_range:
        vol_rows.append({
            "Volatility (σ)": f"{v:.0%}",
            "BS Call":        f"${black_scholes_price(S0, K, r, v, T, 'call'):.4f}",
            "BS Put":         f"${black_scholes_price(S0, K, r, v, T, 'put'):.4f}",
            "Vega":           f"{black_scholes_vega(S0, K, r, v, T):.4f}",
            "Δ Call":         f"{black_scholes_delta(S0, K, r, v, T, 'call'):.4f}",
            "Γ":              f"{black_scholes_gamma(S0, K, r, v, T):.6f}",
        })
    st.dataframe(pd.DataFrame(vol_rows), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "QuantOpt Engine · Black-Scholes + Monte Carlo · "
    "Greeks: Δ Delta · Γ Gamma · ν Vega · Θ Theta · ρ Rho"
)
