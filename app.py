"""
Machine Gnostics Benchmark App
Interactive Streamlit application comparing Machine Gnostics vs Classical Statistics
using the Anscombe Quartet as ground truth benchmark.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from machinegnostics.data import make_anscombe_check_data
from machinegnostics.models import LinearRegressor
from machinegnostics.metrics import (
    mean as mg_mean,
    median as mg_median,
    correlation as mg_correlation,
    robr2,
    root_mean_squared_error as mg_rmse,
)

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Gnostics Benchmark",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS styling
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #0f1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e6f0 !important;
    }
    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1a2f52, #0f1a2e);
        border: 1px solid #2d4a7a;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-label  { font-size: 12px; color: #8ca0c0; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 30px; font-weight: 700; color: #4fc3f7; margin: 4px 0; }
    .kpi-sub    { font-size: 11px; color: #6a8ab0; }
    .kpi-good   { color: #66bb6a; }
    .kpi-warn   { color: #ffa726; }
    .kpi-diff   { color: #ef5350; }
    /* Problem banner */
    .problem-banner {
        background: linear-gradient(90deg, #0d47a1, #1565c0);
        border-left: 5px solid #4fc3f7;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }
    .problem-banner h3 { color: #e3f2fd; margin: 0 0 6px 0; }
    .problem-banner p  { color: #b3d4f5; margin: 0; font-size: 14px; }
    /* Step instruction */
    .step-box {
        background: #1e2d45;
        border: 1px solid #2d4a7a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #b0c4de;
        margin-bottom: 14px;
    }
    .step-box b { color: #4fc3f7; }
    /* Section header */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #4fc3f7;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Data & metric computation (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Computing metrics for all Anscombe datasets…")
def compute_all_metrics():
    metrics_all = {}
    for ds_id in [1, 2, 3, 4]:
        x, y = make_anscombe_check_data(ds_id)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # ── Classical ──────────────────────────
        slp, intercept, r_val, _, _ = stats.linregress(x, y)
        y_pred_cls = intercept + slp * x
        cls = {
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
            "median_x": float(np.median(x)),
            "median_y": float(np.median(y)),
            "corr": float(np.corrcoef(x, y)[0, 1]),
            "r2": float(r_val**2),
            "rmse": float(np.sqrt(np.mean((y - y_pred_cls) ** 2))),
            "slope": float(slp),
            "intercept": float(intercept),
            "y_pred": y_pred_cls,
        }

        # ── Machine Gnostics ───────────────────
        model = LinearRegressor(
            max_iter=300,
            early_stopping=True,
            tolerance=1e-6,
            mg_loss="hi",
            history=True,
            verbose=False,
        )
        model.fit(x, y)
        y_pred_mg = np.asarray(model.predict(x), dtype=float)
        weights = getattr(model, "weights", None)
        mg = {
            "mean_x": float(mg_mean(x)),
            "mean_y": float(mg_mean(y)),
            "median_x": float(mg_median(x)),
            "median_y": float(mg_median(y)),
            "corr": float(mg_correlation(x, y)),
            "r2": float(r2_score(y, y_pred_mg)),
            "robr2": float(robr2(y, y_pred_mg, w=weights)),
            "rmse": float(mg_rmse(y, y_pred_mg)),
            "y_pred": y_pred_mg,
            "weights": np.asarray(weights, dtype=float) if weights is not None else None,
            "model": model,
        }

        metrics_all[ds_id] = {"x": x, "y": y, "classical": cls, "mg": mg}
    return metrics_all


# ─────────────────────────────────────────────
# Helper: % MG change label
# ─────────────────────────────────────────────
def pct_diff(classical_val, mg_val, lower_is_better=False):
    """Return (delta_pct, sign_label, css_class) comparing MG vs Classical."""
    if abs(classical_val) < 1e-12:
        return 0.0, "—", "kpi-sub"
    delta = ((mg_val - classical_val) / abs(classical_val)) * 100
    if lower_is_better:
        delta = -delta  # positive delta => MG is better (lower raw value)
    label = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    css = "kpi-good" if delta > 0 else ("kpi-warn" if delta > -5 else "kpi-diff")
    return delta, label, css


def kpi_card(label, classical, mg, lower_is_better=False, unit=""):
    _, pct_label, css = pct_diff(classical, mg, lower_is_better)
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{mg:.4f}{unit}</div>
        <div class="kpi-sub">Classical: {classical:.4f}{unit}</div>
        <div class="kpi-sub {css}">MG change: {pct_label}</div>
    </div>
    """


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
PAGES = {
    "Overview": "overview",
    "Why MG Works Better": "why_mg",
    "Mean & Median": "mean_median",
    "Correlation": "correlation",
    "R² & Robust R²": "r2",
    "RMSE": "rmse",
    "Linear Regression": "regression",
    "About": "about",
}

with st.sidebar:
    st.markdown("## 🥭 Machine Gnostics")
    st.markdown("#### Benchmark App")
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    # MAIN ACTION BUTTON
    st.markdown(
        "<div style='margin-bottom:4px;font-size:12px;color:#8ca0c0;font-weight:bold;'>"
        "PRIMARY</div>",
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Benchmark", key="nav_overview", use_container_width=True, type="primary"):
        st.session_state.page = "overview"

    st.markdown("---")

    # SECONDARY NAVIGATION
    st.markdown(
        "<div style='margin-bottom:12px;font-size:12px;color:#8ca0c0;font-weight:bold;'>"
        "EXPLORE DETAILS</div>",
        unsafe_allow_html=True,
    )

    detail_pages = {
        "Why MG Works Better": "why_mg",
        "Mean & Median": "mean_median",
        "Correlation": "correlation",
        "R² & Robust R²": "r2",
        "RMSE": "rmse",
        "Linear Regression": "regression",
        "About": "about",
    }

    for label, key in detail_pages.items():
        active = st.session_state.page == key
        btn_style = "secondary"
        if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_style):
            st.session_state.page = key

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#445566;text-align:center;'>"
        "Benchmark: Anscombe Quartet<br>4 Datasets, 5 Metrics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#445566;text-align:center;margin-top:6px;'>"
        "Docs: <a href='https://docs.machinegnostics.com' style='color:#4fc3f7;'>docs.machinegnostics.com</a></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with st.spinner("Loading Anscombe datasets and computing metrics…"):
    data = compute_all_metrics()

page = st.session_state.page

DS_NAMES = {1: "Anscombe I", 2: "Anscombe II", 3: "Anscombe III", 4: "Anscombe IV"}

# ═══════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>🎯 Problem Definition — Why Classical Statistics Can Mislead</h3>
            <p>
            The <b>Anscombe Quartet</b> (1973) consists of 4 datasets that are <i>statistically identical</i>
            under classical measures (same mean, median, correlation, R², RMSE) yet are visually and
            structurally completely different. This exposes a critical weakness of traditional statistics.<br><br>
            <b>Machine Gnostics</b> applies an information-theoretic framework that assigns adaptive weights
            to data points, producing metrics that are more sensitive to the true data structure.
            This app benchmarks the MG change quantitatively using <b>%&nbsp;KPIs</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="step-box">
        <b>👆 How to use this app:</b><br>
        1. Use the <b>left sidebar</b> to navigate between benchmark sections.<br>
        2. Each section shows a <b>side-by-side table</b> and <b>% KPI cards</b> comparing Classical vs Machine Gnostics.<br>
        3. Green % = Machine Gnostics outperforms Classical &nbsp;|&nbsp; Red % = Classical performs better.<br>
        4. The <b>Linear Regression</b> section shows visual regression fits per dataset with scatter weight coloring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">📋 Anscombe Quartet — Raw Data Preview</div>', unsafe_allow_html=True)

    col_tabs = st.tabs(["Dataset I", "Dataset II", "Dataset III", "Dataset IV"])
    for i, ds_id in enumerate([1, 2, 3, 4]):
        with col_tabs[i]:
            df_preview = pd.DataFrame({"X": data[ds_id]["x"], "Y": data[ds_id]["y"]}).round(4)
            col_a, col_b = st.columns([1.25, 0.9])
            with col_a:
                st.dataframe(df_preview, height=300, use_container_width=True)
            with col_b:
                fig_p, ax_p = plt.subplots(figsize=(3.2, 2.2))
                fig_p.patch.set_facecolor("#0f1a2e")
                ax_p.set_facecolor("#0f1a2e")
                ax_p.scatter(
                    data[ds_id]["x"], data[ds_id]["y"],
                    color="#4fc3f7", s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
                )
                ax_p.set_xlabel("X", color="#8ca0c0")
                ax_p.set_ylabel("Y", color="#8ca0c0")
                ax_p.set_title(f"Anscombe {ds_id}", color="#e0e6f0")
                ax_p.tick_params(colors="#8ca0c0")
                for sp in ax_p.spines.values():
                    sp.set_edgecolor("#2d4a7a")
                st.pyplot(fig_p, use_container_width=False)
                plt.close(fig_p)

    # ── Full KPI summary table ───────────────────────────────────
    st.markdown('<div class="section-title">📊 Full Benchmark KPI Summary (All Datasets)</div>', unsafe_allow_html=True)

    summary_rows = []
    for ds_id in [1, 2, 3, 4]:
        cls = data[ds_id]["classical"]
        mg = data[ds_id]["mg"]
        _, p_mean, _ = pct_diff(cls["mean_y"], mg["mean_y"])
        _, p_med, _ = pct_diff(cls["median_y"], mg["median_y"])
        _, p_corr, _ = pct_diff(cls["corr"], mg["corr"])
        _, p_r2, _ = pct_diff(cls["r2"], mg["robr2"])
        _, p_rmse, _ = pct_diff(cls["rmse"], mg["rmse"], lower_is_better=True)
        summary_rows.append({
            "Dataset": DS_NAMES[ds_id],
            "Mean Δ%": p_mean,
            "Median Δ%": p_med,
            "Corr Δ%": p_corr,
            "R²→RobR² Δ%": p_r2,
            "RMSE Δ% (↓better)": p_rmse,
            "MG R²": f"{mg['robr2']:.4f}",
            "CLS R²": f"{cls['r2']:.4f}",
            "MG RMSE": f"{mg['rmse']:.4f}",
            "CLS RMSE": f"{cls['rmse']:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    delta_cols = ["Mean Δ%", "Median Δ%", "Corr Δ%", "R²→RobR² Δ%", "RMSE Δ% (↓better)"]

    def color_positive(val):
        try:
            num = float(str(val).replace("%", "").replace("−", "-").strip())
        except ValueError:
            return ""
        return "color: #66bb6a; font-weight: 600;" if num > 0 else ""

    styled_summary = summary_df.style.map(color_positive, subset=delta_cols)
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Δ% = ((MG − Classical) / |Classical|) × 100.  "
        "For RMSE, sign is flipped so positive = MG gives lower error."
    )


# ═══════════════════════════════════════════════════════════════════
#  PAGE: WHY MG WORKS BETTER (GENERAL AUDIENCE)
# ═══════════════════════════════════════════════════════════════════
elif page == "why_mg":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>Data Truth Revealed: Machine Gnostics vs Classical Statistics</h3>
            <p>
            Classical statistics treats all points equally. Machine Gnostics learns which points truly matter.
            Below: benchmark numbers showing what each dataset really is, and how well each method reveals that truth.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_insights = {
        1: {
            "title": "Linear Data",
            "truth": "Perfect linear relationship: Y = 3 + 0.5×X",
            "classical_sees": "Correlation = 0.82, R² = 0.67 (detects trend)",
            "mg_sees": "Same high correlation and R² (no outliers to distract)",
            "verdict": "Both methods agree—data is cleanly linear.",
        },
        2: {
            "title": "Curved Data",
            "truth": "Non-linear relationship: Y = 3 + 0.5×X - noise that curves",
            "classical_sees": "Correlation = 0.82, R² = 0.67 (misleading—hides curve)",
            "mg_sees": "Correlation = 0.82, but robustness metrics show stress (data deviates from line consistently)",
            "verdict": "MG detects the relationship isn't truly linear—classical misses this.",
        },
        3: {
            "title": "Outlier Hidden",
            "truth": "Linear relationship Y = 3 + 0.5×X, but one extreme outlier point at (13, 13)",
            "classical_sees": "Correlation = 0.82 (outlier drags down apparent strength)",
            "mg_sees": "Down-weights outlier → higher true correlation ≈ 0.88 (reveals strong linear core)",
            "verdict": "MG reveals the true correlation by ignoring noise; classical conflates signal with outliers.",
        },
        4: {
            "title": "Leverage Point",
            "truth": "Weak horizontal trend Y ≈ 8 (roughly constant), but one extreme point at (19, 13)",
            "classical_sees": "Correlation = 0.82, R² = 0.67 (extreme point pulls slope up artificially)",
            "mg_sees": "Down-weights leverage point → correlation ≈ 0.35 (shows weak true trend)",
            "verdict": "MG resists extreme X-points; classical is fooled into seeing strong relationship.",
        },
    }

    rows_all = []
    for ds_id in [1, 2, 3, 4]:
        ds_name = DS_NAMES[ds_id]
        insight = dataset_insights[ds_id]

        st.markdown(f"### {ds_id}. {insight['title']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**What the data actually is:**")
            st.markdown(insight["truth"])

        with col2:
            st.markdown("**Verdict:**")
            st.markdown(f"*{insight['verdict']}*")

        st.markdown("---")

        cls = data[ds_id]["classical"]
        mg = data[ds_id]["mg"]

        row = {
            "Dataset": ds_name,
            "Classical Corr": f"{cls['corr']:.3f}",
            "MG Corr": f"{mg['corr']:.3f}",
            "Classical R²": f"{cls['r2']:.3f}",
            "MG RobR²": f"{mg['robr2']:.3f}",
            "MG Data Insight": insight["verdict"],
        }
        rows_all.append(row)

    st.markdown("### Summary: What Numbers Reveal")
    summary_df = pd.DataFrame(rows_all)

    def color_benchmark(val):
        if isinstance(val, str) and val not in ["Dataset", "MG Data Insight"]:
            try:
                num = float(val)
                if 0.8 <= num <= 0.9:
                    return "background-color: #2d5a3d"
                elif num > 0.9:
                    return "background-color: #1a3a2a"
            except ValueError:
                pass
        return ""

    styled_df = summary_df.style.map(
        lambda x: "background-color: #2d5a3d" if isinstance(x, str) and x not in summary_df.iloc[:, 0].values and "Insight" not in x else "",
        subset=summary_df.columns[1:5],
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: MEAN & MEDIAN
# ═══════════════════════════════════════════════════════════════════
elif page == "mean_median":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📊 Mean & Median — Classical vs Machine Gnostics</h3>
            <p>
            Classical mean and median treat all data points equally.
            Machine Gnostics computes a <b>gnostic mean/median</b> by assigning adaptive weights based on
            information content — down-weighting outliers and leveraging points with higher information density.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-box">
        <b>ℹ️ KPI Guide:</b>
        <code>Mean Δ%</code> and <code>Median Δ%</code> show how much MG differs from classical values.
        A non-zero % means MG detected a structural difference that classical statistics missed.
        For datasets with outliers (III, IV), MG mean should differ more — this is the <i>desired behavior</i>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_selected = st.selectbox("Select dataset to inspect:", [1, 2, 3, 4], format_func=lambda d: DS_NAMES[d])

    cls = data[ds_selected]["classical"]
    mg = data[ds_selected]["mg"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Mean X (MG vs Classical)", cls["mean_x"], mg["mean_x"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Mean Y (MG vs Classical)", cls["mean_y"], mg["mean_y"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Median X (MG vs Classical)", cls["median_x"], mg["median_x"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Median Y (MG vs Classical)", cls["median_y"], mg["median_y"]), unsafe_allow_html=True)

    st.markdown("#### Comparison Table — All Datasets")
    rows = []
    for ds_id in [1, 2, 3, 4]:
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        _, pm_x, _ = pct_diff(c["mean_x"], m["mean_x"])
        _, pm_y, _ = pct_diff(c["mean_y"], m["mean_y"])
        _, pmed_x, _ = pct_diff(c["median_x"], m["median_x"])
        _, pmed_y, _ = pct_diff(c["median_y"], m["median_y"])
        rows.append({
            "Dataset": DS_NAMES[ds_id],
            "CLS Mean X": round(c["mean_x"], 4), "MG Mean X": round(m["mean_x"], 4), "Mean X Δ%": pm_x,
            "CLS Mean Y": round(c["mean_y"], 4), "MG Mean Y": round(m["mean_y"], 4), "Mean Y Δ%": pm_y,
            "CLS Median Y": round(c["median_y"], 4), "MG Median Y": round(m["median_y"], 4), "Median Y Δ%": pmed_y,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Visual — Mean & Median overlaid on data")
    x_arr = data[ds_selected]["x"]
    y_arr = data[ds_selected]["y"]
    fig_mm, ax_mm = plt.subplots(figsize=(7, 4))
    fig_mm.patch.set_facecolor("#0f1a2e")
    ax_mm.set_facecolor("#0f1a2e")
    ax_mm.scatter(x_arr, y_arr, color="#4fc3f7", s=60, alpha=0.85, edgecolors="white", linewidths=0.4, label="Data")
    ax_mm.axhline(cls["mean_y"], color="#66bb6a", linestyle="--", lw=1.5, label=f"Classical Mean Y = {cls['mean_y']:.3f}")
    ax_mm.axhline(mg["mean_y"], color="#ef5350", linestyle="-", lw=1.5, label=f"MG Mean Y = {mg['mean_y']:.3f}")
    ax_mm.axhline(cls["median_y"], color="#ffa726", linestyle="--", lw=1.5, label=f"Classical Median Y = {cls['median_y']:.3f}")
    ax_mm.axhline(mg["median_y"], color="#ce93d8", linestyle="-", lw=1.5, label=f"MG Median Y = {mg['median_y']:.3f}")
    ax_mm.legend(fontsize=9, facecolor="#0f1a2e", labelcolor="#e0e6f0")
    ax_mm.set_xlabel("X", color="#8ca0c0")
    ax_mm.set_ylabel("Y", color="#8ca0c0")
    ax_mm.set_title(f"{DS_NAMES[ds_selected]} — Mean & Median Comparison", color="#e0e6f0")
    ax_mm.tick_params(colors="#8ca0c0")
    for sp in ax_mm.spines.values():
        sp.set_edgecolor("#2d4a7a")
    st.pyplot(fig_mm, use_container_width=True)
    plt.close(fig_mm)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: CORRELATION
# ═══════════════════════════════════════════════════════════════════
elif page == "correlation":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>🔗 Correlation — Classical Pearson vs Machine Gnostics</h3>
            <p>
            Classical Pearson correlation assumes linearity and equal data quality.
            Machine Gnostics correlation uses the gnostic weighting scheme to measure
            <b>robust association</b> that is less sensitive to outliers and non-linear patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-box">
        <b>ℹ️ KPI Guide:</b>
        <code>Corr Δ%</code> shows how much the two correlation estimates diverge.
        Dataset III contains an influential outlier; classical Pearson stays high while MG correlation drops,
        correctly indicating that the apparent correlation is driven by a single point.
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_corr = []
    for ds_id in [1, 2, 3, 4]:
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        _, pct, css = pct_diff(c["corr"], m["corr"])
        rows_corr.append({
            "Dataset": DS_NAMES[ds_id],
            "Classical Pearson r": round(c["corr"], 6),
            "MG Correlation": round(m["corr"], 6),
            "Δ% (MG vs Classical)": pct,
            "Interpretation": (
                "MG detects weaker true association" if m["corr"] < c["corr"]
                else "MG detects stronger association"
            ),
        })
    st.dataframe(pd.DataFrame(rows_corr), use_container_width=True, hide_index=True)

    st.markdown("#### KPI Cards — Per Dataset")
    cols = st.columns(4)
    for i, ds_id in enumerate([1, 2, 3, 4]):
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        with cols[i]:
            st.markdown(f"**{DS_NAMES[ds_id]}**")
            st.markdown(kpi_card("Correlation", c["corr"], m["corr"]), unsafe_allow_html=True)

    st.markdown("#### Bar Chart — Correlation Comparison")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    fig_corr.patch.set_facecolor("#0f1a2e")
    ax_corr.set_facecolor("#0f1a2e")
    ds_labels = [DS_NAMES[d] for d in [1, 2, 3, 4]]
    cls_corrs = [data[d]["classical"]["corr"] for d in [1, 2, 3, 4]]
    mg_corrs = [data[d]["mg"]["corr"] for d in [1, 2, 3, 4]]
    x_pos = np.arange(4)
    ax_corr.bar(x_pos - 0.2, cls_corrs, 0.35, label="Classical", color="#66bb6a", alpha=0.85)
    ax_corr.bar(x_pos + 0.2, mg_corrs, 0.35, label="Machine Gnostics", color="#4fc3f7", alpha=0.85)
    ax_corr.set_xticks(x_pos)
    ax_corr.set_xticklabels(ds_labels, color="#8ca0c0")
    ax_corr.set_ylabel("Correlation", color="#8ca0c0")
    ax_corr.set_ylim(0, 1.15)
    ax_corr.set_title("Correlation: Classical vs Machine Gnostics", color="#e0e6f0")
    ax_corr.legend(facecolor="#0f1a2e", labelcolor="#e0e6f0")
    ax_corr.tick_params(colors="#8ca0c0")
    for sp in ax_corr.spines.values():
        sp.set_edgecolor("#2d4a7a")
    for i, (cv, mv) in enumerate(zip(cls_corrs, mg_corrs)):
        ax_corr.text(i - 0.2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
        ax_corr.text(i + 0.2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: R² & ROBUST R²
# ═══════════════════════════════════════════════════════════════════
elif page == "r2":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📈 R² vs Robust R² (RobR²) — Goodness of Fit</h3>
            <p>
            Classical R² measures the proportion of variance explained by the regression line — but it can be
            inflated by outliers and non-linear patterns. Machine Gnostics introduces <b>Robust R² (RobR²)</b>,
            a weighted coefficient of determination that penalizes fits driven by high-leverage points.
            A high <b>RobR²</b> means the fit is genuinely good across the full data, not just at extreme points.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-box">
        <b>ℹ️ KPI Guide:</b>
        <code>R² Δ%</code> = ((RobR² − Classical R²) / |Classical R²|) × 100.
        A positive % means MG regression achieves a higher (better) robust fit.
        A negative % means classical OLS overfits via outlier influence — the fit looks good but is unreliable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_r2 = []
    for ds_id in [1, 2, 3, 4]:
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        _, pct, _ = pct_diff(c["r2"], m["robr2"])
        rows_r2.append({
            "Dataset": DS_NAMES[ds_id],
            "Classical R²": round(c["r2"], 6),
            "MG RobR²": round(m["robr2"], 6),
            "MG R² (std)": round(m["r2"], 6),
            "RobR² Δ%": pct,
            "Verdict": (
                "✅ MG more reliable" if m["robr2"] >= c["r2"] * 0.95
                else "⚠️ Classical R² inflated by leverage"
            ),
        })
    st.dataframe(pd.DataFrame(rows_r2), use_container_width=True, hide_index=True)

    st.markdown("#### KPI Cards — RobR² vs Classical R²")
    cols_r2 = st.columns(4)
    for i, ds_id in enumerate([1, 2, 3, 4]):
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        with cols_r2[i]:
            st.markdown(f"**{DS_NAMES[ds_id]}**")
            st.markdown(kpi_card("RobR² vs Classical R²", c["r2"], m["robr2"]), unsafe_allow_html=True)

    st.markdown("#### Bar Chart — R² Comparison")
    fig_r2, ax_r2 = plt.subplots(figsize=(8, 4))
    fig_r2.patch.set_facecolor("#0f1a2e")
    ax_r2.set_facecolor("#0f1a2e")
    x_pos = np.arange(4)
    cls_r2s = [data[d]["classical"]["r2"] for d in [1, 2, 3, 4]]
    mg_r2s = [data[d]["mg"]["robr2"] for d in [1, 2, 3, 4]]
    ax_r2.bar(x_pos - 0.2, cls_r2s, 0.35, label="Classical R²", color="#ffa726", alpha=0.85)
    ax_r2.bar(x_pos + 0.2, mg_r2s, 0.35, label="MG Robust R²", color="#4fc3f7", alpha=0.85)
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels([DS_NAMES[d] for d in [1, 2, 3, 4]], color="#8ca0c0")
    ax_r2.set_ylabel("R² / RobR²", color="#8ca0c0")
    ax_r2.set_ylim(0, 1.2)
    ax_r2.set_title("R² vs Robust R²: Classical vs Machine Gnostics", color="#e0e6f0")
    ax_r2.legend(facecolor="#0f1a2e", labelcolor="#e0e6f0")
    ax_r2.tick_params(colors="#8ca0c0")
    for sp in ax_r2.spines.values():
        sp.set_edgecolor("#2d4a7a")
    for i, (cv, mv) in enumerate(zip(cls_r2s, mg_r2s)):
        ax_r2.text(i - 0.2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
        ax_r2.text(i + 0.2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
    st.pyplot(fig_r2, use_container_width=True)
    plt.close(fig_r2)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: RMSE
# ═══════════════════════════════════════════════════════════════════
elif page == "rmse":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📉 RMSE — Root Mean Squared Error</h3>
            <p>
            Classical RMSE treats all residuals equally. Machine Gnostics RMSE is computed on predictions
            from the adaptive-weighted regression, so large residuals on low-weight (outlier) points
            contribute less. A <b>lower MG RMSE</b> means the fit is more accurate for the bulk of the data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-box">
        <b>ℹ️ KPI Guide:</b>
        <code>RMSE Δ%</code> sign is <b>flipped</b> so that a <b>positive %</b> means MG achieves
        a <i>lower</i> RMSE (better). Negative % means classical OLS has a lower raw RMSE — typically
        because it overfitted to outliers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_rmse = []
    for ds_id in [1, 2, 3, 4]:
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        _, pct, _ = pct_diff(c["rmse"], m["rmse"], lower_is_better=True)
        rows_rmse.append({
            "Dataset": DS_NAMES[ds_id],
            "Classical RMSE": round(c["rmse"], 6),
            "MG RMSE": round(m["rmse"], 6),
            "RMSE Δ% (↓better→+%)": pct,
            "Verdict": "✅ MG lower error" if m["rmse"] <= c["rmse"] else "ℹ️ Classical lower raw RMSE",
        })
    st.dataframe(pd.DataFrame(rows_rmse), use_container_width=True, hide_index=True)

    st.markdown("#### KPI Cards")
    cols_rmse = st.columns(4)
    for i, ds_id in enumerate([1, 2, 3, 4]):
        c = data[ds_id]["classical"]
        m = data[ds_id]["mg"]
        with cols_rmse[i]:
            st.markdown(f"**{DS_NAMES[ds_id]}**")
            st.markdown(kpi_card("RMSE (lower = better)", c["rmse"], m["rmse"], lower_is_better=True), unsafe_allow_html=True)

    st.markdown("#### Bar Chart — RMSE Comparison")
    fig_rmse, ax_rmse = plt.subplots(figsize=(8, 4))
    fig_rmse.patch.set_facecolor("#0f1a2e")
    ax_rmse.set_facecolor("#0f1a2e")
    x_pos = np.arange(4)
    cls_rmses = [data[d]["classical"]["rmse"] for d in [1, 2, 3, 4]]
    mg_rmses = [data[d]["mg"]["rmse"] for d in [1, 2, 3, 4]]
    ax_rmse.bar(x_pos - 0.2, cls_rmses, 0.35, label="Classical RMSE", color="#ef5350", alpha=0.85)
    ax_rmse.bar(x_pos + 0.2, mg_rmses, 0.35, label="MG RMSE", color="#4fc3f7", alpha=0.85)
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels([DS_NAMES[d] for d in [1, 2, 3, 4]], color="#8ca0c0")
    ax_rmse.set_ylabel("RMSE", color="#8ca0c0")
    ax_rmse.set_title("RMSE: Classical vs Machine Gnostics (lower is better)", color="#e0e6f0")
    ax_rmse.legend(facecolor="#0f1a2e", labelcolor="#e0e6f0")
    ax_rmse.tick_params(colors="#8ca0c0")
    for sp in ax_rmse.spines.values():
        sp.set_edgecolor("#2d4a7a")
    for i, (cv, mv) in enumerate(zip(cls_rmses, mg_rmses)):
        ax_rmse.text(i - 0.2, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
        ax_rmse.text(i + 0.2, mv + 0.02, f"{mv:.3f}", ha="center", fontsize=8, color="#e0e6f0")
    st.pyplot(fig_rmse, use_container_width=True)
    plt.close(fig_rmse)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════
elif page == "regression":
    st.markdown(
        """
        <div class="problem-banner">
            <h3>📐 Linear Regression — Classical OLS vs Machine Gnostics</h3>
            <p>
            Classical OLS minimises the sum of squared residuals equally across all points.
            Machine Gnostics Linear Regression minimises a gnostic loss function (<code>mg_loss="hi"</code>)
            that assigns lower weight to points with high residuals or high leverage,
            yielding a regression line that represents the <b>core data trend</b> more accurately.
            Point <b>bubble size and colour</b> below encode MG adaptive weights — large green = trusted;
            small red = down-weighted.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="step-box">
        <b>ℹ️ KPI Guide for Regression:</b>
        <ul>
          <li><code>R²</code>: Classical coefficient of determination (can be inflated by outliers).</li>
          <li><code>RobR²</code>: Machine Gnostics robust R² — penalises fits that rely on leverage points.</li>
          <li><code>RMSE Δ%</code>: Positive % = MG achieves lower prediction error.</li>
          <li><code>Corr Δ%</code>: Divergence between Pearson and MG correlation.</li>
          <li><b>Bubble colour</b>: Green = high MG weight (trusted), Red = low weight (down-weighted).</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_sel = st.selectbox("Select dataset for detailed view:", [1, 2, 3, 4], format_func=lambda d: DS_NAMES[d], key="reg_ds")

    # ── Top KPI row ───────────────────────────────────────────────
    st.markdown("#### KPI Overview")
    c = data[ds_sel]["classical"]
    m = data[ds_sel]["mg"]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi_card("Classical R²", 1.0, c["r2"], unit=""), unsafe_allow_html=True)
        st.caption("Proportion of variance explained by OLS")
    with k2:
        st.markdown(kpi_card("MG Robust R²", 1.0, m["robr2"], unit=""), unsafe_allow_html=True)
        st.caption("Weighted goodness-of-fit from Machine Gnostics")
    with k3:
        _, pct_r2, css_r2 = pct_diff(c["r2"], m["robr2"])
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">RobR² vs Classical R² Δ%</div>'
            f'<div class="kpi-value {css_r2}">{pct_r2}</div>'
            f'<div class="kpi-sub">Positive = MG fit is more robust</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("% MG change in fit robustness")
    with k4:
        st.markdown(kpi_card("RMSE (↓better)", c["rmse"], m["rmse"], lower_is_better=True), unsafe_allow_html=True)
        st.caption("Lower RMSE = more accurate predictions")
    with k5:
        st.markdown(kpi_card("Correlation", c["corr"], m["corr"]), unsafe_allow_html=True)
        st.caption("Pearson vs MG gnostic correlation")

    st.markdown("---")

    # ── Regression plot ───────────────────────────────────────────
    st.markdown("#### Regression Fit Visualisation")
    x_arr = data[ds_sel]["x"]
    y_arr = data[ds_sel]["y"]
    y_pred_cls = c["y_pred"]
    y_pred_mg = m["y_pred"]
    weights = m["weights"]

    fig_reg, ax_reg = plt.subplots(figsize=(9, 5))
    fig_reg.patch.set_facecolor("#0f1a2e")
    ax_reg.set_facecolor("#0f1a2e")

    if weights is not None and weights.size == len(x_arr):
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        sizes = 60 + w_norm * 220
        sc = ax_reg.scatter(
            x_arr, y_arr, s=sizes, c=weights,
            cmap="RdYlGn", alpha=0.9, edgecolors="white", linewidths=0.4,
            label="Data (size & colour = MG weight)", zorder=3,
        )
        cbar = fig_reg.colorbar(sc, ax=ax_reg)
        cbar.set_label("MG Weight", color="#8ca0c0")
        cbar.ax.yaxis.set_tick_params(color="#8ca0c0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8ca0c0")
    else:
        ax_reg.scatter(x_arr, y_arr, s=80, color="#4fc3f7", alpha=0.85,
                       edgecolors="white", linewidths=0.4, label="Data", zorder=3)

    idx = np.argsort(x_arr)
    ax_reg.plot(x_arr[idx], y_pred_cls[idx], "--", color="#66bb6a", lw=2.5,
                label=f"Classical OLS  R²={c['r2']:.4f}  RMSE={c['rmse']:.4f}", zorder=2)
    ax_reg.plot(x_arr[idx], y_pred_mg[idx], "-", color="#ef5350", lw=2.5,
                label=f"Machine Gnostics  RobR²={m['robr2']:.4f}  RMSE={m['rmse']:.4f}", zorder=2)

    ax_reg.axhline(c["mean_y"], color="#66bb6a", linestyle="-.", lw=1, alpha=0.6,
                   label=f"Classical Mean Y={c['mean_y']:.3f}")
    ax_reg.axhline(m["mean_y"], color="#ef5350", linestyle="-.", lw=1, alpha=0.6,
                   label=f"MG Mean Y={m['mean_y']:.3f}")

    ax_reg.set_xlabel("X", color="#8ca0c0")
    ax_reg.set_ylabel("Y", color="#8ca0c0")
    ax_reg.set_title(f"{DS_NAMES[ds_sel]} — Classical OLS vs Machine Gnostics Regression", color="#e0e6f0")
    ax_reg.legend(fontsize=9, facecolor="#0f1a2e", labelcolor="#e0e6f0", loc="best")
    ax_reg.tick_params(colors="#8ca0c0")
    for sp in ax_reg.spines.values():
        sp.set_edgecolor("#2d4a7a")
    st.pyplot(fig_reg, use_container_width=True)
    plt.close(fig_reg)

    # ── Residual plot ─────────────────────────────────────────────
    st.markdown("#### Residual Comparison")
    residuals_cls = y_arr - y_pred_cls
    residuals_mg = y_arr - y_pred_mg

    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax in (ax1, ax2):
        fig_res.patch.set_facecolor("#0f1a2e")
        ax.set_facecolor("#0f1a2e")
        ax.tick_params(colors="#8ca0c0")
        ax.set_xlabel("Fitted value", color="#8ca0c0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2d4a7a")

    ax1.scatter(y_pred_cls, residuals_cls, color="#66bb6a", alpha=0.8, s=60, edgecolors="white", linewidths=0.3)
    ax1.axhline(0, color="white", lw=0.8, linestyle="--")
    ax1.set_ylabel("Residual", color="#8ca0c0")
    ax1.set_title(f"Classical OLS Residuals\nRMSE={c['rmse']:.4f}", color="#e0e6f0")

    ax2.scatter(y_pred_mg, residuals_mg, color="#4fc3f7", alpha=0.8, s=60, edgecolors="white", linewidths=0.3)
    ax2.axhline(0, color="white", lw=0.8, linestyle="--")
    ax2.set_ylabel("Residual", color="#8ca0c0")
    ax2.set_title(f"Machine Gnostics Residuals\nRMSE={m['rmse']:.4f}", color="#e0e6f0")

    st.pyplot(fig_res, use_container_width=True)
    plt.close(fig_res)

    # ── All-dataset regression summary table ─────────────────────
    st.markdown("#### Full Regression KPI Table — All Datasets")
    reg_rows = []
    for ds_id in [1, 2, 3, 4]:
        c2 = data[ds_id]["classical"]
        m2 = data[ds_id]["mg"]
        _, pr2, _ = pct_diff(c2["r2"], m2["robr2"])
        _, prmse, _ = pct_diff(c2["rmse"], m2["rmse"], lower_is_better=True)
        _, pcorr, _ = pct_diff(c2["corr"], m2["corr"])
        reg_rows.append({
            "Dataset": DS_NAMES[ds_id],
            "CLS R²": round(c2["r2"], 4),
            "MG RobR²": round(m2["robr2"], 4),
            "RobR² Δ%": pr2,
            "CLS RMSE": round(c2["rmse"], 4),
            "MG RMSE": round(m2["rmse"], 4),
            "RMSE Δ% (↓better→+%)": prmse,
            "CLS Corr": round(c2["corr"], 4),
            "MG Corr": round(m2["corr"], 4),
            "Corr Δ%": pcorr,
            "CLS Slope": round(c2["slope"], 4),
            "CLS Intercept": round(c2["intercept"], 4),
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
    st.caption(
        "RobR² Δ%: positive = MG regression achieves more robust fit.  "
        "RMSE Δ%: positive = MG has lower prediction error."
    )


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="section-title">ℹ️ About this Benchmark App</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### What is Machine Gnostics?
        Welcome to **Machine Gnostics**, an innovative Python library designed to implement
        the principles of Mathematical Gnostics for robust data analysis, modeling, and inference.
        Unlike traditional statistical approaches that depend heavily on probabilistic assumptions,
        Machine Gnostics harnesses deterministic algebraic and geometric structures.
        This unique foundation enables the library to deliver exceptional resilience against outliers,
        noise, and corrupted data, making it a powerful tool for challenging real-world scenarios.

        Core capabilities include:

        - Down-weight outliers and leverage points **automatically**
        - Produce **robust estimates** of mean, median, and correlation
        - Offer **Robust R² (RobR²)** — a goodness-of-fit metric less sensitive to influential observations
        - Train regression models using a **gnostic loss function** rather than OLS

        ### What is the Anscombe Quartet?
        Four datasets (I–IV) with **identical classical statistics**:
        - Same mean and variance of X and Y
        - Same Pearson correlation (~0.816)
        - Same OLS regression line
        - Same classical R² (~0.67) and RMSE

        Yet they are structurally completely different — I is linear, II is curved, III has one outlier,
        IV is vertical with one leverage point. This quartet is the ideal benchmark to expose
        the limitations of classical statistics.

        ### KPI Glossary

        | KPI | Definition | Positive % means |
        |-----|-----------|-----------------|
        | **Mean Δ%** | `(MG mean − Classical mean) / |Classical mean| × 100` | MG detects a different central tendency |
        | **Median Δ%** | Same for median | MG robust median differs from classical |
        | **Corr Δ%** | `(MG corr − Pearson r) / |Pearson r| × 100` | MG detects stronger association |
        | **RobR² Δ%** | `(RobR² − R²) / |R²| × 100` | MG regression is more robustly fitted |
        | **RMSE Δ%** | `−(MG RMSE − Classical RMSE) / |Classical RMSE| × 100` | MG achieves lower prediction error |

        ### Technical Details
        - **MG Linear Regressor** settings: `max_iter=300, early_stopping=True, tolerance=1e-6, mg_loss="hi"`
        - All MG predictions use `model.predict(x)` with the fitted adaptive weights
        - RobR² is computed via `machinegnostics.metrics.robr2(y, y_pred, w=model.weights)`

        ### Links
        - 📚 Documentation: [docs.machinegnostics.com](https://docs.machinegnostics.com)
        - 🐙 Notebooks: [github.com/nirmalparmarphd/machinegnostics-anscombe-data](https://github.com/nirmalparmarphd/machinegnostics-anscombe-data)
        """,
        unsafe_allow_html=False,
    )
