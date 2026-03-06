import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.simulate import run_simulation
from src.stats_engine import run_analysis

st.set_page_config(
    page_title="Spotify Feature Launch Experiment",
    page_icon="🎵",
    layout="wide"
)

@st.cache_data
def load_data():
    df = run_simulation()
    results = run_analysis()
    return df, results

df, results = load_data()


# ── HEADER ─────────────────────────────────────────────────────────────────

st.title("🎵 Spotify Feature Launch Experiment")
st.markdown("""
A three-stage A/B testing pipeline measuring the impact of an AI-powered playlist feature 
on free-to-paid Premium conversion. 10,000 simulated users were randomly assigned to control 
and treatment groups across a feature adoption, notification re-engagement, and Premium 
conversion funnel.
""")

st.divider()

# ── FUNNEL SUMMARY METRICS ──────────────────────────────────────────────────

st.subheader("Funnel Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Users", f"{len(df):,}")
with col2:
    st.metric(
        "Feature Adoption Lift",
        f"{results['adoption']['observed_lift']:.1%}",
        delta=f"{results['adoption']['observed_lift']:.1%}"
    )
with col3:
    st.metric(
        "Re-engagement Lift",
        f"{results['reengagement']['observed_lift']:.1%}",
        delta=f"{results['reengagement']['observed_lift']:.1%}"
    )
with col4:
    st.metric(
        "Conversion Lift",
        f"{results['conversion']['observed_lift']:.1%}",
        delta=f"{results['conversion']['observed_lift']:.1%}"
    )

st.divider()


# ── STAGE 1: FEATURE ADOPTION ──────────────────────────────────────────────

st.subheader("Stage 1: Feature Adoption")

col1, col2 = st.columns(2)

with col1:
    adoption = results["adoption"]
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Control Rate | {adoption['control_rate']:.1%} |
    | Treatment Rate | {adoption['treatment_rate']:.1%} |
    | Observed Lift | {adoption['observed_lift']:.1%} |
    | 95% CI | [{adoption['ci_low']:.1%}, {adoption['ci_high']:.1%}] |
    | Chi-square | {adoption['chi2_statistic']} |
    | P-value | {adoption['p_value']} |
    | Significant | {adoption['significant']} |
    | P(Treatment > Control) | {adoption['prob_treatment_wins']:.1%} |
    """)

with col2:
    fig = go.Figure(go.Bar(
        x=["Control", "Treatment"],
        y=[adoption['control_rate'], adoption['treatment_rate']],
        marker_color=["#535353", "#1DB954"],
        text=[f"{adoption['control_rate']:.1%}", f"{adoption['treatment_rate']:.1%}"],
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Adoption Rate by Group",
        yaxis_tickformat=".0%",
        yaxis_title="Adoption Rate",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── STAGE 2: NOTIFICATION RE-ENGAGEMENT ────────────────────────────────────

st.subheader("Stage 2: Notification Re-engagement")

col1, col2 = st.columns(2)

with col1:
    reengagement = results["reengagement"]
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Control Rate | {reengagement['control_rate']:.1%} |
    | Treatment Rate | {reengagement['treatment_rate']:.1%} |
    | Observed Lift | {reengagement['observed_lift']:.1%} |
    | 95% CI | [{reengagement['ci_low']:.1%}, {reengagement['ci_high']:.1%}] |
    | Chi-square | {reengagement['chi2_statistic']} |
    | P-value | {reengagement['p_value']} |
    | Significant | {reengagement['significant']} |
    | P(Treatment > Control) | {reengagement['prob_treatment_wins']:.1%} |
    """)

with col2:
    fig = go.Figure(go.Bar(
        x=["Control", "Treatment"],
        y=[reengagement['control_rate'], reengagement['treatment_rate']],
        marker_color=["#535353", "#1DB954"],
        text=[f"{reengagement['control_rate']:.1%}", f"{reengagement['treatment_rate']:.1%}"],
        textposition="outside"
    ))
    fig.update_layout(
        title="Re-engagement Rate by Group",
        yaxis_tickformat=".0%",
        yaxis_title="Re-engagement Rate",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── STAGE 3: PREMIUM CONVERSION ─────────────────────────────────────────────

st.subheader("Stage 3: Premium Conversion")

col1, col2 = st.columns(2)

with col1:
    conversion = results["conversion"]
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Control Rate | {conversion['control_rate']:.1%} |
    | Treatment Rate | {conversion['treatment_rate']:.1%} |
    | Observed Lift | {conversion['observed_lift']:.1%} |
    | 95% CI | [{conversion['ci_low']:.1%}, {conversion['ci_high']:.1%}] |
    | Chi-square | {conversion['chi2_statistic']} |
    | T-statistic | {conversion['t_statistic']} |
    | P-value | {conversion['p_value']} |
    | Significant | {conversion['significant']} |
    | P(Treatment > Control) | {conversion['prob_treatment_wins']:.1%} |
    """)

with col2:
    fig = go.Figure(go.Bar(
        x=["Control", "Treatment"],
        y=[conversion['control_rate'], conversion['treatment_rate']],
        marker_color=["#535353", "#1DB954"],
        text=[f"{conversion['control_rate']:.1%}", f"{conversion['treatment_rate']:.1%}"],
        textposition="outside"
    ))
    fig.update_layout(
        title="Premium Conversion Rate by Group",
        yaxis_tickformat=".0%",
        yaxis_title="Conversion Rate",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── MULTIPLE TESTING CORRECTION ─────────────────────────────────────────────

st.subheader("Multiple Testing Correction (Bonferroni)")

bonferroni = results["bonferroni"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Adjusted Alpha", bonferroni["adjusted_alpha"])
with col2:
    st.metric("Tests Run", len(bonferroni["original_p_values"]))
with col3:
    st.metric("Tests Surviving Correction", sum(bonferroni["significant"]))

st.markdown(f"""
| Stage | Original P-value | Adjusted Alpha | Significant |
|-------|-----------------|----------------|-------------|
| Feature Adoption | {bonferroni['original_p_values'][0]} | {bonferroni['adjusted_alpha']} | {bonferroni['significant'][0]} |
| Notification Re-engagement | {bonferroni['original_p_values'][1]} | {bonferroni['adjusted_alpha']} | {bonferroni['significant'][1]} |
| Premium Conversion | {bonferroni['original_p_values'][2]} | {bonferroni['adjusted_alpha']} | {bonferroni['significant'][2]} |
""")

st.divider()

# ── CONCLUSION ───────────────────────────────────────────────────────────────

st.subheader("Conclusion")

st.markdown("""
| Stage | Control | Treatment | Lift | Significant |
|-------|---------|-----------|------|-------------|
| Feature Adoption | 5.3% | 25.7% | +20.4pp | ✅ |
| Notification Re-engagement | 7.5% | 14.3% | +6.8pp | ✅ |
| Premium Conversion | 0.4% | 4.3% | +4.0pp | ✅ |
""")

st.success("""
All three stages of the funnel show statistically significant improvement in the treatment group. 
Results survive Bonferroni correction for multiple comparisons. Bayesian analysis assigns 100% 
posterior probability to treatment outperforming control at every stage. 

**Recommendation: Ship the feature.**
""")