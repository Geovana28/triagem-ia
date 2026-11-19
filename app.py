"""
Clinical Triage AI — Interactive Streamlit Application
Manchester Triage System (MTS) + NEWS2 + Scikit-Learn Random Forest + Asymmetric Clinical Cost Engine.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from model import calculate_triage_score

st.set_page_config(
    page_title="Clinical Triage AI — Manchester & NEWS2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Monochrome Structural Engineering CSS Override
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background-color: #0b0b0d !important; color: #f4f4f6 !important; }
    header[data-testid="stHeader"] { background-color: #0b0b0d !important; border-bottom: 1px solid #18181c; }
    section[data-testid="stSidebar"] { background-color: #121215 !important; border-right: 1px solid #242429; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #a1a1aa !important; font-size: 0.82rem; }
    section[data-testid="stSidebar"] h3 { color: #f4f4f6 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-baseweb="select"] > div { background-color: #0b0b0d !important; border-color: #323238 !important; color: #f4f4f6 !important; border-radius: 4px !important; }

    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        margin-bottom: 22px;
        overflow: hidden;
    }
    @media (max-width: 900px) { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
    .kpi-cell { padding: 15px 18px; border-right: 1px solid #242429; }
    .kpi-cell:last-child { border-right: none; }
    .kpi-lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717a; font-weight: 500; }
    .kpi-num { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #f4f4f6; margin: 5px 0 3px 0; }
    .kpi-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.71rem; color: #a1a1aa; }

    .triage-banner {
        background-color: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }
    .triage-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .triage-title {
        font-size: 1.65rem;
        font-weight: 600;
        margin: 4px 0 8px 0;
    }
    .flag-item {
        background-color: #121215;
        border: 1px solid #323238;
        border-radius: 4px;
        padding: 8px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        margin-bottom: 6px;
        color: #f4f4f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "triage_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

FEATURE_LABELS = [
    "Heart Rate (bpm)",
    "Systolic BP (mmHg)",
    "SpO2 (%)",
    "Resp. Rate (/min)",
    "Temperature (°C)",
    "Glasgow (GCS)",
    "Pain (VAS)",
]

CATEGORY_META = {
    0: {"name": "Non-Urgent (Blue)", "code": "CATEGORY V", "color": "#3b82f6", "time": "≤ 240 min", "priority": 5,
        "rec": "Patient exhibits stable physiological parameters. Eligible for standard outpatient management."},
    1: {"name": "Standard (Green)", "code": "CATEGORY IV", "color": "#22c55e", "time": "≤ 120 min", "priority": 4,
        "rec": "Mild physiological deviation detected. Standard emergency department evaluation queue."},
    2: {"name": "Urgent (Yellow)", "code": "CATEGORY III", "color": "#eab308", "time": "≤ 60 min", "priority": 3,
        "rec": "Moderate clinical risk. Continuous observation and prompt medical assessment required."},
    3: {"name": "Very Urgent (Orange)", "code": "CATEGORY II", "color": "#f97316", "time": "≤ 10 min", "priority": 2,
        "rec": "High risk of acute decompensation. Immediate stretcher placement, ECG/monitoring and physician review."},
    4: {"name": "Immediate / Resuscitation (Red)", "code": "CATEGORY I", "color": "#ef4444", "time": "0 min (Immediate)", "priority": 1,
        "rec": "Life-threatening physiological instability. Activate resuscitation bay and emergency response team immediately."},
}

PRESETS = {
    "Custom / Manual": None,
    "Septic Shock (Hypotensive / Tachycardic)": {"hr": 134, "sbp": 76, "spo2": 89, "rr": 28, "temp": 39.6, "gcs": 12, "pain": 6},
    "Acute Coronary Syndrome (Chest Pain)": {"hr": 118, "sbp": 168, "spo2": 93, "rr": 24, "temp": 36.9, "gcs": 15, "pain": 9},
    "Severe Acute Asthma (Hypoxic)": {"hr": 126, "sbp": 128, "spo2": 84, "rr": 32, "temp": 37.1, "gcs": 14, "pain": 4},
    "Uncomplicated Migraine": {"hr": 84, "sbp": 126, "spo2": 98, "rr": 16, "temp": 36.8, "gcs": 15, "pain": 6},
    "Normal Homeostasis Baseline": {"hr": 76, "sbp": 120, "spo2": 98, "rr": 16, "temp": 36.7, "gcs": 15, "pain": 0},
}


@st.cache_resource(show_spinner="Initializing Random Forest Clinical Triage Model...")
def load_or_train_model():
    """Loads persisted model or retrains cleanly if incompatible across Python/sklearn versions."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            scl = joblib.load(SCALER_PATH)
            if hasattr(clf, "predict_proba") and hasattr(scl, "transform"):
                # Smoke-test inference to catch cross-version pickle incompatibilities
                _ = clf.predict_proba(scl.transform(np.array([[76, 120, 98, 16, 36.7, 15, 0]])))
                return clf, scl
        except Exception:
            pass

    np.random.seed(42)
    n_samples = 6000
    hr = np.random.normal(82, 22, n_samples).clip(35, 180)
    sbp = np.random.normal(122, 24, n_samples).clip(60, 230)
    spo2 = np.random.normal(97, 4, n_samples).clip(70, 100)
    rr = np.random.normal(17, 6, n_samples).clip(8, 45)
    temp = np.random.normal(36.8, 1.1, n_samples).clip(34.0, 41.5)
    gcs = np.random.choice(
        [15, 14, 13, 12, 11, 10, 9, 8, 6, 3],
        size=n_samples,
        p=[0.75, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
    )
    pain = np.random.randint(0, 11, n_samples)

    X = np.column_stack([hr, sbp, spo2, rr, temp, gcs, pain])
    y = np.zeros(n_samples, dtype=int)
    code_to_idx = {"blue": 0, "green": 1, "yellow": 2, "orange": 3, "red": 4}
    for i in range(n_samples):
        res = calculate_triage_score(hr[i], sbp[i], spo2[i], temp[i], rr[i], gcs[i], pain[i])
        y[i] = code_to_idx[res["code"]]

    scl = StandardScaler()
    X_scaled = scl.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scl


clf, scaler = load_or_train_model()

# --- SIDEBAR: CLINICAL INPUTS ---
st.sidebar.markdown("### 01 / Clinical Vignettes")
selected_preset = st.sidebar.selectbox("Reference Scenario", list(PRESETS.keys()), index=1)
preset_vals = PRESETS[selected_preset] or {"hr": 78, "sbp": 120, "spo2": 98, "rr": 16, "temp": 36.8, "gcs": 15, "pain": 2}

st.sidebar.markdown("---")
st.sidebar.markdown("### 02 / Patient Vital Signs")
hr = st.sidebar.slider("Heart Rate (bpm)", 30, 180, int(preset_vals["hr"]))
sbp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 50, 230, int(preset_vals["sbp"]))
spo2 = st.sidebar.slider("Oxygen Saturation SpO2 (%)", 70, 100, int(preset_vals["spo2"]))
rr = st.sidebar.slider("Respiratory Rate (/min)", 6, 45, int(preset_vals["rr"]))
temp = st.sidebar.slider("Core Body Temperature (°C)", 33.0, 42.0, float(preset_vals["temp"]), step=0.1)
gcs = st.sidebar.slider("Glasgow Coma Scale (GCS)", 3, 15, int(preset_vals["gcs"]))
pain = st.sidebar.slider("Pain Rating (VAS 0–10)", 0, 10, int(preset_vals["pain"]))

st.sidebar.markdown("---")
st.sidebar.markdown("### 03 / Asymmetric Clinical Loss")
fn_weight = st.sidebar.slider("False Negative Severity Weight (C_FN / C_FP)", 5, 50, 20, step=5)

# --- INFERENCE ---
raw_x = np.array([[hr, sbp, spo2, rr, temp, gcs, pain]], dtype=float)
scaled_x = scaler.transform(raw_x)
probs = clf.predict_proba(scaled_x)[0]
ml_pred_class = int(np.argmax(probs))

rule_eval = calculate_triage_score(hr, sbp, spo2, temp, rr, gcs, pain)
code_to_idx = {"blue": 0, "green": 1, "yellow": 2, "orange": 3, "red": 4}
rule_class = code_to_idx[rule_eval["code"]]

# Asymmetric safety-adjusted decision: escalate priority if weighted tail risk exceeds threshold
severe_tail_prob = float(probs[3] + probs[4])
expected_fn_loss = severe_tail_prob * fn_weight
final_class = max(ml_pred_class, rule_class)
if expected_fn_loss >= 2.5 and final_class < 3:
    final_class = 3
if expected_fn_loss >= 6.0 and final_class < 4:
    final_class = 4

meta = CATEGORY_META[final_class]
icu_risk_pct = round(float(probs[4] * 0.7 + probs[3] * 0.25) * 100.0, 1)

# --- MAIN HEADER ---
st.markdown(
    """
    <div style="border-bottom:1px solid #242429; padding-bottom:14px; margin-bottom:20px;">
        <div style="font-family:monospace; font-size:0.75rem; color:#71717a; letter-spacing:0.06em;">
            CLINICAL DECISION SUPPORT // MANCHESTER PROTOCOL + NEWS2 + RANDOM FOREST (100 TREES)
        </div>
        <h1 style="margin:4px 0 6px 0; font-size:1.75rem; font-weight:600; color:#f4f4f6;">
            Emergency Clinical Triage & Risk Stratification
        </h1>
        <div style="color:#a1a1aa; font-size:0.9rem;">
            Real-time physiological risk modeling with asymmetric false-negative penalty optimization and feature attribution.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- DYNAMIC MANCHESTER BANNER ---
st.markdown(
    f"""
    <div class="triage-banner" style="border-color:{meta['color']};">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div class="triage-code" style="color:{meta['color']};">{meta['code']} // MANCHESTER PRIORITY {meta['priority']}/5</div>
                <div class="triage-title" style="color:{meta['color']};">{meta['name']}</div>
            </div>
            <div style="background:#0b0b0d; border:1px solid #242429; padding:10px 16px; border-radius:4px; text-align:right;">
                <div style="font-size:0.7rem; color:#71717a; text-transform:uppercase;">Max Target Assessment</div>
                <div style="font-family:monospace; font-size:1.15rem; font-weight:600; color:#f4f4f6;">{meta['time']}</div>
            </div>
        </div>
        <div style="color:#d4d4d8; font-size:0.88rem; margin-top:8px;">{meta['rec']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- UNIFIED KPI STRIP ---
st.markdown(
    f"""
    <div class="kpi-strip">
        <div class="kpi-cell">
            <div class="kpi-lbl">NEWS2 Aggregate Score</div>
            <div class="kpi-num">{rule_eval['score']} pts</div>
            <div class="kpi-meta">Royal College Physicians</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-lbl">ICU / Decompensation Risk</div>
            <div class="kpi-num" style="color:{meta['color']};">{icu_risk_pct}%</div>
            <div class="kpi-meta">RF Tail Probability</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-lbl">Expected FN Loss (Cost)</div>
            <div class="kpi-num">{expected_fn_loss:.2f}</div>
            <div class="kpi-meta">FN Severity Weight: {fn_weight}x</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-lbl">RF Predicted Confidence</div>
            <div class="kpi-num">{probs[ml_pred_class]*100:.1f}%</div>
            <div class="kpi-meta">100 Trees • StandardScaler</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- CHARTS ROW ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Fig 1.0 — Random Forest Posterior Probability by Manchester Tier")
    class_names = ["Blue (V)", "Green (IV)", "Yellow (III)", "Orange (II)", "Red (I)"]
    bar_colors = [meta["color"] if i == final_class else "#27272a" for i in range(5)]
    fig_prob = go.Figure(
        go.Bar(
            x=class_names,
            y=[round(p * 100, 1) for p in probs],
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", color="#a1a1aa", size=11),
        yaxis=dict(title="Probability (%)", range=[0, 112], gridcolor="#1c1c21"),
        xaxis=dict(gridcolor="#1c1c21"),
        margin=dict(t=16, r=16, b=36, l=46),
        height=290,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_right:
    st.markdown("#### Fig 2.0 — Global Random Forest Feature Importance (Gini)")
    imp_df = pd.DataFrame(
        {"Feature": FEATURE_LABELS, "Importance": clf.feature_importances_ * 100.0}
    ).sort_values("Importance", ascending=True)
    fig_imp = go.Figure(
        go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color=["#27272a"] * (len(imp_df) - 1) + [meta["color"]],
            text=[f"{v:.1f}%" for v in imp_df["Importance"]],
            textposition="outside",
        )
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", color="#a1a1aa", size=11),
        xaxis=dict(title="Relative Weight (%)", range=[0, max(imp_df["Importance"]) * 1.22], gridcolor="#1c1c21"),
        yaxis=dict(gridcolor="#1c1c21"),
        margin=dict(t=16, r=36, b=36, l=120),
        height=290,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# --- CLINICAL RED FLAGS ---
st.markdown("#### 04 / Physiological Red Flags Detected (NEWS2 Audit)")
if rule_eval["flags"]:
    for flag in rule_eval["flags"]:
        st.markdown(f'<div class="flag-item">[ALERT] {flag}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="flag-item" style="color:#a1a1aa;">[NOMINAL] No critical vital sign deviations detected.</div>', unsafe_allow_html=True)
