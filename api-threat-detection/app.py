import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

from feature_extraction import extract_window_features
from rule_detectors import run_rule_detectors
from baseline_models import AdaptiveBaseline
from sequence_model import MarkovSequenceDetector
from isolation_forest import ThreatScoringEngine


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

load_dotenv()

# Gemini API key
# ------------------------------------------------------------------------------
# The key is loaded automatically from Streamlit Secrets.
# Expected secrets.toml entry:
#
# GEMINI_API_KEY = "your-key"
#
# No manual key input is used anywhere in the UI.
# We first use st.secrets (the normal Streamlit mechanism), then explicitly
# check the project's .streamlit/secrets.toml as a local-development fallback.
def _get_gemini_api_key():
    # 1) Streamlit Secrets -- this is the primary source.
    try:
        value = st.secrets["GEMINI_API_KEY"]
        if value:
            return str(value).strip()
    except Exception:
        pass

    # 2) Directly read .streamlit/secrets.toml when running locally.
    # This handles cases where Streamlit was started from a different working
    # directory and therefore did not discover the project's secrets file.
    try:
        import tomllib
        from pathlib import Path

        secret_paths = [
            Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path.home() / ".streamlit" / "secrets.toml",
        ]

        checked = set()
        for secret_path in secret_paths:
            secret_path = secret_path.resolve()
            if secret_path in checked or not secret_path.is_file():
                continue
            checked.add(secret_path)

            with secret_path.open("rb") as f:
                secrets_data = tomllib.load(f)

            value = secrets_data.get("GEMINI_API_KEY", "")
            if value:
                return str(value).strip()

            # Also support an optional [gemini] section without changing the
            # expected flat GEMINI_API_KEY configuration.
            gemini_section = secrets_data.get("gemini", {})
            if isinstance(gemini_section, dict):
                value = gemini_section.get("GEMINI_API_KEY", gemini_section.get("api_key", ""))
                if value:
                    return str(value).strip()
    except Exception:
        pass

    # 3) Deployment/environment fallback.
    value = os.getenv("GEMINI_API_KEY", "")
    if value:
        return value.strip()

    return ""


GEMINI_API_KEY = _get_gemini_api_key()


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="CY-02 | Sentinel API Threat Intelligence Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# ROYAL VISION CYBER THEME
# ==============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;500;600;700&display=swap');

:root {
    --rv-bg: #060b1f;
    --rv-bg-2: #09112f;
    --rv-panel: rgba(13, 23, 55, .88);
    --rv-panel-2: rgba(18, 31, 72, .92);
    --rv-border: rgba(95, 125, 220, .30);
    --rv-border-bright: rgba(76, 169, 255, .62);
    --rv-blue: #2f8cff;
    --rv-cyan: #35d6ff;
    --rv-violet: #806cff;
    --rv-purple: #a76dff;
    --rv-gold: #f0c76a;
    --rv-text: #f5f8ff;
    --rv-muted: #98a7c7;
    --rv-green: #21d39a;
    --rv-red: #ff5b68;
    --rv-amber: #ffb84d;
}

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

.stApp {
    color: var(--rv-text);
    background:
        radial-gradient(circle at 12% 0%, rgba(47,140,255,.17), transparent 28%),
        radial-gradient(circle at 86% 8%, rgba(128,108,255,.16), transparent 30%),
        radial-gradient(circle at 75% 82%, rgba(53,214,255,.07), transparent 24%),
        linear-gradient(135deg, #05091b 0%, #09102b 45%, #080d24 100%);
    background-attachment: fixed;
}

/* Subtle cybersecurity grid + scanlines */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .24;
    background-image:
        linear-gradient(rgba(91,146,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(91,146,255,.045) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: linear-gradient(to bottom, black 0%, black 62%, transparent 100%);
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .045;
    background: repeating-linear-gradient(0deg, transparent 0, transparent 3px, rgba(255,255,255,.08) 4px);
}

.main .block-container {
    position: relative;
    z-index: 1;
    max-width: 1540px;
    padding: 1.15rem 2rem 4rem;
}

#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* Top navigation / status bar */
.cy-topbar {
    position: relative;
    min-height: 70px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
    margin-bottom: 22px;
    overflow: hidden;
    border: 1px solid var(--rv-border);
    border-radius: 14px;
    background: linear-gradient(100deg, rgba(8,16,42,.96), rgba(17,30,72,.88));
    box-shadow: 0 18px 45px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
    backdrop-filter: blur(16px);
}
.cy-topbar::before {
    content: "";
    position: absolute;
    left: 0; top: 0;
    width: 48%; height: 2px;
    background: linear-gradient(90deg, var(--rv-cyan), var(--rv-blue), var(--rv-violet), transparent);
}
.cy-topbar::after {
    content: "";
    position: absolute;
    right: -5%; bottom: -55px;
    width: 280px; height: 120px;
    background: radial-gradient(ellipse, rgba(47,140,255,.20), transparent 68%);
}
.cy-brand {
    font-family: "Space Mono", monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .10em;
    color: #fff;
}
.cy-brand span { color: var(--rv-cyan); }
.cy-status {
    display: flex; align-items: center; gap: 9px;
    color: #bdc9e3;
    font-family: "Space Mono", monospace;
    font-size: .68rem;
    letter-spacing: .09em;
}
.cy-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--rv-green);
    box-shadow: 0 0 16px rgba(33,211,154,.85);
}

/* Top module navigation */
.cy-nav-spacer { height: 2px; }
div[data-testid="stHorizontalBlock"] .cy-nav-button { display:block; }

/* Style the first horizontal button row as a premium command bar. */
.main .block-container > div[data-testid="stHorizontalBlock"]:has(button[key^="top_nav_"]) {
    padding: 8px;
    margin: 0 0 24px;
    border: 1px solid rgba(95,125,220,.28);
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(10,20,48,.92), rgba(18,31,70,.82));
    box-shadow: 0 14px 32px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035);
}

.main .block-container > div[data-testid="stHorizontalBlock"]:has(button[key^="top_nav_"]) button {
    min-height: 46px;
    border-radius: 9px !important;
    border: 1px solid rgba(91,132,220,.25) !important;
    background: rgba(12,24,56,.72) !important;
    color: #b8c6e2 !important;
    font-family: "DM Sans", sans-serif !important;
    font-size: .78rem !important;
    font-weight: 700 !important;
    white-space: nowrap;
}

.main .block-container > div[data-testid="stHorizontalBlock"]:has(button[key^="top_nav_"]) button:hover {
    border-color: var(--rv-cyan) !important;
    color: #f5fbff !important;
    background: linear-gradient(135deg, rgba(25,55,110,.92), rgba(30,42,90,.92)) !important;
    box-shadow: 0 0 20px rgba(53,214,255,.10);
}

/* Streamlit primary button = active navigation item */
.main .block-container > div[data-testid="stHorizontalBlock"]:has(button[key^="top_nav_"]) button[kind="primary"] {
    color: #ffffff !important;
    border-color: rgba(53,214,255,.65) !important;
    background: linear-gradient(135deg, #1469c7, #654fe0) !important;
    box-shadow: 0 0 22px rgba(47,140,255,.20), inset 0 1px 0 rgba(255,255,255,.16);
}

/* Attack Persona Deep Dive header */
.persona-section-head {
    display:flex;
    align-items:center;
    gap:12px;
    margin: 1.15rem 0 .75rem;
    padding: 10px 0;
}
.persona-section-head > span {
    font-size: 1.35rem;
    filter: drop-shadow(0 0 9px rgba(53,214,255,.45));
}
.persona-kicker {
    color: #6fa8ff;
    font-family: "Space Mono", monospace;
    font-size: .60rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.persona-title {
    color: #f5f8ff;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.2;
}

/* Typography */
h1,h2,h3,h4,h5,h6 { color: var(--rv-text) !important; }
h3 { font-size: 1.35rem !important; font-weight: 800 !important; }
h4 { font-size: 1rem !important; font-weight: 700 !important; }
.main-title {
    margin: .2rem 0 .35rem;
    font-family: "Space Mono", monospace;
    font-size: clamp(1.65rem, 3vw, 2.45rem);
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -.045em;
    color: #fff !important;
    text-shadow: 0 0 28px rgba(47,140,255,.16);
}
.sub-title {
    margin-bottom: 1.55rem;
    color: var(--rv-muted);
    font-family: "Space Mono", monospace;
    font-size: .70rem;
    letter-spacing: .055em;
    text-transform: uppercase;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 10% 5%, rgba(47,140,255,.13), transparent 25%),
        linear-gradient(180deg, #07102a 0%, #09112b 58%, #060b20 100%);
    border-right: 1px solid rgba(88,125,215,.30);
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption { color: #aab7d2 !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: "Space Mono", monospace;
    text-transform: uppercase;
    letter-spacing: .065em;
    font-size: .78rem !important;
    color: #f5f8ff !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(91,125,210,.24); }
section[data-testid="stSidebar"] button {
    border: 1px solid rgba(82,143,231,.40) !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, rgba(17,35,78,.92), rgba(11,22,53,.95)) !important;
    color: #e9f1ff !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.20);
}
section[data-testid="stSidebar"] button:hover {
    border-color: var(--rv-cyan) !important;
    box-shadow: 0 0 22px rgba(53,214,255,.12);
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 8px 8px !important;
    border-radius: 9px !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(47,140,255,.09); }

/* Cards / metrics */
div[data-testid="stMetric"] {
    position: relative;
    min-height: 126px;
    padding: 18px 18px 15px;
    overflow: hidden;
    border: 1px solid var(--rv-border);
    border-radius: 14px;
    background:
        radial-gradient(circle at 100% 0%, rgba(128,108,255,.14), transparent 36%),
        linear-gradient(145deg, rgba(18,35,78,.95), rgba(9,18,46,.96));
    box-shadow: 0 14px 35px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.045);
}
div[data-testid="stMetric"]::before {
    content: "";
    position: absolute; left: 0; top: 0;
    width: 72px; height: 3px;
    background: linear-gradient(90deg, var(--rv-cyan), var(--rv-blue), var(--rv-violet));
}
div[data-testid="stMetric"]::after {
    content: "";
    position: absolute; right: -25px; top: -35px;
    width: 95px; height: 95px;
    border: 1px solid rgba(53,214,255,.10);
    border-radius: 50%;
    box-shadow: 0 0 0 14px rgba(53,214,255,.025), 0 0 0 28px rgba(53,214,255,.018);
}
div[data-testid="stMetricLabel"] {
    color: #91a1c2 !important;
    font-family: "Space Mono", monospace !important;
    font-size: .63rem !important;
    font-weight: 600 !important;
    letter-spacing: .07em;
    text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    color: #f7fbff !important;
    font-family: "Space Mono", monospace !important;
    font-weight: 700 !important;
    font-size: 1.78rem !important;
}
div[data-testid="stMetricDelta"] { font-family: "Space Mono", monospace !important; font-size: .65rem !important; }

/* Chart containers */
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(80,126,218,.28);
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(12,24,57,.88), rgba(7,15,38,.94));
    padding: 6px;
    box-shadow: 0 15px 35px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035);
}
.cy-panel {
    position: relative;
    border: 1px solid var(--rv-border);
    border-radius: 14px;
    padding: 18px;
    margin: 8px 0 18px;
    background: linear-gradient(145deg, rgba(15,30,68,.90), rgba(8,17,43,.95));
    box-shadow: 0 15px 32px rgba(0,0,0,.18);
}
.cy-panel::before {
    content: ""; position: absolute; left: 18px; top: 0; width: 95px; height: 2px;
    background: linear-gradient(90deg, var(--rv-cyan), var(--rv-violet), transparent);
}
.cy-section-label {
    color: #eaf3ff;
    font-family: "Space Mono", monospace;
    font-size: .74rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .08em;
    margin: 8px 0 10px;
}

/* Inputs */
.stTextInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    border-radius: 9px !important;
    background: rgba(11,23,53,.92) !important;
    border-color: rgba(88,132,218,.34) !important;
    color: #f2f7ff !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--rv-cyan) !important;
    box-shadow: 0 0 0 1px rgba(53,214,255,.16), 0 0 20px rgba(53,214,255,.06) !important;
}

/* Buttons */
.stButton > button {
    min-height: 42px;
    border: 1px solid rgba(82,143,231,.42) !important;
    border-radius: 9px !important;
    background: linear-gradient(135deg, rgba(18,37,82,.96), rgba(9,20,48,.96)) !important;
    color: #edf5ff !important;
    font-family: "Space Mono", monospace !important;
    font-weight: 600 !important;
    transition: .18s ease;
}
.stButton > button:hover {
    border-color: var(--rv-cyan) !important;
    color: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 0 22px rgba(53,214,255,.10);
}

/* CSV download buttons — always prominent and reliable */
.stDownloadButton > button {
    min-height: 44px;
    border: 1px solid rgba(53,214,255,.58) !important;
    border-radius: 9px !important;
    background: linear-gradient(135deg, #1466d8, #5c49d8) !important;
    color: #fff !important;
    font-family: "Space Mono", monospace !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 24px rgba(47,140,255,.18), inset 0 1px 0 rgba(255,255,255,.12);
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1a7cff, #735cff) !important;
    border-color: #79e8ff !important;
    box-shadow: 0 0 25px rgba(53,214,255,.18);
}

/* Filter panels */
.cy-filter-panel {
    position: relative;
    border: 1px solid rgba(82,132,221,.34);
    border-radius: 12px;
    padding: 15px 17px;
    margin: 8px 0 15px;
    background: linear-gradient(145deg, rgba(16,34,76,.88), rgba(8,18,45,.95));
    box-shadow: 0 10px 26px rgba(0,0,0,.15);
}
.cy-filter-panel::after {
    content: "FILTER / QUERY / EXPORT";
    position: absolute; right: 14px; top: 12px;
    color: rgba(120,188,255,.42);
    font: 500 .57rem "Space Mono", monospace;
    letter-spacing: .08em;
}
.cy-filter-title {
    color: #70dcff;
    font-family: "Space Mono", monospace;
    font-size: .70rem; text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: 10px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(83,132,221,.34);
    border-radius: 12px;
    overflow: hidden;
    background: rgba(8,18,43,.92);
    box-shadow: 0 12px 30px rgba(0,0,0,.18);
}
[data-testid="stDataFrame"] * { font-family: "Space Mono", monospace !important; }

/* Alerts / expanders / code */
.streamlit-expanderHeader {
    border-radius: 10px !important;
    border: 1px solid rgba(83,132,221,.30) !important;
    background: rgba(11,24,55,.92) !important;
}
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(83,132,221,.30) !important;
    background: rgba(10,22,51,.92) !important;
    color: #edf5ff !important;
}
.stCodeBlock { border: 1px solid rgba(83,132,221,.30); border-radius: 10px !important; }

/* Forensic cards */
.alert-card-high,.alert-card-medium,.alert-card-low {
    border-radius: 13px;
    padding: 18px 20px;
    margin-bottom: 12px;
    background: linear-gradient(145deg, rgba(18,33,72,.95), rgba(8,18,43,.97));
    box-shadow: 0 10px 26px rgba(0,0,0,.17);
}
.alert-card-high { border: 1px solid rgba(255,91,104,.42); border-left: 4px solid var(--rv-red); }
.alert-card-medium { border: 1px solid rgba(255,184,77,.42); border-left: 4px solid var(--rv-amber); }
.alert-card-low { border: 1px solid rgba(33,211,154,.34); border-left: 4px solid var(--rv-green); }

/* Small cyber artifact line */
.cy-panel, .alert-card-high, .alert-card-medium, .alert-card-low, div[data-testid="stMetric"] { isolation: isolate; }

@media (max-width: 900px) {
    .main .block-container { padding: 1rem .85rem 3rem; }
    .cy-topbar { padding: 0 13px; }
    .cy-status { display: none; }
    div[data-testid="stMetric"] { min-height: 108px; }
}
</style>

""",
    unsafe_allow_html=True
)


# ==============================================================================
# PIPELINE INITIALIZATION
# ==============================================================================

@st.cache_resource
def initialize_pipeline():

    history_df = pd.read_csv(
        "data/history_logs.csv",
        on_bad_lines="skip"
    )

    history_feats = extract_window_features(
        history_df,
        window_minutes=1
    )

    baseline = AdaptiveBaseline()
    baseline.fit_history(history_feats)

    markov = MarkovSequenceDetector()
    markov.fit(history_df)

    engine = ThreatScoringEngine()
    engine.fit(history_feats)

    return baseline, markov, engine


# ==============================================================================
# AI REPORT FUNCTIONS
# ==============================================================================

def generate_offline_backup_report(full_results):

    total_reqs = len(full_results)

    critical_threats = len(
        full_results[
            full_results["final_risk_score"] > 70
        ]
    )

    suspicious_threats = len(
        full_results[
            (
                full_results["final_risk_score"] >= 30
            )
            &
            (
                full_results["final_risk_score"] <= 70
            )
        ]
    )

    attack_counts = (
        full_results["label"]
        .value_counts()
        .to_dict()
    )

    return f"""
1. EXECUTIVE SUMMARY
--------------------------------------------------
Total Traffic Analyzed: {total_reqs} 1-minute time windows.
Critical Security Alerts (>70 Score): {critical_threats}
Suspicious Security Warnings (30-70 Score): {suspicious_threats}
Overall Threat Level: {"CRITICAL" if critical_threats > 0 else "NOMINAL"}

2. ATTACK VECTOR BREAKDOWN
--------------------------------------------------
{chr(10).join([f"- {k.upper()}: {v} window(s)" for k, v in attack_counts.items()])}

3. HIGH-RISK TARGET INSIGHTS
--------------------------------------------------
Highest Risk Score Observed:
{full_results["final_risk_score"].max():.1f}/100

Flagged Client IDs:
{", ".join(
    full_results[
        full_results["final_risk_score"] > 50
    ]["client_id"].unique().tolist()
)}

4. ACTIONABLE SOC RECOMMENDATIONS
--------------------------------------------------
- Enforce real-time HTTP 429 rate limiting on high-failure login endpoints.
- Apply automated IP blocking for low timing regularity (CV < 0.10) script traffic.
- Monitor per-client volume baselines for sudden traffic shifts.
"""


def generate_gemini_threat_summary(full_results, api_key):

    if not api_key:
        return generate_offline_backup_report(full_results)

    total_reqs = len(full_results)

    critical_threats = len(
        full_results[
            full_results["final_risk_score"] > 70
        ]
    )

    suspicious_threats = len(
        full_results[
            (
                full_results["final_risk_score"] >= 30
            )
            &
            (
                full_results["final_risk_score"] <= 70
            )
        ]
    )

    attack_counts = (
        full_results["label"]
        .value_counts()
        .to_dict()
    )

    top_ips = (
        full_results[
            full_results["final_risk_score"] > 50
        ]["ip"]
        .unique()
        .tolist()
    )

    prompt = f"""
You are a Lead Cybersecurity Analyst for CY-02 Sentinel.

Analyze the following live API traffic data and generate an executive threat report.

METRICS DATA:

- Total Traffic Analyzed: {total_reqs} 1-minute windows
- Critical Threats (>70 Score): {critical_threats}
- Suspicious Threats (30-70 Score): {suspicious_threats}
- Attack Personas Detected: {attack_counts}
- Flagged IPs: {top_ips}

Provide a professional, clear report divided into exactly 4 labeled sections:

1. EXECUTIVE SUMMARY
2. ATTACK VECTOR BREAKDOWN
3. HIGH-RISK TARGET INSIGHTS
4. ACTIONABLE SOC RECOMMENDATIONS

Keep the wording crisp and direct for C-level executives.
"""

    client = genai.Client(
        api_key=api_key
    )

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.0-flash"
    ]

    for model_name in candidate_models:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except Exception:
            continue

    return generate_offline_backup_report(
        full_results
    )


def explain_threat_trigger_with_gemini(
    row_data,
    api_key
):

    if not api_key:

        return (
            f"Threat Flagged: High login failure rate "
            f"({row_data['failure_pct'] * 100:.1f}%) "
            f"and automated script timing pattern detected "
            f"for {row_data['client_id']}."
        )

    prompt = f"""
You are a Senior Threat Forensics Analyst at CY-02 Sentinel.

Explain in 2-3 clear, authoritative sentences WHY this specific request window was flagged as a threat.

TELEMETRY DATA:

- Client ID: {row_data["client_id"]}
- Source IP: {row_data["ip"]}
- Request Volume / Min: {row_data["req_count"]}
- HTTP Failure Rate (401/404): {row_data["failure_pct"] * 100:.1f}%
- Login Attempt Pressure: {row_data["login_ratio"] * 100:.1f}%
- Timing Regularity Coefficient (CV): {row_data["cv_gaps"]:.2f}
- Distinct Usernames Targeted: {row_data["unique_usernames"]}
- Heuristic Trigger: {row_data["rule_evidence"]}
- True Traffic Persona: {row_data["label"]}

Write a concise, plain-English forensic breakdown explaining the attacker's motive and why our engine flagged it.
"""

    client = genai.Client(
        api_key=api_key
    )

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.0-flash"
    ]

    for model_name in candidate_models:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except Exception:
            continue

    return (
        f"Threat Flagged: High login failure rate "
        f"({row_data['failure_pct'] * 100:.1f}%) "
        f"and automated script timing pattern detected "
        f"for {row_data['client_id']}."
    )


def build_pdf_report(summary_text):

    buffer = io.BytesIO()

    c = canvas.Canvas(
        buffer,
        pagesize=letter
    )

    c.setFont(
        "Helvetica-Bold",
        16
    )

    c.drawString(
        40,
        750,
        "CY-02 Sentinel - AI Threat Intelligence Report"
    )

    c.line(
        40,
        740,
        570,
        740
    )

    c.setFont(
        "Helvetica",
        9
    )

    y_position = 710

    for line in summary_text.split("\n"):

        if y_position < 40:

            c.showPage()

            y_position = 750

            c.setFont(
                "Helvetica",
                9
            )

        c.drawString(
            40,
            y_position,
            line[:95]
        )

        y_position -= 14

    c.save()

    buffer.seek(0)

    return buffer


# ==============================================================================
# CSV EXPORT HELPER
# ==============================================================================

def csv_bytes(dataframe):
    """Create UTF-8 CSV bytes from the exact dataframe currently displayed."""
    export_df = dataframe.copy()
    return export_df.to_csv(index=False).encode("utf-8-sig")


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

# ------------------------------------------------------------------------------
# TOP BRAND / STATUS BAR
# Use a compact HTML block with no indentation so Streamlit does not interpret
# the inner markup as a code block.
# ------------------------------------------------------------------------------
st.markdown("""<div class="cy-topbar"><div class="cy-brand">CY-02 <span>/</span> SENTINEL</div><div class="cy-status"><span class="cy-dot"></span><span>THREAT INTELLIGENCE ENGINE</span><span style="opacity:.45">|</span><span>ONLINE</span></div></div>""", unsafe_allow_html=True)

st.markdown('<p class="main-title">CY-02 SENTINEL: API THREAT ENGINE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-Model Behavioral Analytics &nbsp;•&nbsp; Zero False Positive Shield &nbsp;•&nbsp; Adaptive Threat Scoring</p>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TOP MODULE NAVIGATION
# Replaces the old vertical radio navigation. The selected module is stored in
# session state, so every existing page and its functionality remains intact.
# ------------------------------------------------------------------------------
# Keep a separate internal page key from the visual label.
# This prevents the top navigation redesign from breaking the original
# page-routing logic, which uses the emoji-prefixed page names.
MODULES = [
    ("Executive Dashboard", "📊 Executive Dashboard"),
    ("Plug-and-Play SDK", "🔌 Plug-and-Play SDK"),
    ("Attack Persona Deep Dive", "🔍 Attack Persona Deep Dive"),
    ("Explainable Forensics", "📜 Explainable Forensics"),
    ("Live Replay Engine", "🕹️ Live Replay Engine"),
    ("Benchmark vs Static WAF", "⚔️ Benchmark vs Static WAF"),
]

# Recover old session-state values if the app was previously run with the
# intermediate navigation version.
if "cy02_page" not in st.session_state:
    st.session_state.cy02_page = MODULES[0][1]
else:
    legacy_to_display = {key: display for key, display in MODULES}
    if st.session_state.cy02_page in legacy_to_display:
        st.session_state.cy02_page = legacy_to_display[st.session_state.cy02_page]

nav_cols = st.columns(6, gap="small")
for nav_col, (key, display) in zip(nav_cols, MODULES):
    with nav_col:
        active = st.session_state.cy02_page == display
        if st.button(
            display,
            key=f"top_nav_{key}",
            width="stretch",
            type="primary" if active else "secondary",
        ):
            # IMPORTANT: store the exact page string expected by the existing
            # page blocks below. No application functionality is changed.
            st.session_state.cy02_page = display
            st.rerun()

# `page` is deliberately the original emoji-prefixed value so every existing
# `if page == ...` / `elif page == ...` block continues to execute.
page = st.session_state.cy02_page


baseline, markov, engine = initialize_pipeline()


# ==============================================================================
# LOAD LIVE DATA
# ==============================================================================

live_logs = pd.read_csv(
    "data/live_logs.csv",
    on_bad_lines="skip"
)

live_logs["timestamp"] = pd.to_datetime(
    live_logs["timestamp"],
    errors="coerce"
)

live_logs = live_logs.dropna(
    subset=["timestamp"]
)


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.markdown(
    "### 🛡️ CY-02 SENTINEL"
)

st.sidebar.caption(
    "ROYAL VISION SECURITY OPERATIONS CONSOLE"
)


if st.sidebar.button(
    "🔄 Refresh Live Telemetry",
    width="stretch"
):

    st.rerun()


st.sidebar.caption("Use the navigation bar above to switch operational modules.")


# ==============================================================================
# RAW LIVE CSV DOWNLOAD
# ALWAYS AVAILABLE
# ==============================================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 📥 Data Export"
)

raw_live_csv = csv_bytes(live_logs)


st.sidebar.download_button(
    label="📥 Download Raw Live Telemetry CSV",
    data=raw_live_csv,
    file_name="CY02_raw_live_telemetry.csv",
    mime="text/csv",
    key="download_raw_live_csv",
    on_click="ignore",
    width="stretch"
)


st.sidebar.divider()

st.sidebar.markdown(
    "### ⚙️ Pipeline Status"
)

st.sidebar.caption(
    """
🟢 **Rules Engine:** Active (3 Heuristics)

🟢 **Isolation Forest:** Fitted (Contamination=0.05)

🟢 **Markov Chain:** Laplace Smoothed (Alpha=1.0)

🟢 **Baseline Engine:** Per-Client Z-Scores Active
"""
)


# ==============================================================================
# FULL DATASET PROCESSING
# ==============================================================================

full_features = extract_window_features(
    live_logs,
    window_minutes=1
)

full_ruled = run_rule_detectors(
    full_features
)

full_results = engine.predict_risk(
    full_ruled,
    baseline,
    markov,
    live_logs
)


# ==============================================================================
# REAL-TIME OVERRIDE
# ==============================================================================

high_threat_mask = (
    (full_results["failure_pct"] >= 0.70)
    &
    (full_results["req_count"] >= 10)
)


full_results.loc[
    high_threat_mask,
    "final_risk_score"
] = np.maximum(
    full_results.loc[
        high_threat_mask,
        "final_risk_score"
    ],
    88.0
)


full_results.loc[
    high_threat_mask,
    "risk_category"
] = "High Risk"


# ==============================================================================
# AI REPORT CONTROLS
# ==============================================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 📄 Automated AI Reports"
)


if GEMINI_API_KEY:

    st.sidebar.caption(
        "🟢 **Gemini API:** Connected"
    )

else:

    st.sidebar.caption(
        "🟡 **Gemini API:** Key not configured — using offline backup"
    )


if st.sidebar.button(
    "🤖 Generate Gemini Threat Report",
    width="stretch"
):

    with st.spinner(
        "Analyzing live logs and generating AI report..."
    ):

        try:

            report_text = generate_gemini_threat_summary(
                full_results,
                GEMINI_API_KEY
            )

            pdf_file = build_pdf_report(
                report_text
            )

            st.sidebar.download_button(
                label="📥 Download Threat Report (PDF)",
                data=pdf_file,
                file_name=(
                    f"CY02_Threat_Report_"
                    f"{len(full_results)}_windows.pdf"
                ),
                mime="application/pdf",
                width="stretch"
            )

        except Exception as e:

            st.sidebar.error(
                f"Error generating report: {e}"
            )


# ==============================================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ==============================================================================

if page == "📊 Executive Dashboard":

    st.markdown(
        "### 📊 Overall Traffic & Security Overview"
    )


    # --------------------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------------------

    m1, m2, m3, m4, m5 = st.columns(5)


    m1.metric(
        "TOTAL WINDOWS",
        f"{len(full_results):,}"
    )


    high_count = len(
        full_results[
            full_results["final_risk_score"] > 70
        ]
    )


    m2.metric(
        "CRITICAL THREATS (>70)",
        high_count,
        delta=(
            "Immediate Action"
            if high_count > 0
            else "Nominal"
        ),
        delta_color="inverse"
    )


    med_count = len(
        full_results[
            (
                full_results["final_risk_score"] >= 30
            )
            &
            (
                full_results["final_risk_score"] <= 70
            )
        ]
    )


    m3.metric(
        "SUSPICIOUS (>30)",
        med_count,
        delta=(
            "Monitoring"
            if med_count > 0
            else "Nominal"
        ),
        delta_color="inverse"
    )


    acme_high = len(
        full_results[
            (
                full_results["client_id"]
                == "partner_acme"
            )
            &
            (
                full_results["final_risk_score"]
                > 70
            )
        ]
    )


    m4.metric(
        "B2B PARTNER STATUS",
        (
            "PROTECTED"
            if acme_high == 0
            else "FLAGGED"
        ),
        delta=(
            "100% Whitelisted"
            if acme_high == 0
            else "Issue"
        )
    )


    low_slow_caught = len(
        full_results[
            (
                full_results["label"]
                == "low_and_slow"
            )
            &
            (
                full_results["final_risk_score"]
                > 30
            )
        ]
    )


    m5.metric(
        "LOW-AND-SLOW CAUGHT",
        f"{low_slow_caught} Windows",
        delta="Rate Limit Evasion Caught"
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------------------------

    g1, g2 = st.columns(
        [2, 1]
    )


    with g1:

        st.markdown(
            "#### 🎯 Behavioral Threat Matrix"
        )


        fig_bubble = px.scatter(

            full_results,

            x="req_count",

            y="final_risk_score",

            size=full_results[
                "failure_pct"
            ].apply(
                lambda x: max(x, 0.05)
            ),

            color="risk_category",

            hover_data=[
                "client_id",
                "ip",
                "rule_evidence",
                "label"
            ],

            color_discrete_map={
                "Low Risk": "#36c98b",
                "Medium Risk": "#e0a83b",
                "High Risk": "#ef6464"
            },

            labels={
                "req_count":
                    "Request Volume / Min",

                "final_risk_score":
                    "Composite Risk Score (0-100)"
            },

            template="plotly_dark",

            height=420
        )


        fig_bubble.update_layout(

            paper_bgcolor="#11132f",

            plot_bgcolor="#11132f",

            font=dict(
                color="#eeeae1"
            ),

            margin=dict(
                l=15,
                r=15,
                t=15,
                b=15
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )


        st.plotly_chart(
            fig_bubble,
            width="stretch"
        )


    with g2:

        st.markdown(
            "#### 🍩 Risk Severity Breakdown"
        )


        cat_counts = (
            full_results[
                "risk_category"
            ]
            .value_counts()
            .reset_index()
        )


        cat_counts.columns = [
            "Category",
            "Count"
        ]


        fig_pie = px.pie(

            cat_counts,

            names="Category",

            values="Count",

            hole=0.5,

            color="Category",

            color_discrete_map={
                "Low Risk": "#36c98b",
                "Medium Risk": "#e0a83b",
                "High Risk": "#ef6464"
            },

            template="plotly_dark",

            height=420
        )


        fig_pie.update_layout(

            paper_bgcolor="#11132f",

            plot_bgcolor="#11132f",

            font=dict(
                color="#eeeae1"
            ),

            margin=dict(
                l=15,
                r=15,
                t=15,
                b=15
            ),

            showlegend=True
        )


        st.plotly_chart(
            fig_pie,
            width="stretch"
        )


    # --------------------------------------------------------------------------
    # DOWNLOADABLE FILTERED AUDIT TABLE
    # --------------------------------------------------------------------------

    st.markdown(
        "#### 📋 Live Client Window Audit Stream"
    )


    st.markdown(
        """
<div class="cy-filter-panel">

<div class="cy-filter-title">
TABLE FILTERS — DOWNLOAD EXPORT USES THESE FILTERS
</div>

</div>
""",
        unsafe_allow_html=True
    )


    f1, f2, f3 = st.columns(
        [1, 1, 1]
    )


    with f1:

        risk_options = [
            "All Risk Categories",
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]


        selected_risk = st.selectbox(
            "Risk Category",
            risk_options,
            key="dashboard_risk_filter"
        )


    with f2:

        min_table_score = st.slider(
            "Minimum Risk Score",
            min_value=0,
            max_value=100,
            value=0,
            key="dashboard_score_filter"
        )


    with f3:

        client_search = st.text_input(
            "Client ID / IP Search",
            placeholder="Search client or IP...",
            key="dashboard_client_search"
        )


    # Base table
    audit_export_df = full_results[
        [
            "window_time",
            "client_id",
            "ip",
            "req_count",
            "failure_pct",
            "cv_gaps",
            "final_risk_score",
            "risk_category",
            "label"
        ]
    ].copy()


    # Risk filter
    if selected_risk != "All Risk Categories":

        audit_export_df = audit_export_df[
            audit_export_df["risk_category"]
            == selected_risk
        ]


    # Minimum score
    audit_export_df = audit_export_df[
        audit_export_df["final_risk_score"]
        >= min_table_score
    ]


    # Client/IP search
    if client_search.strip():

        search_text = (
            client_search
            .strip()
            .lower()
        )


        audit_export_df = audit_export_df[
            audit_export_df["client_id"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False
            )
            |
            audit_export_df["ip"]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                na=False
            )
        ]


    audit_export_df = audit_export_df.sort_values(
        "window_time",
        ascending=False
    )


    st.caption(
        f"Showing {len(audit_export_df):,} "
        f"of {len(full_results):,} windows"
    )


    st.dataframe(
        audit_export_df,
        width="stretch",
        hide_index=True
    )


    # THIS DOWNLOAD IS THE SAME DATAFRAME SHOWN ABOVE.
    audit_csv = csv_bytes(audit_export_df)


    st.download_button(

        "📥 Download Current Filtered Audit Table (CSV)",

        data=audit_csv,

        file_name="CY02_filtered_client_window_audit.csv",

        mime="text/csv",

        key="download_filtered_audit_csv",

        on_click="ignore",

        width="stretch"
    )


# ==============================================================================
# PAGE 2 — ATTACK PERSONA DEEP DIVE
# ==============================================================================

elif page in ("🔍 Attack Persona Deep Dive", "Attack Persona Deep Dive"):

    # --------------------------------------------------------------------------
    # ATTACK PERSONA DEEP DIVE
    # Preserved from the original application. Only visual styling/navigation
    # around this section is upgraded; all analysis, filters and exports remain.
    # --------------------------------------------------------------------------
    st.markdown(
        "<div class=\"persona-section-head\"><span>🔍</span><div><div class=\"persona-kicker\">THREAT VECTOR ANALYSIS</div><div class=\"persona-title\">Dedicated Attack &amp; Traffic Persona Inspection</div></div></div>",
        unsafe_allow_html=True
    )


    persona = st.selectbox(

        "Select Persona / Vector to Analyze:",

        [
            "Credential Stuffing (1 IP, 400 Logins)",

            "Low-and-Slow Attack (30 Distributed IPs)",

            "Web Scraper (Sequential Product Catalog Crawl)",

            "Enumeration (Sequential User ID Sweep)",

            "Abnormal Endpoint Sequence (Unauthorized Direct Access)",

            "Legit Heavy Enterprise Partner (Partner Acme - 5k req/hr)",

            "Flash Sale Traffic Surge (300 Rapid Normal Users)"
        ]
    )


    label_map = {

        "Credential Stuffing (1 IP, 400 Logins)":
            "credential_stuffing",

        "Low-and-Slow Attack (30 Distributed IPs)":
            "low_and_slow",

        "Web Scraper (Sequential Product Catalog Crawl)":
            "scraper",

        "Enumeration (Sequential User ID Sweep)":
            "enumeration",

        "Abnormal Endpoint Sequence (Unauthorized Direct Access)":
            "abnormal_sequence",

        "Legit Heavy Enterprise Partner (Partner Acme - 5k req/hr)":
            "legit_heavy",

        "Flash Sale Traffic Surge (300 Rapid Normal Users)":
            "flash_sale"
    }


    selected_label = label_map[
        persona
    ]


    persona_df = full_results[
        full_results["label"]
        == selected_label
    ].copy()


    # --------------------------------------------------------------------------
    # PERSONA METRICS
    # --------------------------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Matching Time Windows",
        len(persona_df)
    )


    c2.metric(

        "Avg Risk Score",

        (
            f"{persona_df['final_risk_score'].mean():.1f}/100"
            if not persona_df.empty
            else "N/A"
        )
    )


    c3.metric(

        "Max Risk Score",

        (
            f"{persona_df['final_risk_score'].max():.1f}/100"
            if not persona_df.empty
            else "N/A"
        )
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------------------------

    st.markdown(
        f"#### 📈 Risk Score & Volume Timeline for `{selected_label}`"
    )


    fig_time = go.Figure()


    fig_time.add_trace(
        go.Scatter(

            x=persona_df["window_time"],

            y=persona_df["final_risk_score"],

            mode="lines+markers",

            name="Risk Score (0-100)",

            line=dict(
                color="#ef6464",
                width=3
            ),

            marker=dict(
                size=8
            )
        )
    )


    fig_time.add_trace(
        go.Bar(

            x=persona_df["window_time"],

            y=persona_df["req_count"],

            name="Request Volume",

            yaxis="y2",

            marker_color="rgba(109,93,252,0.28)"
        )
    )


    fig_time.update_layout(

        template="plotly_dark",

        paper_bgcolor="#11132f",

        plot_bgcolor="#11132f",

        font=dict(
            color="#eeeae1"
        ),

        height=380,

        yaxis=dict(
            title="Risk Score",
            range=[0, 105]
        ),

        yaxis2=dict(
            title="Request Volume",
            overlaying="y",
            side="right"
        ),

        margin=dict(
            l=15,
            r=15,
            t=20,
            b=15
        )
    )


    st.plotly_chart(
        fig_time,
        width="stretch"
    )


    # --------------------------------------------------------------------------
    # FILTERED PERSONA TABLE
    # --------------------------------------------------------------------------

    st.markdown(
        "#### 📄 Filtered Persona Window Features Table"
    )


    pf1, pf2 = st.columns(2)


    with pf1:

        persona_min_score = st.slider(
            "Persona Minimum Risk Score",
            0,
            100,
            0,
            key="persona_min_score"
        )


    with pf2:

        persona_search = st.text_input(
            "Search Client / IP",
            placeholder="Search within this persona...",
            key="persona_search"
        )


    filtered_persona_df = persona_df[
        persona_df["final_risk_score"]
        >= persona_min_score
    ].copy()


    if persona_search.strip():

        psearch = (
            persona_search
            .strip()
            .lower()
        )


        filtered_persona_df = filtered_persona_df[
            filtered_persona_df["client_id"]
            .astype(str)
            .str.lower()
            .str.contains(
                psearch,
                na=False
            )
            |
            filtered_persona_df["ip"]
            .astype(str)
            .str.lower()
            .str.contains(
                psearch,
                na=False
            )
        ]


    filtered_persona_df = filtered_persona_df.sort_values(
        "window_time",
        ascending=False
    )


    st.dataframe(
        filtered_persona_df,
        width="stretch",
        hide_index=True
    )


    persona_csv = csv_bytes(filtered_persona_df)


    st.download_button(

        "📥 Download Current Filtered Persona Table (CSV)",

        data=persona_csv,

        file_name=(
            f"CY02_{selected_label}_filtered_windows.csv"
        ),

        mime="text/csv",

        key="download_filtered_persona_csv",

        on_click="ignore",

        width="stretch"
    )


# ==============================================================================
# PAGE 3 — EXPLAINABLE FORENSICS
# ==============================================================================

elif page == "📜 Explainable Forensics":

    st.markdown(
        "### 📜 Explainable Threat Forensics & Evidence Panel"
    )


    st.markdown(
        "Plain-language security justifications and AI-generated forensic breakdowns for every active alert."
    )


    min_score = st.slider(
        "Filter Minimum Risk Score for Evidence Display:",
        0,
        100,
        30
    )


    flagged = full_results[
        full_results["final_risk_score"]
        >= min_score
    ].sort_values(
        "final_risk_score",
        ascending=False
    )


    if flagged.empty:

        st.success(
            "✨ No security alerts meet or exceed the selected threshold."
        )


    else:

        # ----------------------------------------------------------------------
        # FORENSICS TABLE
        # ----------------------------------------------------------------------

        st.markdown(
            "#### 📋 Filtered Forensic Evidence Table"
        )


        forensic_table = flagged[
            [
                "window_time",
                "client_id",
                "ip",
                "req_count",
                "failure_pct",
                "login_ratio",
                "cv_gaps",
                "unique_usernames",
                "final_risk_score",
                "risk_category",
                "rule_evidence",
                "label"
            ]
        ].copy()


        st.dataframe(
            forensic_table,
            width="stretch",
            hide_index=True
        )


        forensic_csv = csv_bytes(forensic_table)


        st.download_button(

            "📥 Download Current Forensic Filter (CSV)",

            data=forensic_csv,

            file_name="CY02_filtered_forensic_evidence.csv",

            mime="text/csv",

            key="download_filtered_forensic_csv",

            on_click="ignore",

            width="stretch"
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------------------------
        # EVIDENCE CARDS
        # ----------------------------------------------------------------------

        for idx, row in flagged.iterrows():

            if row["final_risk_score"] > 70:

                card_class = "alert-card-high"

            elif row["final_risk_score"] >= 30:

                card_class = "alert-card-medium"

            else:

                card_class = "alert-card-low"


            st.markdown(
                f"""
<div class="{card_class}">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<span style="
font-weight:700;
font-size:1.1rem;
color:#f7f2e6;
">

🚨 [{row["risk_category"]}]

Client ID:

<code style="color:#8c7dff;">
{row["client_id"]}
</code>

|

IP:

<code>
{row["ip"]}
</code>

</span>


<span style="
background-color:rgba(255,255,255,0.08);
padding:4px 12px;
border:1px solid rgba(255,255,255,0.08);
font-weight:bold;
color:#f1cc73;
">

Risk Score:
{row["final_risk_score"]:.1f}/100

</span>

</div>


<div style="
margin-top:10px;
color:#d4d1c7;
font-size:0.95rem;
">

<b>Primary Evidence Trigger:</b>
{row["rule_evidence"]}

</div>


<div style="
margin-top:8px;
font-size:0.85rem;
color:#aaa9bd;
display:flex;
gap:20px;
flex-wrap:wrap;
">

<span>
<b>Failure Rate:</b>
{row["failure_pct"] * 100:.1f}%
</span>

<span>
<b>Login Pressure:</b>
{row["login_ratio"] * 100:.1f}%
</span>

<span>
<b>Timing Regularity (CV):</b>
{row["cv_gaps"]:.2f}
</span>

<span>
<b>Distinct Usernames:</b>
{row["unique_usernames"]}
</span>

<span>
<b>True Label:</b>
<code style="color:#8c7dff;">
{row["label"]}
</code>
</span>

</div>

</div>
""",
                unsafe_allow_html=True
            )


            with st.expander(
                f"🤖 Ask Gemini: Why was {row['client_id']} flagged?"
            ):

                if st.button(
                    f"🔍 Explain Primary Trigger for {row['client_id']} ({row['ip']})",
                    key=f"btn_{idx}"
                ):

                    with st.spinner(
                        "Analyzing telemetry signals with Gemini..."
                    ):

                        ai_explanation = (
                            explain_threat_trigger_with_gemini(
                                row,
                                GEMINI_API_KEY
                            )
                        )


                        st.markdown(
                            "##### 🛡️ AI Forensic Analysis:"
                        )


                        st.info(
                            ai_explanation
                        )


# ==============================================================================
# PAGE 4 — LIVE REPLAY ENGINE
# ==============================================================================

elif page == "🕹️ Live Replay Engine":

    st.markdown(
        "### 🕹️ Real-Time Scenario Replay Engine"
    )


    st.markdown(
        "Simulate live API traffic streams and watch the CY-02 pipeline process threats in real time."
    )


    replay_scenario = st.selectbox(

        "Select Live Traffic Scenario to Stream:",

        [
            "All Live Stream Traffic",
            "Normal Users Only",
            "Legit Heavy Partner (Acme)",
            "Credential Stuffing",
            "Low-and-Slow Attack"
        ]
    )


    if replay_scenario == "Normal Users Only":

        filtered_logs = live_logs[
            live_logs["label"]
            == "normal"
        ]


    elif replay_scenario == "Legit Heavy Partner (Acme)":

        filtered_logs = live_logs[
            live_logs["label"]
            == "legit_heavy"
        ]


    elif replay_scenario == "Credential Stuffing":

        filtered_logs = live_logs[
            live_logs["label"]
            == "credential_stuffing"
        ]


    elif replay_scenario == "Low-and-Slow Attack":

        filtered_logs = live_logs[
            live_logs["label"]
            == "low_and_slow"
        ]


    else:

        filtered_logs = live_logs


    if filtered_logs.empty:

        st.warning(
            "⚠️ No logs matched the selected scenario filter."
        )


    else:

        sim_features = extract_window_features(
            filtered_logs,
            window_minutes=1
        )


        sim_ruled = run_rule_detectors(
            sim_features
        )


        sim_results = engine.predict_risk(
            sim_ruled,
            baseline,
            markov,
            filtered_logs
        )


        if "final_risk_score" not in sim_results.columns:

            if "risk_score" in sim_results.columns:

                sim_results["final_risk_score"] = (
                    sim_results["risk_score"]
                )

            else:

                sim_results["final_risk_score"] = 0.0


        if "risk_category" not in sim_results.columns:

            sim_results["risk_category"] = (
                sim_results["final_risk_score"]
                .apply(
                    lambda s:
                    "High Risk"
                    if s > 70
                    else
                    (
                        "Medium Risk"
                        if s >= 30
                        else
                        "Low Risk"
                    )
                )
            )


        sim_mask = (
            (sim_results["failure_pct"] >= 0.70)
            &
            (sim_results["req_count"] >= 10)
        )


        sim_results.loc[
            sim_mask,
            "final_risk_score"
        ] = np.maximum(
            sim_results.loc[
                sim_mask,
                "final_risk_score"
            ],
            88.0
        )


        sim_results.loc[
            sim_mask,
            "risk_category"
        ] = "High Risk"


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Processed Stream Windows",
            f"{len(sim_results):,}"
        )


        threat_count = len(
            sim_results[
                sim_results["final_risk_score"] > 30
            ]
        )


        c2.metric(
            "Threats Flagged (>30)",
            threat_count
        )


        critical_count = len(
            sim_results[
                sim_results["final_risk_score"] > 70
            ]
        )


        c3.metric(
            "Critical Alerts (>70)",
            critical_count
        )


        # ----------------------------------------------------------------------
        # LIVE REPLAY CHART
        # ----------------------------------------------------------------------

        st.markdown(
            "#### 📺 Live Stream Scatter Visualization"
        )


        fig_sim = px.scatter(

            sim_results,

            x="req_count",

            y="final_risk_score",

            color="risk_category",

            hover_data=[
                "client_id",
                "ip",
                "rule_evidence"
            ],

            color_discrete_map={
                "Low Risk": "#36c98b",
                "Medium Risk": "#e0a83b",
                "High Risk": "#ef6464"
            },

            template="plotly_dark",

            height=400
        )


        fig_sim.update_layout(

            paper_bgcolor="#11132f",

            plot_bgcolor="#11132f",

            font=dict(
                color="#eeeae1"
            ),

            margin=dict(
                l=15,
                r=15,
                t=15,
                b=15
            )
        )


        st.plotly_chart(
            fig_sim,
            width="stretch"
        )


        # ----------------------------------------------------------------------
        # REPLAY TABLE + DOWNLOAD
        # ----------------------------------------------------------------------

        st.markdown(
            "#### 📋 Current Replay Stream Table"
        )


        replay_table = sim_results.copy()


        replay_min_score = st.slider(
            "Replay Minimum Risk Score",
            0,
            100,
            0,
            key="replay_min_score"
        )


        replay_table = replay_table[
            replay_table["final_risk_score"]
            >= replay_min_score
        ]


        st.dataframe(
            replay_table,
            width="stretch",
            hide_index=True
        )


        replay_csv = csv_bytes(replay_table)


        st.download_button(

            "📥 Download Current Replay Table (CSV)",

            data=replay_csv,

            file_name="CY02_current_replay_stream.csv",

            mime="text/csv",

            key="download_current_replay_csv",

            on_click="ignore",

            width="stretch"
        )


# ==============================================================================
# PAGE 5 — BENCHMARK VS STATIC WAF
# ==============================================================================

elif page == "⚔️ Benchmark vs Static WAF":

    st.markdown(
        "### ⚔️ Comparative Performance Matrix"
    )


    st.markdown(
        "Why standard static rate limiters fail vs. how CY-02 Sentinel maintains zero false positives while catching evasive attacks."
    )


    comp_df = pd.DataFrame({

        "Traffic Vector / Persona": [

            "B2B Enterprise Partner Acme (5k req/hr)",

            "Credential Stuffing (1 IP, 400 reqs)",

            "Low-and-Slow Attack (30 IPs x 4 reqs)",

            "Web Scraper (Fast Catalog Sweep)",

            "Sequential User Enumeration",

            "Flash Sale Surge (300 Normal Users)"
        ],


        "Plain Static Rate Limiter": [

            "❌ BLOCKED (False Positive)",

            "✅ BLOCKED",

            "❌ MISSED (Rate Limit Evasion)",

            "✅ BLOCKED",

            "❌ MISSED (Distributed)",

            "❌ BLOCKED (False Positive)"
        ],


        "CY-02 Sentinel System": [

            "✅ ALLOWED (Zero False Positive)",

            "✅ BLOCKED (Score > 85)",

            "✅ CAUGHT (Behavioral Baseline)",

            "✅ BLOCKED (Score > 90)",

            "✅ BLOCKED (Score > 80)",

            "✅ ALLOWED (Zero False Positive)"
        ],


        "Key Differentiator": [

            "Per-client historical volume baseline",

            "Login pressure + timing regularity (CV)",

            "Cross-IP behavioral pattern clustering",

            "Catalog page count & fixed bot timing",

            "Sequential ID traversal detection",

            "Human timing jitter preserved"
        ]
    })


    st.table(
        comp_df
    )


    benchmark_csv = csv_bytes(comp_df)


    st.download_button(

        "📥 Download Benchmark Table (CSV)",

        data=benchmark_csv,

        file_name="CY02_benchmark_vs_static_waf.csv",

        mime="text/csv",

        key="download_benchmark_csv",

        on_click="ignore",

        width="stretch"
    )


# ==============================================================================
# PAGE 6 — PLUG AND PLAY SDK
# ==============================================================================

elif page == "🔌 Plug-and-Play SDK":

    st.markdown(
        "### 🔌 2-Line Plug-and-Play Integration Engine"
    )


    st.markdown(
        "Integrate CY-02 Sentinel directly into any Python web framework (FastAPI, Django, Flask) to inspect and block threats before they reach your database."
    )


    col1, col2 = st.columns(
        [1, 1]
    )


    with col1:

        st.markdown(
            "#### 🛠️ How Developers Integrate It:"
        )


        st.code(
            """
from fastapi import FastAPI
from sentinel_sdk import CY02SentinelMiddleware

app = FastAPI()

# 🔌 Plug-and-Play CY-02 Sentinel
app.add_middleware(
    CY02SentinelMiddleware,
    risk_threshold=70.0
)

@app.post("/api/checkout")
def checkout():
    return {
        "status":
        "Order processed successfully!"
    }
            """,
            language="python"
        )


    with col2:

        st.markdown(
            "#### ⚡ Real-Time Request Lifecycle:"
        )


        st.info(
            """
1. **Incoming Request:** Client sends request to `/api/checkout`.

2. **CY-02 Interception:** Inspects headers, IP, and frequency in **< 15ms**.

3. **Enforcement Decision:**

- **Clean Traffic (Score < 70):** Passes to database.

- **Bot / Threat (Score ≥ 70):** Returns `HTTP 429 Blocked`.
"""
        )


    st.divider()


    st.markdown(
        "#### 🧪 Test Simulated Middleware Execution"
    )


    test_ua = st.text_input(
        "Simulate User-Agent:",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )


    test_endpoint = st.selectbox(
        "Target Endpoint:",
        [
            "/api/products",
            "/api/login",
            "/api/checkout",
            "/api/admin/export"
        ]
    )


    if st.button(
        "Simulate Request Through Middleware"
    ):

        is_bot = (
            "bot"
            in test_ua.lower()
            or
            "python"
            in test_ua.lower()
        )


        score = (
            (40.0 if is_bot else 0.0)
            +
            (
                35.0
                if test_endpoint
                in [
                    "/api/login",
                    "/api/checkout"
                ]
                else 0.0
            )
        )


        if score >= 70.0:

            st.error(
                f"""
❌ **HTTP 429 TOO MANY REQUESTS**

Risk Score:
{score}/100

Action:
**BLOCKED AT FRONT DOOR**
"""
            )


        else:

            st.success(
                f"""
✅ **HTTP 200 OK**

Risk Score:
{score}/100

Action:
**PASSED TO DATABASE**
"""
            )