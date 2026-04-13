"""
app.py
Improved Streamlit dashboard for early ransomware detection using
system-call sequence analysis and machine learning.
"""

import io
import os
import json
import glob
import pickle
import warnings
from datetime import datetime
from collections import Counter
from typing import Optional

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
APP_TITLE = "EarlyGuard"
MODEL_DIR = "models"
DEFAULT_MIN_SYSCALLS = 20

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif;
        }
        .stApp { background: #060b14; }
        .main .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* ── HERO ── */
        .hero {
            background: linear-gradient(160deg, #0a1628 0%, #060b14 60%, #0a0d18 100%);
            border: 1px solid rgba(56, 189, 248, 0.1);
            border-radius: 24px;
            padding: 4rem 3rem 3.2rem;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        /* Top glow beam */
        .hero::before {
            content: "";
            position: absolute;
            top: -60px; left: 50%;
            transform: translateX(-50%);
            width: 600px; height: 220px;
            background: radial-gradient(ellipse at center,
                rgba(56,189,248,0.10) 0%,
                rgba(99,102,241,0.05) 40%,
                transparent 70%);
            pointer-events: none;
        }
        /* Subtle top border highlight */
        .hero::after {
            content: "";
            position: absolute;
            top: 0; left: 50%;
            transform: translateX(-50%);
            width: 280px; height: 1px;
            background: linear-gradient(90deg,
                transparent,
                rgba(56,189,248,0.5),
                transparent);
        }
        /* Wordmark / logotype */
        .hero-wordmark {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 3.5px;
            text-transform: uppercase;
            color: #38bdf8;
            font-family: 'IBM Plex Mono', monospace;
            margin-bottom: 1rem;
            opacity: 0.7;
        }
        .hero h1 {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(100deg, #e2f8ff 10%, #38bdf8 45%, #818cf8 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 1rem 0;
            line-height: 1.05;
            letter-spacing: -1.5px;
            filter: drop-shadow(0 0 28px rgba(56,189,248,0.18));
        }
        .hero-tagline {
            display: inline-block;
            font-size: 0.95rem;
            color: #64748b;
            max-width: 520px;
            line-height: 1.6;
            letter-spacing: 0.2px;
            text-align: center;
        }
        .feature-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .feature-tag {
            background: rgba(56, 189, 248, 0.05);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.15);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: 0.3px;
            cursor: default;
            transition: transform 0.2s ease, box-shadow 0.2s ease,
                        background 0.2s ease, border-color 0.2s ease;
        }
        .feature-tag:hover {
            transform: scale(1.06);
            background: rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.2);
        }

        /* ── INFO BOX ── */
        .info-box {
            background: rgba(56, 189, 248, 0.06);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #e0f2fe;
            border-radius: 14px;
            padding: 1.1rem 1.5rem;
            margin-bottom: 1.8rem;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* ── METRIC CARD ── */
        .metric-card {
            background: #0d1b2e;
            border-radius: 14px;
            padding: 1.2rem 0.75rem;
            text-align: center;
            height: 100%;
            min-height: 110px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(56, 189, 248, 0.12);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            transition: transform 0.22s ease,
                        box-shadow 0.22s ease,
                        border-color 0.22s ease,
                        background 0.22s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px) scale(1.02);
            background: #101f33;
            border-color: rgba(56, 189, 248, 0.35);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35),
                        0 0 18px rgba(56, 189, 248, 0.12);
        }
        .metric-card h3 {
            font-size: 0.65rem;
            font-weight: 600;
            color: #475569;
            margin: 0 0 0.3rem 0;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'IBM Plex Mono', monospace;
            white-space: nowrap;
            width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-card .value {
            font-size: 1.05rem;
            font-weight: 800;
            color: #38bdf8;
            margin: 0.15rem 0;
            line-height: 1.3;
            /* Normal words wrap at spaces; long unbreakable strings (e.g. filenames)
               wrap at the card edge rather than overflowing */
            word-break: keep-all;
            overflow-wrap: anywhere;
            white-space: normal;
            width: 100%;
            font-family: 'IBM Plex Mono', monospace;
        }
        .metric-card .small-text {
            color: #475569;
            font-size: 0.72rem;
            margin-top: 0.2rem;
            white-space: normal;
            word-break: keep-all;
            overflow-wrap: normal;
            width: 100%;
        }

        /* ── UPLOAD CARD ── */
        .upload-card {
            background: #0d1b2e;
            border: 2px dashed rgba(56, 189, 248, 0.3);
            border-radius: 18px;
            padding: 2.4rem 1.8rem;
            text-align: center;
            color: #94a3b8;
        }
        .upload-card h3 { color: #38bdf8; margin-bottom: 0.5rem; }

        /* ── WORKFLOW STEPS ── */
        .workflow-step {
            background: #0d1b2e;
            border: 1px solid rgba(56, 189, 248, 0.1);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.9rem;
            color: #cbd5e1;
        }
        .step-number {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: white;
            width: 26px; height: 26px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.8rem;
            flex-shrink: 0;
            font-family: 'IBM Plex Mono', monospace;
        }

        /* ── RESULT BOXES ── */
        .result-ransomware {
            background: linear-gradient(135deg, #450a0a, #7f1d1d);
            border: 1px solid rgba(248, 113, 113, 0.4);
            padding: 1.8rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.2rem;
        }
        .result-benign {
            background: linear-gradient(135deg, #052e16, #14532d);
            border: 1px solid rgba(74, 222, 128, 0.4);
            padding: 1.8rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.2rem;
        }
        .result-ransomware h2, .result-benign h2 {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
        }
        .result-ransomware h3, .result-benign h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.3rem 0;
            font-family: 'IBM Plex Mono', monospace;
        }

        /* ── CONFIDENCE BAR ── */
        .conf-bar-wrap {
            background: #0d1b2e;
            border-radius: 9999px;
            height: 12px;
            overflow: hidden;
            margin: 0.6rem 0;
            border: 1px solid rgba(56,189,248,0.12);
        }
        .conf-bar-fill-safe {
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(90deg, #22c55e, #4ade80);
            transition: width 0.5s ease;
        }
        .conf-bar-fill-danger {
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(90deg, #ef4444, #f87171);
            transition: width 0.5s ease;
        }

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {
            background: #060b14;
            border-right: 1px solid rgba(56, 189, 248, 0.08);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #38bdf8;
        }
        .sidebar-model-badge {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 10px;
            padding: 0.7rem 1rem;
            color: #7dd3fc;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.88rem;
            margin: 0.4rem 0;
        }
        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(56,189,248,0.06);
            font-size: 0.88rem;
            color: #94a3b8;
        }
        .sidebar-stat span { color: #38bdf8; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

        /* ── SECTION HEADERS ── */
        h3 { color: #e2e8f0 !important; }

        /* ── TABS ── */
        button[data-baseweb="tab"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.85rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_title(name: str) -> str:
    return name.replace("_", " ").replace(".pkl", "").title()


@st.cache_data(show_spinner=False)
def load_metadata():
    path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_results_table():
    path = os.path.join(MODEL_DIR, "model_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading detection engine...")
def load_model_bundle(selected_model_file: Optional[str] = None):
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(vec_path):
        return None, None, None

    model_files = sorted(
        [
            p for p in glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
            if not os.path.basename(p).startswith("tfidf_vectorizer")
        ]
    )
    if not model_files:
        return None, None, None

    model_path = None
    if selected_model_file:
        candidate = os.path.join(MODEL_DIR, selected_model_file)
        if os.path.exists(candidate):
            model_path = candidate

    if model_path is None:
        metadata = load_metadata()
        best_model = metadata.get("best_model_filename")
        if best_model:
            candidate = os.path.join(MODEL_DIR, best_model)
            if os.path.exists(candidate):
                model_path = candidate

    if model_path is None:
        model_path = model_files[0]

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer, os.path.basename(model_path)


def parse_uploaded_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    parse_errors = []
    for header in (None, "infer"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), header=header)
            if df.empty:
                continue
            first_col = df.columns[0]
            syscalls = (
                df[first_col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            syscalls = [s for s in syscalls if s and s.lower() != "nan"]
            if syscalls:
                return syscalls, None
        except Exception as e:
            parse_errors.append(str(e))
    return [], "; ".join(parse_errors) if parse_errors else "Unknown parsing error."


def predict_with_confidence(model, vectorizer, sequence_text: str):
    X = vectorizer.transform([sequence_text])
    prediction = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[prediction]) * 100
    else:
        confidence = 100.0
    return prediction, confidence


def risk_level(label: int, confidence: float) -> str:
    if label == 0:
        if confidence >= 90:
            return "LOW RISK"
        if confidence >= 70:
            return "MONITORED"
        return "UNCERTAIN"
    if confidence >= 90:
        return "HIGH RISK"
    if confidence >= 70:
        return "MEDIUM RISK"
    return "REVIEW NEEDED"


def render_metric_card(label, value, small_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <h3 title="{label}">{label}</h3>
            <p class="value" title="{value}">{value}</p>
            <p class="small-text" title="{small_text}">{small_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence: float, label: int):
    fill_class = "conf-bar-fill-danger" if label == 1 else "conf-bar-fill-safe"
    width = min(int(confidence), 100)
    st.markdown(
        f"""
        <div class="conf-bar-wrap">
            <div class="{fill_class}" style="width:{width}%"></div>
        </div>
        <p style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#64748b;margin:2px 0 0 0;">
            {confidence:.1f}% confidence
        </p>
        """,
        unsafe_allow_html=True,
    )


def get_top_syscalls(syscalls, top_n=12):
    return Counter(syscalls).most_common(top_n)


def build_report_text(
    filename: str,
    model_display_name: str,
    early_stage_calls: int,
    total_syscalls: int,
    used_syscalls: int,
    prediction_label: int,
    confidence: float,
    risk: str,
    top_calls: list,
) -> str:
    label_text = "Ransomware" if prediction_label == 1 else "Benign"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "EarlyGuard - Analysis Report",
        "=" * 40,
        f"Generated at : {timestamp}",
        f"Uploaded file: {filename}",
        f"Model        : {model_display_name}",
        f"Prediction   : {label_text}",
        f"Confidence   : {confidence:.2f}%",
        f"Risk level   : {risk}",
        f"Total parsed : {total_syscalls} system calls",
        f"Used (window): {used_syscalls} system calls",
        f"Window size  : first {early_stage_calls} system calls",
        "",
        "Top observed system calls:",
    ]
    for name, count in top_calls:
        lines.append(f"  {name:<20} {count}")
    lines.extend([
        "",
        "Note:",
        "Confidence is a model probability score for interpretability.",
        "It is not a guarantee — treat as one signal among many.",
    ])
    return "\n".join(lines)


def run_single_analysis(model, vectorizer, metadata, model_display_name, uploaded_file):
    """Full analysis block for a single uploaded file."""
    raw_syscalls, parse_error = parse_uploaded_file(uploaded_file)
    if not raw_syscalls:
        st.error("Could not extract any system calls from this file.")
        if parse_error:
            st.caption(f"Parsing details: {parse_error}")
        return

    min_syscalls = int(metadata.get("min_syscalls", DEFAULT_MIN_SYSCALLS))
    early_stage_calls = int(metadata.get("early_stage_calls", len(raw_syscalls)))

    if len(raw_syscalls) < min_syscalls:
        st.warning(
            f"This file has only {len(raw_syscalls)} system calls. "
            f"At least {min_syscalls} are recommended for a reliable prediction."
        )
        return

    used_syscalls = raw_syscalls[:early_stage_calls]
    sequence_text = " ".join(used_syscalls)

    label, confidence = predict_with_confidence(model, vectorizer, sequence_text)
    current_risk = risk_level(label, confidence)
    top_calls = get_top_syscalls(used_syscalls, top_n=12)

    preview_col, result_col = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: log preview ──
    with preview_col:
        st.markdown("### Parsed Log Preview")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Total Calls", f"{len(raw_syscalls)}", "Rows extracted")
        with c2:
            render_metric_card("Used (window)", f"{len(used_syscalls)}", "Early-stage slice")
        with c3:
            display_name = uploaded_file.name.rsplit(".", 1)[0]  # strip extension
            render_metric_card("File", display_name, "Target")

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        with st.expander("Preview first 60 system calls", expanded=True):
            st.code(" → ".join(used_syscalls[:60]), language=None)

        # Bar chart for top syscalls
        st.markdown("### Top System Calls")
        if top_calls:
            chart_df = pd.DataFrame(top_calls, columns=["System Call", "Count"])
            st.bar_chart(chart_df.set_index("System Call"), height=280)

    # ── RIGHT: threat result ──
    with result_col:
        st.markdown("### Threat Analysis")

        if label == 1:
            st.markdown(
                f"""
                <div class="result-ransomware">
                    <h2>Ransomware Detected</h2>
                    <h3>{confidence:.1f}% confidence</h3>
                    <p style="color:#fca5a5;font-size:0.95rem;margin:0;">{current_risk}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-benign">
                    <h2>Benign Activity</h2>
                    <h3>{confidence:.1f}% confidence</h3>
                    <p style="color:#86efac;font-size:0.95rem;margin:0;">{current_risk}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_confidence_bar(confidence, label)
        st.caption("Confidence = model probability score. Not a guarantee — use as one signal.")

        st.markdown("#### Classification Details")
        d1, d2, d3 = st.columns(3)
        with d1:
            render_metric_card("Prediction", "Ransomware" if label == 1 else "Benign", "Final class")
        with d2:
            render_metric_card("Confidence", f"{confidence:.1f}%", "Model score")
        with d3:
            render_metric_card("Risk", current_risk, "Threat level")

        st.markdown("#### Technical Details")
        t1, t2, t3 = st.columns(3)
        with t1:
            render_metric_card("Model", model_display_name, "Active classifier")
        with t2:
            render_metric_card("Features", "TF-IDF", "1–3 gram n-grams")
        with t3:
            render_metric_card("Window", len(used_syscalls), "Calls analysed")

        report_text = build_report_text(
            filename=uploaded_file.name,
            model_display_name=model_display_name,
            early_stage_calls=early_stage_calls,
            total_syscalls=len(raw_syscalls),
            used_syscalls=len(used_syscalls),
            prediction_label=label,
            confidence=confidence,
            risk=current_risk,
            top_calls=top_calls,
        )
        st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Analysis Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"ransomguard_{uploaded_file.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain",
        )


# ─────────────────────────────────────────────
# LOAD PERSISTED DATA
# ─────────────────────────────────────────────
metadata = load_metadata()
results_df = load_results_table()
available_model_files = sorted(
    [
        os.path.basename(p) for p in glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
        if os.path.basename(p) != "tfidf_vectorizer.pkl"
    ]
)

default_model_file = metadata.get("best_model_filename")
if default_model_file not in available_model_files and available_model_files:
    default_model_file = available_model_files[0]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Early-stage ransomware behaviour detection")
    st.markdown("---")

    # Model selector
    st.markdown("### Detection Engine")
    if available_model_files:
        default_index = (
            available_model_files.index(default_model_file)
            if default_model_file in available_model_files else 0
        )
        selected_model_file = st.selectbox(
            "Choose classifier",
            options=available_model_files,
            index=default_index,
            format_func=safe_title,
        )
    else:
        selected_model_file = None

    model, vectorizer, loaded_model_file = load_model_bundle(selected_model_file)
    model_display_name = safe_title(loaded_model_file) if loaded_model_file else "Unavailable"

    if loaded_model_file:
        st.markdown(f'<div class="sidebar-model-badge">{model_display_name}</div>', unsafe_allow_html=True)
    else:
        st.warning("No trained model found")

    # Show best model metrics from results CSV
    if not results_df.empty and loaded_model_file:
        active_name = loaded_model_file.replace(".pkl", "")
        row = results_df[results_df["model_name"] == active_name]
        if not row.empty:
            r = row.iloc[0]
            st.markdown("**Model metrics (test set):**")
            for stat_label, col in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1 Score", "f1")]:
                val = r.get(col, None)
                if val is not None:
                    st.markdown(
                        f'<div class="sidebar-stat">{stat_label}<span>{float(val):.4f}</span></div>',
                        unsafe_allow_html=True,
                    )
            cv_mean = r.get("cv_f1_mean", None)
            if cv_mean is not None:
                st.markdown(
                    f'<div class="sidebar-stat">CV F1 Mean<span>{float(cv_mean):.4f}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    with st.expander("About", expanded=False):
        st.caption("Classify system-call logs as ransomware or benign using early-stage behaviour analysis.")
        st.markdown(
            """
**Pipeline**
- CSV log upload
- Sequence extraction & early-stage truncation
- TF-IDF n-gram feature extraction
- ML classifier prediction
- Confidence score + downloadable report
            """
        )

    with st.expander("Config", expanded=False):
        st.markdown(f"**Analysis window:** {metadata.get('early_stage_calls', 'N/A')} system calls")
        st.markdown(f"**Min upload size:** {metadata.get('min_syscalls', DEFAULT_MIN_SYSCALLS)} system calls")

    with st.expander("Labels", expanded=False):
        st.markdown("🔴 &nbsp;**1 = Ransomware**")
        st.markdown("🟢 &nbsp;**0 = Benign**")

    with st.expander("Upload Format", expanded=False):
        st.markdown(
            """
- CSV files only
- First column = system-call names
- One call per row
            """
        )

    st.markdown("---")
    st.caption("Final Year Cybersecurity Project · EarlyGuard")

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-wordmark">Early-stage · Syscall Analysis · ML Detection</div>
        <h1>{APP_TITLE}</h1>
        <div style="width:100%;text-align:center;margin:0 0 2rem 0;">
            <span class="hero-tagline">Detecting ransomware before encryption begins —<br>powered by system-call sequence intelligence.</span>
        </div>
        <div class="feature-tags">
            <span class="feature-tag">Rapid Detection</span>
            <span class="feature-tag">Model Comparison</span>
            <span class="feature-tag">Sequence Insights</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
        <strong>Getting started:</strong> Upload a system-call log as a CSV file — one call per row, first column.
        EarlyGuard will analyse the early execution trace and classify it as ransomware or benign.
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard: no model loaded
if model is None or vectorizer is None:
    st.error("No trained model artifacts found.")
    st.markdown(
        """
        **Setup steps:**
        1. Generate or place CSV data in `dataset/ransomware_calls/` and `dataset/benign_calls/`
        2. Run `python train_model.py`
        3. Refresh this dashboard
        """
    )
    st.stop()

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
top_a, top_b, top_c = st.columns(3)
with top_a:
    render_metric_card("Detection Engine", model_display_name, "Current loaded model")
with top_b:
    render_metric_card("Feature Space", "TF-IDF N-Grams", "Vectorizer from training")
with top_c:
    render_metric_card("Analysis Window", metadata.get("early_stage_calls", "N/A"), "Early-stage syscall count")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_single, tab_batch, tab_models = st.tabs(["Single File Analysis", "Batch Analysis", "Model Comparison"])

# ── TAB 1: Single File ──
with tab_single:
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown("### Upload System-Call Log")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV with system call names in the first column, one per row.",
            label_visibility="collapsed",
            key="single_upload",
        )
        if uploaded_file is None:
            st.markdown(
                """
                <div class="upload-card">
                    <h3>Ready for Analysis</h3>
                    <p>Upload a system-call CSV to preview the sequence, get a prediction,
                    view a top-calls chart, and download an analysis report.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("### Detection Workflow")
        for n, step in enumerate([
            "Read and validate the uploaded CSV file.",
            "Truncate to the early-stage slice used during training.",
            "Convert the sequence to TF-IDF n-gram feature vectors.",
            "Classify as ransomware or benign and report confidence.",
        ], 1):
            st.markdown(
                f'<div class="workflow-step"><span class="step-number">{n}</span>{step}</div>',
                unsafe_allow_html=True,
            )

    if uploaded_file is not None:
        run_single_analysis(model, vectorizer, metadata, model_display_name, uploaded_file)

# ── TAB 2: Batch Analysis ──
with tab_batch:
    st.markdown("### Batch Analysis")
    st.markdown(
        "Upload multiple CSV files at once. Each file is analysed independently "
        "and results are summarised in a downloadable table."
    )

    batch_files = st.file_uploader(
        "Upload multiple CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each CSV should have system-call names in the first column.",
        label_visibility="collapsed",
        key="batch_upload",
    )

    if batch_files:
        min_syscalls = int(metadata.get("min_syscalls", DEFAULT_MIN_SYSCALLS))
        early_stage_calls = int(metadata.get("early_stage_calls", 120))
        batch_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, bf in enumerate(batch_files):
            status_text.caption(f"Analysing {bf.name} ({i+1}/{len(batch_files)})…")
            raw_syscalls, parse_error = parse_uploaded_file(bf)

            if not raw_syscalls or len(raw_syscalls) < min_syscalls:
                batch_results.append({
                    "File": bf.name,
                    "Parsed Calls": len(raw_syscalls) if raw_syscalls else 0,
                    "Prediction": "Skipped",
                    "Confidence (%)": "-",
                    "Risk Level": "Insufficient data",
                })
            else:
                used = raw_syscalls[:early_stage_calls]
                seq = " ".join(used)
                lbl, conf = predict_with_confidence(model, vectorizer, seq)
                risk = risk_level(lbl, conf)
                batch_results.append({
                    "File": bf.name,
                    "Parsed Calls": len(raw_syscalls),
                    "Prediction": "Ransomware" if lbl == 1 else "Benign",
                    "Confidence (%)": f"{conf:.1f}",
                    "Risk Level": risk,
                })

            progress_bar.progress((i + 1) / len(batch_files))

        status_text.empty()
        progress_bar.empty()

        # Plain-text dataframe is the export source (clean, Excel-safe)
        batch_df = pd.DataFrame(batch_results)

        # Build a display-only copy with emoji indicators for the UI table
        display_df = batch_df.copy()
        display_df["Prediction"] = display_df["Prediction"].map(
            {"Ransomware": "🔴 Ransomware", "Benign": "🟢 Benign", "Skipped": "⚪ Skipped"}
        ).fillna(display_df["Prediction"])
        st.dataframe(display_df, use_container_width=True)

        # Summary stats (derived from plain-text batch_df)
        total = len(batch_df)
        n_ransomware = batch_df["Prediction"].eq("Ransomware").sum()
        n_benign = batch_df["Prediction"].eq("Benign").sum()
        n_skipped = batch_df["Prediction"].eq("Skipped").sum()

        bs1, bs2, bs3, bs4 = st.columns(4)
        with bs1:
            render_metric_card("Total Files", total, "Submitted")
        with bs2:
            render_metric_card("Ransomware", n_ransomware, "Detected")
        with bs3:
            render_metric_card("Benign", n_benign, "Detected")
        with bs4:
            render_metric_card("Skipped", n_skipped, "Too short / unreadable")

        st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Batch Results CSV",
            # Export plain-text batch_df with UTF-8 BOM so Excel opens it correctly
            data=batch_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ransomguard_batch_results.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            """
            <div class="upload-card">
                <h3>Upload Multiple Files</h3>
                <p>Select several system-call CSV files to analyse them all at once
                and receive a summary table with predictions and risk levels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── TAB 3: Model Comparison ──
with tab_models:
    st.markdown("### Model Comparison")

    if not results_df.empty:
        # Identify best model for highlighting
        best_model_name = metadata.get("best_model", "")

        display_df = results_df.copy()
        rename_map = {
            "model_name": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "cv_f1_mean": "CV F1 Mean",
            "cv_f1_std": "CV F1 Std",
            "test_samples": "Test Samples",
            "train_samples": "Train Samples",
        }
        display_df = display_df.rename(columns=rename_map)
        if "Model" in display_df.columns:
            display_df["Model"] = display_df["Model"].apply(safe_title)

        # Highlight numeric columns
        numeric_cols = ["Accuracy", "Precision", "Recall", "F1", "CV F1 Mean"]
        existing_numeric = [c for c in numeric_cols if c in display_df.columns]

        def highlight_best(row):
            best = best_model_name.replace("_", " ").title()
            if row.get("Model", "") == best:
                return ["background-color: rgba(56,189,248,0.12); color: #38bdf8"] * len(row)
            return [""] * len(row)

        styled = (
            display_df.style
            .apply(highlight_best, axis=1)
            .format({c: "{:.4f}" for c in existing_numeric if c in display_df.columns})
        )
        st.dataframe(styled, use_container_width=True)

        # Bar chart comparison
        if existing_numeric and "Model" in display_df.columns:
            st.markdown("#### F1 Score Comparison")
            chart_data = display_df.set_index("Model")[["F1"]].sort_values("F1", ascending=False)
            st.bar_chart(chart_data, height=280)

        st.download_button(
            "Download Model Comparison CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Model comparison results will appear here after running `python train_model.py`.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center;color:#334155;font-size:0.88rem;font-family:'IBM Plex Mono',monospace;">
        Developed as a Final Year Cybersecurity Project · EarlyGuard Dashboard
    </p>
    """,
    unsafe_allow_html=True,
)