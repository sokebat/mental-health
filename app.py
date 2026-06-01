import sys
from pathlib import Path

# Ensures src/ imports work on Streamlit Cloud without setting PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.preprocess import preprocess_text

# ── Constants ────────────────────────────────────────────────────────────────

SKLEARN_DIR = Path("models/sklearn")
FIGURES_DIR = Path("reports/figures")

LABEL_COLORS = {
    "Normal":     "#2ecc71",
    "Anxiety":    "#f39c12",
    "Depression": "#3498db",
    "Suicidal":   "#e74c3c",
}

LABEL_EMOJIS = {
    "Normal":     "✅",
    "Anxiety":    "😰",
    "Depression": "😔",
    "Suicidal":   "🚨",
}

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes":         "Naive Bayes",
    "svm":                 "SVM (LinearSVC)",
    "random_forest":       "Random Forest",
}

PERFORMANCE = {
    "Logistic Regression": {"Accuracy": "78.44%", "Macro F1": "0.7695", "Speed": "⚡ Fast"},
    "Naive Bayes":         {"Accuracy": "70.89%", "Macro F1": "0.6524", "Speed": "⚡⚡ Fastest"},
    "SVM (LinearSVC)":     {"Accuracy": "78.28%", "Macro F1": "0.7683", "Speed": "⚡ Fast"},
    "Random Forest":       {"Accuracy": "71.99%", "Macro F1": "0.6777", "Speed": "🐢 Slow"},
}

FIGURES = [
    ("class_distribution.png",   "Class Distribution"),
    ("class_imbalance.png",       "Class Imbalance"),
    ("ml_comparison.png",         "Model Comparison"),
    ("ml_confusion_matrix.png",   "Confusion Matrix"),
    ("per_class_f1_comparison.png","Per-Class F1 Comparison"),
    ("text_length_analysis.png",  "Text Length Analysis"),
    ("top_words_per_class.png",   "Top Words per Class"),
    ("wordclouds_clean.png",      "Word Clouds"),
]

# ── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts(model_type: str):
    if not (SKLEARN_DIR / f"{model_type}.pkl").exists():
        return None, None, None
    tfidf  = joblib.load(SKLEARN_DIR / "tfidf_vectorizer.pkl")
    le     = joblib.load(SKLEARN_DIR / "label_encoder.pkl")
    model  = joblib.load(SKLEARN_DIR / f"{model_type}.pkl")
    return tfidf, le, model


def run_predict(text: str, model_type: str):
    tfidf, le, model = load_artifacts(model_type)
    if model is None:
        return None, None, None

    cleaned = preprocess_text(text)
    X = tfidf.transform([cleaned])

    label = le.inverse_transform(model.predict(X))[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # LinearSVC: decision_function → softmax
        scores = model.decision_function(X)[0]
        scores = scores - scores.max()
        exp_s  = np.exp(scores)
        proba  = exp_s / exp_s.sum()

    confidence = dict(zip(le.classes_, proba.tolist()))
    return label, confidence, cleaned

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mental Health Text Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Mental Health\nText Classifier")
    st.caption(
        "Classifies text into one of four categories: "
        "**Normal**, **Anxiety**, **Depression**, **Suicidal**."
    )
    st.divider()

    st.subheader("Select Model")
    model_type = st.radio(
        "Model",
        options=list(MODEL_DISPLAY.keys()),
        format_func=lambda k: MODEL_DISPLAY[k],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Model Performance")
    perf_df = pd.DataFrame(PERFORMANCE).T
    st.dataframe(perf_df, use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ This tool is for research purposes only. "
        "If you or someone you know is in crisis, please contact a mental health professional."
    )

# ── Main Area ─────────────────────────────────────────────────────────────────

tab_predict, tab_info = st.tabs(["🔍 Predict", "📊 Model Info"])

# ── Predict Tab ───────────────────────────────────────────────────────────────

with tab_predict:
    st.header("Analyze Text")

    # Check model artifacts exist
    tfidf_ok = (SKLEARN_DIR / "tfidf_vectorizer.pkl").exists()
    model_ok  = (SKLEARN_DIR / f"{model_type}.pkl").exists()
    if not (tfidf_ok and model_ok):
        st.error(
            f"Model artifacts not found in `{SKLEARN_DIR}/`. "
            "Run training first:\n\n"
            "```powershell\n"
            "python -m src.training.train --config configs/config.yaml --mode sklearn\n"
            "```"
        )
        st.stop()

    user_text = st.text_area(
        "Enter text to classify:",
        placeholder="Type or paste text here — e.g. 'I feel hopeless and cannot get out of bed...'",
        height=160,
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze = st.button("Analyze", type="primary")

    if analyze:
        if not user_text.strip():
            st.warning("Please enter some text.")
        elif len(user_text.split()) < 3:
            st.warning("Please enter at least 3 words for a meaningful prediction.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence, cleaned = run_predict(user_text, model_type)

            color = LABEL_COLORS[label]
            emoji = LABEL_EMOJIS[label]

            # Result card
            st.markdown(f"""
<div style="
    background:{color}18;
    border-left:5px solid {color};
    border-radius:10px;
    padding:22px 28px;
    margin:18px 0 8px 0;
">
    <div style="font-size:2.2rem;font-weight:800;color:{color};">
        {emoji}&nbsp;&nbsp;{label}
    </div>
    <div style="font-size:0.85rem;color:#999;margin-top:6px;">
        Classified by <b>{MODEL_DISPLAY[model_type]}</b>
    </div>
</div>
""", unsafe_allow_html=True)

            # Confidence scores
            st.subheader("Confidence Scores")
            sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_conf:
                pct = prob * 100
                bar_color = LABEL_COLORS[cls]
                lcol, rcol = st.columns([1, 4])
                with lcol:
                    st.markdown(
                        f'<span style="color:{bar_color};font-weight:600;">{cls}</span>',
                        unsafe_allow_html=True,
                    )
                with rcol:
                    st.progress(float(prob), text=f"{pct:.1f}%")

            # Preprocessed text
            with st.expander("Show preprocessed text"):
                st.code(cleaned, language=None)

    # Disclaimer
    st.divider()
    st.info(
        "This tool is for research and educational purposes only. "
        "It should not be used as a substitute for professional mental health assessment.",
        icon="ℹ️",
    )

# ── Model Info Tab ────────────────────────────────────────────────────────────

with tab_info:
    st.header("Model & Data Insights")

    available_figs = [
        (FIGURES_DIR / fname, title)
        for fname, title in FIGURES
        if (FIGURES_DIR / fname).exists()
    ]

    if not available_figs:
        st.warning("No figures found in `reports/figures/`.")
    else:
        cols = st.columns(2)
        for i, (fig_path, title) in enumerate(available_figs):
            with cols[i % 2]:
                st.subheader(title)
                st.image(str(fig_path), use_column_width=True)
