"""
app.py
--------
SmartShield AI — Hybrid Explainable Spam & Phishing Detection System.

Streamlit entry point. Wires together all src/ modules (preprocessing,
prediction, ensemble, explainability, URL/keyword/emotion/entity
detectors, and visualization) into a multi-page, dark-themed,
production-styled web application.

Run:
    streamlit run app.py
"""

import os
import sys
import json
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.logger import get_logger
from src.preprocessing import preprocess_for_ml
from src.keyword_detector import scan_keywords, KEYWORD_CATEGORIES
from src.url_detector import detect_and_analyze
from src.emotion import analyze_emotions, dominant_emotion
from src.entities import extract_entities
from src.explain import generate_explanation, highlight_keywords_html
from src import visualization as viz

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CSS — dark cybersecurity theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #101823 0%, #0B0F14 45%, #080B0E 100%);
}

section[data-testid="stSidebar"] {
    background-color: #0D1319;
    border-right: 1px solid #1C2530;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
}

.shield-hero {
    padding: 1.6rem 2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(0,229,160,0.08), rgba(0,229,160,0.01));
    border: 1px solid rgba(0,229,160,0.25);
    margin-bottom: 1.4rem;
}

.metric-card {
    background: #121821;
    border: 1px solid #1C2530;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

.reason-chip {
    display: inline-block;
    background: rgba(255,75,110,0.12);
    border: 1px solid rgba(255,75,110,0.4);
    color: #FF8FA3;
    border-radius: 999px;
    padding: 4px 12px;
    margin: 4px 6px 4px 0;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

.verdict-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 999px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
}

.badge-spam { background: rgba(255,75,110,0.15); color: #FF4B6E; border: 1px solid #FF4B6E; }
.badge-suspicious { background: rgba(255,176,32,0.15); color: #FFB020; border: 1px solid #FFB020; }
.badge-ham { background: rgba(0,229,160,0.15); color: #00E5A0; border: 1px solid #00E5A0; }

.code-block {
    font-family: 'JetBrains Mono', monospace;
    background: #0D1319;
    border: 1px solid #1C2530;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.92rem;
    line-height: 1.7;
}

.footer-note {
    color: #5C6773;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
    margin-top: 3rem;
}

div.stButton > button {
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    border: 1px solid #00E5A0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_MESSAGES = {
    "🎉 Lottery Scam": "Congratulations!! You have WON ₹50,000 in the lucky draw. Click here immediately to claim your prize before it expires: http://bit.ly/claim-prize",
    "🏦 Bank Phishing": "URGENT: Your HDFC Bank account has been suspended due to unusual activity. Verify your OTP and password now at hdfc-secure-verify.tk to avoid permanent closure.",
    "📦 Courier Scam": "Your parcel could not be delivered due to an unpaid customs fee of $2.99. Pay immediately at courier-redelivery.xyz to reschedule delivery.",
    "💬 Normal Message": "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if the usual place works for you.",
}

# ---------------------------------------------------------------------------
# CACHED HELPERS
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models (Logistic Regression + LSTM)...")
def _load_predict_module():
    """Import predict.py and eagerly load all 4 model artifacts once per
    app session, so per-message inference timings shown to the user
    reflect only actual inference time, not one-time load cost."""
    from src import predict
    predict.preload_models()
    return predict


@st.cache_data(show_spinner=False)
def _load_dataset():
    df = pd.read_csv(config.SPAM_DATASET_PATH, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text"])
    return df


@st.cache_data(show_spinner=False)
def _load_metrics():
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def _verdict_badge(verdict: str) -> str:
    css_class = {"Spam": "badge-spam", "Suspicious": "badge-suspicious"}.get(verdict, "badge-ham")
    icon = {"Spam": "🚨", "Suspicious": "⚠️"}.get(verdict, "✅")
    return f'<span class="verdict-badge {css_class}">{icon} {verdict}</span>'


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ SmartShield AI")
    st.caption(config.APP_TAGLINE)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Spam Detection", "📊 Dashboard", "⚖️ Model Comparison",
         "📁 Dataset Analysis", "ℹ️ About Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"v{config.APP_VERSION} · Streamlit Cloud Ready")

# ===========================================================================
# PAGE 1 — HOME
# ===========================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="shield-hero">
            <h1>🛡️ SmartShield AI</h1>
            <p style="font-size:1.1rem;color:#9FB0BE;">
            Hybrid Explainable Spam &amp; Phishing Detection System using
            NLP, LSTM, and Ensemble Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    metrics = _load_metrics()
    lr_acc = metrics.get("logistic_regression", {}).get("accuracy", 0) * 100
    lstm_acc = metrics.get("lstm", {}).get("accuracy", 0) * 100
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Logistic Regression Accuracy", f"{lr_acc:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("LSTM Accuracy", f"{lstm_acc:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Detection Channels", "5", "SMS · Email · WhatsApp · Telegram · Social")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Keyword Categories", str(len(KEYWORD_CATEGORIES)))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🚀 Core Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown(
            """
            **🧠 Hybrid AI Engine**
            - TF-IDF + Logistic Regression
            - Embedding + LSTM
            - Weighted ensemble fusion
            - Disagreement-aware "Suspicious" flagging
            """
        )
    with feat_col2:
        st.markdown(
            """
            **🔎 Explainable AI**
            - Keyword category intelligence
            - URL risk analysis
            - Emotion / manipulation detection
            - Named entity recognition
            """
        )
    with feat_col3:
        st.markdown(
            """
            **📊 Analytics Dashboard**
            - Model comparison & ROC curves
            - Confusion matrices
            - Word clouds & frequency charts
            - Dataset quality reports
            """
        )

    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        """
        <div class="code-block">
        Message Input<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;├──▶ NLP Preprocessing (clean → tokenize → lemmatize)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;├──▶ Model 1: TF-IDF + Logistic Regression ──▶ P(spam)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;├──▶ Model 2: Embedding + LSTM ──▶ P(spam)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;├──▶ Ensemble Fusion (weighted avg + disagreement check)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;├──▶ Explainability Layer (keywords · URLs · emotion · entities)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;│<br>
        &nbsp;&nbsp;&nbsp;&nbsp;└──▶ Verdict: Spam / Suspicious / Ham + Reasons
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧰 Technology Stack")
    stack_cols = st.columns(6)
    stack_items = ["Python", "Scikit-learn", "TensorFlow / Keras", "NLTK", "Streamlit", "Plotly"]
    for c, item in zip(stack_cols, stack_items):
        c.markdown(f"<div class='metric-card' style='text-align:center;'>{item}</div>", unsafe_allow_html=True)

    st.markdown("### 👨‍💻 Developer")
    st.info(
        f"**{config.DEVELOPER_NAME}** · {config.DEVELOPER_ROLE} · {config.DEVELOPER_COLLEGE}\n\n"
        f"GitHub: {config.DEVELOPER_GITHUB} · LinkedIn: {config.DEVELOPER_LINKEDIN}"
    )

# ===========================================================================
# PAGE 2 — SPAM DETECTION
# ===========================================================================
elif page == "🔍 Spam Detection":
    st.markdown("## 🔍 Spam & Phishing Detection")
    st.caption("Paste any SMS, email, WhatsApp, Telegram, or social media message below.")

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""

    example_cols = st.columns(len(EXAMPLE_MESSAGES))
    for c, (label, text) in zip(example_cols, EXAMPLE_MESSAGES.items()):
        if c.button(label, width='stretch'):
            st.session_state.message_input = text

    message = st.text_area(
        "Message text",
        value=st.session_state.message_input,
        height=160,
        placeholder="Type or paste a message to analyze...",
        label_visibility="collapsed",
        key="message_area",
    )

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    predict_clicked = btn_col1.button("🛡️ Analyze Message", type="primary", width='stretch')
    clear_clicked = btn_col2.button("🗑️ Clear", width='stretch')

    if clear_clicked:
        st.session_state.message_input = ""
        st.rerun()

    if predict_clicked:
        if not message or not message.strip():
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Running hybrid NLP + LSTM analysis..."):
                predict = _load_predict_module()
                ensemble_result = predict.predict_message(message)
                report = generate_explanation(message, ensemble_result)

            st.markdown("---")
            top1, top2 = st.columns([1, 2])
            with top1:
                st.plotly_chart(
                    viz.plot_confidence_gauge(report["spam_score"], report["verdict"]),
                    width='stretch',
                )
                st.markdown(_verdict_badge(report["verdict"]), unsafe_allow_html=True)
            with top2:
                m1, m2, m3 = st.columns(3)
                m1.metric("Logistic Regression", f"{report['logistic_confidence']}%")
                m2.metric("LSTM (Deep Learning)", f"{report['lstm_confidence']}%")
                m3.metric("Model Disagreement", f"{report['disagreement']}%")

                st.markdown("**🧾 Reasons detected:**")
                chips = "".join(f'<span class="reason-chip">{r}</span>' for r in report["reasons"])
                st.markdown(chips, unsafe_allow_html=True)

            st.markdown("### ✍️ Highlighted Message")
            highlighted = highlight_keywords_html(message, report["keyword_categories"])
            st.markdown(f'<div class="code-block">{highlighted}</div>', unsafe_allow_html=True)

            detail1, detail2 = st.columns(2)
            with detail1:
                st.markdown("### 🔗 Detected URLs")
                if report["url_reports"]:
                    for u in report["url_reports"]:
                        st.write(f"**{u['url']}** — `{u['verdict']}` (risk {u['risk_score']}/100)")
                        for reason in u["reasons"]:
                            st.caption(f"• {reason}")
                else:
                    st.caption("No URLs detected.")

                st.markdown("### 📞 Contact Info Detected")
                st.write(f"**Phone numbers:** {', '.join(report['phone_numbers']) or 'None'}")
                st.write(f"**Emails:** {', '.join(report['emails']) or 'None'}")

            with detail2:
                st.markdown("### 🎭 Emotion / Manipulation Signals")
                if report["emotions"]:
                    for emo, data in report["emotions"].items():
                        st.write(f"**{emo}** — score {data['score']}/100")
                        st.progress(data["score"] / 100)
                else:
                    st.caption("No strong emotional manipulation detected.")

                st.markdown("### 🏷️ Named Entities")
                ent = report["entities"]
                non_empty = {k: v for k, v in ent.items() if v}
                if non_empty:
                    for k, v in non_empty.items():
                        st.write(f"**{k}:** {', '.join(v)}")
                else:
                    st.caption("No named entities detected.")

            st.markdown("### ⏱️ Performance")
            p1, p2, p3 = st.columns(3)
            p1.metric("Logistic Regression Time", f"{ensemble_result['logistic_time_ms']} ms")
            p2.metric("LSTM Time", f"{ensemble_result['lstm_time_ms']} ms")
            p3.metric("Total Inference Time", f"{ensemble_result['total_time_ms']} ms")

# ===========================================================================
# PAGE 3 — DASHBOARD
# ===========================================================================
elif page == "📊 Dashboard":
    st.markdown("## 📊 Analytics Dashboard")
    df = _load_dataset()
    metrics = _load_metrics()

    tab1, tab2, tab3 = st.tabs(["📈 Class & Length Analysis", "☁️ Word Clouds", "🧪 Model Metrics"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.plot_class_distribution(df), width='stretch')
        with c2:
            st.plotly_chart(viz.plot_message_length_distribution(df), width='stretch')

        spam_texts = df[df["label"].str.lower() == "spam"]["text"].apply(preprocess_for_ml).tolist()
        ham_texts = df[df["label"].str.lower() == "ham"]["text"].apply(preprocess_for_ml).tolist()

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(viz.plot_top_words(spam_texts, title="Top Spam Words"), width='stretch')
        with c4:
            st.plotly_chart(viz.plot_top_words(ham_texts, title="Top Ham Words"), width='stretch')

    with tab2:
        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("**Spam Word Cloud**")
            st.image(viz.generate_wordcloud_image(spam_texts, colormap="Reds"), width='stretch')
        with wc2:
            st.markdown("**Ham Word Cloud**")
            st.image(viz.generate_wordcloud_image(ham_texts, colormap="Greens"), width='stretch')

    with tab3:
        if metrics:
            model_metrics = {
                "Logistic Regression": metrics.get("logistic_regression", {}),
                "LSTM": metrics.get("lstm", {}),
            }
            st.plotly_chart(viz.plot_metric_comparison(model_metrics), width='stretch')

            cm1, cm2 = st.columns(2)
            with cm1:
                if "logistic_regression" in metrics:
                    st.plotly_chart(
                        viz.plot_confusion_matrix(
                            metrics["logistic_regression"]["confusion_matrix"],
                            title="Logistic Regression — Confusion Matrix",
                        ),
                        width='stretch',
                    )
            with cm2:
                if "lstm" in metrics:
                    st.plotly_chart(
                        viz.plot_confusion_matrix(
                            metrics["lstm"]["confusion_matrix"],
                            title="LSTM — Confusion Matrix",
                        ),
                        width='stretch',
                    )

            roc1, roc2 = st.columns(2)
            with roc1:
                if "roc_fpr" in metrics.get("logistic_regression", {}):
                    st.plotly_chart(
                        viz.plot_roc_curve(
                            metrics["logistic_regression"]["roc_fpr"],
                            metrics["logistic_regression"]["roc_tpr"],
                            metrics["logistic_regression"]["roc_auc"],
                            title="Logistic Regression — ROC Curve",
                        ),
                        width='stretch',
                    )
            with roc2:
                if "roc_fpr" in metrics.get("lstm", {}):
                    st.plotly_chart(
                        viz.plot_roc_curve(
                            metrics["lstm"]["roc_fpr"],
                            metrics["lstm"]["roc_tpr"],
                            metrics["lstm"]["roc_auc"],
                            title="LSTM — ROC Curve",
                        ),
                        width='stretch',
                    )
        else:
            st.warning("No metrics.json found. Run the training scripts first (see README).")

# ===========================================================================
# PAGE 4 — MODEL COMPARISON
# ===========================================================================
elif page == "⚖️ Model Comparison":
    st.markdown("## ⚖️ Model Comparison")
    metrics = _load_metrics()

    if not metrics:
        st.warning("No metrics.json found. Run the training scripts first (see README).")
    else:
        lr = metrics.get("logistic_regression", {})
        lstm = metrics.get("lstm", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 📐 Logistic Regression")
            st.metric("Accuracy", f"{lr.get('accuracy', 0)*100:.2f}%")
            st.metric("Precision", f"{lr.get('precision', 0)*100:.2f}%")
            st.metric("Recall", f"{lr.get('recall', 0)*100:.2f}%")
            st.metric("F1 Score", f"{lr.get('f1_score', 0)*100:.2f}%")
            st.metric("Train Time", f"{lr.get('train_time_seconds', 0)} s")
        with c2:
            st.markdown("#### 🧠 LSTM")
            st.metric("Accuracy", f"{lstm.get('accuracy', 0)*100:.2f}%")
            st.metric("Precision", f"{lstm.get('precision', 0)*100:.2f}%")
            st.metric("Recall", f"{lstm.get('recall', 0)*100:.2f}%")
            st.metric("F1 Score", f"{lstm.get('f1_score', 0)*100:.2f}%")
            st.metric("Train Time", f"{lstm.get('train_time_seconds', 0)} s")
        with c3:
            st.markdown("#### 🤝 Ensemble Configuration")
            st.metric("LR Weight", f"{config.ENSEMBLE_WEIGHT_LOGISTIC*100:.0f}%")
            st.metric("LSTM Weight", f"{config.ENSEMBLE_WEIGHT_LSTM*100:.0f}%")
            st.metric("Disagreement Threshold", f"{config.DISAGREEMENT_THRESHOLD*100:.0f}%")
            st.metric("Spam Threshold", f"{config.SPAM_THRESHOLD*100:.0f}%")

        st.markdown("### 📊 Side-by-Side Metric Comparison")
        st.plotly_chart(
            viz.plot_metric_comparison({"Logistic Regression": lr, "LSTM": lstm}),
            width='stretch',
        )

        st.markdown("### ⏱️ Training Time Comparison")
        st.plotly_chart(
            viz.plot_execution_time(
                lr.get("train_time_seconds", 0) * 1000,
                lstm.get("train_time_seconds", 0) * 1000,
            ),
            width='stretch',
        )

        st.markdown("### 🔀 Why Ensemble?")
        st.info(
            "Logistic Regression is fast and highly interpretable via TF-IDF weights, "
            "but treats words independently (bag-of-words). LSTM captures word order and "
            "context but needs more data and compute. SmartShield AI combines both: "
            "when they agree, confidence is high; when they disagree significantly, the "
            "message is flagged **Suspicious** for human review instead of a risky hard call."
        )

# ===========================================================================
# PAGE 5 — DATASET ANALYSIS
# ===========================================================================
elif page == "📁 Dataset Analysis":
    st.markdown("## 📁 Dataset Analysis")
    df = _load_dataset()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("### 🔖 Class Distribution")
    st.plotly_chart(viz.plot_class_distribution(df), width='stretch')
    st.dataframe(df["label"].value_counts().rename("count"), width='stretch')

    st.markdown("### 📏 Average Message Length")
    df["length"] = df["text"].astype(str).apply(len)
    st.write(df.groupby("label")["length"].mean().round(1))

    st.markdown("### 🧾 Sample Rows")
    st.dataframe(df.sample(min(10, len(df)), random_state=42), width='stretch')

    st.markdown("### 🔤 Column Info")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [df[c].notnull().sum() for c in df.columns],
    })
    st.dataframe(info_df, width='stretch')

# ===========================================================================
# PAGE 6 — ABOUT
# ===========================================================================
elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About SmartShield AI")

    st.markdown(
        f"""
        ### 🎯 Project Objective
        SmartShield AI is a hybrid, explainable spam and phishing detection system that
        combines classical NLP (TF-IDF + Logistic Regression) with deep learning
        (Embedding + LSTM) into a single ensemble model. Beyond a raw prediction, it
        explains *why* a message is flagged — via keyword intelligence, URL risk
        analysis, emotion/manipulation detection, and named entity recognition — so the
        output is transparent and trustworthy rather than a black box.

        ### 🧰 Technologies Used
        - **Languages:** Python
        - **NLP:** NLTK (tokenization, lemmatization, stopwords), regex-based entity/URL/keyword engines
        - **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression)
        - **Deep Learning:** TensorFlow / Keras (Embedding, LSTM, Dense, Sigmoid)
        - **Visualization:** Plotly, WordCloud
        - **Web App:** Streamlit
        - **Explainable AI:** Custom rule-based + model-confidence hybrid explanation layer

        ### 🔮 Future Scope
        - Fine-tune a transformer model (e.g. DistilBERT) as a third ensemble member
        - Add multilingual spam detection (Hindi, Tamil, Spanish, etc.)
        - Integrate real-time email/SMS gateway APIs for live filtering
        - Add a browser extension for inline WhatsApp Web / Gmail scanning
        - Active-learning feedback loop where user corrections retrain the model

        ### 👨‍💻 Developer Details
        **{config.DEVELOPER_NAME}**
        {config.DEVELOPER_ROLE} — {config.DEVELOPER_COLLEGE}

        GitHub: {config.DEVELOPER_GITHUB}
        LinkedIn: {config.DEVELOPER_LINKEDIN}
        """
    )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="footer-note">SmartShield AI v{config.APP_VERSION} ·
    Built with Streamlit · NLP + LSTM + Ensemble Learning</div>""",
    unsafe_allow_html=True,
)
