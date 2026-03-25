"""
Explain My Data — Interpretation-as-a-Service Platform
Streamlit frontend
"""

import streamlit as st
import os
import io
import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from analyzer import load_dataset, generate_all_charts, build_data_summary
from explainer import explain_dataset, explain_chart, answer_question

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explain My Data",
    # page_icon="logo/ExplainMyDataLogo.png",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Always-dark CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp {
        background-color: #0e1117;
        color: #e8eaf6;
    }
    .main-title {
        font-size: 6.5rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        background: linear-gradient(90deg, #4c72b0, #dd8452) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
        display: block !important;
        margin-bottom: 10px !important;
    }
    .subtitle {
        color: #9fa8c7;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #7c9fd4;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# ── Load API key from .env ─────────────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")

# ── S3 upload helper ───────────────────────────────────────────────────────────
def upload_to_s3(file_obj, filename: str) -> bool:
    """Upload a file-like object to S3. Returns True on success."""
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        bucket = os.environ.get("S3_BUCKET", "explain-my-data-project-bucket")
        buf = io.BytesIO(file_obj.read())
        s3.upload_fileobj(buf, bucket, filename)
        return True
    except (BotoCoreError, ClientError) as e:
        st.warning(f"S3 upload failed: {e}")
        return False



# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex; align-items:center; gap:1rem; margin-bottom:10px;">'
    '<span style="font-size:5rem; line-height:1;">📊</span>'
    '<span class="main-title" style="margin-bottom:0;">Explain My Data</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Upload any dataset and get clear, trustworthy, '
    'plain-English explanations with visual summaries — powered by Groq.</div>',
    unsafe_allow_html=True,
)

# ── File upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your dataset here",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    help="Max 200 MB",
)

if not uploaded_file:
    st.info("Upload a dataset above to get started.")
    st.stop()

# Upload to S3 once per file (tracked in session state)
s3_upload_key = f"s3_uploaded_{uploaded_file.name}"
if s3_upload_key not in st.session_state:
    with st.spinner("Uploading file to S3..."):
        s3_key = f"csv/{uploaded_file.name}"
        success = upload_to_s3(uploaded_file, s3_key)
    if success:
        st.success(f"Uploaded **{uploaded_file.name}** to S3 (`csv/` folder).")
    uploaded_file.seek(0)  # reset so downstream readers can still read the file
    st.session_state[s3_upload_key] = True

if not api_key:
    st.error("ANTHROPIC_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()


# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def cached_load(file):
    return load_dataset(file)


@st.cache_data(show_spinner="Generating charts...")
def cached_charts(df_json):
    df = pd.read_json(df_json, orient="split")
    return generate_all_charts(df)


@st.cache_data(show_spinner="Building data summary...")
def cached_summary(df_json):
    df = pd.read_json(df_json, orient="split")
    return build_data_summary(df)


try:
    df = cached_load(uploaded_file)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

df_json = df.to_json(orient="split")
charts = cached_charts(df_json)
data_summary = cached_summary(df_json)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_charts, tab_qa = st.tabs([
    "🔍 Overview & Explanation",
    "📈 Visual Analysis",
    "💬 Ask a Question",
])


# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab_overview:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)

        with st.expander("Raw Statistics"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    with col_right:
        st.subheader("Overview Chart")
        if "overview" in charts:
            fig = charts["overview"][1]
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("AI Explanation")
        explain_key = f"explain_dataset_{uploaded_file.name}"
        if explain_key not in st.session_state:
            with st.spinner("Groq is reading your data..."):
                try:
                    st.session_state[explain_key] = explain_dataset(
                        data_summary, api_key
                    )
                except Exception as e:
                    st.session_state[explain_key] = f"Error: {e}"

        if explain_key in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state[explain_key])
            if st.button("Clear explanation", key="clear_overview"):
                del st.session_state[explain_key]
                st.rerun()


# ── Tab 2: Visual Analysis ─────────────────────────────────────────────────────
with tab_charts:
    chart_keys = [k for k in charts if k != "overview"]
    if not chart_keys:
        st.info("No additional charts available for this dataset.")
    else:
        for key in chart_keys:
            title, fig = charts[key]
            with st.container():
                st.markdown(f"### {title}")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                chart_explain_key = f"explain_chart_{uploaded_file.name}_{key}"
                exp_col1, exp_col2 = st.columns([1, 5])
                with exp_col1:
                    if chart_explain_key not in st.session_state:
                        if st.button("Explain", key=f"btn_{key}"):
                            with st.spinner("Analysing chart..."):
                                try:
                                    st.session_state[chart_explain_key] = (
                                        explain_chart(title, data_summary, api_key)
                                    )
                                except Exception as e:
                                    st.session_state[chart_explain_key] = f"Error: {e}"
                            st.rerun()
                    else:
                        if st.button("Clear", key=f"clear_{key}"):
                            del st.session_state[chart_explain_key]
                            st.rerun()

                if chart_explain_key in st.session_state:
                    with st.container(border=True):
                        st.markdown(st.session_state[chart_explain_key])

                st.markdown("---")


# ── Tab 3: Q&A ─────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask Anything About Your Data")
    st.markdown(
        "Ask free-form questions — Claude will answer based on your dataset summary."
    )

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Show history
    for item in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

    question = st.chat_input("e.g. Which column has the most outliers? Are there any trends over time?")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(question, data_summary, api_key)
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)
        st.session_state.qa_history.append(
            {"question": question, "answer": answer}
        )
