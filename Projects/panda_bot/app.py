"""
app.py - Data Analysis Agent (Main Streamlit App)

Upload any CSV file, ask a question in plain English,
and the agent writes and runs pandas code to answer it.
Results are shown as a chart, data table, and explanation.

Run: streamlit run app.py
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime

import pandas as pd
import streamlit as st

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from models import AnalysisReport
from analyzer import analyze_csv
from code_generator import generate_code, generate_narrative, is_valid_data_question
from executor import run_code_safely, detect_chart_type
from visualizer import generate_chart
from report_builder import save_report, build_report


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Custom CSS - Dark Purple Aesthetic
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0f0f1a;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #c8c8e8 !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #7c6af7, #a78bfa);
        color: white;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "current_report" not in st.session_state:
    st.session_state.current_report = None

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

if "schema" not in st.session_state:
    st.session_state.schema = None

if "charts_dir" not in st.session_state:
    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    st.session_state.charts_dir = charts_dir


# --------------------------------------------------
# Sidebar - History
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Data Analysis Agent")
    st.caption("Ask questions about your data in plain English")
    st.divider()

    # New analysis button
    if st.button("New Analysis", use_container_width=True):
        st.session_state.current_report  = None
        st.session_state.uploaded_path   = None
        st.session_state.schema          = None
        st.rerun()

    st.divider()

    # Analysis history
    st.markdown("### History")

    if not st.session_state.history:
        st.caption("No analyses yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            real_index = len(st.session_state.history) - 1 - i
            label      = item["question"][:40] + "..." if len(item["question"]) > 40 else item["question"]

            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📄 {label}", key=f"hist_{real_index}", use_container_width=True):
                    st.session_state.current_report = item
                    st.rerun()
            with col2:
                if st.button("✕", key=f"del_{real_index}"):
                    st.session_state.history.pop(real_index)
                    st.rerun()

            st.caption(item.get("timestamp", ""))

    if st.session_state.history:
        st.divider()
        if st.button("Clear All History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()




# --------------------------------------------------
# Main Area - Show Past Report
# --------------------------------------------------

if st.session_state.current_report and "report" in st.session_state.current_report:
    item   = st.session_state.current_report
    report = item["report"]

    st.markdown(f"## {item['question']}")
    st.caption(item.get("timestamp", ""))

    if report.success:
        st.markdown(f"**{report.narrative}**")
        st.divider()

        col1, col2 = st.columns([3, 2])

        with col1:
            if report.chart_path and os.path.exists(report.chart_path):
                st.image(report.chart_path, use_container_width=True)

        with col2:
            st.markdown("**Raw Result**")
            st.code(report.result, language="text")

        with st.expander("Generated Code"):
            st.code(report.code, language="python")

        report_md = build_report(report)
        st.download_button(
            "Download Report",
            data=report_md,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    else:
        st.error(f"Analysis failed: {report.narrative}")

    if st.button("Back to New Analysis"):
        st.session_state.current_report = None
        st.rerun()

    st.stop()


# --------------------------------------------------
# Main Area - New Analysis
# --------------------------------------------------

st.markdown("# 📊 Data Analysis Agent")
st.markdown("Upload a CSV file and ask any question about your data.")
st.divider()

col_upload, col_info = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Supported: any CSV with headers"
    )

with col_info:
    st.markdown("**Example questions:**")
    st.markdown("""
    - Which product had the highest sales?
    - What is the average salary by department?
    - Show monthly revenue trend
    - Which category has the best rating?
    - How many students passed each subject?
    """)


# --------------------------------------------------
# File Upload Handling
# --------------------------------------------------

if uploaded_file:
    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.uploaded_path = temp_path

    # Analyze schema
    schema = analyze_csv(temp_path, uploaded_file.name)
    st.session_state.schema = schema

    # Show dataset preview
    st.divider()
    st.markdown(f"### {uploaded_file.name}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows",    f"{schema.rows:,}")
    m2.metric("Columns", len(schema.columns))
    m3.metric("File",    uploaded_file.name)

    df_preview = pd.read_csv(temp_path)
    with st.expander("Preview data (first 5 rows)"):
        st.dataframe(df_preview.head(5), use_container_width=True)


#--------------------------------------------------
# Question Input and Analysis
# --------------------------------------------------

if st.session_state.schema:
    st.divider()

    with st.form("analysis_form"):
        question       = st.text_input(
            "Ask a question about your data",
            placeholder="e.g. Which product had the highest total sales?",
        )
        analyze_clicked = st.form_submit_button("Analyze")

    if analyze_clicked and question.strip():

        with st.spinner("Analyzing your data..."):

            progress = st.progress(0)
            status   = st.empty()

            # Step 1: Generate code
            status.markdown("**Step 1/3** — Generating pandas code...")
            progress.progress(20)

            generated = generate_code(st.session_state.schema, question)

            # Step 2: Run code
            status.markdown("**Step 2/3** — Running analysis...")
            progress.progress(50)

            exec_result = run_code_safely(
                generated.code,
                st.session_state.uploaded_path
            )

# Step 3: Generate chart and narrative
            status.markdown("**Step 3/3** — Building report...")
            progress.progress(80)

            chart_path = None
            if exec_result.success and exec_result.output:
                chart_type = detect_chart_type(generated.code, exec_result.output, question, schema_columns=st.session_state.schema.columns)
                                
                chart_path = generate_chart(
                    exec_result.output,
                    chart_type,
                    question,
                    st.session_state.charts_dir,
                    csv_path=st.session_state.uploaded_path
                )
            narrative = (
                generate_narrative(question, exec_result.output, st.session_state.schema)
                if exec_result.success
                else exec_result.error or "Analysis failed."
            )

            progress.progress(100)
            status.empty()
            progress.empty()

        # Build report
        report = AnalysisReport(
            question=   question,
            code=       generated.code,
            result=     exec_result.output if exec_result.success else "",
            narrative=  narrative,
            chart_path= chart_path,
            success=    exec_result.success
        )

        # Save report file
        report_path = save_report(report)

        # Save to history
        history_item = {
            "question":  question,
            "report":    report,
            "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
        }
        st.session_state.history.append(history_item)

        # --------------------------------------------------
        # Show Results
        # --------------------------------------------------

        st.divider()
        st.markdown("## Results")

        if report.success:
            # Narrative answer
            st.markdown(
                f'<div class="analysis-card"><strong>{report.narrative}</strong></div>',
                unsafe_allow_html=True
            )

            result_col, chart_col = st.columns([2, 3])

            with result_col:
                st.markdown("**Raw Output**")
                st.code(report.result or "No output", language="text")

            with chart_col:
                if report.chart_path and os.path.exists(report.chart_path):
                    st.image(report.chart_path, use_container_width=True)
                else:
                    st.info("No chart generated for this result type.")

            with st.expander("View Generated Code"):
                st.code(report.code, language="python")

            # Download button
            report_md = build_report(report)
            st.download_button(
                label="Download Report (.md)",
                data=report_md,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        else:
            st.error(f"Analysis failed: {exec_result.error}")
            with st.expander("View Generated Code"):
                st.code(generated.code, language="python")
            st.info("Try rephrasing your question or check that your column names are correct.")


# --------------------------------------------------
# Welcome Screen - when no file is uploaded
# --------------------------------------------------

elif not uploaded_file:
    st.divider()
    st.markdown("### How it works")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. Upload**")
        st.caption("Drop any CSV file")
    with c2:
        st.markdown("**2. Ask**")
        st.caption("Type your question in plain English")
    with c3:
        st.markdown("**3. Analyze**")
        st.caption("Agent writes and runs pandas code")
    with c4:
        st.markdown("**4. Explore**")
        st.caption("See chart, table, and explanation")