"""
Streamlit UI for the Autonomous AI Research Lab.

Runs the pipeline directly (not through the FastAPI layer) so it can show
true phase-by-phase progress as each LangGraph node completes, using
`app.stream()` instead of `app.invoke()`.

Run: streamlit run streamlit_app.py
"""

import streamlit as st

from agents.report_publisher import ReportPublisherAgent
from graph import build_graph
from rag.document_store import DocumentStore

st.set_page_config(page_title="Autonomous AI Research Lab", page_icon="🔬", layout="wide")

PHASE_LABELS = {
    "classify_domain": "🧭 Classifying research domain",
    "assemble_team": "🧩 Dynamically assembling specialist team",
    "generate_hypothesis": "💡 Generating initial hypothesis",
    "evidence_agent": "🔍 Evidence agent investigating (parallel)",
    "critique": "⚖️ Critic Agent reviewing findings",
    "synthesize": "📝 Synthesizing report",
    "peer_review": "🔎 Peer Review Agent doing final check",
    "publish": "📤 Publishing final report",
}

st.title("🔬 Autonomous AI Research Lab")
st.caption(
    "Submit a research question. A dynamically-assembled team of 3–5 specialist "
    "agents will research it end-to-end — with zero human intervention after you hit Run."
)

store = DocumentStore()
with st.expander("📚 Supported research domains (seed document store)"):
    st.write(", ".join(store.available_domains()))

question = st.text_area(
    "Research question",
    placeholder="e.g. Will fault-tolerant quantum computing be capable of breaking RSA encryption before 2035?",
    height=80,
)

run_clicked = st.button("🚀 Run Autonomous Research", type="primary", disabled=not question.strip())

if run_clicked:
    app = build_graph()
    progress_container = st.container()
    phase_status = {}
    status_placeholders = {}

    with progress_container:
        st.subheader("Phase-by-Phase Progress")
        for phase_key, label in PHASE_LABELS.items():
            status_placeholders[phase_key] = st.empty()
            status_placeholders[phase_key].markdown(f"⬜ {label}")

    final_state = None
    evidence_count = 0

    try:
        for chunk in app.stream({
            "question": question, "domain_classification": None, "assembly_plan": None,
            "hypothesis": None, "findings": [], "critic_report": None, "final_findings": [],
            "synthesis": None, "peer_review": None, "report": None, "errors": [],
        }, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "evidence_agent":
                    evidence_count += 1
                    status_placeholders["evidence_agent"].markdown(
                        f"✅ {PHASE_LABELS['evidence_agent']} — {evidence_count} finding(s) received"
                    )
                elif node_name in status_placeholders:
                    status_placeholders[node_name].markdown(f"✅ {PHASE_LABELS[node_name]}")

                if node_name == "assemble_team" and node_output.get("assembly_plan"):
                    plan = node_output["assembly_plan"]
                    with progress_container:
                        st.info(
                            f"Assembled {len(plan.specialists)} specialists: "
                            + ", ".join(s.persona_name for s in plan.specialists)
                        )

                if node_name == "publish" and node_output.get("report"):
                    final_state = node_output

    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline error: {e}")
        st.stop()

    if final_state and final_state.get("report"):
        report = final_state["report"]
        st.success("Research complete!")

        st.header(report.synthesis.title)
        st.caption(f"Domain: {report.domain.value} | Peer review: "
                   f"{'✅ Approved' if report.peer_review.approved else '⚠️ Flagged'}")

        st.subheader("Executive Summary")
        st.write(report.synthesis.executive_summary)

        st.subheader("Hypothesis")
        st.write(f"> {report.hypothesis.statement}")
        st.caption(f"Initial confidence: {report.hypothesis.confidence}")

        st.subheader("Findings")
        for f in report.findings:
            icon = "⚠️" if f.weakened else "✅"
            with st.expander(f"{icon} [{f.finding_id}] {f.persona_name} (confidence={f.confidence})"):
                st.write(f.summary)
                st.caption(f"Sources: {', '.join(f.sources) if f.sources else 'none'}")
                if f.weakened:
                    st.warning(f"Critic note: {f.weakness_note} (was {f.original_confidence})")

        st.subheader("Report Body")
        for section in report.synthesis.sections:
            st.markdown(f"**{section.heading}**")
            st.write(section.content)
            if section.cited_finding_ids:
                st.caption(f"Sources: {', '.join(section.cited_finding_ids)}")

        st.subheader("Conclusion")
        st.write(report.synthesis.conclusion)

        markdown_report = ReportPublisherAgent.to_markdown(report)
        st.download_button(
            "📄 Download full Markdown report", markdown_report,
            file_name="research_report.md", mime="text/markdown",
        )
    else:
        st.error("Pipeline finished but no report was produced. Check the errors above.")
