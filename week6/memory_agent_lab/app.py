"""
app.py -- Memory Inspector (Streamlit)

A dashboard that shows all four memory types of a MemoryAgent side by side:
  - Episodic log       (top-left)
  - Semantic profile    (top-right)
  - Knowledge graph      (bottom-left)
  - Procedural corrections (bottom-right)

"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from agent import MemoryAgent
from seed_data import create_demo_agent


st.set_page_config(page_title="Memory Inspector", page_icon="🧠", layout="wide")



if "agent" not in st.session_state:
    st.session_state.agent = create_demo_agent()

agent: MemoryAgent = st.session_state.agent




with st.sidebar:
    st.header("🧠 Memory Inspector")
    st.caption(
        "Type anything below. The agent always logs it to episodic memory, "
        "and also tries to extract a fact, a correction, or a relationship "
        "from it -- watch which panels update."
    )

    with st.form("observe_form", clear_on_submit=True):
        new_text = st.text_area(
            "New observation",
            placeholder=(
                "Examples:\n"
                "- My name is Jordan.\n"
                "- Jordan works on Project Comet.\n"
                "- Always write commit messages in present tense.\n"
                "- Fixed the login bug today."
            ),
            height=110,
        )
        submitted = st.form_submit_button("Observe", use_container_width=True)
        if submitted and new_text.strip():
            result = agent.observe(new_text.strip())
            stored_in = [k for k, v in result.items() if v]
            st.success(f"Stored in: {', '.join(stored_in)}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Reset demo", use_container_width=True):
            st.session_state.agent = create_demo_agent()
            st.rerun()
    with col_b:
        if st.button("⏳ Decay pass", use_container_width=True):
            forgotten = agent.episodic.decay_and_forget()
            st.info(f"Forgot {len(forgotten)} low-importance episode(s).")

    st.divider()
    st.caption(
        "Episodic: raw event log · Semantic: stable profile facts · "
        "Procedural: learned behavior corrections · Graph: entity relationships"
    )



def fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")




st.title("🧠 Memory Inspector")
st.caption("One agent, four memory types, side by side.")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.subheader("📜 Episodic Log")
    st.caption("Raw, timestamped event history. Importance decays over time.")

    episodes = agent.episodic.recent(limit=50)
    if episodes:
        now = datetime.now(timezone.utc)
        df = pd.DataFrame([
            {
                "Time": fmt_time(e.created_at),
                "Content": e.content,
                "Importance": round(e.importance, 2),
                "Effective now": round(agent.episodic.effective_importance(e, now), 3),
                "Tags": ", ".join(e.tags) if e.tags else "-",
            }
            for e in episodes
        ])
        st.dataframe(df, use_container_width=True, height=280, hide_index=True)
    else:
        st.info("No active episodes yet.")

    if agent.episodic.memory_blocks:
        with st.expander(f"🗜️ {len(agent.episodic.memory_blocks)} consolidated memory block(s)"):
            for b in agent.episodic.memory_blocks:
                st.markdown(
                    f"**{b.episode_count} episodes** "
                    f"({fmt_time(b.time_range[0])} → {fmt_time(b.time_range[1])})"
                )
                st.text(b.summary)
                st.divider()

with row1_col2:
    st.subheader("🗂️ Semantic Profile")
    st.caption("Stable facts about the user. New statements overwrite old ones per key.")

    facts = agent.semantic.get_profile()
    if facts:
        df = pd.DataFrame([
            {
                "Key": f.key,
                "Value": f.value,
                "Confidence": round(f.confidence, 2),
                "Source": f.source,
                "Updated": fmt_time(f.updated_at),
            }
            for f in facts
        ])
        st.dataframe(df, use_container_width=True, height=280, hide_index=True)
    else:
        st.info("No semantic facts learned yet.")

with row2_col1:
    st.subheader("🕸️ Knowledge Graph")
    st.caption("Entities and relationships the agent has learned.")

    if agent.graph.get_triples():
        st.graphviz_chart(agent.graph.to_dot(), use_container_width=True)
        with st.expander(f"📋 {len(agent.graph.get_triples())} raw triple(s)"):
            df = pd.DataFrame(agent.graph.get_triples(), columns=["Subject", "Relation", "Object"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No relationships learned yet.")

with row2_col2:
    st.subheader("🛠️ Procedural Corrections")
    st.caption("Learned behavioral rules, distilled from feedback.")

    corrections = agent.procedural.get_active()
    if corrections:
        df = pd.DataFrame([
            {
                "Trigger": c.trigger_context,
                "Instruction": c.instruction,
                "Applied": c.times_applied,
                "Created": fmt_time(c.created_at),
            }
            for c in corrections
        ])
        st.dataframe(df, use_container_width=True, height=280, hide_index=True)
    else:
        st.info("No corrections learned yet.")



st.divider()
st.subheader("🔎 Unified Context Preview")
st.caption(
    "Shows what the agent pulls from ALL FOUR stores to ground a response to a given query."
)

query = st.text_input("Preview query", value="Tell me about Project Falcon")

if query.strip():
    context = agent.get_context_for_query(query.strip())
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**📜 Episodic**")
        if context["episodic"]:
            for e in context["episodic"]:
                st.markdown(f"- {e.content}")
        else:
            st.markdown("_none matched_")

    with c2:
        st.markdown("**🗂️ Semantic**")
        if context["semantic"]:
            for f in context["semantic"]:
                st.markdown(f"- **{f.key}**: {f.value}")
        else:
            st.markdown("_none matched_")

    with c3:
        st.markdown("**🕸️ Graph**")
        if context["graph"]:
            for s, r, o in context["graph"]:
                st.markdown(f"- {s} → *{r}* → {o}")
        else:
            st.markdown("_none matched_")

    with c4:
        st.markdown("**🛠️ Procedural**")
        if context["procedural"]:
            for c in context["procedural"]:
                st.markdown(f"- ({c.trigger_context}) {c.instruction}")
        else:
            st.markdown("_none matched_")
