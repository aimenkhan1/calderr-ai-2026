"""
app.py -- Streamlit Dashboard (Project 6-I-C)

Three views on the same running agent:
  1. Chat -- talk to the agent live; correct it when it's wrong and watch
     the rule book update.
  2. Rule Book -- every rule the agent has learned, with confidence and
     how many times each has actually been used.
  3. Learning Curve -- the error-rate-over-time chart, proving the agent
     actually improves.

Run with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from agent import SelfImprovingAgent
from demo_dataset import CONVERSATION
from learning_curve import compute_windowed_error_rates, plot_learning_curve

st.set_page_config(page_title="Self-Improving Agent", page_icon="🛠️", layout="wide")


# Session state: keep one agent + conversation history alive across reruns

def fresh_agent() -> SelfImprovingAgent:
    for path in ("rules.db", "performance.db"):
        if os.path.exists(path):
            os.remove(path)
    return SelfImprovingAgent(generator_mode="auto", extractor_mode="auto")


if "agent" not in st.session_state:
    st.session_state.agent = fresh_agent()
    st.session_state.chat_history = []   # list of (speaker, text) tuples
    st.session_state.interaction_number = 0

agent: SelfImprovingAgent = st.session_state.agent


# Sidebar: controls

with st.sidebar:
    st.header("Self-Improving Agent")
    backend_note = "Groq (real LLM)" if agent.response_generator.backend == "groq" else "Mock (offline, deterministic)"
    st.caption(f"Response generator: **{backend_note}**")

    st.divider()
    if st.button("▶️ Run scripted 20-interaction demo", use_container_width=True):
        st.session_state.agent = fresh_agent()
        agent = st.session_state.agent
        st.session_state.chat_history = []
        for i, message in enumerate(CONVERSATION, start=1):
            result = agent.handle_turn(message, interaction_number=i)
            st.session_state.chat_history.append(("user", message))
            st.session_state.chat_history.append(("agent", result.agent_reply))
        st.session_state.interaction_number = len(CONVERSATION)
        st.rerun()

    if st.button("🔄 Reset (empty agent)", use_container_width=True):
        st.session_state.agent = fresh_agent()
        st.session_state.chat_history = []
        st.session_state.interaction_number = 0
        st.rerun()

    st.divider()
    st.caption(
        "Tip: after the agent replies, correct it by starting your next "
        "message with **No,**, **Always...**, or **From now on...** -- "
        "watch the Rule Book tab update."
    )


# Main layout: three tabs

st.title("Procedural Memory & Self-Improving Agent")

tab_chat, tab_rules, tab_curve = st.tabs(["💬 Chat", "📖 Rule Book", "📈 Learning Curve"])

# ---- Tab 1: Chat / correction interface --------------------------------- #
with tab_chat:
    for speaker, text in st.session_state.chat_history:
        with st.chat_message("user" if speaker == "user" else "assistant"):
            st.write(text)

    user_message = st.chat_input("Ask a question, or correct the agent's last answer...")
    if user_message:
        st.session_state.interaction_number += 1
        result = agent.handle_turn(user_message, st.session_state.interaction_number)
        st.session_state.chat_history.append(("user", user_message))
        st.session_state.chat_history.append(("agent", result.agent_reply))
        st.rerun()

# ---- Tab 2: Rule Book ----------------------------------------------------- #
with tab_rules:
    st.subheader("Learned rules")
    st.caption("Distilled from corrections. Confidence rises once a rule has been reinforced enough times.")

    rules = agent.rule_store.get_all_rules()
    if rules:
        df = pd.DataFrame([
            {
                "Rule": r.rule_text,
                "Domain": r.domain,
                "Confidence": round(r.confidence, 2),
                "Evidence count": r.evidence_count,
                "Times applied": r.application_count,
                "Created": r.created_at[:19].replace("T", " "),
            }
            for r in rules
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No rules learned yet -- correct the agent in the Chat tab, or run the scripted demo.")

# ---- Tab 3: Learning Curve -------------------------------------------------- #
with tab_curve:
    st.subheader("Error rate over time")

    all_logs = agent.performance_tracker.get_all()
    known_outcomes = [log for log in all_logs if log.was_corrected is not None]

    if known_outcomes:
        max_interaction = max(log.interaction_number for log in all_logs)
        windows = compute_windowed_error_rates(agent.performance_tracker, total_interactions=max_interaction)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_learning_curve(agent.performance_tracker, total_interactions=max_interaction)
            st.pyplot(fig)
        with col2:
            st.markdown("**Error rate by window**")
            for label, rate in windows:
                st.metric(label, f"{rate:.0%}")

        accuracy = agent.performance_tracker.get_rule_application_accuracy()
        st.metric("Overall rule application accuracy", f"{accuracy:.0%}",
                   help="Of all responses where a learned rule was applied, what fraction were accepted (not corrected)?")
    else:
        st.info("Not enough interactions yet to plot a learning curve -- keep chatting, or run the scripted demo.")
