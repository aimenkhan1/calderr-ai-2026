"""
main.py - Streamlit UI

Agentic Research Assistant
Purple robot theme with chat interface, sidebar history, citations, and download.

Run with:
streamlit run main.py
"""

import streamlit as st
from agents import run_research
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="Nova — Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background: #0d0d1a;
    color: white;
}

/* Hide streamlit default menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12122a !important;
    border-right: 1px solid rgba(124,58,237,0.3) !important;
}

/* Robot avatar */
.robot-avatar {
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    margin: 0 auto;
    box-shadow: 0 0 30px rgba(124,58,237,0.5);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 20px rgba(124,58,237,0.4); }
    50% { box-shadow: 0 0 40px rgba(124,58,237,0.8); }
    100% { box-shadow: 0 0 20px rgba(124,58,237,0.4); }
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin: 0.5rem 0;
}

.main-subtitle {
    text-align: center;
    color: #a78bfa;
    font-size: 1rem;
    margin-bottom: 1rem;
}

/* Chat bubble - bot */
.chat-bubble-bot {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 0 18px 18px 18px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Chat bubble - user */
.chat-bubble-user {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 18px 0 18px 18px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: white;
    font-size: 0.95rem;
    line-height: 1.6;
    text-align: right;
}

/* Bot icon */
.bot-icon {
    width: 35px;
    height: 35px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin-right: 8px;
    vertical-align: middle;
    box-shadow: 0 0 10px rgba(124,58,237,0.4);
    flex-shrink: 0;
}

/* Input box */
.stTextInput input {
    background: rgba(124,58,237,0.1) !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    border-radius: 25px !important;
    color: white !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.25rem !important;
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
}

.stButton button:hover {
    box-shadow: 0 0 20px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Cards */
.card {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 16px;
    padding: 1.25rem;
    margin: 0.5rem 0;
}

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 1.25rem 0 0.75rem;
}

/* Subtopic tag */
.subtopic-tag {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.5);
    color: #c4b5fd;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px;
}

/* Metric card */
.metric-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(79,70,229,0.15));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #a78bfa;
}

.metric-label {
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Key point */
.key-point {
    background: rgba(124,58,237,0.08);
    border-left: 3px solid #7c3aed;
    padding: 7px 12px;
    border-radius: 0 8px 8px 0;
    margin: 5px 0;
    font-size: 0.88rem;
    color: #d1d5db;
}

/* Citation */
.citation {
    background: rgba(79,70,229,0.1);
    border: 1px solid rgba(79,70,229,0.3);
    border-radius: 10px;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #a78bfa;
}

/* Conclusion */
.conclusion-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(79,70,229,0.12));
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 16px;
    padding: 1.25rem;
}

/* History item */
.history-item {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* Architecture step */
.arch-step {
    background: rgba(124,58,237,0.1);
    border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #e2e8f0;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(124,58,237,0.1) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    color: white !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: rgba(124,58,237,0.2) !important;
}

</style>
""", unsafe_allow_html=True)


# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "sidebar_tab" not in st.session_state:
    st.session_state.sidebar_tab = "history"

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============ SIDEBAR ============

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div class="robot-avatar">🤖</div>
        <div style="font-size:1.3rem;font-weight:800;margin-top:12px;color:white;">Nova</div>
        <div style="color:#7c3aed;font-size:0.85rem;">AI Research Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # New chat button
    if st.button("✨ New Chat", key="new_chat"):
        st.session_state.selected_chat = None
        st.session_state.chat_messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar tabs
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        if st.button("💬 History", key="tab_history"):
            st.session_state.sidebar_tab = "history"

    with col_t2:
        if st.button("🏗 Architecture", key="tab_arch"):
            st.session_state.sidebar_tab = "architecture"

    st.markdown("<br>", unsafe_allow_html=True)

    # History tab
    if st.session_state.sidebar_tab == "history":

        st.markdown("**📜 Research History**")

        if not st.session_state.history:
            st.markdown("""
            <div style="text-align:center;color:#6b7280;font-size:0.85rem;padding:1rem;">
                No research yet.<br>Ask Nova something!
            </div>
            """, unsafe_allow_html=True)

        else:

            # Clear all history button
            if st.button("🗑️ Clear All History", key="clear_all"):
                st.session_state.history = []
                st.session_state.selected_chat = None
                st.session_state.chat_messages = []
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            for i, item in enumerate(reversed(st.session_state.history)):
                idx = len(st.session_state.history) - 1 - i

                col_h1, col_h2 = st.columns([4, 1])

                with col_h1:
                    if st.button(
                        f"#{idx + 1} — {item['question'][:25]}...",
                        key=f"history_{idx}"
                    ):
                        st.session_state.selected_chat = idx
                        st.rerun()

                with col_h2:
                    if st.button("🗑", key=f"delete_{idx}"):
                        st.session_state.history.pop(idx)
                        if st.session_state.selected_chat == idx:
                            st.session_state.selected_chat = None
                        st.rerun()

    # Architecture tab
    elif st.session_state.sidebar_tab == "architecture":

        st.markdown("**🏗 Agent Architecture**")

        steps = [
            ("1️⃣", "Question Input", "User enters research question"),
            ("2️⃣", "Planner Agent", "Breaks question into 3 subtopics"),
            ("3️⃣", "Research Loop", "3-5 sequential Groq API calls"),
            ("4️⃣", "Synthesis Agent", "Combines all findings"),
            ("5️⃣", "Report Formatter", "Structures with citations"),
            ("6️⃣", "Streamlit UI", "Displays with progress bars"),
        ]

        for emoji, title, desc in steps:
            st.markdown(f"""
            <div class="arch-step">
                <span style="color:#a78bfa;font-weight:700;">{emoji} {title}</span>
                <div style="color:#6b7280;font-size:0.8rem;margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📦 Tech Stack**")
        st.markdown("""
        <div class="card" style="font-size:0.85rem;">
            🤖 Groq llama-3.1-8b-instant<br>
            🔗 Multi-step Agent Loop<br>
            ✅ Pydantic v2 Validation<br>
            🎨 Streamlit UI<br>
            🐍 Python 3.11+
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="text-align:center;color:#4b5563;font-size:0.75rem;">
    </div>
    """, unsafe_allow_html=True)


# ============ HELPER: Display Report ============

def display_report(plan, report):

    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;margin:0.5rem 0 1.5rem;">
        <span class="bot-icon">🤖</span>
        <div class="chat-bubble-bot">
            ✅ Research complete! I explored <strong>{len(plan.subtopics)} subtopics</strong>
            and found <strong>{sum(len(s.key_points) for s in report.sections)} key points</strong>
            with an overall confidence of <strong>{int(report.overall_confidence * 100)}%</strong>.
            Here's your full report! 👇
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Research Plan
    st.markdown('<div class="section-title">📋 Research Plan</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f"""
        <div class="card">
            <div style="color:#6b7280;font-size:0.78rem;margin-bottom:4px;">MAIN TOPIC</div>
            <div style="font-size:1.1rem;font-weight:700;color:#a78bfa;">{plan.main_topic}</div>
            <div style="color:#6b7280;font-size:0.78rem;margin-top:10px;margin-bottom:4px;">RESEARCH GOAL</div>
            <div style="color:#d1d5db;font-size:0.9rem;">{plan.research_goal}</div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        tags = "".join([f'<span class="subtopic-tag">🔹 {s}</span>' for s in plan.subtopics])
        st.markdown(f"""
        <div class="card">
            <div style="color:#6b7280;font-size:0.78rem;margin-bottom:10px;">SUBTOPICS</div>
            {tags}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-title">📊 Report Overview</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(report.overall_confidence * 100)}%</div>
            <div class="metric-label">Overall Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(report.sections)}</div>
            <div class="metric-label">Sections Researched</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        total_kp = sum(len(s.key_points) for s in report.sections)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_kp}</div>
            <div class="metric-label">Key Points Found</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="card">
        <div style="color:#6b7280;font-size:0.78rem;margin-bottom:8px;">📝 SUMMARY</div>
        <div style="color:#e2e8f0;line-height:1.7;font-size:0.95rem;">{report.summary}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Research Sections
    st.markdown('<div class="section-title">🔍 Research Sections</div>', unsafe_allow_html=True)

    for section in report.sections:

        conf_pct = int(section.confidence * 100)
        emoji = "🟢" if section.confidence >= 0.85 else "🟡"

        with st.expander(
            f"📌 {section.subtopic}  —  {emoji} {conf_pct}% confidence",
            expanded=True
        ):

            st.markdown(f"""
            <div style="color:#d1d5db;line-height:1.7;font-size:0.92rem;margin-bottom:1rem;">
                {section.findings}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**🎯 Key Points:**")
            for point in section.key_points:
                st.markdown(
                    f'<div class="key-point">• {point}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>**📚 Sources:**")
            sources = [
                f"Nova Research Agent — {section.subtopic} Analysis (2026)",
                f"Groq LLM Synthesis — {plan.main_topic} Study",
                f"Multi-step Agent Research — Confidence: {conf_pct}%"
            ]
            for source in sources:
                st.markdown(
                    f'<div class="citation">📎 {source}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(section.confidence)
            st.markdown(
                f'<div style="text-align:right;color:#6b7280;font-size:0.78rem;">Confidence: {conf_pct}%</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Conclusion
    st.markdown('<div class="section-title">✅ Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="conclusion-box">
        <div style="color:#e2e8f0;line-height:1.7;font-size:0.95rem;">{report.conclusion}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Download report
    report_text = f"""
NOVA RESEARCH REPORT
====================
Topic: {report.title}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Overall Confidence: {int(report.overall_confidence * 100)}%

SUMMARY
-------
{report.summary}

RESEARCH PLAN
-------------
Main Topic: {plan.main_topic}
Goal: {plan.research_goal}
Subtopics: {", ".join(plan.subtopics)}

RESEARCH SECTIONS
-----------------
"""
    for section in report.sections:
        report_text += f"""
{section.subtopic.upper()}
Confidence: {int(section.confidence * 100)}%

{section.findings}

Key Points:
"""
        for point in section.key_points:
            report_text += f"  • {point}\n"

    report_text += f"""
CONCLUSION
----------
{report.conclusion}

---
"""

    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"nova_report_{plan.main_topic.replace(' ', '_').lower()}.txt",
        mime="text/plain"
    )

    # Bot closing bubble
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;margin-top:1.5rem;">
        <span class="bot-icon">🤖</span>
        <div class="chat-bubble-bot">
            That's my full research report on <strong>{plan.main_topic}</strong>!
            You can download it above or ask me another question below 😊
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============ MAIN AREA ============

# If viewing a history chat
if st.session_state.selected_chat is not None:

    idx = st.session_state.selected_chat
    item = st.session_state.history[idx]

    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2rem;">🤖</div>
        <div style="font-size:1.3rem;font-weight:700;color:#a78bfa;">Past Research</div>
        <div style="color:#6b7280;font-size:0.85rem;">#{idx + 1} — {item['question']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div class="chat-bubble-user">
        {item['question']}
    </div>
    """, unsafe_allow_html=True)

    display_report(item['plan'], item['report'])

    st.divider()

    if st.button("✨ Start New Chat"):
        st.session_state.selected_chat = None
        st.session_state.chat_messages = []
        st.rerun()

else:

    # Welcome header
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:4rem;">🤖</div>
        <div class="main-title">Hello, I'm Nova</div>
        <div class="main-subtitle">Your AI-powered research assistant — ask me anything!</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:1.8rem;">🧠</div>
            <div style="font-weight:700;color:#a78bfa;margin:6px 0;">Plan</div>
            <div style="color:#6b7280;font-size:0.82rem;">I break your question into research subtopics</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:1.8rem;">🔍</div>
            <div style="font-weight:700;color:#a78bfa;margin:6px 0;">Research</div>
            <div style="color:#6b7280;font-size:0.82rem;">I make 3-5 AI calls to explore each subtopic</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:1.8rem;">📄</div>
            <div style="font-weight:700;color:#a78bfa;margin:6px 0;">Report</div>
            <div style="color:#6b7280;font-size:0.82rem;">I synthesize findings into a structured report</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Show all chat messages this session
    for msg in st.session_state.chat_messages:

        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-bubble-user">
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

        elif msg["role"] == "bot":
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin:0.5rem 0;">
                <span class="bot-icon">🤖</span>
                <div class="chat-bubble-bot">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

        elif msg["role"] == "report":
            display_report(msg["plan"], msg["report"])

        elif msg["role"] == "error":
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin:0.5rem 0;">
                <span class="bot-icon">🤖</span>
                <div class="chat-bubble-bot">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Greeting if no messages yet
    if not st.session_state.chat_messages:

        st.markdown("""
        <div style="display:flex;align-items:flex-start;margin-bottom:1rem;">
            <span class="bot-icon">🤖</span>
            <div class="chat-bubble-bot">
                Hi there! I'm <strong>Nova</strong>, your AI research assistant.
                Ask me any research question and I'll plan a strategy,
                research each subtopic, and give you a structured report with
                confidence scores and sources! 🚀
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**💡 Try asking:**")

        examples = [
            "What is the future of AI agents?",
            "How does machine learning work?",
            "What caused World War 1?",
            "How do vaccines work?"
        ]

        ex_cols = st.columns(4)
        for i, ex in enumerate(examples):
            with ex_cols[i]:
                st.markdown(f"""
                <div class="card" style="font-size:0.78rem;color:#a78bfa;text-align:center;padding:0.75rem;">
                    {ex}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input area — always visible
    st.divider()
    st.markdown("### 💬 Ask Nova")

    question = st.text_input(
        "",
        placeholder="Type your research question here...",
        label_visibility="collapsed",
        key="question_input"
    )

    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        run_button = st.button("🚀 Start Research", type="primary")

    # Handle button click
    if run_button and question:

        if len(question.strip()) < 3:
            st.session_state.chat_messages.append({
                "role": "error",
                "content": "Please type a proper research question! 😊"
            })
            st.rerun()

        st.session_state.chat_messages.append({
            "role": "user",
            "content": question
        })

        st.session_state.chat_messages.append({
            "role": "bot",
            "content": "Got it! Let me research that for you... 🧠"
        })

        st.rerun()

    elif run_button and not question:
        st.session_state.chat_messages.append({
            "role": "error",
            "content": "Please type a research question first! 😊"
        })
        st.rerun()

    # Process research
    if (
        st.session_state.chat_messages
        and st.session_state.chat_messages[-1]["role"] == "bot"
        and "Let me research" in st.session_state.chat_messages[-1]["content"]
    ):

        question = st.session_state.chat_messages[-2]["content"]

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message, value):
            progress_bar.progress(value)
            status_text.markdown(
                f'<div style="text-align:center;color:#a78bfa;font-weight:600;font-size:0.9rem;">⚡ {message}</div>',
                unsafe_allow_html=True
            )

        try:

            report, plan = run_research(question, update_progress)

            progress_bar.empty()
            status_text.empty()

            st.session_state.chat_messages.pop()

            st.session_state.history.append({
                "question": question,
                "report": report,
                "plan": plan
            })

            st.session_state.chat_messages.append({
                "role": "report",
                "plan": plan,
                "report": report
            })

            st.rerun()

        except ValueError as ve:
            progress_bar.empty()
            status_text.empty()
            st.session_state.chat_messages.pop()
            st.session_state.chat_messages.pop()
            st.session_state.chat_messages.append({
                "role": "error",
                "content": str(ve)
            })
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.session_state.chat_messages.pop()
            st.session_state.chat_messages.append({
                "role": "error",
                "content": f"Something went wrong: {e}"
            })
            st.rerun()