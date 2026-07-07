"""Streamlit web UI for the Language Tutor.

Run with:
    streamlit run src/language_tutor/app.py --server.port 8888

Design: dark terminal aesthetic with anime-inspired brand,
warm amber/gold accents, monospace fonts.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from language_tutor import analytics, config, db, llm, srs
from language_tutor.content import fetch_article
from language_tutor.planner import ACTIVITY_REGISTRY, suggest_activities

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Language Tutor",
    page_icon="🗡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark terminal aesthetic
# ---------------------------------------------------------------------------
BRAND_PATH = Path(__file__).parent / "static" / "brand.png"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

    /* Global dark theme */
    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F5A623 !important;
        font-weight: 700 !important;
    }

    h1 { font-size: 2rem !important; letter-spacing: 2px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d0d14;
        border-right: 1px solid #1a1a2e;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00D4FF !important;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: #12121a !important;
        border: 1px solid #1a1a2e;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F5A623 !important;
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #888 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a1a2e !important;
        color: #00D4FF !important;
        border: 1px solid #00D4FF !important;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #00D4FF !important;
        color: #0a0a0f !important;
    }

    /* Text input */
    .stTextInput input, .stTextArea textarea {
        background-color: #12121a !important;
        color: #e0e0e0 !important;
        border: 1px solid #1a1a2e !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 4px;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #00D4FF !important;
    }

    /* Chat input */
    .stChatInput {
        background-color: #12121a !important;
    }

    .stChatInput textarea {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Radio buttons */
    .stRadio label {
        font-family: 'JetBrains Mono', monospace !important;
        color: #ccc !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #0d0d14;
        border-radius: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #888 !important;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF !important;
    }

    /* Dataframes / tables */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Dividers */
    hr {
        border-color: #1a1a2e !important;
    }

    /* Brand watermark */
    .brand-watermark {
        opacity: 0.15;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Correction badges */
    .correction-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .badge-error { background: #3d1515; color: #ff6b6b; border: 1px solid #ff6b6b33; }
    .badge-correct { background: #153d15; color: #6bff6b; border: 1px solid #6bff6b33; }
    .badge-card { background: #15153d; color: #6b6bff; border: 1px solid #6b6bff33; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly dark theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#12121a",
    font=dict(family="JetBrains Mono", color="#e0e0e0", size=12),
    colorway=["#00D4FF", "#F5A623", "#ff6b6b", "#6bff6b", "#b06bff", "#ff6bcd"],
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
    yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialize Streamlit session state."""
    if "conn" not in st.session_state:
        st.session_state.conn = db.get_connection()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tutor" not in st.session_state:
        st.session_state.tutor = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "warmed_up" not in st.session_state:
        st.session_state.warmed_up = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render the sidebar and return the selected page."""
    with st.sidebar:
        # Brand image
        if BRAND_PATH.exists():
            st.image(str(BRAND_PATH), use_container_width=True)

        st.markdown("## ⚔️ LANGUAGE TUTOR")
        st.markdown(
            f"`Level: {config.LEARNER_LEVEL}` · "
            f"`{config.PRIMARY_MODEL.split('/')[-1]}`"
        )
        st.divider()

        page = st.radio(
            "Navigate",
            ["💬 Chat", "📊 Analytics", "🃏 Cards", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats
        conn = st.session_state.conn
        card_stats = db.get_card_stats(conn)
        due_cards = db.get_due_cards(conn)

        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        col1.metric("Cards", card_stats.get("total", 0))
        col2.metric("Due", len(due_cards))

        recent = db.get_recent_sessions(conn, limit=1)
        if recent:
            st.metric("Last session", recent[0]["started_at"][:10])

        return page


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------
def render_chat() -> None:
    """Render the chat/conversation page."""
    st.markdown("# 💬 CONVERSATION")
    conn = st.session_state.conn

    # Activity selection
    if st.session_state.tutor is None:
        st.markdown("### Choose your activity")
        suggestions = suggest_activities(conn)

        cols = st.columns(len(suggestions))
        chosen = None
        for i, act_type in enumerate(suggestions):
            cls = ACTIVITY_REGISTRY[act_type]
            with cols[i]:
                if st.button(f"**{cls.name}**\n\n{cls.description}", key=f"act_{i}",
                             use_container_width=True):
                    chosen = act_type

        if chosen:
            due_cards = db.get_due_cards(conn)
            recent_errors = db.get_recent_errors(conn, limit=5)
            st.session_state.session_id = db.create_session(conn, activity_type=chosen)

            with st.spinner("Connecting to LLM..."):
                llm.warmup()

            tutor = llm.TutorLLM(due_cards=due_cards, recent_errors=recent_errors)

            # Generate opening
            with st.spinner("Preparing session..."):
                opening = tutor.generate_opening()

            st.session_state.tutor = tutor
            st.session_state.messages = [
                {"role": "assistant", "content": opening.message}
            ]
            st.rerun()
        return

    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🗡️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if "corrections" in msg and msg["corrections"]:
                for c in msg["corrections"]:
                    st.markdown(
                        f'<span class="correction-badge badge-error">'
                        f'✗ {c.get("user_said", "")} → {c.get("corrected", "")}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

    # Chat input
    if prompt := st.chat_input("Type in English..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🗡️"):
            with st.spinner("Thinking..."):
                response = st.session_state.tutor.chat(prompt)

            st.markdown(response.message)

            # Show corrections inline
            if response.metadata.corrections:
                for c in response.metadata.corrections:
                    st.markdown(
                        f'<span class="correction-badge badge-error">'
                        f'✗ {c.get("user_said", "")} → {c.get("corrected", "")}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.message,
            "corrections": response.metadata.corrections,
        })

    # End session button
    st.divider()
    if st.button("🛑 End Session", type="secondary"):
        if st.session_state.session_id:
            db.end_session(
                conn, st.session_state.session_id,
                total_turns=len([m for m in st.session_state.messages if m["role"] == "user"]),
                errors_found=sum(len(m.get("corrections", [])) for m in st.session_state.messages),
                cards_reviewed=0,
            )
        st.session_state.tutor = None
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ---------------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------------
def render_analytics() -> None:
    """Render the analytics/stats page."""
    st.markdown("# 📊 ANALYTICS")
    conn = st.session_state.conn

    metrics = analytics._compute_metrics(conn)

    if metrics.total_sessions == 0:
        st.info("No sessions yet. Start a conversation to see your stats!")
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", metrics.total_sessions)
    col2.metric("Turns", metrics.total_turns)
    col3.metric("Errors", metrics.total_corrections)
    col4.metric("Streak", f"{metrics.study_streak}d")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Cards", metrics.total_cards)
    col6.metric("Mastered", metrics.words_mastered)
    col7.metric("Due", len(db.get_due_cards(conn)))
    leeches = len(metrics.leeches)
    col8.metric("Leeches", leeches)

    st.divider()

    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Cards", "🎯 Errors", "🏋️ Practice"])

    with tab1:
        if metrics.cards_by_status:
            fig = go.Figure(data=[go.Pie(
                labels=list(metrics.cards_by_status.keys()),
                values=list(metrics.cards_by_status.values()),
                hole=0.5,
                textfont=dict(family="JetBrains Mono"),
                marker=dict(line=dict(color="#0a0a0f", width=2)),
            )])
            fig.update_layout(
                title="Card Distribution",
                **PLOTLY_LAYOUT,
                showlegend=True,
                legend=dict(font=dict(color="#888")),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if metrics.accuracy_by_type:
            types = list(metrics.accuracy_by_type.keys())
            counts = [v["errors"] for v in metrics.accuracy_by_type.values()]
            fig = go.Figure(data=[go.Bar(
                x=counts, y=types, orientation="h",
                marker=dict(
                    color=counts,
                    colorscale=[[0, "#00D4FF"], [1, "#ff6b6b"]],
                ),
                text=counts, textposition="outside",
                textfont=dict(family="JetBrains Mono", color="#e0e0e0"),
            )])
            fig.update_layout(
                title="Errors by Type",
                **PLOTLY_LAYOUT,
                yaxis=dict(gridcolor="#1a1a2e", autorange="reversed"),
                xaxis=dict(gridcolor="#1a1a2e", title="Count"),
            )
            st.plotly_chart(fig, use_container_width=True)

        if metrics.top_errors:
            st.markdown("### Most Repeated Mistakes")
            for err in metrics.top_errors[:5]:
                st.markdown(
                    f'`{err["user_said"]}` → `{err["corrected"]}` '
                    f'**×{err["count"]}**'
                )

    with tab3:
        if metrics.skills_distribution:
            skills = list(metrics.skills_distribution.keys())
            values = list(metrics.skills_distribution.values())
            fig = go.Figure(data=[go.Scatterpolar(
                r=values + [values[0]],
                theta=skills + [skills[0]],
                fill="toself",
                fillcolor="rgba(0, 212, 255, 0.1)",
                line=dict(color="#00D4FF", width=2),
                marker=dict(size=8, color="#F5A623"),
            )])
            fig.update_layout(
                title="Skills Radar",
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="#12121a",
                    radialaxis=dict(gridcolor="#1a1a2e", color="#888"),
                    angularaxis=dict(gridcolor="#1a1a2e", color="#e0e0e0"),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

    # AI Insights button
    st.divider()
    if st.button("🤖 Generate AI Insights", use_container_width=True):
        with st.spinner("Analyzing your learning data..."):
            report = analytics.generate_report(conn, include_insights=True)
        if report.insights:
            st.markdown(f"### 🧠 AI Analysis\n\n{report.insights}")


# ---------------------------------------------------------------------------
# Cards page
# ---------------------------------------------------------------------------
def render_cards() -> None:
    """Render the card deck management page."""
    st.markdown("# 🃏 CARD DECK")
    conn = st.session_state.conn

    tab1, tab2 = st.tabs(["📋 All Cards", "⏰ Due Now"])

    with tab1:
        rows = conn.execute(
            "SELECT front, back, type, status, times_seen, times_correct, "
            "ease_factor, interval FROM cards ORDER BY updated_at DESC"
        ).fetchall()

        if not rows:
            st.info("No cards yet. Start a conversation to build your deck!")
            return

        for row in rows:
            row = dict(row)
            accuracy = (
                f"{row['times_correct']}/{row['times_seen']}"
                if row["times_seen"] > 0
                else "new"
            )
            status_colors = {
                "new": "🟢", "learning": "🟡", "review": "🔵",
                "relearning": "🟠", "suspended": "🔴",
            }
            icon = status_colors.get(row["status"], "⚪")

            with st.expander(
                f'{icon} **{row["front"]}** — {row["status"]} · {accuracy}'
            ):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Type:** {row['type']}")
                col1.markdown(f"**Definition:** {row['back'] or 'N/A'}")
                col2.markdown(f"**Ease:** {row['ease_factor']:.2f}")
                col2.markdown(f"**Interval:** {row['interval']:.1f} days")

    with tab2:
        due = db.get_due_cards(conn)
        if not due:
            st.success("All caught up! No cards due for review.")
        else:
            st.markdown(f"**{len(due)} card(s) due**")
            for card in due:
                st.markdown(
                    f'- `{card["front"]}` ({card["type"]}, {card["status"]})'
                )


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------
def render_settings() -> None:
    """Render the settings page."""
    st.markdown("# ⚙️ SETTINGS")

    st.markdown("### LLM Provider Cascade")
    for i, model in enumerate(config.LLM_MODELS):
        priority = ["🥇 Primary", "🥈 Fallback", "🥉 Offline"][min(i, 2)]
        st.markdown(f"{priority}: `{model}`")

    st.divider()

    st.markdown("### Current Configuration")
    st.markdown(f"- **Level:** {config.LEARNER_LEVEL}")
    st.markdown(f"- **Target Language:** {config.TARGET_LANGUAGE}")
    st.markdown(f"- **TTS Provider:** {config.TTS_PROVIDER}")
    st.markdown(f"- **STT Provider:** {config.STT_PROVIDER}")
    st.markdown(f"- **Session Duration:** {config.SESSION_DURATION_MINUTES} min")

    st.divider()
    st.markdown(
        "### About\n"
        "**Language Tutor** — LLM conversation + spaced repetition (SRS)\n\n"
        "Built with Qwen3/Llama via Groq, SM-2 algorithm, "
        "edge-tts, Vosk, and Streamlit."
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Main Streamlit app entry point."""
    init_state()
    page = render_sidebar()

    if page == "💬 Chat":
        render_chat()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "🃏 Cards":
        render_cards()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
