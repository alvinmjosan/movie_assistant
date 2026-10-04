import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core_agent import get_agent_response
from mcp.mcp_client import call_email_via_mcp

st.set_page_config(page_title="Movie Intelligence Assistant", layout="centered")

st.title("🎬 Movie Assistant")
st.markdown(
    "Ask about movie plots or dialogue, retrieve cited scenes, or send scene reports by email "
    "via the MCP email tool."
)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_email" not in st.session_state:
    st.session_state.pending_email = None
# NEW: remember the last movie discussed so bare follow-ups resolve correctly.
if "last_movie" not in st.session_state:
    st.session_state.last_movie = None

# --- Sidebar info ---
with st.sidebar:
    st.subheader("MCP status")
    try:
        from mcp.mcp_client import MCPClient
        reachable = MCPClient().is_reachable()
    except Exception:
        reachable = False
    st.write("MCP email server:", "🟢 online" if reachable else "🔴 offline")
    st.caption("Start it with: `uvicorn src.mcp.server:app --port 8000`")

    st.subheader("📽️ Indexed movies")
    try:
        from ingestion.vector_store import list_indexed_movies
        titles = list_indexed_movies()
        if titles:
            for t in titles:
                st.markdown(f"- {t}")
        else:
            st.caption("No movies indexed yet.")
    except Exception as e:
        st.caption(f"Could not list movies: {e}")

    # NEW: show what the agent currently considers the "active" movie for follow-ups.
    if st.session_state.last_movie:
        st.caption(f"Active movie: **{st.session_state.last_movie}**")

# --- Render history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander("🔧 Tool calls"):
                for tc in message["tool_calls"]:
                    st.markdown(
                        f"**{tc['name']}**  \nargs: `{tc['args']}`  \nresult: `{tc['result'][:300]}`"
                    )

# --- Email confirmation panel ---
if st.session_state.pending_email:
    pe = st.session_state.pending_email
    st.warning(f"📧 Email pending confirmation — to **{pe['recipient_email']}**")
    st.text_input("To", pe["recipient_email"], key="conf_to", disabled=True)
    st.text_input("Subject", pe["subject"], key="conf_subj", disabled=True)
    st.text_area("Body", pe["body"], key="conf_body", disabled=True, height=160)

    col1, col2 = st.columns(2)
    if col1.button("✅ Send via MCP"):
        result = call_email_via_mcp(pe["recipient_email"], pe["subject"], pe["body"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Email tool result:** {result}",
            "tool_calls": [{"name": "send_email", "args": pe, "result": result}],
        })
        st.session_state.pending_email = None
        st.rerun()
    if col2.button("❌ Cancel"):
        st.session_state.pending_email = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Email cancelled.",
        })
        st.rerun()

# --- Chat input ---
if prompt := st.chat_input("E.g., What happens to Neo? Or 'Email bob@x.com the quote from Trinity.'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing request, querying vector database, and routing tools..."):
            history_for_agent = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            # NEW: pass the remembered movie into the agent.
            res = get_agent_response(
                prompt,
                history_for_agent,
                auto_send_email=False,
                last_movie=st.session_state.last_movie,
            )

            st.markdown(res["answer"])
            if res["tool_calls"]:
                with st.expander("🔧 Tool calls"):
                    for tc in res["tool_calls"]:
                        st.markdown(
                            f"**{tc['name']}**  \nargs: `{tc['args']}`  \nresult: `{tc['result'][:300]}`"
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "tool_calls": res["tool_calls"],
    })

    # NEW: persist the movie the agent just discussed so follow-ups resolve.
    if res.get("last_movie"):
        st.session_state.last_movie = res["last_movie"]

    if res.get("pending_email"):
        st.session_state.pending_email = res["pending_email"]
        st.rerun()