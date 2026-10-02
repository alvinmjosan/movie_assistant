import streamlit as st
import sys
import os

# Connect to our agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core_agent import get_agent_response

st.set_page_config(page_title="Movie Intelligence Assistant", layout="centered")

st.title("🎬 Movie Intelligence & Follow-up Assistant")
st.markdown("Ask questions about your movie database, retrieve scenes, quote citations, or have me send an email report via MCP!")

# Initialize the chat session history in Streamlit's state memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages on screen
for message in st.session_state.messages:
    # Streamlit uses "user" and "assistant" roles nicely
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Await user input
if prompt := st.chat_input("E.g., What happens to Neo? Or 'Email John at test@test.com the quote.'"):
    
    # 1. Print and save the User's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get the AI Agent's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request and consulting vector database/tools..."):
            
            # We pass the history so the AI has context of previous questions!
            response_text = get_agent_response(user_query=prompt, chat_history=st.session_state.messages[:-1])
            
            st.markdown(response_text)
            
    # 3. Save the Agent's response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
