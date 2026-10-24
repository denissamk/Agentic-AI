import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

st.title("💬 Chatbot with Memory")

# The conversation list IS the memory. We keep it in st.session_state so it
# survives Streamlit's reruns (otherwise it would reset on every message).
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Show the whole conversation so far
for message in st.session_state.conversation:
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["parts"][0]["text"])

# Take new input from the user
user_input = st.chat_input("Say something...")

if user_input:
    # 1. add the user's message to memory and show it
    st.session_state.conversation.append({"role": "user", "parts": [{"text": user_input}]})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. send the ENTIRE conversation so the model has context
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=st.session_state.conversation
        )
        reply = response.text
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"

    # 3. add the AI's reply to memory and show it
    st.session_state.conversation.append({"role": "model", "parts": [{"text": reply}]})
    with st.chat_message("assistant"):
        st.markdown(reply)

# A simple "new chat" button that wipes the memory
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.conversation = []
    st.rerun()
