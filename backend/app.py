import streamlit as st
from backend.src.file_search import ChatAgent

def init_chat_agent():
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = ChatAgent()

init_chat_agent()

st.title("Chat Application")

def get_response(question: str):
    if not question:
        return "No question provided"
    response = st.session_state.chat_agent.query(question)
    return response

user_input = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_input:
        response = get_response(user_input)
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")