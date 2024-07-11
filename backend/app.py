import streamlit as st
from src.file_search import ChatAgent
from pathlib import Path

def init_chat_agent():
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = ChatAgent()

init_chat_agent()

st.set_page_config(
    page_title="Chat Application",
    page_icon=":speech_balloon:",
    layout="wide"
)

st.title("Chat Application :speech_balloon:")

# File Upload Section
st.sidebar.header("Upload Files")
st.sidebar.write("Upload your files here to make them searchable by the chat agent.")

def upload_files():
    uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True, type=["txt", "pdf", "docx"])
    if uploaded_files:
        Path("./data/files").mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            file_contents = file.read()
            file_path = Path(f"./data/files/{file.name}")
            with open(file_path, "wb") as f:
                f.write(file_contents)
            st.sidebar.success(f"Uploaded {file.name}")

upload_files()

# Chat Section
st.header("Ask the Chat Agent")

def get_response(question: str):
    if not question:
        return "No question provided"
    response = st.session_state.chat_agent.query(question)
    return response

user_input = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_input:
        with st.spinner('Getting response...'):
            response = get_response(user_input)
        st.success("Response:")
        st.write(response)
    else:
        st.error("Please enter a question.")
