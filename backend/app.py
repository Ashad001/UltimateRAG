import streamlit as st
from src.file_search import FileAgent
from src.webpage_search import WebpageSearch
from pathlib import Path

if 'agent' not in st.session_state:
    st.session_state.file_agent = FileAgent()
    st.session_state.webpage_search = WebpageSearch()
    st.session_state.file_agent.reset()
    st.session_state.webpage_search.reset()
    
    # st.session_state.webpage_search.feed([""])

st.set_page_config(
    page_title="Chat Application",
    page_icon=":speech_balloon:",
    layout="wide"
)

st.title("Chat Application :speech_balloon:")

col1, col2 = st.columns([1, 3])

# File Upload Section
with col1:
    st.header("Upload Files")
    st.write("Upload your files here to make them searchable by the chat agent.")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["txt", "pdf", "docx"])
    if uploaded_files:
        Path("./data/files").mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            file_contents = file.read()
            file_path = Path(f"./data/files/{file.name}")
            with open(file_path, "wb") as f:
                f.write(file_contents)
            st.success(f"Uploaded {file.name}")
    st.session_state.file_agent.feed_files()
            

# Chat Agent Section
with col2:
    st.header("Ask the Chat Agent")

    user_input = st.text_input("Ask a question:")

    if st.button("Submit"):
        if user_input:
            with st.spinner('Getting response...'):
                response = st.session_state.file_agent.query(user_input)
            st.success("Response:")
            st.write(response)
        else:
            st.error("Please enter a question.")
