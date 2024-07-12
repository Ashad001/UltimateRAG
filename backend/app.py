import regex as re
import streamlit as st
from src.file_search import FileAgent
from src.webpage_search import WebpageSearch
from pathlib import Path

if 'file_agent' not in st.session_state:
    st.session_state.file_agent = FileAgent()
    
if 'webpage_search' not in st.session_state:
    st.session_state.webpage_agent = WebpageSearch()
    
if "search_flag" not in st.session_state:
    st.session_state.search_flag = False

if "files_flag" not in st.session_state:
    st.session_state.files_flag = False

st.set_page_config(
    page_title="Chat Application",
    page_icon=":speech_balloon:",
    layout="wide"
)

# st.title("Chat Application :speech_balloon:")

col1, col2 = st.columns([1, 3])

# File Upload Section
with col1:
    st.header("Upload Files")
    st.write("Upload your files here to make them searchable by the chat agent.")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["txt", "pdf", "docx"])
    if uploaded_files:
        st.session_state.files_flag = True
        Path("./data/files").mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            file_contents = file.read()
            file_path = Path(f"./data/files/{file.name}")
            with open(file_path, "wb") as f:
                f.write(file_contents)
            st.success(f"Uploaded {file.name}")
        st.session_state.file_agent.feed_files()
    
    st.header("Search Webpages")
    search_input = st.text_input("Search for webpages:")
    if st.button("Search"):
        if search_input:
            with st.spinner('Searching webpages...'):
                # use regex to split the urls
                urls = re.findall(r"\bhttps?://\S+\b", search_input)
                print(urls)
                links = st.session_state.webpage_agent.feed(urls)
                st.session_state.search_flag = True
            st.success("Search Results:")
        else:
            st.error("Please enter a search query.")

    if st.button("Reset"):
        st.session_state.file_agent.reset()
        st.session_state.webpage_search.reset()
        st.success("Files have been reset.")
        st.session_state.files_flag = False
        st.session_state.search_flag = False
            

# Chat Agent Section
with col2:
    
    st.header("Ask the Chat Agent")
    # column for webpage search
    user_input = st.text_input("Ask a question:")

    if st.button("Submit"):
        if user_input:
            with st.spinner('Getting response...'):
                if st.session_state.file_agent and "files_flag" in st.session_state and st.session_state.files_flag:
                    file_response = st.session_state.file_agent.query(user_input)
                if st.session_state.webpage_agent and "search_flag" in st.session_state and st.session_state.search_flag:
                    web_response = st.session_state.webpage_agent.query(user_input)
            st.success("Response:")
            if file_response:
                st.write(file_response)
            if web_response:
                st.write(web_response)
        else:
            st.error("Please enter a question.")
        