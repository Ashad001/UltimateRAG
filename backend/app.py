import os
import shutil
import streamlit as st
from typing import Any, Dict, List

from src.directory_reader import (
    MODEL_NAME, MODEL_KWARGS, ENCODE_KWARGS, CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, CHROMA_DB_DIR, METADATA_FILE,
    DocumentProcessor,
    VectorStoreManager,
    ChatRetriever,
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)
llm = ChatGroq(model="llama3-8b-8192")

processor = DocumentProcessor(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
vector_store_manager = VectorStoreManager(embeddings, processor, CHROMA_DB_DIR, METADATA_FILE)
retriever = vector_store_manager.get_vectorstore().as_retriever()
chat_retriever = ChatRetriever(llm)

st.set_page_config(layout="wide")

st.title("Conversational Document Retriever")
st.write("Upload documents, manage them, and have a conversation with the bot.")

# seperate by columns
doc_col, rag_col = st.columns([1, 3])

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = os.urandom(24).hex()

with doc_col:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully.")
        st.experimental_rerun()

    st.header("Remove Documents")
    existing_files = os.listdir(DATA_DIR)
    if existing_files:
        files_to_remove = st.multiselect("Select files to remove", existing_files)
        if st.button("Remove selected files"):
            for file in files_to_remove:
                os.remove(os.path.join(DATA_DIR, file))
            st.success("Selected files removed.")
            st.experimental_rerun()
    else:
        st.write("No files available for removal.")


with rag_col:
    st.header("Chat with the Bot")
    if st.button("Reload Vector Store"):
        retriever = vector_store_manager.get_vectorstore().as_retriever()

    question = st.text_input("Enter your question")
    if st.button("Ask"):
        if question:
            response = chat_retriever.query(retriever, question)
            st.write("Response:", response)
        else:
            st.write("Please enter a question.")

