import os
import time
import hashlib

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Constants
MODEL_NAME = "BAAI/bge-small-en"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": True}
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
DATA_DIR = "./data/files"
CHROMA_DB_DIR = "./chroma_db"
METADATA_FILE = os.path.join(CHROMA_DB_DIR, "metadata.txt")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS
)
llm = ChatGroq(model="llama3-8b-8192")

def get_directory_hash():
    files = sorted(os.listdir(DATA_DIR))
    return hashlib.md5(str(files).encode()).hexdigest()

def load_documents():
    docs = []
    splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            try:
                doc = PyPDFLoader(file_path=os.path.join(DATA_DIR, file)).load()
                split = text_splitter.split_documents(doc)
                docs.extend(doc)
                splits.extend(split)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return docs, splits

def get_vectorstore():
    current_hash = get_directory_hash()
    
    if os.path.exists(CHROMA_DB_DIR) and os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            stored_hash = f.read().strip()
        
        if current_hash == stored_hash:
            print("Loading existing vectorstore...")
            return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    print("Updating vectorstore...")
    _, splits = load_documents()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    with open(METADATA_FILE, 'w') as f:
        f.write(current_hash)
    
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain

def setup_vectorstore():
    return get_vectorstore().as_retriever()

def query(retriever, question):
    rag_chain = create_rag_chain(retriever)
    response = rag_chain.invoke(
        {"input": question},
        config= {
            "configurable": {
                "session_id": "scorp123"
            },
        }
    )
    return response['answer']

def main():
    retriever = setup_vectorstore()
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        response = query(retriever, question)
        print("\nResponse:")
        print(response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()