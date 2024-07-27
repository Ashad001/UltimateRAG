import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings()

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatGroq(model="llama3-8b-8192")


def directory_reader():
    # docs = DirectoryLoader(
    #     path="./data/files",
    #     silent_errors=True,
    #     show_progress=True,         #TODO: change to False
    #     # use_multithreading=True,
    # ).load()
    
   

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        docs = []
        splits = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for file in os.listdir("./data/files"):
            if file.endswith(".pdf"):
                doc = PyPDFLoader(file_path=f"./data/files/{file}").load()
                split = text_splitter.split_documents(doc)
                docs.append(doc)
                splits.extend(split)
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            # persist_directory="./chroma_db",
        )

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke("Summarize the abstract of AI For climate action paper")
    print(response)


if __name__ == "__main__":
    directory_reader()