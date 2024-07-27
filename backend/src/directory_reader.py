import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Watcher:
    def __init__(self, directory_to_watch, callback):
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.observer = Observer()
        self.callback = callback

    def run(self):
        event_handler = Handler(self.callback)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_any_event(self, event):
        if event.is_directory:
            return None
        elif event.event_type == 'created' or event.event_type == 'modified' or event.event_type == 'deleted':
            self.callback()

def update_database():
    docs = DirectoryLoader(
        path="./data/files",
        silent_errors=True,
        show_progress=True,
        use_multithreading=True,
    ).load_documents()

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma.from_directory("./chroma_db")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory="./chroma_db",
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

    response = rag_chain.invoke("Summarize the abstract of Importance of AI in evaluatiing climate change and food safety risk paper")
    print(response)

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings()

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatGroq(model="llama3-8b-8192")

if __name__ == "__main__":
    watcher = Watcher(directory_to_watch="./data/files", callback=update_database)
    watcher.run()
