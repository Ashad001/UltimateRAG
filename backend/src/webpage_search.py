import os
import requests
from typing import List
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from src.get_tools import get_all_tools
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner


class WebpageSearch:
    def __init__(self, urls=[]):
        load_dotenv()
        
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./data/webpages", exist_ok=True)
        
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.search_results = []
        self.all_tools = get_all_tools(folder_path="./data/webpages")
        
    def feed_urls(self, urls: List[str]):
        self.urls = urls
        
    def search(self):
        for url in self.urls:
            url = "https://r.jina.ai/" + url
            response = requests.get(url)
            self.search_results.append(response.text)
    
    def save_results(self):
        for i, result in enumerate(self.search_results):
            with open(f"./data/webpages/search_result_{i}.txt", "w", encoding="utf-8") as f:
                f.write(result)

    def index(self):
        # Initialize the object index and retriever
        self.obj_index = ObjectIndex.from_objects(self.all_tools, index_cls=VectorStoreIndex)
        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=3)
        
        # Initialize the agent worker and runner
        self.agent_worker = self._create_agent_worker()
        self.agent = AgentRunner(self.agent_worker)
    
    def _create_agent_worker(self):
        system_prompt = """
        You are an agent designed to answer queries over a set of given webpages.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.
        """
        return FunctionCallingAgentWorker.from_tools(
            tool_retriever=self.obj_retriever,
            llm=self.llm,
            system_prompt=system_prompt.strip(),
            verbose=False
        )
        
    def query(self, query: str):
        response = self.agent.query(query)
        return str(response)
                
    def reset(self):
        for file in os.listdir("./data/webpages"):
            os.remove(f"./data/webpages/{file}")
        self.search_results = []
        self.documents = []
        self.vector_index = None
        self.sq = None
        
    def feed(self, urls: List[str]):
        self.feed_urls(urls)
        self.search()
        self.save_results()
        self.index()
    

if __name__ == "__main__":
    ws = WebpageSearch()
    ws.reset()
    ws.feed(urls=['https://docs.llamaindex.ai/en/stable/examples/embeddings/jinaai_embeddings/'])
    res = ws.query("How to implement JinaAI's embedding in python?")
    print(res)
    with open("search_results.txt", "w") as f:
        f.write(str(res))