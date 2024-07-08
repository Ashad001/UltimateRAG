import os
import requests
from typing import List
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex


class WebSearch:
    def __init__(self):
        load_dotenv()
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
            embed_batch_size=16,
        )
        self.search_results = []
    
    def feed_urls(self, urls: List[str]):
        self.urls = urls
        
    def search(self):
        for url in self.urls:
            url = "https://r.jina.ai/" + url
            response = requests.get(url)
            self.search_results.append(response.text)
    
    def save_results(self):
        for i, result in enumerate(self.search_results):
            with open(f"./data/search_result_{i}.txt", "w", encoding="utf-8") as f:
                f.write(result)
                
    def ingest(self):
        if len(os.listdir("./data")) == 0:
            self.search()
            self.save_results()
        try:
            self.documents = SimpleDirectoryReader("./data").load_data()
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            
    def index(self):
        self.vector_index = VectorStoreIndex.from_documents(self.documents, embed_model=self.embed_model)
        self.sq = self.vector_index.as_query_engine()
        
    def search_query(self, query: str) -> List[str]:
        return self.sq.query(query)
    
    def reset(self):
        for file in os.listdir("./data"):
            os.remove(f"./data/{file}")
        self.search_results = []
        self.documents = []
        self.vector_index = None
        self.sq = None
        
    def feed(self, urls: List[str]):
        self.feed_urls(urls)
        self.search()
        self.save_results()
        self.ingest()
        self.index()
    

if __name__ == "__main__":
    ws = WebSearch()
    ws.reset()
    ws.feed(urls=['https://github.com/Ashad001'])
    res = ws.search_query("what Pinned projects does he have and what stack has he worked on?")
    print(res)
    with open("search_results.txt", "w") as f:
        f.write(str(res))