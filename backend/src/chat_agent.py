from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from src.get_tools import get_all_tools 
from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import os

class ChatAgent:
    def __init__(self):
        load_dotenv()
        self.all_tools = get_all_tools(folder_path="./data/files")
        # self.gemini_llm = Gemini(model="models/gemini-1.0-pro", api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = OpenAI(model="gpt-3.5-turbo") #! gpt-4o for the final version
        self.obj_index = ObjectIndex.from_objects(
            self.all_tools,
            index_cls=VectorStoreIndex,
        )
        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=3)
        self.agent_worker = FunctionCallingAgentWorker.from_tools(
            tool_retriever=self.obj_retriever,
            llm=self.gemini_llm, 
            system_prompt=""" \
            You are an agent designed to answer queries over a set of given papers.
            Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

            """,
            verbose=False
        )
        self.agent = AgentRunner(self.agent_worker)

    def query(self, question):
        response = self.agent.query(question)
        return str(response)

if __name__=="__main__":
    agent = ChatAgent()
    response = agent.query("whose repo is this? and what tools has he worked on?")
    print(response)
