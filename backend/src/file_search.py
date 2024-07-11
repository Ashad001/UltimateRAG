from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from src.get_tools import get_all_tools
from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import os

class ChatAgent:
    def __init__(self, use_gemini=False, gemini_model="models/gemini-1.0-pro"):
        load_dotenv()
        
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./data/files", exist_ok=True)
        
        self.all_tools = get_all_tools(folder_path="./data/files")
        
        if use_gemini:
            self.llm = self._init_gemini_llm(gemini_model)
        else:
            self.llm = OpenAI(model="gpt-3.5-turbo")  #TODO: Change to "gpt-4o" for the final version

        # Initialize the object index and retriever
        self.obj_index = ObjectIndex.from_objects(self.all_tools, index_cls=VectorStoreIndex)
        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=3)
        
        # Initialize the agent worker and runner
        self.agent_worker = self._create_agent_worker()
        self.agent = AgentRunner(self.agent_worker)

    def _init_gemini_llm(self, model):
        from llama_index.llms.gemini import Gemini
        return Gemini(model=model, api_key=os.getenv("GEMINI_API_KEY"))

    def _create_agent_worker(self):
        system_prompt = """
        You are an agent designed to answer queries over a set of given papers.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.
        """
        return FunctionCallingAgentWorker.from_tools(
            tool_retriever=self.obj_retriever,
            llm=self.llm,
            system_prompt=system_prompt.strip(),
            verbose=False
        )

    def query(self, question):
        response = self.agent.query(question)
        return str(response)

    def reset(self):
        self.agent_worker = self._create_agent_worker()
        self.agent = AgentRunner(self.agent_worker)

if __name__ == "__main__":
    agent = ChatAgent(use_gemini=False)  #? Set to True if using Gemini
    response = agent.query("Whose repo is this? And what tools has he worked on?")
    print(response)
    agent.reset()
    response = agent.query("What is the primary focus of his work?")
    print(response)