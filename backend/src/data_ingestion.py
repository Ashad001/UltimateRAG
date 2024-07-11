import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import Settings

from typing import List, Optional

class QueryTools:
    def __init__(self, file_path: str, name: str):
        load_dotenv()
        self.file_path = file_path
        self.name = name
        
        self.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )
        
        self.documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.splitter = SentenceSplitter(chunk_size=1024)
        
        self.nodes = self.splitter.get_nodes_from_documents(self.documents)
  
        self.vector_index = VectorStoreIndex.from_documents(self.documents, embed_model=self.embed_model)
        self.summary_index = SummaryIndex(self.nodes)

    def vector_query(self, query: str, page_numbers: Optional[List[str]] = None) -> str:
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=3,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
    
    def get_query_tools(self):
        vector_query_tool = FunctionTool.from_defaults(
            name=f"vector_tool_{self.name}",
            fn=self.vector_query
        )
        
        summary_query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        
        summary_tool = QueryEngineTool.from_defaults(
            name=f"summary_tool_{self.name}",
            query_engine=summary_query_engine,
            description=f"Useful for summarization questions related to {self.name}",
        )

        return vector_query_tool, summary_tool
