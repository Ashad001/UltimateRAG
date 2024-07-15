import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.data_ingestion import QueryTools

def get_all_tools(folder_path: str = "./data/files"):
    load_dotenv()
    
    papers = os.listdir(folder_path)

    try:
        paper_to_tools_dict = {}
        for paper in papers:
            try:
                print(f"Getting tools for paper: {paper}")
                tools = QueryTools(f"{folder_path}/{paper}", Path(paper).stem)
                print(Path(paper).stem)
                vector_tool, summary_tool = tools.get_query_tools()
                paper_to_tools_dict[paper] = [vector_tool, summary_tool]
            except Exception as e:
                raise Exception(f"Error occurred: {str(e)} in paper: {paper}")
    except Exception as e:
        raise Exception(f"Error occurred: {str(e)}")

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    return all_tools
