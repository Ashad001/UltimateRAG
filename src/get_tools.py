import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_ingestion import QueryTools

def get_all_tools():
    load_dotenv()

    papers = os.listdir('./data')

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        tools = QueryTools(f"./data/{paper}", Path(paper).stem)
        vector_tool, summary_tool = tools.get_query_tools()
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    print("Done!")

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    return all_tools

# all_tools = get_all_tools()