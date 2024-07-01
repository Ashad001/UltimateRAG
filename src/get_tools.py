import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.data_ingestion import QueryTools

def get_all_tools():
    load_dotenv()

    if os.path.exists("tools.json"):
        with open("tools.json", "r") as f:
            paper_to_tools_dict = json.load(f)
            all_tools = [t for paper in paper_to_tools_dict for t in paper_to_tools_dict[paper]]
            return all_tools

    papers = os.listdir('./data')

    try:
        paper_to_tools_dict = {}
        for paper in papers:
            try:
                print(f"Getting tools for paper: {paper}")
                tools = QueryTools(f"./data/{paper}", Path(paper).stem)
                vector_tool, summary_tool = tools.get_query_tools()
                paper_to_tools_dict[paper] = [vector_tool, summary_tool]
            except Exception as e:
                print(f"Error occurred: {str(e)} in paper: {paper}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    print("Done!")

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    # Save tools to a file
    with open("tools.json", "w") as f:
        json.dump(paper_to_tools_dict, f, indent=4, default=str)
    return all_tools

# all_tools = get_all_tools()