import requests
from bs4 import BeautifulSoup

def search(url):
    url = "https://r.jina.ai/" + url
    response = requests.get(url)
    
    with open("search_results.txt", "w", encoding='utf-8') as f:
        f.write(response.text)
            
print("Searching for 'Jina AI'...")
search("https://docs.n8n.io/learning-path/")