import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import json
import random

def get_telugu_wikipedia_articles(num_articles=1000):
    base_url = "https://te.wikipedia.org/wiki/"
    api_url = "https://te.wikipedia.org/w/api.php"
    
    articles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": "100"
    }
    
    pbar = tqdm(total=num_articles, desc="Collecting articles")
    
    while len(articles) < num_articles:
        response = requests.get(api_url, params=params)
        data = response.json()
        
        for article in data["query"]["random"]:
            if len(articles) >= num_articles:
                break
                
            title = article["title"]
            article_url = base_url + requests.utils.quote(title)
            
            try:
                article_response = requests.get(article_url)
                soup = BeautifulSoup(article_response.content, "html.parser")
                
                # Get main content
                content_div = soup.find("div", {"id": "mw-content-text"})
                if content_div:
                    paragraphs = content_div.find_all("p")
                    text = " ".join([p.get_text() for p in paragraphs])
                    
                    # Clean text
                    text = re.sub(r'\[.*?\]', '', text)  # Remove references
                    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
                    text = text.strip()
                    
                    if len(text) > 100:  # Only keep articles with substantial content
                        articles.append(text)
                        pbar.update(1)
            
            except Exception as e:
                print(f"Error processing article {title}: {str(e)}")
                continue
    
    pbar.close()
    return articles

def save_dataset(articles, output_file="telugu_dataset.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"text": articles}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("Collecting Telugu articles from Wikipedia...")
    articles = get_telugu_wikipedia_articles(1000)  # Collect 1000 articles
    print(f"Collected {len(articles)} articles")
    save_dataset(articles)
    print("Dataset saved to telugu_dataset.json") 