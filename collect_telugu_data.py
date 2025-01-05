import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import json
import random
from typing import List, Dict, Set
import math
from urllib.parse import quote
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sources for Telugu text
SOURCES = {
    "wikipedia": "https://te.wikipedia.org/wiki/",
    "andhrajyothy": "https://www.andhrajyothy.com/",
    "eenadu": "https://www.eenadu.net/",
    "vaartha": "https://www.vaartha.com/",
}

async def fetch_wikipedia_articles(session: aiohttp.ClientSession, num_articles: int = 1000) -> List[str]:
    """Fetch articles from Telugu Wikipedia."""
    api_url = "https://te.wikipedia.org/w/api.php"
    articles = []

    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": "100"
    }

    while len(articles) < num_articles:
        try:
            async with session.get(api_url, params=params) as response:
                data = await response.json()
                for article in data["query"]["random"]:
                    title = article["title"]
                    url = SOURCES["wikipedia"] + quote(title)

                    async with session.get(url) as article_response:
                        html = await article_response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        content = soup.find("div", {"id": "mw-content-text"})

                        if content:
                            # Get all paragraphs and headings
                            text_elements = content.find_all(["p", "h1", "h2", "h3"])
                            text = " ".join([elem.get_text() for elem in text_elements])
                            text = clean_text(text)

                            if len(text) > 500:  # Increased minimum length
                                articles.append(text)
                                logger.info(f"Collected Wikipedia article: {title[:50]}...")
        except Exception as e:
            logger.error(f"Error fetching Wikipedia article: {str(e)}")
            await asyncio.sleep(1)

        if len(articles) % 10 == 0:
            logger.info(f"Collected {len(articles)} Wikipedia articles")

    return articles[:num_articles]

async def fetch_news_articles(session: aiohttp.ClientSession, source: str, url: str) -> List[str]:
    """Fetch articles from Telugu news websites."""
    articles = []
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            # Common selectors for news articles
            article_selectors = [
                "article", ".news-item", ".story-content",
                ".article-content", ".news-content"
            ]

            for selector in article_selectors:
                for article in soup.select(selector):
                    text = article.get_text()
                    text = clean_text(text)
                    if len(text) > 300:
                        articles.append(text)
                        logger.info(f"Collected news article from {source}")
    except Exception as e:
        logger.error(f"Error fetching news from {source}: {str(e)}")

    return articles

def clean_text(text: str) -> str:
    """Clean and normalize Telugu text."""
    # Remove references, citations, and special characters
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^\u0C00-\u0C7F\s\.,!?]', '', text)  # Keep only Telugu characters and basic punctuation

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

async def collect_telugu_text(num_articles: int = 3000) -> List[str]:
    """Collect Telugu text from multiple sources."""
    texts = []
    seen_texts: Set[str] = set()  # To avoid duplicates

    # Configure connection pooling
    conn = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes timeout

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # Fetch Wikipedia articles
        wiki_articles = await fetch_wikipedia_articles(session, num_articles=2000)
        for article in wiki_articles:
            if article not in seen_texts:
                texts.append(article)
                seen_texts.add(article)

        # Fetch from news sources
        for source, url in SOURCES.items():
            if source != "wikipedia":
                news_articles = await fetch_news_articles(session, source, url)
                for article in news_articles:
                    if article not in seen_texts:
                        texts.append(article)
                        seen_texts.add(article)

        logger.info(f"Total collected texts: {len(texts)}")

    # Combine short texts to create longer ones
    combined_texts = []
    buffer = ""

    for text in texts:
        if len(buffer) < 1000:
            buffer += text + " "
        else:
            combined_texts.append(buffer.strip())
            buffer = text + " "

    if buffer:
        combined_texts.append(buffer.strip())

    return combined_texts

def save_dataset(articles: List[str], output_file: str = "telugu_dataset.json") -> None:
    """Save the collected articles to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "text": articles,
            "metadata": {
                "total_articles": len(articles),
                "total_chars": sum(len(text) for text in articles),
                "avg_article_length": sum(len(text) for text in articles) / len(articles),
                "sources": list(SOURCES.keys())
            }
        }, f, ensure_ascii=False, indent=2)

async def main():
    logger.info("Starting Telugu text collection...")
    articles = await collect_telugu_text(3000)
    logger.info(f"Collected {len(articles)} articles")
    save_dataset(articles)
    logger.info("Dataset saved to telugu_dataset.json")

if __name__ == "__main__":
    asyncio.run(main())