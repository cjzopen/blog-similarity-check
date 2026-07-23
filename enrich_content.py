import requests
from bs4 import BeautifulSoup
import json
import time
import re
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Enrich crawled articles with full content")
parser.add_argument("--force", action="store_true", help="Force re-fetch all articles even if content exists")
args = parser.parse_args()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Load raw crawled articles
try:
    with open(r"d:\p\blog-\articles_raw.json", "r", encoding="utf-8") as f:
        raw_articles = json.load(f)
except FileNotFoundError:
    print("articles_raw.json not found! Please run crawl.py first.")
    raw_articles = []

# Load existing enriched articles
try:
    with open(r"d:\p\blog-\articles.json", "r", encoding="utf-8") as f:
        enriched_articles = json.load(f)
except FileNotFoundError:
    enriched_articles = []

# Merge raw crawled list into enriched list (preserving fetched contents)
enriched_map = {art["url"]: art for art in enriched_articles}
articles = []
preserved_count = 0
added_count = 0

for raw in raw_articles:
    url = raw["url"]
    if url in enriched_map:
        existing = enriched_map[url]
        existing["category_id"] = raw["category_id"]
        existing["category_name"] = raw["category_name"]
        # Update metadata from raw list just in case they changed
        existing["title"] = raw.get("title", existing.get("title"))
        existing["description"] = raw.get("description", existing.get("description"))
        articles.append(existing)
        preserved_count += 1
    else:
        articles.append(raw)
        added_count += 1

print(f"Loaded {len(raw_articles)} raw crawled articles.")
print(f"Merged with existing database: preserved {preserved_count} already enriched, added {added_count} new.")

print(f"Total articles to enrich with body content: {len(articles)}")

success_count = 0
for i, art in enumerate(articles):
    url = art["url"]
    # Skip only if content is present and long enough, AND --force is not used
    if not args.force and art.get("content") and len(art["content"]) > 150:
        print(f"[{i+1}/{len(articles)}] Skipping (already has content): {url}")
        success_count += 1
        continue

    print(f"[{i+1}/{len(articles)}] Fetching: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get title, description, keywords, h1 (always override from actual page to ensure accuracy)
        title_tag = soup.find("title")
        if title_tag:
            t = title_tag.get_text(strip=True)
            t = re.sub(r'\s*[|\|｜]\s*鼎新.*$', '', t)
            if t:
                art["title"] = t

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            art["description"] = meta_desc["content"].strip()

        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw and meta_kw.get("content"):
            art["keywords"] = meta_kw["content"].strip()

        h1 = soup.find("h1")
        if h1:
            art["h1"] = h1.get_text(strip=True)

        # Get main article body content from various potential layouts
        body_container = None
        for tag, cls in [
            ("article", "article-body"),
            ("article", "list-case-show"),
            ("div", "content-layout"),
            ("div", "article-content"),
            ("section", "page-content"),
            ("div", "tab-content")
        ]:
            body_container = soup.find(tag, class_=cls)
            if body_container:
                break

        if body_container:
            # Remove scripts, styles, and maybe related articles or other non-content components
            for element in body_container(["script", "style", "iframe", "noscript"]):
                element.extract()
            
            # Extract clean text
            text = body_container.get_text(separator="\n", strip=True)
            art["content"] = text
            success_count += 1
            print(f"  Success! Extracted {len(text)} characters from {body_container.name}.{cls}.")
        else:
            art["content"] = ""
            print("  Warning: no known content container class found.")

    except Exception as e:
        print(f"  Error: {e}")
        art["content"] = ""

    time.sleep(0.3)

    # Save periodically to prevent data loss
    if (i + 1) % 10 == 0 or (i + 1) == len(articles):
        # We must make sure we don't accidentally save empty list if interrupted
        if len(articles) > 0:
            with open(r"d:\p\blog-\articles.json", "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"  Saved progress to articles.json")

# Final save to ensure all updates (including skipped/merged items) are written
if len(articles) > 0:
    with open(r"d:\p\blog-\articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print("Final progress saved to articles.json")

print(f"\nDone! Successfully enriched {success_count}/{len(articles)} articles with body content.")
