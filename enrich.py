import requests
from bs4 import BeautifulSoup
import json
import time
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

with open(r"d:\p\blog-\articles_raw.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Total articles to enrich: {len(articles)}")

for i, art in enumerate(articles):
    url = art["url"]
    print(f"[{i+1}/{len(articles)}] {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get title from <title> tag
        title_tag = soup.find("title")
        if title_tag:
            t = title_tag.get_text(strip=True)
            # Remove site name suffix like "|鼎新數智"
            t = re.sub(r'\s*[|\|｜]\s*鼎新.*$', '', t)
            if t and len(t) > 2:
                art["title"] = t

        # Get meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            art["description"] = meta_desc["content"].strip()

        # Get meta keywords if available
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw and meta_kw.get("content"):
            art["keywords"] = meta_kw["content"].strip()

        # Get h1
        h1 = soup.find("h1")
        if h1:
            art["h1"] = h1.get_text(strip=True)

    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(0.2)

# Save enriched data
with open(r"d:\p\blog-\articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\nDone! Saved enriched data to articles.json")

# Summary
for cid in ["4", "7", "9", "10"]:
    subset = [a for a in articles if a["category_id"] == cid]
    print(f"\n[blog/{cid}] {subset[0]['category_name'] if subset else ''}: {len(subset)} articles")
    for a in subset[:3]:
        print(f"  - {a['title'][:60]}")
