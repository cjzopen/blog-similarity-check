import requests
from bs4 import BeautifulSoup
import json
import re
import time

BASE = "https://www.digiwin.com.tw"

CATEGORIES = [
    {"id": "1", "key": "1/index", "name": "全球動態", "article_pattern": r"^/blog/1/index/(\d+)\.html"},
    {"id": "2", "key": "2/index", "name": "焦點管理", "article_pattern": r"^/blog/2/index/(\d+)\.html"},
    {"id": "3", "key": "3/index", "name": "專家開講", "article_pattern": r"^/blog/3/index/(\d+)\.html"},
    {"id": "4", "key": "4/index", "name": "趨勢議題", "article_pattern": r"^/blog/4/index/(\d+)\.html"},
    {"id": "5", "key": "5/index", "name": "阿傑's科技轉角巷", "article_pattern": r"^/blog/5/index/(\d+)\.html"},
    {"id": "6", "key": "6/index", "name": "影音導讀", "article_pattern": r"^/blog/6/index/(\d+)\.html"},
    {"id": "7", "key": "7/index", "name": "工業電腦智酷", "article_pattern": r"^/blog/7/index/(\d+)\.html"},
    {"id": "9", "key": "9/index", "name": "工廠管理大補帖", "article_pattern": r"^/blog/9/index/(\d+)\.html"},
    {"id": "10", "key": "10/index", "name": "行業管理知識+", "article_pattern": r"^/blog/10/index/(\d+)\.html"},
    {"id": "erp", "key": "erp", "name": "ERP 知識庫", "article_pattern": r"^/blog/erp/(\d+)\.html"},
    {"id": "artificial-intelligence", "key": "artificial-intelligence", "name": "企業AI專欄", "article_pattern": r"^/blog/artificial-intelligence/(\d+)\.html"}
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

all_articles = []
seen_urls = set()

for cat in CATEGORIES:
    cat_id = cat["id"]
    cat_key = cat["key"]
    cat_name = cat["name"]
    pattern = cat["article_pattern"]
    
    page = 1
    consecutive_empty = 0
    
    while True:
        # Build category URL
        url = f"{BASE}/blog/{cat_key}" if page == 1 else f"{BASE}/blog/{cat_key}?page={page}"
        print(f"[{cat_name}] Fetching page {page}: {url}")
        
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.encoding = "utf-8"
        except Exception as e:
            print(f"  Error: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        found = 0

        # Match links using pattern
        for a in soup.find_all("a", href=True):
            href = a["href"]
            match = re.match(pattern, href)
            if not match:
                continue

            full_url = BASE + href
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            title = ""
            desc = ""

            # Get title from link text
            title_text = a.get_text(strip=True)

            # Navigate up to find the article card
            parent = a.find_parent(["div", "li", "article", "dl"])

            # Try multiple approaches for title
            if parent:
                for tag in parent.find_all(["h2", "h3", "h4", "h5", "dt"]):
                    t = tag.get_text(strip=True)
                    if t and len(t) > 2:
                        title = t
                        break

            if not title and title_text and len(title_text) > 3:
                title = title_text

            if not title:
                title = a.get("title", "")

            # Try to get description from parent or sibling
            if parent:
                for tag in parent.find_all(["p", "dd"]):
                    t = tag.get_text(strip=True)
                    if t and len(t) > 10 and t != title:
                        desc = t[:300]
                        break

                if not desc:
                    for tag in parent.find_all("div"):
                        cls = tag.get("class", [])
                        cls_str = " ".join(cls) if cls else ""
                        if any(kw in cls_str for kw in ["desc", "summary", "intro", "text", "des"]):
                            t = tag.get_text(strip=True)
                            if t and len(t) > 10:
                                desc = t[:300]
                                break

            all_articles.append({
                "title": title,
                "description": desc,
                "url": full_url,
                "category_id": cat_id,
                "category_name": cat_name
            })
            found += 1

        print(f"  Found {found} new articles on page {page}")

        if found == 0:
            consecutive_empty += 1
            if consecutive_empty >= 1:
                break
        else:
            consecutive_empty = 0

        # Check max page from pagination
        max_page = page
        for a_tag in soup.find_all("a", href=True):
            # Escaping the cat_key just in case
            escaped_key = re.escape(cat_key)
            m = re.search(rf"blog/{escaped_key}\?page=(\d+)", a_tag["href"])
            if m:
                p = int(m.group(1))
                if p > max_page:
                    max_page = p

        if page >= max_page:
            break

        page += 1
        time.sleep(0.3)

print(f"\n{'='*60}")
print(f"Total articles found: {len(all_articles)}")
for cat in CATEGORIES:
    count = sum(1 for a in all_articles if a['category_id'] == cat['id'])
    print(f"  [{cat['name']}] (blog/{cat['key']}): {count} articles")

# Save raw results
with open(r"d:\p\blog-\articles_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_articles, f, ensure_ascii=False, indent=2)

print(f"\nSaved to articles_raw.json")
