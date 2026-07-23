import requests
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

with open(r"d:\p\blog-\articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Checking status codes for {len(articles)} articles...")

redirects_found = 0
for i, art in enumerate(articles):
    url = art["url"]
    
    try:
        # We use HEAD first as it's much faster and uses less bandwidth
        resp = requests.head(url, headers=headers, allow_redirects=False, timeout=8)
        
        # If HEAD is not allowed, use GET
        if resp.status_code in [405, 400, 501]:
            resp = requests.get(url, headers=headers, allow_redirects=False, timeout=8)
            
        status = resp.status_code
        art["status_code"] = status
        
        if status in [301, 302]:
            loc = resp.headers.get("Location", "")
            art["redirect_url"] = loc
            redirects_found += 1
            print(f"[{i+1}/{len(articles)}] {status} Redirect: {url} -> {loc}")
        else:
            if "redirect_url" in art:
                del art["redirect_url"]
            if status != 200:
                print(f"[{i+1}/{len(articles)}] Status {status}: {url}")
                
    except Exception as e:
        print(f"[{i+1}/{len(articles)}] Error checking {url}: {e}")
        art["status_code"] = 0 # Unknown

    time.sleep(0.05)

    if (i + 1) % 20 == 0 or (i + 1) == len(articles):
        with open(r"d:\p\blog-\articles.json", "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"  Saved progress... Check status: {i+1} done.")

print(f"\nCompleted! Found {redirects_found} redirects out of {len(articles)} articles.")
