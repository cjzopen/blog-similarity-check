import json
import re
import math
import jieba
from collections import Counter, defaultdict

# Load articles
with open(r"d:\p\blog-\articles.json", "r", encoding="utf-8") as f:
    raw_articles = json.load(f)

# Filter out deleted, invalid or empty articles (e.g. content cleared after deletion, or 404)
articles = []
for art in raw_articles:
    # 排除無效狀態碼與沒有實質內容的空殼文章
    if art.get("status_code") in [404, 403, 500]:
        continue
    if not art.get("content") or len(art["content"]) < 100:
        continue
    articles.append(art)

print(f"Loaded {len(raw_articles)} articles, {len(articles)} remaining after filtering empty/invalid ones.")

# ===== NLP Analysis with jieba & TF-IDF =====

# Stop words definition
stops = set("的了是在有和與及為被由從到將會能可以不也都就都要這那些個上下中大小多少新舊，。！？、；：「」（）( )-／/\\_—–|｜#*+=_".split() + list("，。！？、；：「」（）()（）-—／/\\#*+=_[]{}<>:：\"'“‘”’"))
stops.update([
    "我們", "你們", "他們", "自己", "可以", "如何", "什麼", "一個", "一些", "許多", "以及", 
    "因此", "因為", "所以", "但是", "不過", "例如", "包括", "對於", "關於", "目前", 
    "進行", "透過", "提供", "協助", "相關", "重要", "主要", "開始", "需要", "已經", 
    "可能", "同時", "並且", "而且", "為了", "甚至", "第一", "第二", "第三", 
    "部分", "內容", "分享", "文章", "介紹", "進行", "了解", "代表", "影響"
])

print("Tokenizing documents and computing TF-IDF...")

# Tokenize each article
documents = []
for art in articles:
    title = art.get("title", "")
    desc = art.get("description", "")
    body = art.get("content", "")
    
    # Weighting: repeat title and description to emphasize them in TF-IDF
    combined = f"{title} {title} {title} {desc} {desc} {body}"
    
    words = []
    for w in jieba.cut(combined):
        w = w.strip()
        if not w or len(w) < 2 or w in stops:
            continue
        # Standardize English/Acronym terms to uppercase
        if re.match(r'^[A-Za-z0-9+.-]+$', w):
            w = w.upper()
        words.append(w)
    documents.append(words)

# Compute Document Frequency (DF)
N = len(articles)
df_dict = Counter()
for doc in documents:
    for w in set(doc):
        df_dict[w] += 1

# Compute TF-IDF vectors for each article
tf_idf_vectors = []
for i, doc in enumerate(documents):
    tf_dict = Counter(doc)
    doc_len = len(doc)
    vec = {}
    for w, count in tf_dict.items():
        tf = count / doc_len
        # IDF smoothing
        idf = math.log(N / df_dict[w]) + 1.0
        vec[w] = tf * idf
    tf_idf_vectors.append(vec)

# Assign top keywords back to each article
for i, art in enumerate(articles):
    vec = tf_idf_vectors[i]
    sorted_kws = sorted(vec.items(), key=lambda x: x[1], reverse=True)
    # Store top 10 keywords with their TF-IDF scores
    art["extracted_keywords"] = [k for k, v in sorted_kws[:10]]
    art["keyword_weights"] = {k: round(v, 5) for k, v in sorted_kws[:10]}

# Cosine similarity helper
def cosine_similarity(v1, v2):
    intersection = set(v1.keys()) & set(v2.keys())
    if not intersection:
        return 0.0
    numerator = sum(v1[w] * v2[w] for w in intersection)
    sum1 = sum(val**2 for val in v1.values())
    sum2 = sum(val**2 for val in v2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return numerator / denominator

# Character-level Jaccard similarity for titles
def title_jaccard_similarity(a, b):
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0

# Compute all pairwise similarities
print("Computing similarity matrix...")
pairwise_similarities = {}
for i in range(N):
    for j in range(i + 1, N):
        cos_sim = cosine_similarity(tf_idf_vectors[i], tf_idf_vectors[j])
        title_sim = title_jaccard_similarity(articles[i]["title"], articles[j]["title"])
        pairwise_similarities[(i, j)] = {
            "cosine_sim": cos_sim,
            "title_sim": title_sim
        }

# ===== 1. Duplicate & Similarity Analysis =====
print("Analyzing duplicate/similar content...")
duplicates = []
for (i, j), sims in pairwise_similarities.items():
    cos_sim = sims["cosine_sim"]
    title_sim = sims["title_sim"]
    
    # Highly Duplicate: Cosine Sim >= 0.75 OR Title Jaccard >= 0.75
    # Medium-High Risk: Cosine Sim >= 0.65 (or Title Jaccard >= 0.65) but not High
    # Moderately Similar: Cosine Sim between 0.35 and 0.65
    if cos_sim >= 0.75 or title_sim >= 0.75:
        duplicates.append({
            "article_a": articles[i],
            "article_a_idx": i,
            "article_b": articles[j],
            "article_b_idx": j,
            "title_similarity": round(title_sim, 2),
            "body_similarity": round(cos_sim, 2),
            "severity": "high"
        })
    elif cos_sim >= 0.65 or title_sim >= 0.65:
        duplicates.append({
            "article_a": articles[i],
            "article_a_idx": i,
            "article_b": articles[j],
            "article_b_idx": j,
            "title_similarity": round(title_sim, 2),
            "body_similarity": round(cos_sim, 2),
            "severity": "medium_high"
        })
    elif cos_sim >= 0.35:
        duplicates.append({
            "article_a": articles[i],
            "article_a_idx": i,
            "article_b": articles[j],
            "article_b_idx": j,
            "title_similarity": round(title_sim, 2),
            "body_similarity": round(cos_sim, 2),
            "severity": "medium"
        })

# ===== 2. Topic Clustering =====
print("Grouping articles into Topic Clusters...")
# Connected Component Clustering based on similarity threshold
# Threshold = 0.40 represents topical suitability
CLUSTER_THRESHOLD = 0.40
parent = list(range(N))

def find_root(i):
    if parent[i] == i:
        return i
    parent[i] = find_root(parent[i])
    return parent[i]

def union_nodes(i, j):
    root_i = find_root(i)
    root_j = find_root(j)
    if root_i != root_j:
        parent[root_i] = root_j

# Link articles with high similarity
for (i, j), sims in pairwise_similarities.items():
    if sims["cosine_sim"] >= CLUSTER_THRESHOLD:
        union_nodes(i, j)

# Group indices by cluster root
cluster_groups = defaultdict(list)
for i in range(N):
    root = find_root(i)
    cluster_groups[root].append(i)

# Form final clusters with metrics
topic_clusters = []
for root, idxs in cluster_groups.items():
    # Only treat as a cluster if it has 2+ articles, or let it be a single cluster
    # We will list all, but emphasize multi-article clusters
    member_arts = [articles[idx] for idx in idxs]
    
    # Calculate Cohesion: average pairwise cosine similarity
    total_sim = 0.0
    pair_count = 0
    for idx_a in idxs:
        for idx_b in idxs:
            if idx_a < idx_b:
                total_sim += pairwise_similarities[(idx_a, idx_b)]["cosine_sim"]
                pair_count += 1
    
    avg_cohesion = total_sim / pair_count if pair_count > 0 else 1.0
    
    # Determine Cluster Cohesion Level
    if len(idxs) == 1:
        cohesion_level = "獨立主題"
    elif avg_cohesion >= 0.45:
        cohesion_level = "高度聚合"
    elif avg_cohesion >= 0.35:
        cohesion_level = "中度聚合"
    else:
        cohesion_level = "低度聚合"
        
    # Representative Keywords: sum TF-IDF weights across members
    cluster_kws = Counter()
    for idx in idxs:
        for w, val in tf_idf_vectors[idx].items():
            cluster_kws[w] += val
    
    sorted_cluster_kws = [k for k, v in cluster_kws.most_common(10)]
    
    # Identify Pillar Page Candidate:
    # The article in the cluster that has the highest average similarity to all other members
    pillar_idx = idxs[0]
    if len(idxs) > 1:
        max_avg_sim = -1.0
        for i_cand in idxs:
            sum_sim = 0.0
            for i_other in idxs:
                if i_cand != i_other:
                    pair = (min(i_cand, i_other), max(i_cand, i_other))
                    sum_sim += pairwise_similarities[pair]["cosine_sim"]
            avg_sim = sum_sim / (len(idxs) - 1)
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                pillar_idx = i_cand
                
    topic_clusters.append({
        "root_idx": root,
        "indices": idxs,
        "articles": member_arts,
        "cohesion": round(avg_cohesion, 2),
        "cohesion_level": cohesion_level,
        "keywords": sorted_cluster_kws,
        "pillar_article": articles[pillar_idx]
    })

# Sort clusters: multi-article clusters first, sorted by size, then cohesion
topic_clusters.sort(key=lambda x: (len(x["indices"]) > 1, len(x["indices"]), x["cohesion"]), reverse=True)

# Add top cluster connections to each article for the Complete Article table
for i, art in enumerate(articles):
    conns = []
    for j in range(N):
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        sim = pairwise_similarities[pair]["cosine_sim"]
        if sim >= 0.25:
            conns.append((j, sim))
    # Sort connections by similarity descending
    conns.sort(key=lambda x: x[1], reverse=True)
    # Store top 3 related articles
    art["related_connections"] = [
        {
            "title": articles[idx]["title"],
            "url": articles[idx]["url"],
            "similarity": round(sim, 2),
            "category": articles[idx]["category_name"]
        } for idx, sim in conns[:3]
    ]

# ===== 3. Keyword Cannibalization =====
print("Analyzing keyword cannibalization...")
# Group by TF-IDF keywords (if keyword is in top 8 for an article)
keyword_articles = defaultdict(list)
for i, art in enumerate(articles):
    # Use top 8 keywords for cannibalization check
    for kw in art["extracted_keywords"][:8]:
        keyword_articles[kw].append(art)

cannibalization = []
for kw, arts in keyword_articles.items():
    if len(arts) >= 3:
        # Group by category
        cats = defaultdict(list)
        for a in arts:
            cats[a["category_name"]].append(a)
        
        cannibalization.append({
            "keyword": kw,
            "count": len(arts),
            "articles": arts,
            "categories": {k: len(v) for k, v in cats.items()},
            "cross_category": len(cats) > 1
        })

cannibalization.sort(key=lambda x: x["count"], reverse=True)

# ===== 4. Category Keyword Stats =====
cat_stats = {}
for cid in ['1', '2', '3', '4', '5', '6', '7', '9', '10', 'erp', 'artificial-intelligence']:
    cat_arts = [a for a in articles if a['category_id'] == cid]
    if cat_arts:
        # Combine all TF-IDF keywords for articles in this category
        all_cat_kws = []
        for a in cat_arts:
            all_cat_kws.extend(a["extracted_keywords"][:8])
        kw_counter = Counter(all_cat_kws)
        cat_stats[cid] = {
            'name': cat_arts[0]['category_name'],
            'count': len(cat_arts),
            'top_keywords': kw_counter.most_common(12),
            'articles': cat_arts
        }

# ===== 5. DigiKnow External Article Title Similarity & Duplicate Check =====
print("Analyzing external DigiKnow articles title similarity...")
import os
import difflib

# Helper function to clean DigiKnow titles
def clean_dk_title(title):
    t = re.sub(r'^就享知\s*[|｜]\s*', '', title)
    return t.strip()

# Helper function to compute title sequence similarity
def title_seq_similarity(a, b):
    a_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', a.lower())
    b_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', b.lower())
    if not a_clean or not b_clean:
        return 0.0
    return difflib.SequenceMatcher(None, a_clean, b_clean).ratio()

# Load digiknow_titles.json
dk_titles_path = r"C:\Users\User\.gemini\antigravity-ide\brain\a2d66031-f38c-41a6-99d7-1cdf86dde951\scratch\digiknow_titles.json"
if not os.path.exists(dk_titles_path):
    dk_titles_path = "digiknow_titles.json" # Fallback

try:
    with open(dk_titles_path, "r", encoding="utf-8") as f:
        dk_database = json.load(f)
except Exception as e:
    print(f"Error loading digiknow_titles.json: {e}")
    dk_database = {}

digiknow_audit_results = []
dk_duplicate_count = 0
dk_similar_count = 0

# For tagging internal articles as DigiKnow duplicates
# Map internal article index -> list of digiknow matches
internal_dk_matches = defaultdict(list)

for dk_url, dk_info in dk_database.items():
    dk_raw = dk_info.get("raw_title", "")
    dk_clean = clean_dk_title(dk_raw)
    
    # 1. Find redirect sources (articles redirecting to this DigiKnow URL)
    redir_sources = []
    for idx, art in enumerate(articles):
        if art.get("redirect_url") == dk_url:
            redir_sources.append((idx, art))
            
    # 2. Compare against all articles for title similarity
    similar_matches = []
    for idx, art in enumerate(articles):
        # Quick Jaccard filter to save CPU
        jaccard_val = title_jaccard_similarity(dk_clean, art["title"])
        if jaccard_val < 0.30:
            continue
            
        sim_val = title_seq_similarity(dk_clean, art["title"])
        match_score = max(sim_val, jaccard_val)
        
        if match_score >= 0.70:
            severity = "high" # 重複內容 (Title 幾乎一樣)
            label = "重複內容"
            dk_duplicate_count += 1
            similar_matches.append({
                "article": art,
                "index": idx,
                "score": match_score,
                "severity": severity,
                "label": label
            })
            internal_dk_matches[idx].append({
                "dk_url": dk_url,
                "dk_title": dk_clean,
                "score": match_score,
                "severity": severity,
                "label": label
            })
        elif match_score >= 0.65:
            severity = "medium_high" # 高度相似
            label = "高度相似"
            dk_similar_count += 1
            similar_matches.append({
                "article": art,
                "index": idx,
                "score": match_score,
                "severity": severity,
                "label": label
            })
            internal_dk_matches[idx].append({
                "dk_url": dk_url,
                "dk_title": dk_clean,
                "score": match_score,
                "severity": severity,
                "label": label
            })
        elif match_score >= 0.35:
            severity = "medium" # 中度相似
            label = "中度相似"
            similar_matches.append({
                "article": art,
                "index": idx,
                "score": match_score,
                "severity": severity,
                "label": label
            })
            internal_dk_matches[idx].append({
                "dk_url": dk_url,
                "dk_title": dk_clean,
                "score": match_score,
                "severity": severity,
                "label": label
            })

    # Only save if there's any relationship (redirect or similarity matches)
    if redir_sources or similar_matches:
        # Sort similar matches by score desc
        similar_matches.sort(key=lambda x: x["score"], reverse=True)
        
        digiknow_audit_results.append({
            "dk_url": dk_url,
            "dk_raw_title": dk_raw,
            "dk_clean_title": dk_clean,
            "redirect_sources": redir_sources,
            "similar_matches": similar_matches
        })

# ===== Generate HTML Report =====
print("Generating HTML report...")

html = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>鼎新數智 Blog 內容審計報告 (TF-IDF & Topic Cluster)</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+TC:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #0a0e1a;
  --bg-secondary: #111827;
  --bg-card: #1a2035;
  --bg-card-hover: #1f2847;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-purple: #8b5cf6;
  --accent-pink: #ec4899;
  --accent-green: #10b981;
  --accent-orange: #f59e0b;
  --accent-red: #ef4444;
  --border: rgba(255,255,255,0.06);
  --glass: rgba(255,255,255,0.03);
  --radius: 16px;
  --shadow: 0 8px 32px rgba(0,0,0,0.5);
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  font-family: 'Noto Sans TC', 'Inter', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.7;
  min-height: 100vh;
}

body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    radial-gradient(ellipse at 10% 10%, rgba(59,130,246,0.07) 0%, transparent 50%),
    radial-gradient(ellipse at 90% 90%, rgba(139,92,246,0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 60%);
  z-index: 0;
  pointer-events: none;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;
  position: relative;
  z-index: 1;
}

/* Header */
.hero {
  text-align: center;
  padding: 60px 0 45px;
}

.hero h1 {
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 12px;
  letter-spacing: -1px;
}

.hero .subtitle {
  font-size: 1.1rem;
  color: var(--text-secondary);
  font-weight: 300;
}

.hero .meta {
  margin-top: 18px;
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.hero .meta-item {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 20px;
  font-size: 0.9rem;
  color: var(--text-secondary);
  backdrop-filter: blur(10px);
}

.hero .meta-item strong {
  color: var(--accent-cyan);
  font-weight: 700;
  font-size: 1.1rem;
}

/* Summary Box */
.summary-box {
  background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.05));
  border: 1px solid rgba(59,130,246,0.15);
  border-radius: var(--radius);
  padding: 24px;
  margin: 30px 0;
}

.summary-box h3 {
  color: var(--accent-cyan);
  margin-bottom: 14px;
  font-size: 1.25rem;
  font-weight: 700;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding-bottom: 8px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.summary-item h4 {
  font-size: 1rem;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.summary-item ul {
  list-style: none;
  padding: 0;
}

.summary-item li {
  padding: 4px 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.summary-item li::before {
  content: '▸ ';
  color: var(--accent-blue);
  font-weight: bold;
}

/* Stats Grid */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: 3px 3px 0 0;
}

.stat-card:nth-child(1)::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.stat-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink)); }
.stat-card:nth-child(3)::before { background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan)); }
.stat-card:nth-child(4)::before { background: linear-gradient(90deg, var(--accent-orange), var(--accent-pink)); }

.stat-card:hover {
  transform: translateY(-4px);
  border-color: rgba(255,255,255,0.12);
  box-shadow: var(--shadow);
}

.stat-card .cat-name {
  font-size: 0.8rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}

.stat-card .cat-count {
  font-size: 2.5rem;
  font-weight: 800;
  margin-bottom: 2px;
}

.stat-card:nth-child(1) .cat-count { color: var(--accent-blue); }
.stat-card:nth-child(2) .cat-count { color: var(--accent-purple); }
.stat-child(3) .cat-count { color: var(--accent-green); }
.stat-card:nth-child(4) .cat-count { color: var(--accent-orange); }

.stat-card .cat-label {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 12px;
}

.kw-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.kw-tag {
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.75rem;
  color: var(--text-secondary);
  transition: all 0.2s;
}

.kw-tag:hover {
  background: rgba(59,130,246,0.15);
  border-color: var(--accent-blue);
  color: var(--accent-cyan);
}

/* Tabs */
.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
}

.tab-btn {
  background: var(--glass);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 10px 22px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.3s;
  font-family: inherit;
}

.tab-btn:hover { background: rgba(255,255,255,0.06); }

.tab-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
  color: white;
  border-color: transparent;
  font-weight: 600;
  box-shadow: 0 4px 14px rgba(139,92,246,0.3);
}

.tab-content { display: none; }
.tab-content.active { display: block; }

/* Section Header */
.section-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 24px;
}

.section-icon {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
}

.section-icon.purple { background: rgba(139,92,246,0.15); }
.section-icon.red { background: rgba(239,68,68,0.15); }
.section-icon.orange { background: rgba(245,158,11,0.15); }
.section-icon.blue { background: rgba(59,130,246,0.15); }
.section-icon.green { background: rgba(16,185,129,0.15); }

.section-title {
  font-size: 1.5rem;
  font-weight: 700;
}

.section-count {
  background: rgba(139,92,246,0.15);
  color: var(--accent-purple);
  border-radius: 20px;
  padding: 2px 12px;
  font-size: 0.8rem;
  font-weight: 600;
}

.section-count.red { background: rgba(239,68,68,0.15); color: var(--accent-red); }
.section-count.orange { background: rgba(245,158,11,0.15); color: var(--accent-orange); }

/* Topic Cluster Styling */
.cluster-grid {
  display: grid;
  gap: 20px;
}

.cluster-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  transition: all 0.3s;
}

.cluster-card:hover {
  border-color: rgba(139,92,246,0.25);
  box-shadow: var(--shadow);
}

.cluster-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
}

.cluster-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.cluster-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
}

.cohesion-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 8px;
  text-transform: uppercase;
}

.cohesion-badge.high { background: rgba(16,185,129,0.15); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }
.cohesion-badge.medium { background: rgba(59,130,246,0.15); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.3); }
.cohesion-badge.low { background: rgba(245,158,11,0.15); color: var(--accent-orange); border: 1px solid rgba(245,158,11,0.3); }
.cohesion-badge.single { background: rgba(100,116,139,0.15); color: var(--text-secondary); border: 1px solid rgba(100,116,139,0.3); }

.cluster-size {
  background: rgba(255,255,255,0.05);
  color: var(--text-secondary);
  border-radius: 8px;
  padding: 3px 10px;
  font-size: 0.75rem;
  border: 1px solid var(--border);
}

.pillar-box {
  background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(59,130,246,0.04));
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 16px;
}

.pillar-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--accent-cyan);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.pillar-title {
  font-size: 0.95rem;
  font-weight: 600;
}

.pillar-title a {
  color: var(--text-primary);
  text-decoration: none;
  transition: color 0.2s;
}

.pillar-title a:hover { color: var(--accent-cyan); }

.cluster-articles {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.cluster-article-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: rgba(0,0,0,0.15);
  border: 1px solid var(--border);
  border-radius: 10px;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.cluster-article-item:hover {
  background: rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.1);
}

.cluster-article-item a {
  color: var(--text-secondary);
  text-decoration: none;
  flex: 1;
}

.cluster-article-item a:hover { color: var(--accent-blue); }

/* Duplicate Cards */
.dup-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 16px;
  transition: all 0.3s;
}

.dup-card:hover {
  border-color: rgba(239,68,68,0.3);
  box-shadow: 0 0 20px rgba(239,68,68,0.05);
}

.dup-card.medium_high:hover {
  border-color: rgba(249,115,22,0.4);
  box-shadow: 0 0 20px rgba(249,115,22,0.1);
}

.dup-card.medium:hover {
  border-color: rgba(234,179,8,0.3);
  box-shadow: 0 0 20px rgba(234,179,8,0.05);
}

.dup-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.severity-badge {
  border-radius: 8px;
  padding: 4px 12px;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.severity-badge.high {
  background: rgba(239,68,68,0.15);
  color: var(--accent-red);
  border: 1px solid rgba(239,68,68,0.3);
}

.severity-badge.medium_high {
  background: rgba(249,115,22,0.15);
  color: #f97316;
  border: 1px solid rgba(249,115,22,0.3);
}

.severity-badge.medium {
  background: rgba(234,179,8,0.1);
  color: #eab308;
  border: 1px solid rgba(234,179,8,0.2);
}

.sim-score {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.sim-score strong {
  color: var(--text-secondary);
}

.dup-articles {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

@media (max-width: 768px) {
  .dup-articles { grid-template-columns: 1fr; }
}

.dup-article {
  background: rgba(0,0,0,0.2);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid var(--border);
}

.dup-article .cat-badge {
  display: inline-block;
  font-size: 0.7rem;
  padding: 2px 10px;
  border-radius: 12px;
  margin-bottom: 8px;
  font-weight: 600;
}

.cat-badge.cat-1 { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
.cat-badge.cat-2 { background: rgba(139,92,246,0.15); color: var(--accent-purple); }
.cat-badge.cat-3 { background: rgba(6,182,212,0.15); color: var(--accent-cyan); }
.cat-badge.cat-4 { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
.cat-badge.cat-5 { background: rgba(236,72,153,0.15); color: var(--accent-pink); }
.cat-badge.cat-6 { background: rgba(16,185,129,0.15); color: var(--accent-green); }
.cat-badge.cat-7 { background: rgba(139,92,246,0.15); color: var(--accent-purple); }
.cat-badge.cat-9 { background: rgba(16,185,129,0.15); color: var(--accent-green); }
.cat-badge.cat-10 { background: rgba(245,158,11,0.15); color: var(--accent-orange); }
.cat-badge.cat-erp { background: rgba(245,158,11,0.15); color: var(--accent-orange); }
.cat-badge.cat-artificial-intelligence { background: rgba(239,68,68,0.15); color: var(--accent-red); }

.dup-article h4 {
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 6px;
  line-height: 1.5;
}

.dup-article h4 a {
  color: var(--text-primary);
  text-decoration: none;
  transition: color 0.2s;
}

.dup-article h4 a:hover { color: var(--accent-cyan); }

.dup-article .desc {
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Cannibalization Styling */
.cann-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 16px;
}

details.cluster-card summary::-webkit-details-marker,
details.cann-card summary::-webkit-details-marker {
  display: none;
}
details.cluster-card summary,
details.cann-card summary {
  list-style: none;
  outline: none;
}
details.cluster-card[open],
details.cann-card[open] {
  border-color: var(--accent-blue);
  box-shadow: var(--shadow);
}
details.cluster-card .cluster-header,
details.cann-card .cann-header {
  margin-bottom: 0 !important;
}
details.cluster-card[open] .cluster-header {
  margin-bottom: 20px !important;
}
details.cann-card[open] .cann-header {
  margin-bottom: 16px !important;
}

.cann-card:hover { border-color: rgba(245,158,11,0.2); }

.cann-header {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.cann-keyword {
  background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(236,72,153,0.1));
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 10px;
  padding: 6px 16px;
  font-weight: 700;
  font-size: 1rem;
  color: var(--accent-orange);
}

.cann-count {
  font-size: 0.85rem;
  color: var(--text-muted);
}

.cann-cats {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.cann-articles {
  display: grid;
  gap: 8px;
}

.cann-article {
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(0,0,0,0.15);
  border-radius: 10px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  transition: all 0.2s;
}

.cann-article:hover {
  background: rgba(0,0,0,0.25);
  border-color: rgba(255,255,255,0.1);
}

.cann-article a {
  color: var(--text-primary);
  text-decoration: none;
  font-size: 0.85rem;
  flex: 1;
}

.cann-article a:hover { color: var(--accent-cyan); }

/* Heatmap Styling */
.heatmap {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 20px 0;
}

.heat-item {
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: default;
  transition: all 0.3s;
  border: 1px solid transparent;
}

.heat-item:hover {
  transform: scale(1.05);
}

.heat-5 { background: rgba(239,68,68,0.25); color: #fca5a5; border-color: rgba(239,68,68,0.4); font-size: 1.05rem; font-weight: 700; }
.heat-4 { background: rgba(245,158,11,0.2); color: #fcd34d; border-color: rgba(245,158,11,0.3); font-size: 0.95rem; font-weight: 600; }
.heat-3 { background: rgba(59,130,246,0.15); color: #93c5fd; border-color: rgba(59,130,246,0.25); }
.heat-2 { background: rgba(16,185,129,0.1); color: #6ee7b7; border-color: rgba(16,185,129,0.2); }
.heat-1 { background: rgba(100,116,139,0.1); color: var(--text-secondary); border-color: rgba(100,116,139,0.15); font-size: 0.8rem; }

/* Table Styling */
.article-table-wrap {
  overflow-x: auto;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

thead {
  background: rgba(0,0,0,0.3);
}

th {
  padding: 14px 16px;
  text-align: left;
  font-weight: 600;
  color: var(--text-secondary);
  border-bottom: 2px solid var(--border);
  white-space: nowrap;
}

td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}

tr:hover td { background: rgba(255,255,255,0.02); }

td a {
  color: var(--accent-cyan);
  text-decoration: none;
}

td a:hover { text-decoration: underline; }

.td-title {
  font-weight: 500;
  max-width: 300px;
}

.td-desc {
  color: var(--text-muted);
  max-width: 350px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.td-keywords {
  max-width: 200px;
}

.td-relations {
  max-width: 250px;
}

.relation-item {
  font-size: 0.75rem;
  margin-bottom: 4px;
  background: rgba(255,255,255,0.03);
  padding: 4px 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
}

.relation-item a {
  color: var(--text-secondary);
}

.relation-score {
  font-weight: bold;
  color: var(--accent-cyan);
}

/* Filter Bar */
.filter-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  align-items: center;
  flex-wrap: wrap;
}

.filter-input {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 16px;
  color: var(--text-primary);
  font-size: 0.9rem;
  font-family: inherit;
  flex: 1;
  min-width: 200px;
  transition: border-color 0.3s;
}

.filter-input:focus {
  outline: none;
  border-color: var(--accent-blue);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* Cross-category badge */
.cross-badge {
  background: rgba(236,72,153,0.15);
  color: var(--accent-pink);
  border: 1px solid rgba(236,72,153,0.3);
  border-radius: 8px;
  padding: 2px 10px;
  font-size: 0.7rem;
  font-weight: 600;
}

.collapsible-trigger {
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  font-weight: 600;
  color: var(--accent-blue);
  font-size: 0.8rem;
  margin-top: 10px;
}
</style>
</head>
<body>
<div class="container">

<!-- Hero -->
<div class="hero">
  <h1>🔍 鼎新數智 Blog 內容審計報告</h1>
  <p class="subtitle">NLP 關鍵字提取 (jieba+TF-IDF) ｜ 全文重複檢測 ｜ Topic Cluster 關聯性分析</p>
  <div class="meta">
    <div class="meta-item">共計 <strong>""" + str(len(articles)) + """</strong> 篇文章</div>
    <div class="meta-item">深度分析 <strong>11</strong> 個Blog分類</div>
    <div class="meta-item">主題集群 <strong>""" + str(len([c for c in topic_clusters if len(x := c["indices"]) > 1])) + """</strong> 個聚合主題</div>
    <div class="meta-item">內容重複 <strong>""" + str(len(duplicates)) + """</strong> 組相似組</div>
    <div class="meta-item">301/302 轉址 <strong>""" + str(len([a for a in articles if a.get("status_code") in [301, 302]])) + """</strong> 篇</div>
    <div class="meta-item" style="border-color: rgba(239,68,68,0.25);">🌐 DigiKnow 重複 <strong>""" + str(dk_duplicate_count) + """</strong> 篇 (相似 <strong>""" + str(dk_similar_count) + """</strong> 篇)</div>
  </div>
</div>

<!-- Summary Box -->
<div class="summary-box">
  <h3>📊 審計摘要 (TF-IDF & Topic Clusters)</h3>
  <div class="summary-grid">
    <div class="summary-item">
      <h4>📍 關聯性與主題集群 (Topic Clusters)</h4>
      <ul>
        <li>依據 TF-IDF 餘弦相似度對全文進行主題建模，將相似文章歸納為主題集群。</li>
        <li>共檢出 <strong>""" + str(len([c for c in topic_clusters if len(c["indices"]) > 1])) + """</strong> 個包含多篇內容的主題群組，代表適合製作主題集群 (Topic Clusters)。</li>
        <li>為各主題群組自動甄選出 1 篇<strong>柱頁 (Pillar Page) 候選文章</strong>（即群組內與其他文章相似度最高者），引導 SEO 結構化設計。</li>
      </ul>
    </div>
    <div class="summary-item">
      <h4>📍 重複與相似內容排查 (Full Text Similarity)</h4>
      <ul>
        <li>透過<strong>標題 Jaccard 比對 + 內文 TF-IDF 餘弦相似度</strong>，精準比對全文重複度。</li>
        <li><strong>""" + str(len([d for d in duplicates if d['severity'] == 'high'])) + """ 組高度相似/重複</strong>：相似度 ≥ 75%，屬嚴重重複，建議合併。</li>
        <li><strong>""" + str(len([d for d in duplicates if d['severity'] == 'medium_high'])) + """ 組中高度風險重複內容</strong>：關聯度/相似度 ≥ 65%，搜尋權重極易被稀釋，需重點優化。</li>
        <li><strong>""" + str(len([d for d in duplicates if d['severity'] == 'medium'])) + """ 組中度相似/關聯</strong>：相似度在 35% ~ 65% 之間，適合做為 Topic Cluster 的子頁。</li>
        <li>已設定 <strong>""" + str(len([a for a in articles if a.get("status_code") == 301])) + """ 篇 301 重定向轉址</strong>，有效避免內部網址權重分散。</li>
        <li>檢出 <strong>""" + str(dk_duplicate_count) + """ 篇外部 DigiKnow 重複內容</strong> (標題幾乎一樣，直接標記重複)，及 <strong>""" + str(dk_similar_count) + """</strong> 篇高度相似。</li>
      </ul>
    </div>
    <div class="summary-item">
      <h4>📍 關鍵字蠶食風險 (Cannibalization)</h4>
      <ul>
        <li>透過 jieba 分詞並以 TF-IDF 計量，共發現 <strong>""" + str(len(cannibalization)) + """</strong> 個關鍵字存在蠶食風險（被 3+ 篇文章指定為核心關鍵字）。</li>
        <li>高風險關鍵字如：""" + ", ".join([f"「{c['keyword']}」" for c in cannibalization[:4]]) + """ 被多篇文章重複競爭，將分散權重，降低 SEO 排名效率。</li>
      </ul>
    </div>
  </div>
</div>

<!-- Stats Grid -->
<div class="stats-grid">
"""

for i, (cid, stat) in enumerate(cat_stats.items()):
    top_kws = ', '.join([k for k, v in stat['top_keywords'][:8]])
    html += f"""  <div class="stat-card">
    <div class="cat-name">Blog/{cid}</div>
    <div class="cat-count">{stat['count']}</div>
    <div class="cat-label">{stat['name']}</div>
    <div class="kw-tags">
"""
    for kw, cnt in stat['top_keywords'][:8]:
        html += f'      <span class="kw-tag">{kw} ({cnt})</span>\n'
    html += """    </div>
  </div>
"""

html += """</div>

<!-- Tabs Navigation -->
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('clusters')">🔮 Topic Clusters 主題集群 (關聯性)</button>
  <button class="tab-btn" onclick="switchTab('duplicates')">🔴 重複/相似內容</button>
  <button class="tab-btn" onclick="switchTab('digiknow')">🌐 DigiKnow 外部重複</button>
  <button class="tab-btn" onclick="switchTab('cannibalization')">🟡 關鍵字蠶食</button>
  <button class="tab-btn" onclick="switchTab('heatmap')">🔵 關鍵字熱力圖</button>
  <button class="tab-btn" onclick="switchTab('all-articles')">🟢 完整文章清單</button>
</div>

<!-- Tab: Clusters -->
<div class="tab-content active" id="tab-clusters">
  <div class="section">
    <div class="section-header">
      <div class="section-icon purple">🔮</div>
      <h2 class="section-title">主題集群與關聯性分析 (Topic Clusters)</h2>
      <span class="section-count">共 """ + str(len([c for c in topic_clusters if len(c["indices"]) > 1])) + """ 個多篇集群</span>
    </div>
    <p style="color:var(--text-muted);margin-bottom:20px;font-size:0.9rem;">
      關聯性是以全文 TF-IDF 餘弦相似度計算。將關聯度高的文章聚類，便於規劃「柱頁 + 叢頁 (Pillar-Cluster)」結構。<br>
      <strong>柱頁候選文章 (Pillar Page Candidate)</strong> 是群組中與其他文章平均相似度最高的核心，適合規劃為大架構的主題，其他文章則做為子連結 (Cluster Pages)。
    </p>
    
    <div class="cluster-grid">
"""

cluster_num = 0
for cluster in topic_clusters:
    member_count = len(cluster["indices"])
    if member_count <= 1:
        continue # Skip single article clusters to show true clusters
    
    cluster_num += 1
    cohesion = cluster["cohesion"]
    cohesion_level = cluster["cohesion_level"]
    
    cohesion_class = "medium"
    if cohesion_level == "高度聚合":
        cohesion_class = "high"
    elif cohesion_level == "低度聚合":
        cohesion_class = "low"
        
    rep_kws = ', '.join(cluster["keywords"][:6])
    pillar = cluster["pillar_article"]
    
    html += f"""      <details class="cluster-card" name="topic-cluster-group">
        <summary class="cluster-header" style="display:flex; justify-content:space-between; align-items:center;">
          <div class="cluster-meta" style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <span class="cluster-title">🎯 主題集群 #{cluster_num}</span>
            <span class="cohesion-badge {cohesion_class}">{cohesion_level} (凝聚度: {cohesion})</span>
            <span class="cluster-size">{member_count} 篇文章</span>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); max-width: 500px; text-align: right;">
            <strong>主題詞：</strong>{rep_kws}
          </div>
        </summary>
        
        <div style="margin-top:20px;">
          <div class="pillar-box">
            <div class="pillar-label">👑 推薦柱頁文章 (Pillar Page Candidate)</div>
            <div class="pillar-title">
              <span class="cat-badge cat-{pillar['category_id']}">{pillar['category_name']}</span>
              <a href="{pillar['url']}" target="_blank">{pillar['title']}</a>
            </div>
            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:6px;">{pillar.get('description', '')[:140]}...</div>
          </div>
          
          <div style="font-size:0.85rem;color:var(--text-secondary);font-weight:600;margin-bottom:8px;">
            🔗 叢頁文章 (Cluster Pages) 及與柱頁關聯度 (合適度)：
          </div>
          
          <div class="cluster-articles">
"""
    for art in cluster["articles"]:
        if art["url"] == pillar["url"]:
            continue # Skip pillar itself from lists of cluster members or highlight it
        # Compute similarity between this article and pillar
        i_a = articles.index(art)
        i_p = articles.index(pillar)
        pair = (min(i_a, i_p), max(i_a, i_p))
        sim_to_pillar = pairwise_similarities[pair]["cosine_sim"]
        
        warning_badge = ""
        if sim_to_pillar >= 0.65:
            warning_badge = '<span class="severity-badge medium_high" style="font-size:0.7rem;padding:2px 8px;margin-left:8px;white-space:nowrap;display:inline-block;vertical-align:middle;">⚠️ 中高度風險重複</span>'
        
        html += f"""          <div class="cluster-article-item">
            <span class="cat-badge cat-{art['category_id']}">{art['category_name']}</span>
            <a href="{art['url']}" target="_blank">{art['title']}</a>
            {warning_badge}
            <span class="relation-score" title="與柱頁的 TF-IDF 餘弦相似度">關聯度: {int(sim_to_pillar*100)}%</span>
          </div>
"""
        
    html += """        </div>
        </div>
      </details>
"""

html += """    </div>
  </div>
</div>

<!-- Tab: Duplicates -->
<div class="tab-content" id="tab-duplicates">
  <div class="section">
    <div class="section-header">
      <div class="section-icon red">⚠️</div>
      <h2 class="section-title">重複/相似內容排查 (連內文比對)</h2>
      <span class="section-count red">""" + str(len(duplicates)) + """ 組</span>
    </div>
    <p style="color:var(--text-muted);margin-bottom:20px;font-size:0.9rem;">
      透過「標題 Jaccard 比對」與「內文 TF-IDF 餘弦相似度」的加權比對。內文相似度反映真實的文案抄襲與多工發文風險。
    </p>
    
    <div class="filter-bar" style="margin-bottom: 24px;">
      <select class="filter-input" id="dupSeverityFilter" style="flex:0;min-width:220px;" onchange="filterDupCards()">
        <option value="">全部級別</option>
        <option value="high">🔴 重複/高度相似 (相似度 ≥ 75%)</option>
        <option value="medium_high">⚠️ 中高度風險重複 (相似度 65% ~ 75%)</option>
        <option value="medium">🟡 中度相似/主題關聯 (相似度 35% ~ 65%)</option>
      </select>
    </div>
"""

# Sort duplicates by severity then similarity
duplicates.sort(key=lambda x: (0 if x['severity'] == 'high' else (1 if x['severity'] == 'medium_high' else 2), -x['body_similarity'], -x['title_similarity']))

for dup in duplicates:
    a = dup['article_a']
    b = dup['article_b']
    sev = dup['severity']
    label = '高度相似/重複 (建議合併)' if sev == 'high' else ('中高度風險重複內容 (建議重新規劃)' if sev == 'medium_high' else '中度相似/主題關聯')
    html += f"""    <div class="dup-card {sev}">
      <div class="dup-header">
        <span class="severity-badge {sev}">{label}</span>
        <span class="sim-score">內文餘弦相似度 <strong>{int(dup['body_similarity']*100)}%</strong> ｜ 標題相似度 <strong>{int(dup['title_similarity']*100)}%</strong></span>
      </div>
      <div class="dup-articles">
        <div class="dup-article">
          <span class="cat-badge cat-{a['category_id']}">{a['category_name']}</span>
          <h4><a href="{a['url']}" target="_blank">{a['title']}</a></h4>
          <div class="desc">{a.get('description','')[:150]}</div>
          <div style="margin-top:10px;font-size:0.75rem;color:var(--accent-cyan)">
            <strong>核心關鍵字：</strong>{', '.join(a['extracted_keywords'][:5])}
          </div>
        </div>
        <div class="dup-article">
          <span class="cat-badge cat-{b['category_id']}">{b['category_name']}</span>
          <h4><a href="{b['url']}" target="_blank">{b['title']}</a></h4>
          <div class="desc">{b.get('description','')[:150]}</div>
          <div style="margin-top:10px;font-size:0.75rem;color:var(--accent-cyan)">
            <strong>核心關鍵字：</strong>{', '.join(b['extracted_keywords'][:5])}
          </div>
        </div>
      </div>
    </div>
"""

html += """  </div>
</div>
"""

# ===== DigiKnow Tab Content =====
html += """
<!-- Tab: DigiKnow -->
<div class="tab-content" id="tab-digiknow">
  <div class="section">
    <div class="section-header">
      <div class="section-icon blue">🌐</div>
      <h2 class="section-title">DigiKnow 外部重複與相似內容排查 (標題比對)</h2>
      <span class="section-count blue">共 """ + str(len(digiknow_audit_results)) + """ 篇外部文章</span>
    </div>
    <p style="color:var(--text-muted);margin-bottom:20px;font-size:0.9rem;">
      此處比對讀者點擊後會<strong>重定向轉址 (301/302 Redirect)</strong> 至外部網站 <code>https://www.digiknow.com.tw/knowledge/</code> 的文章標題。<br>
      依據需求，<strong>若標題幾乎一樣（字元 SequenceMatcher 相似度 ≥ 70%）直接標示為「重複內容」</strong>，無需檢查內文。這是排查跨域內容重複（Canonicalization / Cross-domain SEO Duplication）的重要依據。
    </p>
    
    <div class="filter-bar" style="margin-bottom: 24px;">
      <select class="filter-input" id="dkSeverityFilter" style="flex:0;min-width:220px;" onchange="filterDkCards()">
        <option value="">全部級別</option>
        <option value="high">🔴 重複內容 (相似度 ≥ 70%)</option>
        <option value="medium_high">🟡 高度相似 (相似度 65% ~ 70%)</option>
        <option value="medium">🔵 中度相似 (相似度 35% ~ 65%)</option>
      </select>
    </div>
    
    <div class="cluster-grid">
"""

for res in digiknow_audit_results:
    dk_url = res["dk_url"]
    dk_raw = res["dk_raw_title"]
    dk_clean = res["dk_clean_title"]
    redirs = res["redirect_sources"]
    matches = res["similar_matches"]
    
    display_matches = [m for m in matches if m["score"] >= 0.35]
    card_sevs = ",".join(list(set([m["severity"] for m in display_matches])))
    
    html += f"""      <div class="cluster-card dk-card" data-severities="{card_sevs}" style="border-left: 4px solid var(--accent-cyan);">
        <div class="cluster-header">
          <div class="cluster-meta">
            <span class="cluster-title" style="color:var(--accent-cyan); font-weight:700;">🌐 外部文章: {dk_clean}</span>
            <span class="cluster-size" style="background:rgba(6,182,212,0.1);color:var(--accent-cyan);border:1px solid rgba(6,182,212,0.2);">就享知平台</span>
          </div>
          <div style="font-size: 0.85rem;">
            <a href="{dk_url}" target="_blank" style="color:var(--accent-cyan);text-decoration:none;font-weight:600;">查看外部原始頁面 ↗</a>
          </div>
        </div>
        
        <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:12px;word-break:break-all;">
          <strong>外部網址：</strong> {dk_url}<br>
          <strong>網頁原始 Title：</strong> {dk_raw}
        </div>
"""
    
    # Show internal redirect sources
    if redirs:
        html += """        <div style="background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.15);border-radius:10px;padding:12px 16px;margin-bottom:16px;">
          <div style="font-size:0.75rem;font-weight:700;color:var(--accent-red);text-transform:uppercase;margin-bottom:6px;">🔗 網站內 301/302 轉址來源：</div>
"""
        for r_idx, r_art in redirs:
            html += f"""          <div style="font-size:0.9rem;display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span class="cat-badge cat-{r_art['category_id']}">{r_art['category_name']}</span>
            <a href="{r_art['url']}" target="_blank" style="color:var(--text-primary);text-decoration:none;font-weight:500;">{r_art['title']}</a>
            <span class="severity-badge high" style="font-size:0.7rem;padding:2px 8px;margin-left:auto;">301 轉址</span>
          </div>"""
        html += """        </div>"""
    else:
        html += """        <div style="background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:10px;padding:10px 16px;margin-bottom:16px;font-size:0.8rem;color:var(--text-muted);">
          ℹ️ 本站無文章直接重定向至此網址
        </div>"""

    # Show title matches
    html += """        <div style="font-size:0.85rem;color:var(--text-secondary);font-weight:600;margin-bottom:8px;">
          🔍 與本站文章標題比對結果：
        </div>
        
        <div class="cluster-articles">
"""
    if display_matches:
        for m in display_matches:
            m_art = m["article"]
            score = m["score"]
            sev = m["severity"]
            
            badge_class = "severity-badge " + ("high" if sev == "high" else ("medium_high" if sev == "medium_high" else "medium"))
            badge_style = "font-size:0.7rem;padding:2px 8px;margin-left:8px;white-space:nowrap;display:inline-block;vertical-align:middle;"
            
            if sev == "high":
                badge_lbl = "🔴 重複內容"
            elif sev == "medium_high":
                badge_lbl = "🟡 高度相似"
            else:
                badge_lbl = "🔵 中度相似"
            
            html += f"""          <div class="cluster-article-item" data-severity="{sev}" style="border-color: { 'rgba(239,68,68,0.2)' if sev=='high' else ('rgba(249,115,22,0.2)' if sev=='medium_high' else 'var(--border)') }">
            <span class="cat-badge cat-{m_art['category_id']}">{m_art['category_name']}</span>
            <a href="{m_art['url']}" target="_blank" style="font-weight:500;">{m_art['title']}</a>
            <span class="{badge_class}" style="{badge_style}">{badge_lbl}</span>
            <span class="relation-score" style="margin-left:auto; color: { 'var(--accent-red)' if sev=='high' else ('var(--accent-orange)' if sev=='medium_high' else 'var(--accent-cyan)') }">相似度: {int(score*100)}%</span>
          </div>
"""
    else:
        html += """          <div style="color:var(--text-muted);font-size:0.8rem;padding:10px;text-align:center;">
            無相似度超過 35% 的文章
          </div>
"""
    html += """        </div>
      </div>
"""

html += """    </div>
  </div>
</div>
"""

# Tab: Cannibalization
html += """<div class="tab-content" id="tab-cannibalization">
  <div class="section">
    <div class="section-header">
      <div class="section-icon orange">🎯</div>
      <h2 class="section-title">關鍵字蠶食分析 (基於 TF-IDF 權重)</h2>
      <span class="section-count orange">""" + str(len([c for c in cannibalization if c['count'] >= 3])) + """ 組</span>
    </div>
    <p style="color:var(--text-muted);margin-bottom:20px;font-size:0.9rem;">
      如果同一個關鍵字在多篇文章中皆擁有極高的 TF-IDF 權重，搜尋引擎在對該關鍵字進行排名時會產生內部競爭，降低首選文章的權重。
    </p>
"""

for cann in cannibalization[:40]:  # Show top 40
    if cann['count'] < 3:
        continue
    html += f"""    <details class="cann-card" name="keyword-cann-group">
      <summary class="cann-header" style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; cursor:pointer;">
        <span class="cann-keyword">{cann['keyword']}</span>
        <span class="cann-count">{cann['count']} 篇文章高度相關</span>
        {"<span class='cross-badge'>跨分類競爭</span>" if cann['cross_category'] else ""}
        <div class="cann-cats" style="display:inline-flex; gap:8px;">
"""
    for cat, cnt in cann['categories'].items():
        html += f'          <span class="kw-tag">{cat}: {cnt}篇</span>\n'
    html += """        </div>
      </summary>
      <div class="cann-articles" style="margin-top:20px;">
"""
    for art in cann['articles'][:8]:
        # Get weight of this keyword in this article
        weight = art.get("keyword_weights", {}).get(cann["keyword"], 0.0)
        html += f"""        <div class="cann-article">
          <span class="cat-badge cat-{art['category_id']}" style="white-space:nowrap">{art['category_name']}</span>
          <a href="{art['url']}" target="_blank">{art['title']}</a>
          <span class="sim-score" style="color:var(--accent-cyan)">權重: {weight:.4f}</span>
        </div>
"""
    if cann['count'] > 8:
        html += f'        <div class="cann-article" style="justify-content:center;color:var(--text-muted);font-size:0.8rem;">... 還有 {cann["count"] - 8} 篇</div>\n'
    html += """      </div>
    </details>
"""

html += """  </div>
</div>

<!-- Tab: Heatmap -->
<div class="tab-content" id="tab-heatmap">
  <div class="section">
    <div class="section-header">
      <div class="section-icon blue">🗺️</div>
      <h2 class="section-title">NLP 提取關鍵字熱力圖 (TF-IDF)</h2>
    </div>
    <p style="color:var(--text-muted);margin-bottom:16px;font-size:0.9rem;">
      基於所有文章計算 TF-IDF 權重後，全域出現頻率最高的關鍵字。字體越大、顏色越暖（偏紅）代表其在各文章中越重要，也伴隨較高的蠶食風險。
    </p>
    <div class="heatmap">
"""

# Global Counter of TF-IDF keywords
all_kws = Counter()
for art in articles:
    for kw in art['extracted_keywords'][:8]:
        all_kws[kw] += 1

for kw, cnt in all_kws.most_common(60):
    if cnt >= 8:
        heat = 'heat-5'
    elif cnt >= 5:
        heat = 'heat-4'
    elif cnt >= 3:
        heat = 'heat-3'
    elif cnt >= 2:
        heat = 'heat-2'
    else:
        heat = 'heat-1'
    html += f'      <span class="heat-item {heat}" title="在 {cnt} 篇文章中為核心詞">{kw} ({cnt})</span>\n'

html += """    </div>

    <h3 style="margin-top:40px;margin-bottom:16px;font-size:1.1rem;color:var(--text-secondary);">各分類關鍵字熱力圖</h3>
"""

for cid, stat in cat_stats.items():
    colors = {
        '1': 'var(--accent-blue)', '2': 'var(--accent-purple)', '3': 'var(--accent-cyan)',
        '4': 'var(--accent-blue)', '5': 'var(--accent-pink)', '6': 'var(--accent-green)',
        '7': 'var(--accent-purple)', '9': 'var(--accent-green)', '10': 'var(--accent-orange)',
        'erp': 'var(--accent-orange)', 'artificial-intelligence': 'var(--accent-red)'
    }
    html += f"""    <div style="margin-bottom:24px;">
      <h4 style="color:{colors[cid]};margin-bottom:10px;font-size:0.95rem;">{stat['name']} (Blog/{cid})</h4>
      <div class="heatmap">
"""
    for kw, cnt in stat['top_keywords']:
        if cnt >= 5:
            heat = 'heat-5'
        elif cnt >= 3:
            heat = 'heat-4'
        elif cnt >= 2:
            heat = 'heat-3'
        else:
            heat = 'heat-2'
        html += f'        <span class="heat-item {heat}" title="{cnt} 篇">{kw} ({cnt})</span>\n'
    html += """      </div>
    </div>
"""

html += """  </div>
</div>

<!-- Tab: All Articles -->
<div class="tab-content" id="tab-all-articles">
  <div class="section">
    <div class="section-header">
      <div class="section-icon green">📋</div>
      <h2 class="section-title">完整文章審計清單</h2>
      <span class="section-count blue">""" + str(len(articles)) + """ 篇</span>
    </div>
    <div class="filter-bar">
      <input type="text" class="filter-input" id="articleFilter" placeholder="🔍 搜尋文章標題、描述或關鍵字..." oninput="filterArticles()">
      <select class="filter-input" id="catFilter" style="flex:0;min-width:160px;" onchange="filterArticles()">
        <option value="">全部分類</option>
        <option value="1">全球動態</option>
        <option value="2">焦點管理</option>
        <option value="3">專家開講</option>
        <option value="4">趨勢議題</option>
        <option value="5">阿傑's科技轉角巷</option>
        <option value="6">影音導讀</option>
        <option value="7">工業電腦智酷</option>
        <option value="9">工廠管理大補帖</option>
        <option value="10">行業管理知識+</option>
        <option value="erp">ERP 知識庫</option>
        <option value="artificial-intelligence">企業AI專欄</option>
      </select>
      <select class="filter-input" id="statusFilter" style="flex:0;min-width:140px;" onchange="filterArticles()">
        <option value="">全部狀態</option>
        <option value="200">正常 (200)</option>
        <option value="301">301 轉址</option>
        <option value="302">302 轉址</option>
      </select>
    </div>
    <div class="article-table-wrap">
      <table id="articlesTable">
        <thead>
          <tr>
            <th style="width: 50px;">#</th>
            <th style="width: 120px;">分類</th>
            <th style="width: 250px;">標題</th>
            <th style="width: 320px;">摘要描述</th>
            <th style="width: 200px;">TF-IDF 關鍵字</th>
            <th style="width: 280px;">Topic Cluster 關聯性推薦</th>
            <th style="width: 80px;">連結</th>
          </tr>
        </thead>
        <tbody>
"""

for i, art in enumerate(articles):
    kws = ', '.join(art.get('extracted_keywords', [])[:6])
    desc_short = art.get('description', '')[:120]
    
    # Generate relation HTML
    rel_html = ""
    relations = art.get("related_connections", [])
    if relations:
        for rel in relations:
            rel_html += f"""          <div class="relation-item">
            <span class="relation-score">{int(rel['similarity']*100)}%</span>
            <a href="{rel['url']}" target="_blank" title="{rel['title']}">{rel['title'][:25]}...</a>
          </div>"""
    else:
        rel_html = "<span style='color:var(--text-muted);font-size:0.75rem;'>無顯著關聯文章</span>"
        
    status = art.get("status_code", 200)
    status_badge = ""
    redirect_info = ""
    if status == 301:
        status_badge = ' <span class="severity-badge" style="font-size:0.7rem;padding:2px 8px;background:rgba(239,68,68,0.15);color:var(--accent-red);border:1px solid rgba(239,68,68,0.3);white-space:nowrap;display:inline-block;vertical-align:middle;margin-left:6px;">301 轉址</span>'
        loc = art.get("redirect_url", "")
        if loc:
            redirect_info = f'<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;word-break:break-all;">轉址至: <a href="{loc}" target="_blank" style="color:var(--accent-cyan);">{loc}</a></div>'
    elif status == 302:
        status_badge = ' <span class="severity-badge" style="font-size:0.7rem;padding:2px 8px;background:rgba(245,158,11,0.15);color:var(--accent-orange);border:1px solid rgba(245,158,11,0.3);white-space:nowrap;display:inline-block;vertical-align:middle;margin-left:6px;">302 轉址</span>'
        loc = art.get("redirect_url", "")
        if loc:
            redirect_info = f'<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;word-break:break-all;">轉址至: <a href="{loc}" target="_blank" style="color:var(--accent-cyan);">{loc}</a></div>'

    dk_badge = ""
    if i in internal_dk_matches:
        best_match = max(internal_dk_matches[i], key=lambda x: x["score"])
        b_score = int(best_match["score"]*100)
        b_sev = best_match["severity"]
        if b_sev == "high":
            dk_badge = f' <span class="severity-badge" style="font-size:0.7rem;padding:2px 8px;background:rgba(239,68,68,0.15);color:var(--accent-red);border:1px solid rgba(239,68,68,0.3);white-space:nowrap;display:inline-block;vertical-align:middle;margin-left:6px;" title="與外部 DigiKnow 文章標題幾乎相同 ({b_score}%)">🔴 DigiKnow 重複</span>'
        elif b_sev == "medium_high":
            dk_badge = f' <span class="severity-badge" style="font-size:0.7rem;padding:2px 8px;background:rgba(245,158,11,0.15);color:var(--accent-orange);border:1px solid rgba(245,158,11,0.3);white-space:nowrap;display:inline-block;vertical-align:middle;margin-left:6px;" title="與外部 DigiKnow 文章標題高度相似 ({b_score}%)">⚠️ DigiKnow 相似</span>'

    html += f"""          <tr data-cat="{art['category_id']}" data-status="{status}" data-text="{art['title'].lower()} {art.get('description','').lower()[:200]} {kws.lower()}">
            <td>{i+1}</td>
            <td><span class="cat-badge cat-{art['category_id']}">{art['category_name']}</span></td>
            <td class="td-title"><strong>{art['title']}</strong>{status_badge}{dk_badge}{redirect_info}</td>
            <td class="td-desc">{desc_short}...</td>
            <td class="td-keywords">
              <div class="kw-tags">
"""
    for k in art.get("extracted_keywords", [])[:5]:
        weight = art.get("keyword_weights", {}).get(k, 0.0)
        html += f'                <span class="kw-tag" title="TF-IDF: {weight:.4f}">{k}</span>\n'
    html += f"""              </div>
            </td>
            <td class="td-relations">{rel_html}</td>
            <td><a href="{art['url']}" target="_blank">開啟 ↗</a></td>
          </tr>
"""

html += """        </tbody>
      </table>
    </div>
  </div>
</div>

</div>

<script>
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  event.target.classList.add('active');
}

function filterArticles() {
  const text = document.getElementById('articleFilter').value.toLowerCase();
  const cat = document.getElementById('catFilter').value;
  const status = document.getElementById('statusFilter').value;
  const rows = document.querySelectorAll('#articlesTable tbody tr');
  rows.forEach(row => {
    const matchText = !text || row.dataset.text.includes(text);
    const matchCat = !cat || row.dataset.cat === cat;
    const matchStatus = !status || row.dataset.status === status;
    row.style.display = (matchText && matchCat && matchStatus) ? '' : 'none';
  });
}

function filterDkCards() {
  const sev = document.getElementById('dkSeverityFilter').value;
  const cards = document.querySelectorAll('#tab-digiknow .dk-card');
  cards.forEach(card => {
    if (!sev) {
      card.style.display = '';
      card.querySelectorAll('.cluster-article-item').forEach(item => item.style.display = '');
      return;
    }
    const cardSevs = card.dataset.severities.split(',');
    if (cardSevs.includes(sev)) {
      card.style.display = '';
      card.querySelectorAll('.cluster-article-item').forEach(item => {
        item.style.display = item.dataset.severity === sev ? '' : 'none';
      });
    } else {
      card.style.display = 'none';
    }
  });
}

function filterDupCards() {
  const sev = document.getElementById('dupSeverityFilter').value;
  const cards = document.querySelectorAll('#tab-duplicates .dup-card');
  cards.forEach(card => {
    if (!sev) {
      card.style.display = '';
      return;
    }
    if (card.classList.contains(sev)) {
      card.style.display = '';
    } else {
      card.style.display = 'none';
    }
  });
}
</script>

<div style="text-align:center;padding:50px 0 60px;color:var(--text-muted);font-size:0.8rem;border-top:1px solid var(--border);margin-top:50px;">
  鼎新數智 Blog 內容審計工具 · 分析共 {N} 篇文章 · 基於 jieba 分詞與自定義 TF-IDF 演算法
</div>

</body>
</html>
"""

# Write to file
with open(r"d:\p\blog-\report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("HTML report successfully generated at: d:\\p\\blog-\\report.html")

# Write enriched articles JSON back to keep data synced
with open(r"d:\p\blog-\articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print("articles.json updated with new TF-IDF keywords and related connections.")
