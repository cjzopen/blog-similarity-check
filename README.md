# Blog Content Audit Tool (鼎新數智 Blog 內容審計工具)

本工具用於自動化爬取、分析與審計鼎新數智 Blog 網站的文章內容。具備 NLP 關鍵字分析 (jieba+TF-IDF)、全文相似度重複排查、Topic Cluster 主題集群關聯性分析以及 HTTP 301 轉址狀態檢測功能。

---

## 🛠️ 環境需求

1. **Python 3.x**
2. **依賴套件安裝**：
   ```bash
   pip install requests beautifulsoup4 jieba
   ```

---

## 🚀 報表生成指令 (一鍵執行)

為了簡化操作，您可以直接在專案根目錄下執行以下指令，系統會自動依序完成所有的爬取與分析步驟（包含列表爬取、強制內文更新、轉址檢查、以及報告生成）：

```bash
python run_all.py
```

若您想要拆解步驟獨立執行，請參考以下 Step-by-Step 說明：

### 1. 爬取文章清單 (Scrape Article Lists)
爬取 Blog 網站中 11 個分類（包含全球動態、趨勢議題、ERP 知識庫、企業AI專欄等）的所有分頁，抓取文章標題、摘要及連結並儲存至 `articles_raw.json`：
```bash
python crawl.py
```

### 2. 爬取全文內文與資料合併 (Scrape Full Text & Merge Data)
讀取 `articles_raw.json`，將新文章與已有的 `articles.json` 資料庫進行合併（保留已爬取的內文）。接著針對新加入的文章，抓取其網頁內文（支援舊版/新版網頁排版）並豐富化資料庫：
```bash
python enrich_content.py
```
*提示：該腳本會自動跳過已有內文的文章。若網站文章有大幅更新或被刪除，且您希望**強制重新拉取所有內文**，請加上 `--force` 參數：*
```bash
python enrich_content.py --force
```

### 3. 檢查 HTTP 狀態碼與 301 轉址 (Check HTTP Status & 301 Redirects)
使用 `requests.head` 對資料庫中的所有網址發送非自動跟隨的請求，檢查網址是否被重定向 (301/302)，並記錄轉址後的目標網址：
```bash
python check_redirects.py
```

### 4. 計算關鍵字、主題集群與生成 HTML 報告 (Analyze & Generate HTML Report)
執行 NLP 斷詞、計算 TF-IDF 權重、以餘弦相似度對全文進行主題建模群聚 (Topic Clusters)、排查相似度大於 65% 的「中高度風險重複內容」，最後生成視覺化報告 `report.html`：
```bash
python generate_report.py
```

---

## 📊 產出結果

執行完上述四個步驟後，您會獲得以下檔案：
- **[report.html]**: 視覺化審計報告 Dashboard，可直接在瀏覽器雙擊打開。
- **[articles.json]**: 已豐富化（包含標題、摘要、內文、關鍵字、狀態碼、轉址 URL、關聯推薦）的 JSON 資料庫。
