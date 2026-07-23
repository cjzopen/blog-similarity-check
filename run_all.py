import subprocess
import sys
import os

def run_step(command, step_name):
    print(f"\n{'='*50}")
    print(f"🚀 正在執行步驟：{step_name}")
    print(f"指令：{command}")
    print(f"{'='*50}")
    
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True
        )
        print(f"✅ {step_name} 執行成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} 執行失敗。")
        print(f"錯誤碼：{e.returncode}")
        sys.exit(1)

def main():
    print("🌟 開始執行 Blog 內容排查自動化流程 🌟")
    
    # 步驟 1: 爬取文章清單
    run_step("python crawl.py", "1. 爬取文章清單 (Scrape Article Lists)")
    
    # 步驟 2: 爬取全文內文與資料合併 (使用 --force 確保資料最新)
    run_step("python enrich_content.py --force", "2. 爬取全文內文與資料合併 (Scrape Full Text)")
    
    # 步驟 3: 檢查 HTTP 狀態碼與 301 轉址
    run_step("python check_redirects.py", "3. 檢查 HTTP 狀態碼與轉址 (Check Redirects)")
    
    # 步驟 4: 計算關鍵字、主題集群與生成 HTML 報告
    run_step("python generate_report.py", "4. 生成 HTML 視覺化報告 (Generate Report)")
    
    print(f"\n{'='*50}")
    print("🎉 所有步驟已完成！")
    print(f"請在瀏覽器中開啟 report.html 查看審計報告。")
    print(f"檔案位置：{os.path.abspath('report.html')}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
