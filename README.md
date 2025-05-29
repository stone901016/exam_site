# 通用期末考題自動解答網站

本專案針對「期末考模擬 2.ppt」中的 10 題，使用 Flask + Python 建置。  
只要把所有輸入檔（例如 B保險.xls、Profit.xls…）放入 `data/` 資料夾，啟動後就能選題、產生答案與圖表。

## 本機執行

1. 建立虛擬環境並安裝套件：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. 把 PPT 指定的所有輸入檔放到 `data/` 目錄。
3. 啟動：
   ```bash
   python app.py
   ```
4. 瀏覽 `http://localhost:5000`

## 部署至 Railway

1. 到 GitHub 建立新 repo，將本專案 push 上去。
2. 到 Railway，新增專案→「Deploy from GitHub」→選擇本 repo→Deploy。
3. 在 Railway 的 Settings → Environment Variables 裡，新增：
   ```
   SECRET_KEY=任你設定的一串隨機字串
   ```
4. Deploy 完成後，Railway 會自動安裝並啟動。
