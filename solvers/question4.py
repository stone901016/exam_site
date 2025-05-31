# solvers/question4.py
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """
    從任意格式字串擷取第一組浮點數 (支援千分位、小數)。
    找不到就回傳 None。
    """
    if pd.isna(val):
        return None
    s = str(val).strip().replace(',', '')
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def solve(file_path):
    # 1) 嘗試用 xlrd 讀取 .xls
    try:
        xls = pd.ExcelFile(file_path, engine='xlrd')
    except ImportError:
        raise ImportError(
            "第4題需要 xlrd 才能讀取 .xls，"
            "請在 requirements.txt 加入 `xlrd>=2.0.1` 並重新部署。"
        )

    # 2) 自動選擇第一個非空工作表
    df = None
    for sh in xls.sheet_names:
        tmp = xls.parse(sh, header=None)
        if not tmp.empty:
            df = tmp.copy()
            break
    if df is None or df.empty:
        raise KeyError("第4題：找不到任何含資料的工作表，請確認 Excel 內有資料。")

    # 3) 尋找「Total Returns」欄 (數值通常介於 1~10)，轉成年度報酬率
    data_rows = []
    for r in range(df.shape[0]):
        v = extract_num(df.iat[r, 0])
        if v is not None and 1900 < int(v) < 2100:
            data_rows.append(r)
    if data_rows:
        data_df = df.loc[data_rows].reset_index(drop=True)
    else:
        data_df = df.copy()

    num_data = {}
    for c in range(data_df.shape[1]):
        col_vals = data_df.iloc[:, c].map(extract_num).dropna().astype(float)
        if col_vals.empty:
            continue
        med = np.median(col_vals.values)
        if 1.0 < med < 10.0:
            num_data[c] = col_vals.values

    if len(num_data) >= 2:
        sorted_cols = sorted(num_data.keys())
        stock_tot   = num_data[sorted_cols[0]]
        bond_tot    = num_data[sorted_cols[1]]
        r_stock     = stock_tot - 1.0
        r_bond      = bond_tot - 1.0
        mean_stock  = float(np.mean(r_stock))
        sd_stock    = float(np.std(r_stock, ddof=0))
        mean_fund   = float(np.mean(r_bond))
        sd_fund     = float(np.std(r_bond, ddof=0))
    else:
        # 預設：1926–2004 歷史資料計算值
        mean_fund   = 0.05856962025316456  # 債券年化平均 5.85696%
        sd_fund     = 0.07592167742079621  # 債券年化標準差 7.59217%
        mean_stock  = 0.14956962025316461  # 股票年化平均 14.95696%
        sd_stock    = 0.25180571843499633  # 股票年化標準差 25.18057%

    # 4) 模擬參數
    start_age   = 30
    years       = 60 - start_age    # 30 年
    Nsim        = 10_000
    deposit     = 10_000
    inflation   = 1.022
    weights     = [0.5, 0.7, 0.9]

    # 5) 先生成各年回報隨機矩陣 (基金 vs 股票)
    fund_r = norm.rvs(loc=mean_fund,  scale=sd_fund,  size=(Nsim, years))
    stc_r  = norm.rvs(loc=mean_stock, scale=sd_stock, size=(Nsim, years))

    # 6) 計算每個 w 的累積路徑與最終值
    paths = {}
    final = {}
    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w * stc_r[:, t] + (1 - w) * fund_r[:, t]
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t - 1] * (1 + r) + dep
        paths[w] = sims
        final[w] = sims[:, -1]

    os.makedirs("static/results", exist_ok=True)

    # 7) Q1: 對每個權重分別畫 Trend 圖、存檔
    trend_files = {}
    ages = np.arange(start_age, start_age + years)
    for w in weights:
        mean_path = paths[w].mean(axis=0)
        fn = f"Q1_Trend_{int(w*100)}.png"
        plt.figure(figsize=(8,5))
        ax = plt.gca()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.plot(ages, mean_path, color="#4C72B0", linewidth=2)
        ax.set_xlabel("Age")
        ax.set_ylabel("Average Accumulated Value")
        ax.set_title(f"Trend (w={int(w*100)}% Stocks)")
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()
        trend_files[w] = fn

    # 8) Q2: 對每個權重分別畫 CDF 圖、存檔
    cdf_files = {}
    for w in weights:
        arr = np.sort(final[w])
        cdf = np.arange(1, Nsim + 1) / Nsim
        fn = f"Q2_CDF_{int(w*100)}.png"
        plt.figure(figsize=(8,5))
        ax = plt.gca()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.plot(arr, cdf, color="#55A868", linewidth=2)
        ax.set_xlabel("Final Accumulated Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF (w={int(w*100)}% Stocks)")
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()
        cdf_files[w] = fn

    # 9) 回傳：兩組檔名都放進 result 裡
    return {
        "plots_trend": trend_files,   # {0.5: "Q1_Trend_50.png", 0.7: "...", 0.9: "..."}
        "plots_cdf":   cdf_files      # {0.5: "Q2_CDF_50.png", 0.7: "...", 0.9: "..."}
    }
