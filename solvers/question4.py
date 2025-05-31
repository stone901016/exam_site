# solvers/question4.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """
    從任意格式字串擷取第一組浮點數 (支援千分位、小數)，找不到就回 None。
    """
    if pd.isna(val):
        return None
    s = str(val).strip().replace(',', '')
    # 找出第一個可能是數字的片段（可帶小數）
    import re
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def solve(file_path):
    """
    第 4 題：模擬投資人從 30 歲到 60 歲 (共 30 年)，每年投入 10,000 元 (考慮 2.2% 通膨)，
    投資標的是基金與股票（皆假設年化報酬為常態分布），模擬次數 10,000 次。
    Q1: 畫出 50%、70%、90% 股票權重下的「年齡 vs. 平均累積價值」趨勢圖 (共 30 點)。
    Q2: 畫出三種權重下的「最終累積價值」CDF 圖 (x 軸：最終累積值，y 軸：累積機率)。
    """
    # 嘗試讀 Excel (支援 .xls/.xlsx)
    try:
        df_all = pd.read_excel(file_path, header=None)
    except Exception as e:
        raise KeyError("第4題：無法讀取 Excel，請確認檔案格式。")

    # 1) 必要時嘗試「找出年化平均與標準差」；若失敗則使用預設值
    #    預設值：1926–2004 年歷史資料計算出的年化 Mean/StdDev
    #    Fund (債券)：Mean≈5.85696%，StdDev≈7.59217%
    #    Stock (股票)：Mean≈14.95696%，StdDev≈25.18057%
    mean_fund   = None
    sd_fund     = None
    mean_stock  = None
    sd_stock    = None

    # 嘗試從 Excel 抓「Total Returns」欄 (若有) 以計算年化報酬
    # 假設至少有兩欄符合「1.0 < 中間值 < 10.0」條件為 yearly total returns
    try:
        # 只拿含年份的那幾列：若第一欄是 1900~2100 之間的數字，就視為年份(row)；保留這些列
        rows_with_year = []
        for r in range(df_all.shape[0]):
            v = extract_num(df_all.iat[r, 0])
            if v is not None and 1900 < int(v) < 2100:
                rows_with_year.append(r)
        if rows_with_year:
            df = df_all.loc[rows_with_year].reset_index(drop=True)
        else:
            df = df_all.copy()

        # 檢查哪兩欄「中位數介於 1.0~10.0」(Total Return)
        numeric_cols = {}
        for c in range(df.shape[1]):
            arr = df.iloc[:, c].map(extract_num).dropna().astype(float)
            if arr.size == 0:
                continue
            med = np.median(arr)
            if 1.0 < med < 10.0:
                numeric_cols[c] = arr.values

        if len(numeric_cols) >= 2:
            # 取排序後最前兩欄
            idxs = sorted(numeric_cols.keys())[:2]
            stock_tot = np.array(numeric_cols[idxs[0]])  # e.g. 1.0595, 1.2980, ...
            bond_tot  = np.array(numeric_cols[idxs[1]])
            # 年化報酬率 = TotalReturn − 1
            r_stock = stock_tot - 1.0
            r_bond  = bond_tot - 1.0
            mean_stock = float(np.mean(r_stock))
            sd_stock   = float(np.std(r_stock, ddof=0))
            mean_fund  = float(np.mean(r_bond))
            sd_fund    = float(np.std(r_bond, ddof=0))
        else:
            # 無法自 Excel 算出，就保留 None，稍後用預設值
            pass
    except Exception:
        # 任何失敗都讓變數保持 None
        pass

    # 如果抓不到，就帶預設值
    if mean_fund is None or sd_fund is None or mean_stock is None or sd_stock is None:
        mean_fund   = 0.05856962025316456   # 年化平均 5.85696%
        sd_fund     = 0.07592167742079621   # 年化標準差 7.59217%
        mean_stock  = 0.14956962025316461   # 年化平均 14.95696%
        sd_stock    = 0.25180571843499633   # 年化標準差 25.18057%

    # 2) 模擬參數設定
    start_age   = 30
    end_age     = 60
    years       = end_age - start_age   # 30 年
    Nsim        = 10_000                 # 模擬次數
    deposit     = 10_000                 # 每年年初投入
    inflation   = 1.022                  # 通貨膨脹 2.2%
    weights     = [0.5, 0.7, 0.9]         # 50%、70%、90% Stocks

    # 3) 在所有模擬跑之前先把每年「基金 vs 股票」的常態隨機矩陣生成好
    #    r_fund.shape = (Nsim, years)，r_stock 同
    rng = np.random.default_rng()  # 使用新一代 RandomGenerator
    r_fund  = rng.normal(loc=mean_fund,  scale=sd_fund,  size=(Nsim, years))
    r_stock = rng.normal(loc=mean_stock, scale=sd_stock, size=(Nsim, years))

    # 4) 計算每個權重 w 的「每年資產路徑」與「最終累積值」
    #    存到 paths, final_value 這兩個 dict
    paths = {}   # paths[w] 為 shape=(Nsim, years) 二維陣列
    final = {}   # final[w] 為 shape=(Nsim,) 最後一年累積值

    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)             # 考慮通膨後的投入
            r   = w * r_stock[:, t] + (1 - w) * r_fund[:, t]  # 當年投資組合報酬
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t - 1] * (1 + r) + dep
        paths[w] = sims
        final[w] = sims[:, -1]   # 第 30 年的最終累積值

    # 5) 繪圖：先建立 results 資料夾
    os.makedirs("static/results", exist_ok=True)

    # 存檔字典
    trend_files = {}
    cdf_files   = {}

    ages = np.arange(start_age, end_age)

    # 6) Q1: 每個權重都畫出「Age vs. 平均累積值」(Trend)
    for w in weights:
        sims = paths[w]                  # shape=(Nsim, years)
        mean_path = sims.mean(axis=0)    # 30 個點：每年所有模擬的平均
        fn = f"Q1_Trend_{int(w*100)}.png"

        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.plot(ages, mean_path, color="#4C72B0", linewidth=2)
        ax.set_title(f"Trend (w={int(w*100)}% Stocks)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Average Accumulated Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()

        trend_files[w] = fn

    # 7) Q2: 每個權重都畫出「Final Value CDF」
    for w in weights:
        arr = np.sort(final[w])                     # 排序後的最終值
        cdf = np.arange(1, Nsim + 1) / Nsim
        fn = f"Q2_CDF_{int(w*100)}.png"

        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.plot(arr, cdf, color="#55A868", linewidth=2)
        ax.set_title(f"CDF (w={int(w*100)}% Stocks)")
        ax.set_xlabel("Final Accumulated Value")
        ax.set_ylabel("Cumulative Probability")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()

        cdf_files[w] = fn

    # 8) 回傳結果：兩個字典，讓 answer.html 直接取用
    return {
        "plots_trend": trend_files,   # {0.5: "Q1_Trend_50.png", 0.7: "Q1_Trend_70.png", 0.9: "Q1_Trend_90.png"}
        "plots_cdf":   cdf_files      # {0.5: "Q2_CDF_50.png",   0.7: "Q2_CDF_70.png",   0.9: "Q2_CDF_90.png"}
    }
