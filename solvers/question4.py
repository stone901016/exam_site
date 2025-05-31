# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 1) 讀取 Excel，預期有 'StockReturn' 與 'BondReturn' 兩欄 (年化報酬率)，
    #    若在檔案中找不到，就嘗試用前兩個數值欄位。
    df = pd.read_excel(file_path)

    if "StockReturn" in df.columns and "BondReturn" in df.columns:
        # 假設原始為百分比，例如 5.95、-2.03 等，轉成小數
        r_stock = pd.to_numeric(df["StockReturn"], errors="coerce").dropna().values / 100.0
        r_fund  = pd.to_numeric(df["BondReturn"],    errors="coerce").dropna().values / 100.0
    else:
        # 如果沒有明確欄位名稱，就找前兩個 numeric 欄位
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            r_stock = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna().values / 100.0
            r_fund  = pd.to_numeric(df[numeric_cols[1]], errors="coerce").dropna().values / 100.0
        else:
            # 若連兩個數值欄位都找不到，就用「專案預設值」
            # 例如：基金年化平均 6%、標準差 12%；股票年化平均 8%、標準差 18%
            DEFAULT_MEAN_FUND  = 0.06
            DEFAULT_SD_FUND    = 0.12
            DEFAULT_MEAN_STOCK = 0.08
            DEFAULT_SD_STOCK   = 0.18
            # 隨機產生 1,000 個樣本（只要用來模擬即可）
            rng = np.random.default_rng()
            r_stock = rng.normal(DEFAULT_MEAN_STOCK, DEFAULT_SD_STOCK, size=1000)
            r_fund  = rng.normal(DEFAULT_MEAN_FUND,  DEFAULT_SD_FUND,  size=1000)

    # 2) 設定模擬參數
    start_age = 30
    end_age   = 60
    years     = end_age - start_age  # 30 年
    N         = 10000               # 模擬次數
    inflation = 1.022
    deposit   = 10000               # 每年投入金額
    
    weights = [0.5, 0.7, 0.9]        # 股票權重：50%、70%、90%

    # 3) 針對每個權重，產生 shape=(N, years) 的模擬累積價值矩陣
    sims_dict = {}
    for w in weights:
        sims = np.zeros((N, years))
        for t in range(years):
            # 當年投入本金(含通膨)
            dep = deposit * (inflation ** t)
            # 隨機抽樣：股票報酬與基金報酬
            r_choice_stock = np.random.choice(r_stock, size=N, replace=True)
            r_choice_fund  = np.random.choice(r_fund,  size=N, replace=True)
            r = w * r_choice_stock + (1.0 - w) * r_choice_fund
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t-1] * (1 + r) + dep
        sims_dict[w] = sims

    # 4) 確保結果資料夾存在
    os.makedirs("static/results", exist_ok=True)

    plots_trend = {}
    plots_cdf   = {}

    # 要畫圖的橫軸：從 30 到 59（共 30 個整年）
    ages = np.arange(start_age, end_age)

    # Q1：Trend——同一張圖畫出 N 條路徑，並突顯平均值
    for w in weights:
        sims = sims_dict[w]         # shape=(N, years)
        mean_path = sims.mean(axis=0)  # 各年的平均累積值

        fn_trend = f"q4_trend_{int(w*100)}.png"
        plt.figure(figsize=(12, 6))  # 稍微放大一些
        # 4.1) 繪製所有模擬路徑
        for i in range(N):
            plt.plot(ages, sims[i, :],
                     color="blue", alpha=0.01, linewidth=0.5)
        # 4.2) 繪製平均路徑
        plt.plot(ages, mean_path,
                 color="red", linewidth=2.5, label="平均累積價值")
        plt.title(f"Trend (w={int(w*100)}% 股票)", fontsize=16)
        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Accumulated Value", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_trend))
        plt.close()

        plots_trend[w] = fn_trend

    # Q2：CDF——分別畫最終累積值的累積機率分布
    for w in weights:
        sims = sims_dict[w]
        final_vals = sims[:, -1]
        sorted_vals = np.sort(final_vals)
        cdf = np.arange(1, N+1) / N

        fn_cdf = f"q4_cdf_{int(w*100)}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_vals, cdf,
                 color="green", linewidth=2)
        plt.title(f"CDF (w={int(w*100)}% 股票)", fontsize=16)
        plt.xlabel("Final Accumulated Value", fontsize=14)
        plt.ylabel("Cumulative Probability", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_cdf))
        plt.close()

        plots_cdf[w] = fn_cdf

    # 回傳兩組對應各權重的檔名
    return {
        "plots_trend": plots_trend,
        "plots_cdf":   plots_cdf
    }
