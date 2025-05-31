# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 1) 讀取 Excel 中的「股票報酬率」與「基金報酬率」欄位（年化，% 格式）
    df = pd.read_excel(file_path)

    # 優先使用欄位名稱；若不存在，就嘗試取前兩個 numeric 欄位；再不行就用預設常態分布參數
    if "StockReturn" in df.columns and "BondReturn" in df.columns:
        r_stock = pd.to_numeric(df["StockReturn"], errors="coerce").dropna().values / 100.0
        r_fund  = pd.to_numeric(df["BondReturn"],    errors="coerce").dropna().values / 100.0
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            r_stock = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna().values / 100.0
            r_fund  = pd.to_numeric(df[numeric_cols[1]], errors="coerce").dropna().values / 100.0
        else:
            # 如果還是不夠，就使用專案中預設的常態分布參數
            DEFAULT_MEAN_FUND  = 0.06   # 年化 6%
            DEFAULT_SD_FUND    = 0.12   # 年化標準差 12%
            DEFAULT_MEAN_STOCK = 0.08   # 年化 8%
            DEFAULT_SD_STOCK   = 0.18   # 年化標準差 18%
            rng = np.random.default_rng()
            r_stock = rng.normal(DEFAULT_MEAN_STOCK, DEFAULT_SD_STOCK, size=1000)
            r_fund  = rng.normal(DEFAULT_MEAN_FUND,  DEFAULT_SD_FUND,  size=1000)

    # 2) 模擬參數
    start_age = 30
    end_age   = 60
    years     = end_age - start_age    # 共 30 年
    N         = 10000                 # 模擬 10,000 次
    inflation = 1.022
    deposit   = 10000                 # 每年投資 10,000 元

    weights = [0.5, 0.7, 0.9]          # 股票權重：50%、70%、90%

    # 3) 針對每個權重 w，模擬 shape=(N, years) 的累積價值矩陣
    sims_dict = {}
    for w in weights:
        sims = np.zeros((N, years))
        for t in range(years):
            # 當年投入考慮通膨
            dep = deposit * (inflation ** t)
            # 隨機從歷史/產生的報酬率中抽樣
            r_choice_stock = np.random.choice(r_stock, size=N, replace=True)
            r_choice_fund  = np.random.choice(r_fund,  size=N, replace=True)
            r = w * r_choice_stock + (1.0 - w) * r_choice_fund
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t-1] * (1 + r) + dep
        sims_dict[w] = sims

    # 4) 確保輸出資料夾存在
    os.makedirs("static/results", exist_ok=True)

    plots_trend = {}
    plots_cdf   = {}

    ages = np.arange(start_age, end_age)  # X 軸：「Age」從 30 到 59

    # Q1: Trend → 同張圖畫出 10,000 條路徑 + 平均值 (粗紅線)
    for w in weights:
        sims = sims_dict[w]             # shape=(N, years)
        mean_path = sims.mean(axis=0)   # 各年份的平均累積值

        fn_trend = f"q4_trend_{int(w*100)}.png"
        plt.figure(figsize=(12, 6))

        # 4.1) 所有模擬路徑 (淡藍線)
        for i in range(N):
            plt.plot(ages, sims[i, :],
                     color="blue", alpha=0.01, linewidth=0.5)
        # 4.2) 平均累積值路徑 (粗紅線)
        plt.plot(ages, mean_path,
                 color="red", linewidth=2.5, label="Average Path")
        plt.title(f"Trend (w={int(w*100)}% stocks)", fontsize=16)
        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Accumulated Value", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_trend))
        plt.close()

        plots_trend[w] = fn_trend

    # Q2: CDF → 每個權重取最終累積值畫累積機率曲線
    for w in weights:
        sims = sims_dict[w]
        final_vals = sims[:, -1]
        sorted_vals = np.sort(final_vals)
        cdf = np.arange(1, N+1) / N

        fn_cdf = f"q4_cdf_{int(w*100)}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_vals, cdf,
                 color="green", linewidth=2)
        plt.title(f"CDF (w={int(w*100)}% stocks)", fontsize=16)
        plt.xlabel("Final Accumulated Value", fontsize=14)
        plt.ylabel("Cumulative Probability", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_cdf))
        plt.close()

        plots_cdf[w] = fn_cdf

    return {
        "plots_trend": plots_trend,
        "plots_cdf":   plots_cdf
    }
