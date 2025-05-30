# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 1) 讀取 Excel
    df = pd.read_excel(file_path)
    # 空表檢查
    if df.empty:
        raise KeyError("第4題：讀取到的資料為空，請確認 Excel 內有資料")
    
    # 2) 擷取「基金報酬率」與「股票報酬率」
    # 優先使用明確欄位名，否則退回到前兩個欄位
    if "FundReturn" in df.columns:
        r_fund = df["FundReturn"].dropna().astype(float)
    else:
        r_fund = df[df.columns[0]].dropna().astype(float)
    if "StockReturn" in df.columns:
        r_stock = df["StockReturn"].dropna().astype(float)
    else:
        if len(df.columns) < 2:
            raise KeyError("第4題：無法找到第二欄作為 StockReturn")
        r_stock = df[df.columns[1]].dropna().astype(float)
    
    # 再次檢查是否都有數值
    if r_fund.empty or r_stock.empty:
        raise KeyError("第4題：基金或股票報酬率欄位沒有任何數值")
    
    # 3) 模擬參數
    years    = 30
    Nsim     = 10_000
    inflation= 1.022
    deposit  = 10_000
    weights  = [0.5, 0.7, 0.9]
    
    # 4) 模擬儲存
    results = {}
    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r = (w * np.random.choice(r_stock,  Nsim) +
                 (1-w) * np.random.choice(r_fund, Nsim))
            if t == 0:
                sims[:,0] = dep * (1 + r)
            else:
                sims[:,t] = sims[:,t-1] * (1 + r) + dep
        results[w] = sims[:,-1]  # 最終年值
    
    os.makedirs("static/results", exist_ok=True)
    
    # 5) 繪圖 Q1：30年平均累積值 Bar Chart
    fn1 = "q4_trend.png"
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
    means = [results[w].mean() for w in weights]
    ax.bar([f"{int(w*100)}% Stocks" for w in weights], means,
           color=["#4C72B0","#55A868","#C44E52"])
    ax.set_xlabel("Portfolio Weight")
    ax.set_ylabel("Average 30-Year Value")
    ax.set_title("30-Year Accumulated Average by Portfolio Mix")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn1))
    plt.close()
    
    # 6) 繪圖 Q2：最終值 CDF
    fn2 = "q4_cdf.png"
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
    x = np.linspace(0, max(v.max() for v in results.values()), 500)
    for w in weights:
        sorted_vals = np.sort(results[w])
        cdf = np.searchsorted(sorted_vals, x, side="right") / Nsim
        ax.plot(x, cdf, label=f"{int(w*100)}% Stocks")
    ax.set_xlabel("Final Value after 30 Years")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of 30-Year Final Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn2))
    plt.close()
    
    # 7) 回傳
    return {
        "plots": {
            "累積趨勢圖": fn1,
            "累積值CDF": fn2
        }
    }
