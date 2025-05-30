# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 1) 自動尋找第一個非空 sheet
    xls = pd.ExcelFile(file_path)
    df = None
    for sheet in xls.sheet_names:
        tmp = pd.read_excel(file_path, sheet_name=sheet)
        if not tmp.empty:
            df = tmp
            break
    if df is None or df.empty:
        raise KeyError("第4題：找不到任何有資料的工作表，請確認 Excel 內有資料")

    # 2) 擷取基金與股票報酬率
    cols = list(df.columns)
    # 優先欄位名，否則用前兩欄
    if "FundReturn" in cols:
        r_fund = df["FundReturn"].dropna().astype(float)
    else:
        r_fund = df[cols[0]].dropna().astype(float)
    if "StockReturn" in cols:
        r_stock = df["StockReturn"].dropna().astype(float)
    else:
        if len(cols) < 2:
            raise KeyError("第4題：找不到第二欄作為 StockReturn")
        r_stock = df[cols[1]].dropna().astype(float)

    if r_fund.empty or r_stock.empty:
        raise KeyError("第4題：基金或股票報酬率欄位沒有任何數值")

    # 3) 模擬參數
    years     = 30
    Nsim      = 10_000
    inflation = 1.022
    deposit   = 10_000
    weights   = [0.5, 0.7, 0.9]
    results   = {}

    # 4) Monte Carlo 模擬累積值
    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w * np.random.choice(r_stock,  Nsim) + \
                  (1-w) * np.random.choice(r_fund,  Nsim)
            if t == 0:
                sims[:,0] = dep * (1 + r)
            else:
                sims[:,t] = sims[:,t-1] * (1 + r) + dep
        results[w] = sims[:,-1]

    os.makedirs("static/results", exist_ok=True)

    # 5) 繪 30 年平均累積值
    fn1 = "q4_trend.png"
    plt.figure(figsize=(10,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    means = [results[w].mean() for w in weights]
    ax.bar([f"{int(w*100)}% Stocks" for w in weights], means,
           color=["#4C72B0","#55A868","#C44E52"])
    ax.set_xlabel("Portfolio Weight")
    ax.set_ylabel("Average Value After 30 Years")
    ax.set_title("30-Year Accumulated Average by Portfolio Mix")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn1))
    plt.close()

    # 6) 繪 CDF
    fn2 = "q4_cdf.png"
    plt.figure(figsize=(10,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    max_val = max(v.max() for v in results.values())
    x = np.linspace(0, max_val, 500)
    for w in weights:
        sorted_vals = np.sort(results[w])
        cdf = np.searchsorted(sorted_vals, x, side="right") / Nsim
        ax.plot(x, cdf, label=f"{int(w*100)}% Stocks")
    ax.set_xlabel("Final Value After 30 Years")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of 30-Year Final Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn2))
    plt.close()

    return {
        "plots": {
            "累積趨勢圖": fn1,
            "累積值CDF": fn2
        }
    }
