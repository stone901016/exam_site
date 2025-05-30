# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 1) 嘗試用 xlrd 讀 .xls
    try:
        xls = pd.ExcelFile(file_path, engine='xlrd')
    except ImportError:
        raise ImportError(
            "第4題需要 xlrd 才能讀取 .xls，"
            "請在 requirements.txt 加入 xlrd>=2.0.1 並重新部署。"
        )

    # 2) 找到第一個 non‐empty sheet
    df = None
    for sheet in xls.sheet_names:
        tmp = xls.parse(sheet)
        if not tmp.empty:
            df = tmp
            break
    if df is None or df.empty:
        raise KeyError("第4題：找不到任何有資料的工作表，請確認 Excel 內有資料")

    # 3) 只保留 numeric 欄位，排除文字欄位
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise KeyError("第4題：數值欄位不足，需至少兩個 numeric 欄位作為 FundReturn、StockReturn")
    # 取前兩個 numeric 欄位作為基金與股票報酬率
    r_fund  = df[num_cols[0]].dropna().astype(float)
    r_stock = df[num_cols[1]].dropna().astype(float)
    if r_fund.empty or r_stock.empty:
        raise KeyError("第4題：基金或股票報酬率欄位沒有任何數值")

    # 4) 模擬參數
    years     = 30
    Nsim      = 10_000
    inflation = 1.022
    deposit   = 10_000
    weights   = [0.5, 0.7, 0.9]
    results   = {}

    # 5) Monte Carlo 模擬累積值
    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w * np.random.choice(r_stock, Nsim) + (1-w) * np.random.choice(r_fund, Nsim)
            if t == 0:
                sims[:,0] = dep * (1 + r)
            else:
                sims[:,t] = sims[:,t-1] * (1 + r) + dep
        results[w] = sims[:,-1]

    os.makedirs("static/results", exist_ok=True)

    # 6) 繪 30 年平均累積值
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

    # 7) 繪 CDF
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
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of 30-Year Final Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn2))
    plt.close()

    # 8) 回傳
    return {
        "plots": {
            "累積趨勢圖": fn1,
            "累積值CDF": fn2
        }
    }
