# solvers/question4.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_num(val):
    """從字串中擷取第一組數字，支援千分位、小數、百分比"""
    if pd.isna(val): 
        return None
    s = str(val).replace(',', '')
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m: 
        return None
    try:
        return float(m.group())
    except:
        return None

def solve(file_path):
    # 1) 用 xlrd 讀 .xls
    try:
        xls = pd.ExcelFile(file_path, engine='xlrd')
    except ImportError:
        raise ImportError(
            "第4題需要 xlrd 才能讀取 .xls，"
            "請在 requirements.txt 加入 xlrd>=2.0.1 並重啟部署。"
        )

    # 2) 自動找第一個有資料的 sheet
    df = None
    for sh in xls.sheet_names:
        tmp = xls.parse(sh)
        if not tmp.empty:
            df = tmp
            break
    if df is None or df.empty:
        raise KeyError("第4題：找不到任何含資料的工作表，請確認 Excel 內有資料")

    # 3) 從每一欄擷取數值，找出前兩個有真正數值的欄位
    numeric_cols = []
    numeric_data = {}
    for col in df.columns:
        nums = df[col].map(extract_num).dropna()
        if len(nums)>0:
            numeric_cols.append(col)
            numeric_data[col] = nums.astype(float)
        if len(numeric_cols) >= 2:
            break

    if len(numeric_cols) < 2:
        raise KeyError("第4題：無法找到兩個可擷取數值的欄位")

    # 前兩個欄位視為基金與股票回報
    r_fund  = numeric_data[numeric_cols[0]]
    r_stock = numeric_data[numeric_cols[1]]

    # 4) 模擬參數
    years     = 30
    Nsim      = 10_000
    inflation = 1.022
    deposit   = 10_000
    weights   = [0.5, 0.7, 0.9]
    results   = {}

    # 5) Monte Carlo 累積值模擬
    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            # 隨機抽樣已有數值 series
            r = (w * np.random.choice(r_stock, Nsim) + 
                 (1-w) * np.random.choice(r_fund, Nsim))
            if t == 0:
                sims[:,0] = dep * (1 + r)
            else:
                sims[:,t] = sims[:,t-1] * (1 + r) + dep
        results[w] = sims[:,-1]

    os.makedirs("static/results", exist_ok=True)

    # 6) 繪 30 年平均累積值
    fn1 = "q4_trend.png"
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
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
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
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
