# solvers/question10.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # 讀 Excel，假設 sheet2 為光寶，或用篩 symbol
    xl = pd.ExcelFile(file_path)
    # 如果有 sheet 名稱 '光寶'
    if "光寶" in xl.sheet_names:
        df = xl.parse("光寶")
    else:
        df = xl.parse(xl.sheet_names[1])  # 第二張
    # 假設欄位 'Date','Close'
    price = df["Close"].astype(float)
    # 計算日報酬
    ret = np.log(price / price.shift(1)).dropna()
    mu = ret.mean() * 252
    sigma = ret.std() * np.sqrt(252)

    # 模擬
    S0 = price.iloc[-1]
    def gbm_sim(days):
        dt = days/252
        return S0 * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn(100000))

    p1 = gbm_sim(12)  # 12日後
    p2 = gbm_sim(25)  # 25日後

    # 圖
    fn = "q10_gbm.png"
    plt.figure()
    plt.hist(p1, bins=50, alpha=0.6, label="12日")
    plt.hist(p2, bins=50, alpha=0.6, label="25日")
    plt.legend()
    plt.title("GBM 模擬股價")
    plt.savefig(os.path.join("static","results",fn))
    plt.close()

    return {
        "mu": mu,
        "sigma": sigma,
        "price_12d_mean": p1.mean(),
        "price_25d_mean": p2.mean(),
        "plots": {"gbm": fn}
    }
