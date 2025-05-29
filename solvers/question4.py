# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    df = pd.read_excel(file_path)
    # 假設欄位: 'FundReturn', 'StockReturn' (年化)
    fund = df.get("FundReturn", df.columns[0])
    stock= df.get("StockReturn", df.columns[1])
    r_fund = df[fund]
    r_stock= df[stock]

    years = 30
    N = 10000
    inflation = 1.022
    deposit = 10000

    weights = [0.5,0.7,0.9]
    results = {}

    for w in weights:
        sims = np.zeros((N, years))
        for t in range(years):
            # 每年投入考慮通膨
            dep = deposit * (inflation**t)
            # 當年投資報酬
            r = w * np.random.choice(r_stock, N) + (1-w) * np.random.choice(r_fund, N)
            if t == 0:
                sims[:,0] = dep * (1+r)
            else:
                sims[:,t] = sims[:,t-1] * (1+r) + dep
        # 取最終值
        final = sims[:,-1]
        results[w] = final

    # 繪 Q1：平均走勢圖 (用每年平均)
    fn1 = "q4_trend.png"
    plt.figure()
    for w in weights:
        mean_path = np.mean(results[w])
        plt.bar(str(w), mean_path)
    plt.title("30 年累積平均值")
    plt.savefig(os.path.join("static","results",fn1))
    plt.close()

    # Q2：CDF 圖
    fn2 = "q4_cdf.png"
    plt.figure()
    for w in weights:
        sortedx = np.sort(results[w])
        cdf = np.arange(1, N+1)/N
        plt.plot(sortedx, cdf, label=f"{int(w*100)}% 股票")
    plt.title("累積機率曲線")
    plt.legend()
    plt.savefig(os.path.join("static","results",fn2))
    plt.close()

    return {
        "plots": {"trend": fn1, "cdf": fn2}
    }
