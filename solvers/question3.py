# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    df = pd.read_excel(file_path)
    # 自動偵測欄位
    S = df.get("S", df.get("s")).mean()
    V = df.get("V", df.get("v")).mean()
    # 取價格欄位，可能是 'P' 或 '價格'
    for c in df.columns:
        if c in ["P","p","價格","Price"]:
            Pcol = c
            break
    P_sd = df[Pcol].std()

    N = 100000
    # 1) P mean = 45, 55
    profits1 = {}
    for m in [45,55]:
        p_sim = norm.rvs(loc=m, scale=P_sd, size=N)
        prof = S * V * p_sim
        profits1[m] = prof

    # 繪圖：Profit 分布比較
    fn1 = "q3_dist.png"
    plt.figure()
    for m, prof in profits1.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static","results",fn1))
    plt.close()

    # 敏感度 (Std of profit)
    sens = {m: np.std(profits1[m]) for m in profits1}

    # 2) P mean 固定，SD = 8,10,12
    prob_gt100 = {}
    P_mean0 = df[Pcol].mean()
    for sd in [8,10,12]:
        p_sim = norm.rvs(loc=P_mean0, scale=sd, size=N)
        prof = S * V * p_sim
        prob_gt100[sd] = np.mean(prof > 100)

    # 繪圖：P(Profit>100) vs SD
    fn2 = "q3_p_gt100.png"
    plt.figure()
    x = list(prob_gt100.keys())
    y = list(prob_gt100.values())
    plt.plot(x, y, marker='o')
    plt.title("P(Profit>100) vs P SD")
    plt.xlabel("P SD")
    plt.ylabel("Probability")
    plt.savefig(os.path.join("static","results",fn2))
    plt.close()

    return {
        "profit_std_by_mean": sens,
        "prob_profit_gt_100": prob_gt100,
        "plots": {"dist": fn1, "p_gt100": fn2}
    }
