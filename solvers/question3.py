# solvers/question3.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(_):
    # Assumptions (from Profit.xls Crystal Ball)
    S = 10.0
    V = 0.6
    P_mean_file = 50.0
    P_sd_file   = 10.0

    N = 100_000

    # 1) SD=10，P mean=45,55 → Profit 分布 & 靈敏度
    profits = {}
    for m in (45, 55):
        p_sim = norm.rvs(loc=m, scale=P_sd_file, size=N)
        profits[m] = S * V * p_sim

    # 繪直方圖
    os.makedirs("static/results", exist_ok=True)
    fn_hist = "q3_dist.png"
    plt.figure(figsize=(16,10))
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.legend()
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_hist))
    plt.close()

    # 靈敏度 (Std)
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # 2) mean=50，SD=8,10,12 → P(Profit>100) 機率
    prob_gt100 = {}
    for sd in (8, 10, 12):
        p_sim = norm.rvs(loc=P_mean_file, scale=sd, size=N)
        prof = S * V * p_sim
        prob_gt100[sd] = float((prof > 100).mean())

    # 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(16,10))
    xs, ys = zip(*sorted(prob_gt100.items()))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("P SD")
    plt.ylabel("P(Profit > 100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    return {
        "靈敏度（收益標準差）": sensitivity,
        "P(Profit>100) 機率": prob_gt100,
        "plots": {
            "分布比較圖": fn_hist,
            "機率變化圖": fn_prob
        }
    }
