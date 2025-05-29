# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    """
    Profit.xls 解題：
    1) 嘗試 header=0,1,2 自動偵測正確的欄位
    2) 取前三個 numeric 欄當作 S, V, P
    3) P mean=45/55 時繪製 Profit 分布 & 靈敏度(Std)
    4) P SD=8,10,12 時計算 P(Profit>100) 並繪圖
    """
    # 嘗試不同 header
    df = None
    for h in [0,1,2]:
        tmp = pd.read_excel(file_path, header=h)
        nums = tmp.select_dtypes(include=[np.number]).columns.tolist()
        if len(nums) >= 3:
            df = tmp
            num_cols = nums
            break
    if df is None:
        raise KeyError(f"Profit.xls 數值欄位不足：僅找到 {nums}")

    # S, V, P 欄位
    S_col, V_col, P_col = num_cols[:3]
    S = df[S_col].mean()
    V = df[V_col].mean()
    P_vals = df[P_col]
    P_mean = P_vals.mean()
    P_sd   = P_vals.std()

    N = 100000
    # (1) mean=45,55
    profits = {}
    for m in [45, 55]:
        p_sim = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S * V * p_sim

    # 繪分布圖
    fn_dist = "q3_dist.png"
    plt.figure()
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static","results",fn_dist))
    plt.close()

    # 靈敏度 (Std)
    sens = {m: float(np.std(profits[m])) for m in profits}

    # (2) SD=8,10,12 的機率
    prob_gt100 = {}
    for sd in [8,10,12]:
        p_sim = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof  = S * V * p_sim
        prob_gt100[sd] = float(np.mean(prof > 100))

    fn_prob = "q3_p_gt100.png"
    plt.figure()
    xs = list(prob_gt100.keys())
    ys = list(prob_gt100.values())
    plt.plot(xs, ys, marker='o')
    plt.title("P(Profit>100) vs P SD")
    plt.xlabel("P SD")
    plt.ylabel("Probability")
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    return {
        "sensitivity_std": sens,
        "prob_profit_gt_100": prob_gt100,
        "plots": {"dist": fn_dist, "p_gt100": fn_prob}
    }
