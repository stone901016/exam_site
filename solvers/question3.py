# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    """
    Profit.xls 解題：
    1) 自動嘗試 header=0~2, 找出至少三個「去除千分位逗號後能轉成數值」的欄位當 S, V, P
    2) P mean=45/55 時模擬 Profit 分布並計算靈敏度 (Std)
    3) P mean 固定，SD=8,10,12時計算 P(Profit>100) 機率並繪折線圖
    4) 回傳敏感度、機率與兩張圖檔
    """
    df = None
    numeric_cols = None

    # 1. 嘗試不同 header
    for h in [0, 1, 2]:
        tmp = pd.read_excel(file_path, header=h)
        conv = {}
        for col in tmp.columns:
            # 先把值轉字串，去除千分位逗號，再轉數字
            s = tmp[col].astype(str).str.replace(",", "")
            nums = pd.to_numeric(s, errors="coerce")
            cnt = nums.notna().sum()
            conv[col] = cnt
        valid = [col for col, cnt in conv.items() if cnt > 0]
        if len(valid) >= 3:
            # 按照可轉數量排序，取前三
            valid.sort(key=lambda c: conv[c], reverse=True)
            numeric_cols = valid[:3]
            df = tmp
            break

    if df is None or numeric_cols is None:
        raise KeyError("Profit.xls 數值欄位不足：無法找到至少三個可轉為數值的欄位")

    # 2. 解析 S, V, P
    S_col, V_col, P_col = numeric_cols
    S = pd.to_numeric(df[S_col].astype(str).str.replace(",", ""), errors="coerce").dropna().astype(float)
    V = pd.to_numeric(df[V_col].astype(str).str.replace(",", ""), errors="coerce").dropna().astype(float)
    P_vals = pd.to_numeric(df[P_col].astype(str).str.replace(",", ""), errors="coerce").dropna().astype(float)

    S_mean = float(S.mean())
    V_mean = float(V.mean())
    P_mean = float(P_vals.mean())
    P_sd   = float(P_vals.std())

    N = 100_000

    # (1) P mean = 45, 55
    profits = {}
    for m in [45, 55]:
        sim_p = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S_mean * V_mean * sim_p

    # 繪分布比較圖
    fn_dist = "q3_dist.png"
    plt.figure()
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static", "results", fn_dist))
    plt.close()

    # 靈敏度 (Std)
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # (2) P mean 固定，SD = 8,10,12
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        sim_p = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof  = S_mean * V_mean * sim_p
        prob_gt100[sd] = float(np.mean(prof > 100))

    # 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure()
    xs = list(prob_gt100.keys())
    ys = list(prob_gt100.values())
    plt.plot(xs, ys, marker='o')
    plt.title("P(Profit > 100) vs P SD")
    plt.xlabel("P SD")
    plt.ylabel("Probability")
    plt.savefig(os.path.join("static", "results", fn_prob))
    plt.close()

    return {
        "sensitivity_std": sensitivity,
        "prob_profit_gt_100": prob_gt100,
        "plots": {"dist": fn_dist, "p_gt100": fn_prob}
    }
