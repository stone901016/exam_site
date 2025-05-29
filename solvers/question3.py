# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    """
    Profit.xls 解題：
    1) 嘗試 header 0~2，自動找出前三個可轉換為數值的欄位當作 S, V, P
    2) P mean=45/55 時模擬 Profit 並繪分布 & 計算靈敏度 (Std)
    3) P mean 固定，SD=8,10,12時計算 P(Profit>100) 並繪折線圖
    """
    df = None
    S, V, P = None, None, None

    # 1. 嘗試不同 header
    for h in [0, 1, 2]:
        tmp = pd.read_excel(file_path, header=h)
        # 計算每欄可轉數值的筆數
        conv_counts = {
            col: pd.to_numeric(tmp[col], errors='coerce').notna().sum()
            for col in tmp.columns
        }
        # 只保留至少一筆數字的欄位
        valid = [(col, cnt) for col, cnt in conv_counts.items() if cnt > 0]
        if len(valid) >= 3:
            # 取筆數最多的前三欄
            valid.sort(key=lambda x: x[1], reverse=True)
            cols = [col for col, _ in valid[:3]]
            # 讀取並轉成 numeric Series
            S = pd.to_numeric(tmp[cols[0]], errors='coerce')
            V = pd.to_numeric(tmp[cols[1]], errors='coerce')
            P = pd.to_numeric(tmp[cols[2]], errors='coerce')
            df = tmp
            break

    if df is None:
        raise KeyError("Profit.xls 數值欄位不足：無法找到至少三個可轉為數值的欄位")

    # 2. 計算 S, V 平均值；P 均值與標準差
    S_mean = float(S.mean())
    V_mean = float(V.mean())
    P_vals = P.dropna()
    P_mean = float(P_vals.mean())
    P_sd   = float(P_vals.std())

    N = 100_000

    # (1) P mean = 45, 55 時的 Profit 模擬
    profits = {}
    for m in [45, 55]:
        p_sim = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S_mean * V_mean * p_sim

    # 繪製 Profit 分布比較圖
    fn_dist = "q3_dist.png"
    plt.figure()
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static", "results", fn_dist))
    plt.close()

    # 計算靈敏度（標準差）
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # (2) P mean 固定，SD = 8, 10, 12 時 P(Profit>100)
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        p_sim = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof  = S_mean * V_mean * p_sim
        prob_gt100[sd] = float(np.mean(prof > 100))

    # 繪製 P(Profit>100) vs SD 折線圖
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
