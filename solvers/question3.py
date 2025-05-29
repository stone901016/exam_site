# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    """
    Profit.xls 解題（更通用）：
    1. 直接讀取帶標頭的 sheet，找到前三個 numeric 欄
    2. 第一欄當 S，第二當 V，第三當 P 樣本
    3. P mean=45/55 模擬 Profit，畫比較圖並算 Std
    4. P 平均固定，SD=8,10,12時計算 P(Profit>100)，畫折線圖
    """
    # 1) 讀標頭 sheet
    df = pd.read_excel(file_path, header=0)
    # 選出所有 numeric 欄
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        raise KeyError(f"Profit.xls 數值欄位不足：只有找到 {len(num_cols)} 個 numeric 欄")
    # 拿前三欄
    col_S, col_V, col_P = num_cols[:3]

    # 2) 解析 S, V, P
    S_vals = df[col_S].dropna().astype(float)
    V_vals = df[col_V].dropna().astype(float)
    P_vals = df[col_P].dropna().astype(float)
    # 若 S、V 多筆取平均
    S = float(S_vals.mean())
    V = float(V_vals.mean())
    P_mean = float(P_vals.mean())
    P_sd   = float(P_vals.std())

    N = 100_000
    # (3) P mean=45,55 模擬 Profit
    profits = {}
    for m in (45, 55):
        sim_p = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S * V * sim_p

    # 繪分布比較
    os.makedirs("static/results", exist_ok=True)
    fn_dist = "q3_dist.png"
    plt.figure(figsize=(8,5))
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.legend()
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_dist))
    plt.close()

    # 靈敏度
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # (4) SD=8,10,12 時 P(Profit>100)
    prob_gt100 = {}
    for sd in (8, 10, 12):
        sim_p = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof  = S * V * sim_p
        prob_gt100[sd] = float((prof > 100).mean())

    # 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(8,5))
    xs, ys = zip(*sorted(prob_gt100.items()))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("P SD")
    plt.ylabel("P(Profit>100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    # 回傳
    return {
        "靈敏度（收益標準差）": sensitivity,
        "P(Profit>100) 機率": prob_gt100,
        "plots": {"分布比較圖": fn_dist, "機率變化圖": fn_prob}
    }
