# solvers/question3.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(file_path):
    """
    Profit.xls 解題：
    1) 自動偵測前三個數值欄位當作 S、V、P
    2) P SD 不變，mean=45/55 時繪製 Profit 分布 & 計算靈敏度 (Std)
    3) P mean 不變，SD=8/10/12 時計算 P(Profit>100) 並繪圖
    4) 回傳敏感度、機率與兩張圖檔
    """
    # 1. 讀檔
    df = pd.read_excel(file_path)

    # 2. 自動偵測數值欄位
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        raise KeyError(f"Profit.xls 數值欄位不足：僅找到 {num_cols}")
    S_col, V_col, P_col = num_cols[:3]

    # 3. 取平均與標準差
    S = df[S_col].mean()
    V = df[V_col].mean()
    P_mean = df[P_col].mean()
    P_sd   = df[P_col].std()

    N = 100000

    # 4. P mean = 45, 55 時的 Profit 模擬
    profits_by_mean = {}
    for m in [45, 55]:
        p_sim = norm.rvs(loc=m, scale=P_sd, size=N)
        profits_by_mean[m] = S * V * p_sim

    # 5. 繪 Profit 分布比較圖
    fn_dist = "q3_dist.png"
    plt.figure()
    for m, prof in profits_by_mean.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static", "results", fn_dist))
    plt.close()

    # 6. 計算靈敏度（Std of Profit）
    sensitivity = {m: np.std(profits_by_mean[m]) for m in profits_by_mean}

    # 7. P mean 不變，SD=8,10,12 時 P(Profit>100)
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        p_sim = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof  = S * V * p_sim
        prob_gt100[sd] = np.mean(prof > 100)

    # 8. 繪 P(Profit>100) vs SD 圖
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
        "plots": {
            "dist": fn_dist,
            "p_gt100": fn_prob
        }
    }
