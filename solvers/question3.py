# solvers/question3.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(_):
    """
    第三題：
    1) S=10, V=0.6, P_SD=10 固定
       P_mean 分別 45、55 → 盈利圖 & 靈敏度分析圖，並說明
    2) P_mean=50 固定，P_SD=8,10,12 → P(Profit>100) 機率變化圖
    """
    # 假設參數
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]

    # 1) 盈利與靈敏度計算
    profits = {m: S * V * m for m in mean_values}
    sensitivity = {m: S * V * SD_const for m in mean_values}

    # 繪「盈利圖」（Bar Chart）
    os.makedirs("static/results", exist_ok=True)
    fn_profit = "q3_profit_bar.png"
    plt.figure(figsize=(16,10))
    xs = mean_values
    ys = [profits[m] for m in xs]
    plt.bar([str(m) for m in xs], ys, color=['#4C72B0','#55A868'])
    plt.xlabel("P 的平均值")
    plt.ylabel("Profit")
    plt.title("不同 P 平均值下的 Profit")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_profit))
    plt.close()

    # 繪「靈敏度分析圖」（Bar Chart of Std）
    fn_sens = "q3_sensitivity_bar.png"
    plt.figure(figsize=(16,10))
    ys2 = [sensitivity[m] for m in xs]
    plt.bar([str(m) for m in xs], ys2, color=['#C44E52','#8172B2'])
    plt.xlabel("P 的平均值")
    plt.ylabel("Profit 標準差（靈敏度）")
    plt.title("不同 P 平均值下的 Profit 標準差")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    explain1 = (
        f"當 P 的平均值從 {mean_values[0]} → {mean_values[1]} 時，"
        f"Profit 從 {profits[45]:.1f} 增加到 {profits[55]:.1f}；"
        f"由於 SD 固定為 {SD_const}，Profit 的標準差（靈敏度）"
        f"均為 {sensitivity[45]:.1f}。"
    )

    # 2) P_mean=50 固定，SD = 8,10,12 → P(Profit>100) 機率
    P_mean_fixed = 50.0
    prob_gt100 = {}
    N = 100_000
    for sd in [8, 10, 12]:
        sim = norm.rvs(loc=P_mean_fixed, scale=sd, size=N)
        prof = S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

    # 繪機率變化圖
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(16,10))
    xs2 = sorted(prob_gt100.keys())
    ys3 = [prob_gt100[sd] for sd in xs2]
    plt.plot(xs2, ys3, marker='o')
    plt.xlabel("P 的標準差")
    plt.ylabel("P(Profit > 100)")
    plt.title("不同 P 標準差下，Profit>100 的機率")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    explain2 = (
        f"當 P 的標準差從 {xs2[0]} → {xs2[-1]} 時，"
        f"P(Profit>100) 從 {prob_gt100[xs2[0]]:.4f} 變為 {prob_gt100[xs2[-1]]:.4f}。"
    )

    return {
        "盈利圖檔案": fn_profit,
        "靈敏度圖檔案": fn_sens,
        "說明（盈利＆靈敏度）": explain1,
        "P(Profit>100) 機率": prob_gt100,
        "說明（機率變化）": explain2,
        "plots": {
            "盈利圖": fn_profit,
            "靈敏度圖": fn_sens,
            "機率變化圖": fn_prob
        }
    }
