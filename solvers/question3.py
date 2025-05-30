# solvers/question3.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def solve(_):
    """
    第三題：
    1) S=10, V=0.6, P_SD=10 固定，
       P_mean 分別為 45、55 → 繪製「盈利圖」與「靈敏度分析圖」，並說明
    2) P_mean=50 固定，P_SD 分別為 8、10、12 → 繪製 P(Profit>100) 機率變化圖，並說明
    """
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]
    N = 100_000

    # 第一部分
    profits    = {m: S * V * m for m in mean_values}
    sensitivity= {m: S * V * SD_const for m in mean_values}

    os.makedirs("static/results", exist_ok=True)

    # 盈利圖
    fn_profit = "q3_profit_bar.png"
    plt.figure(figsize=(16,10))
    xs = [str(m) for m in mean_values]
    ys = [profits[m] for m in mean_values]
    plt.bar(xs, ys, color=["#4C72B0","#55A868"])
    plt.xlabel("P 平均值")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_profit))
    plt.close()

    # 靈敏度分析圖
    fn_sens = "q3_sensitivity_bar.png"
    plt.figure(figsize=(16,10))
    ys2 = [sensitivity[m] for m in mean_values]
    plt.bar(xs, ys2, color=["#C44E52","#8172B2"])
    plt.xlabel("P 平均值")
    plt.ylabel("Profit 標準差 (靈敏度)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    explain1 = (
        f"當 P 平均值從 {mean_values[0]} 增至 {mean_values[1]} 時，"
        f"Profit 從 {profits[mean_values[0]]:.1f} 增加到 {profits[mean_values[1]]:.1f}；"
        f"由於 SD 固定為 {SD_const}，收益標準差均為 {sensitivity[mean_values[0]]:.1f}。"
    )

    # 第二部分
    P_mean_fixed = 50.0
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        sim = norm.rvs(loc=P_mean_fixed, scale=sd, size=N)
        prof= S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(16,10))
    xs2 = sorted(prob_gt100.keys())
    ys3 = [prob_gt100[sd] for sd in xs2]
    plt.plot(xs2, ys3, marker="o")
    plt.xlabel("P 標準差")
    plt.ylabel("P(Profit > 100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    explain2 = (
        f"當 P 標準差從 {xs2[0]}→{xs2[-1]} 時，"
        f"P(Profit>100) 從 {prob_gt100[xs2[0]]:.4f} 變為 {prob_gt100[xs2[-1]]:.4f}。"
    )

    return {
        "盈利圖檔案":         fn_profit,
        "靈敏度圖檔案":       fn_sens,
        "說明（盈利＆靈敏度）": explain1,
        "P(Profit>100) 機率": prob_gt100,
        "說明（機率變化）":     explain2,
        "機率變化圖檔案":     fn_prob,
        "plots": {
            "盈利圖":       fn_profit,
            "靈敏度圖":     fn_sens,
            "機率變化圖":   fn_prob
        }
    }
