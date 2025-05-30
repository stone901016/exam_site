# solvers/question3.py
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# 強制使用通用 sans‐serif 字型，避免缺字
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def solve(_):
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]
    N = 100_000

    # 1) 計算 Profit 和靈敏度
    profits    = {m: S * V * m for m in mean_values}
    sensitivity= {m: S * V * SD_const for m in mean_values}

    os.makedirs("static/results", exist_ok=True)

    # —— 盈利圖 (Profit vs. P Mean) ——  
    fn_profit = "q3_profit_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    xs = [str(m) for m in mean_values]
    ys = [profits[m] for m in mean_values]
    ax.bar(xs, ys, color=['#4C72B0','#55A868'])
    ax.set_xlabel("P Mean")    # 英文
    ax.set_ylabel("Profit")     # 英文
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_profit))
    plt.close()

    # —— 靈敏度圖 (Profit Std Dev vs. P Mean) ——  
    fn_sens = "q3_sensitivity_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    ys2 = [sensitivity[m] for m in mean_values]
    ax.bar(xs, ys2, color=['#C44E52','#8172B2'])
    ax.set_xlabel("P Mean")          # 英文
    ax.set_ylabel("Profit Std Dev")   # 英文
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    # 中文說明
    explain1 = (
        f"當 P 平均值從 {mean_values[0]} 增至 {mean_values[1]} 時，"
        f"Profit 從 {profits[mean_values[0]]:.1f} 增至 {profits[mean_values[1]]:.1f}，"
        f"收益標準差固定為 {sensitivity[mean_values[0]]:.1f}。"
    )

    # 2) P_mean=50, SD=8,10,12 → P(Profit>100) 機率
    P_mean_fixed = 50.0
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        sim = norm.rvs(loc=P_mean_fixed, scale=sd, size=N)
        prof = S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

    # —— 機率變化圖 (P SD vs. P(Profit>100)) ——  
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    xs2 = sorted(prob_gt100.keys())
    ys3 = [prob_gt100[sd] for sd in xs2]
    ax.plot(xs2, ys3, marker='o')
    ax.set_xlabel("P Standard Deviation")  # 英文
    ax.set_ylabel("P(Profit > 100)")        # 英文
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    # 中文說明
    explain2 = (
        f"當 P 標準差從 {xs2[0]} 增至 {xs2[-1]} 時，"
        f"P(Profit>100) 從 {prob_gt100[xs2[0]]:.4f} 變為 {prob_gt100[xs2[-1]]:.4f}。"
    )

    return {
        "盈利圖檔案":       fn_profit,
        "靈敏度圖檔案":     fn_sens,
        "機率變化圖檔案":   fn_prob,
        "說明（盈利＆靈敏度）": explain1,
        "P(Profit>100) 機率":   prob_gt100,
        "說明（機率變化）":     explain2,
        "plots": {
            "盈利圖":     fn_profit,
            "靈敏度圖":   fn_sens,
            "機率變化圖": fn_prob
        }
    }
