# solvers/question3.py
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# —— 强制使用通用 sans-serif 字体，避免中文缺字 ——  
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def solve(_):
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]
    N = 100_000

    # 1) 计算 Profit 与 Sensitivity
    profits    = {m: S * V * m for m in mean_values}
    sensitivity= {m: S * V * SD_const for m in mean_values}

    os.makedirs("static/results", exist_ok=True)

    # —— Profit Bar Chart ——  
    fn_profit = "q3_profit_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    xs = [str(m) for m in mean_values]
    ys = [profits[m] for m in mean_values]
    ax.bar(xs, ys, color=['#4C72B0','#55A868'])
    ax.set_xlabel("P Mean")             # 英文
    ax.set_ylabel("Profit")              # 英文
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_profit))
    plt.close()

    # —— Sensitivity Bar Chart ——  
    fn_sens = "q3_sensitivity_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    ys2 = [sensitivity[m] for m in mean_values]
    ax.bar(xs, ys2, color=['#C44E52','#8172B2'])
    ax.set_xlabel("P Mean")             # 英文
    ax.set_ylabel("Profit Std Dev")      # 英文
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    explain1 = (
        f"When P Mean increases from {mean_values[0]} to {mean_values[1]}, "
        f"Profit goes from {profits[mean_values[0]]:.1f} to {profits[mean_values[1]]:.1f}, "
        f"with constant Std Dev {sensitivity[mean_values[0]]:.1f}."
    )

    # 2) P_mean=50, SD in [8,10,12] → P(Profit>100)
    P_mean_fixed = 50.0
    prob_gt100 = {}
    for sd in [8, 10, 12]:
        sim = norm.rvs(loc=P_mean_fixed, scale=sd, size=N)
        prof = S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

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

    explain2 = (
        f"As P SD moves from {xs2[0]} to {xs2[-1]}, "
        f"P(Profit>100) changes from {prob_gt100[xs2[0]]:.4f} to {prob_gt100[xs2[-1]]:.4f}."
    )

    # —— 返回必须含有这些中文 key ——  
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
