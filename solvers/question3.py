# solvers/question3.py
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# 通用 sans-serif 字型
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def solve(_):
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]
    N = 100_000

    # 1) Profit & Std
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
    ax.set_xlabel("P Mean")                   # English label
    ax.set_ylabel("Profit")                    # English label
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
    ax.set_xlabel("P Mean")                   # English label
    ax.set_ylabel("Profit Std Dev")            # English label
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    explain1 = (
        f"When P Mean changes from {mean_values[0]} to {mean_values[1]}, "
        f"Profit goes from {profits[mean_values[0]]:.1f} to {profits[mean_values[1]]:.1f}, "
        f"with standard deviation fixed at {sensitivity[mean_values[0]]:.1f}."
    )

    # 2) Probability vs P SD
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
    ax.set_xlabel("P Standard Deviation")     # English label
    ax.set_ylabel("P(Profit > 100)")           # English label
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    explain2 = (
        f"As P SD goes from {xs2[0]} to {xs2[-1]}, "
        f"P(Profit>100) changes from {prob_gt100[xs2[0]]:.4f} to {prob_gt100[xs2[-1]]:.4f}."
    )

    return {
        "profit_chart": fn_profit,
        "sensitivity_chart": fn_sens,
        "explanation1": explain1,
        "probabilities": prob_gt100,
        "explanation2": explain2,
        "prob_chart": fn_prob,
        "plots": {
            "Profit Chart": fn_profit,
            "Sensitivity Chart": fn_sens,
            "Probability Chart": fn_prob
        }
    }
