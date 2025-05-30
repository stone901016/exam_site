# solvers/question3.py
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# —— 通用字型設定 ——  
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  
matplotlib.rcParams['axes.unicode_minus'] = False  

def solve(_):
    """
    第三題：
    1) 假設 S=10, V=0.6, P_SD=10 固定，
       P_mean 分別為 45、55 → 繪製「盈利圖」與「靈敏度分析圖」
    2) P_mean=50 固定，P_SD 分別為 8、10、12 → 繪製 P(Profit>100) 機率變化圖
    """
    S, V = 10.0, 0.6
    SD_const = 10.0
    mean_values = [45, 55]
    N = 100_000

    # 1) 計算 Profit 及其標準差
    profits    = {m: S * V * m for m in mean_values}
    sensitivity= {m: S * V * SD_const for m in mean_values}

    os.makedirs("static/results", exist_ok=True)

    # —— 盈利圖 ——  
    fn_profit = "q3_profit_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    xs = [str(m) for m in mean_values]
    ys = [profits[m] for m in mean_values]
    ax.bar(xs, ys, color=['#4C72B0','#55A868'])
    ax.set_xlabel("P 平均值")            # 中文 X 軸
    ax.set_ylabel("Profit")             # 中文或英文 Y 軸都能正確顯示
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_profit))
    plt.close()

    # —— 靈敏度分析圖 ——  
    fn_sens = "q3_sensitivity_bar.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    ys2 = [sensitivity[m] for m in mean_values]
    ax.bar(xs, ys2, color=['#C44E52','#8172B2'])
    ax.set_xlabel("P 平均值")           
    ax.set_ylabel("Profit 標準差 (靈敏度)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_sens))
    plt.close()

    explain1 = (
        f"當 P 平均值從 {mean_values[0]} → {mean_values[1]} 時，"
        f"Profit 從 {profits[mean_values[0]]:.1f} 增至 {profits[mean_values[1]]:.1f}，"
        f"且收益標準差 (靈敏度) 為 {sensitivity[mean_values[0]]:.1f}。"
    )

    # 2) P_mean=50，SD = 8,10,12 時機率變化
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
    ax.set_xlabel("P 標準差")           
    ax.set_ylabel("P(Profit > 100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    explain2 = (
        f"當 P 標準差由 {xs2[0]} → {xs2[-1]} 時，"
        f"P(Profit>100) 由 {prob_gt100[xs2[0]]:.4f} 增至 {prob_gt100[xs2[-1]]:.4f}。"
    )

    return {
        "盈利圖檔案":             fn_profit,
        "靈敏度分析圖檔案":       fn_sens,
        "說明（盈利＆靈敏度）":    explain1,
        "P(Profit>100) 機率":     prob_gt100,
        "說明（機率變化）":        explain2,
        "機率變化圖檔案":         fn_prob,
        "plots": {
            "盈利圖":         fn_profit,
            "靈敏度分析圖":   fn_sens,
            "機率變化圖":     fn_prob
        }
    }
