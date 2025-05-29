# solvers/question3.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, poisson, beta

def solve(_):
    # 1. 參數 (取自 Crystal Ball Report)
    P_mean, P_sd = 50.0, 10.0
    S_lambda     = 10.0
    V_alpha, V_beta = 3, 2

    # 2. 轉換 Lognormal 參數
    sigma_ln = np.sqrt(np.log(1 + (P_sd / P_mean) ** 2))
    mu_ln    = np.log(P_mean) - 0.5 * sigma_ln**2

    N = 100_000

    # 3. 抽樣
    P_sim = lognorm(s=sigma_ln, scale=np.exp(mu_ln)).rvs(size=N)
    S_sim = poisson(mu=S_lambda).rvs(size=N)
    V_sim = beta(a=V_alpha, b=V_beta).rvs(size=N)

    # 4. Profit
    profit = P_sim * S_sim * V_sim

    # 5. 統計量
    mean_profit = float(np.mean(profit))
    std_profit  = float(np.std(profit))
    prob_gt100  = float(np.mean(profit > 100))

    # 6. 繪圖 (放大兩倍)
    os.makedirs("static/results", exist_ok=True)

    # 直方圖
    fn_hist = "q3_hist.png"
    plt.figure(figsize=(16,10))
    plt.hist(profit, bins=50, color="#4C72B0", alpha=0.7)
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_hist))
    plt.close()

    # CDF
    fn_cdf = "q3_cdf.png"
    sorted_p = np.sort(profit)
    cdf = np.arange(1, N+1) / N
    plt.figure(figsize=(16,10))
    plt.plot(sorted_p, cdf, color="#55A868")
    plt.xlabel("Profit")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_cdf))
    plt.close()

    # 7. 回傳結果
    return {
        "平均 Profit": mean_profit,
        "Profit 標準差": std_profit,
        "P(Profit > 100)": prob_gt100,
        "plots": {
            "直方圖": fn_hist,
            "累積機率圖": fn_cdf
        }
    }
