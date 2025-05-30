import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fit_best_dist(data, dists):
    best_name, best_params, best_aic = None, None, np.inf
    for name, dist in dists.items():
        try:
            params = dist.fit(data, floc=0)
            ll = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2*k - 2*ll
            if aic < best_aic:
                best_aic, best_name, best_params = aic, name, params
        except:
            continue
    return best_name, best_params

def compute_var_cvar(arr, alpha):
    v = np.percentile(arr, alpha*100)
    cvar = arr[arr <= v].mean()
    return v, cvar

def solve(file_path):
    # 讀檔
    df      = pd.read_excel(file_path)
    amounts = df["賠償金額"].dropna().astype(float)
    amounts = amounts[amounts > 0]
    if "賠款率" in df:
        ratios = df["賠款率"].dropna().astype(float)
    else:
        ratios = (amounts / df["保險金額"].astype(float)).dropna()
    ratios = ratios[ratios > 0]

    # 挑分布
    candidates = {
        "lognorm":     stats.lognorm,
        "gamma":       stats.gamma,
        "weibull_min": stats.weibull_min
    }
    best_amt, params_amt = fit_best_dist(amounts, candidates)
    best_rat, params_rat = fit_best_dist(ratios, candidates)

    # VaR/CVaR 等計算...（保持不變）

    # 畫圖
    os.makedirs("static/results", exist_ok=True)

    # 賠償金額密度圖
    fn_amt = "q1_pdf_amount.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    # 加背景網格
    ax.grid(True, linestyle='--', alpha=0.5)
    # histogram + pdf
    bins = np.linspace(amounts.min(), amounts.max(), 50)
    ax.hist(amounts, bins=bins, density=True,
            color='skyblue', alpha=0.4, edgecolor='gray')
    x = np.linspace(amounts.min(), amounts.max(), 500)
    y = candidates[best_amt].pdf(x, *params_amt)
    ax.plot(x, y, color='crimson', lw=2)
    ax.fill_between(x, y, color='crimson', alpha=0.2)
    ax.set_xlabel("賠償金額")
    ax.set_ylabel("機率密度")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_amt))
    plt.close()

    # 賠款率密度圖
    fn_rat = "q1_pdf_ratio.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle='--', alpha=0.5)
    bins2 = np.linspace(ratios.min(), ratios.max(), 50)
    ax.hist(ratios, bins=bins2, density=True,
            color='lightgreen', alpha=0.4, edgecolor='gray')
    x2 = np.linspace(ratios.min(), ratios.max(), 500)
    y2 = candidates[best_rat].pdf(x2, *params_rat)
    ax.plot(x2, y2, color='darkorange', lw=2)
    ax.fill_between(x2, y2, color='darkorange', alpha=0.2)
    ax.set_xlabel("賠款率")
    ax.set_ylabel("機率密度")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_rat))
    plt.close()

    return {
        # ... 其餘結果保持 ...
        "plots": {"pdf_amount": fn_amt, "pdf_ratio": fn_rat}
    }
