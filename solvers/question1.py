# solvers/question1.py
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
    # 1. 讀檔並篩正值
    df      = pd.read_excel(file_path)
    amounts = df["賠償金額"].dropna().astype(float)
    amounts = amounts[amounts > 0]
    if "賠款率" in df:
        ratios = df["賠款率"].dropna().astype(float)
    else:
        ratios = (amounts / df["保險金額"].astype(float)).dropna()
    ratios = ratios[ratios > 0]

    # 2. 挑最佳分布
    candidates = {
        "lognorm":     stats.lognorm,
        "gamma":       stats.gamma,
        "weibull_min": stats.weibull_min
    }
    best_amt, params_amt = fit_best_dist(amounts, candidates)
    best_rat, params_rat = fit_best_dist(ratios,  candidates)

    # 3. Monte Carlo 模擬 100k
    Nsim    = 100_000
    sim_amt = candidates[best_amt].rvs(*params_amt, size=Nsim)
    sim_rat = candidates[best_rat].rvs(*params_rat, size=Nsim)

    # 4. 計算 VaR/CVaR （省略，保留你現有邏輯）…

    # 5. 準備畫圖的區間：取 1%–99% 百分位
    amt_low, amt_high   = np.percentile(amounts, [1, 99])
    rat_low, rat_high   = np.percentile(ratios, [1, 99])

    amt_orig_trim = amounts[(amounts >= amt_low) & (amounts <= amt_high)]
    amt_sim_trim  = sim_amt[(sim_amt >= amt_low) & (sim_amt <= amt_high)]
    rat_orig_trim = ratios[(ratios >= rat_low) & (ratios <= rat_high)]
    rat_sim_trim  = sim_rat[(sim_rat >= rat_low) & (sim_rat <= rat_high)]

    os.makedirs("static/results", exist_ok=True)

    # --- 賠償金額機率密度圖 ---
    fn_amt = "q1_pdf_amount.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)

    bins_amt = np.linspace(amt_low, amt_high, 50)
    ax.hist(amt_orig_trim,
            bins=bins_amt, density=True,
            alpha=0.4, color="skyblue", edgecolor="gray",
            label="原始資料 (1%–99%)")
    ax.hist(amt_sim_trim,
            bins=bins_amt, density=True,
            alpha=0.3, color="navy", edgecolor="none",
            label="模擬資料 (100k)")
    x = np.linspace(amt_low, amt_high, 500)
    y = candidates[best_amt].pdf(x, *params_amt)
    ax.plot(x, y, color="crimson", lw=2, label=f"{best_amt} 擬合曲線")
    ax.set_xlim(amt_low, amt_high)
    ax.set_xlabel("賠償金額")
    ax.set_ylabel("機率密度")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_amt))
    plt.close()

    # --- 賠款率機率密度圖 ---
    fn_rat = "q1_pdf_ratio.png"
    plt.figure(figsize=(16,10))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)

    bins_rat = np.linspace(rat_low, rat_high, 50)
    ax.hist(rat_orig_trim,
            bins=bins_rat, density=True,
            alpha=0.4, color="lightgreen", edgecolor="gray",
            label="原始資料 (1%–99%)")
    ax.hist(rat_sim_trim,
            bins=bins_rat, density=True,
            alpha=0.3, color="darkgreen", edgecolor="none",
            label="模擬資料 (100k)")
    x2 = np.linspace(rat_low, rat_high, 500)
    y2 = candidates[best_rat].pdf(x2, *params_rat)
    ax.plot(x2, y2, color="orange", lw=2, label=f"{best_rat} 擬合曲線")
    ax.set_xlim(rat_low, rat_high)
    ax.set_xlabel("賠款率")
    ax.set_ylabel("機率密度")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_rat))
    plt.close()

    # 6. 回傳
    return {
        # 保留其他結果欄位不動…
        "plots": {"pdf_amount": fn_amt, "pdf_ratio": fn_rat}
    }
