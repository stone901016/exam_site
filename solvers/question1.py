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
        except Exception:
            continue
    return best_name, best_params

def compute_var_cvar(arr, alpha):
    v = np.percentile(arr, alpha*100)
    cvar = arr[arr <= v].mean()
    return v, cvar

def solve(file_path):
    # 讀檔與篩正值
    df = pd.read_excel(file_path)
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

    # 計算並轉換百分比
    levels = [0.01, 0.05]
    var_hist_amt, cvar_hist_amt = {}, {}
    var_hist_rat, cvar_hist_rat = {}, {}
    for α in levels:
        key = f"{int(α*100)}%"
        v_a, c_a = compute_var_cvar(amounts, α)
        v_r, c_r = compute_var_cvar(ratios, α)
        var_hist_amt[key]  = v_a
        cvar_hist_amt[key] = c_a
        var_hist_rat[key]  = v_r * 100
        cvar_hist_rat[key] = c_r * 100

    # Monte Carlo
    Nsim = 100_000
    sim_amt = candidates[best_amt].rvs(*params_amt, size=Nsim)
    sim_rat = candidates[best_rat].rvs(*params_rat, size=Nsim)
    var_mc_amt, cvar_mc_amt = {}, {}
    var_mc_rat, cvar_mc_rat = {}, {}
    for α in levels:
        key = f"{int(α*100)}%"
        v_a, c_a = compute_var_cvar(sim_amt, α)
        v_r, c_r = compute_var_cvar(sim_rat, α)
        var_mc_amt[key]  = v_a
        cvar_mc_amt[key] = c_a
        var_mc_rat[key]  = v_r * 100
        cvar_mc_rat[key] = c_r * 100

    # 繪圖──Histogram + Fitted PDF
    os.makedirs("static/results", exist_ok=True)

    # 賠償金額密度圖
    fn_amt = "q1_pdf_amount.png"
    plt.figure(figsize=(16,10))
    # 實際 histogram
    plt.hist(amounts, bins=50, density=True, alpha=0.4,
             color="skyblue", edgecolor="gray", label="資料密度")
    # 擬合分布曲線
    x = np.linspace(amounts.min(), amounts.max(), 500)
    y = candidates[best_amt].pdf(x, *params_amt)
    plt.plot(x, y, color="red", lw=2, label=f"擬合 {best_amt}")
    plt.xlabel("賠償金額")
    plt.ylabel("機率密度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_amt))
    plt.close()

    # 賠款率密度圖
    fn_rat = "q1_pdf_ratio.png"
    plt.figure(figsize=(16,10))
    plt.hist(ratios, bins=50, density=True, alpha=0.4,
             color="lightgreen", edgecolor="gray", label="資料密度")
    x2 = np.linspace(ratios.min(), ratios.max(), 500)
    y2 = candidates[best_rat].pdf(x2, *params_rat)
    plt.plot(x2, y2, color="orange", lw=2, label=f"擬合 {best_rat}")
    plt.xlabel("賠款率")
    plt.ylabel("機率密度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_rat))
    plt.close()

    # 回傳結果
    return {
        "最佳分布（賠償金額）": best_amt,
        "分布參數（賠償金額）": params_amt,
        "歷史VaR（賠償金額）": var_hist_amt,
        "歷史CVaR（賠償金額）": cvar_hist_amt,
        "MC VaR（賠償金額）": var_mc_amt,
        "MC CVaR（賠償金額）": cvar_mc_amt,

        "最佳分布（賠款率）": best_rat,
        "分布參數（賠款率）": params_rat,
        "歷史VaR（賠款率 %）": var_hist_rat,
        "歷史CVaR（賠款率 %）": cvar_hist_rat,
        "MC VaR（賠款率 %）": var_mc_rat,
        "MC CVaR（賠款率 %）": cvar_mc_rat,

        "plots": {
            "pdf_amount": fn_amt,
            "pdf_ratio":  fn_rat
        }
    }
