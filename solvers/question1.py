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
    # 1) 讀檔、計算賠償金額與賠款率
    df      = pd.read_excel(file_path)
    amounts = df["賠償金額"].dropna().astype(float)
    amounts = amounts[amounts > 0]
    ratios  = (df["賠償金額"] / df["保險金額"]).dropna().astype(float)
    ratios  = ratios[ratios > 0]

    # 2) 挑最佳分布
    candidates = {
        "lognorm":     stats.lognorm,
        "gamma":       stats.gamma,
        "weibull_min": stats.weibull_min
    }
    best_amt, params_amt = fit_best_dist(amounts, candidates)
    best_rat, params_rat = fit_best_dist(ratios,  candidates)

    # 3) 計算歷史 VaR/CVaR
    levels = [0.01, 0.05]
    var_hist_amt, cvar_hist_amt = {}, {}
    var_hist_rat, cvar_hist_rat = {}, {}
    for α in levels:
        key = f"{int(α*100)}%"
        va, ca = compute_var_cvar(amounts, α)
        vr, cr = compute_var_cvar(ratios, α)
        var_hist_amt[key]  = va
        cvar_hist_amt[key] = ca
        var_hist_rat[key]  = vr * 100
        cvar_hist_rat[key] = cr * 100

    # 4) Monte Carlo 模擬 100k
    Nsim    = 100_000
    sim_amt = candidates[best_amt].rvs(*params_amt, size=Nsim)
    sim_rat = candidates[best_rat].rvs(*params_rat, size=Nsim)

    var_mc_amt, cvar_mc_amt = {}, {}
    var_mc_rat, cvar_mc_rat = {}, {}
    for α in levels:
        key = f"{int(α*100)}%"
        va, ca = compute_var_cvar(sim_amt, α)
        vr, cr = compute_var_cvar(sim_rat, α)
        var_mc_amt[key]    = va
        cvar_mc_amt[key]   = ca
        var_mc_rat[key]    = vr * 100
        cvar_mc_rat[key]   = cr * 100

    # 5) 畫機率密度函數（Simulation only, 1%-99% 裁切）
    os.makedirs("static/results", exist_ok=True)

    # 裁切邊界
    amt_lo, amt_hi = np.percentile(sim_amt, [1, 99])
    rat_lo, rat_hi = np.percentile(sim_rat, [1, 99])

    sim_amt_trim = sim_amt[(sim_amt>=amt_lo)&(sim_amt<=amt_hi)]
    sim_rat_trim = sim_rat[(sim_rat>=rat_lo)&(sim_rat<=rat_hi)]

    # — 賠償金額 —
    fn_amt = "q1_sim_amount_density.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.hist(sim_amt_trim, bins=50, density=True,
            color="skyblue", edgecolor="gray", alpha=0.7,
            label=f"{best_amt} simulation (n={Nsim})")
    ax.set_xlim(amt_lo, amt_hi)
    ax.set_xlabel("Claim Amount")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_amt))
    plt.close()

    # — 賠款率 —
    fn_rat = "q1_sim_ratio_density.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.hist(sim_rat_trim*100, bins=50, density=True,
            color="lightgreen", edgecolor="gray", alpha=0.7,
            label=f"{best_rat} simulation (n={Nsim})")
    ax.set_xlim(rat_lo*100, rat_hi*100)
    ax.set_xlabel("Claim Ratio (%)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_rat))
    plt.close()

    # 6) 回傳
    return {
        "最佳分布（賠償金額）":    best_amt,
        "分布參數（賠償金額）":    params_amt,
        "最佳分布（賠款率）":      best_rat,
        "分布參數（賠款率）":      params_rat,
        "歷史VaR（賠償金額）":    var_hist_amt,
        "歷史CVaR（賠償金額）":   cvar_hist_amt,
        "歷史VaR（賠款率 %）":    var_hist_rat,
        "歷史CVaR（賠款率 %）":   cvar_hist_rat,
        "MC VaR（賠償金額）":     var_mc_amt,
        "MC CVaR（賠償金額）":    cvar_mc_amt,
        "MC VaR（賠款率 %）":     var_mc_rat,
        "MC CVaR（賠款率 %）":    cvar_mc_rat,
        "plots": {
            "賠償金額機率密度圖": fn_amt,
            "賠款率機率密度圖":   fn_rat
        }
    }
