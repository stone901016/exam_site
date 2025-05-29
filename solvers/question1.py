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

    # 挑最佳分布
    candidates = {
        "lognorm":     stats.lognorm,
        "gamma":       stats.gamma,
        "weibull_min": stats.weibull_min
    }
    best_amt, params_amt = fit_best_dist(amounts, candidates)
    best_rat, params_rat = fit_best_dist(ratios, candidates)

    # 歷史 VaR/CVaR
    levels = [0.01, 0.05]
    var_hist_amt, cvar_hist_amt = {}, {}
    var_hist_rat, cvar_hist_rat = {}, {}
    for α in levels:
        key = f"{int(α*100)}%"
        v_a, c_a = compute_var_cvar(amounts, α)
        v_r, c_r = compute_var_cvar(ratios, α)
        var_hist_amt[key], cvar_hist_amt
