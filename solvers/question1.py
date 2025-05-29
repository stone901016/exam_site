# solvers/question1.py
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fit_best_dist(data, dists):
    best_name, best_params, best_aic = None, None, np.inf
    aic_dict = {}
    for name, dist in dists.items():
        try:
            params = dist.fit(data, floc=0)
            ll = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2*k - 2*ll
            aic_dict[name] = aic
            if aic < best_aic:
                best_aic, best_name, best_params = aic, name, params
        except Exception:
            continue
    return best_name, best_params, aic_dict

def compute_var_cvar(arr, alpha):
    v = np.percentile(arr, alpha*100)
    cvar = arr[arr <= v].mean()
    return v, cvar

def solve(file_path):
    # 1. 讀檔 & 拿欄位
    df = pd.read_excel(file_path)
    if "賠償金額" not in df.columns:
        raise KeyError("找不到「賠償金額」欄位")
    amounts = df["賠償金額"].astype(float)
    amounts = amounts[amounts > 0]

    if "賠款率" in df.columns:
        ratios = df["賠款率"].astype(float)
    elif "保險金額" in df.columns:
        ratios = (amounts / df["保險金額"].astype(float)).dropna()
    else:
        raise KeyError("找不到「賠款率」或「保險金額」欄位")
    ratios = ratios[ratios > 0]

    # 2. 挑最佳分布
    candidates = {
        "lognorm":     stats.lognorm,
        "gamma":       stats.gamma,
        "weibull_min": stats.weibull_min
    }
    best_amt, params_amt, aic_amt = fit_best_dist(amounts, candidates)
    best_rat, params_rat, aic_rat = fit_best_dist(ratios, candidates)

    # 3. 歷史 VaR/CVaR
    levels = [0.01, 0.05]
    var_hist_amt, cvar_hist_amt = {}, {}
    var_hist_rat, cvar_hist_rat = {}, {}
    for α in levels:
        v_a, c_a = compute_var_cvar(amounts, α)
        v_r, c_r = compute_var_cvar(ratios, α)
        key = f"{int(α*100)}%"
        var_hist_amt[key], cvar_hist_amt[key] = v_a, c_a
        var_hist_rat[key], cvar_hist_rat[key] = v_r, c_r

    # 4. 蒙地卡羅 VaR/CVaR
    Nsim = 100_000
    sim_amt = candidates[best_amt].rvs(*params_amt, size=Nsim)
    sim_rat = candidates[best_rat].rvs(*params_rat, size=Nsim)

    var_mc_amt, cvar_mc_amt = {}, {}
    var_mc_rat, cvar_mc_rat = {}, {}
    for α in levels:
        v_a, c_a = compute_var_cvar(sim_amt, α)
        v_r, c_r = compute_var_cvar(sim_rat, α)
        key = f"{int(α*100)}%"
        var_mc_amt[key], cvar_mc_amt[key] = v_a, c_a
        var_mc_rat[key], cvar_mc_rat[key] = v_r, c_r

    # 5. 畫圖
    fn_amt = "q1_pdf_amount.png"
    x = np.linspace(amounts.min(), amounts.max(), 200)
    pdf = candidates[best_amt].pdf(x, *params_amt)
    plt.figure(); plt.plot(x, pdf); plt.title("賠償金額 機率密度"); plt.savefig(os.path.join("static","results",fn_amt)); plt.close()

    fn_rat = "q1_pdf_ratio.png"
    x2 = np.linspace(ratios.min(), ratios.max(), 200)
    pdf2 = candidates[best_rat].pdf(x2, *params_rat)
    plt.figure(); plt.plot(x2, pdf2); plt.title("賠款率 機率密度"); plt.savefig(os.path.join("static","results",fn_rat)); plt.close()

    # 6. 回傳結果
    return {
        "best_dist_amt": best_amt,
        "best_params_amt": tuple(params_amt),
        "aic_amt": aic_amt,
        "best_dist_rat": best_rat,
        "best_params_rat": tuple(params_rat),
        "aic_rat": aic_rat,
        "var_hist_amt": var_hist_amt,
        "cvar_hist_amt": cvar_hist_amt,
        "var_hist_rat": var_hist_rat,
        "cvar_hist_rat": cvar_hist_rat,
        "var_mc_amt": var_mc_amt,
        "cvar_mc_amt": cvar_mc_amt,
        "var_mc_rat": var_mc_rat,
        "cvar_mc_rat": cvar_mc_rat,
        "plots": {
            "pdf_amount": fn_amt,
            "pdf_ratio":  fn_rat
        }
    }
