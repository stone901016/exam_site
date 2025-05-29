# solvers/question6.py
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def solve(file_path):
    df = pd.read_excel(file_path, sheet_name=0)
    # 假設欄位: 'Date','Close','Volume'
    price = df["Close"]
    volume= df["Volume"]

    # 比較常見分布：normal、lognormal、gamma，選擇 AIC 最小
    dists = {
        "normal":  stats.norm,
        "lognorm": stats.lognorm,
        "gamma":   stats.gamma
    }
    best, best_aic = None, np.inf
    for name, dist in dists.items():
        params = dist.fit(price)
        ll = np.sum(dist.logpdf(price, *params))
        k = len(params)
        aic = 2*k - 2*ll
        if aic < best_aic:
            best_aic, best, best_name = aic, params, name

    # 信賴區間
    ci95 = stats.norm.interval(0.95, loc=price.mean(), scale=price.std())
    ci99 = stats.norm.interval(0.99, loc=price.mean(), scale=price.std())

    # 畫圖
    fn = "q6_price_dist.png"
    x = np.linspace(price.min(), price.max(), 200)
    pdf = getattr(stats, best_name).pdf(x, *best)
    plt.figure()
    plt.plot(x, pdf)
    plt.title(f"最佳分布 ({best_name})")
    plt.savefig(os.path.join("static","results",fn))
    plt.close()

    return {
        "best_dist": best_name,
        "params": best,
        "CI95": ci95,
        "CI99": ci99,
        "plots": {"dist": fn}
    }
