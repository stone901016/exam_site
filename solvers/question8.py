# solvers/question8.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def solve(file_path):
    df = pd.read_excel(file_path, header=None)
    # 依題目 C6=第5列、第2欄；C8=第7列、第2欄；C10=第9列、第2欄 (0-index)
    mu1, sd1 = df.iat[5,1], df.iat[5,2]  # Demand
    mu2, sd2 = df.iat[7,1], df.iat[7,2]  # InitPrice
    mu3, sd3 = df.iat[9,1], df.iat[9,2]  # SalePrice

    N = 10000
    def sim(m, s):
        # compute lognormal params
        sigma = np.sqrt(np.log(1 + (s/m)**2))
        mu = np.log(m) - 0.5*sigma**2
        return lognorm(s=sigma, scale=np.exp(mu)).rvs(size=N)

    d = sim(mu1, sd1)
    i = sim(mu2, sd2)
    p = sim(mu3, sd3)
    profit = d * (p - i)

    # 1) CDF
    fn1 = "q8_cdf.png"
    x = np.sort(profit)
    y = np.arange(1, N+1)/N
    plt.figure()
    plt.plot(x, y)
    plt.title("Profit 累積機率密度")
    plt.savefig(os.path.join("static","results",fn1))
    plt.close()

    # 2) 敏感度：皮爾森相關
    corr = {
        "Demand": np.corrcoef(d, profit)[0,1],
        "InitPrice": np.corrcoef(i, profit)[0,1],
        "SalePrice": np.corrcoef(p, profit)[0,1]
    }

    return {
        "corr": corr,
        "plots": {"cdf": fn1}
    }
