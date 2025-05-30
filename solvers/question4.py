# solvers/question4.py
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

def extract_num(val):
    """從字串中擷取第一組浮點數（支援千分位、小數）。"""
    if pd.isna(val): return None
    s = str(val).replace(',', '')
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def solve(file_path):
    # 1) 讀 .xls
    try:
        xls = pd.ExcelFile(file_path, engine='xlrd')
    except ImportError:
        raise ImportError("第4題需 xlrd 讀 .xls，請加 xlrd>=2.0.1")

    # 2) 找出第一個有資料的 sheet
    df = None
    for sh in xls.sheet_names:
        tmp = xls.parse(sh, header=None)
        if not tmp.empty:
            df = tmp; break
    if df is None or df.shape[0]<2 or df.shape[1]<2:
        raise KeyError("第4題：請確認第一列 Mean、第二列 StdDev，且至少兩欄。")

    # 3) 擷取參數
    mean_fund  = extract_num(df.iat[0,0]); sd_fund  = extract_num(df.iat[1,0])
    mean_stock = extract_num(df.iat[0,1]); sd_stock = extract_num(df.iat[1,1])
    if None in (mean_fund, sd_fund, mean_stock, sd_stock):
        raise KeyError("第4題：Mean/StdDev 含非數值，請檢查 Excel。")

    # 4) 模擬設定
    start_age  = 30
    years      = 60 - start_age   # 30 年
    Nsim       = 10_000
    deposit    = 10_000
    inflation  = 1.022
    weights    = [0.5, 0.7, 0.9]

    # 5) 生成各年回報模擬
    fund_r = norm.rvs(loc=mean_fund,  scale=sd_fund,  size=(Nsim, years))
    stc_r  = norm.rvs(loc=mean_stock, scale=sd_stock, size=(Nsim, years))

    paths = {}    # 每年累積路徑
    final = {}    # 最終累積值

    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w*stc_r[:,t] + (1-w)*fund_r[:,t]
            sims[:,t] = dep*(1+r) if t==0 else sims[:,t-1]*(1+r) + dep
        paths[w] = sims
        final[w] = sims[:,-1]

    os.makedirs("static/results", exist_ok=True)

    # 6) Trend 圖
    fn_trend = "q4_trend.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    ages = np.arange(start_age, start_age+years)
    for w,color in zip(weights, ["#4C72B0","#55A868","#C44E52"]):
        ax.plot(ages, paths[w].mean(axis=0),
                label=f"{int(w*100)}% Stocks", color=color, lw=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Accumulated Value")
    ax.set_title("Trend of Accumulated Value (Age 30→60)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static/results",fn_trend))
    plt.close()

    # 7) CDF 圖
    fn_cdf = "q4_cdf.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    for w,color in zip(weights, ["#4C72B0","#55A868","#C44E52"]):
        sorted_vals = np.sort(final[w])
        cdf = np.arange(1, Nsim+1)/Nsim
        ax.plot(sorted_vals, cdf, label=f"{int(w*100)}% Stocks", color=color)
    ax.set_xlabel("Final Accumulated Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Final Accumulated Value at Age 60")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static/results",fn_cdf))
    plt.close()

    # 8) 計算摘要統計
    stats = {}
    for w in weights:
        arr = final[w]
        μ    = arr.mean()
        med  = np.median(arr)
        # 近似 mode：histogram 最大頻率的 bin 中心
        h,edges = np.histogram(arr, bins=50)
        mode = (edges[:-1] + edges[1:]) / 2
        mode = float(mode[np.argmax(h)])
        σ    = arr.std(ddof=0)
        var  = arr.var(ddof=0)
        sk   = skew(arr)
        kurt = kurtosis(arr, fisher=False)
        cv   = σ / μ if μ!=0 else np.nan
        mn   = arr.min()
        mx   = arr.max()
        se   = σ / np.sqrt(Nsim)
        stats[f"{int(w*100)}% Stocks"] = {
            "Trials":           Nsim,
            "Mean":             μ,
            "Median":           med,
            "Mode":             mode,
            "StdDev":           σ,
            "Variance":         var,
            "Skewness":         sk,
            "Kurtosis":         kurt,
            "CoefOfVar":        cv,
            "Minimum":          mn,
            "Maximum":          mx,
            "MeanStdError":     se
        }

    # 9) 回傳
    return {
        "plots": {
            "累積趨勢圖": fn_trend,
            "累積值CDF": fn_cdf
        },
        "summary_stats": stats
    }
