# solvers/question4.py
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """擷取字串中第一組浮點數，支援千分位和小數。"""
    if pd.isna(val): return None
    s = str(val).replace(',', '')
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def solve(file_path):
    # 1) 讀 .xls
    try:
        xls = pd.ExcelFile(file_path, engine='xlrd')
    except ImportError:
        raise ImportError(
            "第4題需安裝 xlrd 才能讀取 .xls，"
            "請在 requirements.txt 加上 xlrd>=2.0.1 並重新部署。"
        )
    # 自動找第一個不空 sheet
    df = None
    for sh in xls.sheet_names:
        tmp = xls.parse(sh, header=None)
        if not tmp.empty:
            df = tmp
            break
    if df is None or df.shape[0] < 2 or df.shape[1] < 2:
        raise KeyError("第4題：請確認 Accumulate.xls 中，第一列 Mean、第二列 StdDev，且最少 2 欄。")

    # 2) 從前兩列、前兩欄擷取參數
    mean_fund  = extract_num(df.iat[0, 0])
    sd_fund    = extract_num(df.iat[1, 0])
    mean_stock = extract_num(df.iat[0, 1])
    sd_stock   = extract_num(df.iat[1, 1])
    if None in (mean_fund, sd_fund, mean_stock, sd_stock):
        raise KeyError("第4題：Mean/StdDev 欄位含非數值，請檢查 Excel 內容。")

    # 3) 參數設定
    start_age  = 30
    years      = 60 - start_age   # 30 年
    Nsim       = 10_000
    deposit    = 10_000
    inflation  = 1.022
    weights    = [0.5, 0.7, 0.9]

    # 4) 模擬
    # shape: (Nsim, years)
    fund_r = norm.rvs(loc=mean_fund,  scale=sd_fund,  size=(Nsim, years))
    stc_r  = norm.rvs(loc=mean_stock, scale=sd_stock, size=(Nsim, years))

    # 存放各 w 的累積路徑
    paths = {}
    final = {}

    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w * stc_r[:, t] + (1 - w) * fund_r[:, t]
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t-1] * (1 + r) + dep
        paths[w] = sims
        final[w] = sims[:, -1]

    os.makedirs("static/results", exist_ok=True)

    # 5) Q1 Trend：年齡 vs 平均累積值
    fn_trend = "q4_trend.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    ages = np.arange(start_age, start_age + years)
    for w, color in zip(weights, ["#4C72B0", "#55A868", "#C44E52"]):
        mean_path = paths[w].mean(axis=0)
        ax.plot(ages, mean_path, label=f"{int(w*100)}% Stocks", color=color, lw=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Accumulated Value")
    ax.set_title("Trend of Accumulated Value (Age 30→60)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn_trend))
    plt.close()

    # 6) Q2 CDF of final accumulation
    fn_cdf = "q4_cdf.png"
    plt.figure(figsize=(12,6))
    ax = plt.gca(); ax.grid(True, linestyle="--", alpha=0.5)
    for w, color in zip(weights, ["#4C72B0", "#55A868", "#C44E52"]):
        sorted_vals = np.sort(final[w])
        cdf = np.arange(1, Nsim+1) / Nsim
        ax.plot(sorted_vals, cdf, label=f"{int(w*100)}% Stocks", color=color)
    ax.set_xlabel("Final Accumulated Value")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Final Accumulated Value at Age 60")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static","results", fn_cdf))
    plt.close()

    # 7) 回傳
    return {
        "plots": {
            "累積趨勢圖": fn_trend,
            "累積值CDF": fn_cdf
        }
    }
