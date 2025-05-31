# solvers/question4.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """
    從任意格式字串擷取第一組浮點數 (支援千分位、小數)，找不到就回 None。
    """
    if pd.isna(val):
        return None
    s = str(val).strip().replace(',', '')
    import re
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def solve(file_path):
    """
    第 4 題：模擬投資人從 30 歲到 60 歲 (共 30 年)，每年投入 10,000 元 (考慮 2.2% 通膨)，
    投資標的是基金與股票（皆假設年化報酬為常態分布），模擬次數 10,000 次。
    Q1: 畫出 50%、70%、90% 股票權重下的「年齡 vs. 平均累積價值」趨勢圖 (共 30 點)。
    Q2: 畫出三種權重下的「最終累積價值」CDF 圖 (x 軸：最終累積值，y 軸：累積機率)。
    圖片放大 1.5 倍 (原本 figsize(6,4)，改為(9,6))。
    """
    # 嘗試讀 Excel (支援 .xls/.xlsx)
    try:
        df_all = pd.read_excel(file_path, header=None)
    except Exception:
        raise KeyError("第4題：無法讀取 Excel，請確認檔案格式。")

    # 1) 把歷史 Total Returns 找出來，若無法抓到就用預設值
    mean_fund   = None
    sd_fund     = None
    mean_stock  = None
    sd_stock    = None

    try:
        rows_with_year = []
        for r in range(df_all.shape[0]):
            v = extract_num(df_all.iat[r, 0])
            if v is not None and 1900 < int(v) < 2100:
                rows_with_year.append(r)
        if rows_with_year:
            df = df_all.loc[rows_with_year].reset_index(drop=True)
        else:
            df = df_all.copy()

        numeric_cols = {}
        for c in range(df.shape[1]):
            arr = df.iloc[:, c].map(extract_num).dropna().astype(float)
            if arr.size == 0:
                continue
            med = np.median(arr)
            if 1.0 < med < 10.0:
                numeric_cols[c] = arr.values

        if len(numeric_cols) >= 2:
            idxs = sorted(numeric_cols.keys())[:2]
            stock_tot = np.array(numeric_cols[idxs[0]])
            bond_tot  = np.array(numeric_cols[idxs[1]])
            r_stock = stock_tot - 1.0
            r_bond  = bond_tot - 1.0
            mean_stock = float(np.mean(r_stock))
            sd_stock   = float(np.std(r_stock, ddof=0))
            mean_fund  = float(np.mean(r_bond))
            sd_fund    = float(np.std(r_bond, ddof=0))
    except Exception:
        pass

    if mean_fund is None or sd_fund is None or mean_stock is None or sd_stock is None:
        mean_fund   = 0.05856962025316456   # 年化平均 5.85696%
        sd_fund     = 0.07592167742079621   # 年化標準差 7.59217%
        mean_stock  = 0.14956962025316461   # 年化平均 14.95696%
        sd_stock    = 0.25180571843499633   # 年化標準差 25.18057%

    # 2) 模擬參數
    start_age   = 30
    end_age     = 60
    years       = end_age - start_age   # 30 年
    Nsim        = 10_000                 # 模擬次數
    deposit     = 10_000                 # 每年年初投入
    inflation   = 1.022                  # 通貨膨脹 2.2%
    weights     = [0.5, 0.7, 0.9]         # 50%、70%、90% Stocks

    # 3) 事先產生 (Nsim × years) 的常態隨機矩陣
    rng = np.random.default_rng()
    r_fund  = rng.normal(loc=mean_fund,  scale=sd_fund,  size=(Nsim, years))
    r_stock = rng.normal(loc=mean_stock, scale=sd_stock, size=(Nsim, years))

    # 4) 對每個 w 計算 paths (Nsim × years) 與 final accumulation (Nsim,)
    paths = {}
    final = {}

    for w in weights:
        sims = np.zeros((Nsim, years))
        for t in range(years):
            dep = deposit * (inflation ** t)
            r   = w * r_stock[:, t] + (1 - w) * r_fund[:, t]
            if t == 0:
                sims[:, 0] = dep * (1 + r)
            else:
                sims[:, t] = sims[:, t - 1] * (1 + r) + dep
        paths[w] = sims
        final[w] = sims[:, -1]

    # 5) 準備存檔資料夾
    os.makedirs("static/results", exist_ok=True)

    trend_files = {}
    cdf_files   = {}

    ages = np.arange(start_age, end_age)

    # 6) Q1: Trend 圖，放大 1.5 倍 => figsize(9,6)
    for w in weights:
        sims = paths[w]
        mean_path = sims.mean(axis=0)
        fn = f"Q1_Trend_{int(w*100)}.png"

        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.plot(ages, mean_path, color="#4C72B0", linewidth=2)
        ax.set_title(f"Trend (w={int(w*100)}% 股票)", fontproperties="sans-serif", size=14)
        ax.set_xlabel("年齡", fontproperties="sans-serif", size=12)
        ax.set_ylabel("平均累積價值", fontproperties="sans-serif", size=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()

        trend_files[w] = fn

    # 7) Q2: CDF 圖，同樣放大 1.5 倍 => figsize(9,6)
    for w in weights:
        arr = np.sort(final[w])
        cdf = np.arange(1, Nsim + 1) / Nsim
        fn = f"Q2_CDF_{int(w*100)}.png"

        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.plot(arr, cdf, color="#55A868", linewidth=2)
        ax.set_title(f"CDF (w={int(w*100)}% 股票)", fontproperties="sans-serif", size=14)
        ax.set_xlabel("最終累積價值", fontproperties="sans-serif", size=12)
        ax.set_ylabel("累積機率", fontproperties="sans-serif", size=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static/results", fn))
        plt.close()

        cdf_files[w] = fn

    return {
        "plots_trend": trend_files,
        "plots_cdf":   cdf_files
    }
