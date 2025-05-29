# solvers/question3.py
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """從任意格式字串擷取第一組數值（含千分位、負號、小數點）"""
    if pd.isna(val): return None
    s = str(val)
    m = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", s)
    if not m: return None
    num = m.group(0).replace(",", "")
    try: return float(num)
    except: return None

def solve(file_path):
    # 1) 全表讀取
    df0 = pd.read_excel(file_path, header=None)
    nrows, ncols = df0.shape

    # 2) 定位 S, V
    S = V = None
    for r in range(nrows):
        for c in range(ncols):
            v = str(df0.iat[r,c]).strip().lower()
            if v == "s":
                S = extract_num(df0.iat[r, c+1])
            elif v == "v":
                V = extract_num(df0.iat[r, c+1])
    if S is None or V is None:
        raise KeyError("找不到 S 或 V 的位置，請確認 Excel 標題")

    # 3) 定位 P 欄
    Pcol = None; Pheader_row = None
    for r in range(nrows):
        for c in range(ncols):
            v = str(df0.iat[r,c]).strip().lower()
            if v in ("p","price","價格"):
                Pcol = c; Pheader_row = r
                break
        if Pcol is not None: break
    if Pcol is None:
        raise KeyError("找不到 P/Price/價格 欄位標題")

    # 4) 擷取 P series
    P_list = []
    for rr in range(Pheader_row+1, nrows):
        num = extract_num(df0.iat[rr, Pcol])
        if num is not None:
            P_list.append(num)
    if len(P_list) < 1:
        raise KeyError("P 欄下方找不到任何數值")

    P = np.array(P_list)
    P_mean, P_sd = P.mean(), P.std()

    # 5) 模擬 Profit = S * V * P
    N = 100_000
    profits = {}
    for m in (45, 55):
        sim = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S * V * sim

    # 6) 繪分布比較圖
    os.makedirs("static/results", exist_ok=True)
    fn_dist = "q3_dist.png"
    plt.figure(figsize=(8,5))
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.legend()
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_dist))
    plt.close()

    # 7) 計算靈敏度（Std）
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # 8) SD=8,10,12時計算 P(Profit>100)
    prob_gt100 = {}
    for sd in (8, 10, 12):
        sim = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof = S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

    # 9) 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(8,5))
    xs, ys = zip(*sorted(prob_gt100.items()))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("P SD")
    plt.ylabel("P(Profit>100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    # 10) 回傳
    return {
        "靈敏度（收益標準差）": sensitivity,
        "P(Profit>100) 機率": prob_gt100,
        "plots": {"分布比較圖": fn_dist, "機率變化圖": fn_prob}
    }
