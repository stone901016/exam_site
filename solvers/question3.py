# solvers/question3.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """從任意格式字串擷取第一組數值（含小數與負號）"""
    if pd.isna(val): 
        return None
    s = str(val)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None

def solve(file_path):
    # 1. 全表讀取（不用 header）
    df0 = pd.read_excel(file_path, header=None, engine='xlrd')
    nrows, ncols = df0.shape

    # 2. 對每一欄計算可轉數值的筆數
    counts = []
    for c in range(ncols):
        cnt = sum(1 for r in range(nrows) if extract_num(df0.iat[r, c]) is not None)
        if cnt > 0:
            counts.append((c, cnt))
    if len(counts) < 3:
        raise KeyError(f"Profit.xls 數值欄位不足：只有找到 {len(counts)} 個可轉數值的欄")
    # 取最多筆的前三欄
    counts.sort(key=lambda x: x[1], reverse=True)
    col_S, col_V, col_P = [c for c, _ in counts[:3]]

    # 3. 解析 S, V, P
    S_vals = [extract_num(df0.iat[r, col_S]) for r in range(nrows)]
    V_vals = [extract_num(df0.iat[r, col_V]) for r in range(nrows)]
    P_vals = [extract_num(df0.iat[r, col_P]) for r in range(nrows)]
    S_arr = np.array([v for v in S_vals if v is not None])
    V_arr = np.array([v for v in V_vals if v is not None])
    P_arr = np.array([v for v in P_vals if v is not None])
    if S_arr.size == 0 or V_arr.size == 0 or P_arr.size == 0:
        raise KeyError("無法從前三欄取得有效的 S/V/P 數值")
    S, V = float(S_arr.mean()), float(V_arr.mean())
    P_mean, P_sd = float(P_arr.mean()), float(P_arr.std())

    N = 100_000
    # 4. P mean=45 & 55 模擬 Profit
    profits = {}
    for m in (45, 55):
        sim = norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = S * V * sim

    # 5. 繪分布比較圖
    os.makedirs("static/results", exist_ok=True)
    fn_dist = "q3_dist.png"
    plt.figure(figsize=(16,10))
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.legend()
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_dist))
    plt.close()

    # 6. 計算靈敏度 (Std)
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # 7. SD=8,10,12 時計算 P(Profit>100)
    prob_gt100 = {}
    for sd in (8, 10, 12):
        sim = norm.rvs(loc=P_mean, scale=sd, size=N)
        prof = S * V * sim
        prob_gt100[sd] = float((prof > 100).mean())

    # 8. 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure(figsize=(16,10))
    xs, ys = zip(*sorted(prob_gt100.items()))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("P SD")
    plt.ylabel("P(Profit > 100)")
    plt.tight_layout()
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    # 9. 回傳結果
    return {
        "靈敏度（收益標準差）": sensitivity,
        "P(Profit>100) 機率": prob_gt100,
        "plots": {
            "分布比較圖": fn_dist,
            "機率變化圖": fn_prob
        }
    }
