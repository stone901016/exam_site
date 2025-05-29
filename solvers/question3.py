# solvers/question3.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_num(val):
    """從字串中擷取第一組數值（含千分位與小數點），回傳 float 或 None。"""
    if pd.isna(val):
        return None
    s = str(val)
    # 找到形如 1,234,567.89 或 123.45 或 123 的字串
    m = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", s)
    if not m:
        return None
    num = m.group(0).replace(",", "")
    try:
        return float(num)
    except:
        return None

def solve(file_path):
    """
    第三題：Profit.xls
    1) 嘗試 header=0~2，對每欄用正則提數值，找出至少三個含數值的欄位作 S, V, P
    2) P mean=45/55 時模擬 Profit 分布 & 靈敏度
    3) P mean 固定，SD=8,10,12時計算 P(Profit>100) 並繪折線圖
    """
    df = None
    numeric_cols = None

    # 逐層 header 嘗試
    for h in [0,1,2]:
        try:
            tmp = pd.read_excel(file_path, header=h)
        except Exception:
            continue
        # 為每欄擷取數值，並計算有效筆數
        counts = {}
        for col in tmp.columns:
            nums = tmp[col].apply(extract_num)
            counts[col] = nums.notna().sum()
        # 篩出「至少一筆數值」的欄位
        cand = [col for col, cnt in counts.items() if cnt>0]
        if len(cand)>=3:
            # 按照可擷取數量排序，取前三
            cand.sort(key=lambda c: counts[c], reverse=True)
            numeric_cols = cand[:3]
            df = tmp
            break

    if numeric_cols is None:
        raise KeyError("Profit.xls 數值欄位不足：無法找到至少三個可擷取數值的欄位")

    # 取欄並轉為純 float series
    S_col, V_col, P_col = numeric_cols
    S = df[S_col].apply(extract_num).dropna().astype(float)
    V = df[V_col].apply(extract_num).dropna().astype(float)
    P = df[P_col].apply(extract_num).dropna().astype(float)

    S_mean = float(S.mean())
    V_mean = float(V.mean())
    P_mean = float(P.mean())
    P_sd   = float(P.std())

    N = 100_000

    # (1) P mean = 45,55 時 Profit 模擬
    profits = {}
    for m in [45,55]:
        prof = S_mean * V_mean * norm.rvs(loc=m, scale=P_sd, size=N)
        profits[m] = prof

    # 繪分布比較圖
    fn_dist = "q3_dist.png"
    plt.figure()
    for m, prof in profits.items():
        plt.hist(prof, bins=50, alpha=0.6, label=f"mean={m}")
    plt.title("Profit 分布比較")
    plt.legend()
    plt.savefig(os.path.join("static","results",fn_dist))
    plt.close()

    # (1b) 計算靈敏度 (Std)
    sensitivity = {m: float(np.std(profits[m])) for m in profits}

    # (2) P mean 固定，SD=8/10/12 計算 P(Profit>100)
    prob_gt100 = {}
    for sd in [8,10,12]:
        prof = S_mean * V_mean * norm.rvs(loc=P_mean, scale=sd, size=N)
        prob_gt100[sd] = float(np.mean(prof>100))

    # 繪折線圖
    fn_prob = "q3_p_gt100.png"
    plt.figure()
    xs = list(prob_gt100.keys())
    ys = list(prob_gt100.values())
    plt.plot(xs, ys, marker='o')
    plt.title("P(Profit > 100) vs P SD")
    plt.xlabel("P SD")
    plt.ylabel("Probability")
    plt.savefig(os.path.join("static","results",fn_prob))
    plt.close()

    return {
        "靈敏度（Std）": sensitivity,
        "P(Profit>100) 機率": prob_gt100,
        "plots": {"dist": fn_dist, "p_gt100": fn_prob}
    }
