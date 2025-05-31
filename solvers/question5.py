# solvers/question5.py

import os
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path=None):
    """
    第 5 題：
    - 投資人 2002 年底 60 歲，退休金 1,000,000 元。
    - 希望每年提領 50,000 元（以 2002 年現值計），通貨膨脹率 3.12%。
    - 歷史報酬率取 1926–2002 共 77 年，四類資產：
        ・LCS (Large Company Stocks)
        ・SCS (Small Company Stocks)
        ・CB  (Corporate Bonds)
        ・USGB(U.S. Government Bonds)
      資料直接硬碼在程式中（％），之後轉換為 multiplier，長度 77。
    - 模擬次數共 100 000，使用兩階段網格搜尋：
      1) 第一階段：step=0.05, Nsim=20 000 → 找到「固定提款 50k 時，2033 底剩餘平均最大」前 20
                          以及「最大 W*（使 2033 底平均剩餘 ≈ 0）」前 20
      2) 第二階段：對上述最多 40 組候選權重，用 Nsim=100 000 精細模擬→最終輸出答案與圖表。
    """

    # ============================================================
    # (A) 硬碼：1926–2002 共 77 年的歷史報酬率百分比（％），
    #     先定義列表，再轉成 numpy array，最後再加 1 變 multiplier。
    #
    #     來源即以下表格（百分比），順序從 1926 年到 2002 年：
    #
    # YEAR | LCS    | SCS    | CB    | USGB
    # 1926 | 11.91  | -4.32  | 5.96  | 6.04
    # 1927 | 36.74  | 27.16  | 7.78  | 5.60
    # 1928 | 41.45  | 42.36  | 0.95  | 0.66
    # 1929 | -8.04  | -51.09 | 3.84  | 5.11
    # 1930 | -25.40 | -41.92 | 7.10  | 5.63
    # 1931 | -44.45 | -49.46 | -3.58 | -4.48
    # 1932 | -8.66  | 2.78   | 11.36 | 11.36
    # 1933 | 54.28  | 165.34 | 5.71  | 0.70
    # 1934 | -1.88  | 24.67  | 12.00 | 9.34
    # 1935 | 46.67  | 54.31  | 7.30  | 6.33
    # 1936 | 33.74  | 74.63  | 6.63  | 4.78
    # 1937 | -35.53 | -55.36 | 1.47  | 1.12
    # 1938 | 30.27  | 28.74  | 5.69  | 5.80
    # 1939 | -0.74  | 0.13   | 4.94  | 4.99
    # 1940 | -9.72  | -8.48  | 4.96  | 3.69
    # 1941 | -11.40 | -11.04 | 1.86  | 0.28
    # 1942 | 20.57  | 47.76  | 3.99  | 2.32
    # 1943 | 26.22  | 94.08  | 3.85  | 2.56
    # 1944 | 20.36  | 57.12  | 4.16  | 2.20
    # 1945 | 36.27  | 77.93  | 5.46  | 5.52
    # 1946 | -8.66  | -12.21 | 0.94  | 0.53
    # 1947 | 5.29   | -1.08  | -1.76 | -0.46
    # 1948 | 5.39   | -4.13  | 3.61  | 2.49
    # 1949 | 18.51  | 20.65  | 4.67  | 3.66
    # 1950 | 32.20  | 42.12  | 0.58  | 0.34
    # 1951 | 23.75  | 8.60   | -2.32 | -1.07
    # 1952 | 18.64  | 4.70   | 2.73  | 1.47
    # 1953 | -1.37  | -6.08  | 3.62  | 3.50
    # 1954 | 52.59  | 62.86  | 5.14  | 3.87
    # 1955 | 31.50  | 21.14  | -0.43 | -0.82
    # 1956 | 6.50   | 4.05   | -5.96 | -2.30
    # 1957 | -10.96 | -14.80 | 9.09  | 7.71
    # 1958 | 43.57  | 67.76  | -2.96 | -2.89
    # 1959 | 12.45  | 17.11  | -2.26 | -1.30
    # 1960 | 0.33   | -4.23  | 11.42 | 12.51
    # 1961 | 27.26  | 31.28  | 2.50  | 1.68
    # 1962 | -8.76  | -14.15 | 7.38  | 6.61
    # 1963 | 22.72  | 17.88  | 0.85  | 1.55
    # 1964 | 16.57  | 21.13  | 4.64  | 4.00
    # 1965 | 12.47  | 39.71  | -0.37 | 1.00
    # 1966 | -10.15 | -7.54  | 1.95  | 4.49
    # 1967 | 24.05  | 93.48  | -6.18 | -2.67
    # 1968 | 11.03  | 43.29  | 0.69  | 2.25
    # 1969 | -8.41  | -28.66 | -7.31 | -2.64
    # 1970 | 4.05   | -16.98 | 15.53 | 14.30
    # 1971 | 14.24  | 17.47  | 14.24 | 10.22
    # 1972 | 19.06  | 1.90   | 6.41  | 4.88
    # 1973 | -14.71 | -35.72 | 1.27  | 2.13
    # 1974 | -26.43 | -24.84 | 1.23  | 5.36
    # 1975 | 37.23  | 61.18  | 11.57 | 7.94
    # 1976 | 23.91  | 56.09  | 14.86 | 14.61
    # 1977 | -7.22  | 23.70  | 1.30  | 0.61
    # 1978 | 6.53   | 22.87  | -2.12 | 0.88
    # 1979 | 18.61  | 43.72  | 2.42  | 2.38
    # 1980 | 32.45  | 37.61  | 5.20  | 0.26
    # 1981 | -4.95  | 10.83  | 1.18  | 5.80
    # 1982 | 21.75  | 27.73  | 24.54 | 34.28
    # 1983 | 22.44  | 37.08  | 2.87  | 4.50
    # 1984 | 6.36   | -10.35 | 16.08 | 14.65
    # 1985 | 32.08  | 26.44  | 31.39 | 24.98
    # 1986 | 18.43  | 5.12   | 21.91 | 18.96
    # 1987 | 5.28   | -11.62 | -1.46 | 0.62
    # 1988 | 16.83  | 22.30  | 9.55  | 7.47
    # 1989 | 31.41  | 9.27   | 17.86 | 15.41
    # 1990 | -3.18  | -24.32 | 6.95  | 8.32
    # 1991 | 30.60  | 47.44  | 19.14 | 17.14
    # 1992 | 7.69   | 25.60  | 8.59  | 7.50
    # 1993 | 9.93   | 20.64  | 14.33 | 13.83
    # 1994 | 1.30   | -0.12  | -6.47 | -5.78
    # 1995 | 37.57  | 33.83  | 29.43 | 22.18
    # 1996 | 23.07  | 17.06  | 0.30  | 1.72
    # 1997 | 33.27  | 22.57  | 14.02 | 10.64
    # 1998 | 28.58  | -4.93  | 12.14 | 10.63
    # 1999 | 21.04  | 25.53  | -8.10 | -3.44
    # 2000 | -9.11  | -3.31  | 16.57 | 14.78
    # 2001 | -11.89 | 10.87  | 7.43  | 6.49
    # 2002 | -22.10 | -17.43 | 16.56 | 13.35
    #
    # total_years = 2002 − 1926 + 1 = 77 年
    # ============================================================
    hist_lcs_pct  = [
        11.91, 36.74, 41.45, -8.04, -25.40, -44.45, -8.66, 54.28, -1.88, 46.67,
        33.74, -35.53, 30.27, -0.74, -9.72, -11.40, 20.57, 26.22, 20.36, 36.27,
        -8.66,  5.29,  5.39, 18.51, 32.20, 23.75, 18.64, -1.37, 52.59, 31.50,
        6.50, -10.96, 43.57, 12.45,  0.33, 27.26, -8.76, 22.72, 16.57, 12.47,
        -10.15, 24.05, 11.03, -8.41,  4.05, 14.24, 19.06, -14.71, -26.43, 37.23,
        23.91, -7.22,  6.53, 18.61, 32.45, -4.95, 21.75, 22.44,  6.36, 32.08,
        18.43,  5.28, 16.83, 31.41, -3.18, 30.60,  7.69,  9.93,  1.30, 37.57,
        23.07, 33.27, 28.58, 21.04, -9.11, -11.89, -22.10
    ]
    hist_scs_pct  = [
        -4.32, 27.16, 42.36, -51.09, -41.92, -49.46,  2.78,165.34, 24.67, 54.31,
         74.63, -55.36, 28.74,  0.13,  -8.48, -11.04, 47.76, 94.08, 57.12, 77.93,
        -12.21, -1.08, -4.13, 20.65, 42.12,  8.60,  4.70, -6.08, 62.86, 21.14,
          4.05, -14.80, 67.76, 17.11,  -4.23, 31.28, -14.15, 17.88, 21.13, 39.71,
         -7.54, 93.48, 43.29, -28.66, -16.98, 17.47,  1.90, -35.72, -24.84, 61.18,
         56.09, 23.70, 22.87, 43.72, 37.61, 10.83, 27.73, 37.08, -10.35, 26.44,
          5.12, -11.62, 22.30,  9.27, -24.32, 47.44, 25.60, 20.64, -0.12, 33.83,
         17.06, 22.57, -4.93, 25.53,  -3.31, 10.87, -17.43
    ]
    hist_cb_pct   = [
         5.96,  7.78,  0.95,  3.84,  7.10, -3.58, 11.36,  5.71, 12.00,  7.30,
          6.63,  1.47,  5.69,  4.94,  4.96,  1.86,  3.99,  3.85,  4.16,  5.46,
          0.94, -1.76,  3.61,  4.67,  0.58, -2.32,  2.73,  3.62,  5.14, -0.43,
         -5.96,  9.09, -2.96, -2.26, 11.42,  2.50,  7.38,  0.85,  4.64, -0.37,
          1.95, -6.18,  0.69, -7.31, 15.53, 14.24,  6.41,  1.27,  1.23, 11.57,
         14.86,  1.30, -2.12,  2.42,  5.20,  1.18, 24.54,  2.87, 16.08, 31.39,
         21.91, -1.46,  9.55, 17.86,  6.95, 19.14,  8.59, 14.33, -6.47, 29.43,
          0.30, 14.02, 12.14, -8.10, 16.57,  7.43, 16.56
    ]
    hist_usgb_pct = [
         6.04,  5.60,  0.66,  5.11,  5.63, -4.48, 11.36,  0.70,  9.34,  6.33,
          4.78,  1.12,  5.80,  4.99,  3.69,  0.28,  2.32,  2.56,  2.20,  5.52,
          0.53, -0.46,  2.49,  3.66,  0.34, -1.07,  1.47,  3.50,  3.87, -0.82,
         -2.30,  7.71, -2.89, -1.30, 12.51,  1.68,  6.61,  1.55,  4.00,  1.00,
          4.49, -2.67,  2.25, -2.64, 14.30, 10.22,  4.88,  2.13,  5.36,  7.94,
         14.61,  0.61,  0.88,  2.38,  0.26,  5.80, 34.28,  4.50, 14.65, 24.98,
         18.96,  0.62,  7.47, 15.41,  8.32, 17.14,  7.50, 13.83, -5.78, 22.18,
          1.72, 10.64, 10.63, -3.44, 14.78,  6.49, 13.35
    ]

    # 把百分比 → multiplier (例如 11.91% → 1.1191；−4.32% → 0.9568)
    pool_lcs   = np.array(hist_lcs_pct)   / 100.0 + 1.0   # shape = (77,)
    pool_scs   = np.array(hist_scs_pct)   / 100.0 + 1.0
    pool_cb    = np.array(hist_cb_pct)    / 100.0 + 1.0
    pool_usgb  = np.array(hist_usgb_pct)  / 100.0 + 1.0

    total_years     = 31          # 2003 年到 2033 年需要模擬的「退休後持續投資」期數
    init_capital    = 1_000_000   # 2002 年底初始資金
    inflation_rate  = 0.0312      # 每年通膨率 3.12%
    fixed_withdraw  = 50_000      # 每年提領 50,000 元 (2002 年現值)

    rng = np.random.default_rng()

    # ============================================================
    # (B) 定義「一次模擬 31 年後剩餘資金」的函式
    def simulate_path(weights, W_withdraw, Nsim_local):
        """
        - weights: tuple (w_lcs, w_scs, w_cb, w_usgb)，四類資產權重
        - W_withdraw: 每年要提領多少 (2002 現值)
        - Nsim_local: 模擬次數

        流程：
        1) 先一次性「有放回」從 77 年的 pool_xxxx 中抽樣，size = (Nsim_local, total_years)
           draw1 = rng.choice(pool_lcs,   size=(Nsim_local, total_years), replace=True)
           draw2 = rng.choice(pool_scs,   size=(Nsim_local, total_years), replace=True)
           draw3 = rng.choice(pool_cb,    size=(Nsim_local, total_years), replace=True)
           draw4 = rng.choice(pool_usgb,  size=(Nsim_local, total_years), replace=True)
        2) 建立 sim_mat = zeros((Nsim_local, total_years+1))，第 0 年＝init_capital
        3) 對 t=1..31:
             real_withdraw = W_withdraw * (1+inflation_rate)^(t-1)
             prev = sim_mat[:, t-1]
             newv = prev * (
                       w_lcs  * draw1[:, t-1] +
                       w_scs  * draw2[:, t-1] +
                       w_cb   * draw3[:, t-1] +
                       w_usgb * draw4[:, t-1]
                   ) - real_withdraw
             newv[newv < 0] = 0
             sim_mat[:, t] = newv
        回傳 sim_mat[:, -1] → (Nsim_local,)，即第 31 年底剩餘
        """
        w_lcs, w_scs, w_cb, w_usgb = weights

        draw1 = rng.choice(pool_lcs,   size=(Nsim_local, total_years), replace=True)
        draw2 = rng.choice(pool_scs,   size=(Nsim_local, total_years), replace=True)
        draw3 = rng.choice(pool_cb,    size=(Nsim_local, total_years), replace=True)
        draw4 = rng.choice(pool_usgb,  size=(Nsim_local, total_years), replace=True)

        sim_mat = np.zeros((Nsim_local, total_years+1), dtype=np.float64)
        sim_mat[:, 0] = init_capital

        for t in range(1, total_years+1):
            real_withdraw = W_withdraw * ((1.0 + inflation_rate) ** (t-1))
            prev = sim_mat[:, t-1]
            newv = prev * (
                    w_lcs  * draw1[:, t-1]
                  + w_scs  * draw2[:, t-1]
                  + w_cb   * draw3[:, t-1]
                  + w_usgb * draw4[:, t-1]
                ) - real_withdraw
            newv[newv < 0.0] = 0.0
            sim_mat[:, t] = newv

        return sim_mat[:, -1]

    # ============================================================
    # (C) 第一階段：粗網格篩選
    step_first  = 0.05
    Nsim_first  = 20_000

    grid_vals = np.arange(0.0, 1.0 + 1e-9, step_first)
    weight_grid1 = []
    for w1 in grid_vals:
        for w2 in grid_vals:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            for w3 in grid_vals:
                if w1 + w2 + w3 > 1.0 + 1e-9:
                    continue
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < -1e-9:
                    continue
                weight_grid1.append((round(w1,4), round(w2,4), round(w3,4), round(w4,4)))

    # --- Part1：固定提款 50k，算 2033 年底剩餘平均，取前 20 組權重 ---
    part1_scores1 = []
    for weights in weight_grid1:
        rems = simulate_path(weights, fixed_withdraw, Nsim_first)
        avg_rem = rems.mean()
        part1_scores1.append((avg_rem, weights))
    part1_scores1.sort(key=lambda x: x[0], reverse=True)
    top20_part1 = part1_scores1[:20]  # (avg_rem, weights)

    # --- Part2：讓 2033 年底平均剩餘 ≈ 0 → 找 W*，取前 20 組權重 ---
    def find_Wstar_small(weights):
        lo, hi = 0.0, 200_000.0
        tol = 5_000.0
        mid_best = 0.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            rems2 = simulate_path(weights, mid, Nsim_first)
            avg_rem2 = rems2.mean()
            if avg_rem2 > tol:
                lo = mid
                mid_best = mid
            else:
                hi = mid
        return mid_best

    part2_scores1 = []
    for weights in weight_grid1:
        W_approx = find_Wstar_small(weights)
        part2_scores1.append((W_approx, weights))
    part2_scores1.sort(key=lambda x: x[0], reverse=True)
    top20_part2 = part2_scores1[:20]  # (W_approx, weights)

    # ============================================================
    # (D) 第二階段：精細模擬（只對 top20_part1 & top20_part2 合併候選做 Nsim_final = 100000）
    candidates = set([w for _, w in top20_part1] + [w for _, w in top20_part2])
    candidates = list(candidates)

    best_avg_rem = -1.0
    best_w1      = None
    best_Wstar   = -1.0
    best_w2      = None

    Part1_final = []
    Part2_final = []

    Nsim_final = 100_000

    for weights in candidates:
        # Part1 精細模擬：固定提款 50k
        rems1 = simulate_path(weights, fixed_withdraw, Nsim_final)
        avg_rem1 = rems1.mean()
        Part1_final.append((avg_rem1, weights))
        if avg_rem1 > best_avg_rem:
            best_avg_rem = avg_rem1
            best_w1      = weights

        # Part2 精細模擬：二分搜尋 W* ≈ 使平均剩餘接近 0
        lo, hi = 0.0, 300_000.0
        mid_best = 0.0
        tol = 1_000.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            rems2 = simulate_path(weights, mid, Nsim_final)
            avg_rem2 = rems2.mean()
            if avg_rem2 > tol:
                lo = mid
                mid_best = mid
            else:
                hi = mid
        Part2_final.append((mid_best, weights))
        if mid_best > best_Wstar:
            best_Wstar = mid_best
            best_w2    = weights

    # ============================================================
    # (E) 繪圖並存檔
    os.makedirs("static/results", exist_ok=True)

    # 把候選權重組合轉成「LCS/SCS/CB/USGB 百分比字串」
    combos_f = [
        f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%"
        for w in candidates
    ]

    # Part1 長條圖：2033 年底「平均剩餘」(顏色標紅最優)
    values1_f = [v for v, w in Part1_final]
    fn1 = "q5_part1.png"
    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(combos_f, values1_f, color='tab:blue')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("2033 年底平均剩餘 (元)")
    plt.title("Part1：固定提款 50,000 元，候選權重組合 2033 年度平均剩餘")
    # 標紅最優
    best_str1 = f"{int(best_w1[0]*100)}%/" \
                f"{int(best_w1[1]*100)}%/" \
                f"{int(best_w1[2]*100)}%/" \
                f"{int(best_w1[3]*100)}%"
    idx_best1 = combos_f.index(best_str1)
    bars1[idx_best1].set_color('tab:red')
    plt.text(idx_best1,
             values1_f[idx_best1] * 1.02,
             f"{int(values1_f[idx_best1]):,}",
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn1))
    plt.close()

    # Part2 長條圖：最大 W* 值 (顏色標紅最優)
    values2_f = [v for v, w in Part2_final]
    fn2 = "q5_part2.png"
    plt.figure(figsize=(14, 6))
    bars2 = plt.bar(combos_f, values2_f, color='tab:orange')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("最大可提領 W* (元)")
    plt.title("Part2：最大 W* 值 (讓 2033 年底平均剩餘 ≈ 0)")
    # 標紅最優
    best_str2 = f"{int(best_w2[0]*100)}%/" \
                f"{int(best_w2[1]*100)}%/" \
                f"{int(best_w2[2]*100)}%/" \
                f"{int(best_w2[3]*100)}%"
    idx_best2 = combos_f.index(best_str2)
    bars2[idx_best2].set_color('tab:red')
    plt.text(idx_best2,
             values2_f[idx_best2] * 1.02,
             f"{int(values2_f[idx_best2]):,}",
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn2))
    plt.close()

    # ============================================================
    # (F) 回傳結果字典
    return {
        "Part1_best_portfolio":      f"{int(best_w1[0]*100)}%/"
                                     f"{int(best_w1[1]*100)}%/"
                                     f"{int(best_w1[2]*100)}%/"
                                     f"{int(best_w1[3]*100)}%",
        "Part1_best_avg_remaining":  best_avg_rem,
        "Part2_best_portfolio":      f"{int(best_w2[0]*100)}%/"
                                     f"{int(best_w2[1]*100)}%/"
                                     f"{int(best_w2[2]*100)}%/"
                                     f"{int(best_w2[3]*100)}%",
        "Part2_best_Wstar":          best_Wstar,
        "plots": {
            "Part1_固定提款_年50k_平均剩餘長條圖": fn1,
            "Part2_最大可提領_W*_長條圖":     fn2
        }
    }
