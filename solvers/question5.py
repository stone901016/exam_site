# solvers/question5.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve(file_path):
    """
    第 5 題（網格搜尋版，高精度設定）：
    2002 年底，60 歲，退休金 1,000,000 元。希望每年提領 50,000 以上（2002 現值），
    通貨膨脹率 3.12%。模擬 100,000 次，四種類別資產：LCS、SCS、CB、USGB，
    用網格搜尋找到：
      1. 固定提款 50,000 時，2033 年底「平均剩餘」最高的權重組合。
      2. 求出對每組權重，使得 2033 年底「平均剩餘」≈ 0 時，最大可提領 W*，
         並找出 W* 最大的那組權重。

    回傳字典格式：
      {
        "Part1_best_portfolio": "25%/12%/0%/63%",
        "Part1_best_avg_remaining": 1234567.0,
        "Part2_best_portfolio": "30%/10%/10%/50%",
        "Part2_best_Wstar": 47890.0,
        "plots": {
            "Part 1（固定提款 50,000）：平均剩餘長條圖": "q5_part1.png",
            "Part 2（最大 W*）：提領長條圖": "q5_part2.png"
        }
      }
    """

    # ---------------------------------------------------------------------
    # (A) 嘗試直接從 Excel 讀取 2003~2033 年度的「Total Return LCS、SCS、CB、USGB」
    use_manual = False
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        required_cols = [
            "Year",
            "Total Return Large Company Stocks",
            "Total Return Small Company Stocks",
            "Total Return Corporate Bonds",
            "Total Return U.S. Government Bonds"
        ]
        if all(col in df.columns for col in required_cols):
            df["Year"] = df["Year"].astype(int)
            df2 = df.loc[(df["Year"] >= 2003) & (df["Year"] <= 2033), required_cols].copy()
            if len(df2) == 31:
                # 直接將每年 multiplier（已經是 1.xxx 格式）
                lcs_returns_excel  = df2["Total Return Large Company Stocks"].values.astype(float)
                scs_returns_excel  = df2["Total Return Small Company Stocks"].values.astype(float)
                cb_returns_excel   = df2["Total Return Corporate Bonds"].values.astype(float)
                usgb_returns_excel = df2["Total Return U.S. Government Bonds"].values.astype(float)
            else:
                use_manual = True
        else:
            use_manual = True
    except Exception:
        use_manual = True

    # ---------------------------------------------------------------------
    # (B) 如果無法讀到 2003~2033，改用手動「1926~2002」歷史年化百分比 Bootstrap
    if not use_manual:
        lcs_returns = lcs_returns_excel.copy()
        scs_returns = scs_returns_excel.copy()
        cb_returns  = cb_returns_excel.copy()
        usgb_returns = usgb_returns_excel.copy()
    else:
        # ---------------------------------------------------------------------
        # （B1）請務必把下面 77 年的「% 數」跟您手上的 Excel 一字不差地填齊，
        #      否則 Bootstrap 出來的分布不正確！
        MANUAL_LCS_PERC = np.array([
            11.91, 36.74, 41.45, -8.04, -25.40, -44.45, -8.66, 54.28, -1.88, 46.67,
            33.74, -35.53, 30.27, -0.74, -9.72, -11.40, 20.57, 26.22, 20.36, 36.27,
            -8.66,  5.29,  5.39, 18.51, 32.20, 23.75, 18.64, -1.37, 52.59, 31.50,
             6.50, -10.96, 43.57, 12.45,  0.33, 27.26, -8.76, 22.72, 16.57, 12.47,
           -10.15, 24.05, 11.03, -8.41,  4.05, 14.24, 19.06, -14.71, -26.43, 37.23,
            23.91, -7.22,  6.53, 18.61, 32.45, -4.95, 21.75, 22.44,  6.36, 32.08,
            18.43,  5.28, 16.83, 31.41, -3.18, 30.60,  7.69,  9.93,  1.30, 37.57,
            23.07, 33.27, 28.58, 21.04, -9.11, -11.89, -22.10
        ], dtype=float)

        MANUAL_SCS_PERC = np.array([
            -4.32, 27.16, 42.36, -51.09, -41.92, -49.46,  2.78, 165.34, 24.67, 54.31,
             74.63, -55.36, 28.74,   0.13,  -8.48,  -11.04, 47.76, 94.08, 57.12, 77.93,
            -12.21,  -1.08,  -4.13, 20.65, 42.12,  8.60,   4.70,  -6.08, 62.86, 21.14,
              4.05,  -14.80, 67.76, 17.11,  -4.23, 31.28, -14.15, 17.88, 21.13, 39.71,
            -7.54,  93.48, 43.29, -28.66, -16.98, 17.47,   1.90, -35.72, -24.84, 61.18,
             56.09,  23.70, 22.87, 43.72, 37.61, 10.83,  27.73, 37.08, -10.35, 26.44,
              5.12, -11.62, 22.30,  9.27, -24.32, 47.44,  25.60, 20.64,  -0.12, 33.83,
             17.06,  22.57,  -4.93, 25.53,  -3.31, 10.87,  -17.43
        ], dtype=float)

        MANUAL_CB_PERC = np.array([
             5.96,  7.78,  0.95,  3.84,  7.10, -3.58, 11.36,  5.71, 12.00,  7.30,
             6.63,  1.47,  5.69,  4.94,  4.96,  1.86,  3.99,  3.85,  4.16,  5.46,
             0.94, -1.76,  3.61,  4.67,  0.58, -2.32,  2.73,  3.62,  5.14, -0.43,
            -5.96,  9.09, -2.96, -2.26, 11.42,  2.50,  7.38,  0.85,  4.64, -0.37,
             1.95, -6.18,  0.69, -7.31, 15.53, 14.24,  6.41,   1.27,   1.23, 11.57,
            14.86,   1.30,  -2.12,  2.42,   5.20,   1.18,  24.54,   2.87,  16.08, 31.39,
            21.91,  -1.46,   9.55,  17.86,   6.95,  19.14,   8.59,  14.33,  -6.47, 29.43,
             0.30,  14.02,  12.14,  -8.10,  16.57,   7.43,  16.56
        ], dtype=float)

        MANUAL_USGB_PERC = np.array([
             6.04,  5.60,  0.66,  5.11,  5.63, -4.48, 11.36,  0.70,  9.34,  6.33,
             4.78,  1.12,  5.80,  4.99,  3.69,  0.28,  2.32,  2.56,  2.20,  5.52,
             0.53, -0.46,  2.49,  3.66,  0.34, -1.07,  1.47,  3.50,  3.87, -0.82,
            -2.30,  7.71, -2.89, -1.30, 12.51,  1.68,  6.61,  1.55,  4.00,  1.00,
             4.49, -2.67,  2.25, -2.64, 14.30, 10.22,  4.88,  2.13,  5.36,  7.94,
            14.61,  0.61,  0.88,  2.38,  0.26,  5.80, 34.28,  4.50, 14.65, 24.98,
            18.96,  0.62,  7.47, 15.41,  8.32, 17.14,  7.50, 13.83, -5.78, 22.18,
             1.72, 10.64, 10.63, -3.44, 14.78,  6.49, 13.35
        ], dtype=float)

        if not (len(MANUAL_LCS_PERC)==77 and len(MANUAL_SCS_PERC)==77 
                and len(MANUAL_CB_PERC)==77 and len(MANUAL_USGB_PERC)==77):
            raise KeyError("第5題：手動填入的 LCS/SCS/CB/USGB 必須各有 77 筆 (1926~2002)。")

        # 轉成 multiplier (1 + %/100)
        lcs_returns  = 1.0 + (MANUAL_LCS_PERC  / 100.0)
        scs_returns  = 1.0 + (MANUAL_SCS_PERC  / 100.0)
        cb_returns   = 1.0 + (MANUAL_CB_PERC   / 100.0)
        usgb_returns = 1.0 + (MANUAL_USGB_PERC / 100.0)

    # ---------------------------------------------------------------------
    # (C) 共用參數設定
    Nsim = 100_000         # 現在改回 100,000 次模擬
    total_years = 31       # 2003–2033 共 31 年
    init_capital = 1_000_000  # 初始退休金 (2002 年底)
    inflation = 0.0312        # 年通膨率 3.12%
    fixed_withdraw = 50_000   # Part1 固定提款 50,000

    # 將四類資產的 multiplier 轉成 np.array，方便 index 隨機抽樣
    pool_lcs  = np.array(lcs_returns)
    pool_scs  = np.array(scs_returns)
    pool_cb   = np.array(cb_returns)
    pool_usgb = np.array(usgb_returns)

    rng = np.random.default_rng()


    def simulate_path(weights, W_withdraw):
        """
        模擬 Nsim 條路徑，四元權重 weights = [w_LCS, w_SCS, w_CB, w_USGB]，
        每年先本金配權，走到 31 年後輸出「2033 年底的剩餘資產 (np.array of shape (Nsim,))」。
        """
        w_lcs, w_scs, w_cb, w_usgb = weights

        # 建立 (Nsim, total_years+1) 矩陣，第一年的資產 = init_capital
        sim_mat = np.zeros((Nsim, total_years + 1), dtype=float)
        sim_mat[:, 0] = init_capital

        # 抽樣：若 pool 長度剛好=31，直接 tile；否則 Bootstrap
        if pool_lcs.shape[0] == total_years:
            draw_lcs  = np.tile(pool_lcs,  (Nsim, 1))
            draw_scs  = np.tile(pool_scs,  (Nsim, 1))
            draw_cb   = np.tile(pool_cb,   (Nsim, 1))
            draw_usgb = np.tile(pool_usgb, (Nsim, 1))
        else:
            draw_lcs  = rng.choice(pool_lcs,  size=(Nsim, total_years), replace=True)
            draw_scs  = rng.choice(pool_scs,  size=(Nsim, total_years), replace=True)
            draw_cb   = rng.choice(pool_cb,   size=(Nsim, total_years), replace=True)
            draw_usgb = rng.choice(pool_usgb, size=(Nsim, total_years), replace=True)

        # 逐年 (t=1~31) 計算
        for t in range(1, total_years + 1):
            real_withdraw = W_withdraw * ((1.0 + inflation) ** (t - 1))

            prev = sim_mat[:, t-1]
            r1   = draw_lcs[:, t-1]
            r2   = draw_scs[:, t-1]
            r3   = draw_cb[:, t-1]
            r4   = draw_usgb[:, t-1]

            new_val = prev * ( w_lcs * r1
                              + w_scs * r2
                              + w_cb  * r3
                              + w_usgb* r4 ) - real_withdraw

            new_val[new_val < 0] = 0.0
            sim_mat[:, t] = new_val

        return sim_mat[:, -1]


    # ---------------------------------------------------------------------
    # (D) 產生「所有可能的四元權重」網格（步進 0.01，即 1% 一格）
    step = 0.01
    grid_values = np.arange(0.0, 1.0 + 1e-9, step)  # [0.00, 0.01, 0.02, …, 1.00]

    weight_grid = []
    for w1 in grid_values:
        for w2 in grid_values:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            for w3 in grid_values:
                if w1 + w2 + w3 > 1.0 + 1e-9:
                    continue
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < -1e-9:
                    continue
                # 四元 tuple，取到小數第四位
                weight_grid.append((round(w1, 4), round(w2, 4), round(w3, 4), round(w4, 4)))

    # 語意上可能猜測總共組合數為 C(101+3,3) ≈ 176,851 組（實際看範圍）
    # print("Total weight combinations:", len(weight_grid))

    # ---------------------------------------------------------------------
    # (E) Part1：固定提款 50,000 時，遍歷權重網格，計算 2033 年底的平均剩餘
    best_avg_rem = -1.0
    best_w_part1 = None
    part1_records = {}

    for idx, w in enumerate(weight_grid):
        avg_rem = np.mean(simulate_path(w, fixed_withdraw))
        part1_records[w] = avg_rem
        if avg_rem > best_avg_rem:
            best_avg_rem = avg_rem
            best_w_part1 = w
        # （可選）進度印出
        if (idx + 1) % 20_000 == 0:
            print(f"[Part1] 已處理 {idx+1} / {len(weight_grid)} 組權重")

    # ---------------------------------------------------------------------
    # (F) Part2：對每組 w，二分搜尋最大可提領 W*，使 2033 年底平均剩餘 ≈ 0
    def find_Wstar(weights):
        lo, hi = 0.0, 200_000.0
        tol = 1_000.0    # 平均剩餘誤差 ±1,000
        best_mid = 0.0
        best_avg = None

        for _ in range(30):  # 迭代 30 次可達非常高精度
            mid = (lo + hi) / 2.0
            rems = simulate_path(weights, mid)
            avg_rem = np.mean(rems)

            if avg_rem > tol:
                lo = mid
                best_mid = mid
                best_avg = avg_rem
            else:
                hi = mid

        return best_mid, best_avg

    best_Wstar = -1.0
    best_w_part2 = None
    part2_records = {}

    for idx, w in enumerate(weight_grid):
        Wstar, avg_rem_at_star = find_Wstar(w)
        part2_records[w] = (Wstar, avg_rem_at_star)
        if Wstar > best_Wstar:
            best_Wstar = Wstar
            best_w_part2 = w
        if (idx + 1) % 20_000 == 0:
            print(f"[Part2] 已處理 {idx+1} / {len(weight_grid)} 組權重")

    # ---------------------------------------------------------------------
    # (G) 繪圖：Part1 & Part2 長條圖
    os.makedirs("static/results", exist_ok=True)

    # 1) Part1：2033 年底平均剩餘長條
    fn1 = "q5_part1.png"
    plt.figure(figsize=(12, 6))
    combos = [f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%" 
              for w in weight_grid]
    values1 = [part1_records[w] for w in weight_grid]
    bars = plt.bar(combos, values1, color='tab:blue')
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("2033 年底平均剩餘 (元)")
    plt.title("Part 1：固定提款 50,000 時，各權重組合 2033 年底平均剩餘")
    idx_best1 = weight_grid.index(best_w_part1)
    bars[idx_best1].set_color('tab:red')
    plt.text(idx_best1, values1[idx_best1] * 1.02,
             f"{int(values1[idx_best1]):,}", 
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn1))
    plt.close()

    # 2) Part2：最大可提領 W* 長條
    fn2 = "q5_part2.png"
    plt.figure(figsize=(12, 6))
    values2 = [part2_records[w][0] for w in weight_grid]
    bars2 = plt.bar(combos, values2, color='tab:orange')
    plt.xticks(rotation=90, fontsize=6)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("最大可提領 W* (2002 現值, 元)")
    plt.title("Part 2：各權重組合的最大 W* (2033 年底平均剩餘 ≈ 0)")
    idx_best2 = weight_grid.index(best_w_part2)
    bars2[idx_best2].set_color('tab:red')
    plt.text(idx_best2, values2[idx_best2] * 1.02,
             f"{int(values2[idx_best2]):,}",
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn2))
    plt.close()

    # ---------------------------------------------------------------------
    # (H) 組合並回傳
    return {
        # Part1
        "Part1_best_portfolio": f"{int(best_w_part1[0]*100)}%/"
                                f"{int(best_w_part1[1]*100)}%/"
                                f"{int(best_w_part1[2]*100)}%/"
                                f"{int(best_w_part1[3]*100)}%",
        "Part1_best_avg_remaining": best_avg_rem,

        # Part2
        "Part2_best_portfolio": f"{int(best_w_part2[0]*100)}%/"
                                f"{int(best_w_part2[1]*100)}%/"
                                f"{int(best_w_part2[2]*100)}%/"
                                f"{int(best_w_part2[3]*100)}%",
        "Part2_best_Wstar": best_Wstar,

        # 圖檔
        "plots": {
            "Part 1（固定提款 50,000）：平均剩餘長條圖": fn1,
            "Part 2（最大 W*）：提領長條圖": fn2
        }
    }
