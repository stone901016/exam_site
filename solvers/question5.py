# solvers/question5.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve(file_path):
    """
    第 5 題：
    2002 年底，60 歲，退休金 1,000,000 元。希望每年提領 50,000 以上（2002 現值），
    通貨膨脹率 3.12%。模擬 100,000 次，四種投資組合 (各含 4 類資產：LCS、SCS、CB、USGB)，
    Part1：固定提款 50,000，模擬到 2033 年底各組合平均剩餘，找最佳投資組合。
    Part2：對每個組合，用二分搜尋找可提款 W*（2002 現值），使 2033 年底平均剩餘 ≈ 0，並找最佳組合。

    資料主要使用 sustainableRetirementWithdrawls.xls，但若 Excel 拿不到 2003~2033 年度的
    Total Return 欄位，就退回使用 1926–2002 年的 LCS、SCS、CB、USGB 歷史報酬率為
    母本 (bootstrap)。
    """

    # ---------------------------------------------------------------------
    # 嘗試從 Excel 擷取 2003–2033 年的 Total Return Stocks / Bonds
    use_manual = False
    try:
        df = pd.read_excel(file_path, sheet_name=0)

        # 如果能夠讀到 2003~2033 的「Total Returns Stocks」和「Total Returns Bonds」，
        # 但我們現在想要四類資產：LCS、SCS、CB、USGB，Presumably Excel 裡至少能給
        # 這四個欄位（或是 Total Return Stocks/ Bonds 就對應到 LCS＋SCS / CB＋USGB）。
        # 以下示範如果 Excel 有「Year、Total Returns Large Company Stocks、
        # Total Returns Small Company Stocks、Total Returns Corporate Bonds、
        # Total Returns U.S. Government Bonds」這四個欄位，就直接用：
        #
        #    "Total Return Large Company Stocks" => LCS
        #    "Total Return Small Company Stocks" => SCS
        #    "Total Return Corporate Bonds"      => CB
        #    "Total Return U.S. Government Bonds" => USGB
        #
        # 如果缺少其中任何一個，就回退到 use_manual = True。
        #
        required_cols = [
            "Year",
            "Total Return Large Company Stocks",
            "Total Return Small Company Stocks",
            "Total Return Corporate Bonds",
            "Total Return U.S. Government Bonds"
        ]
        if all(col in df.columns for col in required_cols):
            df["Year"] = df["Year"].astype(int)
            retires = df.loc[(df["Year"] >= 2003) & (df["Year"] <= 2033), required_cols].copy()

            if len(retires) == 31:
                # 直接拿到 2003~2033 的每年 multiplier，例如 1.2980、1.0696、1.0168 etc.
                lcs_returns_excel  = retires["Total Return Large Company Stocks"].values.astype(float)
                scs_returns_excel  = retires["Total Return Small Company Stocks"].values.astype(float)
                cb_returns_excel   = retires["Total Return Corporate Bonds"].values.astype(float)
                usgb_returns_excel = retires["Total Return U.S. Government Bonds"].values.astype(float)
            else:
                use_manual = True
        else:
            use_manual = True

    except Exception:
        # 任何讀取失敗都回退到手動方式
        use_manual = True

    if not use_manual:
        # 如果從 Excel 成功拿到 31 年的回報，直接使用
        lcs_returns = lcs_returns_excel.copy()
        scs_returns = scs_returns_excel.copy()
        cb_returns  = cb_returns_excel.copy()
        usgb_returns = usgb_returns_excel.copy()

    else:
        # ================================================================
        # 回退手動：使用者提供的 1926–2002 年 LCS、SCS、CB、USGB 年化百分比 (%)
        #
        # 以下一定要對照您提供的「1926~2002」那張表，把 LCS、SCS、CB、USGB
        # 這四列 77 個數字逐年填齊。程式裡示範已經把您貼的表格完整打進來，請務必核對一致！
        # ================================================================
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

        # 確認都是 77 年
        if not (len(MANUAL_LCS_PERC) == len(MANUAL_SCS_PERC) == len(MANUAL_CB_PERC) == len(MANUAL_USGB_PERC) == 77):
            raise KeyError("第5題：手動填入的 LCS/SCS/CB/USGB 資料筆數必須各為 77 年 (1926~2002)。")

        # 轉成 multiplier (1 + 百分比/100)
        lcs_returns = 1.0 + (MANUAL_LCS_PERC / 100.0)
        scs_returns = 1.0 + (MANUAL_SCS_PERC / 100.0)
        cb_returns  = 1.0 + (MANUAL_CB_PERC  / 100.0)
        usgb_returns = 1.0 + (MANUAL_USGB_PERC/ 100.0)

    # ------------------------------------------------------------------------------
    # 到這裡：lcs_returns、scs_returns、cb_returns、usgb_returns
    #   或長度為 31（直接從 Excel 讀取 2003~2033），
    #   或長度為 77（從 1926~2002 Bootstrap）。
    # ------------------------------------------------------------------------------
    # 共用參數
    Nsim = 100_000            # 蒙地卡羅路徑數
    total_years = 31          # 2003–2033 共 31 年
    init_capital = 1_000_000  # 初始退休金
    inflation = 0.0312        # 年通膨率 3.12%

    # ------------------------------------------------
    # 【請務必在此自行定義「四種投資組合」的四元權重】
    #
    # weight_list 中包含 4 個元素，每個元素都是一個 list/tuple，內含 4 個數字：
    #   [w_LCS, w_SCS, w_CB, w_USGB]，且其總和約等於 1.0。
    #
    # 範例（請依照題目需求將下列四組示範值改成您真正想要的四種組合）：
    #
    #   1. [0.25, 0.12, 0.00, 0.63] → LCS 25%、SCS 12%、CB 0%、USGB 63%
    #   2. [0.30, 0.10, 0.10, 0.50] → LCS 30%、SCS 10%、CB 10%、USGB 50%
    #   3. [0.40, 0.20, 0.10, 0.30] → LCS 40%、SCS 20%、CB 10%、USGB 30%
    #   4. [0.60, 0.20, 0.10, 0.10] → LCS 60%、SCS 20%、CB 10%、USGB 10%
    #
    # 如果您有不同的四種 mix，請直接改 weight_list：
    weight_list = [
        [0.25, 0.12, 0.00, 0.63],
        [0.30, 0.10, 0.10, 0.50],
        [0.40, 0.20, 0.10, 0.30],
        [0.60, 0.20, 0.10, 0.10]
    ]
    # ------------------------------------------------

    # 驗證 weight_list 中每組的總和是否約 1.0（允許極小浮點誤差）
    for w in weight_list:
        if abs(sum(w) - 1.0) > 1e-6:
            raise KeyError(f"第5題：權重總和必須=1.0，但您提供 {w} 的總和 = {sum(w)}。")

    # 建立 NumPy array 代表四類資產的母本
    lcs_pool  = np.array(lcs_returns)
    scs_pool  = np.array(scs_returns)
    cb_pool   = np.array(cb_returns)
    usgb_pool = np.array(usgb_returns)

    rng = np.random.default_rng()

    def simulate_path(four_weights, W_withdraw):
        """
        模擬單一路徑，給定四元權重 four_weights = [w_LCS, w_SCS, w_CB, w_USGB]，
        以及 W_withdraw（2002 現值）之後，回傳形狀 (Nsim, ) 的 2033 年底剩餘資產。
        """
        # 建立 (Nsim, total_years+1) 矩陣，第一年 (t=0) 皆為 init_capital
        sim_mat = np.zeros((Nsim, total_years + 1), dtype=float)
        sim_mat[:, 0] = init_capital

        # 如果母本長度 = 31，就直接重複給所有模擬路徑
        # 如果母本長度 = 77，就 bootstrap 隨機抽樣 (有放回) 共 31 年
        if lcs_pool.shape[0] == total_years:
            # 直接把 31 年資料 tile 成 (Nsim,31)
            stock_lcs_draws  = np.tile(lcs_pool,  (Nsim, 1))
            stock_scs_draws  = np.tile(scs_pool,  (Nsim, 1))
            bond_cb_draws    = np.tile(cb_pool,   (Nsim, 1))
            bond_usgb_draws  = np.tile(usgb_pool, (Nsim, 1))
        else:
            # Bootstrap：從 77 年母本隨機抽 31 年
            stock_lcs_draws  = rng.choice(lcs_pool, size=(Nsim, total_years), replace=True)
            stock_scs_draws  = rng.choice(scs_pool, size=(Nsim, total_years), replace=True)
            bond_cb_draws    = rng.choice(cb_pool,   size=(Nsim, total_years), replace=True)
            bond_usgb_draws  = rng.choice(usgb_pool, size=(Nsim, total_years), replace=True)

        w_lcs, w_scs, w_cb, w_usgb = four_weights

        # 逐年模擬 2003~2033
        for t in range(1, total_years + 1):
            # 計算「當年」依通膨調整後要提領的實際金額
            real_withdraw = W_withdraw * ((1.0 + inflation) ** (t - 1))

            prev_val = sim_mat[:, t - 1]  # 去年 (t-1) 年底資產
            r1 = stock_lcs_draws[:, t - 1]   # LCS 的 multiplier
            r2 = stock_scs_draws[:, t - 1]   # SCS 的 multiplier
            r3 = bond_cb_draws[:, t - 1]     # CB  的 multiplier
            r4 = bond_usgb_draws[:, t - 1]   # USGB 的 multiplier

            # 計算「當年」(t 年底) 的資產 = 去年資產 × (w_LCS*r1 + w_SCS*r2 + w_CB*r3 + w_USGB*r4) - real_withdraw
            new_val = prev_val * (w_lcs * r1 + w_scs * r2 + w_cb * r3 + w_usgb * r4) - real_withdraw

            # 如果 new_val < 0，就強制設為 0
            new_val[new_val < 0] = 0.0
            sim_mat[:, t] = new_val

        # 回傳「2033 年底」(t=31) 的那一欄，形狀 (Nsim,)
        return sim_mat[:, -1]


    # =====================================================================
    # Part 1：固定提款 50,000 (2002 現值)，模擬各組合到 2033 年底的「平均剩餘」
    fixed_withdraw = 50_000
    part1_avg_remaining = {}  # key = tuple(weight)，value = float(平均剩餘)
    for w in weight_list:
        rems = simulate_path(w, fixed_withdraw)
        part1_avg_remaining[tuple(w)] = np.mean(rems)

    # 找出 Part1 最佳組合 (平均剩餘最大)
    best_portfolio_part1 = max(part1_avg_remaining, key=lambda x: part1_avg_remaining[x])


    # =====================================================================
    # Part 2：對每個組合 w，用二分搜尋找最大可提款 W*（2002 現值），使 2033 年底「平均剩餘」≈ 0
    def find_max_withdrawal_for_weights(weights):
        """
        對於給定 weights (四元向量)，用二分搜尋找「最大可提領 W*」，
        使 31 年後 (2033 年底) 的平均剩餘 ∼ 0（允許誤差 tol）。
        返回：(最佳 W*, 在此 W* 下的平均剩餘)
        """
        lo, hi = 0.0, 200000.0  # 可以提款的範圍 (2002 現值)，上限先設 200k
        tol = 1_000.0           # 平均剩餘允許誤差 ±1,000
        best_mid, best_avg = 0.0, None

        for _ in range(25):
            mid = (lo + hi) / 2.0
            remainders = simulate_path(weights, mid)
            avg_mid = np.mean(remainders)
            if avg_mid > tol:
                # 如果平均剩餘大於 tol，表示還可以再提高提款
                lo = mid
                best_mid = mid
                best_avg = avg_mid
            else:
                # 如果平均剩餘過低 (<= tol)，提款太多，要往下調
                hi = mid

        return best_mid, best_avg

    part2_max_withdrawal = {}
    part2_avg_remaining_at_star = {}
    for w in weight_list:
        W_star, avg_at_star = find_max_withdrawal_for_weights(w)
        part2_max_withdrawal[tuple(w)] = W_star
        part2_avg_remaining_at_star[tuple(w)] = avg_at_star

    # Part 2 最佳組合：看哪個 W* 最大
    best_portfolio_part2 = max(part2_max_withdrawal, key=lambda x: part2_max_withdrawal[x])


    # =====================================================================
    # 繪圖：Part1 & Part2 各畫一張長條圖，存到 static/results/
    os.makedirs("static/results", exist_ok=True)

    # (1) Part1 長條：X-軸為四種權重、Y-軸為「2033 年底平均剩餘」
    fn_part1 = "q5_part1_avg_remaining.png"
    plt.figure(figsize=(8, 5))
    x_labels = [f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%"
                for w in weight_list]
    y_vals1 = [part1_avg_remaining[tuple(w)] for w in weight_list]
    bars1 = plt.bar(x_labels, y_vals1, color='tab:blue')
    plt.xlabel("投資組合權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("2033 年底平均剩餘資產 (元)")
    plt.title("Part 1：固定提領 50,000 時，各組合 2033 年底平均剩餘")
    plt.xticks(rotation=15, ha='right')
    for idx, v in enumerate(y_vals1):
        plt.text(idx, v * 1.02, f"{v:,.0f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn_part1))
    plt.close()

    # (2) Part2 長條：X-軸為四種權重、Y-軸為最大可提領 W*
    fn_part2 = "q5_part2_max_withdraw.png"
    plt.figure(figsize=(8, 5))
    y_vals2 = [part2_max_withdrawal[tuple(w)] for w in weight_list]
    bars2 = plt.bar(x_labels, y_vals2, color='tab:orange')
    plt.xlabel("投資組合權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("最大可提領 W* (2002 現值，元)")
    plt.title("Part 2：各投資組合的最大可提領 W* (2033 年底平均剩餘 ≈ 0)")
    plt.xticks(rotation=15, ha='right')
    for idx, v in enumerate(y_vals2):
        plt.text(idx, v * 1.02, f"{v:,.0f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn_part2))
    plt.close()


    # =====================================================================
    # 最終回傳字典，前端 template (answer.html) 會用到
    return {
        # Part1 結果
        "Part1_avg_remaining": {f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%": 
                                part1_avg_remaining[tuple(w)] for w in weight_list},
        "Part1_best_portfolio": f"{int(best_portfolio_part1[0]*100)}%/" +
                                f"{int(best_portfolio_part1[1]*100)}%/" +
                                f"{int(best_portfolio_part1[2]*100)}%/" +
                                f"{int(best_portfolio_part1[3]*100)}%",

        # Part2 結果
        "Part2_max_withdrawal": {f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%":
                                 part2_max_withdrawal[tuple(w)] for w in weight_list},
        "Part2_avg_remaining_at_star": {f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%":
                                        part2_avg_remaining_at_star[tuple(w)] for w in weight_list},
        "Part2_best_portfolio": f"{int(best_portfolio_part2[0]*100)}%/" +
                                f"{int(best_portfolio_part2[1]*100)}%/" +
                                f"{int(best_portfolio_part2[2]*100)}%/" +
                                f"{int(best_portfolio_part2[3]*100)}%",

        # 圖檔
        "plots": {
            "Part 1：固定提領 50k 平均剩餘": fn_part1,
            "Part 2：最大可提領 W*":       fn_part2
        }
    }
