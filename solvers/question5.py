# solvers/question5.py

import os
import numpy as np
import matplotlib.pyplot as plt

# 這裡定義題目資訊，用來在 answer.html 裡顯示「第 5 題，題目標題」等文字
QUESTION = {
    "id": 5,
    "title": "題目 5 標題"
}

def run_full_simulation(file_path, n_sim=100_000):
    """
    這是第 5 題的「真正耗時」函式，會被放在 background thread/process 內執行。
    參數：
      - file_path：上傳檔案的實際路徑 (本例中無實際使用，因為我們直接寫死歷史報酬率)
      - n_sim：模擬次數 (本例預設 100_000)
    回傳值：
      一個 dict，包含第 5 題要顯示的所有欄位。例如：
       {
         "最佳組合(2033_剩餘平均)": {"LCS": w1, "SCS": w2, "CB": w3, "USGB": w4},
         "統計結果(2033_剩餘)": {"Mean": ..., "StdDev": ..., "Median": ...},
         "最佳組合(2033_提款平均)": {...},
         "統計結果(2033_提款)": {...},
         "plots": {"Plot_Remain": "q5_remain_dist.png", "Plot_Withdraw": "q5_withdraw_dist.png"}
       }
    """

    # ------------------------------------------------------------
    # 1) 先印出一些除錯訊息，讓您在 Logstream 可以看到背景工作啟動
    print(f"[run_full_simulation] 檔案路徑 (unused): {file_path}")
    print(f"[run_full_simulation] 開始進行 {n_sim} 次模擬 + 權重搜尋")

    # ------------------------------------------------------------
    # 2) 直接寫死 1926-2002 的歷史總回報 (multipliers)：
    #    LCS, SCS, CB, USGB 各有 77 年 (1926 ~ 2002)，將百分比轉成 1+rate/100
    #    例如：11.91% 變成 1.1191
    lcs_pct = np.array([
        11.91, 36.74, 41.45, -8.04, -25.40, -44.45, -8.66, 54.28, -1.88, 46.67, 33.74, -35.53,
        30.27, -0.74, -9.72, -11.40, 20.57, 26.22, 20.36, 36.27, -8.66, 5.29, 5.39, 18.51,
        32.20, 23.75, 18.64, -1.37, 52.59, 31.50, 6.50, -10.96, 43.57, 12.45, 0.33, 27.26,
        -8.76, 22.72, 16.57, 12.47, -10.15, 24.05, 11.03, -8.41, 4.05, 14.24, 19.06, -14.71,
        -26.43, 37.23, 23.91, -7.22, 6.53, 18.61, 32.45, -4.95, 21.75, 22.44, 6.36, 32.08,
        18.43, 5.28, 16.83, 31.41, -3.18, 30.60, 7.69, 9.93, 1.30, 37.57, 23.07, 33.27,
        28.58, 21.04, -9.11, -11.89, -22.10
    ]) / 100.0 + 1.0  # length 77

    scs_pct = np.array([
        -4.32, 27.16, 42.36, -51.09, -41.92, -49.46, 2.78, 165.34, 24.67, 54.31, 74.63,
        -55.36, 28.74, 0.13, -8.48, -11.04, 47.76, 94.08, 57.12, 77.93, -12.21, -1.08,
        -4.13, 20.65, 42.12, 8.60, 4.70, -6.08, 62.86, 21.14, 4.05, -14.80, 67.76, 17.11,
        -4.23, 31.28, -14.15, 17.88, 21.13, 39.71, -7.54, 93.48, 43.29, -28.66, -16.98,
        17.47, 1.90, -35.72, -24.84, 61.18, 56.09, 23.70, 22.87, 43.72, 37.61, 10.83, 27.73,
        37.08, -10.35, 26.44, 5.12, -11.62, 22.30, 9.27, -24.32, 47.44, 25.60, 20.64, -0.12,
        33.83, 17.06, 22.57, -4.93, 25.53, -3.31, 10.87, -17.43
    ]) / 100.0 + 1.0  # length 77

    cb_pct = np.array([
        5.96, 7.78, 0.95, 3.84, 7.10, -3.58, 11.36, 5.71, 12.00, 7.30, 6.63, 1.47,
        5.69, 4.94, 4.96, 1.86, 3.99, 3.85, 4.16, 5.46, 0.94, -1.76, 3.61, 4.67,
        0.58, -2.32, 2.73, 3.62, 5.14, -0.43, -5.96, 9.09, -2.96, -2.26, 11.42, 2.50,
        7.38, 0.85, 4.64, -0.37, 1.95, -6.18, 0.69, -7.31, 15.53, 14.24, 6.41, 1.27,
        1.23, 11.57, 14.86, 1.30, -2.12, 2.42, 5.20, 1.18, 24.54, 2.87, 16.08, 31.39,
        21.91, -1.46, 9.55, 17.86, 6.95, 19.14, 8.59, 14.33, -6.47, 29.43, 0.30, 14.02,
        12.14, -8.10, 16.57, 7.43, 16.56
    ]) / 100.0 + 1.0  # length 77

    usgb_pct = np.array([
        6.04, 5.60, 0.66, 5.11, 5.63, -4.48, 11.36, 0.70, 9.34, 6.33, 4.78, 1.12,
        5.80, 4.99, 3.69, 0.28, 2.32, 2.56, 2.20, 5.52, 0.53, -0.46, 2.49, 3.66,
        0.34, -1.07, 1.47, 3.50, 3.87, -0.82, -2.30, 7.71, -2.89, -1.30, 12.51, 1.68,
        6.61, 1.55, 4.00, 1.00, 4.49, -2.67, 2.25, -2.64, 14.30, 10.22, 4.88, 2.13,
        5.36, 7.94, 14.61, 0.61, 0.88, 2.38, 0.26, 5.80, 34.28, 4.50, 14.65, 24.98,
        18.96, 0.62, 7.47, 15.41, 8.32, 17.14, 7.50, 13.83, -5.78, 22.18, 1.72, 10.64,
        10.63, -3.44, 14.78, 6.49, 13.35
    ]) / 100.0 + 1.0  # length 77

    # ------------------------------------------------------------
    # 3) 基本設定
    #    - 初始本金 1,000,000（2002年底）
    #    - 每年提領 50,000 (2002 現值)，未來各年要依 3.12% 通膨作調整
    #    - 後續每年要從投資組合裡取出「當年通膨後的提款金額」
    #    - 投資期間：60 歲開始(2002) → 到 2033 結束（30 年期）
    initial_balance = 1_000_000.0
    withdraw_2002 = 50_000.0
    inflation_rate = 0.0312  # 3.12% 每年通膨

    # 投資年限：2002→2033 共 31 個年度（包含 2002 當年提款，最後到 2033 結束）
    years = list(range(2002, 2033 + 1))

    # 權重格點：LCS, SCS, CB, USGB 四種資產
    # w1 + w2 + w3 + w4 = 1；我們只用 w1, w2, w3，w4 = 1 - (w1+w2+w3)
    # 每個 w 都從 0.00、0.01、0.02 … 走到 1.00（共 101 個可能值）
    weights = np.arange(0.0, 1.0 + 1e-9, 0.01)  # 0.00, 0.01, …, 1.00

    total_combinations = 0
    # 我們預先算一下「合法的」權重組合個數，以供印出進度時參考
    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                w_sum = w1 + w2 + w3
                if w_sum <= 1.0:
                    total_combinations += 1
    print(f"[run_full_simulation] 權重格點總數：{total_combinations} 個")

    # 準備儲存「2033 年剩餘金額」與「2033 年當年可提款金額」的二維陣列
    # shape 大小： (total_combinations, n_sim)
    remain_results = np.zeros((total_combinations, n_sim), dtype=np.float64)
    withdraw_results = np.zeros((total_combinations, n_sim), dtype=np.float64)

    combo_index = 0
    start_time = time.time()

    # ------------------------------------------------------------
    # 4) 開始「權重搜尋 + 蒙地卡羅模擬」的巢狀迴圈
    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < -1e-12:
                    # 如果 w1 + w2 + w3 > 1，就不合法，跳過
                    continue
                # 確保 w4 不會因為浮點誤差稍微變成 -1e-16 之類，強制夾 0
                w4 = max(w4, 0.0)

                # 每跑 500 個組合，就印一次進度
                if combo_index % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"[run_full_simulation] 已跑 {combo_index}/{total_combinations} 組合，已耗時 {elapsed:.1f} 秒…")

                # 對這個權重組合，跑 n_sim 次模擬
                # 每一次模擬：從 2002 開始
                # 1) 2002 年：本金 1,000,000，先領取 withdraw_2002 (不折現、當年立刻扣除)
                #             剩餘本金 * (投資報酬率) => 2003 年初本金
                # 2) 2003 年：先調整通膨後提款額 = withdraw_2002 * (1 + inflation_rate)^(2003-2002)
                #             從 2003 年初本金中扣除 → 剩餘本金 * 投資報酬率 => 2004 年初本金
                # …
                # 一直跑到 2033 為止，紀錄「當年提款金額」與「2033 年年末剩餘本金」

                # 先隨機從 77 年報酬率分別抽 n_sim 次 (bootstrap)
                # （抽出來就是一個形狀 (77, n_sim) 的矩陣，每一列代表某年所有模擬樣本的 multiplier）
                idxs = np.random.randint(0, 77, size=(len(years), n_sim))
                # idxs[i, j] = 第 i 個年度 (2002 + i) 的樣本 j 抽到哪一年「歷史報酬」
                # 我們接著要把 idxs 轉成各資產的 multiplier 案例
                lcs_sim = lcs_pct[idxs]    # shape = (31, n_sim)
                scs_sim = scs_pct[idxs]    # 同上
                cb_sim  = cb_pct[idxs]
                usgb_sim= usgb_pct[idxs]

                # 現在從 2002 年到 2033 年，共 32 個年度
                # 我們用「陣列記錄」方式，同時計算 n_sim 個模擬
                # 先宣告一個 (years, n_sim) 的矩陣儲存 balance 每年的年初金額
                balance = np.zeros((len(years), n_sim), dtype=np.float64)

                # 年度 0 (2002 年年初前) 先把 balance 設為 initial_balance
                balance[0, :] = initial_balance

                # 2002 年先領一次 withdraw_2002，然後乘以 (w1*lcs_sim[0] + w2*scs_sim[0] + w3*cb_sim[0] + w4*usgb_sim[0])
                # 才是 2003 年年初的本金
                # 但如果 withdraw_2002 > initial_balance，就代表破產 => 後面就視為 0
                withdraw_2002_amount = withdraw_2002
                # 模擬 2002 年提款
                post_withdraw = np.maximum(balance[0, :] - withdraw_2002_amount, 0.0)
                # 2003 年年初的本金 = post_withdraw × 投資報酬率(2002 年度)
                balance[1, :] = post_withdraw * (
                    w1 * lcs_sim[0, :] + w2 * scs_sim[0, :] + w3 * cb_sim[0, :] + w4 * usgb_sim[0, :]
                )

                # 從 i=1 (2003 年) 開始：
                for i in range(1, len(years)):
                    # 先計算「當年度實際提款額」：withdraw_2002 × (1+inflation_rate)^i
                    this_withdraw = withdraw_2002 * ((1+inflation_rate) ** i)
                    # 每次提款後剩餘本金
                    post_withdraw = np.maximum(balance[i, :] - this_withdraw, 0.0)
                    if i < len(years)-1:
                        # 把領完後的 post_withdraw，乘上當年度投資報酬率，得到下一年年初本金
                        balance[i+1, :] = post_withdraw * (
                            w1 * lcs_sim[i, :] + w2 * scs_sim[i, :] + w3 * cb_sim[i, :] + w4 * usgb_sim[i, :]
                        )

                # 模擬 2033 年「提款金額」(we stored in this_withdraw when i==31)
                # 2033 年提款 = withdraw_2002 × (1+inflation_rate)^31
                withdraw_2033 = withdraw_2002 * ((1+inflation_rate) ** (len(years)-1))
                # 只要 balance[31, :] 代表 2033 年年初本金，提款前；後面沒再投資，不必乘投報率
                remain_2033 = balance[-1, :]  # 當年度提款前剩餘

                # 把結果放到 remain_results / withdraw_results
                remain_results[combo_index, :] = remain_2033
                withdraw_results[combo_index, :] = np.full(n_sim, withdraw_2033, dtype=np.float64)

                combo_index += 1

    # 權重搜尋 + 模擬全跑完
    total_elapsed = time.time() - start_time
    print(f"[run_full_simulation] 權重搜尋全部完成，共耗時 {total_elapsed:.1f} 秒")

    # ------------------------------------------------------------
    # 5) 列出所有 combo 的「2033 年剩餘平均」與「2033 年提款平均」，找出最佳
    #    a) 2033 年剩餘金額最多(平均) 的 index
    mean_remain = remain_results.mean(axis=1)    # shape = (total_combinations,)
    best_idx_remain = np.argmax(mean_remain)     # 找到最大平均值對應的 index

    #    b) 2033 年提款金額最多(平均) 的 index（其實提款金額對所有模擬都是一樣，但不同權重組合的提款條件不同，可不一樣）
    mean_withdraw = withdraw_results.mean(axis=1)  # 其實 withdraw_results 每列都是一樣，但為一致性示範
    best_idx_withdraw = np.argmax(mean_withdraw)

    # 將 best_idx 轉回權重組合 (w1, w2, w3, w4)
    # 由於我們是以三重巢狀迴圈走 weights，combo_index 的順序就是：
    #   (0,0,0), (0,0,0.01), (0,0,0.02), ..., (0,0,1.0),
    #   (0,0.01,0), (0,0.01,0.01), ..., 一直到 (1.0,0,0), … (1.0,0,0.0)
    def idx_to_weights(idx):
        count = 0
        for w1 in weights:
            for w2 in weights:
                for w3 in weights:
                    w4 = 1.0 - (w1 + w2 + w3)
                    if w4 < -1e-12:
                        continue
                    w4 = max(w4, 0.0)
                    if count == idx:
                        return (w1, w2, w3, w4)
                    count += 1
        return (0, 0, 0, 0)  # 不可能到這

    best_w_remain = idx_to_weights(best_idx_remain)
    best_w_withdraw = idx_to_weights(best_idx_withdraw)

    # 統計「2033 年剩餘」和「2033 年提款」的各種指標 (Mean/StdDev/Median)
    remain_2033_all = remain_results[best_idx_remain, :]
    withdraw_2033_all = withdraw_results[best_idx_withdraw, :]

    stats_remain = {
        "Mean": float(np.mean(remain_2033_all)),
        "StdDev": float(np.std(remain_2033_all, ddof=1)),
        "Median": float(np.median(remain_2033_all)),
    }
    stats_withdraw = {
        "Mean": float(np.mean(withdraw_2033_all)),
        "StdDev": float(np.std(withdraw_2033_all, ddof=1)),
        "Median": float(np.median(withdraw_2033_all)),
    }

    # ------------------------------------------------------------
    # 6) 為了方便顯示，把最佳權重以 dict 形式回傳
    best_comb_remain = {
        "LCS": best_w_remain[0],
        "SCS": best_w_remain[1],
        "CB":  best_w_remain[2],
        "USGB": best_w_remain[3]
    }
    best_comb_withdraw = {
        "LCS": best_w_withdraw[0],
        "SCS": best_w_withdraw[1],
        "CB":  best_w_withdraw[2],
        "USGB": best_w_withdraw[3]
    }

    # ------------------------------------------------------------
    # 7) 繪製 2033 年「剩餘金額分布圖」與 2033 年「提款分布圖」
    #    並存成 q5_remain_dist.png / q5_withdraw_dist.png
    result_plots = {}

    # (a) 2033 Year-End 剩餘金額分布圖
    fig1 = plt.figure(figsize=(6,4))
    plt.hist(remain_2033_all, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.title("2033 Remain Balance Distribution")
    plt.xlabel("Balance (NT$)")
    plt.ylabel("Probability Density")
    fname1 = f"q5_remain_dist.png"
    fig1.tight_layout()
    fig1.savefig(os.path.join("static", "results", fname1))
    plt.close(fig1)
    result_plots["Plot_Remain"] = fname1

    # (b) 2033 年提款分布圖
    fig2 = plt.figure(figsize=(6,4))
    plt.hist(withdraw_2033_all, bins=50, color='salmon', edgecolor='black', density=True)
    plt.title("2033 Annual Withdrawal Distribution")
    plt.xlabel("Annual Withdrawal (NT$)")
    plt.ylabel("Probability Density")
    fname2 = f"q5_withdraw_dist.png"
    fig2.tight_layout()
    fig2.savefig(os.path.join("static", "results", fname2))
    plt.close(fig2)
    result_plots["Plot_Withdraw"] = fname2

    # ------------------------------------------------------------
    # 8) 最後回傳整個 dict 給 background_question5，並存到 jobs[job_id]['result']
    return {
        "最佳組合(2033_剩餘平均)": best_comb_remain,
        "統計結果(2033_剩餘)": stats_remain,
        "最佳組合(2033_提款平均)": best_comb_withdraw,
        "統計結果(2033_提款)": stats_withdraw,
        "plots": result_plots
    }
