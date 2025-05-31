# solvers/question5.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve(file_path):
    """
    第五題：
    2002年底，60歲，退休金1,000,000。希望每年提領50,000以上(2002 現值)，
    通貨膨脹率3.12%。模擬100000次，四種投資組合(股票權重20%, 40%, 60%, 80%)，
    1. Part1：固定提款50,000，模擬到2033年底各組合平均剩餘，找最佳組合 w。
    2. Part2：搜尋最大可提領 W*（2002 現值），使 2033 年底平均剩餘 ≈ 0，並找最佳組合 w。
    資料使用 sustainableRetirementWithdrawls.xls，取出 Total Returns（Stocks, Bonds）列。
    """

    # 讀取 Excel
    df = pd.read_excel(file_path, sheet_name=0)

    # 將檔案中 'Year' 欄位轉為 int，取 2003 ~ 2033 的 'Total Returns Stocks'、'Total Returns Bonds'
    # 假設欄位名稱正確如下，若與您手上的檔案不同，請調整字串名稱：
    #   'Year'、'Total Returns Stocks'、'Total Returns Bonds'
    if 'Year' not in df.columns:
        raise KeyError("第5題：找不到 'Year' 欄位，請確認 sustainableRetirementWithdrawls.xls 格式")
    if 'Total Returns Stocks' not in df.columns or 'Total Returns Bonds' not in df.columns:
        raise KeyError("第5題：找不到 'Total Returns Stocks' 或 'Total Returns Bonds' 欄位，請確認檔案")

    df['Year'] = df['Year'].astype(int)
    # 取出 2003 ~ 2033 年份的 returns
    returns_df = df.loc[(df['Year'] >= 2003) & (df['Year'] <= 2033), ['Year', 'Total Returns Stocks', 'Total Returns Bonds']].copy()
    # 確保共 31 個年度
    if len(returns_df) != 31:
        raise KeyError(f"第5題：2003~2033 年份資料筆數應為 31，實際為 {len(returns_df)}，請確認 sustainableRetirementWithdrawls.xls")

    stock_returns = returns_df['Total Returns Stocks'].values   # numpy array, shape (31,)
    bond_returns  = returns_df['Total Returns Bonds'].values    # numpy array, shape (31,)

    # 模擬參數
    Nsim = 100_000              # 模擬路徑數
    years = 31                  # 從 2003 到 2033 共 31 年
    init_capital = 1_000_000    # 2002 年底初始退休金
    inflation = 0.0312          # 年通膨率 3.12%

    # 四種投資權重 (股票比重)：20%, 40%, 60%, 80%
    weight_list = [0.20, 0.40, 0.60, 0.80]

    # 先定義一個模擬函式：給定 weight、withdrawal (2002 現值)，回傳 2033 年底所有路徑的剩餘資產陣列
    def simulate_path(weight, W_withdraw):
        """
        weight      : 股票比重 (float between 0 and 1)
        W_withdraw  : 2002 現值下，每年提領金額（第1年2003年提 W_withdraw；第2年2004年提 W_withdraw*(1+inflation)；以此類推）
        回傳值：shape=(Nsim, years+1) 的矩陣，第一欄消耗 2002 底初始資金，
        最後一欄是 2033 年底剩餘資產。 
        """
        # 先建立一個 shape=(Nsim, years+1) 的矩陣，columns[0] 為 2002 年底的初始資金
        result = np.zeros((Nsim, years+1))
        result[:, 0] = init_capital

        # 將 returns 隨機重抽 (with replacement) Nsim x years 的樣本
        # stocks_draws[i, t] = 當第 i 條路徑，第 t 年 (2003+t-1) 抽到的股票 return
        # bonds_draws   對應 債券
        rng = np.random.default_rng()
        stocks_draws = rng.choice(stock_returns,  size=(Nsim, years), replace=True)
        bonds_draws  = rng.choice(bond_returns,   size=(Nsim, years), replace=True)

        for t in range(1, years+1):
            # 計算「真實提款金額」：
            # 第 t 年 (t=1 => 2003)：提款 = W_withdraw * (1+inflation)^(t-1)
            real_withdraw = W_withdraw * ((1+inflation)**(t-1))

            # 當年資產成長 = 上一年資產 * (weight * R_stock + (1-weight)*R_bond) - real_withdraw
            # R_stock 與 R_bond 讀自 stocks_draws[:, t-1], bonds_draws[:, t-1]
            prev_capital = result[:, t-1]
            r_stock = stocks_draws[:, t-1]
            r_bond  = bonds_draws[:,  t-1]
            # 當年成長後提款
            new_capital = prev_capital * ( weight*r_stock + (1-weight)*r_bond ) - real_withdraw

            # 如果負值，強制歸零：
            new_capital[new_capital < 0] = 0.0

            result[:, t] = new_capital

        # 回傳形狀 (Nsim, years+1)，但實際只要最後一欄 (第 31 年度 2033 年底) 的結果
        return result[:, -1]   # 回傳 shape=(Nsim,) 的 2033 年底剩餘資產

    # ===== Part1：固定提款 W=50,000，模擬各權重到 2033 平均剩餘 =====
    fixed_withdraw = 50_000
    avg_remaining_part1 = {}
    for w in weight_list:
        remainders = simulate_path(w, fixed_withdraw)  # 回傳 shape=(Nsim,)
        avg_remaining = np.mean(remainders)
        avg_remaining_part1[w] = avg_remaining

    # 找 Part1 下「平均剩餘最多」的最佳 w
    best_w_part1 = max(avg_remaining_part1, key=lambda x: avg_remaining_part1[x])

    # ===== Part2：搜尋最大可提領 W*，讓 2033 年底平均剩餘 ≈ 0 =====
    # 對每種權重，使用二分搜尋找到 W*，使平均剩餘逼近 0（允許±1000 的誤差）
    max_withdrawals = {}
    avg_remainder_at_wstar = {}

    def find_max_withdraw(weight):
        """
        使用簡單的二分搜尋，搜尋 W 在 [0, 200000] 範圍內，
        使得 simulate_path(weight, W).mean() ≈ 0（容許 ±1000）。
        回傳 (W_star, avg_remaining_at_Wstar)。
        """
        lo, hi = 0.0, 200000.0   # 假設最大不超過 200k
        target = 0.0
        tol = 1_000.0            # 容許平均剩餘 ±1000 為接近 0
        W_star = 0.0
        avg_rem = None

        for _ in range(25):   # 約 2^25 ≈ 3.3e7 的解析度，足夠
            mid = (lo + hi) / 2.0
            rems = simulate_path(weight, mid)
            avg_mid = np.mean(rems)
            # 如果平均剩餘過高 (> +tol)，代表提款太保守，提款額可再增大
            if avg_mid > tol:
                lo = mid
            # 如果平均剩餘過低 (< -tol)，代表提款過高，提款額要降低
            else:
                hi = mid
            W_star = mid
            avg_rem = avg_mid

        return W_star, avg_rem

    for w in weight_list:
        W_star, avg_rem = find_max_withdraw(w)
        max_withdrawals[w] = W_star
        avg_remainder_at_wstar[w] = avg_rem

    # 找 Part2 下，「最大可提領金額 W* 最大」的最佳 w
    best_w_part2 = max(max_withdrawals, key=lambda x: max_withdrawals[x])

    # ===== 畫圖：Part1 平均剩餘長條圖；Part2 最大可提領金額長條圖 =====
    os.makedirs("static/results", exist_ok=True)

    # (1) Part1：固定提款50,000 時，各 w→avg_remaining 比較條形圖
    fn_part1 = "q5_part1_avg_remaining.png"
    plt.figure(figsize=(8, 5))
    ws = [int(w*100) for w in weight_list]
    avg_vals = [avg_remaining_part1[w] for w in weight_list]
    plt.bar(ws, avg_vals, color='tab:blue')
    plt.xlabel("股票權重 (%)")       # 圖上 X 軸仍顯示英文也可，但為了清楚，我直接寫中文
    plt.ylabel("2033 年底剩餘平均資產 (元)")
    plt.title("Part 1：固定提款 50,000 時，各組合 2033 年底平均剩餘")
    for idx, val in enumerate(avg_vals):
        plt.text(ws[idx], val * 1.02, f"{val:,.0f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn_part1))
    plt.close()

    # (2) Part2：各 w→對應最大可提領金額 W*（2002 現值）長條圖
    fn_part2 = "q5_part2_max_withdraw.png"
    plt.figure(figsize=(8, 5))
    w_vals = [int(w*100) for w in weight_list]
    Wstars = [max_withdrawals[w] for w in weight_list]
    plt.bar(w_vals, Wstars, color='tab:orange')
    plt.xlabel("股票權重 (%)")
    plt.ylabel("最大可提領（2002 現值，元）")
    plt.title("Part 2：各組合 最大可提領金額 (2033 平均剩餘 ≈ 0)")
    for idx, val in enumerate(Wstars):
        plt.text(w_vals[idx], val * 1.02, f"{val:,.0f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn_part2))
    plt.close()

    # ===== 回傳結果 =====
    return {
        # Part1 資料
        "Part1_avg_remaining": avg_remaining_part1,
        "Part1_best_w": best_w_part1,
        # Part2 資料
        "Part2_max_withdrawal": max_withdrawals,
        "Part2_avg_remaining_at_Wstar": avg_remainder_at_wstar,
        "Part2_best_w": best_w_part2,
        # 圖檔檔名
        "plots": {
            "Part1 固定提款 50k 平均剩餘": fn_part1,
            "Part2 最大可提領 W*":        fn_part2
        }
    }
