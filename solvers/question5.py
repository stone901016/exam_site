# solvers/question5.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve(file_path):
    """
    第 5 題：使用 sustainableRetirementWithdrawals.xls 裡 2003–2033 四類資產的「總回報乘數」(multiplier)，
    模擬 100000 次，步進 0.01 的全網格搜索，並且用「先粗步進筛选 Top 候选，再对候选做精细模拟」的思路，
    最终输出：
      1) Part1：2033 年底「平均剩餘」最大的權重組合；
      2) Part2：可提領金額 W*（使 2033 年底平均剩餘接近零）最大的權重組合。
    """

    # ------------------------------------------------------------
    # (A) 先从 Excel 读数据——需要 2003~2033 共 31 年，且包含下列四个「总回报乘数」列：
    #     - "Total Return Large Company Stocks"
    #     - "Total Return Small Company Stocks"
    #     - "Total Return Corporate Bonds"
    #     - "Total Return U.S. Government Bonds"
    #
    #     这四列应当已经是「乘数」形式 (例如 1.1191 代表 11.91% 的年回报)。
    #
    # 如果无法满足，就直接抛错并提示上传的 Excel 有问题。
    try:
        df = pd.read_excel(file_path, sheet_name=0)

        # 确认包含我们需要的列
        required_cols = [
            "Year",
            "Total Return Large Company Stocks",
            "Total Return Small Company Stocks",
            "Total Return Corporate Bonds",
            "Total Return U.S. Government Bonds"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"第5題：Excel 缺少必要欄位「{col}」")

        # 取 2003–2033 共 31 年
        df_sel = df.loc[(df["Year"] >= 2003) & (df["Year"] <= 2033), required_cols].copy()
        if df_sel.shape[0] != 31:
            raise KeyError(f"第5題：Excel 中 2003~2033 年份資料不足，共有 {df_sel.shape[0]} 行，而非 31 行")

        # 把这四列取出为 numpy 数组 (长度 31)，保留「总回报乘数」
        lcs_returns   = df_sel["Total Return Large Company Stocks"].values.astype(float)
        scs_returns   = df_sel["Total Return Small Company Stocks"].values.astype(float)
        cb_returns    = df_sel["Total Return Corporate Bonds"].values.astype(float)
        usgb_returns  = df_sel["Total Return U.S. Government Bonds"].values.astype(float)

    except Exception as e:
        # 只要读不到，就直接报错，让用户检查 Excel
        raise KeyError(f"第5題：讀取 Excel 失敗，請確認 sustainableRetirementWithdrawals.xls 內含 2003~2033 年份，且有「總回報乘數」欄位。詳細錯誤：{e}")

    # ------------------------------------------------------------
    # (B) 其餘共用參數
    total_years     = 31        # 2003 年到 2033 年（共 31 年）
    init_capital    = 1_000_000 # 2002 年底資金 1,000,000 元
    inflation_rate  = 0.0312    # 通膨率 3.12%
    fixed_withdraw  = 50_000    # 每年提領 50,000 元 (2002 年現值)

    # 转成 numpy 向量
    pool_lcs   = np.array(lcs_returns)   # shape=(31,)
    pool_scs   = np.array(scs_returns)   # shape=(31,)
    pool_cb    = np.array(cb_returns)    # shape=(31,)
    pool_usgb  = np.array(usgb_returns)  # shape=(31,)

    rng = np.random.default_rng()

    # ------------------------------------------------------------
    # (C) 定義「一次模擬 31 年後剩餘資金」的函式
    def simulate_path(weights, W_withdraw, Nsim_local):
        """
        weights: (w_lcs, w_scs, w_cb, w_usgb)，四類資產权重，和须近似等于 1
        W_withdraw: 每年要提領多少（2002 年現值）
        Nsim_local: 模拟次数

        流程：
        - sim_mat 是 (Nsim_local × (total_years+1)) 的矩陣，第 0 列初始化為 init_capital。
        - 每年 t = 1..31:
            - 先通膨累計：當年第 t 年實際提款 = W_withdraw * (1 + inflation_rate)^(t-1)
            - 抽样 31 年的收益率矩阵：r1, r2, r3, r4，shape 都是 (Nsim_local, 31)
            - 对当年第 t 年：
                newv = prev * (w_lcs * r1[:,t-1] + w_scs * r2[:,t-1]
                               + w_cb * r3[:,t-1] + w_usgb * r4[:,t-1])
                       - real_withdraw
                如果 newv < 0，就置 0（代表破产后就剩 0 不再增长）
        - 返回 sim_mat[:, -1]，也就是 31 年后 2033 年底的剩余值，一共 Nsim_local 条路径。
        """
        w_lcs, w_scs, w_cb, w_usgb = weights

        # 用 numpy 一次性抽样 31 年；因为总共有 4 类资产，我们对每个资产都抽样。
        # 这里我们假设历史数据（2003~2033）的「乘数」就是这 31 年每年固定的顺序，
        # 进行“有放回抽样”，shape 都是 (Nsim_local, 31)。
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

        return sim_mat[:, -1]  # Return shape = (Nsim_local,)

    # ------------------------------------------------------------
    # (D)【第一階段】粗網格篩選（step = 0.05, Nsim = 20_000）
    step_first   = 0.05
    Nsim_first   = 20_000
    grid_vals    = np.arange(0.0, 1.0 + 1e-9, step_first)

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

    # Part 1 第一階段：固定提款 50k，计算 31 年后「平均剩余」，取 Top 20 权重
    part1_scores1 = []
    for weights in weight_grid1:
        rems = simulate_path(weights, fixed_withdraw, Nsim_first)
        avg_rem = rems.mean()
        part1_scores1.append((avg_rem, weights))
    part1_scores1.sort(key=lambda x: x[0], reverse=True)
    top20_part1 = part1_scores1[:20]   # (avg_rem, weights)

    # Part 2 第一階段：对每组权重，用 Nsim_first 做「二分搜 W*」让平均剩余≈0
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
    top20_part2 = part2_scores1[:20]   # (W_approx, weights)

    # ------------------------------------------------------------
    # (E)【第二階段】精細計算（只对 top20_part1 中的 20 组和 top20_part2 中的 20 组做 Nsim_final = 100_000）
    # 合并去重，得到候选权重 <= 40
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
        # Part1 精细：固定提款 50k
        rems1 = simulate_path(weights, fixed_withdraw, Nsim_final)
        avg_rem1 = rems1.mean()
        Part1_final.append((avg_rem1, weights))
        if avg_rem1 > best_avg_rem:
            best_avg_rem = avg_rem1
            best_w1      = weights

        # Part2 精细：二分搜 W*
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

    # ------------------------------------------------------------
    # (F) 繪圖并存檔
    os.makedirs("static/results", exist_ok=True)

    # 候选权重标签（用 "LCS/SCS/CB/USGB" 四个数字% 表示）
    combos_f = [
        f"{int(w[0]*100)}%/{int(w[1]*100)}%/{int(w[2]*100)}%/{int(w[3]*100)}%"
        for w in candidates
    ]

    # Part1 绘图 (2033 年底平均剩余), 标红最优
    values1_f = [v for v, w in Part1_final]
    fn1 = "q5_part1.png"
    plt.figure(figsize=(14, 6))
    bars1 = plt.bar(combos_f, values1_f, color='tab:blue')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("2033 年底平均剩餘 (元)")
    plt.title("Part1：固定提款 50,000，候選權重組合 2033 年度平均剩餘")
    idx_best1 = combos_f.index(f"{int(best_w1[0]*100)}%/"
                                f"{int(best_w1[1]*100)}%/"
                                f"{int(best_w1[2]*100)}%/"
                                f"{int(best_w1[3]*100)}%")
    bars1[idx_best1].set_color('tab:red')
    plt.text(idx_best1,
             values1_f[idx_best1] * 1.02,
             f"{int(values1_f[idx_best1]):,}",
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn1))
    plt.close()

    # Part2 绘图 (最大 W*), 标红最优
    values2_f = [v for v, w in Part2_final]
    fn2 = "q5_part2.png"
    plt.figure(figsize=(14, 6))
    bars2 = plt.bar(combos_f, values2_f, color='tab:orange')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("資產權重 (LCS/SCS/CB/USGB)")
    plt.ylabel("最大可提領 W* (元)")
    plt.title("Part2：最大 W* 值 (讓 2033 年底平均剩餘約 ≈ 0)")
    idx_best2 = combos_f.index(f"{int(best_w2[0]*100)}%/"
                                f"{int(best_w2[1]*100)}%/"
                                f"{int(best_w2[2]*100)}%/"
                                f"{int(best_w2[3]*100)}%")
    bars2[idx_best2].set_color('tab:red')
    plt.text(idx_best2,
             values2_f[idx_best2] * 1.02,
             f"{int(values2_f[idx_best2]):,}",
             ha='center', va='bottom', color='red', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn2))
    plt.close()

    # ------------------------------------------------------------
    # (G) 返回结果
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
