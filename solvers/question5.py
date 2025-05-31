# solvers/question5.py
import os
import numpy as np
import matplotlib.pyplot as plt

QUESTION = {
    "id": 5,
    "title": "題目 5 標題",
}

def run_full_simulation(file_path, n_sim=100_000):
    """
    耗時函式：跑 100,000 次蒙地卡羅 + 權重格點搜尋，回傳要在 answer.html 顯示的結果 dict。
    """

    # (A) 內嵌歷史報酬率 (1926–2002，共 77 年)
    lcs_pct = np.array([
        11.91, 36.74, 41.45, -8.04, -25.40, -44.45, -8.66, 54.28, -1.88, 46.67,
        33.74, -35.53, 30.27, -0.74, -9.72, -11.40, 20.57, 26.22, 20.36, 36.27,
        -8.66, 5.29, 5.39, 18.51, 32.20, 23.75, 18.64, -1.37, 52.59, 31.50,
        6.50, -10.96, 43.57, 12.45, 0.33, 27.26, -8.76, 22.72, 16.57, 12.47,
        -10.15, 24.05, 11.03, -8.41, 4.05, 14.24, 19.06, -14.71, -26.43, 37.23,
        23.91, -7.22, 6.53, 18.61, 32.45, -4.95, 21.75, 22.44, 6.36, 32.08,
        18.43, 5.28, 16.83, 31.41, -3.18, 30.60, 7.69, 9.93, 1.30, 37.57,
        23.07, 33.27, 28.58, 21.04, -9.11, -11.89, -22.10
    ]) / 100.0 + 1.0

    scs_pct = np.array([
        -4.32, 27.16, 42.36, -51.09, -41.92, -49.46, 2.78, 165.34, 24.67, 54.31,
        74.63, -55.36, 28.74, 0.13, -8.48, -11.04, 47.76, 94.08, 57.12, 77.93,
        -12.21, -1.08, -4.13, 20.65, 42.12, 8.60, 4.70, -6.08, 62.86, 21.14,
        4.05, -14.80, 67.76, 17.11, -4.23, 31.28, -14.15, 17.88, 21.13, 39.71,
        -7.54, 93.48, 43.29, -28.66, -16.98, 17.47, 1.90, -35.72, -24.84, 61.18,
        56.09, 23.70, 22.87, 43.72, 37.61, 10.83, 27.73, 37.08, -10.35, 26.44,
        5.12, -11.62, 22.30, 9.27, -24.32, 47.44, 25.60, 20.64, -0.12, 33.83,
        17.06, 22.57, -4.93, 25.53, -3.31, 10.87, -17.43
    ]) / 100.0 + 1.0

    cb_pct = np.array([
        5.96, 7.78, 0.95, 3.84, 7.10, -3.58, 11.36, 5.71, 12.00, 7.30,
        6.63, 1.47, 5.69, 4.94, 4.96, 1.86, 3.99, 3.85, 4.16, 5.46,
        0.94, -1.76, 3.61, 4.67, 0.58, -2.32, 2.73, 3.62, 5.14, -0.43,
        -5.96, 9.09, -2.96, -2.26, 11.42, 2.50, 7.38, 0.85, 4.64, -0.37,
        1.95, -6.18, 0.69, -7.31, 15.53, 14.24, 6.41, 1.27, 1.23, 11.57,
        14.86, 1.30, -2.12, 2.42, 5.20, 1.18, 24.54, 2.87, 16.08, 31.39,
        21.91, -1.46, 9.55, 17.86, 6.95, 19.14, 8.59, 14.33, -6.47, 29.43,
        0.30, 14.02, 12.14, -8.10, 16.57, 7.43, 16.56
    ]) / 100.0 + 1.0

    usgb_pct = np.array([
        6.04, 5.60, 0.66, 5.11, 5.63, -4.48, 11.36, 0.70, 9.34, 6.33,
        4.78, 1.12, 5.80, 4.99, 3.69, 0.28, 2.32, 2.56, 2.20, 5.52,
        0.53, -0.46, 2.49, 3.66, 0.34, -1.07, 1.47, 3.50, 3.87, -0.82,
        -2.30, 7.71, -2.89, -1.30, 12.51, 1.68, 6.61, 1.55, 4.00, 1.00,
        4.49, -2.67, 2.25, -2.64, 14.30, 10.22, 4.88, 2.13, 5.36, 7.94,
        14.61, 0.61, 0.88, 2.38, 0.26, 5.80, 34.28, 4.50, 14.65, 24.98,
        18.96, 0.62, 7.47, 15.41, -3.18, -24.32, 47.44, 25.60, 20.64, -0.12,
        33.83, 17.06, 22.57, -4.93, 25.53, -3.31, 10.87, -17.43
    ]) / 100.0 + 1.0

    assert len(lcs_pct) == len(scs_pct) == len(cb_pct) == len(usgb_pct) == 77

    # (B) 參數
    initial_balance = 1_000_000.0
    annual_withdraw_real = 50_000.0
    inflation_rate = 0.0312
    years = 30
    N = n_sim

    # 計算每年需要提領的 nominal 值
    withdraws = np.array([annual_withdraw_real * ((1 + inflation_rate) ** t) for t in range(years + 1)])

    # (C) 權重格點
    weight_step = 0.01
    weight_list = []
    rng = np.arange(0.0, 1.0 + 1e-8, weight_step)
    for w_lcs in rng:
        for w_scs in rng:
            for w_cb in rng:
                w_usgb = 1.0 - (w_lcs + w_scs + w_cb)
                if w_usgb < 0 or w_usgb > 1.0:
                    continue
                weight_list.append((w_lcs, w_scs, w_cb, w_usgb))

    # (D) 暴力搜尋 + Monte Carlo
    all_results = []
    for (w_lcs, w_scs, w_cb, w_usgb) in weight_list:
        sims = np.zeros((N, years + 1))
        sims[:, 0] = initial_balance

        for t in range(years):
            sims[:, t] -= withdraws[t]
            sims[:, t] = np.where(sims[:, t] < 0, 0.0, sims[:, t])

            r_lcs = np.random.choice(lcs_pct, size=N)
            r_scs = np.random.choice(scs_pct, size=N)
            r_cb = np.random.choice(cb_pct, size=N)
            r_usgb = np.random.choice(usgb_pct, size=N)

            sims[:, t + 1] = (
                sims[:, t] * w_lcs * r_lcs
                + sims[:, t] * w_scs * r_scs
                + sims[:, t] * w_cb * r_cb
                + sims[:, t] * w_usgb * r_usgb
            )

        final_balances = sims[:, -1]
        avg_remaining = np.mean(final_balances)
        all_results.append({"w": (w_lcs, w_scs, w_cb, w_usgb), "avg_remain": avg_remaining})

    best_remain = max(all_results, key=lambda x: x["avg_remain"])
    best_w = best_remain["w"]
    best_avg_remain = best_remain["avg_remain"]

    # (F) 繪製 100,000 條路徑 + 平均走勢 (Trend)
    w_lcs, w_scs, w_cb, w_usgb = best_w
    sims_all = np.zeros((N, years + 1))
    sims_all[:, 0] = initial_balance
    for t in range(years):
        sims_all[:, t] -= withdraws[t]
        sims_all[:, t] = np.where(sims_all[:, t] < 0, 0.0, sims_all[:, t])
        r_lcs = np.random.choice(lcs_pct, size=N)
        r_scs = np.random.choice(scs_pct, size=N)
        r_cb = np.random.choice(cb_pct, size=N)
        r_usgb = np.random.choice(usgb_pct, size=N)

        sims_all[:, t + 1] = (
            sims_all[:, t] * w_lcs * r_lcs
            + sims_all[:, t] * w_scs * r_scs
            + sims_all[:, t] * w_cb * r_cb
            + sims_all[:, t] * w_usgb * r_usgb
        )

    mean_path = sims_all.mean(axis=0)
    os.makedirs("static/results", exist_ok=True)
    fn_trend = f"q5_trend_{int(w_lcs*100)}_{int(w_scs*100)}_{int(w_cb*100)}_{int(w_usgb*100)}.png"
    plt.figure(figsize=(12, 8))
    ages_plot = np.arange(60, 60 + years + 1)
    for i in range(N):
        plt.plot(ages_plot, sims_all[i, :], color="skyblue", alpha=0.03, linewidth=0.5)
    plt.plot(ages_plot, mean_path, color="red", linewidth=2, label="平均走勢")
    plt.xlabel("Age (歲數)")
    plt.ylabel("Accumulated Value (元)")
    plt.title(f"Trend (最佳配置: LCS={w_lcs:.2f}, SCS={w_scs:.2f}, CB={w_cb:.2f}, USGB={w_usgb:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("static", "results", fn_trend))
    plt.close()

    # (G) 回傳 dict 給 answer.html
    result = {
        "best_remain_w": best_w,
        "best_avg_remain": float(best_avg_remain),
        "trend_plot": fn_trend,
    }
    return result
