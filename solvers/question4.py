# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # -------------------------------------------------------------------
    # 1) 嘗試從 Excel 讀取「基金 Mean/StdDev」與「股票 Mean/StdDev」
    #    如果讀不到，就用預設值（請自行改成專案要求的真實數字）
    # -------------------------------------------------------------------
    df = pd.read_excel(file_path)

    def try_get_float(col_name):
        if col_name in df.columns:
            tmp = df[col_name].dropna()
            try:
                return float(tmp.iloc[0])
            except:
                return None
        return None

    mean_fund  = try_get_float("基金 Mean")
    sd_fund    = try_get_float("基金 StdDev")
    mean_stock = try_get_float("股票 Mean")
    sd_stock   = try_get_float("股票 StdDev")

    if mean_fund is None or sd_fund is None or mean_stock is None or sd_stock is None:
        # 以下數值請改成你專案真實計算後得到的結果
        DEFAULT_MEAN_FUND  = 0.059   # 例如：基金年化平均 5.9%
        DEFAULT_SD_FUND    = 0.115   # 例如：基金年化標準差 11.5%
        DEFAULT_MEAN_STOCK = 0.147   # 例如：股票年化平均 14.7%
        DEFAULT_SD_STOCK   = 0.295   # 例如：股票年化標準差 29.5%

        mean_fund  = DEFAULT_MEAN_FUND
        sd_fund    = DEFAULT_SD_FUND
        mean_stock = DEFAULT_MEAN_STOCK
        sd_stock   = DEFAULT_SD_STOCK

    # -------------------------------------------------------------------
    # 2) 模擬參數設定
    #    - 投資人 30 歲 → 60 歲：共 31 年 (含 30、60)
    #    - 每年投入 10,000 元(考慮 2.2% 通膨)
    #    - 投資標的：股票與基金，均假設常態分布
    #    - 模擬次數 N = 10,000
    # -------------------------------------------------------------------
    years = 31
    N = 100_000
    inflation = 1.022  # 年通膨率 2.2%
    deposit = 10_000   # 每年投入金額

    weights = [0.5, 0.7, 0.9]  # 三種股票權重：50%、70%、90%

    # 回傳用字典：各權重對應的 Trend/CDF 檔名
    plots_trend = {}
    plots_cdf   = {}

    os.makedirs("static/results", exist_ok=True)

    ages = np.arange(30, 30 + years)  # [30,31,...,60]

    for w in weights:
        # sims[i,t] = 第 i 條模擬路徑，在第 t 年(30+t 歲) 的累積價值
        sims = np.zeros((N, years))

        # 針對 100,000 次模擬，各年投入並產生隨機報酬
        for i in range(N):
            for t in range(years):
                # 當年實際投入金額 (考慮通膨)
                dep = deposit * (inflation ** t)
                # 分別從常態分布抽取該年的股票報酬、基金報酬
                r_stock = np.random.normal(mean_stock, sd_stock)
                r_fund  = np.random.normal(mean_fund, sd_fund)
                # 投資組合報酬率
                r_port = w * r_stock + (1 - w) * r_fund

                if t == 0:
                    sims[i, t] = dep * (1 + r_port)
                else:
                    sims[i, t] = sims[i, t - 1] * (1 + r_port) + dep

        # -------------------- Trend 圖 --------------------
        fn_trend = f"q4_trend_{int(w*100)}.png"
        plt.figure(figsize=(9, 6))

        # 1) 先畫所有 100,000 條路徑 (淡藍色)
        for i in range(N):
            plt.plot(
                ages,
                sims[i, :],
                color="skyblue",
                linewidth=0.4,
                alpha=0.03
            )

        # 2) 再畫平均路徑 (粗紅線)
        mean_path = sims.mean(axis=0)
        plt.plot(
            ages,
            mean_path,
            color="red",
            linewidth=2.5,
            label="Average Path"
        )

        # X/Y 軸改回英文
        plt.xlabel("Age", fontsize=12)
        plt.ylabel("Accumulated Value", fontsize=12)
        plt.title(f"Trend (w={int(w*100)}% Stocks)", fontsize=14)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_trend))
        plt.close()
        plots_trend[w] = fn_trend

        # -------------------- CDF 圖 --------------------
        final_vals = sims[:, -1]           # 每條路徑在 60 歲 的最終累積值
        sorted_vals = np.sort(final_vals)  # 由小到大排序
        cdf = np.arange(1, N + 1) / N

        fn_cdf = f"q4_cdf_{int(w*100)}.png"
        plt.figure(figsize=(9, 6))
        plt.plot(
            sorted_vals,
            cdf,
            color="green",
            linewidth=2
        )

        # CDF 圖 X/Y 軸也用英文
        plt.xlabel("Final Accumulated Value", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.title(f"CDF (w={int(w*100)}% Stocks)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_cdf))
        plt.close()
        plots_cdf[w] = fn_cdf

    # 回傳給樣板
    return {
        "plots_trend": plots_trend,
        "plots_cdf": plots_cdf
    }
