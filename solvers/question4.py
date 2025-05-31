# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # ----------------------------------------------------
    # 1) 嘗試從 Excel 讀取「基金 Mean/StdDev」與「股票 Mean/StdDev」
    #    若找不到，就用預設值（請自行改成專案真的計算值）
    # ----------------------------------------------------
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
        # 下面範例數值請改成專案實際計算後的結果
        DEFAULT_MEAN_FUND  = 0.059   # 例如：基金年化平均 5.9%
        DEFAULT_SD_FUND    = 0.115   # 例如：基金年化標準差 11.5%
        DEFAULT_MEAN_STOCK = 0.147   # 例如：股票年化平均 14.7%
        DEFAULT_SD_STOCK   = 0.295   # 例如：股票年化標準差 29.5%

        mean_fund  = DEFAULT_MEAN_FUND
        sd_fund    = DEFAULT_SD_FUND
        mean_stock = DEFAULT_MEAN_STOCK
        sd_stock   = DEFAULT_SD_STOCK

    # ----------------------------------------------------
    # 2) 模擬參數設定
    #    - 投資年齡：30→60 歲，共 31 年
    #    - 每年投入 10,000 元（含 2.2% 通膨）
    #    - 投資標的：股票與基金（皆假設常態分布）
    #    - 模擬次數 N = 100,000（由需求指定）
    # ----------------------------------------------------
    years = 31
    N = 100_000            # 模擬 100,000 次
    inflation = 1.022      # 年通膨率 2.2%
    deposit = 10_000       # 每年投入金額

    weights = [0.5, 0.7, 0.9]  # 三種股票權重（50%、70%、90%）

    # 回傳用：各權重對應的趨勢圖 / CDF 圖檔名
    plots_trend = {}
    plots_cdf   = {}

    os.makedirs("static/results", exist_ok=True)
    ages = np.arange(30, 30 + years)  # [30, 31, ..., 60]

    for w in weights:
        # 1) 建立 sims 儲存所有 N 條路徑、31 年的累積價值
        sims = np.zeros((N, years))

        # 2) 進行 N 次模擬
        #    注意：此迴圈可能需要一些運算時間
        for i in range(N):
            for t in range(years):
                dep = deposit * (inflation ** t)
                # 每年先從常態分布抽取該年的股票報酬、基金報酬
                r_stock = np.random.normal(mean_stock, sd_stock)
                r_fund  = np.random.normal(mean_fund, sd_fund)
                # 投資組合報酬率
                r_port  = w * r_stock + (1 - w) * r_fund

                if t == 0:
                    sims[i, t] = dep * (1 + r_port)
                else:
                    sims[i, t] = sims[i, t - 1] * (1 + r_port) + dep

        # ---------------------------------------------
        # (A) Trend 圖：隨機抽 2,000 條路徑 + 平均路徑
        # ---------------------------------------------
        fn_trend = f"q4_trend_{int(w*100)}.png"
        plt.figure(figsize=(12, 8))

        # 隨機抽樣 2,000 條作為代表
        if N > 2000:
            idxs = np.random.choice(N, size=2000, replace=False)
        else:
            idxs = np.arange(N)

        for idx in idxs:
            plt.plot(
                ages,
                sims[idx, :],
                color="skyblue",
                linewidth=0.5,
                alpha=0.2
            )

        # 計算所有 100,000 條路徑的平均累積走勢
        mean_path = sims.mean(axis=0)
        plt.plot(
            ages,
            mean_path,
            color="red",
            linewidth=3,
            label="Average Path"
        )

        plt.xlabel("Age", fontsize=14)
        plt.ylabel("Accumulated Value", fontsize=14)
        plt.title(f"Trend (w={int(w*100)}% Stocks)", fontsize=16)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_trend))
        plt.close()
        plots_trend[w] = fn_trend

        # ---------------------------------------------
        # (B) CDF 圖：針對 60 歲的最終累積值
        # ---------------------------------------------
        final_vals = sims[:, -1]
        sorted_vals = np.sort(final_vals)
        cdf = np.arange(1, N + 1) / N

        fn_cdf = f"q4_cdf_{int(w*100)}.png"
        plt.figure(figsize=(12, 8))
        plt.plot(
            sorted_vals,
            cdf,
            color="green",
            linewidth=2
        )
        plt.xlabel("Final Accumulated Value", fontsize=14)
        plt.ylabel("Cumulative Probability", fontsize=14)
        plt.title(f"CDF (w={int(w*100)}% Stocks)", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_cdf))
        plt.close()
        plots_cdf[w] = fn_cdf

    # 回傳給 answer.html
    return {
        "plots_trend": plots_trend,
        "plots_cdf": plots_cdf
    }
