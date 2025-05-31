# solvers/question4.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    # -------------------------------------------------------------------
    # 1) 先從 Excel 嘗試抓取「基金 Mean」、「基金 StdDev」、「股票 Mean」、「股票 StdDev」
    # 格式化檢查：如果欄位不存在或含非數值，則套用預設值
    # 預設值示例：基金年化平均 6%、年化標準差 12%；股票年化平均 8%、年化標準差 18%
    # 你可依照專案實際數值自行調整 DEFAULT_* 這四個數字
    # -------------------------------------------------------------------
    df = pd.read_excel(file_path)

    # 嘗試自 Excel 找到該欄位並轉成 float
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

    # 如果其中任一值抓不到，就套用專案給定的預設數值
    if mean_fund is None or sd_fund is None or mean_stock is None or sd_stock is None:
        # 以下四個數值需改成你專案「Accumulate.xls」或其他題目中所提供的資料計算出來的結果
        DEFAULT_MEAN_FUND  = 0.059  # 例如：基金年化平均約 5.9%
        DEFAULT_SD_FUND    = 0.115  # 例如：基金年化標準差約 11.5%
        DEFAULT_MEAN_STOCK = 0.147  # 例如：股票年化平均約 14.7%
        DEFAULT_SD_STOCK   = 0.295  # 例如：股票年化標準差約 29.5%

        mean_fund  = DEFAULT_MEAN_FUND
        sd_fund    = DEFAULT_SD_FUND
        mean_stock = DEFAULT_MEAN_STOCK
        sd_stock   = DEFAULT_SD_STOCK

    # -------------------------------------------------------------------
    # 2) 模擬參數設定
    # 投資人從 30 歲到 60 歲退休：共 31 年(含 30、60)
    # 每年投入 10,000 元(考慮 2.2% 通膨)
    # 每年投資部分放在股票、部分放在基金；各自採常態分布隨機
    # 模擬次數 N = 10,000
    # -------------------------------------------------------------------
    years = 31
    N = 10_000
    inflation = 1.022  # 年通膨率 2.2%
    deposit = 10_000  # 每年投入金額

    # 欲比較的三種股票權重
    weights = [0.5, 0.7, 0.9]

    # 最後要回傳：trend 圖和 cdf 圖的檔名
    plots_trend = {}  # { 0.5: "q4_trend_50.png", ... }
    plots_cdf   = {}

    # 建立目錄
    os.makedirs("static/results", exist_ok=True)

    # 每個權重依序模擬
    for w in weights:
        # sims[i,t] = 第 i 次模擬，到第 t 年後的累積價值 (t=0..30)
        sims = np.zeros((N, years))
        ages = np.arange(30, 30 + years)  # [30,31,...,60]

        # 每次模擬各年投入 + 投資報酬
        for i in range(N):
            # 每條路徑從零開始
            prev_val = 0.0
            for t in range(years):
                # 當年投入考量通膨：第 t 年的折算金額 = deposit * (inflation ** t)
                dep = deposit * (inflation ** t)

                # 隨機取一筆股票回報、基金回報 (年化常態)
                r_stock = np.random.normal(mean_stock, sd_stock)
                r_fund  = np.random.normal(mean_fund, sd_fund)

                # 投資組合當年報酬
                r_portfolio = w * r_stock + (1 - w) * r_fund

                # 計算當年 End Value
                if t == 0:
                    sims[i, t] = dep * (1 + r_portfolio)
                else:
                    sims[i, t] = sims[i, t - 1] * (1 + r_portfolio) + dep

        # 2.1) 繪製 Trend 圖 (所有路徑 + 平均路徑)
        fn_trend = f"q4_trend_{int(w*100)}.png"
        plt.figure(figsize=(9, 6))
        # 先把所有 N 條路徑都畫出來 (淡藍色)
        for i in range(N):
            plt.plot(ages, sims[i, :], color="skyblue", linewidth=0.5, alpha=0.02)

        # 再畫上「平均路徑」(粗紅線)
        mean_path = sims.mean(axis=0)  # 每個年度的平均累積值
        plt.plot(ages, mean_path, color="red", linewidth=2.5, label="平均路徑")

        # 中文 X/Y 軸
        plt.xlabel("年齡", fontsize=12)
        plt.ylabel("累積價值 (元)", fontsize=12)
        plt.title(f"Trend (w={int(w*100)}% 股票)", fontsize=14)
        plt.legend(loc="upper left", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_trend))
        plt.close()
        plots_trend[w] = fn_trend

        # 2.2) 繪製 CDF 圖 (取 sims[:, -1]：最後一年 60 歲 的累積值)
        final_vals = sims[:, -1]
        sorted_vals = np.sort(final_vals)
        cdf = np.arange(1, N + 1) / N

        fn_cdf = f"q4_cdf_{int(w*100)}.png"
        plt.figure(figsize=(9, 6))
        plt.plot(sorted_vals, cdf, color="green", linewidth=2)
        plt.xlabel("最終累積價值 (元)", fontsize=12)
        plt.ylabel("累積機率", fontsize=12)
        plt.title(f"CDF (w={int(w*100)}% 股票)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("static", "results", fn_cdf))
        plt.close()
        plots_cdf[w] = fn_cdf

    # 回傳：只需要把兩組字典交給樣板即可
    return {
        "plots_trend": plots_trend,
        "plots_cdf": plots_cdf
    }
