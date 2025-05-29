# solvers/question1.py
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def solve(file_path):
    # 1. 讀資料
    df = pd.read_excel(file_path)
    if "賠償金額" not in df.columns:
        raise KeyError("檔案內找不到「賠償金額」欄位")
    amounts = df["賠償金額"]

    # 2. 取或算賠款率
    if "賠款率" in df.columns:
        ratios = df["賠款率"]
    elif "保險金額" in df.columns:
        ratios = amounts / df["保險金額"]
    else:
        raise KeyError("檔案內找不到「賠款率」或「保險金額」欄位")

    # 3. 參數擬合（以 lognormal 為例）
    params_amt = stats.lognorm.fit(amounts, floc=0)
    params_rat = stats.lognorm.fit(ratios,  floc=0)

    # 4. 繪製賠償金額 PDF
    x1 = np.linspace(amounts.min(), amounts.max(), 200)
    pdf1 = stats.lognorm.pdf(x1, *params_amt)
    fn_amt = "q1_amount.png"
    plt.figure()
    plt.plot(x1, pdf1)
    plt.title("賠償金額分布")
    plt.savefig(os.path.join("static","results",fn_amt))
    plt.close()

    # 5. VaR / CVaR（歷史值）
    var01 = np.percentile(amounts, 1)
    var05 = np.percentile(amounts, 5)
    cvar01 = amounts[amounts<=var01].mean()
    cvar05 = amounts[amounts<=var05].mean()

    # 6. VaR / CVaR（蒙地卡羅）
    sim = stats.lognorm.rvs(*params_amt, size=100000)
    mc_var1 = np.percentile(sim, 1)
    mc_cvar1 = sim[sim<=mc_var1].mean()

    return {
        "params_amt": params_amt,
        "params_rat": params_rat,
        "var": {"1%": var01, "5%": var05},
        "cvar": {"1%": cvar01, "5%": cvar05},
        "mc": {"var1%": mc_var1, "cvar1%": mc_cvar1},
        "plots": {"amount_dist": fn_amt}
    }
