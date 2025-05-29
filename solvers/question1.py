# solvers/question1.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def solve(file_path):
    df = pd.read_excel(file_path)
    amounts = df["賠償金額"]
    ratios  = df["賠款率"]

    params_amt = stats.lognorm.fit(amounts, floc=0)
    x = np.linspace(amounts.min(), amounts.max(), 200)
    pdf = stats.lognorm.pdf(x, *params_amt)
    out = "q1_amount.png"
    plt.figure()
    plt.plot(x, pdf)
    plt.title("賠償金額分布")
    plt.savefig(os.path.join("static/results", out))
    plt.close()

    var01 = np.percentile(amounts, 1)
    cvar01 = amounts[amounts <= var01].mean()

    return {
        "params_amt": params_amt,
        "var": {"1%": var01},
        "cvar": {"1%": cvar01},
        "plots": {"amount_dist": out}
    }
