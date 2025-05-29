# solvers/question9.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solve(file_path):
    df = pd.read_excel(file_path)
    # 假設欄位: 'DiscountRate','MarketShare','NPV'
    # Tornado：對每個變數分別取最大 / 最小 NPV 差距
    base = df[(df["DiscountRate"]==df["DiscountRate"].median()) & 
              (df["MarketShare"]==15)]["NPV"].mean()

    deltas = {}
    # DiscountRate 變動
    for v in sorted(df["DiscountRate"].unique()):
        val = df[(df["DiscountRate"]==v)&(df["MarketShare"]==15)]["NPV"].mean()
        deltas[f"DR {v}%"] = val - base
    # MarketShare 變動 (12,15,18)
    for ms in [12,15,18]:
        val = df[(df["DiscountRate"]==50)&(df["MarketShare"]==ms)]["NPV"].mean()
        deltas[f"MS {ms}"] = val - base

    # 畫 Tornado
    fn = "q9_tornado.png"
    items = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
    labels, values = zip(*items)
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(6,4))
    plt.barh(y_pos, values, align='center')
    plt.yticks(y_pos, labels)
    plt.title("Tornado Chart")
    plt.savefig(os.path.join("static","results",fn))
    plt.close()

    return {
        "deltas": deltas,
        "plots": {"tornado": fn}
    }
