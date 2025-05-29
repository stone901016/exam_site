# solvers/question5.py
import os
import pandas as pd
import numpy as np

def solve(file_path):
    df = pd.read_excel(file_path)
    # 假設欄位: 'Portfolio'、'Year'、'Balance'、'Withdrawal'
    # 篩 2033 年
    df33 = df[df["Year"]==2033]
    # 分組計算平均
    grp = df33.groupby("Portfolio").agg({
        "Balance":"mean",
        "Withdrawal":"mean"
    }).reset_index()

    # 找最優
    best_balance = grp.loc[grp["Balance"].idxmax()]
    best_withdraw= grp.loc[grp["Withdrawal"].idxmax()]

    return {
        "最佳剩餘組合": best_balance["Portfolio"],
        "最佳提款組合": best_withdraw["Portfolio"]
    }
