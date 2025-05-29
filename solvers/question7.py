# solvers/question7.py
import pandas as pd
import numpy as np

def solve(file_path):
    df = pd.read_excel(file_path)
    # 假設欄位: 'Abandon','Expansion','NPV3'
    results = {}
    for q in [0.95, 0.99]:
        best_val, best_combo = -np.inf, None
        for (a, e), g in df.groupby(["Abandon","Expansion"]):
            v = g["NPV3"].quantile(q)
            if v > best_val:
                best_val, best_combo = v, (a, e)
        results[f"q{int(q*100)}"] = {
            "Abandon": best_combo[0],
            "Expansion": best_combo[1],
            "QuantileValue": best_val
        }
    return results
