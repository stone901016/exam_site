# solvers/question2.py
import numpy as np
from scipy.stats import lognorm

def solve(_):
    # 投影片裡給的各科參數
    subjects = {
        "微積分":       {"mean":60, "sd":5, "credits":3},
        "統計學":       {"mean":60, "sd":5, "credits":3},
        "經濟學":       {"mean":65, "sd":5, "credits":3},
        "風管與保險":   {"mean":60, "sd":10,"credits":3},
        "計算機概論":   {"mean":70, "sd":10,"credits":2},
        "英文":         {"mean":70, "sd":10,"credits":2},
        "中國文學":     {"mean":70, "sd":5, "credits":3},
        "體育":         {"mean":75, "sd":10,"credits":1},
        "民法":         {"mean":70, "sd":5, "credits":2},
    }
    N = 100000
    # 1) 模擬每科成績，累加「及格後」學分
    total_credits = np.zeros(N)
    for sub, p in subjects.items():
        # lognormal 拟合底下參數 μ,σ
        mu = np.log(p["mean"]**2 / np.sqrt(p["sd"]**2 + p["mean"]**2))
        sigma = np.sqrt(np.log(1 + (p["sd"]**2)/(p["mean"]**2)))
        scores = lognorm(s=sigma, scale=np.exp(mu)).rvs(size=N)
        total_credits += p["credits"] * (scores >= 60)

    prob_ge16 = np.mean(total_credits >= 16)

    # 2) 三科同時及格
    prob_pass_three = 1
    for sub in ["微積分","統計學","民法"]:
        p = subjects[sub]
        mu = np.log(p["mean"]**2 / np.sqrt(p["sd"]**2 + p["mean"]**2))
        sigma = np.sqrt(np.log(1 + (p["sd"]**2)/(p["mean"]**2)))
        cdf60 = lognorm(s=sigma, scale=np.exp(mu)).cdf(60)
        prob_ge60 = 1 - cdf60
        prob_pass_three *= prob_ge60

    return {
        "P(學分≥16)": prob_ge16,
        "P(微積/統計/民法皆及格)": prob_pass_three
    }
