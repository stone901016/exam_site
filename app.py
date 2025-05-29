# -*- coding: utf-8 -*-
import os
import importlib
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import QUESTIONS

app = Flask(__name__)
# 部署時用環境變數設定 SECRET_KEY
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER  = os.path.join(BASE_DIR, "data")
RESULT_FOLDER= os.path.join(BASE_DIR, "static", "results")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    # 找到題目設定
    q = next((x for x in QUESTIONS if x["id"] == qid), None)
    if not q:
        return "題目不存在", 404

    if request.method == "POST":
        # 1. 處理上傳檔案
        f = request.files.get("data_file")
        if not f or f.filename == "":
            flash("請先選擇一個檔案來上傳")
            return redirect(url_for("question", qid=qid))

        filename = secure_filename(f.filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        f.save(file_path)

        # 2. 動態載入對應 solver 並計算
        mod = importlib.import_module(f"solvers.{q['module']}")
        result = mod.solve(file_path)

        # 3. 回傳答案頁
        return render_template("answer.html", question=q, result=result)

    # GET 則顯示上傳表單
    return render_template("question.html", question=q)

if __name__ == "__main__":
    # PORT 環境變數由 Railway 自動注入
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
