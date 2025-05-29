# -*- coding: utf-8 -*-
import os
import importlib
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import QUESTIONS

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER   = os.path.join(BASE_DIR, "data")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    q = next((x for x in QUESTIONS if x["id"] == qid), None)
    if not q:
        return "題目不存在", 404

    if request.method == "POST":
        f = request.files.get("data_file")
        if not f or f.filename == "":
            flash("請先選擇要上傳的檔案")
            return redirect(url_for("question", qid=qid))

        filename = secure_filename(f.filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        f.save(file_path)

        # ── 在這裡包 try/except ──
        try:
            mod    = importlib.import_module(f"solvers.{q['module']}")
            result = mod.solve(file_path)
            return render_template("answer.html", question=q, result=result)
        except Exception as e:
            tb = traceback.format_exc()
            app.logger.error(tb)  # 寫進 server log
            flash(f"伺服器執行錯誤：{e}")  # 前端顯示簡短錯誤
            return redirect(url_for("question", qid=qid))

    return render_template("question.html", question=q)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
