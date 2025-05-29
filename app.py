# -*- coding: utf-8 -*-
import os
import importlib
from flask import Flask, render_template, request, redirect, url_for, flash
from config import QUESTIONS

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER   = os.path.join(BASE_DIR, "data")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html", questions=QUESTIONS)

@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    q = next((x for x in QUESTIONS if x["id"]==qid), None)
    if not q:
        return "題目不存在", 404

    files = os.listdir(DATA_FOLDER)
    if request.method == "POST":
        fname = request.form.get("data_file")
        if not fname:
            flash("請選擇一個檔案")
            return redirect(url_for("question", qid=qid))

        file_path = os.path.join(DATA_FOLDER, fname)
        mod = importlib.import_module(f"solvers.{q['module']}")
        result = mod.solve(file_path)

        return render_template("answer.html", question=q, result=result)
    return render_template("question.html", question=q, files=files)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=True)
