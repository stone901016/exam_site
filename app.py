# app.py
import os
import uuid
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ProcessPoolExecutor
from werkzeug.utils import secure_filename

# 請確保安裝了 Flask：
# pip install Flask

app = Flask(__name__)
# 上傳檔案要先存在某個暫存資料夾，以下以 /tmp 為例
app.config["UPLOAD_FOLDER"] = "/tmp"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB 上限

# ---------------------------------------------
# 1. 全域字典 `jobs`：儲存所有 background-job 的狀態與結果
#    格式範例：
#    {
#      "some-uuid-1": {
#         "status": "pending",    # pending/running/finished/error
#         "result": None,         # 等到 finished 時，放真正要回前端的 result_dict
#         "error": None           # 如果背景發生 Exception，放錯誤訊息 (str)
#      },
#      ...
#    }
# ---------------------------------------------
jobs = {}

# ---------------------------------------------
# 2. 建立一個 ProcessPoolExecutor，讓第 5 題的計算可以丟到背景執行
#    max_workers 可依照您的 CPU 核心調整，以下示範 2 個核心
# ---------------------------------------------
executor = ProcessPoolExecutor(max_workers=2)


# ---------------------------------------------
# 3. background_question5(job_id, saved_file_path)
#    真正的背景工作：呼叫 solvers/question5.py 裡耗時的函式
#    然後把結果填回 jobs[job_id]
# ---------------------------------------------
def background_question5(job_id, saved_file_path):
    # 以避免 circular import，我們在函式內才 import
    from solvers.question5 import run_full_simulation

    try:
        # 先把狀態標記成 running
        jobs[job_id]["status"] = "running"
        print(f"[BACKGROUND] ({job_id}) 開始執行 run_full_simulation…") 

        # 呼叫真正耗時的函式，並取得 result_dict
        # 您的 run_full_simulation 會做 100000 次模擬 + 格點搜尋 + 繪圖，最後回傳 dict
        result_dict = run_full_simulation(saved_file_path, n_sim=100_000)
        print(f"[BACKGROUND] ({job_id}) run_full_simulation 執行完畢，準備存結果。")

        # 計算成功：把狀態改成 finished，並把 result 塞進去
        jobs[job_id]["status"] = "finished"
        jobs[job_id]["result"] = result_dict

    except Exception as e:
        # 若任何錯誤，就把狀態標示成 error，並儲存錯誤字串
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ---------------------------------------------
# 4. /question/<qid>：所有題目的入口
#    - 如果 qid == 5，改用「背景執行 + 回傳 job_id」的流程
#    - 其他題目 (1~4)，維持同步呼叫各自 solvers.questionX.solve(...)
# ---------------------------------------------
@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    # 針對第 5 題做特別處理：Background Task
    if qid == 5:
        # GET /question/5：單純顯示 answer.html (result=None)，讓使用者上傳檔案
        if request.method == "GET":
            from solvers.question5 import QUESTION as q5_info
            return render_template("answer.html", question=q5_info, result=None)

        # POST /question/5：使用者按下「產生答案」，啟動背景工作並回傳 job_id
        if "file" not in request.files:
            return "請務必上傳 sustainableRetirementWithdrawals.xls", 400

        uploaded_file = request.files["file"]
        filename = secure_filename(uploaded_file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(saved_path)

        # 產生一個唯一的 job_id，並在 jobs dict 裡標記為 pending
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "result": None, "error": None}

        # 把真正耗時計算的函式丟給背景去跑 (不等待)
        executor.submit(background_question5, job_id, saved_path)

        # 立即回傳 job_id 給前端，HTTP 狀態 202 (Accepted)
        return jsonify({"job_id": job_id}), 202

    # 如果 qid != 5，維持原本同步流程
    from solvers import question1, question2, question3, question4
    solvers_map = {
        1: question1,
        2: question2,
        3: question3,
        4: question4
    }
    if qid not in solvers_map:
        return "此題目尚未實作", 404

    mod = solvers_map[qid]
    if request.method == "GET":
        # 顯示 answer.html (result=None)，讓使用者上傳檔案
        return render_template("answer.html", question=mod.QUESTION, result=None)

    # 同步 POST：存檔 & 呼叫 mod.solve(...) 並 render
    if "file" not in request.files:
        return "請務必上傳檔案", 400

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(saved_path)

    try:
        result = mod.solve(saved_path)
    except Exception as e:
        return f"伺服器執行錯誤：{str(e)}", 500

    return render_template("answer.html", question=mod.QUESTION, result=result)


# ---------------------------------------------
# 5. /question/5/status/<job_id>：查詢第 5 題背景工作狀態
#    前端會以 AJAX / fetch 每隔幾秒輪詢這個 endpoint，直到狀態為 "finished" 或 "error"
# ---------------------------------------------
@app.route("/question/5/status/<job_id>", methods=["GET"])
def question5_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "job_id not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "status": job["status"],   # pending / running / finished / error
        "error": job["error"],     # 如果有 error，在前端顯示
        "result": job["result"]    # 只有 status == "finished" 時，這裡才會是 dict
    })


# ---------------------------------------------
# 6. 首頁或其他路由
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # 本地開發可保持 debug=True。部署時請改用 gunicorn，並設定 --timeout 600 等參數
    app.run(host="0.0.0.0", port=8080, debug=True)
