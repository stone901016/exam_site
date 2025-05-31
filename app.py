# app.py
import os
import uuid
import time
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ProcessPoolExecutor
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/tmp"  
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  

# ─── 1) 全域 jobs 字典，用來記錄各 job 狀態與結果 ─────────────────────────────
# 格式範例：
# jobs = {
#   "uuid-string-1": {
#       "status": "pending",  # 可為 "pending" / "running" / "finished" / "error"
#       "result": None,       # 最後 run_full_simulation 回傳的 dict
#       "error": None         # 如果執行錯誤，就放 Exception 字串
#   },
#   "uuid-string-2": { … }
# }
jobs = {}

# ─── 2) 建立背景工作池 (ProcessPoolExecutor) ───────────────────────────────────
executor = ProcessPoolExecutor(max_workers=2)


# ─── 3) 背景工作函式：真正執行第 5 題的模擬，跑完之後更新 jobs ────────────────
def background_question5(job_id, saved_file_path):
    """
    這邊會在另一個 process/thread 中執行 (由 executor.submit 呼叫)，
    用來跑 run_full_simulation(...) 這個耗時巨大的函式。
    成功跑完之後，把 jobs[job_id]['status'] 設成 "finished"，並把 result 設回字典。
    如果發生 Exception，則把 status 設為 "error"，並把錯誤訊息存到 jobs[job_id]['error']。
    """
    from solvers.question5 import run_full_simulation

    try:
        # 1) 將狀態從 pending -> running
        jobs[job_id]["status"] = "running"
        print(f"[BACKGROUND] ({job_id}) 開始執行 run_full_simulation…")

        # 2) 真正跑 100,000 次模擬＋權重格點搜尋
        result_dict = run_full_simulation(saved_file_path, n_sim=100_000)

        # 3) 完成後，更新狀態與結果
        jobs[job_id]["status"] = "finished"
        jobs[job_id]["result"] = result_dict
        print(f"[BACKGROUND] ({job_id}) run_full_simulation 已完成。")

    except Exception as e:
        # 若發生任何 Exception，就把狀態設成 error，並且把錯誤訊息放進去
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"[BACKGROUND] ({job_id}) 執行時出錯：{e}")


# ─── 4) /question/<qid> 路由：第 1～4 題為同步呼叫，第 5 題改成背景 + 輪詢 ─────────
@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    # 如果 qid == 5，啟動特殊的「背景執行」流程
    if qid == 5:
        from solvers.question5 import QUESTION as q5_info

        if request.method == "GET":
            # 只顯示回答頁面模板 (還未產生結果)
            return render_template("answer.html", question=q5_info, result=None)

        # POST：上傳檔案後，立即開一個 background job，並回傳 job_id
        if "file" not in request.files:
            return "請務必上傳檔案", 400

        uploaded_file = request.files["file"]
        filename = secure_filename(uploaded_file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(saved_path)

        # 產生一個 job_id
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "result": None, "error": None}

        # 提交給背景工作池去執行，立刻回應不用等
        executor.submit(background_question5, job_id, saved_path)

        # 回傳 job_id 給前端 (HTTP 202 Accepted)
        return jsonify({"job_id": job_id}), 202

    # 如果 qid 在 1~4 範圍內，維持原本同步執行的流程
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
        return render_template("answer.html", question=mod.QUESTION, result=None)

    # POST：上傳檔案後，同步呼叫 mod.solve(...)
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


# ─── 5) /question/5/status/<job_id>：查詢第 5 題背景工作狀態 ──────────────────────
@app.route("/question/5/status/<job_id>", methods=["GET"])
def question5_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "job_id not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "status": job["status"],   # pending / running / finished / error
        "error": job["error"],     # 如果執行失敗，顯示錯誤訊息
        "result": job["result"]    # 當 status=="finished" 時，才會是完整的 dict
    })


# ─── 6) 首頁（無須修改） ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Gunicorn 上線時請自行把 timeout 設長一點 (或根本不需理會，因為已改成 background)
    app.run(host="0.0.0.0", port=8080, debug=True)
