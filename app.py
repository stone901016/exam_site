import os
import uuid
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ProcessPoolExecutor
from werkzeug.utils import secure_filename
from types import SimpleNamespace

# 注意：請先安裝 Flask
#   pip install Flask

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/tmp"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 例如 50MB

# ---------------------------------------------
# (一) 全域 job 狀態字典：儲存第 5 題背景工作進度與結果
# ---------------------------------------------
jobs = {}
# 範例格式：
# jobs = {
#   "some-uuid-1": {
#       "status": "pending",    # pending / running / finished / error
#       "result": None,         # 完成時由 run_full_simulation 回傳的 dict
#       "error": None           # 若背景工作發生例外，把錯誤字串放在這裡
#   },
#   ...
# }

# ---------------------------------------------
# (二) 建立 ProcessPoolExecutor，專門跑第 5 題的耗時模擬
# ---------------------------------------------
executor = ProcessPoolExecutor(max_workers=2)


# ---------------------------------------------
# (三) 第 5 題的背景執行函式：真正呼叫 solvers.question5.run_full_simulation()
# ---------------------------------------------
def background_question5(job_id, saved_file_path):
    """
    在背景中執行第5題的模擬流程。成功時把結果寫到 jobs[job_id]["result"]，
    若例外則記錄到 jobs[job_id]["error"]，並把狀態標成 'error'。
    """
    try:
        from solvers.question5 import run_full_simulation
    except ImportError:
        # 如果找不到 question5.py，直接把狀態設成 error
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = "無法匯入 solvers.question5 模組，請檢查是否存在。"
        return

    # 把 job 狀態設為 running
    jobs[job_id]["status"] = "running"
    try:
        # n_sim 參數預設改為 100000
        result_dict = run_full_simulation(saved_file_path, n_sim=100000)
        jobs[job_id]["status"] = "finished"
        jobs[job_id]["result"] = result_dict
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ---------------------------------------------
# (四) /question/<qid>：第 1~4 題 (同步) & 第 5 題 (背景 + 非同步輪詢)
# ---------------------------------------------
@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    # 先把模組對應表準備好
    from solvers import question1, question2, question3, question4
    solvers_map = {
        1: question1,
        2: question2,
        3: question3,
        4: question4
        # 5 題不放在這裡，因為 5 題要改成 Background Task
    }

    # 如果是第 5 題，要啟動「背景工作 + 輪詢」流程
    if qid == 5:
        # GET: 只顯示「上傳檔 + 產生答案」表單，並留空 result（顯示 status=pending）
        if request.method == "GET":
            # 要把 question 也傳給 template：先嘗試取 solvers.question5.QUESTION
            # 若不存在，就自行產生一個簡易的 Namespace
            try:
                from solvers.question5 import QUESTION as q5_info
                question_info = q5_info
            except (ImportError, AttributeError):
                question_info = SimpleNamespace(id=5, title="題目 5 標題")

            return render_template("answer.html", question=question_info, result=None)

        # POST: 收到檔案後，就立刻回傳 job_id，並把背景工作交給 executor
        if "file" not in request.files:
            return "請務必上傳檔案", 400

        uploaded_file = request.files["file"]
        filename = secure_filename(uploaded_file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(saved_path)

        # 產生 job_id，並在 jobs 字典中紀錄初始狀態
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending", "result": None, "error": None}

        # 提交給背景執行，不等待結果
        executor.submit(background_question5, job_id, saved_path)

        # 立刻回傳 job_id 給前端，HTTP status 202 (Accepted)
        return jsonify({"job_id": job_id}), 202

    # 如果不是第 5 題，就走原本同步邏輯 (第 1~4 題)
    if qid not in solvers_map:
        return "此題目尚未實作", 404

    mod = solvers_map[qid]
    # GET: 僅顯示「上傳檔 + 產生答案」表單
    if request.method == "GET":
        # 同樣的，先嘗試用 mod.QUESTION 並傳給前端；若不存在，就隨便產生一個 Namespace
        try:
            question_info = mod.QUESTION
        except AttributeError:
            question_info = SimpleNamespace(id=qid, title=f"題目 {qid} 標題")
        return render_template("answer.html", question=question_info, result=None)

    # POST: 使用者上傳檔案，同步呼叫 mod.solve()，再 render 結果
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

    # 同樣地，保險起見給 question_info
    try:
        question_info = mod.QUESTION
    except AttributeError:
        question_info = SimpleNamespace(id=qid, title=f"題目 {qid} 標題")

    return render_template("answer.html", question=question_info, result=result)


# ---------------------------------------------
# (五) 第 5 題的輪詢 API：/question/5/status/<job_id>
# ---------------------------------------------
@app.route("/question/5/status/<job_id>", methods=["GET"])
def question5_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "job_id not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "status": job["status"],
        "error": job["error"],
        "result": job["result"]
    })


# ---------------------------------------------
# (六) 首頁 & 其他
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(404)
def page_not_found(e):
    return "404: 此題目尚未實作", 404


if __name__ == "__main__":
    # 注意：如果直接用 flask run 或 python app.py，debug=True 方便開發。
    app.run(host="0.0.0.0", port=8080, debug=True)
