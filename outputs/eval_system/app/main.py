from fastapi import FastAPI, Request, Form, Depends, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import threading
import time

from . import models, auth, rate_limiter, queue_manager, security, compiler, profiler

app = FastAPI(title="CUDA Online Evaluation System", version="1.0.0")
bearer_scheme = HTTPBearer(auto_error=False)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "app", "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

models.init_db()

# --- Auth dependency ---
def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    token = request.cookies.get("session_token")
    if token:
        user = auth.decode_token(token)
        if user:
            return user
    if credentials:
        user = auth.decode_token(credentials.credentials)
        if user:
            return user
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = auth.authenticate_api_key(api_key)
        if user:
            return user
    return None


def require_user(user: dict | None = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def require_admin(user: dict = Depends(require_user)):
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# --- Web pages ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: dict = Depends(require_user)):
    jobs = queue_manager.get_user_jobs(user["id"])
    rate_info = rate_limiter.get_user_rate_limit_info(user["id"])
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user, "jobs": jobs, "rate_info": rate_info,
    })


@app.get("/submit", response_class=HTMLResponse)
async def submit_page(request: Request, user: dict = Depends(require_user)):
    rate_info = rate_limiter.get_user_rate_limit_info(user["id"])
    return templates.TemplateResponse("submit.html", {
        "request": request, "user": user, "rate_info": rate_info,
    })


@app.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request, user: dict = Depends(require_user)):
    jobs = queue_manager.get_user_jobs(user["id"])
    return templates.TemplateResponse("queue.html", {
        "request": request, "user": user, "jobs": jobs,
    })


@app.get("/result/{job_id}", response_class=HTMLResponse)
async def result_page(request: Request, job_id: int, user: dict = Depends(require_user)):
    job = queue_manager.get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="任务不存在")
    return templates.TemplateResponse("result.html", {
        "request": request, "user": user, "job": job,
    })


# --- API endpoints ---
@app.post("/api/auth/login")
async def api_login(
    username: str = Form(...),
    password: str = Form(...),
    use_cookie: bool = Form(False),
):
    user = auth.authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = auth.create_token(user["id"], user["username"], user["is_admin"])
    if use_cookie:
        response = JSONResponse({"ok": True, "user": user})
        response.set_cookie("session_token", token, httponly=True)
        return response
    return {"token": token, "user": user}


@app.post("/api/auth/logout")
async def api_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie("session_token")
    return response


@app.post("/api/submit")
async def api_submit(
    request: Request,
    user: dict = Depends(require_user),
    code: str = Form(""),
    file: UploadFile | None = File(None),
):
    if file and file.filename:
        source_code = (await file.read()).decode("utf-8", errors="replace")
    elif code:
        source_code = code
    else:
        raise HTTPException(status_code=400, detail="请提供代码（粘贴或上传文件）")

    allowed, remaining, retry = rate_limiter.check_rate_limit(user["id"])
    if not allowed:
        raise HTTPException(status_code=429, detail=f"提交频率超限，请{retry}秒后重试")

    safe, error_msg = security.scan_code(source_code)
    if not safe:
        raise HTTPException(status_code=400, detail=f"安全检查未通过：{error_msg}")

    job_id = queue_manager.enqueue_job(user["id"], source_code)
    position = queue_manager.get_user_queue_position(user["id"], job_id)

    return {
        "ok": True,
        "job_id": job_id,
        "queue_position": position,
        "remaining_tokens": remaining,
    }


@app.get("/api/queue")
async def api_queue(user: dict = Depends(require_user)):
    jobs = queue_manager.get_user_jobs(user["id"])
    rate_info = rate_limiter.get_user_rate_limit_info(user["id"])
    return {"jobs": jobs, "rate_info": rate_info}


@app.get("/api/result/{job_id}")
async def api_result(job_id: int, user: dict = Depends(require_user)):
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权访问")
    return {"job": job}


@app.get("/api/download/{job_id}")
async def api_download(job_id: int, user: dict = Depends(require_user)):
    job = queue_manager.get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="文件不存在")
    if not job["ncu_report_path"]:
        raise HTTPException(status_code=404, detail="NCU报告尚未生成")
    path = os.path.join(profiler.RESULTS_DIR, job["ncu_report_path"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="报告文件丢失")
    return FileResponse(path, filename=job["ncu_report_path"])


# --- Admin pages ---
@app.get("/admin/", response_class=HTMLResponse)
async def admin_index(request: Request, user: dict = Depends(require_admin)):
    jobs = queue_manager.get_all_jobs(100)
    db = models.get_db()
    users = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    db.close()
    return templates.TemplateResponse("admin/index.html", {
        "request": request, "user": user, "jobs": jobs, "all_users": [dict(u) for u in users],
    })


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request, user: dict = Depends(require_admin)):
    db = models.get_db()
    users = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    db.close()
    return templates.TemplateResponse("admin/users.html", {
        "request": request, "user": user, "all_users": [dict(u) for u in users],
    })


@app.post("/admin/users/create")
async def admin_create_user(
    username: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    rate_limit: int = Form(5),
    admin_user: dict = Depends(require_admin),
):
    try:
        new_user = auth.create_user(username, password, is_admin, rate_limit)
        return {"ok": True, "user": new_user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/users/{target_id}/toggle")
async def admin_toggle_user(target_id: int, admin_user: dict = Depends(require_admin)):
    db = models.get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (target_id,)).fetchone()
    if user:
        new_active = 0 if user["is_active"] else 1
        db.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_active, target_id))
        db.commit()
    db.close()
    return {"ok": True}


@app.post("/admin/users/{target_id}/rate-limit")
async def admin_update_rate_limit(
    target_id: int,
    rate_limit: int = Form(...),
    admin_user: dict = Depends(require_admin),
):
    rate_limiter.update_user_rate_limit(target_id, rate_limit)
    return {"ok": True}


@app.delete("/admin/queue/{job_id}")
async def admin_cancel_job(job_id: int, admin_user: dict = Depends(require_admin)):
    queue_manager.cancel_job(job_id)
    return {"ok": True}


@app.get("/admin/config", response_class=HTMLResponse)
async def admin_config_page(request: Request, user: dict = Depends(require_admin)):
    db = models.get_db()
    configs = db.execute("SELECT * FROM config").fetchall()
    db.close()
    return templates.TemplateResponse("admin/config.html", {
        "request": request, "user": user, "configs": [dict(c) for c in configs],
    })


@app.post("/admin/config")
async def admin_update_config(
    key: str = Form(...),
    value: str = Form(...),
    admin_user: dict = Depends(require_admin),
):
    db = models.get_db()
    db.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value))
    db.commit()
    db.close()
    return {"ok": True}


# --- Background worker ---
def worker_loop():
    """Background thread that processes jobs from the queue."""
    while True:
        try:
            job = queue_manager.process_next_job()
            if job is None:
                time.sleep(2)
                continue

            job_id = job["id"]
            source = job["source_code"]

            success, output, binary_path = compiler.compile_cuda(source)
            queue_manager.update_job_status(job_id, "compiling", compile_output=output)

            if not success or binary_path is None:
                queue_manager.update_job_status(
                    job_id, "failed",
                    error_message=f"编译失败：{output}",
                )
                continue

            queue_manager.update_job_status(job_id, "profiling")

            success, summary, report_name = profiler.run_ncu_profile(binary_path)
            if success and report_name:
                queue_manager.update_job_status(
                    job_id, "completed",
                    profile_summary=summary,
                    ncu_report_path=report_name,
                )
            else:
                queue_manager.update_job_status(
                    job_id, "failed",
                    error_message=summary,
                )

        except Exception as e:
            try:
                queue_manager.update_job_status(job_id, "failed", error_message=str(e))
            except Exception:
                pass
            time.sleep(1)


@app.on_event("startup")
async def startup():
    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()
