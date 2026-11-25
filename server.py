import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

import auth

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", 10000))

# Subprocesses for the 3 internal services
processes = []

INTERNAL_SERVICES = [
    {
        "name": "CRM Dashboard & API",
        "port": 8000,
        "command": [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
    },
    {
        "name": "Monday.com Task Board",
        "port": 8001,
        "command": [sys.executable, "-m", "uvicorn", "task_board:app", "--host", "127.0.0.1", "--port", "8001"],
    },
    {
        "name": "Voice Operations AI Agent",
        "port": 8002,
        "command": [sys.executable, "-m", "uvicorn", "agent_server:app", "--host", "127.0.0.1", "--port", "8002"],
    },
]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_internal_services():
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    env = os.environ.copy()
    if "RENDER_EXTERNAL_URL" in os.environ and "PUBLIC_URL" not in env:
        env["PUBLIC_URL"] = os.environ["RENDER_EXTERNAL_URL"]

    for svc in INTERNAL_SERVICES:
        if is_port_in_use(svc["port"]):
            print(f"Service {svc['name']} is already active on 127.0.0.1:{svc['port']}.")
            continue
        print(f"Starting internal service {svc['name']} on 127.0.0.1:{svc['port']}...")
        proc = subprocess.Popen(
            svc["command"],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            env=env,
        )
        processes.append((proc, svc))
    time.sleep(2.5)


def stop_internal_services():
    print("Shutting down internal services...")
    for proc, svc in processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_internal_services()
    try:
        auth.init_users_table()
    except Exception as e:
        print(f"[Auth Init Warning] Could not initialize users table: {e}")
    yield
    stop_internal_services()


gateway_app = FastAPI(
    title="Presales Ecosystem Cloud Gateway",
    description="Unified single-port gateway with Authentication, CRM, Tasks, and Voice AI Agent",
    lifespan=lifespan,
)

gateway_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

http_client = httpx.AsyncClient(timeout=120.0)


# -----------------------------------------------------------------------------
# HTML Templates: Login & Self Sign Up
# -----------------------------------------------------------------------------
LOGIN_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تسجيل الدخول | منظومة Presales المتكاملة</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0a0e17;
            --card-bg: #111827;
            --card-border: #1f293d;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
        }
        body {
            background-color: var(--bg-body);
            font-family: 'Tajawal', 'Plus Jakarta Sans', system-ui, sans-serif;
            color: #f3f4f6;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .auth-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6);
            max-width: 450px;
            width: 100%;
            padding: 36px 30px;
        }
        .brand-icon {
            width: 60px;
            height: 60px;
            border-radius: 15px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            color: white;
            margin: 0 auto 16px;
            box-shadow: 0 4px 18px rgba(37,99,235,0.4);
        }
        .form-control:focus {
            background-color: #0f172a;
            border-color: #3b82f6;
            box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25);
            color: #fff;
        }
        .btn-submit {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border: none;
            padding: 11px 18px;
            font-weight: 700;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(37,99,235,0.35);
            transition: all 0.2s ease;
        }
        .btn-submit:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
        }
        .demo-box {
            background: rgba(16, 185, 129, 0.08);
            border: 1px dashed rgba(16, 185, 129, 0.35);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 0.82rem;
        }
        .password-toggle {
            cursor: pointer;
            border-start-end-radius: 0;
            border-end-end-radius: 0;
        }
    </style>
</head>
<body>
    <div class="auth-card">
        <div class="text-center mb-4">
            <div class="brand-icon">
                <i class="bi bi-shield-lock-fill"></i>
            </div>
            <h4 class="fw-bold mb-1 text-white">منظومة Presales المتكاملة</h4>
            <div class="small text-secondary">تسجيل الدخول للنظام (CRM • Tasks • Voice Agent)</div>
        </div>

        <div id="errorAlert" class="alert alert-danger py-2 small d-none mb-3" role="alert"></div>
        <div id="successAlert" class="alert alert-success py-2 small d-none mb-3" role="alert"></div>

        <form id="loginForm" onsubmit="handleLogin(event)">
            <input type="hidden" id="nextUrl" value="">
            <div class="mb-3 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">اسم المستخدم أو البريد الإلكتروني (Username / Email):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-person-fill"></i></span>
                    <input type="text" id="username" class="form-control bg-dark text-light border-secondary" placeholder="msafia or user@company.com" required autofocus>
                </div>
            </div>

            <div class="mb-3 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">كلمة المرور (Password):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-key-fill"></i></span>
                    <input type="password" id="password" class="form-control bg-dark text-light border-secondary" placeholder="••••••••" required>
                    <button class="btn btn-outline-secondary" type="button" onclick="togglePasswordVisibility()"><i class="bi bi-eye" id="toggleIcon"></i></button>
                </div>
            </div>

            <div class="d-flex justify-content-between align-items-center mb-4">
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="rememberMe" checked>
                    <label class="form-check-label small text-secondary" for="rememberMe">تذكرني على هذا الجهاز</label>
                </div>
            </div>

            <button type="submit" class="btn btn-primary w-100 btn-submit mb-3" id="submitBtn">
                <i class="bi bi-box-arrow-in-left me-1"></i> تسجيل الدخول (Sign In)
            </button>
        </form>

        <!-- Admin Credentials Info Box -->
        <div class="demo-box mb-3 text-light">
            <div class="fw-bold text-success mb-1 d-flex align-items-center gap-1">
                <i class="bi bi-shield-check"></i> بيانات المسؤول الافتراضية (Admin Account):
            </div>
            <div class="d-flex justify-content-between align-items-center small font-monospace mt-1">
                <span>Username: <strong class="text-white">msafia</strong></span>
                <span>Password: <strong class="text-white">Asdasdf1@</strong></span>
            </div>
        </div>

        <div class="text-center pt-2 border-top border-secondary border-opacity-25">
            <span class="small text-muted">ليس لديك حساب؟</span>
            <a href="/signup" class="small text-info text-decoration-none fw-bold ms-1">إنشاء حساب جديد (Self Sign Up) <i class="bi bi-arrow-left"></i></a>
        </div>
    </div>

    <script>
        // Extract 'next' query param if present
        const urlParams = new URLSearchParams(window.location.search);
        const nextTarget = urlParams.get('next') || '/dashboard';
        document.getElementById('nextUrl').value = nextTarget;

        function togglePasswordVisibility() {
            const pwd = document.getElementById('password');
            const icon = document.getElementById('toggleIcon');
            if (pwd.type === 'password') {
                pwd.type = 'text';
                icon.className = 'bi bi-eye-slash';
            } else {
                pwd.type = 'password';
                icon.className = 'bi bi-eye';
            }
        }

        async function handleLogin(e) {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const errAlert = document.getElementById('errorAlert');
            const succAlert = document.getElementById('successAlert');
            errAlert.classList.add('d-none');
            succAlert.classList.add('d-none');

            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value;

            if (!username || !password) {
                errAlert.textContent = 'يرجى إدخال اسم المستخدم وكلمة المرور.';
                errAlert.classList.remove('d-none');
                return;
            }

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>جارٍ التحقق...';

            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });

                const data = await res.json();
                if (res.ok) {
                    succAlert.textContent = 'تم تسجيل الدخول بنجاح! جارٍ تحويلك...';
                    succAlert.classList.remove('d-none');
                    setTimeout(() => {
                        window.location.href = document.getElementById('nextUrl').value || '/dashboard';
                    }, 500);
                } else {
                    errAlert.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-1"></i> ` + (data.detail || 'بيانات الدخول غير صحيحة.');
                    errAlert.classList.remove('d-none');
                }
            } catch (err) {
                errAlert.textContent = 'خطأ في الاتصال بالخادم: ' + err.message;
                errAlert.classList.remove('d-none');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-box-arrow-in-left me-1"></i> تسجيل الدخول (Sign In)';
            }
        }
    </script>
</body>
</html>
"""

SIGNUP_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إنشاء حساب جديد | Presales Ecosystem</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0a0e17;
            --card-bg: #111827;
            --card-border: #1f293d;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
        }
        body {
            background-color: var(--bg-body);
            font-family: 'Tajawal', 'Plus Jakarta Sans', system-ui, sans-serif;
            color: #f3f4f6;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .auth-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6);
            max-width: 480px;
            width: 100%;
            padding: 36px 30px;
        }
        .brand-icon {
            width: 60px;
            height: 60px;
            border-radius: 15px;
            background: linear-gradient(135deg, #10b981, #059669);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            color: white;
            margin: 0 auto 16px;
            box-shadow: 0 4px 18px rgba(16,185,129,0.4);
        }
        .form-control:focus {
            background-color: #0f172a;
            border-color: #10b981;
            box-shadow: 0 0 0 0.25rem rgba(16, 185, 129, 0.25);
            color: #fff;
        }
        .btn-submit {
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            padding: 11px 18px;
            font-weight: 700;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(16,185,129,0.35);
            transition: all 0.2s ease;
        }
        .btn-submit:hover {
            background: linear-gradient(135deg, #059669, #047857);
            transform: translateY(-1px);
        }
    </style>
</head>
<body>
    <div class="auth-card">
        <div class="text-center mb-4">
            <div class="brand-icon">
                <i class="bi bi-person-plus-fill"></i>
            </div>
            <h4 class="fw-bold mb-1 text-white">إنشاء حساب جديد (Self Sign Up)</h4>
            <div class="small text-secondary">انضم إلى فريق Presales وأدر الصفقات والمهام والتحديثات الصوتية</div>
        </div>

        <div id="errorAlert" class="alert alert-danger py-2 small d-none mb-3" role="alert"></div>
        <div id="successAlert" class="alert alert-success py-2 small d-none mb-3" role="alert"></div>

        <form id="signupForm" onsubmit="handleSignup(event)">
            <div class="mb-3">
                <label class="form-label small text-light fw-semibold">الاسم الكامل (Full Name):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-card-text"></i></span>
                    <input type="text" id="fullName" class="form-control bg-dark text-light border-secondary" placeholder="م. عبدالله القحطاني" required autofocus>
                </div>
            </div>

            <div class="mb-3 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">اسم المستخدم (Username):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-person"></i></span>
                    <input type="text" id="username" class="form-control bg-dark text-light border-secondary" placeholder="eng_abdullah" required>
                </div>
            </div>

            <div class="mb-3 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">البريد الإلكتروني (Email Address):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-envelope"></i></span>
                    <input type="email" id="email" class="form-control bg-dark text-light border-secondary" placeholder="abdullah@company.com" required>
                </div>
            </div>

            <div class="mb-3 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">كلمة المرور (Password):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-lock"></i></span>
                    <input type="password" id="password" class="form-control bg-dark text-light border-secondary" placeholder="•••••••• (6+ أحرف)" required minlength="6">
                </div>
            </div>

            <div class="mb-4 text-start" dir="ltr">
                <label class="form-label small text-light fw-semibold">تأكيد كلمة المرور (Confirm Password):</label>
                <div class="input-group">
                    <span class="input-group-text bg-dark border-secondary text-secondary"><i class="bi bi-check2-circle"></i></span>
                    <input type="password" id="confirmPassword" class="form-control bg-dark text-light border-secondary" placeholder="••••••••" required minlength="6">
                </div>
            </div>

            <button type="submit" class="btn btn-success w-100 btn-submit mb-3" id="submitBtn">
                <i class="bi bi-person-plus-fill me-1"></i> إرسال طلب إنشاء الحساب (Register)
            </button>
        </form>

        <div class="text-center pt-2 border-top border-secondary border-opacity-25" id="footerLink">
            <span class="small text-muted">لديك حساب مسجل بالفعل؟</span>
            <a href="/login" class="small text-primary text-decoration-none fw-bold ms-1">تسجيل الدخول (Sign In) <i class="bi bi-arrow-left"></i></a>
        </div>
    </div>

    <script>
        async function handleSignup(e) {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const errAlert = document.getElementById('errorAlert');
            const succAlert = document.getElementById('successAlert');
            errAlert.classList.add('d-none');
            succAlert.classList.add('d-none');

            const full_name = document.getElementById('fullName').value.trim();
            const username = document.getElementById('username').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            const confirm_password = document.getElementById('confirmPassword').value;

            if (password !== confirm_password) {
                errAlert.textContent = 'كلمتا المرور غير متطابقتين.';
                errAlert.classList.remove('d-none');
                return;
            }

            if (password.length < 6) {
                errAlert.textContent = 'كلمة المرور يجب أن لا تقل عن 6 أحرف أو أرقام.';
                errAlert.classList.remove('d-none');
                return;
            }

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>جارٍ إرسال الطلب...';

            try {
                const res = await fetch('/api/auth/signup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ full_name, username, email, password })
                });

                const data = await res.json();
                if (res.ok) {
                    succAlert.innerHTML = `
                        <div class="fw-bold mb-1"><i class="bi bi-clock-history me-1"></i> تم استلام طلب التسجيل بنجاح!</div>
                        <div class="small mb-2">حسابك الآن قيد المراجعة وبانتظار موافقة مسؤول النظام (Admin) لاعتماده وتفعيله لتتمكن من تسجيل الدخول.</div>
                        <div class="mt-3 text-center">
                            <a href="/login" class="btn btn-outline-success btn-sm px-3 fw-bold">
                                <i class="bi bi-box-arrow-in-left me-1"></i> الانتقال لصفحة تسجيل الدخول
                            </a>
                        </div>
                    `;
                    succAlert.classList.remove('d-none');
                    document.getElementById('signupForm').style.display = 'none';
                    const foot = document.getElementById('footerLink');
                    if (foot) foot.style.display = 'none';
                } else {
                    errAlert.textContent = data.detail || 'فشل إنشاء الحساب.';
                    errAlert.classList.remove('d-none');
                }
            } catch (err) {
                errAlert.textContent = 'خطأ في الاتصال بالخادم: ' + err.message;
                errAlert.classList.remove('d-none');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-person-plus-fill me-1"></i> إرسال طلب إنشاء الحساب (Register)';
            }
        }
    </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Pydantic Schemas for Auth API
# -----------------------------------------------------------------------------
class LoginPayload(BaseModel):
    username: str
    password: str


class SignupPayload(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


# -----------------------------------------------------------------------------
# Auth Helper & Injection Utilities
# -----------------------------------------------------------------------------
def is_public_path(path: str) -> bool:
    """Returns True if the path can be accessed without authentication."""
    p = path.strip("/")
    if p in ("", "login", "signup", "docs", "openapi.json"):
        return True
    if p.startswith("api/auth/"):
        return True
    if p in ("favicon.ico", "api/health"):
        return True
    return False


def inject_user_badge(html: str, user: Dict[str, Any]) -> str:
    """Injects authenticated user badge and Logout button into the navigation header, plus admin management modal for admins."""
    username = user.get("username", "user")
    role = user.get("role", "presales")
    role_badge = "Admin" if role == "admin" else "Presales"

    admin_btn = ""
    if role == "admin":
        admin_btn = """
        <button type="button" class="btn btn-sm btn-warning py-0 px-2 d-inline-flex align-items-center gap-1 fw-bold shadow-sm" style="font-size:0.75rem;" onclick="openAdminUsersModal()" title="إدارة المستخدمين وطلبات التسجيل">
            <i class="bi bi-people-fill"></i> المستخدمين
            <span id="adminPendingBadge" class="badge bg-danger rounded-pill px-1 ms-1" style="font-size:0.65rem; display:none;">0</span>
        </button>
        """

    badge_html = f"""
    <!-- Authenticated User Profile & Logout -->
    <div class="d-flex align-items-center gap-2 ms-2 ps-2 border-start border-secondary user-auth-badge">
        {admin_btn}
        <span class="badge bg-primary-subtle text-primary border border-primary-subtle px-2 py-1 d-inline-flex align-items-center gap-1" style="font-size:0.75rem;" title="Logged-in Presales Account">
            <i class="bi bi-person-fill"></i> {username} ({role_badge})
        </span>
        <a href="/api/auth/logout" class="btn btn-sm btn-outline-danger py-0 px-2 d-flex align-items-center gap-1" style="font-size:0.75rem;" title="Sign Out">
            <i class="bi bi-box-arrow-right"></i> Logout
        </a>
    </div>
    """

    # Inject right before the version info container across all 3 apps
    pattern = r'(<div class="d-flex align-items-center ms[^>]*border-start[^>]*>)'
    if re.search(pattern, html):
        html = re.sub(pattern, badge_html + r"\1", html, count=1)

    # For Admin, also inject the Admin Users Management Modal before </body>
    if role == "admin" and "adminUsersModal" not in html and "</body>" in html:
        admin_modal_html = """
        <!-- Admin User Management & Registration Approval Modal -->
        <div class="modal fade" id="adminUsersModal" tabindex="-1" aria-labelledby="adminUsersModalLabel" aria-hidden="true" dir="rtl">
            <div class="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <div class="modal-content text-light" style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px;">
                    <div class="modal-header border-bottom border-secondary border-opacity-50 py-3">
                        <div class="d-flex align-items-center gap-2">
                            <div class="bg-warning bg-opacity-10 p-2 rounded-circle text-warning">
                                <i class="bi bi-shield-lock-fill fs-5"></i>
                            </div>
                            <div>
                                <h5 class="modal-title fw-bold text-white mb-0" id="adminUsersModalLabel">إدارة المستخدمين والموافقات</h5>
                                <small class="text-muted">مراجعة طلبات التسجيل الجديدة وتفعيل الحسابات</small>
                            </div>
                        </div>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body p-4" style="background-color: #0f172a;">
                        <div id="adminActionAlert" class="alert d-none py-2 px-3 mb-3 small fw-bold"></div>

                        <!-- Section: Pending Registration Approvals -->
                        <div class="card mb-4 border-warning border-opacity-50" style="background-color: #1e293b;">
                            <div class="card-header d-flex justify-content-between align-items-center py-2 px-3 border-warning border-opacity-25" style="background-color: rgba(234, 179, 8, 0.1);">
                                <span class="fw-bold text-warning d-flex align-items-center gap-2">
                                    <i class="bi bi-hourglass-split"></i> طلبات التسجيل الجديدة (بانتظار الموافقة)
                                </span>
                                <span id="modalPendingCountBadge" class="badge bg-warning text-dark fw-bold">0</span>
                            </div>
                            <div class="card-body p-0">
                                <div class="table-responsive">
                                    <table class="table table-dark table-hover mb-0 align-middle" style="background-color: transparent; font-size: 0.85rem;">
                                        <thead class="table-dark text-secondary">
                                            <tr>
                                                <th class="ps-3">اسم المستخدم</th>
                                                <th>الاسم الكامل</th>
                                                <th>البريد الإلكتروني</th>
                                                <th>تاريخ الطلب</th>
                                                <th class="text-center pe-3">الإجراء المطلوب</th>
                                            </tr>
                                        </thead>
                                        <tbody id="pendingUsersTableBody">
                                            <tr><td colspan="5" class="text-center text-muted py-3">جارٍ التحميل...</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        <!-- Section: Active Users -->
                        <div class="card border-secondary border-opacity-50" style="background-color: #1e293b;">
                            <div class="card-header d-flex justify-content-between align-items-center py-2 px-3 border-secondary border-opacity-25" style="background-color: rgba(148, 163, 184, 0.05);">
                                <span class="fw-bold text-info d-flex align-items-center gap-2">
                                    <i class="bi bi-people-fill"></i> المستخدمون المفعلون (Active Users)
                                </span>
                                <span id="modalActiveCountBadge" class="badge bg-secondary">0</span>
                            </div>
                            <div class="card-body p-0">
                                <div class="table-responsive">
                                    <table class="table table-dark table-hover mb-0 align-middle" style="background-color: transparent; font-size: 0.85rem;">
                                        <thead class="table-dark text-secondary">
                                            <tr>
                                                <th class="ps-3">اسم المستخدم</th>
                                                <th>الاسم الكامل</th>
                                                <th>البريد الإلكتروني</th>
                                                <th>الصلاحية</th>
                                                <th>آخر تسجيل دخول</th>
                                                <th class="text-center pe-3">الإجراء</th>
                                            </tr>
                                        </thead>
                                        <tbody id="activeUsersTableBody">
                                            <tr><td colspan="6" class="text-center text-muted py-3">جارٍ التحميل...</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer border-top border-secondary border-opacity-50 py-2">
                        <button type="button" class="btn btn-secondary btn-sm px-3" data-bs-dismiss="modal">إغلاق</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
        let adminUsersModalInstance = null;

        async function checkAdminPendingUsers() {
            try {
                const res = await fetch('/api/admin/users');
                if (!res.ok) return;
                const data = await res.json();
                const badge = document.getElementById('adminPendingBadge');
                if (badge) {
                    if (data.pending_count > 0) {
                        badge.textContent = data.pending_count;
                        badge.style.display = 'inline-block';
                    } else {
                        badge.style.display = 'none';
                    }
                }
            } catch(e) {
                console.error('Error fetching admin pending users count:', e);
            }
        }

        async function loadAdminUsersData() {
            const pendingTbody = document.getElementById('pendingUsersTableBody');
            const activeTbody = document.getElementById('activeUsersTableBody');
            const pendingBadge = document.getElementById('modalPendingCountBadge');
            const activeBadge = document.getElementById('modalActiveCountBadge');

            try {
                const res = await fetch('/api/admin/users');
                if (!res.ok) {
                    throw new Error('Failed to load users: ' + res.statusText);
                }
                const data = await res.json();

                if (pendingBadge) pendingBadge.textContent = data.pending_count;
                if (activeBadge) activeBadge.textContent = data.active.length;
                const navBadge = document.getElementById('adminPendingBadge');
                if (navBadge) {
                    navBadge.textContent = data.pending_count;
                    navBadge.style.display = data.pending_count > 0 ? 'inline-block' : 'none';
                }

                if (data.pending.length === 0) {
                    pendingTbody.innerHTML = '<tr><td colspan="5" class="text-center text-success py-3"><i class="bi bi-check-circle me-1"></i> لا توجد طلبات تسجيل جديدة بانتظار الموافقة.</td></tr>';
                } else {
                    pendingTbody.innerHTML = data.pending.map(u => `
                        <tr>
                            <td class="ps-3 fw-bold text-white">${u.username}</td>
                            <td>${u.full_name || '-'}</td>
                            <td class="text-muted">${u.email}</td>
                            <td class="small text-muted">${u.created_at || '-'}</td>
                            <td class="text-center pe-3">
                                <button class="btn btn-sm btn-success py-1 px-2 fw-bold" onclick="approveAdminUser(${u.id}, '${u.username}')" title="موافقة وتفعيل">
                                    <i class="bi bi-check-lg me-1"></i> موافقة
                                </button>
                                <button class="btn btn-sm btn-outline-danger py-1 px-2 ms-1" onclick="deleteAdminUser(${u.id}, '${u.username}')" title="رفض وحذف">
                                    <i class="bi bi-trash"></i> رفض
                                </button>
                            </td>
                        </tr>
                    `).join('');
                }

                if (data.active.length === 0) {
                    activeTbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-3">لا يوجد مستخدمون مفعلون.</td></tr>';
                } else {
                    activeTbody.innerHTML = data.active.map(u => `
                        <tr>
                            <td class="ps-3 fw-bold text-white">${u.username}</td>
                            <td>${u.full_name || '-'}</td>
                            <td class="text-muted">${u.email}</td>
                            <td><span class="badge ${u.role === 'admin' ? 'bg-primary' : 'bg-secondary'}">${u.role}</span></td>
                            <td class="small text-muted">${u.last_login || 'لم يسجل دخول بعد'}</td>
                            <td class="text-center pe-3">
                                ${u.username === 'msafia' ? '<span class="badge bg-primary-subtle text-primary border border-primary-subtle">المدير العام (Protected)</span>' : `
                                    <button class="btn btn-sm btn-outline-warning py-0 px-2" onclick="deactivateAdminUser(${u.id}, '${u.username}')" title="تعليق وتجميد الحساب">
                                        <i class="bi bi-pause-circle me-1"></i> تعطيل
                                    </button>
                                    <button class="btn btn-sm btn-outline-danger py-0 px-1 ms-1" onclick="deleteAdminUser(${u.id}, '${u.username}')" title="حذف الحساب">
                                        <i class="bi bi-x-lg"></i>
                                    </button>
                                `}
                            </td>
                        </tr>
                    `).join('');
                }

            } catch (err) {
                showAdminAlert('danger', 'خطأ أثناء جلب بيانات المستخدمين: ' + err.message);
            }
        }

        function showAdminAlert(type, msg) {
            const alertBox = document.getElementById('adminActionAlert');
            if (!alertBox) return;
            alertBox.className = `alert alert-${type} py-2 px-3 mb-3 small fw-bold`;
            alertBox.innerHTML = msg;
            alertBox.classList.remove('d-none');
            setTimeout(() => {
                alertBox.classList.add('d-none');
            }, 4000);
        }

        function openAdminUsersModal() {
            const modalEl = document.getElementById('adminUsersModal');
            if (!modalEl) return;
            if (!adminUsersModalInstance && typeof bootstrap !== 'undefined') {
                adminUsersModalInstance = new bootstrap.Modal(modalEl);
            }
            loadAdminUsersData();
            if (adminUsersModalInstance) {
                adminUsersModalInstance.show();
            }
        }

        async function approveAdminUser(id, username) {
            try {
                const res = await fetch('/api/admin/users/' + id + '/approve', { method: 'POST' });
                const resData = await res.json();
                if (res.ok) {
                    showAdminAlert('success', `<i class="bi bi-check-circle-fill me-1"></i> تم تفعيل واعتماد حساب <strong>${username}</strong> بنجاح! يمكنه الآن تسجيل الدخول.`);
                    loadAdminUsersData();
                } else {
                    showAdminAlert('danger', resData.detail || 'فشل تفعيل الحساب.');
                }
            } catch (err) {
                showAdminAlert('danger', 'خطأ بالاتصال: ' + err.message);
            }
        }

        async function deactivateAdminUser(id, username) {
            if (!confirm(`هل أنت متأكد من رغبتك في تعطيل حساب ${username}؟ لن يتمكن من تسجيل الدخول حتى تعيد تفعيله.`)) return;
            try {
                const res = await fetch('/api/admin/users/' + id + '/deactivate', { method: 'POST' });
                const resData = await res.json();
                if (res.ok) {
                    showAdminAlert('warning', `<i class="bi bi-pause-fill me-1"></i> تم تعطيل حساب <strong>${username}</strong>.`);
                    loadAdminUsersData();
                } else {
                    showAdminAlert('danger', resData.detail || 'فشل تعطيل الحساب.');
                }
            } catch (err) {
                showAdminAlert('danger', 'خطأ بالاتصال: ' + err.message);
            }
        }

        async function deleteAdminUser(id, username) {
            if (!confirm(`تحذير: هل أنت متأكد من رفض وحذف حساب ${username} نهائياً؟`)) return;
            try {
                const res = await fetch('/api/admin/users/' + id, { method: 'DELETE' });
                const resData = await res.json();
                if (res.ok) {
                    showAdminAlert('danger', `<i class="bi bi-trash-fill me-1"></i> تم حذف حساب <strong>${username}</strong> بنجاح.`);
                    loadAdminUsersData();
                } else {
                    showAdminAlert('danger', resData.detail || 'فشل حذف الحساب.');
                }
            } catch (err) {
                showAdminAlert('danger', 'خطأ بالاتصال: ' + err.message);
            }
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', checkAdminPendingUsers);
        } else {
            checkAdminPendingUsers();
        }
        </script>
        """
        html = html.replace("</body>", admin_modal_html + "\n</body>")

    return html


# -----------------------------------------------------------------------------
# Auth Routes (Login, Signup, Logout, Current User)
# -----------------------------------------------------------------------------
@gateway_app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serves the Login page. If user is already authenticated, redirects to dashboard."""
    user = auth.get_current_user_from_request(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return HTMLResponse(content=LOGIN_HTML)


@gateway_app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Serves the Self Sign Up page."""
    user = auth.get_current_user_from_request(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return HTMLResponse(content=SIGNUP_HTML)


@gateway_app.post("/api/auth/login")
async def api_login(payload: LoginPayload):
    """Authenticates user and returns session cookie if active."""
    auth_res = auth.authenticate_user(payload.username, payload.password)
    if not auth_res.get("success"):
        status_code = status.HTTP_403_FORBIDDEN if auth_res.get("status") == "pending_approval" else status.HTTP_401_UNAUTHORIZED
        raise HTTPException(status_code=status_code, detail=auth_res.get("message", "بيانات الدخول غير صحيحة."))

    user = auth_res["user"]
    token = auth.create_session_token(
        user_id=user["id"],
        username=user["username"],
        role=user.get("role", "presales"),
        full_name=user.get("full_name", ""),
    )

    response = JSONResponse(content={
        "status": "success",
        "message": "Login successful",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user.get("full_name"),
            "role": user.get("role"),
        },
        "token": token,
    })

    # Set secure session cookie (30 days)
    response.set_cookie(
        key=auth.SESSION_COOKIE_NAME,
        value=token,
        max_age=30 * 86400,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True on HTTPS/production
    )
    return response


@gateway_app.post("/api/auth/signup")
async def api_signup(payload: SignupPayload):
    """Registers a new user in pending state (is_active=0) awaiting admin approval."""
    try:
        user = auth.create_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            role="presales",
            is_active=0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {e}")

    # Gated account: pending admin approval (no session cookie issued)
    return JSONResponse(content={
        "status": "pending_approval",
        "message": "تم استلام طلب التسجيل بنجاح! حسابك قيد المراجعة وبانتظار موافقة مسؤول النظام (Admin) لتفعيله.",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user.get("full_name"),
            "role": user.get("role"),
            "is_active": user.get("is_active", 0),
        },
    })


# -----------------------------------------------------------------------------
# Admin User Management APIs
# -----------------------------------------------------------------------------
@gateway_app.get("/api/admin/users")
async def admin_list_users(request: Request):
    """Lists pending and active users for admin."""
    user = auth.get_current_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="صلاحيات مسؤول النظام مطلوبة (Admin privileges required).")
    return auth.list_users()


@gateway_app.post("/api/admin/users/{user_id}/approve")
async def admin_approve_user(user_id: int, request: Request):
    """Approves and activates a pending user account."""
    user = auth.get_current_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="صلاحيات مسؤول النظام مطلوبة (Admin privileges required).")
    success = auth.approve_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود.")
    return {"status": "success", "message": f"تم تفعيل واعتماد المستخدم بنجاح."}


@gateway_app.post("/api/admin/users/{user_id}/deactivate")
async def admin_deactivate_user(user_id: int, request: Request):
    """Deactivates an active user account."""
    user = auth.get_current_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="صلاحيات مسؤول النظام مطلوبة (Admin privileges required).")
    success = auth.deactivate_user(user_id)
    if not success:
        raise HTTPException(status_code=400, detail="لا يمكن تعطيل هذا الحساب أو المستخدم غير موجود.")
    return {"status": "success", "message": f"تم تعطيل الحساب بنجاح."}


@gateway_app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, request: Request):
    """Rejects registration or deletes a user account."""
    user = auth.get_current_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="صلاحيات مسؤول النظام مطلوبة (Admin privileges required).")
    success = auth.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=400, detail="لا يمكن حذف هذا الحساب أو المستخدم غير موجود.")
    return {"status": "success", "message": f"تم حذف الطلب / الحساب بنجاح."}


@gateway_app.get("/api/auth/logout")
@gateway_app.post("/api/auth/logout")
async def logout():
    """Clears session cookie and redirects to /login."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=auth.SESSION_COOKIE_NAME)
    return response


@gateway_app.get("/api/auth/me")
async def get_me(request: Request):
    """Returns profile of currently authenticated user."""
    user = auth.get_current_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


# -----------------------------------------------------------------------------
# Protected Application Endpoints
# -----------------------------------------------------------------------------
@gateway_app.get("/", response_class=HTMLResponse)
async def root_redirect(request: Request):
    user = auth.get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@gateway_app.get("/dashboard", response_class=HTMLResponse)
@gateway_app.get("/dashboard/", response_class=HTMLResponse)
async def dashboard_ui(request: Request):
    """Proxies CRM Dashboard with authentication check and user badge injection."""
    user = auth.get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login?next=/dashboard", status_code=303)

    try:
        r = await http_client.get("http://127.0.0.1:8000/dashboard")
        content = r.text
        content = re.sub(r"http://127\.0\.0\.1:8001", "/tasks", content)
        content = re.sub(r"http://127\.0\.0\.1:8002", "/voice", content)
        content = inject_user_badge(content, user)
        return HTMLResponse(content=content, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(f"<h3>CRM Dashboard is starting up... ({e})</h3>", status_code=503)


@gateway_app.get("/tasks", response_class=HTMLResponse)
@gateway_app.get("/tasks/", response_class=HTMLResponse)
async def tasks_ui(request: Request):
    """Proxies Task Board HTML with authentication check and user badge injection."""
    user = auth.get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login?next=/tasks", status_code=303)

    try:
        r = await http_client.get("http://127.0.0.1:8001/")
        content = r.text
        content = re.sub(r"http://127\.0\.0\.1:8000/dashboard", "/dashboard", content)
        content = re.sub(r"http://127\.0\.0\.1:8002", "/voice", content)
        content = inject_user_badge(content, user)
        return HTMLResponse(content=content, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(f"<h3>Task Board is starting up... ({e})</h3>", status_code=503)


@gateway_app.get("/voice", response_class=HTMLResponse)
@gateway_app.get("/voice/", response_class=HTMLResponse)
@gateway_app.get("/agent", response_class=HTMLResponse)
@gateway_app.get("/agent/", response_class=HTMLResponse)
async def voice_agent_ui(request: Request):
    """Proxies Voice Agent HTML with authentication check and user badge injection."""
    user = auth.get_current_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login?next=/voice", status_code=303)

    try:
        r = await http_client.get("http://127.0.0.1:8002/")
        content = r.text
        content = re.sub(r"http://127\.0\.0\.1:8000/dashboard", "/dashboard", content)
        content = re.sub(r"http://127\.0\.0\.1:8001", "/tasks", content)
        content = inject_user_badge(content, user)
        return HTMLResponse(content=content, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(f"<h3>Voice Agent is starting up... ({e})</h3>", status_code=503)


# -----------------------------------------------------------------------------
# Dynamic Reverse Proxy for All Sub-routes & APIs
# -----------------------------------------------------------------------------
@gateway_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy(request: Request, path: str):
    """Dynamic reverse proxy dispatching to the 3 internal services with authentication gating."""
    # Check authentication for non-public paths
    if not is_public_path(path):
        client_host = request.client.host if request.client else ""
        is_local_internal = client_host in ("127.0.0.1", "localhost", "::1")
        user = auth.get_current_user_from_request(request)

        if not user and not is_local_internal:
            # Check if this is an API call or a page navigation
            if path.startswith("api/"):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required. Please log in at /login"},
                )
            else:
                return RedirectResponse(url=f"/login?next=/{path}", status_code=303)

    # 1. Voice Agent Endpoints (Port 8002)
    agent_endpoints = [
        "api/process-audio",
        "api/audit-state",
        "api/followup-opportunities",
        "api/pre-meeting-questions",
        "api/health",
        "api/set-api-key",
        "api/approve-sync",
    ]
    if any(path.startswith(ep) for ep in agent_endpoints):
        target_port = 8002
    # 2. Tasks API Endpoints (Task Board - Port 8001)
    elif (
        path.startswith("api/tasks")
        or path.startswith("api/deals-lookup")
        or path.startswith("api/customers-lookup")
        or (path.startswith("api/deals/") and path.endswith("/tasks"))
        or (request.method == "DELETE" and path.startswith("api/deals/"))
    ):
        target_port = 8001
    # 3. CRM Dashboard & All Other Endpoints (Port 8000)
    else:
        target_port = 8000

    target_url = f"http://127.0.0.1:{target_port}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        resp = await http_client.send(req, stream=True)

        # Intercept HTML responses to rewrite relative URLs and inject user badge
        if "text/html" in resp.headers.get("content-type", ""):
            resp_content = await resp.aread()
            text = resp_content.decode("utf-8", errors="replace")
            text = re.sub(r"http://127\.0\.0\.1:8001", "/tasks", text)
            text = re.sub(r"http://127\.0\.0\.1:8002", "/voice", text)
            user = auth.get_current_user_from_request(request)
            if user:
                text = inject_user_badge(text, user)
            return HTMLResponse(content=text, status_code=resp.status_code)

        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as e:
        return Response(content=f"Gateway service communication error: {e}", status_code=502)


if __name__ == "__main__":
    print("=" * 70)
    print(f"Starting Authenticated Presales Cloud Gateway on 0.0.0.0:{PORT}...")
    print("=" * 70)
    uvicorn.run("server:gateway_app", host="0.0.0.0", port=PORT, reload=False)
