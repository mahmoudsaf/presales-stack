import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

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
    yield
    stop_internal_services()


gateway_app = FastAPI(
    title="Presales Ecosystem Cloud Gateway",
    description="Unified single-port gateway routing CRM, Tasks, and Voice AI Agent",
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


@gateway_app.get("/tasks", response_class=HTMLResponse)
@gateway_app.get("/tasks/", response_class=HTMLResponse)
async def tasks_ui():
    """Proxies Task Board HTML with rewritten relative asset paths."""
    try:
        r = await http_client.get("http://127.0.0.1:8001/")
        content = r.text
        # Ensure links to CRM point to /dashboard
        content = re.sub(r"http://127\.0\.0\.1:8000/dashboard", "/dashboard", content)
        content = re.sub(r"http://127\.0\.0\.1:8002", "/voice", content)
        return HTMLResponse(content=content, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(f"<h3>Task Board is starting up... ({e})</h3>", status_code=503)


@gateway_app.get("/voice", response_class=HTMLResponse)
@gateway_app.get("/voice/", response_class=HTMLResponse)
@gateway_app.get("/agent", response_class=HTMLResponse)
@gateway_app.get("/agent/", response_class=HTMLResponse)
async def voice_agent_ui():
    """Proxies Voice Agent HTML with rewritten relative asset paths."""
    try:
        r = await http_client.get("http://127.0.0.1:8002/")
        content = r.text
        content = re.sub(r"http://127\.0\.0\.1:8000/dashboard", "/dashboard", content)
        content = re.sub(r"http://127\.0\.0\.1:8001", "/tasks", content)
        return HTMLResponse(content=content, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(f"<h3>Voice Agent is starting up... ({e})</h3>", status_code=503)


@gateway_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def reverse_proxy(request: Request, path: str):
    """Dynamic reverse proxy dispatching to the 3 internal services seamlessly."""
    # 1. Voice Agent Endpoints (Port 8002)
    agent_endpoints = [
        "api/process-audio",
        "api/audit-state",
        "api/followup-opportunities",
        "api/pre-meeting-questions",
        "api/health",
        "api/set-api-key",
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

        # Intercept HTML responses from CRM to ensure relative links
        if "text/html" in resp.headers.get("content-type", ""):
            resp_content = await resp.aread()
            text = resp_content.decode("utf-8", errors="replace")
            text = re.sub(r"http://127\.0\.0\.1:8001", "/tasks", text)
            text = re.sub(r"http://127\.0\.0\.1:8002", "/voice", text)
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
    print(f"Starting Presales Cloud Gateway on 0.0.0.0:{PORT}...")
    print("=" * 70)
    uvicorn.run("server:gateway_app", host="0.0.0.0", port=PORT, reload=False)
