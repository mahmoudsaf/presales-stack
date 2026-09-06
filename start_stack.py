import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Working directory
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

PLACEHOLDER_SUBSTRINGS = ["your_actual", "placeholder", "aizasyyouractual"]


def is_placeholder(val: str) -> bool:
    if not val:
        return True
    s = val.strip().lower()
    return any(p in s for p in PLACEHOLDER_SUBSTRINGS) or s.startswith("your_")


# Auto-load .env file if present
if ENV_FILE.exists():
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k and v and not is_placeholder(v):
                        curr = os.environ.get(k, "")
                        if not curr or is_placeholder(curr) or k == "GEMINI_API_KEY":
                            os.environ[k] = v
    except Exception:
        pass

SERVICES = [
    {
        "name": "Local CRM API & Pipeline Dashboard",
        "command": [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        "port": 8000,
        "url": "http://127.0.0.1:8000",
        "docs": "http://127.0.0.1:8000/docs",
    },
    {
        "name": "Monday.com-Style Task Board API",
        "command": [sys.executable, "-m", "uvicorn", "task_board:app", "--host", "127.0.0.1", "--port", "8001", "--reload"],
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "docs": "http://127.0.0.1:8001/docs",
    },
    {
        "name": "Bilingual Voice Operations AI Agent",
        "command": [sys.executable, "-m", "uvicorn", "agent_server:app", "--host", "127.0.0.1", "--port", "8002", "--reload"],
        "port": 8002,
        "url": "http://127.0.0.1:8002",
        "docs": "http://127.0.0.1:8002/docs",
    },
]

processes = []


def check_service_ready(url: str, max_retries: int = 25, delay: float = 0.5) -> bool:
    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status in (200, 404):
                    return True
        except Exception:
            time.sleep(delay)
    return False


def stop_all_services(signum=None, frame=None):
    print("\n" + "=" * 65)
    print("Shutting down all Presales Stack services gracefully...")
    print("=" * 65)
    for p, svc in zip(processes, SERVICES):
        if p.poll() is None:
            print(f"Terminating {svc['name']} (PID: {p.pid})...")
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
    print("All services stopped.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, stop_all_services)

    print("=" * 70)
    print("   PRESALES AUTOMATION ECOSYSTEM - CONCURRENT STACK LAUNCHER   ")
    print("=" * 70)

    # Launch subprocesses
    for svc in SERVICES:
        print(f"Starting {svc['name']} on port {svc['port']}...")
        proc = subprocess.Popen(
            svc["command"],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)

    print("\nWaiting for all 3 servers to initialize...")

    all_ready = True
    for svc in SERVICES:
        ready = check_service_ready(svc["url"])
        if ready:
            print(f"  [ONLINE]  {svc['name']:<38} -> {svc['url']}")
        else:
            print(f"  [WARNING] {svc['name']:<38} did not respond in time.")
            all_ready = False

    print("=" * 70)
    if all_ready:
        print(">>> ALL 3 SERVICES RUNNING AND READY <<<")
    else:
        print(">>> Services started (some still initializing). <<<")
    print("\nAccess Dashboards & APIs:")
    for svc in SERVICES:
        print(f" - {svc['name']}: {svc['url']} (Swagger: {svc['docs']})")
    print("\nPress CTRL+C in this console to shut down all 3 servers at once.")
    print("=" * 70 + "\n")

    # Monitor processes loop
    try:
        while True:
            for p, svc in zip(processes, SERVICES):
                retcode = p.poll()
                if retcode is not None:
                    print(f"Warning: {svc['name']} exited unexpectedly with code {retcode}.")
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_all_services()


if __name__ == "__main__":
    main()
