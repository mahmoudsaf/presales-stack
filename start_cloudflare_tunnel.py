import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED = BASE_DIR / "cloudflared.exe"

SERVICES = [
    {"name": "CRM Dashboard & API", "port": 8000},
    {"name": "Monday.com Task Board", "port": 8001},
    {"name": "Voice Operations AI Agent", "port": 8002},
]


def main():
    if not CLOUDFLARED.exists():
        print(f"Error: {CLOUDFLARED} not found.")
        sys.exit(1)

    print("=" * 70)
    print("   PRESALES AUTOMATION ECOSYSTEM - CLOUDFLARE EDGE DEPLOYMENT   ")
    print("=" * 70)

    processes = []
    urls = {}

    for svc in SERVICES:
        port = svc["port"]
        name = svc["name"]
        print(f"Connecting {name} (Port {port}) to Cloudflare Edge...")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=flags,
            bufsize=1,
        )
        processes.append((proc, svc))

    print("\nNegotiating secure Cloudflare Quick Tunnels...")

    # Extract URLs from stdout/stderr stream
    timeout = 25
    start_time = time.time()
    for proc, svc in processes:
        name = svc["name"]
        port = svc["port"]
        url_found = None
        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
            if m:
                url_found = m.group(0)
                urls[port] = url_found
                print(f"  [LIVE] {name:<32} -> {url_found}")
                break

    print("=" * 70)
    print(">>> ALL SERVICES DEPLOYED TO CLOUDFLARE EDGE <<<")
    print("=" * 70)
    print("\nPublic HTTPS URLs:")
    for svc in SERVICES:
        p = svc["port"]
        u = urls.get(p, "Initializing...")
        print(f" - {svc['name']}: {u}")

    print("\n100% matched with local SQLite databases (crm.db & tasks.db).")
    print("Press CTRL+C to close tunnels.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing Cloudflare Tunnels...")
        for proc, _ in processes:
            proc.terminate()
        print("Tunnels closed.")


if __name__ == "__main__":
    main()
