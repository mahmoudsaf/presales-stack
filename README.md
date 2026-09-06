# Enterprise Presales Ecosystem & Bilingual Voice Operations AI Agent

A lightweight, local, full-stack Presales operations platform powered by **Python**, **FastAPI**, **SQLite**, and **Google Gemini 3.6 Flash**.

---

## Architecture & Ports Overview

| Component | Port | File | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Visual Analytics & CRM Dashboard** | `8000` | `app.py` | Executive visual metrics, Chart.js analytics, customer accounts, multi-vendor deal pipeline, and presales assignment. Access at `/` or `/dashboard`. |
| **Monday.com Task Board** | `8001` | `task_board.py` | RFP technical ownership, distributed scopes, priority, and management blockers. |
| **Voice Operations AI Agent** | `8002` | `agent_server.py` | Bilingual (Saudi Arabic/English) audio standup processor, pre-meeting state audit, and automated REST API sync. |
| **Concurrent Stack Launcher** | - | `start_stack.py` | Spawns and manages all 3 services concurrently with unified lifecycle management. |

---

## Quick Start Guide

### 1. Launch All Services Concurrently
From the project folder:

```powershell
cd C:\Users\msafi\Documents\Antigraphity\Team1
python start_stack.py
```

All 3 servers will start automatically:
- **CRM Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (API Docs: `/docs`)
- **Task Board**: [http://127.0.0.1:8001/](http://127.0.0.1:8001/) (API Docs: `/docs`)
- **Voice Agent**: [http://127.0.0.1:8002/](http://127.0.0.1:8002/) (API Docs: `/docs`)

---

## Gemini API Key Persistence (Never Asked Twice)

The application now supports **automatic permanent persistence** for your `GEMINI_API_KEY`:

### Method A: From the Web UI (Easiest)
1. Open [http://127.0.0.1:8002/](http://127.0.0.1:8002/).
2. Click **"Configure Gemini Key"** in the top right.
3. Enter your Gemini API key and click **Save**.
4. The key is **automatically written to `.env`** on your disk. Every time the agent starts or reloads, it reads this file automatically.

### Method B: Manual `.env` File
Create or edit `.env` in this directory:
```bash
GEMINI_API_KEY="AIzaSyYourActualKeyHere"
```

### Method C: Windows Permanent Environment Variable
To make it available across all terminals permanently:
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIzaSyYourActualKeyHere', 'User')
```

---

## How to Backup / Save This Application

All source code and database files are self-contained in this folder:
`C:\Users\msafi\Documents\Antigraphity\Team1`

To create a quick compressed ZIP backup at any time, run:

```powershell
Compress-Archive -Path * -DestinationPath ..\Presales_CRM_Backup.zip -Force
```
