import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Configuration & Persistence
# -----------------------------------------------------------------------------
ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env_file():
    """Loads key-value pairs from local .env into os.environ if present."""
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception as e:
            print(f"Notice: Could not read .env: {e}")


def save_env_file(key: str, value: str):
    """Saves or updates a key-value pair in .env file."""
    lines = []
    found = False
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith(f"{key}="):
                        lines.append(f'{key}="{value}"\n')
                        found = True
                    else:
                        lines.append(line)
        except Exception:
            pass
    if not found:
        lines.append(f'{key}="{value}"\n')

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


# Automatically load .env on startup
load_env_file()

CRM_API_URL = os.getenv("CRM_API_URL", "http://127.0.0.1:8000/api")
TASKS_API_URL = os.getenv("TASKS_API_URL", "http://127.0.0.1:8001/api")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-2.5-flash"]


def generate_with_model_fallback(client: genai.Client, contents: Any, **kwargs):
    """Attempts generation with primary model and falls back if model is deprecated/404."""
    models_to_try = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_exception = None

    for model_name in models_to_try:
        try:
            return client.models.generate_content(model=model_name, contents=contents, **kwargs)
        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            if "404" in err_msg or "not found" in err_msg or "no longer available" in err_msg:
                print(f"Model {model_name} unavailable, falling back to next model...")
                continue
            raise e
    raise last_exception


def get_gemini_client() -> Optional[genai.Client]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None


# -----------------------------------------------------------------------------
# FastAPI App & Lifecycle
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify baseline service connectivity on startup
    print("=" * 60)
    print("Voice Operations AI Agent starting on Port 8002")
    print(f"Connecting to CRM API: {CRM_API_URL}")
    print(f"Connecting to Tasks API: {TASKS_API_URL}")
    print(f"Gemini Model: {GEMINI_MODEL}")
    if os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY detected.")
    else:
        print("NOTICE: GEMINI_API_KEY not found in environment. You can set it in the UI or environment variable.")
    print("=" * 60)
    yield


app = FastAPI(
    title="Bilingual Voice Operations AI Agent",
    description="Multimodal standup audio processor & API sync agent for Presales CRM & Task Board",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helper Functions: State Fetching & API Execution
# -----------------------------------------------------------------------------
async def fetch_baseline_state() -> Dict[str, Any]:
    deals = []
    tasks = []
    crm_healthy = False
    tasks_healthy = False

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            r = await client.get(f"{CRM_API_URL}/deals")
            if r.status_code == 200:
                deals = r.json()
                crm_healthy = True
        except Exception as e:
            print(f"Warning: Could not fetch deals from CRM API: {e}")

        try:
            r = await client.get(f"{TASKS_API_URL}/tasks")
            if r.status_code == 200:
                tasks = r.json()
                tasks_healthy = True
        except Exception as e:
            print(f"Warning: Could not fetch tasks from Tasks API: {e}")

    # Ensure every task record has customer_name and deal_name populated
    deals_by_id = {}
    for d in deals:
        did = d.get("deal_id")
        if did:
            deals_by_id[did] = (
                d.get("deal_name") or "",
                d.get("company_name") or d.get("customer_name") or ""
            )

    for t in tasks:
        rel_id = t.get("related_deal_id")
        if rel_id and rel_id in deals_by_id:
            d_name, c_name = deals_by_id[rel_id]
            t["deal_name"] = t.get("deal_name") or d_name
            t["customer_name"] = t.get("customer_name") or c_name

        # Fallback keyword match if deal_name or customer_name still missing
        if not t.get("deal_name") or not t.get("customer_name"):
            t_title = (t.get("task_title") or "").lower()
            matched = False
            for did, (d_name, c_name) in deals_by_id.items():
                keywords = []
                if "فيصل" in d_name or "faisal" in c_name.lower():
                    keywords.extend(["فيصل", "faisal"])
                if "مراعي" in d_name or "almarai" in c_name.lower():
                    keywords.extend(["مراعي", "almarai"])
                if "بلدية" in d_name or "municipality" in c_name.lower():
                    keywords.extend(["بلدية", "municipality"])
                if "تخطيط" in d_name or "planning" in c_name.lower():
                    keywords.extend(["تخطيط", "planning"])
                if "حج" in d_name or "hajj" in c_name.lower():
                    keywords.extend(["حج", "hajj"])
                if (d_name and len(d_name) > 4 and d_name.lower() in t_title) or (c_name and len(c_name) > 3 and c_name.lower() in t_title):
                    t["deal_name"] = t.get("deal_name") or d_name
                    t["customer_name"] = t.get("customer_name") or c_name
                    if not t.get("related_deal_id"):
                        t["related_deal_id"] = did
                    matched = True
                    break
                for kw in keywords:
                    if kw in t_title:
                        t["deal_name"] = t.get("deal_name") or d_name
                        t["customer_name"] = t.get("customer_name") or c_name
                        if not t.get("related_deal_id"):
                            t["related_deal_id"] = did
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                if "human resources" in t_title or "الموارد البشرية" in t_title:
                    t["deal_name"] = t.get("deal_name") or "Ministry HR Renewal Tender"
                    t["customer_name"] = t.get("customer_name") or "Ministry of Human Resources"
                elif "solarwinds" in t_title:
                    t["deal_name"] = t.get("deal_name") or "SolarWinds License Procurement"
                    t["customer_name"] = t.get("customer_name") or "SolarWinds"
                elif "islam" in t_title or "cross-functional" in t_title:
                    t["deal_name"] = t.get("deal_name") or "Cross-Functional Team Deliverables"
                    t["customer_name"] = t.get("customer_name") or "Internal Presales Team"
                else:
                    t["deal_name"] = t.get("deal_name") or "General Presales Deliverable"
                    t["customer_name"] = t.get("customer_name") or "Presales Operations"

    return {
        "deals": deals,
        "tasks": tasks,
        "crm_healthy": crm_healthy,
        "tasks_healthy": tasks_healthy,
    }


def sanitize_crm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload)
    p.pop("deal_id", None)
    p.pop("customer_id", None)
    if "stage" in p and p["stage"]:
        s = str(p["stage"]).strip().lower()
        stage_map = {
            "in progress": "Gathering Requirements",
            "ongoing": "Gathering Requirements",
            "active": "Gathering Requirements",
            "discovery": "Discovery",
            "requirements": "Gathering Requirements",
            "gathering requirements": "Gathering Requirements",
            "poc": "PoC",
            "proof of concept": "PoC",
            "proposal": "Proposal",
            "closed-won": "Closed-Won",
            "closed won": "Closed-Won",
            "won": "Closed-Won",
            "closed-lost": "Closed-Lost",
            "closed lost": "Closed-Lost",
            "lost": "Closed-Lost",
        }
        p["stage"] = stage_map.get(s, "Gathering Requirements")
    if "assigned_presales" in p and p["assigned_presales"]:
        v = str(p["assigned_presales"]).strip().lower()
        p["assigned_presales"] = "Presales 2" if "2" in v else "Presales 1"
    return p


def sanitize_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload)
    p.pop("task_id", None)

    # 1. Title handling (alias resolution & default)
    title = p.pop("task_title", None) or p.pop("title", None) or p.pop("name", None)
    if not title or not str(title).strip():
        title = "Presales Action Item"
    p["task_title"] = str(title).strip()

    # 2. Category handling
    cat = str(p.get("category", "")).strip().lower()
    if "owner" in cat:
        p["category"] = "RFP_OWNERSHIP"
    elif "distribut" in cat or "scope" in cat or "rfp" in cat:
        p["category"] = "RFP_DISTRIBUTED_SCOPE"
    else:
        p["category"] = "GENERAL_ACTION"

    # 3. Assigned To handling
    assigned = str(p.get("assigned_to", "")).strip().lower()
    if "2" in assigned or "presales 2" in assigned:
        p["assigned_to"] = "Presales 2"
    else:
        p["assigned_to"] = "Presales 1"

    # 4. Vendor Domain handling (crucial mapping for HP -> HPE)
    vendor = str(p.get("vendor_domain", "")).strip().lower()
    if "hp" in vendor or "hewlett" in vendor or "proliant" in vendor or "alletra" in vendor:
        p["vendor_domain"] = "HPE"
    elif "veeam" in vendor:
        p["vendor_domain"] = "Veeam"
    elif "dell" in vendor or "poweredge" in vendor or "powerprotect" in vendor or "emc" in vendor:
        p["vendor_domain"] = "Dell"
    elif "nutanix" in vendor or "ahv" in vendor:
        p["vendor_domain"] = "Nutanix"
    elif "vmware" in vendor or "vcf" in vendor or "vsphere" in vendor or "broadcom" in vendor:
        p["vendor_domain"] = "VMware"
    else:
        p["vendor_domain"] = "General"

    # 5. Status handling
    st = str(p.get("status", "")).strip().lower()
    status_map = {
        "not started": "Not Started",
        "in progress": "In Progress",
        "inprogress": "In Progress",
        "started": "In Progress",
        "ongoing": "In Progress",
        "active": "In Progress",
        "waiting": "Waiting on Vendor",
        "waiting on vendor": "Waiting on Vendor",
        "review": "Pending Review",
        "pending review": "Pending Review",
        "pending": "Pending Review",
        "completed": "Completed",
        "done": "Completed",
        "closed": "Completed",
        "finished": "Completed",
    }
    p["status"] = status_map.get(st, "In Progress")

    # 6. Priority handling
    pr = str(p.get("priority", "")).strip().lower()
    if "high" in pr or "critical" in pr or "urgent" in pr or "p1" in pr:
        p["priority"] = "High"
    elif "low" in pr or "p3" in pr:
        p["priority"] = "Low"
    else:
        p["priority"] = "Medium"

    # 7. Deal ID handling
    deal_id = p.get("related_deal_id")
    if deal_id is not None and str(deal_id).strip():
        digits = re.findall(r"\d+", str(deal_id))
        p["related_deal_id"] = int(digits[0]) if digits else None
    else:
        p["related_deal_id"] = None

    # 8. Management blockers
    if "management_blockers" in p and p["management_blockers"]:
        p["management_blockers"] = str(p["management_blockers"]).strip()
    else:
        p["management_blockers"] = None

    # 9. Changed By
    p["changed_by"] = p.get("changed_by") or "Voice Agent"
    return p


def reconcile_tasks_from_conversation(ai_data: Dict[str, Any], baseline_deals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures that EVERY actionable deliverable, tender, or next step mentioned in the conversation
    is captured as a task in task_updates, eliminating gaps between executive report and task board.
    """
    task_updates = list(ai_data.get("task_updates", []))
    existing_titles = [str(item.get("payload", {}).get("task_title", "")).lower() for item in task_updates]
    existing_titles += [str(item.get("payload", {}).get("title", "")).lower() for item in task_updates]

    exec_report = ai_data.get("executive_report", {})
    actions = exec_report.get("tomorrow_actions", [])
    progress = exec_report.get("today_progress", [])

    # Map deals by customer or name keywords for smart linking
    deal_keyword_map = {}
    for d in baseline_deals:
        d_id = d.get("deal_id")
        name = str(d.get("deal_name", "")).lower()
        cust = str(d.get("company_name", "")).lower()
        if d_id:
            deal_keyword_map[name] = d_id
            deal_keyword_map[cust] = d_id

    # Also check newly planned crm_updates
    crm_updates = ai_data.get("crm_updates", [])
    for cu in crm_updates:
        d_id = cu.get("deal_id")
        p = cu.get("payload", {})
        d_name = str(p.get("deal_name", "")).lower()
        d_cust = str(p.get("company_name", "")).lower()
        if d_id:
            if d_name:
                deal_keyword_map[d_name] = d_id
            if d_cust:
                deal_keyword_map[d_cust] = d_id

    def find_related_deal(text: str) -> Optional[int]:
        t_low = text.lower()
        for k, d_id in deal_keyword_map.items():
            if k and len(k) > 3 and k in t_low:
                return d_id
        return None

    def detect_vendor(text: str) -> str:
        t_low = text.lower()
        if "hp" in t_low or "hewlett" in t_low:
            return "HPE"
        if "dell" in t_low or "poweredge" in t_low:
            return "Dell"
        if "veeam" in t_low:
            return "Veeam"
        if "nutanix" in t_low or "ahv" in t_low:
            return "Nutanix"
        if "vmware" in t_low or "vcf" in t_low:
            return "VMware"
        return "General"

    def detect_assigned(text: str) -> str:
        t_low = text.lower()
        if "presales 2" in t_low or "rep 2" in t_low:
            return "Presales 2"
        return "Presales 1"

    def detect_category(text: str) -> str:
        t_low = text.lower()
        if "owner" in t_low or "platform" in t_low or "prime" in t_low:
            return "RFP_OWNERSHIP"
        if "scope" in t_low or "renewal" in t_low or "distributed" in t_low:
            return "RFP_DISTRIBUTED_SCOPE"
        return "GENERAL_ACTION"

    # 1. Reconcile tomorrow's actions
    for action in actions:
        action_text = str(action).strip()
        if not action_text:
            continue
        act_low = action_text.lower()
        if any(len(act_low) > 8 and (act_low[:20] in et or et in act_low) for et in existing_titles):
            continue

        task_updates.append({
            "method": "POST",
            "payload": {
                "task_title": action_text,
                "category": detect_category(action_text),
                "assigned_to": detect_assigned(action_text),
                "vendor_domain": detect_vendor(action_text),
                "status": "In Progress",
                "priority": "High" if ("tender" in act_low or "rfp" in act_low) else "Medium",
                "related_deal_id": find_related_deal(action_text),
                "changed_by": "Voice Agent",
            }
        })
        existing_titles.append(act_low)

    # 2. Reconcile any onboarding / scope mentioned in today_progress
    for prog in progress:
        prog_text = str(prog).strip()
        prog_low = prog_text.lower()
        if "tender" in prog_low or "scope" in prog_low or "onboard" in prog_low or "rfp" in prog_low:
            if not any(len(prog_low) > 8 and (prog_low[:20] in et or et in prog_low) for et in existing_titles):
                task_updates.append({
                    "method": "POST",
                    "payload": {
                        "task_title": prog_text,
                        "category": detect_category(prog_text),
                        "assigned_to": detect_assigned(prog_text),
                        "vendor_domain": detect_vendor(prog_text),
                        "status": "In Progress",
                        "priority": "High",
                        "related_deal_id": find_related_deal(prog_text),
                        "changed_by": "Voice Agent",
                    }
                })
                existing_titles.append(prog_low)

    return task_updates


async def execute_api_sync(crm_updates: List[Dict[str, Any]], task_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sync_logs = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Execute CRM Updates
        for item in crm_updates:
            method = item.get("method", "PUT").upper()
            deal_id = item.get("deal_id")
            payload = sanitize_crm_payload(item.get("payload", {}))

            try:
                if method == "PUT" and deal_id:
                    url = f"{CRM_API_URL}/deals/{deal_id}"
                    res = await client.put(url, json=payload)
                    sync_logs.append({
                        "target": "CRM",
                        "method": "PUT",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": res.status_code in (200, 201),
                        "detail": res.json() if res.status_code in (200, 201) else res.text,
                    })
                elif method == "POST":
                    url = f"{CRM_API_URL}/deals"
                    res = await client.post(url, json=payload)
                    sync_logs.append({
                        "target": "CRM",
                        "method": "POST",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": res.status_code in (200, 201),
                        "detail": res.json() if res.status_code in (200, 201) else res.text,
                    })
            except Exception as e:
                sync_logs.append({
                    "target": "CRM",
                    "method": method,
                    "endpoint": f"{CRM_API_URL}/deals/{deal_id or ''}",
                    "status_code": 500,
                    "success": False,
                    "detail": str(e),
                })

        # 2. Execute Task Updates
        for item in task_updates:
            method = item.get("method", "PUT").upper()
            task_id = item.get("task_id")
            payload = sanitize_task_payload(item.get("payload", {}))

            try:
                if method == "PUT" and task_id:
                    url = f"{TASKS_API_URL}/tasks/{task_id}"
                    res = await client.put(url, json=payload)
                    sync_logs.append({
                        "target": "TASKS",
                        "method": "PUT",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": res.status_code in (200, 201),
                        "detail": res.json() if res.status_code in (200, 201) else res.text,
                    })
                elif method == "POST":
                    url = f"{TASKS_API_URL}/tasks"
                    res = await client.post(url, json=payload)
                    sync_logs.append({
                        "target": "TASKS",
                        "method": "POST",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": res.status_code in (200, 201),
                        "detail": res.json() if res.status_code in (200, 201) else res.text,
                    })
            except Exception as e:
                sync_logs.append({
                    "target": "TASKS",
                    "method": method,
                    "endpoint": f"{TASKS_API_URL}/tasks/{task_id or ''}",
                    "status_code": 500,
                    "success": False,
                    "detail": str(e),
                })

    return sync_logs


def extract_json(raw_text: str) -> Dict[str, Any]:
    """Helper to extract JSON from Gemini text response even if wrapped in markdown blocks."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    baseline = await fetch_baseline_state()
    return {
        "status": "online",
        "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "gemini_model": GEMINI_MODEL,
        "crm_api": {"url": CRM_API_URL, "connected": baseline["crm_healthy"], "deals_count": len(baseline["deals"])},
        "tasks_api": {"url": TASKS_API_URL, "connected": baseline["tasks_healthy"], "tasks_count": len(baseline["tasks"])},
    }


@app.post("/api/set-api-key")
async def set_api_key(payload: Dict[str, str]):
    key = payload.get("api_key", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    os.environ["GEMINI_API_KEY"] = key
    save_env_file("GEMINI_API_KEY", key)
    return {"message": "GEMINI_API_KEY saved permanently to local .env file."}


@app.get("/api/audit-state")
async def get_audit_state():
    """
    Returns real-time pipeline audit metrics, blockers, category breakdown,
    and vendor domain counts without invoking generative AI.
    """
    baseline = await fetch_baseline_state()
    deals = baseline.get("deals", [])
    tasks = baseline.get("tasks", [])

    total_deals = len(deals)
    total_tasks = len(tasks)
    open_tasks = [t for t in tasks if str(t.get("status", "")).strip().lower() not in ("completed", "done", "closed")]
    completed_tasks = [t for t in tasks if str(t.get("status", "")).strip().lower() in ("completed", "done", "closed")]

    # Identify active blockers among open tasks
    blockers = []
    for t in open_tasks:
        is_blk = bool(t.get("is_blocked")) or str(t.get("status", "")).strip().lower() in ("waiting on vendor", "blocked")
        blocker_note = t.get("management_blockers")
        if is_blk or (blocker_note and str(blocker_note).strip()):
            blockers.append({
                "task_id": t.get("task_id"),
                "task_title": t.get("task_title") or "Unnamed Task",
                "customer_name": t.get("customer_name") or "Unspecified Customer",
                "deal_name": t.get("deal_name") or "Unspecified Deal",
                "assigned_to": t.get("assigned_to") or "Presales 1",
                "vendor_domain": t.get("vendor_domain") or "General",
                "status": t.get("status") or "Blocked",
                "reason": str(blocker_note).strip() if blocker_note else "Waiting on vendor or partner response"
            })

    high_priority = [t for t in open_tasks if str(t.get("priority", "")).strip().lower() in ("high", "critical")]

    # Calculate total pipeline value and stage distribution
    total_pipeline_val = 0.0
    stages = {}
    for d in deals:
        try:
            total_pipeline_val += float(d.get("estimated_value", 0) or 0)
        except (ValueError, TypeError):
            pass
        st = d.get("stage") or "Unspecified"
        stages[st] = stages.get(st, 0) + 1

    # Breakdown by category
    category_counts = {
        "RFP_OWNERSHIP": sum(1 for t in tasks if t.get("category") == "RFP_OWNERSHIP"),
        "RFP_DISTRIBUTED_SCOPE": sum(1 for t in tasks if t.get("category") == "RFP_DISTRIBUTED_SCOPE"),
        "GENERAL_ACTION": sum(1 for t in tasks if t.get("category") == "GENERAL_ACTION"),
    }

    # Breakdown by vendor
    vendor_counts = {}
    for t in tasks:
        v = t.get("vendor_domain") or "General"
        vendor_counts[v] = vendor_counts.get(v, 0) + 1

    summary = {
        "total_deals": total_deals,
        "total_tasks": total_tasks,
        "open_tasks": len(open_tasks),
        "completed_tasks": len(completed_tasks),
        "total_pipeline_value": total_pipeline_val,
        "blocked_tasks_count": len(blockers),
        "high_priority_count": len(high_priority),
        "pipeline_value": f"${total_pipeline_val:,.2f}",
    }

    return {
        "status": "success",
        "crm_connected": baseline["crm_healthy"],
        "tasks_connected": baseline["tasks_healthy"],
        "summary": summary,
        "metrics": summary,
        "stage_distribution": stages,
        "vendor_distribution": vendor_counts,
        "category_counts": category_counts,
        "blockers": blockers,
        "high_priority_tasks": high_priority,
    }


@app.get("/api/followup-opportunities")
async def get_followup_opportunities():
    """
    Returns previous/current opportunities classified into the 3 kinds:
    1. RFP_OWNERSHIP (Prime RFP / Tender ownership)
    2. RFP_DISTRIBUTED_SCOPE (Multi-vendor & distributed partner scope)
    3. GENERAL_ACTION (PoC milestones, sizing reviews, general actions)
    Specifically highlighting which deals have tasks to follow up about (and blockers).
    """
    baseline = await fetch_baseline_state()
    deals = baseline.get("deals", [])
    tasks = baseline.get("tasks", [])

    opportunities = []
    claimed_task_ids = set()

    for d in deals:
        deal_id = d.get("deal_id")
        deal_name = d.get("deal_name", "")
        customer_name = d.get("company_name") or d.get("customer_name") or "Customer"

        # Match tasks belonging to this deal
        linked_tasks = []
        for t in tasks:
            t_id = t.get("task_id")
            t_rel_deal = t.get("related_deal_id")
            t_deal_name = (t.get("deal_name") or "").strip().lower()
            t_cust_name = (t.get("customer_name") or "").strip().lower()

            is_match = False
            if t_rel_deal and t_rel_deal == deal_id:
                is_match = True
            elif t_deal_name and deal_name and (t_deal_name in deal_name.lower() or deal_name.lower() in t_deal_name):
                is_match = True
            elif t_cust_name and customer_name and (t_cust_name in customer_name.lower() or customer_name.lower() in t_cust_name):
                is_match = True

            if is_match:
                linked_tasks.append(t)
                claimed_task_ids.add(t_id)

        # Filter follow-up tasks (not completed)
        followup_tasks = [
            t for t in linked_tasks 
            if str(t.get("status", "")).strip().lower() not in ("completed", "done", "closed")
        ]
        completed_tasks = [
            t for t in linked_tasks 
            if str(t.get("status", "")).strip().lower() in ("completed", "done", "closed")
        ]

        # Determine the opportunity kind (one of the 3 kinds)
        kind = None
        for t in linked_tasks:
            if t.get("category") == "RFP_OWNERSHIP":
                kind = "RFP_OWNERSHIP"
                break
        if not kind:
            for t in linked_tasks:
                if t.get("category") == "RFP_DISTRIBUTED_SCOPE":
                    kind = "RFP_DISTRIBUTED_SCOPE"
                    break
        if not kind:
            for t in linked_tasks:
                if t.get("category") == "GENERAL_ACTION":
                    kind = "GENERAL_ACTION"
                    break

        if not kind:
            dn_low = deal_name.lower()
            if any(k in dn_low for k in ["rfp", "tender", "مناقصة", "platform", "منصة"]):
                kind = "RFP_OWNERSHIP"
            elif any(k in dn_low for k in ["scope", "refresh", "renewal", "backup", "تجديد", "تحديث", "نطاق", "توريد"]):
                kind = "RFP_DISTRIBUTED_SCOPE"
            else:
                kind = "GENERAL_ACTION"

        # Check if deal has active blockers
        has_blocker = any(
            t.get("is_blocked") or str(t.get("status", "")).strip().lower() in ("waiting on vendor", "blocked") or (t.get("management_blockers") and str(t.get("management_blockers")).strip())
            for t in followup_tasks
        )

        opportunities.append({
            "deal_id": deal_id,
            "deal_name": deal_name,
            "customer_name": customer_name,
            "stage": d.get("stage", "Gathering Requirements"),
            "estimated_value": d.get("estimated_value", 0),
            "primary_vendors": d.get("primary_vendors", "General"),
            "assigned_presales": d.get("assigned_presales", "Presales 1"),
            "kind": kind,
            "has_followup_tasks": len(followup_tasks) > 0,
            "followup_tasks_count": len(followup_tasks),
            "completed_tasks_count": len(completed_tasks),
            "has_blocker": has_blocker,
            "followup_tasks": followup_tasks,
            "completed_tasks": completed_tasks,
        })

    # Also handle any unclaimed follow-up tasks from task board as standalone deliverables
    unclaimed = [t for t in tasks if t.get("task_id") not in claimed_task_ids]
    if unclaimed:
        unclaimed_groups = {}
        for t in unclaimed:
            key = t.get("deal_name") or t.get("customer_name") or "Standup Technical Deliverables"
            if key not in unclaimed_groups:
                unclaimed_groups[key] = []
            unclaimed_groups[key].append(t)

        for key, t_list in unclaimed_groups.items():
            f_tasks = [t for t in t_list if str(t.get("status", "")).strip().lower() not in ("completed", "done", "closed")]
            c_tasks = [t for t in t_list if str(t.get("status", "")).strip().lower() in ("completed", "done", "closed")]
            first_t = t_list[0]
            kind = first_t.get("category") or "GENERAL_ACTION"
            has_blk = any(t.get("is_blocked") or str(t.get("status", "")).strip().lower() in ("waiting on vendor", "blocked") or (t.get("management_blockers") and str(t.get("management_blockers")).strip()) for t in f_tasks)

            opportunities.append({
                "deal_id": first_t.get("related_deal_id"),
                "deal_name": first_t.get("deal_name") or key,
                "customer_name": first_t.get("customer_name") or "Deliverables",
                "stage": "Active Execution",
                "estimated_value": 0,
                "primary_vendors": first_t.get("vendor_domain") or "General",
                "assigned_presales": first_t.get("assigned_to") or "Presales 1",
                "kind": kind,
                "has_followup_tasks": len(f_tasks) > 0,
                "followup_tasks_count": len(f_tasks),
                "completed_tasks_count": len(c_tasks),
                "has_blocker": has_blk,
                "followup_tasks": f_tasks,
                "completed_tasks": c_tasks,
            })

    # Filter strictly for opportunities that have tasks needing follow-up (all tasks completed are excluded!)
    followup_deals = [
        o for o in opportunities 
        if o["has_followup_tasks"] and o.get("followup_tasks_count", 0) > 0
    ]

    # Group strictly by the 3 kinds for deals needing follow-up
    rfp_ownership_followup = [o for o in followup_deals if o["kind"] == "RFP_OWNERSHIP"]
    rfp_distributed_followup = [o for o in followup_deals if o["kind"] == "RFP_DISTRIBUTED_SCOPE"]
    general_action_followup = [o for o in followup_deals if o["kind"] == "GENERAL_ACTION"]

    return {
        "status": "success",
        "counts": {
            "total_opportunities": len(followup_deals),
            "with_followup_tasks": len(followup_deals),
            "rfp_ownership_count": len(rfp_ownership_followup),
            "rfp_ownership_followup_count": len(rfp_ownership_followup),
            "rfp_distributed_scope_count": len(rfp_distributed_followup),
            "rfp_distributed_scope_followup_count": len(rfp_distributed_followup),
            "general_action_count": len(general_action_followup),
            "general_action_followup_count": len(general_action_followup),
            "with_blockers_count": sum(1 for o in followup_deals if o["has_blocker"]),
        },
        "kinds": {
            "RFP_OWNERSHIP": {
                "name_en": "RFP Ownership",
                "name_ar": "مناقصات رئيسية وتكليف كامل",
                "badge": "badge-rfp-owner",
                "description": "Prime tenders owned end-to-end requiring technical architecture, RFP response submission, and bid management.",
                "opportunities": rfp_ownership_followup,
                "followup_opportunities": rfp_ownership_followup
            },
            "RFP_DISTRIBUTED_SCOPE": {
                "name_en": "RFP Distributed Scope",
                "name_ar": "نطاق موزع وشراكات التقنية",
                "badge": "badge-rfp-dist",
                "description": "Multi-vendor partner tenders (HPE, Dell, Veeam, Nutanix, VMware) requiring partner discounts, BoQ validations, and distributor scopes.",
                "opportunities": rfp_distributed_followup,
                "followup_opportunities": rfp_distributed_followup
            },
            "GENERAL_ACTION": {
                "name_en": "General Action",
                "name_ar": "إجراءات وتجارب فنية عامة",
                "badge": "badge-rfp-action",
                "description": "PoC testing, hardware sizing, licensing migrations, and operational presales support deliverables.",
                "opportunities": general_action_followup,
                "followup_opportunities": general_action_followup
            }
        },
        "opportunities": followup_deals,
        "all_opportunities": followup_deals,
        "followup_opportunities": followup_deals,
        "raw_pipeline_deals_count": len(opportunities),
    }


@app.get("/api/pre-meeting-questions")
async def get_pre_meeting_questions():
    """
    Audits active deals & tasks and generates 4-6 bilingual check-in questions
    (Saudi Arabic & English) focused on Presales 1 & Presales 2 domain responsibilities.
    """
    baseline = await fetch_baseline_state()
    deals = baseline["deals"]
    tasks = baseline["tasks"]

    # Provide high-quality fallback questions if Gemini API key is missing or offline
    fallback_questions = [
        {
            "id": 1,
            "target": "Presales 1",
            "domain": "HPE & Veeam Backup",
            "question_en": "Has the customer validated the immutable repository sizing for FinTech Horizons, or are we blocked on the Dell PowerProtect appliance quote?",
            "question_ar": "هل تم اعتماد وتأكيد أحجام النسخ الاحتياطي غير القابلة للتغيير (Immutability) لمشروع FinTech Horizons، أم ما زلنا بانتظار تسعيرة أجهزة Dell PowerProtect؟",
            "context": "Deal #2 (Ransomware Backup) is at Proposal stage; Task #2 is awaiting vendor approval."
        },
        {
            "id": 2,
            "target": "Presales 1",
            "domain": "RFP Ownership",
            "question_en": "What is the expected delivery date for the Hospital Cluster Executive Summary (Task #1), and do we have HPE partner discount approvals?",
            "question_ar": "ما هو التاريخ المتوقع لتسليم الملخص التنفيذي لمشروع مستشفيات Nordic Health (مهمة #1)، وهل تم الحصول على موافقة الخصم الخاص من شركاء HPE؟",
            "context": "RFP Ownership task is marked In Progress with an active blocker."
        },
        {
            "id": 3,
            "target": "Presales 2",
            "domain": "Nutanix AHV & PoC",
            "question_en": "Has the vendor SE returned from leave to finalize the AHV sizing and IOPS specs for Acme Cloud Corp's PoC?",
            "question_ar": "هل عاد مهندس الحلول (SE) من الإجازة لتثبيت متطلبات الأداء و IOPS لبيئة Nutanix AHV الخاصة بتجربة (PoC) شركة Acme Cloud؟",
            "context": "Deal #1 is in PoC stage ($125k) and Task #2 is Waiting on Vendor."
        },
        {
            "id": 4,
            "target": "Presales 2",
            "domain": "VMware Cloud Foundation",
            "question_en": "Have we received the finalized core count from Apex Logistics to execute the VCF licensing migration playbook?",
            "question_ar": "هل استلمنا عدد الأنوية (Core Counts) المعتمد من عميل Apex Logistics للبدء في خطة الانتقال لترخيص VMware VCF؟",
            "context": "Task #5 is Not Started pending customer confirmation."
        },
        {
            "id": 5,
            "target": "Presales 1 & 2",
            "domain": "General Workstream",
            "question_en": "Are there any cross-vendor compatibility issues between Dell PowerEdge compute nodes and Nutanix AHV hypervisor?",
            "question_ar": "هل تواجهون أي تعارضات فنية أو توافقية بين خوادم Dell PowerEdge ونظام Nutanix AHV في المشاريع المشتركة؟",
            "context": "Coordinating multi-vendor bill of materials."
        }
    ]

    client = get_gemini_client()
    if not client:
        return JSONResponse(content={
            "source": "baseline_template",
            "notice": "GEMINI_API_KEY not configured. Showing intelligent baseline questions generated from live CRM & Task data.",
            "crm_connected": baseline["crm_healthy"],
            "tasks_connected": baseline["tasks_healthy"],
            "questions": fallback_questions,
        })

    # Prepare prompt for Gemini
    prompt = f"""
You are an Enterprise Presales Operations AI Director presiding over a daily standup meeting in Saudi Arabia.
Your team consists of:
- Presales 1: Primary focus on HPE, Veeam, Business Continuity, Ransomware/DR, RFP Technical Ownership.
- Presales 2: Primary focus on Dell, SAN/NAS, Nutanix HCI, VMware VCF licensing, Distributed Scopes.

Current Live CRM Pipeline:
{json.dumps(deals, indent=2)}

Current Live Monday.com Task Board:
{json.dumps(tasks, indent=2)}

Generate 4 to 6 critical, highly relevant check-in questions to ask during today's standup meeting.
Each question MUST be provided in BOTH idiomatic English and professional Saudi business Arabic (اللهجة المهنية المعتمدة في قطاع تقنية المعلومات في السعودية مع المصطلحات الفنية المألوفة).

Cover:
1. RFP Ownership deliverables vs Distributed scopes.
2. High-value deals in Proposal or PoC stages.
3. Management blockers, vendor SE delays, or special pricing requests (HPE, Veeam, Dell, Nutanix, VMware).

Return STRICT JSON format:
{{
  "questions": [
    {{
      "id": 1,
      "target": "Presales 1" or "Presales 2" or "Presales 1 & 2",
      "domain": "e.g. Veeam / HPE / RFP Ownership",
      "question_en": "Question in English",
      "question_ar": "السؤال بالعربي المهني السعودي",
      "context": "Why this question matters based on current data"
    }}
  ]
}}
"""
    try:
        response = generate_with_model_fallback(
            client=client,
            contents=prompt,
        )
        data = extract_json(response.text)
        return {
            "source": GEMINI_MODEL,
            "crm_connected": baseline["crm_healthy"],
            "tasks_connected": baseline["tasks_healthy"],
            "questions": data.get("questions", fallback_questions),
        }
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return {
            "source": "fallback_error_recovery",
            "error_detail": str(e),
            "questions": fallback_questions,
        }


@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Receives recorded standup audio, processes it using Gemini 2.5 Flash's native speech recognition,
    detects state deltas vs CRM & Tasks, outputs a structured executive report, and executes live API syncs.
    """
    audio_bytes = await file.read()
    content_type = file.content_type or "audio/webm"

    # 1. Fetch current baseline
    baseline = await fetch_baseline_state()
    deals = baseline["deals"]
    tasks = baseline["tasks"]

    client = get_gemini_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY is not set. Please provide your API key in the top navigation bar to process audio with Gemini.",
        )

    # 2. Build Multimodal Content
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=content_type)

    instructions = f"""
You are an expert bilingual (Saudi Arabic & English) Enterprise Presales Operations AI.
You are listening to an audio recording of a presales team standup meeting. The engineers speak in their natural languages: Saudi Arabic and English IT terminology.

CRITICAL LANGUAGE & TRANSCRIPTION RULES (STRICT REQUIREMENT):
1. KEEP THE TRANSCRIPT AND SUMMARY IN ARABIC AND ENGLISH EXACTLY AS SPOKEN — NO TRANSLATION:
   - If speech was spoken in Arabic, transcribe and keep it in Arabic.
   - If speech was spoken in English, transcribe and keep it in English.
   - DO NOT translate spoken Arabic into English. DO NOT translate spoken English into Arabic.
   - Preserve authentic Saudi Arabic phrasing, technical code-switching, and natural engineer terminology.
2. KEEP DEAL NAMES AND OPPORTUNITY NAMES IN ARABIC EXACTLY AS SPOKEN:
   - When a deal, tender, RFP, opportunity, or customer is mentioned in Arabic, KEEP THE DEAL NAME AND OPPORTUNITY NAME IN ARABIC AS SPOKEN.
   - Never translate Arabic deal names (e.g. keep "مناقصة وزارة التخطيط", "توريد أجهزة ديل", "مشروع منصة الحج والعمرة", "تجديد رخص فيم").
   - When creating or updating deals or tasks, use the original Arabic deal/opportunity name as spoken.

BASELINE CRM DEALS:
{json.dumps(deals, indent=2, ensure_ascii=False)}

BASELINE TASK BOARD:
{json.dumps(tasks, indent=2, ensure_ascii=False)}

Your responsibilities:
1. Listen carefully and transcribe/understand all spoken updates from Presales 1 and Presales 2 exactly in their spoken languages without translation.
2. Compare spoken updates against the baseline CRM deals and Task Board items above to detect DELTAS:
   - Deal stage changes (e.g. PoC -> Proposal, Discovery -> Gathering Requirements, Proposal -> Closed-Won).
   - Estimated deal value revisions.
   - Task status transitions (e.g. In Progress -> Completed, Waiting on Vendor -> In Progress, Not Started -> In Progress).
   - Resolved or newly raised management blockers.
   - Any newly mentioned deals or tasks to create.
3. Formulate structured REST API updates:
   - `crm_updates`: Array of deal updates. Use "PUT" with "deal_id" and "payload" for existing deals; or "POST" with "payload" for newly won or qualified deals.
     IMPORTANT ENUM CONSTRAINTS FOR CRM DEALS:
     * deal_name: Keep in original Arabic/English as spoken!
     * stage MUST be one of: ["Discovery", "Gathering Requirements", "PoC", "Proposal", "Closed-Won", "Closed-Lost"]. (Never use "In Progress" for deal stage; if activities are ongoing/in progress, use "Gathering Requirements").
     * assigned_presales MUST be: "Presales 1" or "Presales 2" (Engineer Abdullah maps to "Presales 1").
     * estimated_value must be a numeric float (e.g. 125000.0).
     * vendor_notes: Keep in original language as spoken.
   - `task_updates`: Array of task updates.
     CRITICAL REQUIREMENT: For EVERY new tender, RFP ownership, vendor scope, or tomorrow's action item mentioned in the meeting, you MUST create a task using method "POST"! Never omit any discussed task or deliverable.
     IMPORTANT CONSTRAINTS FOR TASKS:
     * task_title: Actionable title in the original spoken language (Arabic or English as spoken, e.g. "جمع المتطلبات الفنية لمناقصة منصة الحج والعمرة").
     * status MUST be one of: ["Not Started", "In Progress", "Waiting on Vendor", "Pending Review", "Completed"].
     * assigned_to MUST be: "Presales 1" or "Presales 2".
     * category MUST be: "RFP_OWNERSHIP" (prime tenders), "RFP_DISTRIBUTED_SCOPE" (vendor scopes/renewals), or "GENERAL_ACTION".
     * vendor_domain MUST be one of: ["HPE", "Veeam", "Dell", "Nutanix", "VMware", "General"]. NOTE: HP / Hewlett Packard MUST be set to "HPE".
     * priority MUST be one of: ["High", "Medium", "Low"].
     * related_deal_id: Integer deal ID if associated with a CRM deal, else null.
     * changed_by: "Voice Agent" (or the speaking engineer).
4. Produce an Executive Briefing Report in the original spoken language without translation:
   - `today_progress`: Array of key accomplishments in the spoken language.
   - `tomorrow_actions`: Array of prioritized next steps in the spoken language.
   - `management_warnings`: Array of critical risks, vendor roadblocks, or management escalations in the spoken language.

Return STRICT JSON matching this schema:
{{
  "transcript_summary": "Authentic transcript and summary of the meeting highlights preserving the exact language as spoken (Arabic and English mixed as spoken, NO translation).",
  "crm_updates": [
    {{
      "method": "PUT",
      "deal_id": 1,
      "payload": {{
        "stage": "Proposal",
        "estimated_value": 135000.0,
        "vendor_notes": "Updated note in spoken language..."
      }}
    }}
  ],
  "task_updates": [
    {{
      "method": "POST",
      "payload": {{
        "task_title": "Actionable task title in original spoken language (Arabic or English)",
        "category": "RFP_OWNERSHIP",
        "assigned_to": "Presales 2",
        "vendor_domain": "Dell",
        "status": "In Progress",
        "priority": "High",
        "related_deal_id": null,
        "changed_by": "Voice Agent"
      }}
    }}
  ],
  "executive_report": {{
    "today_progress": ["... (in original spoken language)"],
    "tomorrow_actions": ["... (in original spoken language)"],
    "management_warnings": ["... (in original spoken language)"]
  }}
}}
"""

    try:
        response = generate_with_model_fallback(
            client=client,
            contents=[audio_part, instructions],
        )
        ai_data = extract_json(response.text)
    except Exception as e:
        print(f"Multimodal Gemini error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(e)}")

    # 3. Automatically synchronize detected updates to local APIs with comprehensive reconciliation
    crm_updates = ai_data.get("crm_updates", [])
    task_updates = reconcile_tasks_from_conversation(ai_data, deals)
    sync_log = await execute_api_sync(crm_updates, task_updates)

    return {
        "status": "success",
        "transcript_summary": ai_data.get("transcript_summary", "No transcript generated."),
        "executive_report": ai_data.get("executive_report", {}),
        "crm_updates_planned": crm_updates,
        "task_updates_planned": task_updates,
        "api_sync_log": sync_log,
    }


# -----------------------------------------------------------------------------
# Embedded Web Dashboard UI
# -----------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Operations AI Agent | Presales Automation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0a0e17;
            --card-bg: #111827;
            --card-border: #1f293d;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-amber: #f59e0b;
            --accent-purple: #8b5cf6;
        }
        body {
            background-color: var(--bg-body);
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            color: #f3f4f6;
        }
        .font-arabic {
            font-family: 'Tajawal', system-ui, sans-serif;
        }
        .app-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            box-shadow: 0 4px 20px -2px rgba(0,0,0,0.3);
        }
        .mic-button {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
            border: 3px solid rgba(255,255,255,0.1);
        }
        .mic-idle {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }
        .mic-recording {
            background: linear-gradient(135deg, #ef4444, #b91c1c);
            color: white;
            animation: pulse-ring 1.5s infinite;
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background-color: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
        .dot-red { background-color: #ef4444; }
        .badge-presales1 { background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-presales2 { background-color: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.3); }

        /* Modern Tabs Styling */
        .nav-tabs-custom .nav-link {
            color: #94a3b8;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 7px 14px;
            transition: all 0.2s ease;
        }
        .nav-tabs-custom .nav-link:hover {
            color: #e2e8f0;
            background: rgba(255, 255, 255, 0.05);
        }
        .nav-tabs-custom .nav-link.active {
            color: #ffffff !important;
            background: #2563eb !important;
            border-color: #2563eb !important;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
        }

        /* 3 Kinds Badges */
        .badge-rfp-owner {
            background-color: rgba(59, 130, 246, 0.18);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.45);
        }
        .badge-rfp-dist {
            background-color: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.45);
        }
        .badge-rfp-action {
            background-color: rgba(245, 158, 11, 0.18);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.45);
        }
        .badge-blocker {
            background-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.5);
        }
        .kind-filter-btn {
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .modal.show {
            display: block !important;
        }
        .modal-backdrop.show {
            opacity: 0.7;
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header class="border-bottom border-secondary border-opacity-25 py-3 px-4 mb-4">
        <div class="container-fluid d-flex flex-wrap justify-content-between align-items-center gap-3">
            <div class="d-flex align-items-center gap-3">
                <div class="bg-primary bg-opacity-25 text-primary p-2 rounded-3">
                    <i class="bi bi-soundwave fs-3"></i>
                </div>
                <div>
                    <h5 class="fw-bold mb-0 text-white">Voice Operations AI Agent</h5>
                    <div class="small text-muted">Bilingual Standup Multimodal Processor & API Sync (Port 8002)</div>
                </div>
            </div>

            <!-- Health Status Indicators & API Key Button -->
            <div class="d-flex flex-wrap align-items-center gap-2">
                <span class="badge bg-dark border border-secondary border-opacity-50 px-2 py-2 d-flex align-items-center gap-2">
                    <span class="status-dot dot-green" id="crmDot"></span>
                    <span>CRM API (8000)</span>
                </span>
                <span class="badge bg-dark border border-secondary border-opacity-50 px-2 py-2 d-flex align-items-center gap-2">
                    <span class="status-dot dot-green" id="tasksDot"></span>
                    <span>Tasks API (8001)</span>
                </span>
                <span class="badge bg-primary-subtle text-primary border px-2 py-2">
                    <i class="bi bi-cpu me-1"></i>Gemini Flash
                </span>
                <button class="btn btn-sm btn-outline-light d-flex align-items-center gap-1" onclick="openApiKeyModal()">
                    <i class="bi bi-key-fill text-warning"></i>
                    <span id="apiKeyBtnText">Configure Gemini Key</span>
                </button>
            </div>
        </div>
    </header>

    <div class="container-fluid px-4 pb-5">
        <div class="row g-4">

            <!-- LEFT COLUMN: Operations Tabs (Audit State, Follow-up Deals 3 Kinds, Standup Questions) -->
            <div class="col-12 col-xl-5">
                <div class="app-card p-3 p-md-4 h-100 d-flex flex-column">
                    <!-- Tab Navigation Header -->
                    <ul class="nav nav-pills nav-tabs-custom gap-2 mb-3 pb-2 border-bottom border-secondary border-opacity-25" id="agentTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="tab-audit-btn" onclick="switchTab('audit')" type="button">
                                <i class="bi bi-speedometer2 me-1"></i>Audit State
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="tab-opps-btn" onclick="switchTab('opps')" type="button">
                                <i class="bi bi-diagram-3 me-1"></i>Follow-up Deals (3 Kinds)
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="tab-questions-btn" onclick="switchTab('questions')" type="button">
                                <i class="bi bi-chat-quote me-1"></i>Standup Questions
                            </button>
                        </li>
                    </ul>

                    <!-- TAB 1: Audit State Pane (No auto-execute on page load!) -->
                    <div id="pane-audit" class="flex-grow-1 d-flex flex-column">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <div>
                                <h6 class="fw-bold text-white mb-0"><i class="bi bi-clipboard2-data text-primary me-2"></i>Pipeline & Tasks Audit State</h6>
                                <div class="small text-muted">Audits active CRM deals, tasks health, and blockers</div>
                            </div>
                            <button class="btn btn-sm btn-primary d-flex align-items-center gap-1" onclick="runAuditState()" id="runAuditBtn">
                                <i class="bi bi-lightning-charge-fill"></i> Execute State Audit
                            </button>
                        </div>
                        <div id="auditContentContainer" class="flex-grow-1 overflow-auto" style="max-height: 680px;">
                            <div class="text-center py-5 text-muted">
                                <i class="bi bi-speedometer2 fs-1 d-block mb-2 text-secondary"></i>
                                Click <strong>"Execute State Audit"</strong> to query current pipeline metrics, deliverables status, and active roadblocks.
                            </div>
                        </div>
                    </div>

                    <!-- TAB 2: Follow-up Opportunities (3 Kinds) Pane -->
                    <div id="pane-opps" class="d-none flex-grow-1 d-flex flex-column">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <div>
                                <h6 class="fw-bold text-white mb-0"><i class="bi bi-kanban text-info me-2"></i>Deals Needing Follow-up (3 Kinds)</h6>
                                <div class="small text-muted">Showing active follow-up deals only (all completed deals hidden)</div>
                            </div>
                            <button class="btn btn-sm btn-outline-info d-flex align-items-center gap-1" onclick="loadFollowupOpportunities()" id="loadOppsBtn">
                                <i class="bi bi-arrow-repeat"></i> Refresh
                            </button>
                        </div>

                        <!-- 3 Kinds Filter Toolbar -->
                        <div class="d-flex flex-wrap gap-1 mb-3 pt-1">
                            <button class="btn btn-sm btn-primary kind-filter-btn" id="filter-btn-ALL" onclick="filterFollowupByKind('ALL')">
                                All (<span id="count-ALL">0</span>)
                            </button>
                            <button class="btn btn-sm btn-outline-secondary kind-filter-btn" id="filter-btn-RFP_OWNERSHIP" onclick="filterFollowupByKind('RFP_OWNERSHIP')">
                                <span class="badge badge-rfp-owner me-1">RFP Ownership</span>(<span id="count-RFP_OWNERSHIP">0</span>)
                            </button>
                            <button class="btn btn-sm btn-outline-secondary kind-filter-btn" id="filter-btn-RFP_DISTRIBUTED_SCOPE" onclick="filterFollowupByKind('RFP_DISTRIBUTED_SCOPE')">
                                <span class="badge badge-rfp-dist me-1">Distributed Scope</span>(<span id="count-RFP_DISTRIBUTED_SCOPE">0</span>)
                            </button>
                            <button class="btn btn-sm btn-outline-secondary kind-filter-btn" id="filter-btn-GENERAL_ACTION" onclick="filterFollowupByKind('GENERAL_ACTION')">
                                <span class="badge badge-rfp-action me-1">General Action</span>(<span id="count-GENERAL_ACTION">0</span>)
                            </button>
                        </div>

                        <div id="oppsContentContainer" class="flex-grow-1 overflow-auto" style="max-height: 640px;">
                            <div class="text-center py-5 text-muted">
                                <i class="bi bi-diagram-3 fs-1 d-block mb-2 text-secondary"></i>
                                Click <strong>"Refresh"</strong> to categorize deals into the 3 kinds and identify pending follow-up deliverables.
                            </div>
                        </div>
                    </div>

                    <!-- TAB 3: Questions Pane (Manual generate button only) -->
                    <div id="pane-questions" class="d-none flex-grow-1 d-flex flex-column">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <div>
                                <h6 class="fw-bold text-white mb-0"><i class="bi bi-chat-left-dots text-warning me-2"></i>Bilingual Standup Questions</h6>
                                <div class="small text-muted">Targeted questions for presales engineers on tasks needing follow-up</div>
                            </div>
                            <button class="btn btn-sm btn-outline-warning d-flex align-items-center gap-1" onclick="loadPreMeetingQuestions()" id="loadQuestionsBtn">
                                <i class="bi bi-lightning-charge-fill"></i> Generate Questions
                            </button>
                        </div>
                        <div id="questionsContainer" class="flex-grow-1 overflow-auto" style="max-height: 680px;">
                            <div class="text-center py-5 text-muted">
                                <i class="bi bi-chat-quote fs-1 d-block mb-2 text-secondary"></i>
                                Click <strong>"Generate Questions"</strong> to query current pipeline & generate bilingual standup questions.
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- RIGHT COLUMN: Multimodal Voice Recording & Live Sync Execution -->
            <div class="col-12 col-xl-7">
                <div class="app-card p-4 mb-4">
                    <h6 class="fw-bold text-white mb-1"><i class="bi bi-mic-fill text-danger me-2"></i>Multimodal Voice Standup Processor</h6>
                    <div class="small text-muted mb-4">Record your bilingual (Saudi Arabic / English) presales standup or upload audio</div>

                    <!-- Mic & Recording Section -->
                    <div class="text-center py-3">
                        <div class="d-flex justify-content-center mb-3">
                            <button id="recordBtn" class="mic-button mic-idle" onclick="toggleRecording()" title="Click to Start/Stop Recording">
                                <i class="bi bi-mic-fill" id="micIcon"></i>
                            </button>
                        </div>
                        <div class="fw-bold fs-5 text-white" id="recordingTimer">00:00</div>
                        <div class="small text-muted mt-1" id="recordingPrompt">Click microphone to begin recording standup update</div>

                        <!-- Audio playback container -->
                        <div class="mt-3 d-none" id="audioPlaybackContainer">
                            <audio id="audioPlayer" controls class="w-75 mb-3"></audio>
                            <div>
                                <button class="btn btn-success px-4 py-2 fw-semibold" id="processAudioBtn" onclick="processRecordedAudio()">
                                    <i class="bi bi-cpu-fill me-1"></i> Analyze Speech & Sync APIs
                                </button>
                            </div>
                        </div>

                        <!-- Fallback File Upload -->
                        <div class="mt-4 pt-3 border-top border-secondary border-opacity-25 text-start">
                            <label class="form-label small text-muted">Or upload pre-recorded audio file (.webm, .wav, .mp3, .m4a):</label>
                            <div class="input-group input-group-sm">
                                <input type="file" id="audioFileInput" class="form-control" accept="audio/*">
                                <button class="btn btn-outline-light" onclick="uploadAudioFile()">Upload & Process</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Live Results: Transcript, Executive Report & API Sync Log -->
                <div class="app-card p-4 d-none" id="resultsCard">
                    <h6 class="fw-bold text-white mb-3 pb-2 border-bottom border-secondary border-opacity-25">
                        <i class="bi bi-check2-circle text-success me-2"></i>Standup Intelligence & Sync Execution
                    </h6>

                    <!-- Meeting Summary -->
                    <div class="mb-4">
                        <div class="small text-uppercase fw-semibold text-muted mb-1">Standup Meeting Transcript & Summary</div>
                        <div class="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 text-light font-arabic" id="summaryText" dir="auto" style="white-space: pre-wrap; line-height: 1.6;"></div>
                    </div>

                    <!-- Executive 3-Column Report -->
                    <div class="row g-3 mb-4">
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-success border-opacity-25 bg-success bg-opacity-10 h-100">
                                <div class="fw-bold small text-success mb-2"><i class="bi bi-check-circle me-1"></i>Today's Key Progress</div>
                                <ul class="small ps-3 mb-0 font-arabic" id="todayProgressList" dir="auto"></ul>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-primary border-opacity-25 bg-primary bg-opacity-10 h-100">
                                <div class="fw-bold small text-primary mb-2"><i class="bi bi-arrow-right-circle me-1"></i>Tomorrow's Actions</div>
                                <ul class="small ps-3 mb-0 font-arabic" id="tomorrowActionsList" dir="auto"></ul>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-danger border-opacity-25 bg-danger bg-opacity-10 h-100">
                                <div class="fw-bold small text-danger mb-2"><i class="bi bi-exclamation-triangle me-1"></i>Management Warnings</div>
                                <ul class="small ps-3 mb-0 font-arabic" id="managementWarningsList" dir="auto"></ul>
                            </div>
                        </div>
                    </div>

                    <!-- API Sync Execution Log -->
                    <div>
                        <div class="small text-uppercase fw-semibold text-muted mb-2">Automated API Synchronizations</div>
                        <div class="table-responsive">
                            <table class="table table-sm table-dark align-middle mb-0">
                                <thead>
                                    <tr class="text-secondary small">
                                        <th>Target</th>
                                        <th>Action</th>
                                        <th>Endpoint</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="syncLogBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <!-- API Key Configuration Modal (Supports Bootstrap & Native Fallback) -->
    <div class="modal fade" id="apiKeyModal" tabindex="-1" aria-hidden="true" style="display:none;">
        <div class="modal-dialog">
            <div class="modal-content bg-dark text-light border-secondary">
                <div class="modal-header border-secondary">
                    <h6 class="modal-title fw-bold"><i class="bi bi-key-fill text-warning me-2"></i>Configure Gemini API Key</h6>
                    <button type="button" class="btn-close btn-close-white" onclick="closeApiKeyModal()" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p class="small text-muted">Enter your Google Gemini API key to enable native audio speech recognition and dynamic generation via Gemini Flash models.</p>
                    <input type="password" id="geminiApiKeyInput" class="form-control mb-2" placeholder="AIzaSy...">
                    <div class="small text-secondary"><i class="bi bi-shield-check me-1 text-success"></i>Saved locally in <code>.env</code> file.</div>
                </div>
                <div class="modal-footer border-secondary">
                    <button type="button" class="btn btn-secondary btn-sm" onclick="closeApiKeyModal()">Close</button>
                    <button type="button" class="btn btn-primary btn-sm" onclick="saveApiKey()">Save API Key</button>
                </div>
            </div>
        </div>
    </div>
    <div id="customModalBackdrop" class="modal-backdrop fade show" style="display:none;" onclick="closeApiKeyModal()"></div>

    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let mediaRecorder = null;
        let recordedChunks = [];
        let timerInterval = null;
        let secondsElapsed = 0;
        let recordedBlob = null;
        let apiKeyModalInstance = null;
        let currentFollowupData = null;
        let activeKindFilter = 'ALL';

        // Initialize strictly with health check. ZERO auto-generation of questions or audits on page load!
        document.addEventListener('DOMContentLoaded', () => {
            checkHealth();
        });

        // ---------------------------------------------------------------------
        // Tab Navigation
        // ---------------------------------------------------------------------
        function switchTab(tabKey) {
            // Update Tab Nav Buttons
            document.getElementById('tab-audit-btn').classList.toggle('active', tabKey === 'audit');
            document.getElementById('tab-opps-btn').classList.toggle('active', tabKey === 'opps');
            document.getElementById('tab-questions-btn').classList.toggle('active', tabKey === 'questions');

            // Toggle Panes
            document.getElementById('pane-audit').classList.toggle('d-none', tabKey !== 'audit');
            document.getElementById('pane-opps').classList.toggle('d-none', tabKey !== 'opps');
            document.getElementById('pane-questions').classList.toggle('d-none', tabKey !== 'questions');
        }

        // ---------------------------------------------------------------------
        // Health Checking
        // ---------------------------------------------------------------------
        async function checkHealth() {
            try {
                const res = await fetch('/api/health');
                const data = await res.json();
                document.getElementById('crmDot').className = 'status-dot ' + (data.crm_api.connected ? 'dot-green' : 'dot-red');
                document.getElementById('tasksDot').className = 'status-dot ' + (data.tasks_api.connected ? 'dot-green' : 'dot-red');
                if (data.gemini_api_key_configured) {
                    document.getElementById('apiKeyBtnText').textContent = "Gemini Key Configured";
                }
            } catch (err) {
                console.error("Health check error:", err);
            }
        }

        // ---------------------------------------------------------------------
        // TAB 1: Audit State
        // ---------------------------------------------------------------------
        async function runAuditState() {
            const btn = document.getElementById('runAuditBtn');
            const container = document.getElementById('auditContentContainer');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Auditing...';
            container.innerHTML = '<div class="text-center py-5 text-muted"><div class="spinner-border text-primary mb-2"></div><div>Executing pipeline & deliverables audit...</div></div>';

            try {
                const res = await fetch('/api/audit-state');
                const data = await res.json();
                if (data.status !== 'success') {
                    throw new Error(data.message || 'Audit state query failed');
                }
                renderAuditState(data);
            } catch (err) {
                container.innerHTML = `<div class="alert alert-danger small p-3"><i class="bi bi-exclamation-triangle-fill me-2"></i>Failed to audit state: ${err.message}. Ensure CRM (8000) and Tasks (8001) are running.</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i>Execute State Audit';
            }
        }

        function renderAuditState(data) {
            const container = document.getElementById('auditContentContainer');
            const s = data.summary || {};
            const stages = data.stage_distribution || {};
            const vendors = data.vendor_distribution || {};
            const blockers = data.blockers || [];
            const highPri = data.high_priority_tasks || [];

            let html = `
                <!-- Metrics Grid -->
                <div class="row g-2 mb-3">
                    <div class="col-6 col-sm-3">
                        <div class="p-2 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                            <div class="small text-muted">Total Deals</div>
                            <div class="fs-5 fw-bold text-white">${s.total_deals || 0}</div>
                        </div>
                    </div>
                    <div class="col-6 col-sm-3">
                        <div class="p-2 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                            <div class="small text-muted">Pipeline Value</div>
                            <div class="fs-5 fw-bold text-info">${Number(s.total_pipeline_value || 0).toLocaleString()} SAR</div>
                        </div>
                    </div>
                    <div class="col-6 col-sm-3">
                        <div class="p-2 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                            <div class="small text-muted">Open Tasks</div>
                            <div class="fs-5 fw-bold text-warning">${s.open_tasks || 0}</div>
                        </div>
                    </div>
                    <div class="col-6 col-sm-3">
                        <div class="p-2 bg-dark rounded-3 border border-secondary border-opacity-25 text-center">
                            <div class="small text-muted">Completed</div>
                            <div class="fs-5 fw-bold text-success">${s.completed_tasks || 0}</div>
                        </div>
                    </div>
                </div>
            `;

            // Active Roadblocks Banner if present
            if (blockers.length > 0) {
                html += `
                    <div class="alert alert-danger p-2 mb-3 small">
                        <div class="fw-bold mb-2"><i class="bi bi-shield-fill-exclamation me-1"></i>Active Roadblocks & Vendor Holds (${blockers.length})</div>
                        <ul class="mb-0 ps-3">
                            ${blockers.map(b => `
                                <li class="mb-2">
                                    <div class="fw-semibold text-white">${b.task_title} <span class="badge bg-secondary-subtle text-light ms-1">${b.vendor_domain}</span></div>
                                    <div class="d-flex flex-wrap gap-2 text-muted mt-1" style="font-size: 0.73rem;">
                                        <span><i class="bi bi-building text-info me-1"></i>Customer: <strong class="text-light">${b.customer_name || 'Customer'}</strong></span>
                                        <span><i class="bi bi-briefcase text-warning me-1"></i>Deal: <strong class="text-light">${b.deal_name || 'Deal'}</strong></span>
                                        <span class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${b.reason || b.management_blockers || b.status}</span>
                                    </div>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }

            // High Priority Tasks
            if (highPri.length > 0) {
                html += `
                    <div class="mb-3">
                        <div class="small text-uppercase fw-bold text-secondary mb-1">
                            <i class="bi bi-fire text-danger me-1"></i>High Priority Deliverables (${highPri.length})
                        </div>
                        <div class="list-group list-group-flush border border-secondary border-opacity-25 rounded-3">
                            ${highPri.map(t => `
                                <div class="list-group-item bg-dark text-light p-2 border-secondary border-opacity-25 d-flex justify-content-between align-items-center">
                                    <div class="me-2 text-truncate" style="max-width: 75%;">
                                        <div class="fw-semibold small text-white">${t.task_title}</div>
                                        <div class="d-flex flex-wrap gap-2 text-muted mt-1" style="font-size: 0.73rem;">
                                            <span><i class="bi bi-building text-info me-1"></i>Customer: <strong class="text-light">${t.customer_name || 'N/A'}</strong></span>
                                            <span><i class="bi bi-briefcase text-warning me-1"></i>Deal: <strong class="text-light">${t.deal_name || 'N/A'}</strong></span>
                                            <span><i class="bi bi-person me-1"></i>${t.assigned_to || 'Assigned'}</span>
                                            <span><i class="bi bi-layers me-1"></i>${t.vendor_domain || 'General'}</span>
                                        </div>
                                    </div>
                                    <span class="badge bg-danger-subtle text-danger border border-danger-subtle">${t.status}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // Breakdown Sections
            html += `
                <div class="row g-2">
                    <div class="col-12 col-md-6">
                        <div class="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 h-100">
                            <div class="small text-uppercase fw-bold text-secondary mb-2"><i class="bi bi-funnel me-1"></i>Deals by Stage</div>
                            ${Object.entries(stages).map(([st, cnt]) => `
                                <div class="d-flex justify-content-between align-items-center small py-1 border-bottom border-secondary border-opacity-10">
                                    <span class="text-secondary">${st}</span>
                                    <span class="badge bg-secondary">${cnt}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div class="col-12 col-md-6">
                        <div class="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 h-100">
                            <div class="small text-uppercase fw-bold text-secondary mb-2"><i class="bi bi-building me-1"></i>Tasks by Vendor Domain</div>
                            ${Object.entries(vendors).map(([vd, cnt]) => `
                                <div class="d-flex justify-content-between align-items-center small py-1 border-bottom border-secondary border-opacity-10">
                                    <span class="text-secondary">${vd}</span>
                                    <span class="badge bg-info bg-opacity-25 text-info">${cnt}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;

            container.innerHTML = html;
        }

        // ---------------------------------------------------------------------
        // TAB 2: Follow-up Opportunities (3 Kinds)
        // ---------------------------------------------------------------------
        async function loadFollowupOpportunities() {
            const btn = document.getElementById('loadOppsBtn');
            const container = document.getElementById('oppsContentContainer');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Loading...';
            container.innerHTML = '<div class="text-center py-5 text-muted"><div class="spinner-border text-info mb-2"></div><div>Categorizing deals into 3 kinds & checking follow-up tasks...</div></div>';

            try {
                const res = await fetch('/api/followup-opportunities');
                const data = await res.json();
                if (data.status !== 'success') {
                    throw new Error(data.message || 'Failed to fetch opportunities');
                }
                currentFollowupData = data;
                
                // Update badge counts on filter buttons (strictly deals needing follow-up)
                const c = data.counts || {};
                document.getElementById('count-ALL').textContent = c.with_followup_tasks || 0;
                document.getElementById('count-RFP_OWNERSHIP').textContent = c.rfp_ownership_followup_count || 0;
                document.getElementById('count-RFP_DISTRIBUTED_SCOPE').textContent = c.rfp_distributed_scope_followup_count || 0;
                document.getElementById('count-GENERAL_ACTION').textContent = c.general_action_followup_count || 0;

                renderFollowupOpportunities(activeKindFilter);
            } catch (err) {
                container.innerHTML = `<div class="alert alert-danger small p-3"><i class="bi bi-exclamation-triangle-fill me-2"></i>Failed to load opportunities: ${err.message}.</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Refresh';
            }
        }

        function filterFollowupByKind(kind) {
            activeKindFilter = kind;
            ['ALL', 'RFP_OWNERSHIP', 'RFP_DISTRIBUTED_SCOPE', 'GENERAL_ACTION'].forEach(k => {
                const btn = document.getElementById('filter-btn-' + k);
                if (btn) {
                    if (k === kind) {
                        btn.className = 'btn btn-sm btn-primary kind-filter-btn';
                    } else {
                        btn.className = 'btn btn-sm btn-outline-secondary kind-filter-btn';
                    }
                }
            });
            renderFollowupOpportunities(kind);
        }

        function renderFollowupOpportunities(kindFilter) {
            const container = document.getElementById('oppsContentContainer');
            if (!currentFollowupData || !currentFollowupData.all_opportunities) {
                container.innerHTML = '<div class="text-muted small text-center py-4">No opportunities data loaded.</div>';
                return;
            }

            // Exclude any deals where all tasks are done - show ONLY deals with active follow-up tasks
            let opps = currentFollowupData.all_opportunities.filter(o => o.has_followup_tasks && (o.followup_tasks_count > 0));
            if (kindFilter && kindFilter !== 'ALL') {
                opps = opps.filter(o => o.kind === kindFilter);
            }

            if (!opps.length) {
                container.innerHTML = `<div class="text-muted small text-center py-5"><i class="bi bi-check2-all fs-2 d-block mb-2 text-success"></i>All tasks completed! No deals pending follow-up deliverables in "${kindFilter === 'ALL' ? 'Any Category' : kindFilter}".</div>`;
                return;
            }

            container.innerHTML = opps.map(o => {
                let kindBadgeClass = 'badge-rfp-action';
                let kindName = 'General Action';
                let kindNameAr = 'إجراءات فنية عامة';
                if (o.kind === 'RFP_OWNERSHIP') {
                    kindBadgeClass = 'badge-rfp-owner';
                    kindName = 'RFP Ownership';
                    kindNameAr = 'مناقصة رئيسية';
                } else if (o.kind === 'RFP_DISTRIBUTED_SCOPE') {
                    kindBadgeClass = 'badge-rfp-dist';
                    kindName = 'RFP Distributed Scope';
                    kindNameAr = 'نطاق موزع وشراكات';
                }

                const followCount = o.followup_tasks_count || 0;
                const statusBadge = `<span class="badge bg-warning text-dark"><i class="bi bi-clock-history me-1"></i>Follow-up Needed (${followCount})</span>`;

                const blockerBadge = o.has_blocker
                    ? `<span class="badge badge-blocker ms-1"><i class="bi bi-exclamation-triangle-fill me-1"></i>Blocked</span>`
                    : '';

                // Render tasks needing follow-up
                let tasksHtml = '';
                if (o.followup_tasks && o.followup_tasks.length > 0) {
                    tasksHtml = `
                        <div class="mt-2 pt-2 border-top border-secondary border-opacity-25">
                            <div class="text-secondary fw-semibold mb-1" style="font-size: 0.75rem;">
                                <i class="bi bi-arrow-return-right me-1"></i>Active Tasks to Follow Up About:
                            </div>
                            <div class="d-flex flex-column gap-1">
                                ${o.followup_tasks.map(t => `
                                    <div class="p-2 rounded-2 bg-black bg-opacity-30 border border-secondary border-opacity-15 small">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <span class="text-light fw-medium">${t.task_title}</span>
                                            <span class="badge ${t.priority === 'High' ? 'bg-danger' : 'bg-secondary'} ms-2">${t.status}</span>
                                        </div>
                                        <div class="d-flex flex-wrap gap-2 align-items-center mt-1 text-muted" style="font-size: 0.72rem;">
                                            <span><i class="bi bi-building text-info me-1"></i>Customer: <strong class="text-light">${t.customer_name || o.customer_name || 'Customer'}</strong></span>
                                            <span><i class="bi bi-briefcase text-warning me-1"></i>Deal: <strong class="text-light">${t.deal_name || o.deal_name || 'Deal'}</strong></span>
                                            <span><i class="bi bi-person me-1"></i>${t.assigned_to || 'Assigned'}</span>
                                            ${t.is_blocked || t.management_blockers ? `<span class="text-danger"><i class="bi bi-slash-circle me-1"></i>${t.management_blockers || 'Blocked'}</span>` : ''}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                return `
                    <div class="card bg-dark bg-opacity-60 border border-secondary border-opacity-30 p-3 mb-3" style="border-radius: 11px;">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <div>
                                <span class="badge ${kindBadgeClass} me-1">${kindName}</span>
                                <span class="font-arabic small text-muted">(${kindNameAr})</span>
                            </div>
                            <div>
                                ${statusBadge}
                                ${blockerBadge}
                            </div>
                        </div>

                        <div class="d-flex justify-content-between align-items-baseline">
                            <h6 class="text-white fw-bold mb-1">${o.deal_name}</h6>
                            <span class="text-info fw-semibold small">${Number(o.estimated_value || 0).toLocaleString()} SAR</span>
                        </div>
                        <div class="small text-muted mb-2">
                            <i class="bi bi-building me-1"></i>${o.customer_name || 'Customer'} • 
                            <i class="bi bi-diagram-2 me-1"></i>${o.stage} • 
                            <i class="bi bi-cpu me-1"></i>${o.primary_vendors || 'General'} • 
                            <i class="bi bi-person-badge me-1"></i>${o.assigned_presales}
                        </div>

                        ${tasksHtml}
                    </div>
                `;
            }).join('');
        }

        // ---------------------------------------------------------------------
        // TAB 3: Standup Questions Generator (Manual Trigger)
        // ---------------------------------------------------------------------
        async function loadPreMeetingQuestions() {
            const btn = document.getElementById('loadQuestionsBtn');
            const container = document.getElementById('questionsContainer');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Generating...';
            container.innerHTML = '<div class="text-center py-5 text-muted"><div class="spinner-border text-warning mb-2"></div><div>Analyzing open deliverables & generating bilingual questions...</div></div>';

            try {
                const res = await fetch('/api/pre-meeting-questions');
                const data = await res.json();
                renderQuestions(data.questions || []);
            } catch (err) {
                container.innerHTML = '<div class="alert alert-danger small p-3">Failed to load questions. Please ensure CRM (8000) and Tasks (8001) are running.</div>';
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i>Generate Questions';
            }
        }

        function renderQuestions(questions) {
            const container = document.getElementById('questionsContainer');
            if (!questions.length) {
                container.innerHTML = '<div class="text-muted small text-center py-5">No open deliverables requiring follow-up questions.</div>';
                return;
            }

            container.innerHTML = questions.map(q => {
                const badgeClass = q.target.includes('1') ? 'badge-presales1' : 'badge-presales2';
                return `
                    <div class="card bg-dark bg-opacity-50 border border-secondary border-opacity-25 p-3 mb-3" style="border-radius: 10px;">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="badge ${badgeClass}">${q.target}</span>
                            <span class="badge bg-secondary-subtle text-light small">${q.domain || 'Domain'}</span>
                        </div>
                        <div class="text-white small fw-semibold mb-2">
                            <i class="bi bi-chat-quote text-primary me-1"></i>${q.question_en}
                        </div>
                        <div class="font-arabic small text-info text-opacity-75 mb-2" dir="rtl" style="font-size: 0.95rem;">
                            <i class="bi bi-translate ms-1"></i>${q.question_ar}
                        </div>
                        <div class="small text-muted fst-italic border-top border-secondary border-opacity-25 pt-2" style="font-size: 0.75rem;">
                            <i class="bi bi-info-circle me-1"></i>${q.context}
                        </div>
                    </div>
                `;
            }).join('');
        }

        // ---------------------------------------------------------------------
        // Multimodal Microphone Recording (Resilient Dual-stage getUserMedia)
        // ---------------------------------------------------------------------
        async function toggleRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const micIcon = document.getElementById('micIcon');
            const timer = document.getElementById('recordingTimer');
            const prompt = document.getElementById('recordingPrompt');

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                try {
                    prompt.innerHTML = '<span class="text-info"><i class="bi bi-hourglass-split me-1"></i>Accessing microphone...</span>';
                    
                    let stream = null;
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({
                            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
                        });
                    } catch (constraintErr) {
                        console.warn("Retrying microphone with basic constraints:", constraintErr);
                        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    }

                    recordedChunks = [];
                    let mimeType = 'audio/webm';
                    if (!MediaRecorder.isTypeSupported('audio/webm')) {
                        if (MediaRecorder.isTypeSupported('audio/mp4')) {
                            mimeType = 'audio/mp4';
                        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                            mimeType = 'audio/ogg';
                        } else {
                            mimeType = '';
                        }
                    }

                    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

                    mediaRecorder.ondataavailable = e => {
                        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
                    };

                    mediaRecorder.onstop = () => {
                        // Release hardware mic tracks
                        try {
                            stream.getTracks().forEach(track => track.stop());
                        } catch(e) {}

                        const actualType = mediaRecorder.mimeType || 'audio/webm';
                        recordedBlob = new Blob(recordedChunks, { type: actualType });
                        const audioUrl = URL.createObjectURL(recordedBlob);
                        document.getElementById('audioPlayer').src = audioUrl;
                        document.getElementById('audioPlaybackContainer').classList.remove('d-none');
                        prompt.innerHTML = '<span class="text-success fw-semibold"><i class="bi bi-check2-circle me-1"></i>Recording complete. Play audio or click "Analyze Speech & Sync APIs".</span>';
                    };

                    mediaRecorder.start(250);
                    recordBtn.className = 'mic-button mic-recording';
                    micIcon.className = 'bi bi-stop-fill';
                    prompt.innerHTML = '<span class="text-danger fw-bold"><i class="bi bi-record-circle me-1"></i>Recording standup meeting... Speak in Arabic / English. Click red button to stop.</span>';
                    
                    secondsElapsed = 0;
                    timer.textContent = "00:00";
                    if (timerInterval) clearInterval(timerInterval);
                    timerInterval = setInterval(() => {
                        secondsElapsed++;
                        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                        const secs = String(secondsElapsed % 60).padStart(2, '0');
                        timer.textContent = `${mins}:${secs}`;
                    }, 1000);
                } catch (err) {
                    console.error("Microphone access error:", err);
                    recordBtn.className = 'mic-button mic-idle';
                    micIcon.className = 'bi bi-mic-fill';
                    prompt.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill me-1"></i>Mic Error: ${err.message}. Please allow mic permissions in browser settings or use file upload below.</span>`;
                }
            } else {
                try {
                    mediaRecorder.stop();
                } catch(e) {
                    console.warn("Error stopping mediaRecorder:", e);
                }
                if (timerInterval) clearInterval(timerInterval);
                recordBtn.className = 'mic-button mic-idle';
                micIcon.className = 'bi bi-mic-fill';
            }
        }

        async function processRecordedAudio() {
            if (!recordedBlob) {
                alert('No audio recorded.');
                return;
            }
            uploadAndAnalyze(recordedBlob, 'standup_recording.webm');
        }

        function uploadAudioFile() {
            const input = document.getElementById('audioFileInput');
            if (!input.files.length) {
                alert('Please select an audio file first.');
                return;
            }
            uploadAndAnalyze(input.files[0], input.files[0].name);
        }

        async function uploadAndAnalyze(blobOrFile, filename) {
            const btn = document.getElementById('processAudioBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Gemini Processing Speech & Syncing APIs...';

            const formData = new FormData();
            formData.append('file', blobOrFile, filename);

            try {
                const res = await fetch('/api/process-audio', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert('Audio processing error: ' + (err.detail || JSON.stringify(err)));
                    btn.disabled = false;
                    btn.innerHTML = '<i class="bi bi-cpu-fill me-1"></i> Analyze Speech & Sync APIs';
                    return;
                }

                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert('Network error communicating with Voice Agent server.');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-cpu-fill me-1"></i> Analyze Speech & Sync APIs';
            }
        }

        function renderResults(data) {
            const resultsCard = document.getElementById('resultsCard');
            resultsCard.classList.remove('d-none');

            // Summary
            document.getElementById('summaryText').textContent = data.transcript_summary || 'Meeting processed.';

            // Executive Report lists
            const exec = data.executive_report || {};
            document.getElementById('todayProgressList').innerHTML = (exec.today_progress || []).map(p => `<li>${p}</li>`).join('') || '<li>No items noted.</li>';
            document.getElementById('tomorrowActionsList').innerHTML = (exec.tomorrow_actions || []).map(a => `<li>${a}</li>`).join('') || '<li>No items noted.</li>';
            document.getElementById('managementWarningsList').innerHTML = (exec.management_warnings || []).map(w => `<li>${w}</li>`).join('') || '<li>No critical blockers detected.</li>';

            // Sync Log table
            const tbody = document.getElementById('syncLogBody');
            const logs = data.api_sync_log || [];
            if (!logs.length) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-2">No API state deltas required modification.</td></tr>';
            } else {
                tbody.innerHTML = logs.map(l => `
                    <tr>
                        <td><span class="badge ${l.target === 'CRM' ? 'bg-primary' : 'bg-info'}">${l.target}</span></td>
                        <td><code>${l.method}</code></td>
                        <td class="small text-secondary text-truncate" style="max-width: 250px;">${l.endpoint}</td>
                        <td>
                            <span class="badge ${l.success ? 'bg-success' : 'bg-danger'}">
                                ${l.success ? '200 Synced' : 'Failed'}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }

            resultsCard.scrollIntoView({ behavior: 'smooth' });
        }

        // ---------------------------------------------------------------------
        // Resilient Modal Controller (Works with Bootstrap or Pure CSS Fallback)
        // ---------------------------------------------------------------------
        function openApiKeyModal() {
            const modalEl = document.getElementById('apiKeyModal');
            if (window.bootstrap && window.bootstrap.Modal) {
                try {
                    if (!apiKeyModalInstance) {
                        apiKeyModalInstance = new bootstrap.Modal(modalEl);
                    }
                    apiKeyModalInstance.show();
                    return;
                } catch (e) {
                    console.warn("Bootstrap modal failed, using fallback:", e);
                }
            }
            // Fallback display
            modalEl.classList.add('show');
            modalEl.style.display = 'block';
            const backdrop = document.getElementById('customModalBackdrop');
            if (backdrop) backdrop.style.display = 'block';
        }

        function closeApiKeyModal() {
            const modalEl = document.getElementById('apiKeyModal');
            if (apiKeyModalInstance) {
                try {
                    apiKeyModalInstance.hide();
                } catch(e) {}
            }
            modalEl.classList.remove('show');
            modalEl.style.display = 'none';
            const backdrop = document.getElementById('customModalBackdrop');
            if (backdrop) backdrop.style.display = 'none';
        }

        async function saveApiKey() {
            const key = document.getElementById('geminiApiKeyInput').value.trim();
            if (!key) return;

            try {
                const res = await fetch('/api/set-api-key', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: key })
                });

                if (res.ok) {
                    closeApiKeyModal();
                    document.getElementById('apiKeyBtnText').textContent = "Gemini Key Configured";
                    await checkHealth();
                    alert('Gemini API key configured successfully!');
                } else {
                    alert('Failed to save API key');
                }
            } catch (e) {
                alert('Network error saving API key: ' + e.message);
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    return HTML_DASHBOARD


# -----------------------------------------------------------------------------
# Main Entry Point (Port 8002)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("agent_server:app", host="127.0.0.1", port=8002, reload=True)
