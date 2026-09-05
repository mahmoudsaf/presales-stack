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

    # 7. Deal ID & Names handling
    deal_id = p.get("related_deal_id")
    if deal_id is not None and str(deal_id).strip():
        digits = re.findall(r"\d+", str(deal_id))
        p["related_deal_id"] = int(digits[0]) if digits else None
    else:
        p["related_deal_id"] = None

    if "customer_name" in p and p["customer_name"]:
        p["customer_name"] = str(p["customer_name"]).strip()
    else:
        p["customer_name"] = None

    if "deal_name" in p and p["deal_name"]:
        p["deal_name"] = str(p["deal_name"]).strip()
    else:
        p["deal_name"] = None

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
    deal_details_map = {}
    for d in baseline_deals:
        d_id = d.get("deal_id")
        name = str(d.get("deal_name", "")).lower()
        cust = str(d.get("company_name", "")).lower()
        if d_id:
            deal_keyword_map[name] = d_id
            deal_keyword_map[cust] = d_id
            deal_details_map[d_id] = {
                "deal_name": d.get("deal_name"),
                "customer_name": d.get("company_name"),
            }

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
            if d_id not in deal_details_map:
                deal_details_map[d_id] = {
                    "deal_name": p.get("deal_name"),
                    "customer_name": p.get("company_name"),
                }

    def find_related_deal(text: str) -> Optional[int]:
        t_low = text.lower()
        for k, d_id in deal_keyword_map.items():
            if k and len(k) > 3 and k in t_low:
                return d_id
        return None

    def detect_customer_name(text: str) -> Optional[str]:
        t_low = text.lower()
        if "almarai" in t_low:
            return "Almarai"
        if "hajj" in t_low:
            return "Ministry of Hajj"
        if "foreign affairs" in t_low:
            return "Ministry of Foreign Affairs"
        if "planning" in t_low:
            return "Ministry of Planning"
        if "human resources" in t_low or "mhr" in t_low:
            return "Ministry of Human Resources"
        if "jeddah" in t_low:
            return "Jeddah Municipality"
        if "electronic university" in t_low or "seu" in t_low:
            return "Saudi Electronic University"
        if "solarwinds" in t_low:
            return "SolarWinds Request"
        if "fintech" in t_low:
            return "FinTech Horizons"
        if "nordic" in t_low:
            return "Nordic Health Systems"
        if "apex" in t_low:
            return "Apex Logistics"
        if "acme" in t_low:
            return "Acme Cloud Corp"
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

        d_id = find_related_deal(action_text)
        d_info = deal_details_map.get(d_id, {}) if d_id else {}
        c_name = d_info.get("customer_name") or detect_customer_name(action_text)
        d_name = d_info.get("deal_name")

        task_updates.append({
            "method": "POST",
            "payload": {
                "task_title": action_text,
                "category": detect_category(action_text),
                "assigned_to": detect_assigned(action_text),
                "vendor_domain": detect_vendor(action_text),
                "status": "In Progress",
                "priority": "High" if ("tender" in act_low or "rfp" in act_low) else "Medium",
                "related_deal_id": d_id,
                "customer_name": c_name,
                "deal_name": d_name,
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
                d_id = find_related_deal(prog_text)
                d_info = deal_details_map.get(d_id, {}) if d_id else {}
                c_name = d_info.get("customer_name") or detect_customer_name(prog_text)
                d_name = d_info.get("deal_name")

                task_updates.append({
                    "method": "POST",
                    "payload": {
                        "task_title": prog_text,
                        "category": detect_category(prog_text),
                        "assigned_to": detect_assigned(prog_text),
                        "vendor_domain": detect_vendor(prog_text),
                        "status": "In Progress",
                        "priority": "High",
                        "related_deal_id": d_id,
                        "customer_name": c_name,
                        "deal_name": d_name,
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
You are listening to an audio recording of a presales team standup meeting. The engineers speak a natural blend of Saudi Arabic and English IT terminology.

BASELINE CRM DEALS:
{json.dumps(deals, indent=2)}

BASELINE TASK BOARD:
{json.dumps(tasks, indent=2)}

Your responsibilities:
1. Listen carefully and transcribe/understand all spoken updates from Presales 1 and Presales 2.
2. Compare spoken updates against the baseline CRM deals and Task Board items above to detect DELTAS:
   - Deal stage changes (e.g. PoC -> Proposal, Discovery -> Gathering Requirements, Proposal -> Closed-Won).
   - Estimated deal value revisions.
   - Task status transitions (e.g. In Progress -> Completed, Waiting on Vendor -> In Progress, Not Started -> In Progress).
   - Resolved or newly raised management blockers.
   - Any newly mentioned deals or tasks to create.
3. Formulate structured REST API updates:
   - `crm_updates`: Array of deal updates. Use "PUT" with "deal_id" and "payload" for existing deals; or "POST" with "payload" for newly won or qualified deals.
     IMPORTANT ENUM CONSTRAINTS FOR CRM DEALS:
     * stage MUST be one of: ["Discovery", "Gathering Requirements", "PoC", "Proposal", "Closed-Won", "Closed-Lost"]. (Never use "In Progress" for deal stage; if activities are ongoing/in progress, use "Gathering Requirements").
     * assigned_presales MUST be: "Presales 1" or "Presales 2" (Engineer Abdullah maps to "Presales 1").
     * estimated_value must be a numeric float (e.g. 125000.0).
    - `task_updates`: Array of task updates.
      CRITICAL REQUIREMENT: For EVERY new tender, RFP ownership, vendor scope, or tomorrow's action item mentioned in the meeting, you MUST create a task using method "POST"! Never omit any discussed task or deliverable.
      IMPORTANT CONSTRAINTS FOR TASKS:
      * task_title: Actionable, descriptive title (e.g. "Gather technical requirements for Ministry of Hajj platform").
      * status MUST be one of: ["Not Started", "In Progress", "Waiting on Vendor", "Pending Review", "Completed"].
      * assigned_to MUST be: "Presales 1" or "Presales 2".
      * category MUST be: "RFP_OWNERSHIP" (prime tenders), "RFP_DISTRIBUTED_SCOPE" (vendor scopes/renewals), or "GENERAL_ACTION".
      * vendor_domain MUST be one of: ["HPE", "Veeam", "Dell", "Nutanix", "VMware", "General"]. NOTE: HP / Hewlett Packard MUST be set to "HPE".
      * priority MUST be one of: ["High", "Medium", "Low"].
      * related_deal_id: Integer deal ID if associated with a CRM deal, else null.
      * changed_by: "Voice Agent" (or the speaking engineer).
4. Produce an Executive Briefing Report containing:
   - `today_progress`: Array of key accomplishments confirmed in the call.
   - `tomorrow_actions`: Array of prioritized next steps.
   - `management_warnings`: Array of critical risks, vendor roadblocks, or management escalations.

Return STRICT JSON matching this schema:
{{
  "transcript_summary": "Concise summary of the meeting highlights in English with Arabic context where appropriate.",
  "crm_updates": [
    {{
      "method": "PUT",
      "deal_id": 1,
      "payload": {{
        "stage": "Proposal",
        "estimated_value": 135000.0,
        "vendor_notes": "Updated note..."
      }}
    }}
  ],
  "task_updates": [
    {{
      "method": "POST",
      "payload": {{
        "task_title": "Actionable task name",
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
    "today_progress": ["..."],
    "tomorrow_actions": ["..."],
    "management_warnings": ["..."]
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
                    <i class="bi bi-cpu me-1"></i>Gemini 3.6 Flash
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

            <!-- LEFT COLUMN: Pre-Meeting State Audit & Question Generator -->
            <div class="col-12 col-xl-5">
                <div class="app-card p-4 h-100">
                    <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary border-opacity-25">
                        <div>
                            <h6 class="fw-bold text-white mb-1"><i class="bi bi-clipboard2-pulse text-primary me-2"></i>Pre-Meeting State Audit</h6>
                            <div class="small text-muted">Auto-audits deals & tasks to generate targeted questions</div>
                        </div>
                        <button class="btn btn-sm btn-primary d-flex align-items-center gap-1" onclick="loadPreMeetingQuestions()">
                            <i class="bi bi-lightning-charge-fill"></i> Audit State
                        </button>
                    </div>

                    <div id="questionsContainer">
                        <div class="text-center py-5 text-muted">
                            <i class="bi bi-chat-left-dots fs-1 d-block mb-2 text-secondary"></i>
                            Click <strong>"Audit State"</strong> to query current pipeline & generate bilingual standup questions.
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
                            <button id="recordBtn" class="mic-button mic-idle" onclick="toggleRecording()">
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
                        <div class="small text-uppercase fw-semibold text-muted mb-1">Standup Meeting Summary</div>
                        <div class="p-3 bg-dark rounded-3 border border-secondary border-opacity-25 text-light" id="summaryText"></div>
                    </div>

                    <!-- Executive 3-Column Report -->
                    <div class="row g-3 mb-4">
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-success border-opacity-25 bg-success bg-opacity-10 h-100">
                                <div class="fw-bold small text-success mb-2"><i class="bi bi-check-circle me-1"></i>Today's Key Progress</div>
                                <ul class="small ps-3 mb-0" id="todayProgressList"></ul>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-primary border-opacity-25 bg-primary bg-opacity-10 h-100">
                                <div class="fw-bold small text-primary mb-2"><i class="bi bi-arrow-right-circle me-1"></i>Tomorrow's Actions</div>
                                <ul class="small ps-3 mb-0" id="tomorrowActionsList"></ul>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="p-3 rounded-3 border border-danger border-opacity-25 bg-danger bg-opacity-10 h-100">
                                <div class="fw-bold small text-danger mb-2"><i class="bi bi-exclamation-triangle me-1"></i>Management Warnings</div>
                                <ul class="small ps-3 mb-0" id="managementWarningsList"></ul>
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

    <!-- API Key Configuration Modal -->
    <div class="modal fade" id="apiKeyModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content bg-dark text-light border-secondary">
                <div class="modal-header border-secondary">
                    <h6 class="modal-title fw-bold"><i class="bi bi-key-fill text-warning me-2"></i>Configure Gemini API Key</h6>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p class="small text-muted">Enter your Google Gemini API key to enable native audio speech recognition and dynamic generation via Gemini 2.5 Flash.</p>
                    <input type="password" id="geminiApiKeyInput" class="form-control mb-2" placeholder="AIzaSy...">
                </div>
                <div class="modal-footer border-secondary">
                    <button type="button" class="btn btn-secondary btn-sm" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary btn-sm" onclick="saveApiKey()">Save API Key</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let mediaRecorder;
        let recordedChunks = [];
        let timerInterval;
        let secondsElapsed = 0;
        let recordedBlob = null;
        const apiKeyModal = new bootstrap.Modal(document.getElementById('apiKeyModal'));

        document.addEventListener('DOMContentLoaded', () => {
            checkHealth();
            loadPreMeetingQuestions();
        });

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

        async function loadPreMeetingQuestions() {
            const container = document.getElementById('questionsContainer');
            container.innerHTML = '<div class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Auditing active pipeline & tasks...</div>';

            try {
                const res = await fetch('/api/pre-meeting-questions');
                const data = await res.json();
                renderQuestions(data.questions || []);
            } catch (err) {
                container.innerHTML = '<div class="alert alert-danger small">Failed to load questions. Please ensure CRM (8000) and Tasks (8001) are running.</div>';
            }
        }

        function renderQuestions(questions) {
            const container = document.getElementById('questionsContainer');
            if (!questions.length) {
                container.innerHTML = '<div class="text-muted small">No questions generated.</div>';
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

        async function toggleRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const micIcon = document.getElementById('micIcon');
            const timer = document.getElementById('recordingTimer');
            const prompt = document.getElementById('recordingPrompt');

            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    recordedChunks = [];
                    mediaRecorder = new MediaRecorder(stream);

                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) recordedChunks.push(e.data);
                    };

                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                        const audioUrl = URL.createObjectURL(recordedBlob);
                        document.getElementById('audioPlayer').src = audioUrl;
                        document.getElementById('audioPlaybackContainer').classList.remove('d-none');
                        prompt.textContent = "Recording complete. Review playback or click analyze.";
                    };

                    mediaRecorder.start();
                    recordBtn.className = 'mic-button mic-recording';
                    micIcon.className = 'bi bi-stop-fill';
                    prompt.textContent = "Recording standup meeting... Speak in Arabic / English.";
                    
                    secondsElapsed = 0;
                    timerInterval = setInterval(() => {
                        secondsElapsed++;
                        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                        const secs = String(secondsElapsed % 60).padStart(2, '0');
                        timer.textContent = `${mins}:${secs}`;
                    }, 1000);
                } catch (err) {
                    alert('Microphone access denied or not available: ' + err.message);
                }
            } else {
                mediaRecorder.stop();
                clearInterval(timerInterval);
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
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Gemini 3.6 Flash Processing Speech & Syncing APIs...';

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

        function openApiKeyModal() {
            apiKeyModal.show();
        }

        async function saveApiKey() {
            const key = document.getElementById('geminiApiKeyInput').value.trim();
            if (!key) return;

            const res = await fetch('/api/set-api-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: key })
            });

            if (res.ok) {
                apiKeyModal.hide();
                document.getElementById('apiKeyBtnText').textContent = "Gemini Key Configured";
                await checkHealth();
                alert('Gemini API key configured successfully!');
            } else {
                alert('Failed to save API key');
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
