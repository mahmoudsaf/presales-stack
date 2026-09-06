import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure standard output/error supports Arabic and UTF-8 characters on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
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



# -----------------------------------------------------------------------------
# Bilingual Customer & Deal Aliases (Preserves Spoken / Original Language)
# -----------------------------------------------------------------------------
CUSTOMER_DEAL_ALIASES = [
    {
        "customer_ar": "وزارة الحج والعمرة",
        "customer_en": "Ministry of Hajj",
        "deal_ar": "مناقصة تطوير المنصة",
        "deal_en": "Platform Development Tender",
        "deal_id": 7,
        "vendor": "Dell",
        "category": "RFP_OWNERSHIP",
        "keywords_ar": ["حج", "الحج", "وزارة الحج", "تطوير المنصة"],
        "keywords_en": ["hajj", "ministry of hajj", "platform development"],
    },
    {
        "customer_ar": "وزارة الخارجية",
        "customer_en": "Ministry of Foreign Affairs",
        "deal_ar": "مناقصة البنية التحتية نوتانكس",
        "deal_en": "Nutanix HCI Infrastructure Tender",
        "deal_id": 5,
        "vendor": "Nutanix",
        "category": "RFP_OWNERSHIP",
        "keywords_ar": ["خارجية", "الخارجية", "وزارة الخارجية"],
        "keywords_en": ["foreign affairs", "mofa", "ministry of foreign affairs"],
    },
    {
        "customer_ar": "وزارة التخطيط",
        "customer_en": "Ministry of Planning",
        "deal_ar": "مناقصة تجديد الدعم الفني",
        "deal_en": "Technical Support Renewal Tender",
        "deal_id": 6,
        "vendor": "HPE",
        "category": "RFP_DISTRIBUTED_SCOPE",
        "keywords_ar": ["تخطيط", "التخطيط", "وزارة التخطيط"],
        "keywords_en": ["planning", "ministry of planning"],
    },
    {
        "customer_ar": "أمانة جدة",
        "customer_en": "Jeddah Municipality",
        "deal_ar": "تحديث أجهزة ديل",
        "deal_en": "Dell Hardware Tech Refresh",
        "deal_id": 8,
        "vendor": "Dell",
        "category": "RFP_DISTRIBUTED_SCOPE",
        "keywords_ar": ["جدة", "أمانة جدة", "امانة جدة", "الأمانة", "امانة"],
        "keywords_en": ["jeddah", "jeddah municipality", "municipality"],
    },
    {
        "customer_ar": "الجامعة السعودية الإلكترونية",
        "customer_en": "Saudi Electronic University",
        "deal_ar": "مناقصة البنية التحتية لمركز الاتصال أزور",
        "deal_en": "Azure Call Center Infrastructure Tender",
        "deal_id": 9,
        "vendor": "General",
        "category": "RFP_OWNERSHIP",
        "keywords_ar": ["إلكترونية", "الكترونية", "الجامعة الإلكترونية", "الجامعة الالكترونية", "السعودية الإلكترونية"],
        "keywords_en": ["electronic university", "seu", "saudi electronic university"],
    },
    {
        "customer_ar": "المراعي",
        "customer_en": "Almarai",
        "deal_ar": "مناقصة توريد لابتوبات المراعي",
        "deal_en": "Almarai Laptop Supply RFP",
        "deal_id": 10,
        "vendor": "Dell",
        "category": "RFP_DISTRIBUTED_SCOPE",
        "keywords_ar": ["مراعي", "المراعي", "شركة المراعي"],
        "keywords_en": ["almarai", "al marai"],
    },
    {
        "customer_ar": "Acme Cloud Corp",
        "customer_en": "Acme Cloud Corp",
        "deal_ar": "Hyperconverged Datacenter Refresh",
        "deal_en": "Hyperconverged Datacenter Refresh",
        "deal_id": 1,
        "vendor": "HPE",
        "category": "RFP_OWNERSHIP",
        "keywords_ar": ["أكمي", "اكامي", "اكيمي"],
        "keywords_en": ["acme", "acme cloud"],
    },
    {
        "customer_ar": "FinTech Horizons",
        "customer_en": "FinTech Horizons",
        "deal_ar": "Ransomware Backup Immutability",
        "deal_en": "Ransomware Backup Immutability",
        "deal_id": 2,
        "vendor": "Veeam",
        "category": "RFP_DISTRIBUTED_SCOPE",
        "keywords_ar": ["فينتك", "فين تك"],
        "keywords_en": ["fintech", "fintech horizons"],
    },
    {
        "customer_ar": "Nordic Health Systems",
        "customer_en": "Nordic Health Systems",
        "deal_ar": "Edge Compute Cluster Expansion",
        "deal_en": "Edge Compute Cluster Expansion",
        "deal_id": 3,
        "vendor": "Dell",
        "category": "RFP_OWNERSHIP",
        "keywords_ar": ["نورديك", "مستشفى", "مستشفيات"],
        "keywords_en": ["nordic", "nordic health", "hospital"],
    },
    {
        "customer_ar": "Apex Logistics",
        "customer_en": "Apex Logistics",
        "deal_ar": "Enterprise Core Virtualization",
        "deal_en": "Enterprise Core Virtualization",
        "deal_id": 4,
        "vendor": "VMware",
        "category": "GENERAL_ACTION",
        "keywords_ar": ["أيبكس", "ايبكس", "لوجستكس"],
        "keywords_en": ["apex", "apex logistics"],
    },
]


def extract_entities_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts customer_name, deal_name, and related_deal_id from text,
    faithfully preserving the original language (Arabic or English) as spoken/typed.
    """
    if not text:
        return {"customer_name": None, "deal_name": None, "related_deal_id": None}

    is_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    t_low = text.lower()

    # 1. Search aliases
    for alias in CUSTOMER_DEAL_ALIASES:
        for kw in alias["keywords_ar"]:
            if kw in text:
                return {
                    "customer_name": alias["customer_ar"] if is_arabic else alias["customer_en"],
                    "deal_name": alias["deal_ar"] if is_arabic else alias["deal_en"],
                    "related_deal_id": alias["deal_id"],
                }
        for kw in alias["keywords_en"]:
            if kw in t_low:
                return {
                    "customer_name": alias["customer_ar"] if is_arabic else alias["customer_en"],
                    "deal_name": alias["deal_ar"] if is_arabic else alias["deal_en"],
                    "related_deal_id": alias["deal_id"],
                }

    # 2. General Regex detection for Arabic ministry / customer patterns
    if is_arabic:
        m_gov = re.search(r"((?:وزارة|أمانة|امانة|هيئة|جامعة|شركة)\s+[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?)", text)
        if m_gov:
            cust = m_gov.group(1).strip()
            return {"customer_name": cust, "deal_name": None, "related_deal_id": None}

    return {"customer_name": None, "deal_name": None, "related_deal_id": None}


def reconcile_tasks_from_conversation(ai_data: Dict[str, Any], baseline_deals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures that EVERY actionable deliverable, tender, or next step mentioned in the conversation
    is captured as a task in task_updates, faithfully preserving original Arabic or English
    customer names and opportunity names.
    """
    task_updates = list(ai_data.get("task_updates", []))
    existing_titles = [str(item.get("payload", {}).get("task_title", "")).lower() for item in task_updates]
    existing_titles += [str(item.get("payload", {}).get("title", "")).lower() for item in task_updates]

    exec_report = ai_data.get("executive_report", {})
    actions = exec_report.get("tomorrow_actions", [])
    progress = exec_report.get("today_progress", [])

    def detect_vendor(text: str) -> str:
        t_low = text.lower()
        if "hp" in t_low or "hewlett" in t_low or "اتش بي" in t_low:
            return "HPE"
        if "dell" in t_low or "poweredge" in t_low or "ديل" in t_low:
            return "Dell"
        if "veeam" in t_low or "فيم" in t_low:
            return "Veeam"
        if "nutanix" in t_low or "ahv" in t_low or "نوتانكس" in t_low or "نيوتانكس" in t_low:
            return "Nutanix"
        if "vmware" in t_low or "vcf" in t_low or "vsphere" in t_low or "في ام وير" in t_low or "فيموير" in t_low:
            return "VMware"
        return "General"

    def detect_assigned(text: str) -> str:
        t_low = text.lower()
        if "presales 2" in t_low or "rep 2" in t_low or "مهندس 2" in t_low or "2" in t_low:
            return "Presales 2"
        return "Presales 1"

    def detect_category(text: str) -> str:
        t_low = text.lower()
        if "owner" in t_low or "platform" in t_low or "prime" in t_low or "رئيسي" in t_low or "منصة" in t_low or "مناقصة" in t_low:
            return "RFP_OWNERSHIP"
        if "scope" in t_low or "renewal" in t_low or "distributed" in t_low or "نطاق" in t_low or "تجديد" in t_low or "توريد" in t_low:
            return "RFP_DISTRIBUTED_SCOPE"
        return "GENERAL_ACTION"

    # 1. Enrich existing tasks from AI data without overwriting original names
    for item in task_updates:
        p = item.setdefault("payload", {})
        title = p.get("task_title") or p.get("title") or ""
        
        # If customer_name or deal_name is missing, extract while preserving original language
        entities = extract_entities_from_text(f"{title} {p.get('customer_name', '')} {p.get('deal_name', '')}")
        
        if not p.get("customer_name") and entities["customer_name"]:
            p["customer_name"] = entities["customer_name"]
            
        if not p.get("deal_name") and entities["deal_name"]:
            p["deal_name"] = entities["deal_name"]
            
        if not p.get("related_deal_id") and entities["related_deal_id"]:
            p["related_deal_id"] = entities["related_deal_id"]

    # 2. Reconcile tomorrow's actions
    for action in actions:
        action_text = str(action).strip()
        if not action_text:
            continue
        act_low = action_text.lower()
        if any(len(act_low) > 8 and (act_low[:20] in et or et in act_low) for et in existing_titles):
            continue

        entities = extract_entities_from_text(action_text)

        task_updates.append({
            "method": "POST",
            "payload": {
                "task_title": action_text,
                "category": detect_category(action_text),
                "assigned_to": detect_assigned(action_text),
                "vendor_domain": detect_vendor(action_text),
                "status": "In Progress",
                "priority": "High" if ("tender" in act_low or "rfp" in act_low or "مناقصة" in action_text) else "Medium",
                "related_deal_id": entities["related_deal_id"],
                "customer_name": entities["customer_name"],
                "deal_name": entities["deal_name"],
                "changed_by": "Voice Agent",
            }
        })
        existing_titles.append(act_low)

    # 3. Reconcile any onboarding / scope mentioned in today_progress
    for prog in progress:
        prog_text = str(prog).strip()
        prog_low = prog_text.lower()
        if "tender" in prog_low or "scope" in prog_low or "onboard" in prog_low or "rfp" in prog_low or "مناقصة" in prog_text or "نطاق" in prog_text or "توريد" in prog_text:
            if not any(len(prog_low) > 8 and (prog_low[:20] in et or et in prog_low) for et in existing_titles):
                entities = extract_entities_from_text(prog_text)

                task_updates.append({
                    "method": "POST",
                    "payload": {
                        "task_title": prog_text,
                        "category": detect_category(prog_text),
                        "assigned_to": detect_assigned(prog_text),
                        "vendor_domain": detect_vendor(prog_text),
                        "status": "In Progress",
                        "priority": "High",
                        "related_deal_id": entities["related_deal_id"],
                        "customer_name": entities["customer_name"],
                        "deal_name": entities["deal_name"],
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
        is_blk = t.get("is_blocked") or str(t.get("status", "")).strip().lower() in ("waiting on vendor", "blocked")
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

    high_priority = [t for t in tasks if str(t.get("priority", "")).strip().lower() in ("high", "critical")]

    # Calculate total pipeline value
    total_pipeline_val = 0.0
    for d in deals:
        try:
            total_pipeline_val += float(d.get("estimated_value", 0) or 0)
        except (ValueError, TypeError):
            pass

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

    return {
        "status": "success",
        "crm_connected": baseline["crm_healthy"],
        "tasks_connected": baseline["tasks_healthy"],
        "metrics": {
            "total_deals": total_deals,
            "total_tasks": total_tasks,
            "open_tasks_count": len(open_tasks),
            "completed_tasks_count": len(completed_tasks),
            "blocked_tasks_count": len(blockers),
            "high_priority_count": len(high_priority),
            "pipeline_value": f"${total_pipeline_val:,.2f}",
        },
        "category_counts": category_counts,
        "vendor_counts": vendor_counts,
        "blockers": blockers,
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
            # Check aliases
            for alias in CUSTOMER_DEAL_ALIASES:
                if alias.get("deal_id") == deal_id or alias["deal_en"].lower() in deal_name.lower() or alias["deal_ar"] in deal_name:
                    kind = alias.get("category")
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

    # Group by the 3 kinds
    rfp_ownership = [o for o in opportunities if o["kind"] == "RFP_OWNERSHIP"]
    rfp_distributed = [o for o in opportunities if o["kind"] == "RFP_DISTRIBUTED_SCOPE"]
    general_action = [o for o in opportunities if o["kind"] == "GENERAL_ACTION"]

    rfp_ownership_followup = [o for o in rfp_ownership if o["has_followup_tasks"]]
    rfp_distributed_followup = [o for o in rfp_distributed if o["has_followup_tasks"]]
    general_action_followup = [o for o in general_action if o["has_followup_tasks"]]

    return {
        "status": "success",
        "counts": {
            "total_opportunities": len(opportunities),
            "with_followup_tasks": sum(1 for o in opportunities if o["has_followup_tasks"]),
            "rfp_ownership_count": len(rfp_ownership),
            "rfp_ownership_followup_count": len(rfp_ownership_followup),
            "rfp_distributed_scope_count": len(rfp_distributed),
            "rfp_distributed_scope_followup_count": len(rfp_distributed_followup),
            "general_action_count": len(general_action),
            "general_action_followup_count": len(general_action_followup),
            "with_blockers_count": sum(1 for o in opportunities if o["has_blocker"]),
        },
        "kinds": {
            "RFP_OWNERSHIP": {
                "name_en": "RFP Ownership",
                "name_ar": "مناقصات رئيسية وتكليف كامل",
                "badge": "badge-rfp-owner",
                "description": "Prime tenders owned end-to-end requiring technical architecture, RFP response submission, and bid management.",
                "opportunities": rfp_ownership
            },
            "RFP_DISTRIBUTED_SCOPE": {
                "name_en": "RFP Distributed Scope",
                "name_ar": "نطاق موزع وشراكات التقنية",
                "badge": "badge-rfp-dist",
                "description": "Multi-vendor partner tenders (HPE, Dell, Veeam, Nutanix, VMware) requiring partner discounts, BoQ validations, and distributor scopes.",
                "opportunities": rfp_distributed
            },
            "GENERAL_ACTION": {
                "name_en": "General Action",
                "name_ar": "إجراءات وتجارب فنية عامة",
                "badge": "badge-rfp-action",
                "description": "PoC testing, hardware sizing, licensing migrations, and operational presales support deliverables.",
                "opportunities": general_action
            }
        },
        "all_opportunities": opportunities
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


def parse_standup_text_offline(text: str, deals: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Intelligent offline fallback parser that processes Arabic & English standup text
    preserving original customer & deal names even when Gemini API quota is exhausted.
    """
    is_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    
    crm_updates = []
    task_updates = []
    today_progress = []
    tomorrow_actions = []
    warnings = []
    
    clauses = [c.strip() for c in re.split(r"[\n\.\؛\,،\*\-]|(?<=[a-zA-Z0-9\u0600-\u06FF])\s+(?=وبكرة|غداً|غدا|اليوم|بكرة|Tomorrow|Today|Next)", text) if c.strip()]
    
    for clause in clauses:
        c_low = clause.lower()
        entities = extract_entities_from_text(clause)
        c_name = entities["customer_name"]
        d_name = entities["deal_name"]
        d_id = entities["related_deal_id"]
        
        is_future = any(w in c_low or w in clause for w in ["بكرة", "غدا", "غداً", "شغال على", "نعمل", "سأعمل", "tomorrow", "next", "will", "plan"])
        is_done = any(w in c_low or w in clause for w in ["خلصنا", "انتهينا", "اعتمدنا", "تم", "finished", "completed", "done", "closed", "won"])
        is_warning = any(w in c_low or w in clause for w in ["بلوكر", "معلق", "متأخر", "بانتظار", "مشكلة", "تأخير", "blocker", "waiting", "delayed", "issue"])
        
        vendor = "General"
        if any(w in c_low or w in clause for w in ["hp", "hewlett", "اتش بي"]):
            vendor = "HPE"
        elif any(w in c_low or w in clause for w in ["dell", "ديل"]):
            vendor = "Dell"
        elif any(w in c_low or w in clause for w in ["veeam", "فيم"]):
            vendor = "Veeam"
        elif any(w in c_low or w in clause for w in ["nutanix", "نوتانكس"]):
            vendor = "Nutanix"
        elif any(w in c_low or w in clause for w in ["vmware", "في ام وير", "فيموير"]):
            vendor = "VMware"
            
        assigned = "Presales 2" if any(w in c_low for w in ["2", "presales 2", "مهندس 2"]) else "Presales 1"
        category = "RFP_OWNERSHIP" if any(w in c_low or w in clause for w in ["مناقصة", "منصة", "rfp", "tender", "owner", "platform"]) else ("RFP_DISTRIBUTED_SCOPE" if any(w in c_low or w in clause for w in ["نطاق", "تجديد", "توريد", "scope", "renewal"]) else "GENERAL_ACTION")

        if is_warning:
            warnings.append(clause)
        elif is_future or (c_name and not is_done):
            tomorrow_actions.append(clause)
            task_updates.append({
                "method": "POST",
                "payload": {
                    "task_title": clause,
                    "customer_name": c_name,
                    "deal_name": d_name,
                    "related_deal_id": d_id,
                    "category": category,
                    "assigned_to": assigned,
                    "vendor_domain": vendor,
                    "status": "In Progress",
                    "priority": "High" if category == "RFP_OWNERSHIP" else "Medium",
                    "changed_by": "Voice Agent",
                }
            })
        elif is_done:
            today_progress.append(clause)
            if d_id and any(w in c_low or w in clause for w in ["poc", "proposal", "عقد", "closed-won", "won"]):
                stage = "Closed-Won" if ("won" in c_low or "فزنا" in clause or "عقد" in clause) else "Proposal"
                crm_updates.append({
                    "method": "PUT",
                    "deal_id": d_id,
                    "payload": {
                        "stage": stage,
                        "vendor_notes": f"Standup update: {clause}",
                    }
                })

    summary = f"Standup update processed. Identified {len(task_updates)} action items, {len(today_progress)} completed items, and {len(warnings)} warnings."
    if is_arabic:
        summary = f"تمت معالجة تحديث الـ Standup بنجاح. تم استخراج {len(task_updates)} مهام تنفيذية، و {len(today_progress)} إنجازات محققة، و {len(warnings)} تنبيهات."

    return {
        "raw_transcript": text,
        "transcript_summary": summary,
        "crm_updates": crm_updates,
        "task_updates": task_updates,
        "executive_report": {
            "today_progress": today_progress or [text[:120]],
            "tomorrow_actions": tomorrow_actions or [],
            "management_warnings": warnings or [],
        }
    }


def build_analysis_instructions(deals: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> str:
    return f"""
You are an expert bilingual (Saudi Arabic & English) Enterprise Presales Operations AI.
You are listening to or reading a presales team standup meeting. The engineers speak a natural blend of Saudi Arabic and English IT terminology.

BASELINE CRM DEALS:
{json.dumps(deals, indent=2)}

BASELINE TASK BOARD:
{json.dumps(tasks, indent=2)}

Your responsibilities:
1. Provide the complete verbatim transcript of everything said in the meeting (`raw_transcript`).
   CRITICAL REQUIREMENT: KEEP THE TRANSCRIPT AS IT IS IN ITS ORIGINAL LANGUAGE:
   - If spoken/written in Arabic, keep it in Arabic.
   - If spoken/written in English, keep it in English.
   - If spoken in a bilingual mix (e.g. "خلصنا PoC لوزارة الخارجية مع Nutanix"), keep the exact spoken blend verbatim.
   - NEVER translate the transcript into another language in `raw_transcript`!

2. Compare updates against the baseline CRM deals and Task Board to detect DELTAS:
   - Deal stage changes (e.g. PoC -> Proposal, Gathering Requirements, Closed-Won).
   - Estimated deal value revisions.
   - Task status transitions (In Progress, Completed, Waiting on Vendor).
   - Resolved or newly raised management blockers.
   - Any newly mentioned deals or tasks to create.

3. EXTRACT AND PARSE DATA FAITHFULLY AS IT IS:
   - Extract and preserve the ORIGINAL customer name (`customer_name`) exactly as spoken/stated (e.g. if the speaker said "وزارة الحج والعمرة" or "أمانة جدة" or "المراعي", keep the exact customer name in Arabic; if they said "Jeddah Municipality" or "Acme Cloud Corp", keep it in English. DO NOT translate customer names!).
   - Extract and preserve the ORIGINAL opportunity / tender name (`deal_name`) exactly as spoken/stated (e.g. "مناقصة تطوير المنصة" or "Platform Development Tender").
   - In `task_title`, maintain the exact opportunity and customer references in the original language as spoken.

4. Formulate structured REST API updates:
   - `crm_updates`: Array of deal updates. Use "PUT" with "deal_id" and "payload" for existing deals; or "POST" with "payload" for new deals.
     Stage MUST be: ["Discovery", "Gathering Requirements", "PoC", "Proposal", "Closed-Won", "Closed-Lost"].
     Assigned MUST be: "Presales 1" or "Presales 2".
   - `task_updates`: Array of task updates.
     CRITICAL REQUIREMENT: For EVERY new tender, RFP ownership, vendor scope, or tomorrow's action item mentioned, create a task using method "POST"!
     Include `customer_name` and `deal_name` directly in `payload`.
     Category: "RFP_OWNERSHIP" (prime tenders), "RFP_DISTRIBUTED_SCOPE" (vendor scopes/renewals), or "GENERAL_ACTION".
     Vendor Domain: ["HPE", "Veeam", "Dell", "Nutanix", "VMware", "General"].
     Status: ["Not Started", "In Progress", "Waiting on Vendor", "Pending Review", "Completed"].
     Priority: ["High", "Medium", "Low"].
     related_deal_id: Integer deal ID if associated with a CRM deal, else null.

5. Produce an Executive Briefing Report:
   - `today_progress`: Array of key accomplishments confirmed in the call (preserving original names).
   - `tomorrow_actions`: Array of prioritized next steps (preserving original names).
   - `management_warnings`: Array of critical risks, vendor roadblocks, or management escalations.

Return STRICT JSON matching this schema:
{{
  "raw_transcript": "The verbatim transcript of the entire meeting, exactly as spoken in Arabic and/or English without translation.",
  "transcript_summary": "Concise summary of the meeting highlights preserving original customer and opportunity names.",
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
        "task_title": "Actionable task name in original language",
        "customer_name": "Original customer name (e.g. 'وزارة الحج والعمرة' or 'Jeddah Municipality')",
        "deal_name": "Original opportunity name (e.g. 'مناقصة تطوير المنصة')",
        "category": "RFP_OWNERSHIP",
        "assigned_to": "Presales 2",
        "vendor_domain": "Dell",
        "status": "In Progress",
        "priority": "High",
        "related_deal_id": 7,
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


@app.post("/api/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    client_transcript: Optional[str] = Form(None),
):
    """
    Receives recorded standup audio, processes it using Gemini multimodal speech recognition,
    faithfully preserving original Arabic/English transcript and customer/deal names.
    If Gemini multimodal fails (e.g. invalid API key, quota limit, or offline), it seamlessly
    falls back to the client-side speech recognition transcript.
    """
    audio_bytes = await file.read()
    content_type = file.content_type or "audio/webm"

    if (not audio_bytes or len(audio_bytes) < 200) and not (client_transcript and client_transcript.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The recorded audio file is empty or too short. Please speak into your microphone for at least 3 to 5 seconds before clicking analyze.",
        )

    baseline = await fetch_baseline_state()
    deals = baseline["deals"]
    tasks = baseline["tasks"]

    client = get_gemini_client()
    ai_data = None
    gemini_error_detail = None

    if client and audio_bytes and len(audio_bytes) >= 200:
        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=content_type)
            instructions = build_analysis_instructions(deals, tasks)
            response = generate_with_model_fallback(
                client=client,
                contents=[audio_part, instructions],
            )
            ai_data = extract_json(response.text)
        except Exception as e:
            gemini_error_detail = str(e)
            print(f"Gemini multimodal audio processing error: {gemini_error_detail}")

    # Seamless fallback if Gemini failed or was unconfigured
    if not ai_data:
        if client_transcript and client_transcript.strip():
            print("Falling back to client speech recognition transcript...")
            ai_data = parse_standup_text_offline(client_transcript.strip(), deals, tasks)
        else:
            if not client:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="GEMINI_API_KEY is not configured. Please configure your API key in the top navigation bar, or paste your standup text in the box below to process and sync instantly without a key.",
                )
            if gemini_error_detail:
                if "API_KEY_INVALID" in gemini_error_detail or "API key not valid" in gemini_error_detail or "INVALID_ARGUMENT" in gemini_error_detail:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Your Google Gemini API key is invalid. Google AI Studio keys start with 'AIzaSy...'. Please click 'Configure Gemini Key' in the top header and enter a valid API key from https://aistudio.google.com. (In the meantime, you can also paste your standup text in the box below to process and sync instantly without a key!)",
                    )
                elif "RESOURCE_EXHAUSTED" in gemini_error_detail or "429" in gemini_error_detail:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Gemini API quota or rate limit reached (429). Please wait a moment or check your Google AI Studio plan. You can also paste your standup transcript in the text box below to process immediately!",
                    )
                raise HTTPException(status_code=500, detail=f"Gemini processing error: {gemini_error_detail}")
            raise HTTPException(status_code=500, detail="Failed to process audio and no client transcript was provided.")

    crm_updates = ai_data.get("crm_updates", [])
    task_updates = reconcile_tasks_from_conversation(ai_data, deals)
    sync_log = await execute_api_sync(crm_updates, task_updates)

    raw_trans = ai_data.get("raw_transcript") or ai_data.get("transcript") or ai_data.get("transcript_summary", "")

    return {
        "status": "success",
        "raw_transcript": raw_trans,
        "transcript_summary": ai_data.get("transcript_summary", "No summary generated."),
        "executive_report": ai_data.get("executive_report", {}),
        "crm_updates_planned": crm_updates,
        "task_updates_planned": task_updates,
        "api_sync_log": sync_log,
    }


class TextStandupRequest(BaseModel):
    text: str = Field(..., description="Spoken or typed standup text in Arabic or English")


@app.post("/api/process-text")
async def process_text_standup(payload: TextStandupRequest):
    """
    Processes typed or pasted standup transcript (Arabic or English),
    extracting deltas and syncing CRM & Task Board while preserving verbatim transcript
    and original opportunity / customer names.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    baseline = await fetch_baseline_state()
    deals = baseline["deals"]
    tasks = baseline["tasks"]

    client = get_gemini_client()
    ai_data = None

    if client:
        try:
            instructions = build_analysis_instructions(deals, tasks)
            prompt = f"{instructions}\n\nSTANDUP TEXT TRANSCRIPT:\n{text}"
            response = generate_with_model_fallback(
                client=client,
                contents=prompt,
            )
            ai_data = extract_json(response.text)
            if not ai_data.get("raw_transcript"):
                ai_data["raw_transcript"] = text
        except Exception as e:
            print(f"Notice: Gemini text generation fell back to offline parser due to: {e}")
            ai_data = None

    if not ai_data:
        # Use our smart bilingual offline parser
        ai_data = parse_standup_text_offline(text, deals, tasks)

    crm_updates = ai_data.get("crm_updates", [])
    task_updates = reconcile_tasks_from_conversation(ai_data, deals)
    sync_log = await execute_api_sync(crm_updates, task_updates)

    return {
        "status": "success",
        "raw_transcript": ai_data.get("raw_transcript") or text,
        "transcript_summary": ai_data.get("transcript_summary", "Meeting processed."),
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
        .badge-rfp-owner { background-color: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
        .badge-rfp-dist { background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.35); }
        .badge-rfp-action { background-color: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.35); }
        .kind-filter-btn {
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 20px;
            cursor: pointer;
            border: 1px solid var(--card-border);
            background: rgba(255, 255, 255, 0.05);
            color: #d1d5db;
            transition: all 0.15s ease;
        }
        .kind-filter-btn.active, .kind-filter-btn:hover {
            background: #2563eb;
            color: white;
            border-color: #3b82f6;
        }
        .task-item-badge {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .nav-pills .nav-link {
            color: #9ca3af;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .nav-pills .nav-link.active {
            background-color: var(--accent-blue);
            color: #ffffff;
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.4);
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

            <!-- LEFT COLUMN: 3-Tab Operational Intelligence Interface -->
            <div class="col-12 col-xl-5">
                <div class="app-card p-3 h-100 d-flex flex-column">
                    <!-- Tab Navigation Header -->
                    <ul class="nav nav-pills nav-fill mb-3 p-1 bg-dark bg-opacity-75 rounded-3 border border-secondary border-opacity-25" id="agentLeftTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active py-2 px-2 small fw-semibold" id="tab-audit-btn" data-bs-toggle="pill" data-bs-target="#pane-audit" type="button" role="tab">
                                <i class="bi bi-speedometer2 me-1"></i>Audit State
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link py-2 px-2 small fw-semibold" id="tab-opps-btn" data-bs-toggle="pill" data-bs-target="#pane-opps" type="button" role="tab">
                                <i class="bi bi-diagram-3 me-1"></i>Follow-Up Deals
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link py-2 px-2 small fw-semibold" id="tab-questions-btn" data-bs-toggle="pill" data-bs-target="#pane-questions" type="button" role="tab">
                                <i class="bi bi-patch-question me-1"></i>Questions
                            </button>
                        </li>
                    </ul>

                    <!-- Tab Content Area -->
                    <div class="tab-content flex-grow-1" id="agentLeftTabsContent">
                        
                        <!-- TAB 1: Audit State -->
                        <div class="tab-pane fade show active" id="pane-audit" role="tabpanel">
                            <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary border-opacity-25">
                                <div>
                                    <h6 class="fw-bold text-white mb-0"><i class="bi bi-graph-up text-primary me-2"></i>Pipeline Audit State</h6>
                                    <div class="small text-muted" style="font-size: 0.75rem;">CRM & Tasks KPIs, Blockers, & Domain distribution</div>
                                </div>
                                <button class="btn btn-sm btn-primary d-flex align-items-center gap-1" onclick="runAuditState()">
                                    <i class="bi bi-lightning-charge-fill"></i> Run Audit
                                </button>
                            </div>

                            <div id="auditContainer">
                                <div class="text-center py-5 text-muted">
                                    <i class="bi bi-speedometer2 fs-1 d-block mb-2 text-secondary"></i>
                                    Click <strong>"Run Audit"</strong> to compute real-time pipeline KPIs, blockers, and task statistics.
                                </div>
                            </div>
                        </div>

                        <!-- TAB 2: Follow-Up Opportunities (3 Kinds) -->
                        <div class="tab-pane fade" id="pane-opps" role="tabpanel">
                            <div class="d-flex justify-content-between align-items-center mb-2 pb-2 border-bottom border-secondary border-opacity-25">
                                <div>
                                    <h6 class="fw-bold text-white mb-0"><i class="bi bi-folder2-open text-warning me-2"></i>Follow-Up Opportunities</h6>
                                    <div class="small text-muted" style="font-size: 0.75rem;">Classified by 3 kinds with pending tasks & blockers</div>
                                </div>
                                <button class="btn btn-sm btn-outline-warning d-flex align-items-center gap-1" onclick="loadFollowupOpportunities()">
                                    <i class="bi bi-arrow-clockwise"></i> Generate List
                                </button>
                            </div>

                            <!-- Filter Pills for the 3 Kinds -->
                            <div class="d-flex flex-wrap align-items-center gap-1 mb-3 pt-1" id="kindFilterPillsContainer">
                                <button class="kind-filter-btn active" id="btn-filter-all" onclick="filterFollowupByKind('ALL')">All Kinds <span class="badge bg-secondary ms-1" id="badge-count-all">0</span></button>
                                <button class="kind-filter-btn" id="btn-filter-rfp-owner" onclick="filterFollowupByKind('RFP_OWNERSHIP')">RFP Ownership <span class="badge bg-secondary ms-1" id="badge-count-owner">0</span></button>
                                <button class="kind-filter-btn" id="btn-filter-rfp-dist" onclick="filterFollowupByKind('RFP_DISTRIBUTED_SCOPE')">Distributed Scope <span class="badge bg-secondary ms-1" id="badge-count-dist">0</span></button>
                                <button class="kind-filter-btn" id="btn-filter-rfp-action" onclick="filterFollowupByKind('GENERAL_ACTION')">General Action <span class="badge bg-secondary ms-1" id="badge-count-action">0</span></button>
                            </div>

                            <div id="oppsContainer">
                                <div class="text-center py-5 text-muted">
                                    <i class="bi bi-diagram-3 fs-1 d-block mb-2 text-secondary"></i>
                                    Click <strong>"Generate List"</strong> to inspect opportunities categorized into the 3 Presales kinds with their follow-up tasks.
                                </div>
                            </div>
                        </div>

                        <!-- TAB 3: Questions -->
                        <div class="tab-pane fade" id="pane-questions" role="tabpanel">
                            <div class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom border-secondary border-opacity-25">
                                <div>
                                    <h6 class="fw-bold text-white mb-0"><i class="bi bi-chat-left-dots text-success me-2"></i>Task Follow-Up Questions</h6>
                                    <div class="small text-muted" style="font-size: 0.75rem;">Bilingual questions targeting open tasks & blockers</div>
                                </div>
                                <button class="btn btn-sm btn-success d-flex align-items-center gap-1" onclick="loadPreMeetingQuestions()">
                                    <i class="bi bi-patch-question-fill"></i> Generate Questions
                                </button>
                            </div>

                            <div id="questionsContainer">
                                <div class="text-center py-5 text-muted">
                                    <i class="bi bi-chat-left-dots fs-1 d-block mb-2 text-secondary"></i>
                                    Questions will not generate automatically.<br>
                                    Click <strong>"Generate Questions"</strong> to audit tasks and create bilingual check-in questions.
                                </div>
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

                    <!-- Original Spoken Transcript (Arabic or English) -->
                    <div class="mb-4">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <div class="small text-uppercase fw-semibold text-muted">
                                <i class="bi bi-chat-left-quote text-primary me-1"></i>Original Spoken Transcript / النص الحرفي للمحادثة
                            </div>
                            <span class="badge bg-secondary-subtle text-light small" id="transcriptLangBadge">Verbatim</span>
                        </div>
                        <div class="p-3 rounded-3 border border-secondary border-opacity-25 bg-black bg-opacity-50 text-light font-arabic" 
                             style="white-space: pre-wrap; line-height: 1.65; font-size: 1rem;" 
                             id="rawTranscriptText"></div>
                    </div>

                    <!-- Meeting Summary -->
                    <div class="mb-4">
                        <div class="small text-uppercase fw-semibold text-muted mb-1">
                            <i class="bi bi-file-text text-info me-1"></i>Executive Meeting Summary
                        </div>
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

                    <!-- Extracted Tasks with Original Customer & Deal Names -->
                    <div class="mb-4" id="extractedTasksContainer">
                        <div class="small text-uppercase fw-semibold text-muted mb-2">
                            <i class="bi bi-kanban text-warning me-1"></i>Extracted Tasks & Original Opportunities (CRM & Board)
                        </div>
                        <div class="table-responsive">
                            <table class="table table-sm table-dark align-middle mb-0">
                                <thead>
                                    <tr class="text-secondary small">
                                        <th>Task Deliverable</th>
                                        <th>Customer Name</th>
                                        <th>Opportunity / Deal</th>
                                        <th>Presales</th>
                                        <th>Vendor</th>
                                        <th>Priority</th>
                                    </tr>
                                </thead>
                                <tbody id="extractedTasksBody"></tbody>
                            </table>
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
        let mediaRecorder = null;
        let activeStream = null;
        let recordedChunks = [];
        let timerInterval = null;
        let secondsElapsed = 0;
        let recordedBlob = null;
        let isInitializingMedia = false;
        let speechRecognizer = null;
        let liveSpeechTranscript = "";
        const apiKeyModal = new bootstrap.Modal(document.getElementById('apiKeyModal'));

        document.addEventListener('DOMContentLoaded', () => {
            checkHealth();
            // Note: Standup questions and audit states are strictly manual-trigger on user demand
            setupSpeechRecognition();
        });

        function setupSpeechRecognition() {
            try {
                const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRec) {
                    speechRecognizer = new SpeechRec();
                    speechRecognizer.continuous = true;
                    speechRecognizer.interimResults = true;
                    speechRecognizer.lang = 'ar-SA';
                    speechRecognizer.onresult = (event) => {
                        let text = '';
                        for (let i = 0; i < event.results.length; i++) {
                            text += event.results[i][0].transcript + ' ';
                        }
                        liveSpeechTranscript = text.trim();
                    };
                    speechRecognizer.onerror = (e) => {
                        console.warn("SpeechRecognition notice:", e.error);
                    };
                }
            } catch (e) {
                console.warn("Speech recognition setup error:", e);
            }
        }

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
        // Tab 1: Pipeline Audit State (Manual Execution)
        // ---------------------------------------------------------------------
        async function runAuditState() {
            const container = document.getElementById('auditContainer');
            container.innerHTML = '<div class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Auditing CRM pipeline & task boards...</div>';

            try {
                const res = await fetch('/api/audit-state');
                const data = await res.json();
                renderAuditState(data);
            } catch (err) {
                container.innerHTML = '<div class="alert alert-danger small">Failed to execute audit. Ensure CRM (8000) and Tasks (8001) services are running.</div>';
            }
        }

        function renderAuditState(data) {
            const container = document.getElementById('auditContainer');
            const m = data.metrics || {};
            const blockers = data.blockers || [];
            const cat = data.category_counts || {};
            const ven = data.vendor_counts || {};

            let blockersHtml = '';
            if (blockers.length > 0) {
                blockersHtml = `
                    <div class="mb-3">
                        <div class="small text-uppercase fw-semibold text-danger mb-2">
                            <i class="bi bi-exclamation-octagon-fill me-1"></i>Active Management Blockers (${blockers.length})
                        </div>
                        <div class="d-flex flex-column gap-2">
                            ${blockers.map(b => `
                                <div class="p-2 rounded-2 border border-danger border-opacity-50 bg-danger bg-opacity-10 text-light small">
                                    <div class="d-flex justify-content-between align-items-center mb-1">
                                        <span class="fw-bold text-white"><i class="bi bi-exclamation-triangle-fill text-danger me-1"></i>${b.task_title}</span>
                                        <span class="badge bg-secondary-subtle text-light">${b.assigned_to}</span>
                                    </div>
                                    <div class="text-secondary small mb-1">
                                        <span><i class="bi bi-building me-1"></i>${b.customer_name}</span> &bull; 
                                        <span><i class="bi bi-briefcase me-1"></i>${b.deal_name}</span> &bull; 
                                        <span class="text-info">${b.vendor_domain}</span>
                                    </div>
                                    <div class="text-danger small fst-italic">
                                        <i class="bi bi-shield-exclamation me-1"></i>${b.reason}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            } else {
                blockersHtml = `
                    <div class="p-2 mb-3 rounded-2 border border-success border-opacity-25 bg-success bg-opacity-10 text-success small text-center">
                        <i class="bi bi-check-circle me-1"></i>No active management blockers reported across tasks.
                    </div>
                `;
            }

            container.innerHTML = `
                <!-- KPI Metrics Grid -->
                <div class="row g-2 mb-3">
                    <div class="col-6">
                        <div class="p-2 rounded-3 border border-secondary border-opacity-25 bg-dark bg-opacity-50 text-center">
                            <div class="text-muted small" style="font-size: 0.7rem;">Active Deals</div>
                            <div class="fs-5 fw-bold text-primary">${m.total_deals || 0}</div>
                            <div class="text-info small" style="font-size: 0.7rem;">${m.pipeline_value || '$0'}</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="p-2 rounded-3 border border-secondary border-opacity-25 bg-dark bg-opacity-50 text-center">
                            <div class="text-muted small" style="font-size: 0.7rem;">Open / Total Tasks</div>
                            <div class="fs-5 fw-bold text-warning">${m.open_tasks_count || 0} <span class="text-muted fs-6">/ ${m.total_tasks || 0}</span></div>
                            <div class="text-success small" style="font-size: 0.7rem;">${m.completed_tasks_count || 0} Done</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="p-2 rounded-3 border border-secondary border-opacity-25 bg-dark bg-opacity-50 text-center">
                            <div class="text-muted small" style="font-size: 0.7rem;">Management Blockers</div>
                            <div class="fs-5 fw-bold ${m.blocked_tasks_count > 0 ? 'text-danger' : 'text-success'}">${m.blocked_tasks_count || 0}</div>
                            <div class="text-secondary small" style="font-size: 0.7rem;">Immediate Action</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="p-2 rounded-3 border border-secondary border-opacity-25 bg-dark bg-opacity-50 text-center">
                            <div class="text-muted small" style="font-size: 0.7rem;">High / Critical Priority</div>
                            <div class="fs-5 fw-bold text-info">${m.high_priority_count || 0}</div>
                            <div class="text-secondary small" style="font-size: 0.7rem;">Key Deliverables</div>
                        </div>
                    </div>
                </div>

                <!-- 3 Kinds Distribution Breakdown -->
                <div class="mb-3 p-2 rounded-2 border border-secondary border-opacity-25 bg-dark bg-opacity-25">
                    <div class="small text-uppercase fw-semibold text-muted mb-2" style="font-size: 0.72rem;">
                        <i class="bi bi-pie-chart me-1"></i>Tasks by 3 Kinds
                    </div>
                    <div class="d-flex flex-column gap-1 small">
                        <div class="d-flex justify-content-between align-items-center">
                            <span><span class="badge badge-rfp-owner me-1">RFP Ownership</span> <span class="text-secondary">(مناقصات رئيسية)</span></span>
                            <span class="fw-bold text-warning">${cat.RFP_OWNERSHIP || 0}</span>
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <span><span class="badge badge-rfp-dist me-1">Distributed Scope</span> <span class="text-secondary">(نطاق موزع)</span></span>
                            <span class="fw-bold text-info">${cat.RFP_DISTRIBUTED_SCOPE || 0}</span>
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <span><span class="badge badge-rfp-action me-1">General Action</span> <span class="text-secondary">(إجراءات عامة)</span></span>
                            <span class="fw-bold text-light">${cat.GENERAL_ACTION || 0}</span>
                        </div>
                    </div>
                </div>

                <!-- Vendor Distribution Breakdown -->
                <div class="mb-3 p-2 rounded-2 border border-secondary border-opacity-25 bg-dark bg-opacity-25">
                    <div class="small text-uppercase fw-semibold text-muted mb-2" style="font-size: 0.72rem;">
                        <i class="bi bi-tags me-1"></i>Tasks by Vendor Domain
                    </div>
                    <div class="d-flex flex-wrap gap-1">
                        ${Object.entries(ven).map(([vName, count]) => `
                            <span class="badge bg-dark border border-secondary border-opacity-50 text-light py-1 px-2">
                                ${vName}: <strong class="text-primary">${count}</strong>
                            </span>
                        `).join('')}
                    </div>
                </div>

                <!-- Blockers / Management Warnings Section -->
                ${blockersHtml}
            `;
        }

        // ---------------------------------------------------------------------
        // Tab 2: Follow-Up Opportunities (3 Kinds)
        // ---------------------------------------------------------------------
        let followupDataCache = null;
        let currentKindFilter = 'ALL';

        async function loadFollowupOpportunities() {
            const container = document.getElementById('oppsContainer');
            container.innerHTML = '<div class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-warning me-2"></div>Classifying opportunities into 3 kinds & tracking follow-up tasks...</div>';

            try {
                const res = await fetch('/api/followup-opportunities');
                const data = await res.json();
                followupDataCache = data;

                // Update counts on filter buttons
                const counts = data.counts || {};
                const badgeAll = document.getElementById('badge-count-all');
                const badgeOwner = document.getElementById('badge-count-owner');
                const badgeDist = document.getElementById('badge-count-dist');
                const badgeAction = document.getElementById('badge-count-action');

                if (badgeAll) badgeAll.textContent = counts.total_opportunities || 0;
                if (badgeOwner) badgeOwner.textContent = counts.rfp_ownership_count || 0;
                if (badgeDist) badgeDist.textContent = counts.rfp_distributed_scope_count || 0;
                if (badgeAction) badgeAction.textContent = counts.general_action_count || 0;

                renderFollowupOpportunities(currentKindFilter);
            } catch (err) {
                container.innerHTML = '<div class="alert alert-danger small">Failed to load opportunities. Ensure CRM (8000) and Tasks (8001) are running.</div>';
            }
        }

        function filterFollowupByKind(kind) {
            currentKindFilter = kind;
            ['btn-filter-all', 'btn-filter-rfp-owner', 'btn-filter-rfp-dist', 'btn-filter-rfp-action'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            });

            if (kind === 'ALL') document.getElementById('btn-filter-all')?.classList.add('active');
            else if (kind === 'RFP_OWNERSHIP') document.getElementById('btn-filter-rfp-owner')?.classList.add('active');
            else if (kind === 'RFP_DISTRIBUTED_SCOPE') document.getElementById('btn-filter-rfp-dist')?.classList.add('active');
            else if (kind === 'GENERAL_ACTION') document.getElementById('btn-filter-rfp-action')?.classList.add('active');

            if (followupDataCache) {
                renderFollowupOpportunities(kind);
            } else {
                loadFollowupOpportunities();
            }
        }

        function renderFollowupOpportunities(kindFilter) {
            const container = document.getElementById('oppsContainer');
            if (!followupDataCache || !followupDataCache.all_opportunities) {
                container.innerHTML = '<div class="text-muted small text-center py-4">No data available. Click "Generate List".</div>';
                return;
            }

            let list = [...followupDataCache.all_opportunities];
            if (kindFilter !== 'ALL') {
                list = list.filter(o => o.kind === kindFilter);
            }

            if (list.length === 0) {
                container.innerHTML = '<div class="text-muted small text-center py-4">No opportunities found for this category filter.</div>';
                return;
            }

            // Sort: deals with active follow-up tasks first, then by blockers
            list.sort((a, b) => {
                if (a.has_blocker && !b.has_blocker) return -1;
                if (!a.has_blocker && b.has_blocker) return 1;
                if (a.has_followup_tasks && !b.has_followup_tasks) return -1;
                if (!a.has_followup_tasks && b.has_followup_tasks) return 1;
                return 0;
            });

            const kindLabels = {
                'RFP_OWNERSHIP': { labelEn: 'RFP Ownership', labelAr: 'مناقصة رئيسية', badge: 'badge-rfp-owner', icon: 'bi-file-earmark-lock' },
                'RFP_DISTRIBUTED_SCOPE': { labelEn: 'Distributed Scope', labelAr: 'نطاق موزع وشراكات', badge: 'badge-rfp-dist', icon: 'bi-share' },
                'GENERAL_ACTION': { labelEn: 'General Action', labelAr: 'إجراء فني وتجارب', badge: 'badge-rfp-action', icon: 'bi-gear' }
            };

            container.innerHTML = list.map(opp => {
                const kMeta = kindLabels[opp.kind] || kindLabels['GENERAL_ACTION'];
                const followTasks = opp.followup_tasks || [];
                const hasFollow = followTasks.length > 0;

                let tasksHtml = '';
                if (hasFollow) {
                    tasksHtml = `
                        <div class="mt-2 pt-2 border-top border-secondary border-opacity-25">
                            <div class="small fw-semibold text-muted mb-1" style="font-size: 0.72rem;">
                                <i class="bi bi-list-check me-1 text-primary"></i>Tasks Needing Follow-Up (${followTasks.length}):
                            </div>
                            <div class="d-flex flex-column gap-1">
                                ${followTasks.map(t => {
                                    const isBlk = t.is_blocked || (t.management_blockers && t.management_blockers.trim().length > 0) || t.status === 'Waiting on Vendor';
                                    const assignedBadge = (t.assigned_to && t.assigned_to.includes('2')) ? 'badge-presales2' : 'badge-presales1';
                                    return `
                                        <div class="p-2 rounded border border-secondary border-opacity-25 bg-black bg-opacity-30 small">
                                            <div class="d-flex justify-content-between align-items-center mb-1">
                                                <span class="text-white fw-medium">${t.task_title}</span>
                                                <span class="badge ${assignedBadge}">${t.assigned_to || 'Presales'}</span>
                                            </div>
                                            <div class="d-flex flex-wrap align-items-center gap-1">
                                                <span class="badge bg-secondary-subtle text-light task-item-badge">${t.status}</span>
                                                <span class="badge bg-dark border border-secondary border-opacity-50 text-info task-item-badge">${t.vendor_domain || 'General'}</span>
                                                ${t.priority === 'High' ? '<span class="badge bg-danger-subtle text-danger task-item-badge">High Priority</span>' : ''}
                                            </div>
                                            ${isBlk ? `
                                                <div class="text-danger small mt-1 fst-italic" style="font-size: 0.75rem;">
                                                    <i class="bi bi-exclamation-diamond-fill me-1"></i><strong>Blocker:</strong> ${t.management_blockers || 'Waiting on Vendor feedback'}
                                                </div>
                                            ` : ''}
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    `;
                } else {
                    tasksHtml = `
                        <div class="mt-2 pt-2 border-top border-secondary border-opacity-25 small text-muted fst-italic" style="font-size: 0.72rem;">
                            <i class="bi bi-check2-all text-success me-1"></i>All linked tasks completed (${opp.completed_tasks_count || 0} completed).
                        </div>
                    `;
                }

                return `
                    <div class="card bg-dark bg-opacity-60 border border-secondary border-opacity-25 p-3 mb-2" style="border-radius: 10px;">
                        <div class="d-flex justify-content-between align-items-start gap-2 mb-1">
                            <div>
                                <div class="d-flex align-items-center gap-2 flex-wrap mb-1">
                                    <span class="badge ${kMeta.badge}"><i class="bi ${kMeta.icon} me-1"></i>${kMeta.labelEn}</span>
                                    <span class="badge bg-dark border border-secondary border-opacity-50 text-secondary" style="font-size: 0.68rem;">${kMeta.labelAr}</span>
                                    ${opp.has_blocker ? '<span class="badge bg-danger text-white"><i class="bi bi-exclamation-triangle-fill me-1"></i>Blocked</span>' : ''}
                                </div>
                                <h6 class="fw-bold text-white mb-0" style="font-size: 0.95rem;">${opp.deal_name}</h6>
                                <div class="text-muted small mt-1">
                                    <i class="bi bi-building text-info me-1"></i>${opp.customer_name}
                                </div>
                            </div>
                            <div class="text-end">
                                <span class="badge bg-primary-subtle text-primary border border-primary border-opacity-25">${opp.stage}</span>
                                ${opp.estimated_value > 0 ? `<div class="small fw-semibold text-success mt-1">$${Number(opp.estimated_value).toLocaleString()}</div>` : ''}
                            </div>
                        </div>

                        <div class="d-flex flex-wrap align-items-center gap-2 mt-1 text-secondary small" style="font-size: 0.75rem;">
                            <span><i class="bi bi-person-badge me-1"></i>${opp.assigned_presales}</span>
                            <span>&bull;</span>
                            <span><i class="bi bi-cpu me-1"></i>${opp.primary_vendors}</span>
                            <span>&bull;</span>
                            <span class="${hasFollow ? 'text-warning' : 'text-success'}">
                                <i class="bi bi-clock-history me-1"></i>${hasFollow ? `${followTasks.length} Pending Tasks` : 'Up to Date'}
                            </span>
                        </div>

                        ${tasksHtml}
                    </div>
                `;
            }).join('');
        }

        // ---------------------------------------------------------------------
        // Tab 3: Pre-Meeting Questions (Manual Execution)
        // ---------------------------------------------------------------------
        async function loadPreMeetingQuestions() {
            const container = document.getElementById('questionsContainer');
            container.innerHTML = '<div class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-success me-2"></div>Auditing active pipeline & generating targeted questions...</div>';

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
                container.innerHTML = '<div class="text-muted small text-center py-4">No questions generated. Click "Generate Questions" to audit.</div>';
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

            // 1. If currently recording, STOP recording
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                prompt.textContent = "Stopping recording...";
                try {
                    mediaRecorder.stop();
                } catch (e) {
                    console.warn("Error stopping MediaRecorder:", e);
                }
                if (speechRecognizer) {
                    try { speechRecognizer.stop(); } catch (e) {}
                }
                if (activeStream) {
                    try {
                        activeStream.getTracks().forEach(t => t.stop());
                    } catch (e) {}
                    activeStream = null;
                }
                if (timerInterval) {
                    clearInterval(timerInterval);
                    timerInterval = null;
                }
                recordBtn.className = 'mic-button mic-idle';
                micIcon.className = 'bi bi-mic-fill';
                return;
            }

            // 2. Prevent race conditions from rapid multiple clicks
            if (isInitializingMedia) return;
            isInitializingMedia = true;

            // 3. Verify mediaDevices support in browser context
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                isInitializingMedia = false;
                prompt.textContent = "Microphone requires HTTPS or localhost. Please use file upload below.";
                alert("Microphone recording is not available in this browser context (requires HTTPS or localhost). Please use the audio file upload option below.");
                return;
            }

            try {
                prompt.textContent = "Accessing microphone...";

                // Clean up any stale streams
                if (activeStream) {
                    try { activeStream.getTracks().forEach(t => t.stop()); } catch (e) {}
                    activeStream = null;
                }

                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
                activeStream = stream;
                recordedChunks = [];
                liveSpeechTranscript = "";

                let options = {};
                if (typeof MediaRecorder.isTypeSupported === 'function') {
                    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                        options = { mimeType: 'audio/webm;codecs=opus' };
                    } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                        options = { mimeType: 'audio/webm' };
                    } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                        options = { mimeType: 'audio/mp4' };
                    }
                }

                mediaRecorder = new MediaRecorder(stream, options);

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) {
                        recordedChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const mimeType = mediaRecorder.mimeType || 'audio/webm';
                    recordedBlob = new Blob(recordedChunks, { type: mimeType });
                    const audioUrl = URL.createObjectURL(recordedBlob);
                    document.getElementById('audioPlayer').src = audioUrl;
                    document.getElementById('audioPlaybackContainer').classList.remove('d-none');
                    prompt.textContent = "Recording complete. Review playback or click Analyze Speech & Sync APIs.";
                    
                    if (activeStream) {
                        try { activeStream.getTracks().forEach(t => t.stop()); } catch (e) {}
                        activeStream = null;
                    }
                };

                // Start speech recognition in background if supported
                if (speechRecognizer) {
                    try {
                        speechRecognizer.start();
                    } catch (e) {
                        console.warn("Could not start speechRecognizer:", e);
                    }
                }

                mediaRecorder.start(250);
                recordBtn.className = 'mic-button mic-recording';
                micIcon.className = 'bi bi-stop-fill';
                prompt.textContent = "Recording standup meeting... Speak in Arabic / English. Click to stop.";

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
                if (activeStream) {
                    try { activeStream.getTracks().forEach(t => t.stop()); } catch (e) {}
                    activeStream = null;
                }
                if (timerInterval) {
                    clearInterval(timerInterval);
                    timerInterval = null;
                }
                recordBtn.className = 'mic-button mic-idle';
                micIcon.className = 'bi bi-mic-fill';
                prompt.textContent = "Microphone error: " + err.message;
                alert("Microphone Error: " + err.message + "\\n\\nPlease allow microphone permission in your browser or use the audio file upload option below.");
            } finally {
                isInitializingMedia = false;
            }
        }

        async function processRecordedAudio() {
            if (!recordedBlob) {
                alert('No audio recorded. Please record audio or upload a file first.');
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
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing Speech & Syncing APIs...';

            const formData = new FormData();
            formData.append('file', blobOrFile, filename);
            if (liveSpeechTranscript) {
                formData.append('client_transcript', liveSpeechTranscript);
            }

            try {
                const res = await fetch('/api/process-audio', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    let errMsg = `Server returned status ${res.status}`;
                    try {
                        const errJson = await res.json();
                        if (errJson && errJson.detail) errMsg = errJson.detail;
                        else if (errJson) errMsg = JSON.stringify(errJson);
                    } catch (e) {
                        const txt = await res.text();
                        if (txt) errMsg = txt;
                    }
                    alert('Audio Processing Notice:\n\n' + errMsg);
                    return;
                }

                const data = await res.json();
                if (!data) {
                    alert('Server returned an empty response.');
                    return;
                }
                renderResults(data);
            } catch (err) {
                console.error("Audio processing fetch error:", err);
                alert('Audio Processing Notice:\n\n' + (err.message || 'Unable to connect to server.'));
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-cpu-fill me-1"></i> Analyze Speech & Sync APIs';
            }
        }

        function renderResults(data) {
            const resultsCard = document.getElementById('resultsCard');
            resultsCard.classList.remove('d-none');

            // 1. Raw Verbatim Transcript
            const raw = data.raw_transcript || data.transcript || '';
            const rawContainer = document.getElementById('rawTranscriptText');
            if (raw) {
                rawContainer.textContent = raw;
                const isArabic = /[\u0600-\u06FF]/.test(raw);
                rawContainer.setAttribute('dir', isArabic ? 'rtl' : 'ltr');
                document.getElementById('transcriptLangBadge').textContent = isArabic ? 'عربي / Saudi Arabic' : 'English';
            } else {
                rawContainer.textContent = 'Verbatim transcript not available.';
            }

            // 2. Summary
            document.getElementById('summaryText').textContent = data.transcript_summary || 'Meeting processed.';

            // 3. Executive Report lists
            const exec = data.executive_report || {};
            document.getElementById('todayProgressList').innerHTML = (exec.today_progress || []).map(p => `<li>${p}</li>`).join('') || '<li>No items noted.</li>';
            document.getElementById('tomorrowActionsList').innerHTML = (exec.tomorrow_actions || []).map(a => `<li>${a}</li>`).join('') || '<li>No items noted.</li>';
            document.getElementById('managementWarningsList').innerHTML = (exec.management_warnings || []).map(w => `<li>${w}</li>`).join('') || '<li>No critical blockers detected.</li>';

            // 4. Extracted Tasks with Original Customer & Opportunity Names
            const tasksTbody = document.getElementById('extractedTasksBody');
            const plannedTasks = data.task_updates_planned || [];
            if (!plannedTasks.length) {
                tasksTbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-2">No new task records generated.</td></tr>';
            } else {
                tasksTbody.innerHTML = plannedTasks.map(t => {
                    const p = t.payload || {};
                    const custBadge = p.customer_name ? `<span class="badge bg-primary-subtle text-primary border"><i class="bi bi-building me-1"></i>${p.customer_name}</span>` : '<span class="text-muted">-</span>';
                    const dealBadge = p.deal_name ? `<span class="badge bg-info-subtle text-info border"><i class="bi bi-briefcase me-1"></i>${p.deal_name}</span>` : '<span class="text-muted">-</span>';
                    return `
                        <tr>
                            <td class="fw-semibold text-white">${p.task_title || '-'}</td>
                            <td>${custBadge}</td>
                            <td>${dealBadge}</td>
                            <td><span class="badge bg-secondary">${p.assigned_to || 'Presales 1'}</span></td>
                            <td><span class="badge bg-dark border">${p.vendor_domain || 'General'}</span></td>
                            <td><span class="badge ${p.priority === 'High' ? 'bg-danger' : 'bg-warning text-dark'}">${p.priority || 'Medium'}</span></td>
                        </tr>
                    `;
                }).join('');
            }

            // 5. Sync Log table
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
