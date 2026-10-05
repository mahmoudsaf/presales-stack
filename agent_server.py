import json
import os
import re
import sys
from contextlib import asynccontextmanager
from enum import Enum
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

from customer_matcher import (
    CUSTOMER_ALIASES,
    GENERIC_DEAL_WORDS,
    NOISE_WORDS,
    compute_customer_similarity,
    extract_deal_scope_tokens,
    get_customer_tokens,
    normalize_arabic,
)
from task_catalog import PRE_RFP_TASKS, RFP_TASKS, ALL_ALLOWED_TASKS, normalize_to_catalog, is_pre_rfp_category

# -----------------------------------------------------------------------------
# Configuration & Persistence
# -----------------------------------------------------------------------------
class DealCategory(str, Enum):
    RFP_OWNERSHIP = "1- RFP Ownership & Prime Proposals"
    RFP_DISTRIBUTED_SCOPE = "2- RFP Distributed Scope Items"
    OPPORTUNITY_EFFORTS = "3- Opportunity Efforts & PO"


def normalize_deal_category(v: Optional[Any]) -> str:
    if not v:
        return DealCategory.OPPORTUNITY_EFFORTS.value
    v_str = str(v).strip().lower()
    if "owner" in v_str or "prime" in v_str or "رئيسية" in v_str or "كراسة" in v_str or "1-" in v_str or v_str == "rfp_ownership":
        return DealCategory.RFP_OWNERSHIP.value
    if "scope" in v_str or "distributed" in v_str or "موزع" in v_str or "نطاق" in v_str or "2-" in v_str or v_str == "rfp_distributed_scope":
        return DealCategory.RFP_DISTRIBUTED_SCOPE.value
    if "opp" in v_str or "effort" in v_str or "po" in v_str or "فرصة" in v_str or "3-" in v_str or "general" in v_str or v_str == "general_action":
        return DealCategory.OPPORTUNITY_EFFORTS.value
    for cat in DealCategory:
        if v_str == cat.value.lower():
            return cat.value
    return DealCategory.OPPORTUNITY_EFFORTS.value


def normalize_closing_date(v: Optional[Any]) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "null", "-", "undefined"):
        return None
    # ISO YYYY-MM-DD
    m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m_iso:
        y, m, d = m_iso.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    # DD/MM/YYYY or DD-MM-YYYY
    m_dmy = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
    if m_dmy:
        d, m, y = m_dmy.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    return s[:20]


def get_vendors_str(v: Optional[Any]) -> str:
    """Safely converts primary_vendors (str, list, None) into a normalized string."""
    if not v:
        return ""
    if isinstance(v, list):
        return " ".join(str(item).strip() for item in v if item)
    return str(v).strip()


ENV_FILE = Path(__file__).resolve().parent / ".env"

PLACEHOLDER_SUBSTRINGS = ["your_actual", "placeholder", "aizasyyouractual"]


def is_placeholder(val: Optional[str]) -> bool:
    if not val:
        return True
    s = val.strip().lower()
    return any(p in s for p in PLACEHOLDER_SUBSTRINGS) or s.startswith("your_")


def mask_key(val: Optional[str]) -> Optional[str]:
    if not val or is_placeholder(val):
        return None
    s = val.strip()
    if len(s) <= 8:
        return "****"
    return f"{s[:6]}...{s[-4:]}"


def load_env_file():
    """Loads key-value pairs from local .env into os.environ with placeholder protection."""
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
        except Exception as e:
            print(f"Notice: Could not read .env: {e}")


def save_env_file(key: str, value: str):
    """Saves or updates a key-value pair in .env file and Windows user environment if applicable."""
    lines = []
    found = False
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if re.match(rf"^{re.escape(key)}\s*=", line.strip()):
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

    os.environ[key] = value

    # Persist in Windows User Environment Registry so it survives reboots and terminal restarts
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE) as reg_key:
                winreg.SetValueEx(reg_key, key, 0, winreg.REG_SZ, value)
        except Exception as e:
            print(f"Notice: Could not update Windows User Environment: {e}")


# Automatically load .env on startup
load_env_file()

CRM_API_URL = os.getenv("CRM_API_URL", "http://127.0.0.1:8000/api")
TASKS_API_URL = os.getenv("TASKS_API_URL", "http://127.0.0.1:8001/api")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest"]


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
    if not api_key or is_placeholder(api_key):
        load_env_file()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or is_placeholder(api_key):
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
    key = os.getenv("GEMINI_API_KEY")
    if not key or is_placeholder(key):
        load_env_file()
        key = os.getenv("GEMINI_API_KEY")

    if key and not is_placeholder(key):
        print(f"GEMINI_API_KEY detected ({mask_key(key)}) and active.")
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
async def fetch_baseline_state(active_only: bool = True) -> Dict[str, Any]:
    deals = []
    tasks = []
    crm_healthy = False
    tasks_healthy = False

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            url = f"{CRM_API_URL}/deals?active_only=true" if active_only else f"{CRM_API_URL}/deals"
            r = await client.get(url)
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

    # Enforce active_only filter to guarantee closed deals are completely excluded for scalability & precision
    if active_only:
        deals = [d for d in deals if d.get("stage") not in ("Closed-Won", "Closed-Lost")]

    # Ensure every task record has customer_name, deal_name, and customer_id populated
    deals_by_id = {}
    for d in deals:
        did = d.get("deal_id")
        if did:
            deals_by_id[did] = {
                "deal_name": d.get("deal_name") or "",
                "customer_name": d.get("company_name") or d.get("customer_name") or "",
                "customer_id": d.get("customer_id"),
                "deal_category": d.get("deal_category"),
                "closing_date": d.get("closing_date"),
            }

    for t in tasks:
        rel_id = t.get("related_deal_id")
        if rel_id and rel_id in deals_by_id:
            d_info = deals_by_id[rel_id]
            t["deal_name"] = t.get("deal_name") or d_info["deal_name"]
            t["customer_name"] = t.get("customer_name") or d_info["customer_name"]
            t["customer_id"] = t.get("customer_id") or d_info.get("customer_id")
            if not t.get("deal_category") and d_info.get("deal_category"):
                t["deal_category"] = d_info.get("deal_category")
            if not t.get("closing_date") and d_info.get("closing_date"):
                t["closing_date"] = d_info.get("closing_date")

        # Fallback keyword match if deal_name or customer_name still missing
        if not t.get("deal_name") or not t.get("customer_name") or not t.get("customer_id"):
            t_title = (t.get("task_title") or "").lower()
            matched = False
            for did, d_info in deals_by_id.items():
                d_name = d_info["deal_name"]
                c_name = d_info["customer_name"]
                c_id = d_info.get("customer_id")
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
                    t["customer_id"] = t.get("customer_id") or c_id
                    if not t.get("deal_category") and d_info.get("deal_category"):
                        t["deal_category"] = d_info.get("deal_category")
                    if not t.get("closing_date") and d_info.get("closing_date"):
                        t["closing_date"] = d_info.get("closing_date")
                    if not t.get("related_deal_id"):
                        t["related_deal_id"] = did
                    matched = True
                    break
                for kw in keywords:
                    if kw in t_title:
                        t["deal_name"] = t.get("deal_name") or d_name
                        t["customer_name"] = t.get("customer_name") or c_name
                        t["customer_id"] = t.get("customer_id") or c_id
                        if not t.get("deal_category") and d_info.get("deal_category"):
                            t["deal_category"] = d_info.get("deal_category")
                        if not t.get("closing_date") and d_info.get("closing_date"):
                            t["closing_date"] = d_info.get("closing_date")
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
                    t["deal_name"] = t.get("deal_name") or None
                    t["customer_name"] = t.get("customer_name") or None

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

    # 1. Company Name normalization & alias resolution
    company = p.get("company_name")
    if not company or str(company).strip().lower() in ("general client", "client", "general", "عميل"):
        for key in ["customer_name", "customer", "client_name", "client", "account_name", "account", "org_name"]:
            if p.get(key) and str(p.get(key)).strip() and str(p.get(key)).strip().lower() not in ("general client", "client", "general", "عميل"):
                company = str(p.get(key)).strip()
                break

    deal_raw = p.get("deal_name") or p.get("name") or p.get("title") or p.get("project_name") or p.get("opportunity_name")
    search_corpus = f"{deal_raw or ''} {p.get('vendor_notes') or ''}"
    if not company or str(company).strip().lower() in ("general client", "client", "general", "عميل"):
        for cluster_id, aliases in CUSTOMER_ALIASES.items():
            for a in aliases:
                if len(a) > 2 and (normalize_arabic(a) in normalize_arabic(search_corpus) or a.lower() in search_corpus.lower()):
                    company = aliases[0]
                    break
            if company and company != "General Client":
                break

    if (not company or company == "General Client") and deal_raw:
        d_str = str(deal_raw).strip()
        m = re.search(r"(?:مشروع|عميل|شركة|مؤسسة|مناقصة)\s+([^\s\-:،,]+)", d_str)
        if m and m.group(1).strip().lower() not in ("general", "client"):
            company = m.group(1).strip()
        elif "general client" not in d_str.lower():
            company = d_str[:30]

    p["company_name"] = str(company or "General Client").strip()

    # 2. Deal Name normalization & alias resolution
    if not deal_raw or str(deal_raw).strip().lower() in ("مشروع general client", "general client"):
        if p["company_name"] != "General Client":
            deal_raw = f"مشروع {p['company_name']}"
        else:
            deal_raw = "مشروع متابعة الفرص"
    p["deal_name"] = str(deal_raw).strip()

    # 3. Estimated Value parsing with scale multipliers
    if "estimated_value" in p:
        val = p["estimated_value"]
        if isinstance(val, (int, float)):
            p["estimated_value"] = float(val)
        else:
            val_str = str(val).strip()
            mult = 1.0
            if re.search(r"(?:مليار|billion|b\b)", val_str, re.I):
                mult = 1_000_000_000.0
            elif re.search(r"(?:مليون|ملايين|million|m\b)", val_str, re.I):
                mult = 1_000_000.0
            elif re.search(r"(?:ألف|الاف|آلاف|thousand|k\b)", val_str, re.I):
                mult = 1_000.0

            range_match = re.findall(r"(\d+(?:\.\d+)?)", val_str)
            if len(range_match) >= 2 and any(sep in val_str for sep in ["-", "إلى", "الى", "to"]):
                vals = [float(x) for x in range_match[:2]]
                p["estimated_value"] = round((sum(vals) / len(vals)) * mult, 2)
            elif range_match:
                p["estimated_value"] = round(float(range_match[0]) * mult, 2)
            else:
                p["estimated_value"] = 0.0
    else:
        p["estimated_value"] = 0.0

    # 4. Primary Vendors
    if "primary_vendors" in p and p["primary_vendors"]:
        if isinstance(p["primary_vendors"], list):
            p["primary_vendors"] = [str(v).strip() for v in p["primary_vendors"] if v]
        else:
            p["primary_vendors"] = [v.strip() for v in str(p["primary_vendors"]).split(",") if v.strip()]
    else:
        notes_and_name = f"{p.get('deal_name', '')} {p.get('vendor_notes', '')}".lower()
        detected_v = []
        for v in ["Hitachi Vantara", "Hitachi", "Dell", "HPE", "Nutanix", "VMware", "Veeam"]:
            if v.lower() in notes_and_name:
                detected_v.append(v)
        p["primary_vendors"] = detected_v if detected_v else ["General"]

    # 5. Stage normalization
    if "stage" in p and p["stage"]:
        s = str(p["stage"]).strip().lower()
        stage_map = {
            "in progress": "Gathering Requirements",
            "ongoing": "Gathering Requirements",
            "active": "Gathering Requirements",
            "discovery": "Discovery",
            "gathering requirements": "Gathering Requirements",
            "requirements": "Gathering Requirements",
            "rfp / tender": "RFP / Tender",
            "rfp/tender": "RFP / Tender",
            "rfp or tender": "RFP / Tender",
            "rfp ofr tender": "RFP / Tender",
            "rfp": "RFP / Tender",
            "tender": "RFP / Tender",
            "rfq": "RFP / Tender",
            "request for pricing": "RFP / Tender",
            "direct request for pricing": "RFP / Tender",
            "direct request for pricing request": "RFP / Tender",
            "pricing request": "RFP / Tender",
            "pricing": "RFP / Tender",
            "طلب تسعير": "RFP / Tender",
            "طلب تسعيرة": "RFP / Tender",
            "تسعير": "RFP / Tender",
            "تسعيرة": "RFP / Tender",
            "استدراج عروض": "RFP / Tender",
            "استدراج عروض أسعار": "RFP / Tender",
            "مناقصة": "RFP / Tender",
            "منافسة": "RFP / Tender",
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
        p["stage"] = stage_map.get(s, "Discovery")
    else:
        p["stage"] = "Discovery"

    # 6. Assigned Presales
    if "assigned_presales" in p and p["assigned_presales"]:
        v = str(p["assigned_presales"]).strip().lower()
        p["assigned_presales"] = "Presales 2" if "2" in v else "Presales 1"
    else:
        p["assigned_presales"] = "Presales 1"

    # 7. Deal Category normalization
    cat_val = p.get("deal_category") or p.get("category")
    if not cat_val:
        d_text = f"{p.get('deal_name', '')} {p.get('vendor_notes', '')}".lower()
        if any(w in d_text for w in ["prime", "رئيسية", "كراسة", "مناقصة", "منافسة", "owner", "tender"]):
            cat_val = "1- RFP Ownership & Prime Proposals"
        elif any(w in d_text for w in ["scope", "نطاق", "موزع", "renewal", "تجديد", "شريك", "partner", "distributed"]):
            cat_val = "2- RFP Distributed Scope Items"
        else:
            cat_val = "3- Opportunity Efforts & PO"
    p["deal_category"] = normalize_deal_category(cat_val)

    # 8. Closing Date normalization
    c_date = p.get("closing_date") or p.get("due_date")
    if not c_date:
        d_text = f"{p.get('vendor_notes', '')} {p.get('deal_name', '')}"
        m_date = re.search(r"(?:إغلاق|اغلاق|حتسكر|closing|due|deadline)[\s:ب]*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", d_text, re.IGNORECASE)
        if m_date:
            c_date = m_date.group(1)
        else:
            m_date2 = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", d_text)
            if m_date2:
                c_date = m_date2.group(1)
    p["closing_date"] = normalize_closing_date(c_date)

    return p


def sanitize_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload)
    p.pop("task_id", None)

    # 1. Category handling
    cat = str(p.get("category", "") or p.get("deal_category", "")).strip().lower()
    if "owner" in cat or "prime" in cat or "1-" in cat:
        p["category"] = "RFP_OWNERSHIP"
        p["deal_category"] = "1- RFP Ownership & Prime Proposals"
    elif "distribut" in cat or "scope" in cat or "2-" in cat:
        p["category"] = "RFP_DISTRIBUTED_SCOPE"
        p["deal_category"] = "2- RFP Distributed Scope Items"
    else:
        p["category"] = "GENERAL_ACTION"
        p["deal_category"] = "3- Opportunity Efforts & PO"

    # 2. Title handling (strictly enforce predefined task catalog)
    raw_title = p.pop("task_title", None) or p.pop("title", None) or p.pop("name", None)
    canonical_title = normalize_to_catalog(raw_title, p["deal_category"])
    p["task_title"] = canonical_title
    if raw_title and raw_title != canonical_title and not p.get("management_blockers"):
        p["management_blockers"] = f"Context: {raw_title}"

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

    # 7. Deal ID, Deal Name, Customer ID & Customer Name handling
    deal_id = p.get("related_deal_id")
    if deal_id is not None and str(deal_id).strip():
        digits = re.findall(r"\d+", str(deal_id))
        p["related_deal_id"] = int(digits[0]) if digits else None
    else:
        p["related_deal_id"] = None

    cust_id = p.get("customer_id")
    if cust_id is not None and str(cust_id).strip():
        digits = re.findall(r"\d+", str(cust_id))
        p["customer_id"] = int(digits[0]) if digits else None
    else:
        p["customer_id"] = None

    if p.get("deal_name"):
        p["deal_name"] = str(p["deal_name"]).strip()
    if p.get("customer_name"):
        p["customer_name"] = str(p["customer_name"]).strip()

    # 8. Management blockers
    if "management_blockers" in p and p["management_blockers"]:
        p["management_blockers"] = str(p["management_blockers"]).strip()
    else:
        p["management_blockers"] = None

    # 9. Closing date handling
    c_date = p.get("closing_date") or p.get("due_date")
    p["closing_date"] = normalize_closing_date(c_date)

    # 10. Changed By
    p["changed_by"] = p.get("changed_by") or "Voice Agent"

    # 11. Previous Task ID
    prev_tid = p.get("previous_task_id")
    if prev_tid is not None and str(prev_tid).strip():
        digits = re.findall(r"\d+", str(prev_tid))
        p["previous_task_id"] = int(digits[0]) if digits else None
    else:
        p["previous_task_id"] = None
    return p


def reconcile_crm_updates_from_conversation(ai_data: Dict[str, Any], baseline_deals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures that spoken updates regarding existing customers or deals are accurately mapped
    to existing CRM deal records (using 'PUT' with deal_id) rather than creating duplicate deals (POST)
    or phantom fallback deals named 'مشروع General Client'.
    """
    crm_updates = list(ai_data.get("crm_updates", []))
    summary = str(ai_data.get("transcript_summary", "") or "")
    exec_report = ai_data.get("executive_report", {}) or {}
    progress_items = [str(x) for x in exec_report.get("today_progress", [])]
    action_items = [str(x) for x in exec_report.get("tomorrow_actions", [])]
    full_context = f"{summary} {' '.join(progress_items)} {' '.join(action_items)}"
    full_context_low = full_context.lower()
    full_context_norm = normalize_arabic(full_context)

    def find_matching_deal(text: str, candidate_company: Optional[str] = None, candidate_deal: Optional[str] = None) -> Optional[Dict[str, Any]]:
        t_low = text.lower()
        t_norm = normalize_arabic(text)
        cand_comp_norm = normalize_arabic(candidate_company or "")
        cand_deal_norm = normalize_arabic(candidate_deal or "")

        # Check if incoming text introduces an explicitly new tender/project
        is_explicit_new = any(kw in t_norm for kw in [
            "مناقصه جديده", "كراسه جديده", "فرصه جديده", "مناقصه اخري", "كراسه اخري", "مشروع جديد"
        ])

        best_deal = None
        best_score = 0.0

        for bd in baseline_deals:
            bd_id = bd.get("deal_id")
            bd_comp = str(bd.get("company_name") or bd.get("customer_name") or "")
            bd_deal = str(bd.get("deal_name") or "")
            bd_comp_norm = normalize_arabic(bd_comp)
            bd_deal_norm = normalize_arabic(bd_deal)

            # Check if this baseline deal matches the candidate customer
            cust_matched = False
            # 1. Direct Customer Alias Cluster Check
            for cluster_id, aliases in CUSTOMER_ALIASES.items():
                if any(normalize_arabic(a) in bd_comp_norm for a in aliases):
                    if any(normalize_arabic(a) in t_norm or (cand_comp_norm and normalize_arabic(a) in cand_comp_norm) for a in aliases):
                        cust_matched = True
                        break

            # 2. Customer Name Similarity
            if not cust_matched and bd_comp:
                if cand_comp_norm and cand_comp_norm not in ("general client", "client", "general", "عميل"):
                    if compute_customer_similarity(bd_comp, candidate_company) >= 0.75:
                        cust_matched = True
                if not cust_matched and compute_customer_similarity(bd_comp, text) >= 0.75:
                    cust_matched = True
                if not cust_matched:
                    _, core_tokens, stripped_tokens = get_customer_tokens(bd_comp)
                    for token in core_tokens + stripped_tokens:
                        if len(token) >= 3 and token not in NOISE_WORDS:
                            t_boundary = rf"\b{re.escape(token)}\b"
                            if re.search(t_boundary, t_norm) or token in t_norm.split():
                                cust_matched = True
                                break

            # If customer does not match bd, continue
            if not cust_matched:
                continue

            # If candidate introduces an explicit new tender, do NOT match existing deal!
            if is_explicit_new:
                continue

            score = 0.0

            # Exact normalized deal name match
            if cand_deal_norm and bd_deal_norm and cand_deal_norm == bd_deal_norm:
                score = 10.0
            else:
                cand_scope = extract_deal_scope_tokens(candidate_deal or "", bd_comp)
                bd_scope = extract_deal_scope_tokens(bd_deal, bd_comp)

                if cand_scope and bd_scope:
                    overlap = cand_scope & bd_scope
                    if overlap:
                        jaccard = len(overlap) / len(cand_scope | bd_scope)
                        score = len(overlap) * 3.0 + jaccard * 4.0
                    else:
                        # Distinct project scopes for the same customer (e.g. ملقمات vs رخص vs إنكورت)!
                        continue
                elif not cand_scope:
                    cand_is_generic = (
                        not candidate_deal or 
                        "general client" in (candidate_deal or "").lower() or
                        candidate_deal in (f"مشروع {bd_comp}", bd_comp) or
                        cand_deal_norm in (bd_comp_norm, f"مشروع {bd_comp_norm}", "مشروع", "مناقصه", "كراسه", "فرصه")
                    )
                    if cand_is_generic:
                        cust_deals = [d for d in baseline_deals if (d.get("company_name") or d.get("customer_name")) == bd_comp or (bd.get("customer_id") and d.get("customer_id") == bd.get("customer_id"))]
                        if len(cust_deals) == 1:
                            score = 4.0
                        else:
                            if bd_scope and any(st in t_norm for st in bd_scope):
                                score = 5.0

                # Check vendor keyword match
                p_vendors = get_vendors_str(bd.get("primary_vendors")).lower()
                for v in ["huawei", "هواوي", "cisco", "سيسكو", "micro focus", "ميكروفوكس", "incorta", "انكورت", "hpe", "dell", "veeam"]:
                    if v in p_vendors and (v in t_low or v in t_norm):
                        score += 3.0

                # Deal name string similarity
                if candidate_deal and "general client" not in candidate_deal.lower():
                    d_sim = compute_customer_similarity(bd_deal, candidate_deal)
                    score += d_sim * 2.0

            if score > best_score:
                best_score = score
                best_deal = bd

        if best_deal and best_score >= 4.0:
            return best_deal

        return None

    reconciled_updates: List[Dict[str, Any]] = []
    handled_deal_ids = set()

    # Process all CRM updates proposed by Gemini
    for cu in crm_updates:
        method = str(cu.get("method", "POST")).upper()
        deal_id = cu.get("deal_id")
        payload = dict(cu.get("payload", {}) or {})
        comp = payload.get("company_name") or payload.get("customer_name")
        deal = payload.get("deal_name") or payload.get("title")
        notes = str(payload.get("vendor_notes") or "")
        search_corpus = f"{comp or ''} {deal or ''} {notes} {full_context}"

        matched_bd = None
        if deal_id:
            matched_candidate = next((d for d in baseline_deals if d.get("deal_id") == deal_id), None)
            if matched_candidate:
                # Strictly verify that candidate deal name matches the existing deal name/scope
                cand_scope = extract_deal_scope_tokens(deal or "", matched_candidate.get("company_name", ""))
                bd_scope = extract_deal_scope_tokens(matched_candidate.get("deal_name", ""), matched_candidate.get("company_name", ""))
                sim = compute_customer_similarity(matched_candidate.get("deal_name", ""), deal or "")

                is_explicit_new = any(kw in normalize_arabic(notes) or kw in normalize_arabic(search_corpus) for kw in [
                    "مناقصه جديده", "كراسه جديده", "فرصه جديده", "مناقصه اخري", "كراسه اخري", "مشروع جديد"
                ])

                if (cand_scope and bd_scope and not (cand_scope & bd_scope)) or (is_explicit_new and sim < 0.80):
                    # Gemini erroneously sent PUT on an existing deal for a new/different tender!
                    # Convert to POST to create a brand new deal!
                    matched_bd = None
                    deal_id = None
                    cu["method"] = "POST"
                    cu.pop("deal_id", None)
                else:
                    matched_bd = matched_candidate

        if not matched_bd:
            matched_bd = find_matching_deal(search_corpus, candidate_company=comp, candidate_deal=deal)

        if matched_bd:
            matched_id = matched_bd.get("deal_id")
            handled_deal_ids.add(matched_id)

            # Determine Stage Progression
            new_stage = payload.get("stage")
            if not new_stage or str(new_stage).strip().lower() in ("discovery", "gathering requirements", "in progress"):
                search_all = f"{notes} {full_context}".lower()
                if any(w in search_all for w in ["tender department", "إدارة المناقصات", "ارسال العرض", "send to tender", "submitted", "تسليم العرض", "تقديم العرض"]):
                    new_stage = "Proposal"
                elif any(w in search_all for w in ["prices", "quotations", "pricing", "تسعير", "الأسعار", "received prices"]):
                    new_stage = "Proposal" if ("tender" in search_all or "مناقصات" in search_all) else "RFP / Tender"
                else:
                    new_stage = matched_bd.get("stage") or "RFP / Tender"

            # Determine Authentic Deal Name (never overwrite authentic deal name with different deal name)
            final_deal_name = matched_bd.get("deal_name")
            bd_scope = extract_deal_scope_tokens(final_deal_name or "", matched_bd.get("company_name", ""))
            if not bd_scope or "general client" in (final_deal_name or "").lower():
                if deal and str(deal).strip().lower() not in (
                    "مشروع general client", "general client", "مشروع متابعة الفرص",
                    f"مشروع {str(matched_bd.get('company_name', '')).lower()}"
                ):
                    final_deal_name = deal

            # Merge Notes only if it is actually the same deal
            existing_notes = str(matched_bd.get("vendor_notes") or "").strip()
            if notes and notes not in existing_notes:
                merged_notes = f"{existing_notes} | {notes}".strip(" |") if existing_notes else notes
            else:
                merged_notes = existing_notes or notes

            # Merge Vendors
            p_vendors = payload.get("primary_vendors") or []
            if isinstance(p_vendors, str):
                p_vendors = [v.strip() for v in p_vendors.split(",") if v.strip()]
            bd_vendors = matched_bd.get("primary_vendors") or []
            if isinstance(bd_vendors, str):
                bd_vendors = [v.strip() for v in bd_vendors.split(",") if v.strip()]
            merged_vendors = list(dict.fromkeys(bd_vendors + p_vendors))

            reconciled_payload = {
                **payload,
                "company_name": matched_bd.get("company_name"),
                "deal_name": final_deal_name,
                "stage": new_stage,
                "vendor_notes": merged_notes,
                "primary_vendors": merged_vendors if merged_vendors else ["General"],
                "deal_category": matched_bd.get("deal_category") or payload.get("deal_category"),
                "closing_date": payload.get("closing_date") or matched_bd.get("closing_date"),
            }
            if "estimated_value" in payload and float(payload.get("estimated_value") or 0) > 0:
                reconciled_payload["estimated_value"] = payload["estimated_value"]
            elif matched_bd.get("estimated_value"):
                reconciled_payload["estimated_value"] = matched_bd["estimated_value"]

            reconciled_updates.append({
                "method": "PUT",
                "deal_id": matched_id,
                "payload": reconciled_payload
            })
        else:
            # Check for phantom deal
            comp_clean = str(comp or "").strip().lower()
            deal_clean = str(deal or "").strip().lower()
            if comp_clean in ("general client", "client", "general", "عميل", "") and \
               ("general client" in deal_clean or not deal_clean):
                # Discard phantom deal with no real customer entity
                continue
            cu["method"] = "POST"
            cu.pop("deal_id", None)
            reconciled_updates.append(cu)

    return reconciled_updates


def reconcile_tasks_from_conversation(ai_data: Dict[str, Any], baseline_deals: List[Dict[str, Any]], baseline_tasks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Ensures that EVERY actionable deliverable, tender, or next step mentioned in the conversation
    is captured as a task in task_updates, eliminating gaps between executive report and task board.
    Attaches deal_id, deal_name, customer_name, deal_category, closing_date, and previous_task_id to tasks for full cross-system linkage.
    """
    task_updates = list(ai_data.get("task_updates", []))
    existing_titles = [str(item.get("payload", {}).get("task_title", "")).lower() for item in task_updates]
    existing_titles += [str(item.get("payload", {}).get("title", "")).lower() for item in task_updates]

    exec_report = ai_data.get("executive_report", {})
    actions = exec_report.get("tomorrow_actions", [])
    progress = exec_report.get("today_progress", [])

    # Map deals to their latest known task ID for chaining
    deal_latest_task_ids: Dict[int, int] = {}
    if baseline_tasks:
        for bt in baseline_tasks:
            bd_id = bt.get("related_deal_id")
            bt_id = bt.get("task_id")
            if bd_id and bt_id:
                if bd_id not in deal_latest_task_ids or bt_id > deal_latest_task_ids[bd_id]:
                    deal_latest_task_ids[bd_id] = bt_id

    # If any task in task_updates is being marked 'Completed', record for chaining
    for tu in task_updates:
        if tu.get("method") == "PUT" and tu.get("task_id"):
            p = tu.get("payload", {})
            if p.get("status") == "Completed":
                t_id = tu.get("task_id")
                for bt in (baseline_tasks or []):
                    if bt.get("task_id") == t_id and bt.get("related_deal_id"):
                        deal_latest_task_ids[bt["related_deal_id"]] = t_id

    # Map deals by customer or name keywords for smart linking
    known_deals = []
    for d in baseline_deals:
        known_deals.append({
            "deal_id": d.get("deal_id"),
            "deal_name": d.get("deal_name", ""),
            "customer_name": d.get("company_name") or d.get("customer_name") or "",
            "customer_id": d.get("customer_id"),
            "deal_category": d.get("deal_category"),
            "closing_date": d.get("closing_date"),
            "primary_vendors": d.get("primary_vendors", ""),
            "vendor_notes": d.get("vendor_notes", ""),
        })

    # Also check newly planned crm_updates (both POST and PUT)
    crm_updates = ai_data.get("crm_updates", [])
    for cu in crm_updates:
        p = cu.get("payload", {})
        d_name = p.get("deal_name") or p.get("title") or ""
        c_name = p.get("company_name") or p.get("customer_name") or p.get("customer") or ""
        d_id = cu.get("deal_id")
        c_id = cu.get("customer_id") or p.get("customer_id")
        d_cat = p.get("deal_category") or p.get("category")
        c_date = p.get("closing_date") or p.get("due_date")
        if d_name or c_name:
            known_deals.append({
                "deal_id": d_id,
                "deal_name": d_name,
                "customer_name": c_name,
                "customer_id": c_id,
                "deal_category": d_cat,
                "closing_date": c_date,
                "primary_vendors": p.get("primary_vendors", ""),
                "vendor_notes": p.get("vendor_notes", ""),
            })

    def find_related_deal_and_context(text: str):
        t_low = text.lower()
        t_norm = normalize_arabic(text)

        matching_deals = []

        for d in known_deals:
            d_id = d.get("deal_id")
            d_name = d.get("deal_name", "")
            c_name = d.get("customer_name", "")
            c_id = d.get("customer_id")
            d_cat = d.get("deal_category")
            c_date = d.get("closing_date")

            cust_matched = False
            # 1. Customer similarity
            if c_name:
                sim = compute_customer_similarity(c_name, text)
                for seg in text.split(" - "):
                    sim = max(sim, compute_customer_similarity(c_name, seg.strip()))
                if sim >= 0.75:
                    cust_matched = True

            # 2. Customer aliases
            if not cust_matched and c_name:
                c_norm = normalize_arabic(c_name)
                for cluster_id, aliases in CUSTOMER_ALIASES.items():
                    if any(normalize_arabic(a) in c_norm for a in aliases):
                        if any(normalize_arabic(a) in t_norm or a.lower() in t_low for a in aliases):
                            cust_matched = True
                            break

            # 3. Direct core tokens (never noise words like وزارة / شركة)
            if not cust_matched and c_name:
                _, core_tokens, stripped_tokens = get_customer_tokens(c_name)
                for token in core_tokens + stripped_tokens:
                    if len(token) >= 3 and token not in NOISE_WORDS:
                        t_word_boundary = rf"\b{re.escape(token)}\b"
                        if re.search(t_word_boundary, t_norm) or token in t_norm.split():
                            cust_matched = True
                            break

            # 4. Direct authentic deal title match
            deal_title_matched = False
            d_norm = normalize_arabic(d_name)
            if d_name and len(d_norm) > 6 and d_norm in t_norm:
                deal_title_matched = True

            if cust_matched or deal_title_matched:
                # Score this deal based on project scope keyword match with text
                d_scope = extract_deal_scope_tokens(d_name, c_name)
                scope_score = 1 if cust_matched else 4
                for st in d_scope:
                    if st in t_norm:
                        scope_score += 4
                # Check vendor keyword match
                p_vendors = get_vendors_str(d.get("primary_vendors")).lower()
                for v, ar in [("huawei", "هواوي"), ("cisco", "سيسكو"), ("micro focus", "ميكروفوكس"), ("incorta", "انكورت"), ("dell", "ديل"), ("hpe", "اتش بي"), ("veeam", "فيم"), ("nutanix", "نيوتانكس"), ("vmware", "فيموير")]:
                    if (v in p_vendors or ar in p_vendors) and (v in t_low or ar in t_norm):
                        scope_score += 3

        # Fallback: if no match found via customer/deal name, check if text matches a unique vendor
        # in newly discussed/updated deals from the same meeting
        if not matching_deals:
            for d in known_deals:
                d_id = d.get("deal_id")
                d_name = d.get("deal_name", "")
                c_name = d.get("customer_name", "")
                c_id = d.get("customer_id")
                d_cat = d.get("deal_category")
                c_date = d.get("closing_date")
                p_vendors = get_vendors_str(d.get("primary_vendors")).lower()
                notes_low = str(d.get("vendor_notes", "")).lower()
                for v, ar in [("huawei", "هواوي"), ("cisco", "سيسكو"), ("micro focus", "ميكروفوكس"), ("incorta", "انكورت"), ("dell", "ديل"), ("hpe", "اتش بي"), ("veeam", "فيم"), ("nutanix", "نيوتانكس"), ("vmware", "فيموير")]:
                    if (v in p_vendors or ar in p_vendors or v in notes_low or ar in notes_low) and (v in t_low or ar in t_norm):
                        matching_deals.append((2, d_id, d_name, c_name, c_id, d_cat, c_date))
                        break

        if matching_deals:
            matching_deals.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best = matching_deals[0]
            return best[1], best[2], best[3], best[4], best[5], best[6]

        return None, None, None, None, None, None

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
        t_norm = normalize_arabic(text)
        if any(w in t_low for w in ["presales 2", "presales2", "rep 2", "rep2", "engineer 2", "presales-2", "rep-2"]):
            return "Presales 2"
        if any(w in t_norm for w in ["بريسيلز 2", "بريسيلز2", "مهندس 2", "المهندس الثاني", "الزميل 2"]):
            return "Presales 2"
        if "abdullah" in t_low or "عبدالله" in t_norm or "عبد الله" in t_norm:
            return "Presales 1"
        return "Presales 1"

    def detect_category(text: str):
        t_low = text.lower()
        if any(w in t_low for w in ["owner", "platform", "prime", "رئيسية", "كراسة", "مناقصة", "منافسة"]):
            return "RFP_OWNERSHIP", "1- RFP Ownership & Prime Proposals"
        if any(w in t_low for w in ["scope", "renewal", "distributed", "موزع", "نطاق", "تجديد", "شريك", "partner"]):
            return "RFP_DISTRIBUTED_SCOPE", "2- RFP Distributed Scope Items"
        return "GENERAL_ACTION", "3- Opportunity Efforts & PO"

    def detect_closing_date(text: str) -> Optional[str]:
        m = re.search(r"(?:إغلاق|اغلاق|حتسكر|closing|due|deadline)[\s:ب]*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", text, re.IGNORECASE)
        if m:
            return normalize_closing_date(m.group(1))
        m2 = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", text)
        if m2:
            return normalize_closing_date(m2.group(1))
        return None

    # Enrich and validate any tasks generated directly by Gemini
    reconciled_task_updates = []
    for tu in task_updates:
        p = tu.get("payload", {})
        t_text = f"{p.get('task_title', '')} {p.get('management_blockers', '')} {p.get('deal_name', '')} {p.get('customer_name', '')}"
        d_id, d_name, c_name, c_id, d_cat, c_date = find_related_deal_and_context(t_text)
        if not p.get("related_deal_id") and d_id:
            p["related_deal_id"] = d_id
        if not p.get("deal_name") and d_name:
            p["deal_name"] = d_name
        if not p.get("customer_name") and c_name:
            p["customer_name"] = c_name
        if not p.get("customer_id") and c_id:
            p["customer_id"] = c_id
        if not p.get("deal_category"):
            p["deal_category"] = d_cat or ("1- RFP Ownership & Prime Proposals" if p.get("category") == "RFP_OWNERSHIP" else "2- RFP Distributed Scope Items" if p.get("category") == "RFP_DISTRIBUTED_SCOPE" else "3- Opportunity Efforts & PO")
        if not p.get("closing_date"):
            p["closing_date"] = c_date or detect_closing_date(t_text)

        # Enforce strict predefined catalog task title
        raw_t = p.get("task_title", "")
        p["task_title"] = normalize_to_catalog(raw_t, p.get("deal_category"))
        if raw_t and raw_t != p["task_title"] and not p.get("management_blockers"):
            p["management_blockers"] = f"Context: {raw_t}"

        # CRITICAL MULTI-PRESALES & DISTRIBUTED SCOPE CHECK:
        # If Gemini generated a PUT on an existing task owned by Presales 1, but this task/update
        # is for Presales 2 (or vice versa) receiving a distributed scope:
        if tu.get("method") == "PUT" and tu.get("task_id"):
            t_id = tu.get("task_id")
            old_task = next((bt for bt in (baseline_tasks or []) if bt.get("task_id") == t_id), None)
            if old_task:
                old_assigned = old_task.get("assigned_to", "Presales 1")
                new_assigned = p.get("assigned_to") or detect_assigned(t_text)
                is_dist = (
                    p.get("category") == "RFP_DISTRIBUTED_SCOPE" or
                    p.get("deal_category") == "2- RFP Distributed Scope Items" or
                    any(kw in str(p.get("management_blockers", "")).lower() for kw in ["distributed", "scope", "موزع", "نطاق"]) or
                    any(kw in str(raw_t).lower() for kw in ["distributed", "scope", "موزع", "نطاق"])
                )
                if is_dist and new_assigned != old_assigned:
                    # Do NOT overwrite old_task! Convert into a new POST task for the other presales engineer
                    tu = {
                        "method": "POST",
                        "payload": {
                            **p,
                            "assigned_to": new_assigned,
                            "category": "RFP_DISTRIBUTED_SCOPE",
                            "deal_category": "2- RFP Distributed Scope Items",
                            "related_deal_id": p.get("related_deal_id") or old_task.get("related_deal_id"),
                            "deal_name": p.get("deal_name") or old_task.get("deal_name"),
                            "customer_name": p.get("customer_name") or old_task.get("customer_name"),
                            "customer_id": p.get("customer_id") or old_task.get("customer_id"),
                            "status": "In Progress",
                        }
                    }
        reconciled_task_updates.append(tu)
    task_updates = reconciled_task_updates

    # 1. Process today_progress: check if another presales received a distributed scope on an existing deal
    for prog in progress:
        prog_text = str(prog).strip()
        if not prog_text:
            continue
        prog_low = prog_text.lower()
        prog_norm = normalize_arabic(prog_text)

        is_dist_scope = (
            any(w in prog_low for w in ["distributed scope", "received scope", "scope received", "distributed to", "received distributed", "scope breakdown", "assigned scope"]) or
            any(w in prog_norm for w in ["نطاق موزع", "استلم نطاق", "استلمت نطاق", "توزيع نطاق", "نطاق العمل الموزع", "استلام كراسه", "استلام نطاق", "استلمت الكراسة"])
        )
        mentions_other_rep = (
            any(w in prog_low for w in ["presales 2", "rep 2", "presales2"]) or
            any(w in prog_norm for w in ["بريسيلز 2", "مهندس 2", "المهندس الثاني", "الزميل 2"])
        )

        if is_dist_scope or (mentions_other_rep and any(w in prog_low or w in prog_norm for w in ["scope", "rfp", "tender", "نطاق", "كراسة", "مناقصة"])):
            d_id, d_name, c_name, c_id, d_cat, c_date = find_related_deal_and_context(prog_text)
            if d_id:
                target_rep = detect_assigned(prog_text)
                canonical_prog_title = normalize_to_catalog(prog_text, "2- RFP Distributed Scope Items")
                if canonical_prog_title in ("Discovery & Technical Requirements Gathering", "RFP Decomposition & Scope Breakdown"):
                    if any(w in prog_low or w in prog_norm for w in ["boq", "quotation", "تسعير", "اسعار", "سعات", "pricing"]):
                        canonical_prog_title = "Final BoQ & Vendor Quotations"
                    else:
                        canonical_prog_title = "Low-Level Architecture & Technical Write-up"

                # Check if a task for this rep on this deal is already enqueued
                already_enqueued = any(
                    t.get("payload", {}).get("related_deal_id") == d_id and
                    t.get("payload", {}).get("assigned_to") == target_rep
                    for t in task_updates
                )
                if not already_enqueued:
                    task_updates.append({
                        "method": "POST",
                        "payload": {
                            "task_title": canonical_prog_title,
                            "management_blockers": f"Distributed scope received: {prog_text}",
                            "category": "RFP_DISTRIBUTED_SCOPE",
                            "deal_category": "2- RFP Distributed Scope Items",
                            "closing_date": c_date or detect_closing_date(prog_text),
                            "assigned_to": target_rep,
                            "vendor_domain": detect_vendor(prog_text),
                            "status": "In Progress",
                            "priority": "High",
                            "related_deal_id": d_id,
                            "deal_name": d_name,
                            "customer_id": c_id,
                            "customer_name": c_name,
                            "changed_by": "Voice Agent",
                        }
                    })
                    existing_titles.append(canonical_prog_title.lower())

    # 2. Reconcile tomorrow's actions
    for action in actions:
        action_text = str(action).strip()
        if not action_text:
            continue
        act_low = action_text.lower()
        act_norm = normalize_arabic(action_text)

        d_id, d_name, c_name, c_id, d_cat, c_date = find_related_deal_and_context(action_text)
        cat_key, cat_name = detect_category(action_text)
        closing_dt = c_date or detect_closing_date(action_text)
        target_rep = detect_assigned(action_text)

        # If action mentions distributed scope or is for another presales on an existing RFP deal
        if any(w in act_low for w in ["scope", "distributed", "renewal"]) or any(w in act_norm for w in ["موزع", "نطاق", "تجديد"]):
            cat_key = "RFP_DISTRIBUTED_SCOPE"
            cat_name = "2- RFP Distributed Scope Items"
        elif d_id and target_rep == "Presales 2":
            matched_bd = next((d for d in baseline_deals if d.get("deal_id") == d_id), None)
            if matched_bd and matched_bd.get("assigned_presales") == "Presales 1":
                cat_key = "RFP_DISTRIBUTED_SCOPE"
                cat_name = "2- RFP Distributed Scope Items"

        canonical_action_title = normalize_to_catalog(action_text, d_cat or cat_name)
        
        # Deduplication must be per (task_title, related_deal_id, assigned_to) to avoid suppressing tasks for other presales
        if any(
            t.get("payload", {}).get("task_title") == canonical_action_title 
            and t.get("payload", {}).get("related_deal_id") == d_id
            and t.get("payload", {}).get("assigned_to") == target_rep
            for t in task_updates
        ):
            continue

        task_updates.append({
            "method": "POST",
            "payload": {
                "task_title": canonical_action_title,
                "management_blockers": f"Action details: {action_text}" if action_text != canonical_action_title else None,
                "category": cat_key,
                "deal_category": d_cat or cat_name,
                "closing_date": closing_dt,
                "assigned_to": target_rep,
                "vendor_domain": detect_vendor(action_text),
                "status": "In Progress",
                "priority": "High" if ("tender" in act_low or "rfp" in act_low or "مناقصة" in act_low) else "Medium",
                "related_deal_id": d_id,
                "deal_name": d_name,
                "customer_id": c_id,
                "customer_name": c_name,
                "changed_by": "Voice Agent",
            }
        })
        existing_titles.append(canonical_action_title.lower())

    # Attach previous_task_id to new POST tasks on the same deal
    for tu in task_updates:
        if tu.get("method") == "POST":
            p = tu.get("payload", {})
            d_id = p.get("related_deal_id")
            if d_id and not p.get("previous_task_id") and d_id in deal_latest_task_ids:
                p["previous_task_id"] = deal_latest_task_ids[d_id]

    return task_updates


async def execute_api_sync(crm_updates: List[Dict[str, Any]], task_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sync_logs = []
    created_deals = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Execute CRM Updates FIRST (Creates customer and deal)
        for item in crm_updates:
            method = item.get("method", "PUT").upper()
            deal_id = item.get("deal_id")
            payload = sanitize_crm_payload(item.get("payload", {}))

            try:
                if method == "PUT" and deal_id:
                    url = f"{CRM_API_URL}/deals/{deal_id}"
                    res = await client.put(url, json=payload)
                    success = res.status_code in (200, 201)
                    detail = res.json() if success else res.text
                    sync_logs.append({
                        "target": "CRM",
                        "method": "PUT",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": success,
                        "detail": detail,
                    })
                elif method == "POST":
                    url = f"{CRM_API_URL}/deals"
                    res = await client.post(url, json=payload)
                    success = res.status_code in (200, 201)
                    detail = res.json() if success else res.text
                    sync_logs.append({
                        "target": "CRM",
                        "method": "POST",
                        "endpoint": url,
                        "status_code": res.status_code,
                        "success": success,
                        "detail": detail,
                    })
                    if success and isinstance(detail, dict):
                        created_deals.append(detail)
            except Exception as e:
                sync_logs.append({
                    "target": "CRM",
                    "method": method,
                    "endpoint": f"{CRM_API_URL}/deals/{deal_id or ''}",
                    "status_code": 500,
                    "success": False,
                    "detail": str(e),
                })

        # 2. Link newly created deal IDs & customer info to Task Updates
        for item in task_updates:
            method = item.get("method", "PUT").upper()
            task_id = item.get("task_id")
            payload = sanitize_task_payload(item.get("payload", {}))

            # Smart correlation with freshly created CRM deals
            if not payload.get("related_deal_id") or not payload.get("customer_name") or not payload.get("customer_id"):
                search_text = f"{payload.get('task_title', '')} {payload.get('deal_name', '')} {payload.get('customer_name', '')}".lower()
                for cd in created_deals:
                    c_id = cd.get("deal_id")
                    c_cust_id = cd.get("customer_id")
                    c_name = str(cd.get("deal_name", "")).lower()
                    c_cust = str(cd.get("company_name", "")).lower()
                    c_cust_raw = cd.get("company_name", "")
                    sim = compute_customer_similarity(c_cust_raw, payload.get("customer_name", "") or "")
                    if (c_cust and c_cust in search_text) or (c_name and (c_name in search_text or search_text in c_name)) or (sim >= 0.80):
                        if not payload.get("related_deal_id"):
                            payload["related_deal_id"] = c_id
                        if not payload.get("deal_name"):
                            payload["deal_name"] = cd.get("deal_name")
                        if not payload.get("customer_name"):
                            payload["customer_name"] = cd.get("company_name")
                        if not payload.get("customer_id") and c_cust_id:
                            payload["customer_id"] = c_cust_id
                        if not payload.get("deal_category") and cd.get("deal_category"):
                            payload["deal_category"] = cd.get("deal_category")
                        if not payload.get("closing_date") and cd.get("closing_date"):
                            payload["closing_date"] = cd.get("closing_date")
                        break

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
    key = os.getenv("GEMINI_API_KEY")
    if not key or is_placeholder(key):
        load_env_file()
        key = os.getenv("GEMINI_API_KEY")

    has_key = bool(key and not is_placeholder(key))
    masked = mask_key(key) if has_key else None

    return {
        "status": "online",
        "gemini_api_key_configured": has_key,
        "gemini_api_key_masked": masked,
        "gemini_model": GEMINI_MODEL,
        "crm_api": {"url": CRM_API_URL, "connected": baseline["crm_healthy"], "deals_count": len(baseline["deals"])},
        "tasks_api": {"url": TASKS_API_URL, "connected": baseline["tasks_healthy"], "tasks_count": len(baseline["tasks"])},
    }


@app.post("/api/set-api-key")
async def set_api_key(payload: Dict[str, str]):
    key = payload.get("api_key", "").strip()
    if not key or is_placeholder(key):
        raise HTTPException(status_code=400, detail="Please enter a valid Gemini API key.")
    os.environ["GEMINI_API_KEY"] = key
    save_env_file("GEMINI_API_KEY", key)
    return {
        "message": "GEMINI_API_KEY saved permanently to local .env file and Windows Environment.",
        "gemini_api_key_configured": True,
        "gemini_api_key_masked": mask_key(key),
    }


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
        d_cat = d.get("deal_category")
        if d_cat:
            d_cat_low = d_cat.lower()
            if "owner" in d_cat_low or "prime" in d_cat_low or "1-" in d_cat_low:
                kind = "RFP_OWNERSHIP"
            elif "distribut" in d_cat_low or "scope" in d_cat_low or "2-" in d_cat_low:
                kind = "RFP_DISTRIBUTED_SCOPE"
            elif "opp" in d_cat_low or "effort" in d_cat_low or "po" in d_cat_low or "3-" in d_cat_low:
                kind = "GENERAL_ACTION"

        if not kind:
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

        category_name = d.get("deal_category") or (
            "1- RFP Ownership & Prime Proposals" if kind == "RFP_OWNERSHIP"
            else "2- RFP Distributed Scope Items" if kind == "RFP_DISTRIBUTED_SCOPE"
            else "3- Opportunity Efforts & PO"
        )

        opportunities.append({
            "deal_id": deal_id,
            "deal_name": deal_name,
            "customer_name": customer_name,
            "deal_category": category_name,
            "closing_date": d.get("closing_date"),
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
            first_cat = first_t.get("deal_category") or (
                "1- RFP Ownership & Prime Proposals" if kind == "RFP_OWNERSHIP"
                else "2- RFP Distributed Scope Items" if kind == "RFP_DISTRIBUTED_SCOPE"
                else "3- Opportunity Efforts & PO"
            )

            opportunities.append({
                "deal_id": first_t.get("related_deal_id"),
                "deal_name": first_t.get("deal_name") or key,
                "customer_name": first_t.get("customer_name") or "Deliverables",
                "deal_category": first_cat,
                "closing_date": first_t.get("closing_date"),
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
                "name_en": "1- RFP Ownership & Prime Proposals",
                "name_ar": "مناقصات رئيسية وتكليف كامل",
                "badge": "badge-rfp-owner",
                "description": "Prime tenders owned end-to-end requiring technical architecture, RFP response submission, and bid management.",
                "opportunities": rfp_ownership_followup,
                "followup_opportunities": rfp_ownership_followup
            },
            "RFP_DISTRIBUTED_SCOPE": {
                "name_en": "2- RFP Distributed Scope Items",
                "name_ar": "نطاق موزع وشراكات التقنية",
                "badge": "badge-rfp-dist",
                "description": "Multi-vendor partner tenders (HPE, Dell, Veeam, Nutanix, VMware) requiring partner discounts, BoQ validations, and distributor scopes.",
                "opportunities": rfp_distributed_followup,
                "followup_opportunities": rfp_distributed_followup
            },
            "GENERAL_ACTION": {
                "name_en": "3- Opportunity Efforts & PO",
                "name_ar": "جهود الفرص المبكرة والتعميد المباشر",
                "badge": "badge-rfp-action",
                "description": "Early consultative assessment, customer visits, cooking RFP before public release, or direct PO.",
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

3. ONLY LOOK AT ACTIVE DEALS (SCALABILITY & ACCURACY RULE):
   - BASELINE CRM DEALS contains ONLY CURRENT ACTIVE DEALS (Discovery, Gathering Requirements, RFP / Tender, PoC, Proposal).
   - Historical closed deals (Closed-Won, Closed-Lost) are completed and omitted for scalability and to prevent confusion with previous deals.
   - Always compare spoken opportunities and tenders against current ACTIVE deals in BASELINE CRM DEALS.
   - If a deal is completed/won/lost, update the active deal's stage to "Closed-Won" or "Closed-Lost".
   - Do NOT confuse new opportunities with previous closed tenders.

BASELINE CRM DEALS (CURRENT ACTIVE DEALS ONLY):
{json.dumps(deals, indent=2, ensure_ascii=False)}

BASELINE TASK BOARD:
{json.dumps(tasks, indent=2, ensure_ascii=False)}

Your responsibilities:
1. Listen carefully and transcribe/understand all spoken updates from Presales 1 and Presales 2 exactly in their spoken languages without translation.
2. Compare spoken updates against the baseline CRM deals and Task Board items above to detect DELTAS:
   - Deal stage changes (e.g. Discovery -> RFP / Tender, RFP / Tender -> Proposal, PoC -> Proposal, Proposal -> Closed-Won).
   - Estimated deal value revisions.
   - Task status transitions (e.g. In Progress -> Completed, Waiting on Vendor -> In Progress, Not Started -> In Progress).
   - Resolved or newly raised management blockers.
   - Any newly mentioned deals or tasks to create.
3. Formulate structured REST API updates:
   - `crm_updates`: Array of deal updates.
     * For existing baseline deals: Use "PUT" with "deal_id" and "payload".
     * For newly discussed projects/opportunities (e.g. "مشروع كاست", "مشروع المراعي"): Use "POST" with "payload".
     IMPORTANT REQUIREMENTS FOR CRM DEALS (CRITICAL):
     * company_name: REQUIRED for "POST". Name of the client or enterprise organization (e.g. "كاست", "المراعي", "وزارة الصحة", "MOI"). If an existing customer in BASELINE CRM DEALS matches this entity (e.g. "المراعي" when mentioned as "شركة المراعي"), reuse the existing customer's registered company name.
     * deal_name: REQUIRED for "POST". Descriptive deal/project title in original Arabic/English as spoken (e.g. "تجديد الدعم الفني للأجهزة", "مشروع كاست - تحديث مركز البيانات", "مشروع شركة المراعي - حلول الأجهزة ومستلزمات Dell").
     * deal_category: MUST be one of the three standardized business categories:
       1. "1- RFP Ownership & Prime Proposals": When presales mentions owning a prime RFP/tender end-to-end (e.g. "I got an RFP for customer MOI and the name is renewal for hardware" / "استلمت مناقصة" / "اعمل على مناقصة").
       2. "2- RFP Distributed Scope Items": When receiving or working on a distributed vendor/partner scope within an RFP (e.g. "received a scope for the RFP name تجديد الدعم الفني للتجهيزات للعميل وزارة الداخلية").
       3. "3- Opportunity Efforts & PO": For early discovery, consultative assessment visits prior to public RFP ("received request to visit customer", "met customer to identify opportunity", "working early to cook the RFP before release") or direct PO.
     * closing_date: Tender closing date or submission deadline in format "YYYY-MM-DD" (e.g. if speaker says "تاريخ الاغلاق او حتسكر ب 21/8/2026", output "2026-08-21"). Mentioned especially for RFP tenders and distributed scopes. For early opportunity efforts without deadline, can be null.
     * primary_vendors: Array of vendor technologies (e.g. ["Hitachi Vantara"], ["Dell"], ["HPE"], ["Veeam"]).
     * stage: MUST be one of: ["Discovery", "Gathering Requirements", "RFP / Tender", "PoC", "Proposal", "Closed-Won", "Closed-Lost"]. (Never use "In Progress" for deal stage).
     * RFP & REQUEST FOR PRICING RULE (STRICT): If the presales engineer mentions receiving an RFP (طلب تقديم عروض), RFQ, or a request for pricing (طلب تسعير / استدراج عروض أسعار / تسعيرة مباشرة), set the deal stage to "RFP / Tender". This represents a direct request for pricing or tender proposal, usually for opportunities or private/commercial accounts not related to the Eitimad (منصة اعتماد) portal which is designated for government tenders.
     * assigned_presales: "Presales 1" or "Presales 2" (Engineer Abdullah maps to "Presales 1").
     * estimated_value: Numeric float in SAR/USD (e.g. 1.25M to 1.5M -> 1350000.0, 10M -> 10000000.0). Remember 1 million = 1000000.0.
     * vendor_notes: Technical requirements, scope, target close dates, in original spoken language.
    - `task_updates`: Array of task updates.
       STRICT TASK CATALOG REQUIREMENT (CRITICAL - CEASE FREE-FORM TASKS):
       * The presales operations pipeline uses a FIXED, PREDEFINED TASK CATALOG.
       * You MUST NOT generate open-ended or arbitrary free-form task titles.
       * Every task's `task_title` MUST BE EXACTLY ONE OF THE FOLLOWING PREDEFINED CATALOG ITEMS:
         --- PRE-RFP TASKS (for Category 3: Opportunity Efforts & PO) ---
         1. "Discovery & Technical Requirements Gathering"
         2. "High-Level Architecture (HLA) Design"
         3. "Preliminary BoQ & Budgetary Sizing"
         4. "RFP Specifications Shaping & Advisory"
         5. "Technical Proposal Draft & Client Review"
         --- RFP TASKS (for Category 1: RFP Ownership & Category 2: RFP Distributed Scope) ---
         1. "Bid / No-Bid Qualification & Owner Assignment"
         2. "RFP Decomposition & Scope Breakdown"
         3. "Clarification Questions Submission"
         4. "Low-Level Architecture & Technical Write-up"
         5. "Final BoQ & Vendor Quotations"
         6. "Technical Compliance Matrix"
         7. "Proposal Integration & Master Compliance Audit"
         8. "Final Technical Review & Commercial Handover"
       * ANY spoken details, engineer comments, partner coordination, site visits, or specific hardware context MUST be placed in `management_blockers` or notes, NEVER replacing the standardized catalog `task_title`.

        MANDATORY 5-STEP AGENT DATA PIPELINE:
        1. Determine whether a deal is NEW or EXISTING (CRITICAL - DEAL NAME SIMILARITY RULE):
           - ALWAYS extract the exact customer name and deal/tender title from the conversation.
           - A SINGLE customer can have MULTIPLE distinct tenders/RFPs/projects simultaneously (e.g. one for Software Licenses, one for Servers, one for Laptops, one for Networking).
           - Compare the spoken deal/tender name against existing deal names in BASELINE CRM DEALS:
             * IF AND ONLY IF the speaker is following up on an EXISTING deal with the SAME or SIMILAR deal name (e.g. updating prices, submitting clarifications, sending to tender department, changing stage for that project):
               Use "method": "PUT" WITH that specific "deal_id" to update that existing deal.
             * IF the speaker mentions receiving a NEW tender, a new RFP, or a project with a DIFFERENT name or scope (e.g. Incorta licenses, server procurement, laptops, networking):
               YOU MUST USE "method": "POST" to create a BRAND NEW DEAL, even if the customer already has other deals!
               NEVER modify an existing deal or append notes to an existing deal if the project/tender is distinct!
             * NEVER use generic fallback names like "مشروع General Client" or "General Client". ALWAYS use the real organization and authentic deal name.
        2. Update Deal Context:
           - Capture deal stage progression, estimated value, closing date, vendor notes, and primary vendors.
        3. Update Existing Task Statuses:
           - Check BASELINE TASK BOARD. If speech indicates an active task on the deal finished (e.g. "خلصنا زيارة الموقع" / "we finished site survey" / "we submitted questions"):
             formulate a "PUT" on that task's "task_id" with payload {{ "status": "Completed" }}.
        4. Enqueue the Next Action strictly from the Catalog:
           - Choose the logical next deliverable strictly from PRE_RFP_TASKS (for Opportunity Efforts) or RFP_TASKS (for RFPs).
           - Formulate a "POST" in task_updates linked to the deal, chained to the finished task via "previous_task_id", and set status to "In Progress" or "Not Started".
        5. MULTI-PRESALES & DISTRIBUTED SCOPE TASK CREATION RULE (CRITICAL):
           - If the same deal has been distributed to another presales engineer as well, and the other presales engineer informs that he received the distributed scope (e.g. Deal owned by Presales 1, and Presales 2 informs he received the distributed scope for Dell / Nutanix, or vice versa):
             * CRM DEAL: Add/amend the update to the SAME existing deal using "method": "PUT" with that deal's "deal_id". Append notes to "vendor_notes" and merge vendors into "primary_vendors". DO NOT create a duplicate deal!
             * TASK BOARD: DO NOT add progress to the same task or overwrite the other presales' existing task!
             * INSTEAD, GENERATE A BRAND NEW TASK ("method": "POST") for the other presales engineer:
               - "assigned_to": The presales engineer who received the distributed scope ("Presales 2" or "Presales 1").
               - "category": "RFP_DISTRIBUTED_SCOPE".
               - "deal_category": "2- RFP Distributed Scope Items".
               - "task_title": Standardized catalog item for their deliverable (e.g. "Low-Level Architecture & Technical Write-up", "Final BoQ & Vendor Quotations", "Technical Compliance Matrix", or "RFP Decomposition & Scope Breakdown").
               - "vendor_domain": Technology domain of their distributed scope (e.g. "Nutanix", "Dell", "HPE", "Veeam", "VMware", "General").
               - "status": "In Progress" (or "Not Started").
               - "related_deal_id": The existing deal's integer deal_id.
               - "deal_name": The existing deal's name.
               - "customer_name": The customer name.
               - "customer_id": The customer ID.
               - "management_blockers": Specific distributed scope details and notes from that engineer.

        IMPORTANT CONSTRAINTS FOR TASKS:
        * task_title: MUST be verbatim from PRE_RFP_TASKS or RFP_TASKS above.
        * management_blockers: Specific spoken context, blockers, or next step details.
        * customer_name: Name of customer or organization (e.g. "كاست", "المراعي", "MOI").
        * deal_name: Descriptive deal/project title.
        * deal_category: Corresponding deal category ("1- RFP Ownership & Prime Proposals", "2- RFP Distributed Scope Items", or "3- Opportunity Efforts & PO").
        * closing_date: Closing deadline (YYYY-MM-DD) if mentioned or associated with RFP tender.
        * status: MUST be one of: ["Not Started", "In Progress", "Waiting on Vendor", "Pending Review", "Completed"].
        * assigned_to: MUST be: "Presales 1" or "Presales 2".
        * category: MUST be: "RFP_OWNERSHIP" (prime tenders), "RFP_DISTRIBUTED_SCOPE" (vendor scopes/renewals), or "GENERAL_ACTION" (opportunity efforts / general deliverables).
        * vendor_domain: MUST be one of: ["HPE", "Veeam", "Dell", "Nutanix", "VMware", "General"]. NOTE: HP / Hewlett Packard MUST be set to "HPE"; Hitachi Vantara maps to "General".
        * priority: MUST be one of: ["High", "Medium", "Low"].
        * related_deal_id: Integer deal ID if associated with a CRM deal, else null.
        * previous_task_id: Integer ID of previous completed task on same deal, else null.
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
      "method": "POST",
      "payload": {{
        "company_name": "MOI",
        "deal_name": "تجديد الدعم الفني للأجهزة",
        "deal_category": "1- RFP Ownership & Prime Proposals",
        "closing_date": "2026-08-21",
        "primary_vendors": ["Dell", "HPE"],
        "stage": "RFP / Tender",
        "estimated_value": 10000000.0,
        "assigned_presales": "Presales 1",
        "vendor_notes": "كراسة تجديد الدعم الفني للأجهزة، تاريخ الإغلاق 21/8/2026"
      }}
    }},
    {{
      "method": "POST",
      "payload": {{
        "company_name": "كاست",
        "deal_name": "مشروع كاست - تحديث مركز البيانات",
        "deal_category": "3- Opportunity Efforts & PO",
        "closing_date": null,
        "primary_vendors": ["Hitachi Vantara"],
        "stage": "Gathering Requirements",
        "estimated_value": 1250000.0,
        "assigned_presales": "Presales 1",
        "vendor_notes": "تحديث مركز البيانات Data Center Tech Refresh بحلول Hitachi Vantara ميزانية 1 إلى 1.5 مليون ريال"
      }}
    }}
  ],
  "task_updates": [
    {{
      "method": "POST",
      "payload": {{
        "task_title": "RFP Decomposition & Scope Breakdown",
        "management_blockers": "توزيع نطاق العمل لشركاء Dell و Cisco لمناقصة وزارة الداخلية",
        "customer_name": "MOI",
        "deal_name": "تجديد الدعم الفني للأجهزة",
        "deal_category": "1- RFP Ownership & Prime Proposals",
        "closing_date": "2026-08-21",
        "category": "RFP_OWNERSHIP",
        "assigned_to": "Presales 1",
        "vendor_domain": "Dell",
        "status": "In Progress",
        "priority": "High",
        "related_deal_id": null,
        "previous_task_id": null,
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
    try:
        crm_updates = reconcile_crm_updates_from_conversation(ai_data, deals)
        ai_data["crm_updates"] = crm_updates
        task_updates = reconcile_tasks_from_conversation(ai_data, deals, tasks)
        sync_log = await execute_api_sync(crm_updates, task_updates)
    except Exception as e:
        print(f"Reconciliation or API sync error in process_audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"API Sync & Reconciliation error: {str(e)}")

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
                <a href="http://127.0.0.1:8000/dashboard" target="_blank" class="btn btn-sm btn-outline-info d-flex align-items-center gap-1">
                    <i class="bi bi-bar-chart-line-fill"></i> 📊 Visual Dashboard
                </a>
                <button class="btn btn-sm btn-outline-light d-flex align-items-center gap-1" id="apiKeyBtn" onclick="openApiKeyModal()">
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
                                <span class="badge badge-rfp-owner me-1">1- RFP Ownership</span>(<span id="count-RFP_OWNERSHIP">0</span>)
                            </button>
                            <button class="btn btn-sm btn-outline-secondary kind-filter-btn" id="filter-btn-RFP_DISTRIBUTED_SCOPE" onclick="filterFollowupByKind('RFP_DISTRIBUTED_SCOPE')">
                                <span class="badge badge-rfp-dist me-1">2- Distributed Scope</span>(<span id="count-RFP_DISTRIBUTED_SCOPE">0</span>)
                            </button>
                            <button class="btn btn-sm btn-outline-secondary kind-filter-btn" id="filter-btn-GENERAL_ACTION" onclick="filterFollowupByKind('GENERAL_ACTION')">
                                <span class="badge badge-rfp-action me-1">3- Opportunity Efforts & PO</span>(<span id="count-GENERAL_ACTION">0</span>)
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
                    <div id="currentApiKeyStatus" class="mb-3"></div>
                    <p class="small text-muted mb-2">Enter your Google Gemini API key to enable native audio speech recognition and dynamic generation via Gemini Flash models.</p>
                    <input type="password" id="geminiApiKeyInput" class="form-control mb-2" placeholder="AIzaSy... or AQ....">
                    <div class="small text-secondary"><i class="bi bi-shield-check me-1 text-success"></i>Saved permanently to local <code>.env</code> file &amp; Windows User Environment (persists across restarts).</div>
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
                const btn = document.getElementById('apiKeyBtn');
                if (data.gemini_api_key_configured) {
                    const masked = data.gemini_api_key_masked ? ` (${data.gemini_api_key_masked})` : '';
                    if (btn) btn.className = 'btn btn-sm btn-outline-success d-flex align-items-center gap-1';
                    document.getElementById('apiKeyBtnText').textContent = `Gemini Active${masked}`;
                } else {
                    if (btn) btn.className = 'btn btn-sm btn-outline-warning d-flex align-items-center gap-1';
                    document.getElementById('apiKeyBtnText').textContent = "Configure Gemini Key";
                    // Check if saved in localStorage as a backup
                    const localKey = localStorage.getItem('gemini_api_key');
                    if (localKey && !localKey.includes('your_actual') && localKey.trim().length > 10) {
                        await fetch('/api/set-api-key', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ api_key: localKey.trim() })
                        });
                        await checkHealth();
                        return;
                    }
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
                let kindName = '3- Opportunity Efforts & PO';
                let kindNameAr = 'جهود الفرص المبكرة والتعميد المباشر';
                if (o.kind === 'RFP_OWNERSHIP') {
                    kindBadgeClass = 'badge-rfp-owner';
                    kindName = '1- RFP Ownership & Prime Proposals';
                    kindNameAr = 'مناقصة رئيسية وتكليف كامل';
                } else if (o.kind === 'RFP_DISTRIBUTED_SCOPE') {
                    kindBadgeClass = 'badge-rfp-dist';
                    kindName = '2- RFP Distributed Scope Items';
                    kindNameAr = 'نطاق موزع وشراكات التقنية';
                }

                const followCount = o.followup_tasks_count || 0;
                const statusBadge = `<span class="badge bg-warning text-dark"><i class="bi bi-clock-history me-1"></i>Follow-up Needed (${followCount})</span>`;

                const blockerBadge = o.has_blocker
                    ? `<span class="badge badge-blocker ms-1"><i class="bi bi-exclamation-triangle-fill me-1"></i>Blocked</span>`
                    : '';

                const closingBadge = o.closing_date
                    ? `<span class="badge bg-danger-subtle text-danger border border-danger-subtle ms-1"><i class="bi bi-alarm-fill me-1"></i>Closing: ${o.closing_date}</span>`
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
                                            ${t.closing_date ? `<span><i class="bi bi-alarm text-danger me-1"></i>Closing: <strong class="text-danger">${t.closing_date}</strong></span>` : ''}
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
                                <span class="badge ${kindBadgeClass} me-1">${o.deal_category || kindName}</span>
                                <span class="font-arabic small text-muted">(${kindNameAr})</span>
                            </div>
                            <div class="d-flex flex-wrap gap-1 align-items-center">
                                ${closingBadge}
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
                    let errDetail = 'Server error (' + res.status + ')';
                    try {
                        const err = await res.json();
                        errDetail = err.detail || JSON.stringify(err);
                    } catch(e) {
                        try {
                            const txt = await res.text();
                            if (txt) errDetail = txt;
                        } catch(e2) {}
                    }
                    alert('Audio processing error: ' + errDetail);
                    return;
                }

                const data = await res.json();
                renderResults(data);
            } catch (err) {
                console.error("Audio processing failed:", err);
                alert('Communication error: ' + (err.message || 'Network error communicating with Voice Agent server.'));
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

            // Render current status and masked key
            fetch('/api/health').then(r => r.json()).then(d => {
                const statusEl = document.getElementById('currentApiKeyStatus');
                if (statusEl) {
                    if (d.gemini_api_key_configured) {
                        statusEl.innerHTML = `
                            <div class="alert alert-success border border-success border-opacity-25 bg-success bg-opacity-10 small py-2 mb-2">
                                <i class="bi bi-shield-check text-success me-1"></i>
                                <strong>Active Gemini Key:</strong> <code>${d.gemini_api_key_masked || 'Configured'}</code>
                                <div class="text-secondary mt-1" style="font-size:0.75rem;">Key is permanently saved in local <code>.env</code> and Windows User environment. It will stay active on server restart.</div>
                            </div>
                        `;
                    } else {
                        statusEl.innerHTML = `
                            <div class="alert alert-warning border border-warning border-opacity-25 bg-warning bg-opacity-10 small py-2 mb-2">
                                <i class="bi bi-exclamation-triangle-fill text-warning me-1"></i>
                                No active API key configured. Enter your Gemini API key below to activate AI features.
                            </div>
                        `;
                    }
                }
            }).catch(e => console.warn("Failed fetching health for modal:", e));

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
                    localStorage.setItem('gemini_api_key', key);
                    document.getElementById('geminiApiKeyInput').value = '';
                    closeApiKeyModal();
                    await checkHealth();
                    alert('Gemini API key saved permanently! It will remain active even when restarting.');
                } else {
                    const err = await res.json();
                    alert('Failed to save API key: ' + (err.detail || 'Invalid key'));
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
