import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DB_FILE = Path(__file__).resolve().parent / "tasks.db"
CRM_DB_FILE = Path(__file__).resolve().parent / "crm.db"


def resolve_deal_details(
    deal_id: Optional[int], 
    custom_deal: Optional[str] = None, 
    custom_customer: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Resolves deal title and customer company name from crm.db if not explicitly provided."""
    d_name = str(custom_deal).strip() if custom_deal and str(custom_deal).strip() else None
    c_name = str(custom_customer).strip() if custom_customer and str(custom_customer).strip() else None

    if deal_id and CRM_DB_FILE.exists() and (not d_name or not c_name):
        try:
            with sqlite3.connect(CRM_DB_FILE) as crm_conn:
                crm_conn.row_factory = sqlite3.Row
                row = crm_conn.execute("""
                    SELECT d.deal_name, c.company_name 
                    FROM deals d 
                    JOIN customers c ON d.customer_id = c.customer_id 
                    WHERE d.deal_id = ?;
                """, (deal_id,)).fetchone()
                if row:
                    if not d_name:
                        d_name = row["deal_name"]
                    if not c_name:
                        c_name = row["company_name"]
        except Exception as e:
            print(f"Notice: Failed to query crm.db: {e}")

    return d_name, c_name


class TaskCategory(str, Enum):
    RFP_OWNERSHIP = "RFP_OWNERSHIP"
    RFP_DISTRIBUTED_SCOPE = "RFP_DISTRIBUTED_SCOPE"
    GENERAL_ACTION = "GENERAL_ACTION"


class TaskStatus(str, Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    WAITING_ON_VENDOR = "Waiting on Vendor"
    PENDING_REVIEW = "Pending Review"
    COMPLETED = "Completed"


class TaskPriority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class PresalesRep(str, Enum):
    PRESALES_1 = "Presales 1"
    PRESALES_2 = "Presales 2"


class VendorDomain(str, Enum):
    HPE = "HPE"
    VEEAM = "Veeam"
    DELL = "Dell"
    NUTANIX = "Nutanix"
    VMWARE = "VMware"
    GENERAL = "General"


# -----------------------------------------------------------------------------
# Database Setup & Initialization
# -----------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_title TEXT NOT NULL,
        category TEXT NOT NULL CHECK(category IN ('RFP_OWNERSHIP', 'RFP_DISTRIBUTED_SCOPE', 'GENERAL_ACTION')),
        related_deal_id INTEGER,
        assigned_to TEXT NOT NULL CHECK(assigned_to IN ('Presales 1', 'Presales 2')),
        vendor_domain TEXT NOT NULL CHECK(vendor_domain IN ('HPE', 'Veeam', 'Dell', 'Nutanix', 'VMware', 'General')),
        status TEXT NOT NULL CHECK(status IN ('Not Started', 'In Progress', 'Waiting on Vendor', 'Pending Review', 'Completed')),
        priority TEXT NOT NULL CHECK(priority IN ('High', 'Medium', 'Low')),
        due_date TEXT,
        management_blockers TEXT,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Schema migration check: ensure started_at, completed_at, customer_name, deal_name exist
    cursor.execute("PRAGMA table_info(tasks);")
    existing_cols = [r["name"] for r in cursor.fetchall()]
    if "started_at" not in existing_cols:
        cursor.execute("ALTER TABLE tasks ADD COLUMN started_at TIMESTAMP;")
    if "completed_at" not in existing_cols:
        cursor.execute("ALTER TABLE tasks ADD COLUMN completed_at TIMESTAMP;")
    if "customer_name" not in existing_cols:
        cursor.execute("ALTER TABLE tasks ADD COLUMN customer_name TEXT;")
    if "deal_name" not in existing_cols:
        cursor.execute("ALTER TABLE tasks ADD COLUMN deal_name TEXT;")

    # Backfill customer_name & deal_name from crm.db where available
    if CRM_DB_FILE.exists():
        try:
            with sqlite3.connect(CRM_DB_FILE) as crm_conn:
                crm_conn.row_factory = sqlite3.Row
                crm_rows = crm_conn.execute("""
                    SELECT d.deal_id, d.deal_name, c.company_name 
                    FROM deals d 
                    JOIN customers c ON d.customer_id = c.customer_id;
                """).fetchall()
                deal_map = {r["deal_id"]: (r["deal_name"], r["company_name"]) for r in crm_rows}

                cursor.execute("SELECT task_id, related_deal_id, task_title, customer_name, deal_name FROM tasks;")
                for t in cursor.fetchall():
                    tid = t["task_id"]
                    did = t["related_deal_id"]
                    t_title = t["task_title"] or ""
                    c_name = t["customer_name"]
                    d_name = t["deal_name"]

                    if did and did in deal_map:
                        crm_deal, crm_cust = deal_map[did]
                        if not d_name:
                            d_name = crm_deal
                        if not c_name:
                            c_name = crm_cust
                    else:
                        low_title = t_title.lower()
                        for _did, (_dname, _cname) in deal_map.items():
                            if _cname.lower() in low_title or _dname.lower() in low_title:
                                if not c_name:
                                    c_name = _cname
                                if not d_name:
                                    d_name = _dname
                                break
                        if not c_name:
                            if "almarai" in low_title:
                                c_name = "Almarai"
                            elif "human resources" in low_title or "hr" in low_title:
                                c_name = "Ministry of Human Resources"
                            elif "solarwinds" in low_title:
                                c_name = "SolarWinds Request"
                            elif "hajj" in low_title:
                                c_name = "Ministry of Hajj"
                            elif "foreign affairs" in low_title:
                                c_name = "Ministry of Foreign Affairs"
                            elif "planning" in low_title:
                                c_name = "Ministry of Planning"
                            elif "jeddah" in low_title:
                                c_name = "Jeddah Municipality"
                            elif "electronic university" in low_title:
                                c_name = "Saudi Electronic University"

                    if c_name or d_name:
                        cursor.execute(
                            "UPDATE tasks SET customer_name = ?, deal_name = ? WHERE task_id = ?;",
                            (c_name, d_name, tid)
                        )
        except Exception as e:
            print(f"Notice during deal backfill: {e}")


    # Activity event log table for full lifecycle tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        from_status TEXT,
        to_status TEXT NOT NULL,
        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
    );
    """)

    # Seed illustrative default tasks if database is empty
    cursor.execute("SELECT COUNT(*) FROM tasks;")
    if cursor.fetchone()[0] == 0:
        seed_tasks = [
            (
                "Author Executive Technical Summary for Hospital Cluster",
                "RFP_OWNERSHIP",
                1,
                "Presales 1",
                "HPE",
                "In Progress",
                "High",
                "2026-09-18",
                "Awaiting final sizing quote from HPE partner portal.",
                "2026-09-02 09:00:00",
                None,
            ),
            (
                "Validate Nutanix AHV Sizing & IOPS Specs",
                "RFP_DISTRIBUTED_SCOPE",
                1,
                "Presales 2",
                "Nutanix",
                "Waiting on Vendor",
                "High",
                "2026-09-15",
                "Vendor SE is on annual leave until Tuesday.",
                "2026-09-03 11:30:00",
                None,
            ),
            (
                "Build Veeam v12.2 Immutability Bill of Materials (BOM)",
                "RFP_DISTRIBUTED_SCOPE",
                2,
                "Presales 2",
                "Veeam",
                "Pending Review",
                "Medium",
                "2026-09-22",
                None,
                "2026-09-04 14:00:00",
                None,
            ),
            (
                "Dell PowerProtect Sizing Calculator Verification",
                "RFP_DISTRIBUTED_SCOPE",
                2,
                "Presales 1",
                "Dell",
                "Completed",
                "High",
                "2026-09-10",
                None,
                "2026-09-01 08:30:00",
                "2026-09-04 16:45:00",
            ),
            (
                "Review VMware VCF Licensing Migration Playbook",
                "GENERAL_ACTION",
                4,
                "Presales 2",
                "VMware",
                "Not Started",
                "Medium",
                "2026-09-25",
                "Pending customer confirmation of core counts.",
                None,
                None,
            ),
            (
                "Schedule Lab PoC Readiness Call for FinTech Horizons",
                "GENERAL_ACTION",
                2,
                "Presales 1",
                "General",
                "In Progress",
                "Low",
                "2026-09-12",
                None,
                "2026-09-04 10:00:00",
                None,
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO tasks (
                task_title, category, related_deal_id, assigned_to, vendor_domain, status, priority, due_date, management_blockers, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            seed_tasks,
        )

    # Backfill timestamps for any completed/in-progress tasks if null
    cursor.execute("UPDATE tasks SET started_at = created_at WHERE status != 'Not Started' AND started_at IS NULL;")
    cursor.execute("UPDATE tasks SET completed_at = updated_at WHERE status = 'Completed' AND completed_at IS NULL;")

    # Seed initial activity logs for tasks if activity table is empty
    cursor.execute("SELECT COUNT(*) FROM task_activity_log;")
    if cursor.fetchone()[0] == 0:
        cursor.execute("SELECT task_id, status, assigned_to, created_at, started_at, completed_at FROM tasks;")
        for t_row in cursor.fetchall():
            # Initial creation log
            cursor.execute(
                """
                INSERT INTO task_activity_log (task_id, from_status, to_status, changed_at, user_id)
                VALUES (?, ?, ?, ?, ?);
                """,
                (t_row["task_id"], None, "Not Started", t_row["created_at"], t_row["assigned_to"]),
            )
            # If transitioned to started/in progress
            if t_row["started_at"] and t_row["status"] != "Not Started":
                cursor.execute(
                    """
                    INSERT INTO task_activity_log (task_id, from_status, to_status, changed_at, user_id)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (t_row["task_id"], "Not Started", "In Progress", t_row["started_at"], t_row["assigned_to"]),
                )
            # If completed
            if t_row["completed_at"] and t_row["status"] == "Completed":
                cursor.execute(
                    """
                    INSERT INTO task_activity_log (task_id, from_status, to_status, changed_at, user_id)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (t_row["task_id"], "In Progress", "Completed", t_row["completed_at"], t_row["assigned_to"]),
                )

    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# -----------------------------------------------------------------------------
# FastAPI Application & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Monday-Style Presales Task Board",
    description="Task management API for RFP Ownership, Distributed Scopes, and General Presales Actions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local agent HTTP requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class TaskBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_title: str = Field(default="Presales Action Item", description="Actionable title for the task")
    category: Union[TaskCategory, str] = Field(default=TaskCategory.GENERAL_ACTION, description="RFP_OWNERSHIP, RFP_DISTRIBUTED_SCOPE, or GENERAL_ACTION")
    related_deal_id: Optional[Union[int, str]] = Field(None, description="Optional foreign deal ID reference")
    customer_name: Optional[str] = Field(None, description="Customer or client organization name")
    deal_name: Optional[str] = Field(None, description="Associated CRM deal / tender title")
    assigned_to: Union[PresalesRep, str] = Field(default=PresalesRep.PRESALES_1, description="Presales 1 or Presales 2")
    vendor_domain: Union[VendorDomain, str] = Field(default=VendorDomain.GENERAL, description="Vendor scope")
    status: Union[TaskStatus, str] = Field(default=TaskStatus.NOT_STARTED, description="Current workflow state")
    priority: Union[TaskPriority, str] = Field(default=TaskPriority.MEDIUM, description="Task urgency")
    due_date: Optional[str] = Field(None, description="Target completion date (YYYY-MM-DD)")
    management_blockers: Optional[str] = Field(None, description="Escalations, vendor delays, or blockers")

    @field_validator("task_title", mode="before")
    @classmethod
    def normalize_title(cls, v):
        if not v or not str(v).strip():
            return "Presales Action Item"
        return str(v).strip()

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if not v:
            return TaskCategory.GENERAL_ACTION
        v_str = str(v).strip().lower()
        if "owner" in v_str:
            return TaskCategory.RFP_OWNERSHIP
        if "distribut" in v_str or "scope" in v_str or "rfp" in v_str:
            return TaskCategory.RFP_DISTRIBUTED_SCOPE
        return TaskCategory.GENERAL_ACTION

    @field_validator("assigned_to", mode="before")
    @classmethod
    def normalize_assigned(cls, v):
        if not v:
            return PresalesRep.PRESALES_1
        v_str = str(v).strip().lower()
        if "2" in v_str:
            return PresalesRep.PRESALES_2
        return PresalesRep.PRESALES_1

    @field_validator("vendor_domain", mode="before")
    @classmethod
    def normalize_vendor(cls, v):
        if not v:
            return VendorDomain.GENERAL
        v_str = str(v).strip().lower()
        if "hp" in v_str or "hewlett" in v_str or "proliant" in v_str or "alletra" in v_str:
            return VendorDomain.HPE
        if "veeam" in v_str:
            return VendorDomain.VEEAM
        if "dell" in v_str or "poweredge" in v_str or "powerprotect" in v_str or "emc" in v_str:
            return VendorDomain.DELL
        if "nutanix" in v_str or "ahv" in v_str:
            return VendorDomain.NUTANIX
        if "vmware" in v_str or "vcf" in v_str or "vsphere" in v_str or "broadcom" in v_str:
            return VendorDomain.VMWARE
        return VendorDomain.GENERAL

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v):
        if not v:
            return TaskStatus.NOT_STARTED
        v_str = str(v).strip().lower()
        if "completed" in v_str or "done" in v_str or "closed" in v_str or "finished" in v_str or "won" in v_str:
            return TaskStatus.COMPLETED
        if "waiting" in v_str or "vendor" in v_str:
            return TaskStatus.WAITING_ON_VENDOR
        if "review" in v_str or "pending" in v_str:
            return TaskStatus.PENDING_REVIEW
        if "progress" in v_str or "ongoing" in v_str or "active" in v_str or "started" in v_str:
            return TaskStatus.IN_PROGRESS
        return TaskStatus.NOT_STARTED

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v):
        if not v:
            return TaskPriority.MEDIUM
        v_str = str(v).strip().lower()
        if "high" in v_str or "critical" in v_str or "urgent" in v_str or "p1" in v_str:
            return TaskPriority.HIGH
        if "low" in v_str or "p3" in v_str:
            return TaskPriority.LOW
        return TaskPriority.MEDIUM

    @field_validator("related_deal_id", mode="before")
    @classmethod
    def normalize_deal_id(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        nums = re.findall(r"\d+", str(v))
        if nums:
            return int(nums[0])
        return None


class TaskCreate(TaskBase):
    changed_by: Optional[str] = Field(default="Voice Agent", description="User or agent recording the creation")

    @model_validator(mode="before")
    @classmethod
    def check_aliases(cls, data):
        if isinstance(data, dict):
            if "title" in data and "task_title" not in data:
                data["task_title"] = data["title"]
        return data


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_title: Optional[str] = None
    category: Optional[Union[TaskCategory, str]] = None
    related_deal_id: Optional[Union[int, str]] = None
    customer_name: Optional[str] = None
    deal_name: Optional[str] = None
    assigned_to: Optional[Union[PresalesRep, str]] = None
    vendor_domain: Optional[Union[VendorDomain, str]] = None
    status: Optional[Union[TaskStatus, str]] = None
    priority: Optional[Union[TaskPriority, str]] = None
    due_date: Optional[str] = None
    management_blockers: Optional[str] = None
    changed_by: Optional[str] = Field(default="Voice Agent", description="User or agent moving the task")

    @model_validator(mode="before")
    @classmethod
    def check_aliases(cls, data):
        if isinstance(data, dict):
            if "title" in data and "task_title" not in data:
                data["task_title"] = data["title"]
        return data

    @field_validator("task_title", mode="before")
    @classmethod
    def normalize_title(cls, v):
        if v is not None:
            return str(v).strip()
        return None

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if "owner" in v_str:
            return TaskCategory.RFP_OWNERSHIP
        if "distribut" in v_str or "scope" in v_str or "rfp" in v_str:
            return TaskCategory.RFP_DISTRIBUTED_SCOPE
        return TaskCategory.GENERAL_ACTION

    @field_validator("assigned_to", mode="before")
    @classmethod
    def normalize_assigned(cls, v):
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if "2" in v_str:
            return PresalesRep.PRESALES_2
        return PresalesRep.PRESALES_1

    @field_validator("vendor_domain", mode="before")
    @classmethod
    def normalize_vendor(cls, v):
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if "hp" in v_str or "hewlett" in v_str or "proliant" in v_str or "alletra" in v_str:
            return VendorDomain.HPE
        if "veeam" in v_str:
            return VendorDomain.VEEAM
        if "dell" in v_str or "poweredge" in v_str or "powerprotect" in v_str or "emc" in v_str:
            return VendorDomain.DELL
        if "nutanix" in v_str or "ahv" in v_str:
            return VendorDomain.NUTANIX
        if "vmware" in v_str or "vcf" in v_str or "vsphere" in v_str or "broadcom" in v_str:
            return VendorDomain.VMWARE
        return VendorDomain.GENERAL

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v):
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if "completed" in v_str or "done" in v_str or "closed" in v_str or "finished" in v_str or "won" in v_str:
            return TaskStatus.COMPLETED
        if "waiting" in v_str or "vendor" in v_str:
            return TaskStatus.WAITING_ON_VENDOR
        if "review" in v_str or "pending" in v_str:
            return TaskStatus.PENDING_REVIEW
        if "progress" in v_str or "ongoing" in v_str or "active" in v_str or "started" in v_str:
            return TaskStatus.IN_PROGRESS
        return TaskStatus.NOT_STARTED

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v):
        if v is None:
            return None
        v_str = str(v).strip().lower()
        if "high" in v_str or "critical" in v_str or "urgent" in v_str or "p1" in v_str:
            return TaskPriority.HIGH
        if "low" in v_str or "p3" in v_str:
            return TaskPriority.LOW
        return TaskPriority.MEDIUM

    @field_validator("related_deal_id", mode="before")
    @classmethod
    def normalize_deal_id(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        nums = re.findall(r"\d+", str(v))
        if nums:
            return int(nums[0])
        return None


class TaskOut(TaskBase):
    task_id: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    lead_time: Optional[str] = None
    cycle_time: Optional[str] = None
    created_at: str
    updated_at: str


class TaskActivityOut(BaseModel):
    id: int
    task_id: int
    from_status: Optional[str] = None
    to_status: str
    changed_at: str
    user_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_duration(start_str: Optional[str], end_str: Optional[str]) -> Optional[str]:
    """Calculates human-readable elapsed duration (days or hours)."""
    if not start_str or not end_str:
        return None
    try:
        s = datetime.strptime(start_str.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")
        e = datetime.strptime(end_str.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")
        diff = e - s
        total_seconds = diff.total_seconds()
        if total_seconds < 0:
            return None
        hours = round(total_seconds / 3600.0, 1)
        if hours >= 24:
            days = round(hours / 24.0, 1)
            return f"{days} d"
        return f"{hours} h"
    except Exception:
        return None


def row_to_task(row: sqlite3.Row) -> TaskOut:
    created_at = str(row["created_at"])
    started_at = str(row["started_at"]) if row["started_at"] else None
    completed_at = str(row["completed_at"]) if row["completed_at"] else None

    # Performance Metrics:
    # Lead Time: Completed Date - Created Date (total time from creation to completion)
    # Cycle Time: Completed Date - Started Date (actual active working time)
    lead_time = compute_duration(created_at, completed_at)
    cycle_time = compute_duration(started_at, completed_at)

    cols = row.keys()
    c_name = row["customer_name"] if "customer_name" in cols and row["customer_name"] else None
    d_name = row["deal_name"] if "deal_name" in cols and row["deal_name"] else None
    d_id = row["related_deal_id"]

    if (not c_name or not d_name) and d_id:
        auto_deal, auto_cust = resolve_deal_details(d_id, d_name, c_name)
        d_name = d_name or auto_deal
        c_name = c_name or auto_cust

    return TaskOut(
        task_id=row["task_id"],
        task_title=row["task_title"],
        category=row["category"],
        related_deal_id=d_id,
        customer_name=c_name,
        deal_name=d_name,
        assigned_to=row["assigned_to"],
        vendor_domain=row["vendor_domain"],
        status=row["status"],
        priority=row["priority"],
        due_date=row["due_date"],
        management_blockers=row["management_blockers"],
        started_at=started_at,
        completed_at=completed_at,
        lead_time=lead_time,
        cycle_time=cycle_time,
        created_at=created_at,
        updated_at=str(row["updated_at"]),
    )



# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/tasks", tags=["Tasks"])
def get_tasks(
    group_by: Optional[str] = Query(None, description="Optional grouping: 'status', 'assigned_to', or 'category'"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned user"),
    active_only: bool = Query(False, description="Exclude completed tasks"),
    created_today: bool = Query(False, description="Filter items created today"),
    completed_today: bool = Query(False, description="Filter items completed today"),
    completed_this_week: bool = Query(False, description="Filter items completed in the past 7 days"),
):
    """
    Retrieve all tasks, with filtering by daily velocity (created today, completed this week)
    and optional grouping by status or assigned user.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM tasks WHERE 1=1"
    params = []

    if active_only:
        query += " AND status != 'Completed'"
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)
    if assigned_to:
        query += " AND assigned_to = ?"
        params.append(assigned_to)
    if created_today:
        query += " AND date(created_at) = date('now')"
    if completed_today:
        query += " AND date(completed_at) = date('now')"
    if completed_this_week:
        query += " AND date(completed_at) >= date('now', '-7 days')"

    query += " ORDER BY CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 3 ELSE 4 END, due_date ASC;"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    tasks = [row_to_task(r) for r in rows]

    if group_by:
        grouped: Dict[str, List[TaskOut]] = {}
        for t in tasks:
            key = getattr(t, group_by, "Other")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(t)
        return {"group_by": group_by, "grouped": grouped, "total": len(tasks), "items": tasks}

    return tasks


@app.get("/api/tasks/metrics/velocity", tags=["Tasks"])
def get_velocity_metrics():
    """
    Returns board throughput, daily standup counts, Lead Time, Cycle Time, and activity transitions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM tasks WHERE date(created_at) = date('now');")
    created_today = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tasks WHERE date(completed_at) = date('now');")
    completed_today = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tasks WHERE date(completed_at) >= date('now', '-7 days');")
    completed_this_week = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tasks WHERE status != 'Completed' AND started_at IS NOT NULL;")
    in_flight = cursor.fetchone()[0]

    # Compute average cycle time (completed_at - started_at) and lead time (completed_at - created_at)
    cursor.execute("""
        SELECT 
            AVG((julianday(completed_at) - julianday(started_at)) * 24.0) as avg_cycle_hours,
            AVG((julianday(completed_at) - julianday(created_at)) * 24.0) as avg_lead_hours
        FROM tasks 
        WHERE status = 'Completed' AND completed_at IS NOT NULL AND started_at IS NOT NULL;
    """)
    metrics_row = cursor.fetchone()
    avg_cycle_hours = round(metrics_row["avg_cycle_hours"], 1) if metrics_row["avg_cycle_hours"] else 0.0
    avg_lead_hours = round(metrics_row["avg_lead_hours"], 1) if metrics_row["avg_lead_hours"] else 0.0

    # Recent 10 activity log transitions
    cursor.execute("""
        SELECT a.id, a.task_id, t.task_title, a.from_status, a.to_status, a.changed_at, a.user_id
        FROM task_activity_log a
        JOIN tasks t ON a.task_id = t.task_id
        ORDER BY a.changed_at DESC, a.id DESC
        LIMIT 10;
    """)
    recent_activities = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return {
        "created_today": created_today,
        "completed_today": completed_today,
        "completed_this_week": completed_this_week,
        "in_flight": in_flight,
        "avg_cycle_time": f"{round(avg_cycle_hours / 24.0, 1)} d" if avg_cycle_hours >= 24 else f"{avg_cycle_hours} h",
        "avg_lead_time": f"{round(avg_lead_hours / 24.0, 1)} d" if avg_lead_hours >= 24 else f"{avg_lead_hours} h",
        "recent_activity": recent_activities,
    }


@app.get("/api/tasks/{task_id}/history", response_model=List[TaskActivityOut], tags=["Tasks"])
def get_task_history(task_id: int):
    """
    Returns the audit event log of all state transitions for a specific task.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, task_id, from_status, to_status, changed_at, user_id
        FROM task_activity_log
        WHERE task_id = ?
        ORDER BY changed_at DESC, id DESC;
    """, (task_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        TaskActivityOut(
            id=r["id"],
            task_id=r["task_id"],
            from_status=r["from_status"],
            to_status=r["to_status"],
            changed_at=str(r["changed_at"]),
            user_id=r["user_id"],
        )
        for r in rows
    ]


@app.get("/api/tasks/rfp", response_model=List[TaskOut], tags=["Tasks"])
def get_rfp_tasks():
    """
    Get all active RFP ownership and distributed scope items (status != 'Completed').
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM tasks
        WHERE category IN ('RFP_OWNERSHIP', 'RFP_DISTRIBUTED_SCOPE')
          AND status != 'Completed'
        ORDER BY CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, due_date ASC;
    """)
    rows = cursor.fetchall()
    conn.close()

    return [row_to_task(r) for r in rows]


@app.post("/api/tasks", response_model=TaskOut, status_code=status.HTTP_201_CREATED, tags=["Tasks"])
def create_task(payload: TaskCreate):
    """
    Create a new presales task with automatic started_at and activity logging.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    cat_val = payload.category.value if hasattr(payload.category, "value") else str(payload.category)
    assigned_val = payload.assigned_to.value if hasattr(payload.assigned_to, "value") else str(payload.assigned_to)
    vendor_val = payload.vendor_domain.value if hasattr(payload.vendor_domain, "value") else str(payload.vendor_domain)
    status_val = payload.status.value if hasattr(payload.status, "value") else str(payload.status)
    priority_val = payload.priority.value if hasattr(payload.priority, "value") else str(payload.priority)

    # Timestamps tracking
    started_at = now_iso if status_val != TaskStatus.NOT_STARTED.value else None
    completed_at = now_iso if status_val == TaskStatus.COMPLETED.value else None

    # Resolve customer & deal names
    deal_name, customer_name = resolve_deal_details(
        payload.related_deal_id,
        payload.deal_name,
        payload.customer_name
    )

    cursor.execute(
        """
        INSERT INTO tasks (
            task_title, category, related_deal_id, customer_name, deal_name, assigned_to, vendor_domain, status, priority, due_date, management_blockers, started_at, completed_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            payload.task_title.strip(),
            cat_val,
            payload.related_deal_id,
            customer_name,
            deal_name,
            assigned_val,
            vendor_val,
            status_val,
            priority_val,
            payload.due_date.strip() if payload.due_date else None,
            payload.management_blockers.strip() if payload.management_blockers else None,
            started_at,
            completed_at,
            now_iso,
            now_iso,
        ),
    )

    task_id = cursor.lastrowid

    # Record creation transition in task_activity_log
    actor = payload.changed_by or assigned_val
    cursor.execute(
        """
        INSERT INTO task_activity_log (task_id, from_status, to_status, changed_at, user_id)
        VALUES (?, ?, ?, ?, ?);
        """,
        (task_id, None, status_val, now_iso, str(actor)),
    )

    conn.commit()

    cursor.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,))
    row = cursor.fetchone()
    conn.close()

    return row_to_task(row)


@app.put("/api/tasks/{task_id}", response_model=TaskOut, tags=["Tasks"])
def update_task(task_id: int, payload: TaskUpdate):
    """
    Update task status, blockers, due date, priority, or any other field dynamically.
    Tracks lifecycle timestamps (started_at, completed_at) and logs status transitions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,))
    current_row = cursor.fetchone()
    if not current_row:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task #{task_id} not found.")

    old_status = current_row["status"]
    old_started_at = current_row["started_at"]
    old_completed_at = current_row["completed_at"]
    old_assigned = current_row["assigned_to"]
    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    update_clauses = []
    params = []

    if payload.task_title is not None:
        update_clauses.append("task_title = ?")
        params.append(payload.task_title.strip())

    if payload.category is not None:
        cat_val = payload.category.value if hasattr(payload.category, "value") else str(payload.category)
        update_clauses.append("category = ?")
        params.append(cat_val)

    if payload.related_deal_id is not None:
        update_clauses.append("related_deal_id = ?")
        params.append(payload.related_deal_id)

    if payload.customer_name is not None:
        update_clauses.append("customer_name = ?")
        params.append(payload.customer_name.strip() if payload.customer_name else None)

    if payload.deal_name is not None:
        update_clauses.append("deal_name = ?")
        params.append(payload.deal_name.strip() if payload.deal_name else None)

    # Auto-resolve deal & customer name if related_deal_id was updated and names were not explicitly passed
    if payload.related_deal_id is not None and payload.customer_name is None and payload.deal_name is None:
        auto_deal, auto_cust = resolve_deal_details(payload.related_deal_id)
        if auto_deal:
            update_clauses.append("deal_name = ?")
            params.append(auto_deal)
        if auto_cust:
            update_clauses.append("customer_name = ?")
            params.append(auto_cust)

    if payload.assigned_to is not None:

        assigned_val = payload.assigned_to.value if hasattr(payload.assigned_to, "value") else str(payload.assigned_to)
        update_clauses.append("assigned_to = ?")
        params.append(assigned_val)

    if payload.vendor_domain is not None:
        vendor_val = payload.vendor_domain.value if hasattr(payload.vendor_domain, "value") else str(payload.vendor_domain)
        update_clauses.append("vendor_domain = ?")
        params.append(vendor_val)

    if payload.priority is not None:
        priority_val = payload.priority.value if hasattr(payload.priority, "value") else str(payload.priority)
        update_clauses.append("priority = ?")
        params.append(priority_val)

    if payload.due_date is not None:
        update_clauses.append("due_date = ?")
        params.append(payload.due_date.strip() if payload.due_date else None)

    if payload.management_blockers is not None:
        update_clauses.append("management_blockers = ?")
        params.append(payload.management_blockers.strip() if payload.management_blockers else None)

    # Status transition & lifecycle timestamp handling
    new_status = payload.status.value if hasattr(payload.status, "value") else (str(payload.status) if payload.status is not None else None)
    if new_status is not None:
        update_clauses.append("status = ?")
        params.append(new_status)

        if new_status != old_status:
            # 1. Initiated/Started: Moving from 'Not Started' to an active state
            if new_status != TaskStatus.NOT_STARTED.value and not old_started_at:
                update_clauses.append("started_at = ?")
                params.append(now_iso)

            # 2. Completed: Entering 'Completed'
            if new_status == TaskStatus.COMPLETED.value:
                update_clauses.append("completed_at = ?")
                params.append(now_iso)
            # 3. Reopened: Re-opening an item from 'Completed' back to an active or pending state
            elif old_status == TaskStatus.COMPLETED.value and new_status != TaskStatus.COMPLETED.value:
                update_clauses.append("completed_at = NULL")

            # 4. Record transition in task_activity_log
            actor = payload.changed_by or old_assigned
            cursor.execute(
                """
                INSERT INTO task_activity_log (task_id, from_status, to_status, changed_at, user_id)
                VALUES (?, ?, ?, ?, ?);
                """,
                (task_id, old_status, new_status, now_iso, str(actor)),
            )

    update_clauses.append("updated_at = ?")
    params.append(now_iso)

    params.append(task_id)

    cursor.execute(
        f"UPDATE tasks SET {', '.join(update_clauses)} WHERE task_id = ?;",
        params,
    )
    conn.commit()

    cursor.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,))
    row = cursor.fetchone()
    conn.close()

    return row_to_task(row)


@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
def delete_task(task_id: int):
    """Delete a task by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE task_id = ?;", (task_id,))
    conn.commit()
    conn.close()
    return None


@app.get("/api/crm-deals", tags=["Tasks"])
def get_crm_deals():
    """Returns all available CRM deals with customer names for dropdowns and linking."""
    if not CRM_DB_FILE.exists():
        return []
    try:
        with sqlite3.connect(CRM_DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT d.deal_id, d.deal_name, d.stage, d.assigned_presales, c.company_name as customer_name
                FROM deals d
                JOIN customers c ON d.customer_id = c.customer_id
                ORDER BY d.deal_id DESC;
            """).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching CRM deals: {e}")
        return []



# -----------------------------------------------------------------------------
# Embedded Monday.com-Style Task Board Dashboard
# -----------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monday-Style Presales Task Board</title>
    <!-- Google Fonts & Bootstrap 5 -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        :root {
            --monday-blue: #0073ea;
            --monday-hover: #0060b9;
            --monday-bg: #f5f6f8;
            --monday-card: #ffffff;
            --monday-border: #d0d4e4;
            --color-completed: #00c875;
            --color-inprogress: #fdab3d;
            --color-waiting: #784bd1;
            --color-review: #579bfc;
            --color-notstarted: #c4c4c4;
            --color-high: #e2445c;
            --color-medium: #fdab3d;
            --color-low: #579bfc;
        }

        body {
            background-color: var(--monday-bg);
            font-family: 'Figtree', system-ui, -apple-system, sans-serif;
            color: #323338;
        }

        .board-header {
            background: #ffffff;
            border-bottom: 1px solid var(--monday-border);
            padding: 1rem 2rem;
        }

        .board-title {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .monday-pill {
            font-size: 0.82rem;
            font-weight: 600;
            color: #fff;
            padding: 6px 14px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: opacity 0.15s;
            user-select: none;
            min-width: 130px;
        }
        .monday-pill:hover {
            opacity: 0.9;
        }

        .status-Completed { background-color: var(--color-completed); }
        .status-In-Progress { background-color: var(--color-inprogress); }
        .status-Waiting-on-Vendor { background-color: var(--color-waiting); }
        .status-Pending-Review { background-color: var(--color-review); }
        .status-Not-Started { background-color: var(--color-notstarted); }

        .priority-High { background-color: var(--color-high); }
        .priority-Medium { background-color: var(--color-medium); }
        .priority-Low { background-color: var(--color-low); }

        .group-section {
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid var(--monday-border);
            margin-bottom: 2rem;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        .group-header {
            padding: 12px 18px;
            font-weight: 700;
            font-size: 1.05rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #f0f3f6;
        }

        .group-RFP_OWNERSHIP .group-header {
            border-left: 6px solid #579bfc;
            color: #0051b3;
        }
        .group-RFP_DISTRIBUTED_SCOPE .group-header {
            border-left: 6px solid #a25ddc;
            color: #6323a6;
        }
        .group-GENERAL_ACTION .group-header {
            border-left: 6px solid #00c875;
            color: #0b804d;
        }

        .task-table {
            margin-bottom: 0;
            font-size: 0.9rem;
        }

        .task-table th {
            background-color: #f6f7fb;
            color: #676879;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            border-bottom: 1px solid #e1e5ee;
            padding: 8px 12px;
            text-align: center;
        }

        .task-table th.text-start {
            text-align: left;
        }

        .task-table td {
            vertical-align: middle;
            padding: 8px 12px;
            border-bottom: 1px solid #eceff5;
            text-align: center;
        }

        .task-table td.text-start {
            text-align: left;
        }

        .task-table tr:hover {
            background-color: #f9fbfd;
        }

        .avatar-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #eef2f8;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.82rem;
            font-weight: 600;
            color: #334155;
        }

        .deal-tag {
            font-size: 0.75rem;
            background: #e2e8f0;
            color: #475569;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }

        .customer-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            max-width: 175px;
            white-space: normal;
            word-break: break-word;
            text-align: left;
            line-height: 1.25;
        }

        .deal-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: #f8fafc;
            color: #334155;
            border: 1px solid #e2e8f0;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 500;
            max-width: 185px;
            white-space: normal;
            word-break: break-word;
            text-align: left;
            line-height: 1.25;
        }

        .blocker-chip {

            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            background: #fee2e2;
            color: #b91c1c;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            display: inline-block;
        }

        .filter-btn.active {
            background-color: var(--monday-blue);
            color: #fff;
            border-color: var(--monday-blue);
        }

        .progress-bar-segment {
            height: 8px;
            transition: width 0.3s;
        }
    </style>
</head>
<body>

    <!-- Top Monday.com Board Header -->
    <header class="board-header">
        <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
            <div>
                <div class="board-title">
                    <i class="bi bi-kanban-fill text-primary"></i>
                    <span>Presales Workstream & RFP Execution Board</span>
                    <span class="badge bg-secondary-subtle text-dark border ms-2">tasks.db</span>
                </div>
                <div class="text-muted small">Monday.com-style tracking for RFP Ownership, Distributed Scopes & Technical Actions</div>
            </div>
            <div class="d-flex align-items-center gap-2">
                <a href="/docs" target="_blank" class="btn btn-sm btn-outline-secondary">
                    <i class="bi bi-file-earmark-code me-1"></i>Swagger Docs
                </a>
                <button class="btn btn-sm btn-primary d-flex align-items-center gap-1" onclick="openNewTaskModal()">
                    <i class="bi bi-plus-lg"></i> New Item
                </button>
            </div>
        </div>

        <!-- Filter Toolbar -->
        <div class="d-flex flex-wrap align-items-center justify-content-between gap-3 mt-3 pt-2 border-top">
            <div class="btn-group btn-group-sm flex-wrap" role="group">
                <button type="button" class="btn btn-outline-secondary filter-btn active" onclick="setFilter('ALL', this)">All Items</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('CREATED_TODAY', this)"><i class="bi bi-calendar-plus me-1 text-primary"></i>Created Today</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('COMPLETED_WEEK', this)"><i class="bi bi-calendar-check me-1 text-success"></i>Completed This Week</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('RFP_ONLY', this)">RFP Workstream</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('PRESALES_1', this)">Presales 1</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('PRESALES_2', this)">Presales 2</button>
                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setFilter('ACTIVE_ONLY', this)">Active Only</button>
            </div>

            <div class="d-flex align-items-center gap-2">
                <div class="input-group input-group-sm" style="width: 250px;">
                    <span class="input-group-text bg-white"><i class="bi bi-search"></i></span>
                    <input type="text" id="searchInput" class="form-control" placeholder="Search tasks, blockers..." oninput="renderBoard()">
                </div>
                <button class="btn btn-sm btn-light border" onclick="loadTasks()" title="Reload Data">
                    <i class="bi bi-arrow-clockwise"></i>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Board Area -->
    <main class="container-fluid px-4 py-4">

        <!-- Velocity & Performance Metrics Strip -->
        <div class="row g-3 mb-4">
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #0073ea !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Created Today</div>
                    <div class="fs-4 fw-bold text-dark mt-1" id="metricCreatedToday">0</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Items logged today</div>
                </div>
            </div>
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #00c875 !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Completed Today</div>
                    <div class="fs-4 fw-bold text-success mt-1" id="metricCompletedToday">0</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Finished today</div>
                </div>
            </div>
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #10b981 !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Completed (7d)</div>
                    <div class="fs-4 fw-bold text-success mt-1" id="metricCompletedWeek">0</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Weekly velocity</div>
                </div>
            </div>
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #fdab3d !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Active In-Flight</div>
                    <div class="fs-4 fw-bold text-warning mt-1" id="metricInFlight">0</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Started / working</div>
                </div>
            </div>
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #784bd1 !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Avg Cycle Time</div>
                    <div class="fs-4 fw-bold mt-1" style="color: #784bd1;" id="metricAvgCycle">-</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Started &rarr; Completed</div>
                </div>
            </div>
            <div class="col-6 col-md-2">
                <div class="card border-0 shadow-sm p-3 bg-white h-100" style="border-radius: 8px; border-left: 4px solid #579bfc !important;">
                    <div class="text-muted small text-uppercase fw-semibold">Avg Lead Time</div>
                    <div class="fs-4 fw-bold text-primary mt-1" id="metricAvgLead">-</div>
                    <div class="text-muted" style="font-size: 0.72rem;">Created &rarr; Completed</div>
                </div>
            </div>
        </div>

        <!-- Macro Progress KPI Tracker -->
        <div class="card mb-4 border-0 shadow-sm p-3 bg-white" style="border-radius: 8px;">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="fw-bold small text-uppercase text-muted">Overall Workstream Completion</span>
                <span class="fw-bold small" id="completionRateLabel">0%</span>
            </div>
            <div class="progress" style="height: 10px;" id="progressBarContainer">
                <div class="progress-bar bg-success" id="progressCompleted" style="width: 0%"></div>
                <div class="progress-bar bg-warning" id="progressInProgress" style="width: 0%"></div>
                <div class="progress-bar bg-primary" id="progressReview" style="width: 0%"></div>
                <div class="progress-bar" style="width: 0%; background-color: var(--color-waiting);" id="progressWaiting"></div>
            </div>
        </div>

        <!-- Groups Container -->
        <div id="groupsContainer">
            <div class="text-center py-5 text-muted">
                <div class="spinner-border text-primary spinner-border-sm me-2"></div>Loading Task Board...
            </div>
        </div>

    </main>

    <!-- Create Task Modal -->
    <div class="modal fade" id="newTaskModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content" style="border-radius: 12px;">
                <form id="newTaskForm" onsubmit="submitNewTask(event)">
                    <div class="modal-header border-0 pb-0">
                        <h5 class="modal-title fw-bold"><i class="bi bi-plus-circle text-primary me-2"></i>Create New Board Item</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <div class="row g-3">
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Task / Item Title *</label>
                                <input type="text" id="newTitle" class="form-control" placeholder="e.g. Complete Technical RFP Section 3.2" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Workstream Category *</label>
                                <select id="newCategory" class="form-select" required>
                                    <option value="RFP_OWNERSHIP">RFP Ownership</option>
                                    <option value="RFP_DISTRIBUTED_SCOPE">RFP Distributed Scope</option>
                                    <option value="GENERAL_ACTION">General Action</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Assigned Presales Lead *</label>
                                <select id="newAssigned" class="form-select" required>
                                    <option value="Presales 1">Presales 1</option>
                                    <option value="Presales 2">Presales 2</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Vendor Domain</label>
                                <select id="newVendor" class="form-select">
                                    <option value="General">General</option>
                                    <option value="HPE">HPE</option>
                                    <option value="Veeam">Veeam</option>
                                    <option value="Dell">Dell</option>
                                    <option value="Nutanix">Nutanix</option>
                                    <option value="VMware">VMware</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Status *</label>
                                <select id="newStatus" class="form-select" required>
                                    <option value="Not Started">Not Started</option>
                                    <option value="In Progress">In Progress</option>
                                    <option value="Waiting on Vendor">Waiting on Vendor</option>
                                    <option value="Pending Review">Pending Review</option>
                                    <option value="Completed">Completed</option>
                                </select>
                            </div>
                            <div class="col-md-12">
                                <label class="form-label small fw-semibold"><i class="bi bi-link-45deg me-1 text-primary"></i>Link to CRM Deal & Customer (Optional)</label>
                                <select id="newDealSelect" class="form-select" onchange="onDealSelectChanged('new')">
                                    <option value="">-- Standalone / Unlinked Task --</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Customer / Organization Name</label>
                                <input type="text" id="newCustomerName" class="form-control" placeholder="e.g. Ministry of Foreign Affairs">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Deal / Opportunity Name</label>
                                <input type="text" id="newDealName" class="form-control" placeholder="e.g. Nutanix HCI Infrastructure Tender">
                            </div>
                            <input type="hidden" id="newDealId">
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Due Date</label>
                                <input type="date" id="newDueDate" class="form-control">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Priority *</label>
                                <select id="newPriority" class="form-select" required>
                                    <option value="High">High</option>
                                    <option value="Medium" selected>Medium</option>
                                    <option value="Low">Low</option>
                                </select>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Management / Escalation Blockers</label>
                                <textarea id="newBlockers" class="form-control" rows="2" placeholder="Note any vendor delays, sizing dependencies, or management help required..."></textarea>
                            </div>

                        </div>
                    </div>
                    <div class="modal-footer border-0 pt-0">
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary px-4"><i class="bi bi-check-lg me-1"></i>Add Item</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Edit Task Modal -->
    <div class="modal fade" id="editTaskModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content" style="border-radius: 12px;">
                <form id="editTaskForm" onsubmit="submitEditTask(event)">
                    <div class="modal-header border-0 pb-0">
                        <h5 class="modal-title fw-bold"><i class="bi bi-pencil-square text-primary me-2"></i>Update Task Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <input type="hidden" id="editTaskId">
                        <div class="row g-3">
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Task Title *</label>
                                <input type="text" id="editTitle" class="form-control" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Category *</label>
                                <select id="editCategory" class="form-select" required>
                                    <option value="RFP_OWNERSHIP">RFP Ownership</option>
                                    <option value="RFP_DISTRIBUTED_SCOPE">RFP Distributed Scope</option>
                                    <option value="GENERAL_ACTION">General Action</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Assigned To *</label>
                                <select id="editAssigned" class="form-select" required>
                                    <option value="Presales 1">Presales 1</option>
                                    <option value="Presales 2">Presales 2</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Vendor Domain</label>
                                <select id="editVendor" class="form-select">
                                    <option value="General">General</option>
                                    <option value="HPE">HPE</option>
                                    <option value="Veeam">Veeam</option>
                                    <option value="Dell">Dell</option>
                                    <option value="Nutanix">Nutanix</option>
                                    <option value="VMware">VMware</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Status *</label>
                                <select id="editStatus" class="form-select" required>
                                    <option value="Not Started">Not Started</option>
                                    <option value="In Progress">In Progress</option>
                                    <option value="Waiting on Vendor">Waiting on Vendor</option>
                                    <option value="Pending Review">Pending Review</option>
                                    <option value="Completed">Completed</option>
                                </select>
                            </div>
                            <div class="col-md-12">
                                <label class="form-label small fw-semibold"><i class="bi bi-link-45deg me-1 text-primary"></i>Link to CRM Deal & Customer</label>
                                <select id="editDealSelect" class="form-select" onchange="onDealSelectChanged('edit')">
                                    <option value="">-- Standalone / Custom --</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Customer / Organization Name</label>
                                <input type="text" id="editCustomerName" class="form-control">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Deal / Opportunity Name</label>
                                <input type="text" id="editDealName" class="form-control">
                            </div>
                            <input type="hidden" id="editDealId">
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Due Date</label>
                                <input type="date" id="editDueDate" class="form-control">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Priority *</label>
                                <select id="editPriority" class="form-select" required>
                                    <option value="High">High</option>
                                    <option value="Medium">Medium</option>
                                    <option value="Low">Low</option>
                                </select>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Management Blockers</label>
                                <textarea id="editBlockers" class="form-control" rows="3"></textarea>
                            </div>

                        </div>
                    </div>
                    <div class="modal-footer border-0 pt-0">
                        <button type="button" class="btn btn-outline-danger me-auto" onclick="deleteCurrentTask()"><i class="bi bi-trash me-1"></i>Delete</button>
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary"><i class="bi bi-save me-1"></i>Save Changes</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Task Lifecycle & Audit History Modal -->
    <div class="modal fade" id="historyModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
                <div class="modal-header border-0 pb-0">
                    <div>
                        <h5 class="modal-title fw-bold">
                            <i class="bi bi-clock-history text-primary me-2"></i>Task Lifecycle Audit Log
                        </h5>
                        <div class="text-muted small" id="historyTaskTitle">Item Transitions</div>
                    </div>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body p-4">
                    <div id="historyTimeline">
                        <div class="text-center py-3 text-muted">
                            <div class="spinner-border spinner-border-sm text-primary me-2"></div>Loading audit trail...
                        </div>
                    </div>
                </div>
                <div class="modal-footer border-0 pt-0">
                    <button type="button" class="btn btn-light btn-sm" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap Bundle JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let allTasks = [];
        let crmDeals = [];
        let currentFilter = 'ALL';
        const newModal = new bootstrap.Modal(document.getElementById('newTaskModal'));
        const editModal = new bootstrap.Modal(document.getElementById('editTaskModal'));
        const historyModal = new bootstrap.Modal(document.getElementById('historyModal'));

        const STATUS_FLOW = ["Not Started", "In Progress", "Waiting on Vendor", "Pending Review", "Completed"];
        const PRIORITY_FLOW = ["Low", "Medium", "High"];

        document.addEventListener('DOMContentLoaded', () => {
            loadTasks();
            loadCrmDeals();
        });

        async function loadTasks() {
            try {
                const res = await fetch('/api/tasks');
                allTasks = await res.json();
                renderBoard();
                loadVelocityMetrics();
            } catch (err) {
                console.error("Failed to load tasks:", err);
            }
        }

        async function loadCrmDeals() {
            try {
                const res = await fetch('/api/crm-deals');
                if (res.ok) {
                    crmDeals = await res.json();
                    populateDealDropdown('newDealSelect');
                    populateDealDropdown('editDealSelect');
                }
            } catch (err) {
                console.error("Failed to load CRM deals:", err);
            }
        }

        function populateDealDropdown(selectId) {
            const sel = document.getElementById(selectId);
            if (!sel) return;
            const currentVal = sel.value;
            sel.innerHTML = '<option value="">-- Standalone / Custom --</option>' +
                crmDeals.map(d => `<option value="${d.deal_id}">Deal #${d.deal_id}: ${d.customer_name} — ${d.deal_name}</option>`).join('');
            if (currentVal) sel.value = currentVal;
        }

        function onDealSelectChanged(prefix) {
            const sel = document.getElementById(prefix + 'DealSelect');
            const did = sel.value ? parseInt(sel.value) : null;
            document.getElementById(prefix + 'DealId').value = did || '';
            if (did) {
                const found = crmDeals.find(d => d.deal_id === did);
                if (found) {
                    document.getElementById(prefix + 'CustomerName').value = found.customer_name || '';
                    document.getElementById(prefix + 'DealName').value = found.deal_name || '';
                }
            }
        }

        async function loadVelocityMetrics() {
            try {
                const res = await fetch('/api/tasks/metrics/velocity');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('metricCreatedToday').textContent = data.created_today ?? 0;
                    document.getElementById('metricCompletedToday').textContent = data.completed_today ?? 0;
                    document.getElementById('metricCompletedWeek').textContent = data.completed_this_week ?? 0;
                    document.getElementById('metricInFlight').textContent = data.in_flight ?? 0;
                    document.getElementById('metricAvgCycle').textContent = data.avg_cycle_time || '-';
                    document.getElementById('metricAvgLead').textContent = data.avg_lead_time || '-';
                }
            } catch (err) {
                console.error("Failed to load velocity metrics:", err);
            }
        }

        function setFilter(filterType, btn) {
            currentFilter = filterType;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            renderBoard();
        }

        function renderBoard() {
            const search = document.getElementById('searchInput').value.toLowerCase().trim();

            const filtered = allTasks.filter(t => {
                const matchesSearch = !search ||
                    t.task_title.toLowerCase().includes(search) ||
                    (t.customer_name && t.customer_name.toLowerCase().includes(search)) ||
                    (t.deal_name && t.deal_name.toLowerCase().includes(search)) ||
                    (t.management_blockers && t.management_blockers.toLowerCase().includes(search)) ||
                    t.vendor_domain.toLowerCase().includes(search) ||
                    t.assigned_to.toLowerCase().includes(search);


                let matchesFilter = true;
                if (currentFilter === 'RFP_ONLY') {
                    matchesFilter = (t.category === 'RFP_OWNERSHIP' || t.category === 'RFP_DISTRIBUTED_SCOPE');
                } else if (currentFilter === 'PRESALES_1') {
                    matchesFilter = t.assigned_to === 'Presales 1';
                } else if (currentFilter === 'PRESALES_2') {
                    matchesFilter = t.assigned_to === 'Presales 2';
                } else if (currentFilter === 'ACTIVE_ONLY') {
                    matchesFilter = t.status !== 'Completed';
                } else if (currentFilter === 'CREATED_TODAY') {
                    const todayStr = new Date().toISOString().substring(0, 10);
                    matchesFilter = t.created_at && t.created_at.startsWith(todayStr);
                } else if (currentFilter === 'COMPLETED_WEEK') {
                    if (t.status !== 'Completed' || !t.completed_at) {
                        matchesFilter = false;
                    } else {
                        const compDate = new Date(t.completed_at.replace(' ', 'T'));
                        const weekAgo = new Date();
                        weekAgo.setDate(weekAgo.getDate() - 7);
                        matchesFilter = compDate >= weekAgo;
                    }
                }

                return matchesSearch && matchesFilter;
            });

            updateMacroProgress(allTasks);

            const groups = {
                'RFP_OWNERSHIP': {
                    title: 'RFP Ownership & Prime Proposals',
                    color: '#0073ea',
                    tasks: filtered.filter(t => t.category === 'RFP_OWNERSHIP')
                },
                'RFP_DISTRIBUTED_SCOPE': {
                    title: 'RFP Distributed Vendor Scope Items',
                    color: '#a25ddc',
                    tasks: filtered.filter(t => t.category === 'RFP_DISTRIBUTED_SCOPE')
                },
                'GENERAL_ACTION': {
                    title: 'General Technical Actions & PoC Tasks',
                    color: '#00c875',
                    tasks: filtered.filter(t => t.category === 'GENERAL_ACTION')
                }
            };

            const container = document.getElementById('groupsContainer');
            container.innerHTML = Object.entries(groups).map(([catKey, group]) => `
                <div class="group-section group-${catKey}">
                    <div class="group-header">
                        <div class="d-flex align-items-center gap-2">
                            <span>${group.title}</span>
                            <span class="badge bg-light text-dark border">${group.tasks.length} items</span>
                        </div>
                        <button class="btn btn-sm btn-link text-decoration-none p-0" onclick="openNewTaskModal('${catKey}')">
                            <i class="bi bi-plus-circle me-1"></i>Add Item
                        </button>
                    </div>

                    <div class="table-responsive">
                        <table class="table task-table">
                            <thead>
                                <tr>
                                    <th class="text-start" style="width: 22%;">Item Title</th>
                                    <th style="width: 14%;">Customer</th>
                                    <th style="width: 14%;">Deal / Tender</th>
                                    <th style="width: 10%;">Assignee</th>
                                    <th style="width: 11%;">Status</th>
                                    <th style="width: 7%;">Priority</th>
                                    <th style="width: 7%;">Vendor</th>
                                    <th style="width: 10%;">Timing & Velocity</th>
                                    <th style="width: 5%;">Due</th>
                                    <th style="width: 4%;">Log</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${group.tasks.length === 0 ? `
                                    <tr>
                                        <td colspan="10" class="text-center py-4 text-muted fst-italic">
                                            No tasks in this workstream matching filters.
                                        </td>
                                    </tr>
                                ` : group.tasks.map(t => renderTaskRow(t)).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `).join('');
        }

        function renderTaskRow(t) {
            const statusClass = 'status-' + t.status.replace(/\\s+/g, '-');
            const priorityClass = 'priority-' + t.priority;

            const customerDisplay = t.customer_name 
                ? `<span class="customer-badge" title="${t.customer_name}"><i class="bi bi-building text-primary"></i>${t.customer_name}</span>`
                : '<span class="text-muted small">—</span>';

            const dealText = t.deal_name || (t.related_deal_id ? `Deal #${t.related_deal_id}` : '');
            const dealDisplay = dealText 
                ? `<span class="deal-badge" title="${dealText}"><i class="bi bi-briefcase text-secondary"></i>${dealText}</span>`
                : '<span class="text-muted small">—</span>';

            const blockerDisplay = t.management_blockers 
                ? `<div class="mt-1"><span class="blocker-chip" title="${t.management_blockers}" onclick="openEditTaskModal(${t.task_id})"><i class="bi bi-exclamation-triangle-fill me-1"></i>${t.management_blockers}</span></div>`
                : '';

            let velocityBadge = '';
            if (t.status === 'Completed') {
                const cycle = t.cycle_time || '-';
                const lead = t.lead_time || '-';
                velocityBadge = `
                    <div class="d-flex flex-column gap-1 align-items-center">
                        <span class="badge bg-success-subtle text-success border border-success-subtle py-1 px-2" style="font-size: 0.72rem;" title="Cycle Time (Working duration)">
                            <i class="bi bi-stopwatch me-1"></i>Cycle: ${cycle}
                        </span>
                        <span class="badge bg-light text-muted border py-1 px-2" style="font-size: 0.70rem;" title="Lead Time (Total turnaround)">
                            <i class="bi bi-flag me-1"></i>Lead: ${lead}
                        </span>
                    </div>
                `;
            } else if (t.started_at) {
                const startedDate = t.started_at.substring(0, 10);
                velocityBadge = `
                    <span class="badge bg-warning-subtle text-warning-emphasis border border-warning-subtle py-1 px-2" style="font-size: 0.72rem;" title="Initiated at ${t.started_at}">
                        <i class="bi bi-play-fill text-warning me-1"></i>Started ${startedDate}
                    </span>
                `;
            } else {
                const createdDate = t.created_at.substring(0, 10);
                velocityBadge = `
                    <span class="badge bg-light text-secondary border py-1 px-2" style="font-size: 0.72rem;" title="Created at ${t.created_at}">
                        <i class="bi bi-hourglass me-1"></i>Queued ${createdDate}
                    </span>
                `;
            }

            return `
                <tr>
                    <td class="text-start">
                        <a href="javascript:void(0)" class="fw-semibold text-dark text-decoration-none" onclick="openEditTaskModal(${t.task_id})">
                            ${t.task_title}
                        </a>
                        ${blockerDisplay}
                    </td>
                    <td>${customerDisplay}</td>
                    <td>${dealDisplay}</td>
                    <td>
                        <span class="avatar-badge">
                            <i class="bi bi-person-fill text-primary"></i>${t.assigned_to}
                        </span>
                    </td>
                    <td>
                        <div class="dropdown">
                            <span class="monday-pill ${statusClass} dropdown-toggle" data-bs-toggle="dropdown">
                                ${t.status}
                            </span>
                            <ul class="dropdown-menu shadow-sm">
                                ${STATUS_FLOW.map(st => `
                                    <li><a class="dropdown-item small" href="javascript:void(0)" onclick="quickUpdateStatus(${t.task_id}, '${st}')">${st}</a></li>
                                `).join('')}
                            </ul>
                        </div>
                    </td>
                    <td>
                        <span class="monday-pill ${priorityClass}" style="min-width: 75px;" onclick="cyclePriority(${t.task_id}, '${t.priority}')" title="Click to cycle priority">
                            ${t.priority}
                        </span>
                    </td>
                    <td>
                        <span class="badge bg-light text-secondary border px-2 py-1">${t.vendor_domain}</span>
                    </td>
                    <td>
                        ${velocityBadge}
                    </td>
                    <td>
                        <span class="small ${isOverdue(t.due_date, t.status) ? 'text-danger fw-bold' : 'text-muted'}">
                            ${t.due_date ? t.due_date.substring(5) : '-'}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-secondary py-1 px-2 border-0" title="Lifecycle Audit Log" onclick="openHistoryModal(${t.task_id}, '${t.task_title.replace(/'/g, "\\'")}')">
                            <i class="bi bi-clock-history text-primary fs-6"></i>
                        </button>
                    </td>
                </tr>
            `;
        }


        function isOverdue(dateStr, status) {
            if (!dateStr || status === 'Completed') return false;
            const due = new Date(dateStr);
            const now = new Date();
            now.setHours(0,0,0,0);
            return due < now;
        }

        function updateMacroProgress(tasks) {
            if (!tasks.length) return;
            const completed = tasks.filter(t => t.status === 'Completed').length;
            const inProgress = tasks.filter(t => t.status === 'In Progress').length;
            const review = tasks.filter(t => t.status === 'Pending Review').length;
            const waiting = tasks.filter(t => t.status === 'Waiting on Vendor').length;

            const total = tasks.length;
            const pct = Math.round((completed / total) * 100);

            document.getElementById('completionRateLabel').textContent = `${pct}% (${completed}/${total} completed)`;
            document.getElementById('progressCompleted').style.width = `${(completed / total) * 100}%`;
            document.getElementById('progressInProgress').style.width = `${(inProgress / total) * 100}%`;
            document.getElementById('progressReview').style.width = `${(review / total) * 100}%`;
            document.getElementById('progressWaiting').style.width = `${(waiting / total) * 100}%`;
        }

        async function quickUpdateStatus(taskId, newStatus) {
            try {
                const res = await fetch(`/api/tasks/${taskId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status: newStatus })
                });
                if (res.ok) {
                    await loadTasks();
                } else {
                    alert('Failed to update task status.');
                }
            } catch (err) {
                console.error("Error updating status:", err);
            }
        }

        async function cyclePriority(taskId, currentPriority) {
            const idx = PRIORITY_FLOW.indexOf(currentPriority);
            const nextPriority = PRIORITY_FLOW[(idx + 1) % PRIORITY_FLOW.length];
            try {
                const res = await fetch(`/api/tasks/${taskId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ priority: nextPriority })
                });
                if (res.ok) {
                    await loadTasks();
                }
            } catch (err) {
                console.error("Error cycling priority:", err);
            }
        }

        function openNewTaskModal(presetCategory) {
            document.getElementById('newTaskForm').reset();
            document.getElementById('newDealId').value = '';
            document.getElementById('newDealSelect').value = '';
            document.getElementById('newCustomerName').value = '';
            document.getElementById('newDealName').value = '';
            if (presetCategory) {
                document.getElementById('newCategory').value = presetCategory;
            }
            newModal.show();
        }

        async function submitNewTask(e) {
            e.preventDefault();
            const payload = {
                task_title: document.getElementById('newTitle').value,
                category: document.getElementById('newCategory').value,
                assigned_to: document.getElementById('newAssigned').value,
                vendor_domain: document.getElementById('newVendor').value,
                status: document.getElementById('newStatus').value,
                priority: document.getElementById('newPriority').value,
                due_date: document.getElementById('newDueDate').value || null,
                related_deal_id: parseInt(document.getElementById('newDealId').value) || null,
                customer_name: document.getElementById('newCustomerName').value.trim() || null,
                deal_name: document.getElementById('newDealName').value.trim() || null,
                management_blockers: document.getElementById('newBlockers').value || null
            };

            try {
                const res = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    newModal.hide();
                    await loadTasks();
                } else {
                    const err = await res.json();
                    alert('Error creating task: ' + JSON.stringify(err));
                }
            } catch (err) {
                console.error("Failed to submit task:", err);
            }
        }

        function openEditTaskModal(taskId) {
            const task = allTasks.find(t => t.task_id === taskId);
            if (!task) return;

            document.getElementById('editTaskId').value = task.task_id;
            document.getElementById('editTitle').value = task.task_title;
            document.getElementById('editCategory').value = task.category;
            document.getElementById('editAssigned').value = task.assigned_to;
            document.getElementById('editVendor').value = task.vendor_domain;
            document.getElementById('editStatus').value = task.status;
            document.getElementById('editPriority').value = task.priority;
            document.getElementById('editDueDate').value = task.due_date || '';
            document.getElementById('editDealId').value = task.related_deal_id || '';
            document.getElementById('editCustomerName').value = task.customer_name || '';
            document.getElementById('editDealName').value = task.deal_name || '';
            document.getElementById('editBlockers').value = task.management_blockers || '';

            const sel = document.getElementById('editDealSelect');
            if (sel) {
                sel.value = task.related_deal_id ? String(task.related_deal_id) : '';
            }

            editModal.show();
        }

        async function submitEditTask(e) {
            e.preventDefault();
            const taskId = document.getElementById('editTaskId').value;
            const payload = {
                task_title: document.getElementById('editTitle').value,
                category: document.getElementById('editCategory').value,
                assigned_to: document.getElementById('editAssigned').value,
                vendor_domain: document.getElementById('editVendor').value,
                status: document.getElementById('editStatus').value,
                priority: document.getElementById('editPriority').value,
                due_date: document.getElementById('editDueDate').value || null,
                related_deal_id: parseInt(document.getElementById('editDealId').value) || null,
                customer_name: document.getElementById('editCustomerName').value.trim() || null,
                deal_name: document.getElementById('editDealName').value.trim() || null,
                management_blockers: document.getElementById('editBlockers').value || null
            };


            try {
                const res = await fetch(`/api/tasks/${taskId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    editModal.hide();
                    await loadTasks();
                } else {
                    const err = await res.json();
                    alert('Error updating task: ' + JSON.stringify(err));
                }
            } catch (err) {
                console.error("Failed to update task:", err);
            }
        }

        async function deleteCurrentTask() {
            const taskId = document.getElementById('editTaskId').value;
            if (!confirm('Are you sure you want to delete this item?')) return;

            try {
                const res = await fetch(`/api/tasks/${taskId}`, { method: 'DELETE' });
                if (res.ok) {
                    editModal.hide();
                    await loadTasks();
                }
            } catch (err) {
                console.error("Failed to delete task:", err);
            }
        }

        async function openHistoryModal(taskId, taskTitle) {
            document.getElementById('historyTaskTitle').textContent = `Item #${taskId}: ${taskTitle}`;
            const timeline = document.getElementById('historyTimeline');
            timeline.innerHTML = '<div class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Loading audit trail...</div>';
            historyModal.show();

            try {
                const res = await fetch(`/api/tasks/${taskId}/history`);
                const logs = await res.json();
                if (!logs || !logs.length) {
                    timeline.innerHTML = '<div class="text-center text-muted py-4">No activity transitions recorded yet.</div>';
                    return;
                }

                timeline.innerHTML = `
                    <div class="timeline-container ps-3 border-start border-2 border-primary-subtle ms-2" style="position: relative;">
                        ${logs.map((log) => {
                            const fromBadge = log.from_status 
                                ? `<span class="badge bg-secondary-subtle text-dark border me-1">${log.from_status}</span>` 
                                : `<span class="badge bg-info-subtle text-info-emphasis border me-1">Created</span>`;
                            const toBadge = `<span class="badge bg-primary-subtle text-primary border">${log.to_status}</span>`;
                            const actorBadge = log.user_id 
                                ? `<span class="badge bg-light text-dark border ms-1"><i class="bi bi-person me-1 text-secondary"></i>${log.user_id}</span>` 
                                : '';
                            const timeStr = log.changed_at ? log.changed_at.replace('T', ' ').substring(0, 19) : '-';

                            return `
                                <div class="mb-3 position-relative ps-2">
                                    <div style="position: absolute; left: -21px; top: 4px; width: 10px; height: 10px; border-radius: 50%; background-color: var(--monday-blue); border: 2px solid #fff;"></div>
                                    <div class="d-flex align-items-center flex-wrap gap-1">
                                        ${fromBadge} <i class="bi bi-arrow-right text-muted small"></i> ${toBadge} ${actorBadge}
                                    </div>
                                    <div class="text-muted small mt-1" style="font-size: 0.78rem;">
                                        <i class="bi bi-clock me-1"></i>${timeStr}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            } catch (err) {
                timeline.innerHTML = `<div class="alert alert-danger py-2 small">Error loading transition log: ${err}</div>`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    """Serves the Monday.com-style visual Task Board."""
    return HTML_DASHBOARD


# -----------------------------------------------------------------------------
# Main Entry Point (Configured on Port 8001 to run concurrently with CRM app)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("task_board:app", host="127.0.0.1", port=8001, reload=True)
