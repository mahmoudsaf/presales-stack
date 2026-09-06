import os
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DB_FILE = Path(__file__).resolve().parent / "crm.db"

VALID_VENDORS = ["HPE", "Veeam", "Dell", "Nutanix", "VMware"]


class DealStage(str, Enum):
    DISCOVERY = "Discovery"
    GATHERING_REQUIREMENTS = "Gathering Requirements"
    RFP_TENDER = "RFP / Tender"
    POC = "PoC"
    PROPOSAL = "Proposal"
    CLOSED_WON = "Closed-Won"
    CLOSED_LOST = "Closed-Lost"


class PresalesRep(str, Enum):
    PRESALES_1 = "Presales 1"
    PRESALES_2 = "Presales 2"


# -----------------------------------------------------------------------------
# Database Helpers & Initialization
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
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT UNIQUE NOT NULL,
        contact_name TEXT,
        contact_email TEXT,
        contact_phone TEXT
    );
    """)

    # Schema migration check: ensure deals table includes 'RFP / Tender' in its CHECK constraint
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='deals';")
    deals_row = cursor.fetchone()
    if deals_row and "RFP / Tender" not in deals_row["sql"]:
        cursor.execute("PRAGMA foreign_keys = OFF;")
        cursor.execute("""
        CREATE TABLE deals_new (
            deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            deal_name TEXT NOT NULL,
            primary_vendors TEXT,
            stage TEXT NOT NULL CHECK(stage IN (
                'Discovery', 'Gathering Requirements', 'RFP / Tender', 'PoC', 'Proposal', 'Closed-Won', 'Closed-Lost'
            )),
            estimated_value REAL NOT NULL DEFAULT 0.0,
            assigned_presales TEXT CHECK(assigned_presales IN ('Presales 1', 'Presales 2')),
            vendor_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """)
        cursor.execute("INSERT INTO deals_new SELECT * FROM deals;")
        cursor.execute("DROP TABLE deals;")
        cursor.execute("ALTER TABLE deals_new RENAME TO deals;")
        cursor.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
    elif not deals_row:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS deals (
            deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            deal_name TEXT NOT NULL,
            primary_vendors TEXT,
            stage TEXT NOT NULL CHECK(stage IN (
                'Discovery', 'Gathering Requirements', 'RFP / Tender', 'PoC', 'Proposal', 'Closed-Won', 'Closed-Lost'
            )),
            estimated_value REAL NOT NULL DEFAULT 0.0,
            assigned_presales TEXT CHECK(assigned_presales IN ('Presales 1', 'Presales 2')),
            vendor_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """)

    # Seed demo data only if explicitly requested via environment variable
    if os.environ.get("SEED_DEMO_DATA", "").lower() == "true":
        cursor.execute("SELECT COUNT(*) FROM customers;")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                """
                INSERT INTO customers (company_name, contact_name, contact_email, contact_phone)
                VALUES (?, ?, ?, ?);
                """,
                [
                    ("Acme Cloud Corp", "Alice Smith", "alice@acmecloud.com", "+1-555-0192"),
                    ("FinTech Horizons", "Brian Miller", "bmiller@fintechhorizons.io", "+1-555-0341"),
                    ("Nordic Health Systems", "Clara Lind", "clara.lind@nordichealth.org", "+46-8-123-456"),
                    ("Apex Logistics", "David Vance", "dvance@apexlogistics.com", "+1-555-0872"),
                ],
            )

            sample_deals = [
                (
                    1,
                    "Hyperconverged Datacenter Refresh",
                    "HPE, Nutanix",
                    "PoC",
                    125000.0,
                    "Presales 1",
                    "Customer evaluating Nutanix AHV vs VMware migration on HPE ProLiant DX nodes.",
                ),
                (
                    2,
                    "Ransomware Backup Immutability",
                    "Veeam, Dell",
                    "Proposal",
                    68000.0,
                    "Presales 2",
                    "Veeam Data Platform Enterprise Plus paired with Dell PowerProtect DD backup appliances.",
                ),
                (
                    3,
                    "Edge Compute Cluster Expansion",
                    "Dell, VMware",
                    "Gathering Requirements",
                    45000.0,
                    "Presales 1",
                    "Assessing 4 remote hospital sites with dual-node clusters.",
                ),
                (
                    4,
                    "Enterprise Core Virtualization",
                    "VMware, HPE",
                    "Closed-Won",
                    195000.0,
                    "Presales 2",
                    "Contract executed for 3-year VMware Cloud Foundation subscription and HPE Alletra storage.",
                ),
            ]

            cursor.executemany(
                """
                INSERT INTO deals (
                    customer_id, deal_name, primary_vendors, stage, estimated_value, assigned_presales, vendor_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                sample_deals,
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
    title="Lightweight Local CRM",
    description="Local CRM with FastAPI, SQLite, and Presales Deal Tracking",
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
class CustomerBase(BaseModel):
    company_name: str = Field(..., description="Unique company name")
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


class CustomerOut(CustomerBase):
    customer_id: int
    total_deals: Optional[int] = 0
    total_pipeline: Optional[float] = 0.0


class CustomerCreate(CustomerBase):
    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Company name cannot be empty")
        return s


class CustomerUpdate(BaseModel):
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            s = v.strip()
            if not s:
                raise ValueError("Company name cannot be empty")
            return s
        return v


class DealCreate(BaseModel):
    # Customer identifier & contact details
    company_name: str = Field(..., description="Target company name (auto-created if not exists)")
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None

    # Deal details
    deal_name: str = Field(..., description="Descriptive deal or project name")
    primary_vendors: Optional[Union[List[str], str]] = Field(
        default=[], description="Vendor array or CSV string (e.g. HPE, Veeam, Dell, Nutanix, VMware)"
    )
    model_config = ConfigDict(extra="ignore")
    stage: Union[DealStage, str] = Field(default=DealStage.DISCOVERY, description="Current deal stage")
    estimated_value: Union[float, str] = Field(default=0.0, description="Estimated deal value in USD")
    assigned_presales: Union[PresalesRep, str] = Field(default=PresalesRep.PRESALES_1, description="Assigned presales engineer")
    vendor_notes: Optional[str] = Field(default="", description="Technical requirements, bill of materials, notes")

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Alias company_name (customer_name, customer, client, etc.)
            if not data.get("company_name"):
                for key in ["customer_name", "customer", "client_name", "client", "account_name", "account", "org_name", "organization"]:
                    if data.get(key) and str(data.get(key)).strip():
                        data["company_name"] = str(data.get(key)).strip()
                        break
            
            # Infer from deal_name if company_name is still missing
            deal_raw = data.get("deal_name") or data.get("opportunity_name") or data.get("project_name") or data.get("title") or data.get("name")
            if not data.get("company_name") and deal_raw:
                d_str = str(deal_raw).strip()
                m = re.search(r"(?:مشروع|عميل|شركة|مؤسسة|مناقصة)\s+([^\s\-:،,]+)", d_str)
                if m:
                    data["company_name"] = m.group(1).strip()
                else:
                    data["company_name"] = d_str[:30]
            elif not data.get("company_name"):
                data["company_name"] = "General Client"

            # 2. Alias deal_name
            if not data.get("deal_name"):
                for key in ["project_name", "opportunity_name", "title", "name", "tender_name", "deal"]:
                    if data.get(key) and str(data.get(key)).strip():
                        data["deal_name"] = str(data.get(key)).strip()
                        break
            if not data.get("deal_name"):
                data["deal_name"] = f"مشروع {data.get('company_name', 'العميل')}"

        return data

    @field_validator("stage", mode="before")
    @classmethod
    def normalize_stage(cls, v):
        if not v:
            return DealStage.DISCOVERY
        v_str = str(v).strip().lower()
        mapping = {
            "discovery": DealStage.DISCOVERY,
            "gathering requirements": DealStage.GATHERING_REQUIREMENTS,
            "requirements": DealStage.GATHERING_REQUIREMENTS,
            "in progress": DealStage.GATHERING_REQUIREMENTS,
            "ongoing": DealStage.GATHERING_REQUIREMENTS,
            "active": DealStage.GATHERING_REQUIREMENTS,
            "rfp / tender": DealStage.RFP_TENDER,
            "rfp/tender": DealStage.RFP_TENDER,
            "rfp or tender": DealStage.RFP_TENDER,
            "rfp ofr tender": DealStage.RFP_TENDER,
            "rfp": DealStage.RFP_TENDER,
            "tender": DealStage.RFP_TENDER,
            "rfq": DealStage.RFP_TENDER,
            "request for pricing": DealStage.RFP_TENDER,
            "direct request for pricing": DealStage.RFP_TENDER,
            "direct request for pricing request": DealStage.RFP_TENDER,
            "pricing request": DealStage.RFP_TENDER,
            "pricing": DealStage.RFP_TENDER,
            "طلب تسعير": DealStage.RFP_TENDER,
            "طلب تسعيرة": DealStage.RFP_TENDER,
            "تسعير": DealStage.RFP_TENDER,
            "تسعيرة": DealStage.RFP_TENDER,
            "استدراج عروض": DealStage.RFP_TENDER,
            "استدراج عروض أسعار": DealStage.RFP_TENDER,
            "مناقصة": DealStage.RFP_TENDER,
            "منافسة": DealStage.RFP_TENDER,
            "poc": DealStage.POC,
            "proof of concept": DealStage.POC,
            "proposal": DealStage.PROPOSAL,
            "closed-won": DealStage.CLOSED_WON,
            "closed won": DealStage.CLOSED_WON,
            "won": DealStage.CLOSED_WON,
            "closed-lost": DealStage.CLOSED_LOST,
            "closed lost": DealStage.CLOSED_LOST,
            "lost": DealStage.CLOSED_LOST,
        }
        return mapping.get(v_str, DealStage.DISCOVERY)

    @field_validator("assigned_presales", mode="before")
    @classmethod
    def normalize_presales(cls, v):
        if not v:
            return PresalesRep.PRESALES_1
        v_str = str(v).strip().lower()
        if "2" in v_str:
            return PresalesRep.PRESALES_2
        return PresalesRep.PRESALES_1

    @field_validator("estimated_value", mode="before")
    @classmethod
    def normalize_value(cls, v):
        if v is None or v == "":
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        v_str = str(v).strip()
        # Scale multipliers (Arabic & English)
        mult = 1.0
        if re.search(r"(?:مليار|billion|b\b)", v_str, re.I):
            mult = 1_000_000_000.0
        elif re.search(r"(?:مليون|ملايين|million|m\b)", v_str, re.I):
            mult = 1_000_000.0
        elif re.search(r"(?:ألف|الاف|آلاف|thousand|k\b)", v_str, re.I):
            mult = 1_000.0

        range_match = re.findall(r"(\d+(?:\.\d+)?)", v_str)
        if len(range_match) >= 2 and any(sep in v_str for sep in ["-", "إلى", "الى", "to"]):
            vals = [float(x) for x in range_match[:2]]
            avg_val = sum(vals) / len(vals)
            return round(avg_val * mult, 2)
        elif range_match:
            return round(float(range_match[0]) * mult, 2)

        clean = re.sub(r"[^\d.]", "", v_str)
        try:
            return float(clean) if clean else 0.0
        except Exception:
            return 0.0


class DealUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    deal_name: Optional[str] = None
    primary_vendors: Optional[Union[List[str], str]] = None
    stage: Optional[Union[DealStage, str]] = None
    estimated_value: Optional[Union[float, str]] = None
    assigned_presales: Optional[Union[PresalesRep, str]] = None
    vendor_notes: Optional[str] = None

    @field_validator("stage", mode="before")
    @classmethod
    def normalize_stage_update(cls, v):
        if v is None or v == "":
            return None
        v_str = str(v).strip().lower()
        mapping = {
            "discovery": DealStage.DISCOVERY,
            "gathering requirements": DealStage.GATHERING_REQUIREMENTS,
            "requirements": DealStage.GATHERING_REQUIREMENTS,
            "in progress": DealStage.GATHERING_REQUIREMENTS,
            "ongoing": DealStage.GATHERING_REQUIREMENTS,
            "active": DealStage.GATHERING_REQUIREMENTS,
            "rfp / tender": DealStage.RFP_TENDER,
            "rfp/tender": DealStage.RFP_TENDER,
            "rfp or tender": DealStage.RFP_TENDER,
            "rfp ofr tender": DealStage.RFP_TENDER,
            "rfp": DealStage.RFP_TENDER,
            "tender": DealStage.RFP_TENDER,
            "rfq": DealStage.RFP_TENDER,
            "request for pricing": DealStage.RFP_TENDER,
            "direct request for pricing": DealStage.RFP_TENDER,
            "direct request for pricing request": DealStage.RFP_TENDER,
            "pricing request": DealStage.RFP_TENDER,
            "pricing": DealStage.RFP_TENDER,
            "طلب تسعير": DealStage.RFP_TENDER,
            "طلب تسعيرة": DealStage.RFP_TENDER,
            "تسعير": DealStage.RFP_TENDER,
            "تسعيرة": DealStage.RFP_TENDER,
            "استدراج عروض": DealStage.RFP_TENDER,
            "استدراج عروض أسعار": DealStage.RFP_TENDER,
            "مناقصة": DealStage.RFP_TENDER,
            "منافسة": DealStage.RFP_TENDER,
            "poc": DealStage.POC,
            "proof of concept": DealStage.POC,
            "proposal": DealStage.PROPOSAL,
            "closed-won": DealStage.CLOSED_WON,
            "closed won": DealStage.CLOSED_WON,
            "won": DealStage.CLOSED_WON,
            "closed-lost": DealStage.CLOSED_LOST,
            "closed lost": DealStage.CLOSED_LOST,
            "lost": DealStage.CLOSED_LOST,
        }
        return mapping.get(v_str, DealStage.GATHERING_REQUIREMENTS)

    @field_validator("assigned_presales", mode="before")
    @classmethod
    def normalize_presales_update(cls, v):
        if v is None or v == "":
            return None
        v_str = str(v).strip().lower()
        if "2" in v_str:
            return PresalesRep.PRESALES_2
        return PresalesRep.PRESALES_1

    @field_validator("estimated_value", mode="before")
    @classmethod
    def normalize_value_update(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        v_str = str(v).strip()
        mult = 1.0
        if re.search(r"(?:مليار|billion|b\b)", v_str, re.I):
            mult = 1_000_000_000.0
        elif re.search(r"(?:مليون|ملايين|million|m\b)", v_str, re.I):
            mult = 1_000_000.0
        elif re.search(r"(?:ألف|الاف|آلاف|thousand|k\b)", v_str, re.I):
            mult = 1_000.0

        range_match = re.findall(r"(\d+(?:\.\d+)?)", v_str)
        if len(range_match) >= 2 and any(sep in v_str for sep in ["-", "إلى", "الى", "to"]):
            vals = [float(x) for x in range_match[:2]]
            avg_val = sum(vals) / len(vals)
            return round(avg_val * mult, 2)
        elif range_match:
            return round(float(range_match[0]) * mult, 2)

        clean = re.sub(r"[^\d.]", "", v_str)
        try:
            return float(clean) if clean else None
        except Exception:
            return None


class DealOut(BaseModel):
    deal_id: int
    customer_id: int
    company_name: str
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    deal_name: str
    primary_vendors: str
    stage: str
    estimated_value: float
    assigned_presales: Optional[str] = None
    vendor_notes: Optional[str] = None
    created_at: str
    updated_at: str


def normalize_vendors(vendors: Optional[Union[List[str], str]]) -> str:
    if not vendors:
        return ""
    if isinstance(vendors, list):
        clean_list = [v.strip() for v in vendors if v and v.strip()]
        return ", ".join(clean_list)
    return ", ".join([v.strip() for v in str(vendors).split(",") if v.strip()])


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/customers", response_model=List[CustomerOut], tags=["Customers"])
def get_customers():
    """Returns list of customers along with aggregated deal count and pipeline value."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            c.customer_id,
            c.company_name,
            c.contact_name,
            c.contact_email,
            c.contact_phone,
            COUNT(d.deal_id) AS total_deals,
            COALESCE(SUM(d.estimated_value), 0.0) AS total_pipeline
        FROM customers c
        LEFT JOIN deals d ON c.customer_id = d.customer_id
        GROUP BY c.customer_id
        ORDER BY c.company_name COLLATE NOCASE ASC;
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        CustomerOut(
            customer_id=row["customer_id"],
            company_name=row["company_name"],
            contact_name=row["contact_name"],
            contact_email=row["contact_email"],
            contact_phone=row["contact_phone"],
            total_deals=row["total_deals"],
            total_pipeline=row["total_pipeline"],
        )
        for row in rows
    ]


@app.get("/api/customers/{customer_id}", response_model=CustomerOut, tags=["Customers"])
def get_customer(customer_id: int):
    """Returns details of a single customer account with aggregated deal and pipeline metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            c.customer_id,
            c.company_name,
            c.contact_name,
            c.contact_email,
            c.contact_phone,
            COUNT(d.deal_id) AS total_deals,
            COALESCE(SUM(d.estimated_value), 0.0) AS total_pipeline
        FROM customers c
        LEFT JOIN deals d ON c.customer_id = d.customer_id
        WHERE c.customer_id = ?
        GROUP BY c.customer_id;
    """, (customer_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer #{customer_id} not found")

    return CustomerOut(
        customer_id=row["customer_id"],
        company_name=row["company_name"],
        contact_name=row["contact_name"],
        contact_email=row["contact_email"],
        contact_phone=row["contact_phone"],
        total_deals=row["total_deals"],
        total_pipeline=row["total_pipeline"],
    )


@app.post("/api/customers", response_model=CustomerOut, status_code=status.HTTP_201_CREATED, tags=["Customers"])
def create_customer(payload: CustomerCreate):
    """
    Creates a new customer account directly.
    Checks for duplicate company name (case-insensitive).
    """
    clean_company = payload.company_name.strip()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT customer_id FROM customers WHERE LOWER(company_name) = LOWER(?);",
        (clean_company,),
    )
    existing = cursor.fetchone()
    if existing:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Customer account with company name '{clean_company}' already exists (ID #{existing['customer_id']})."
        )

    cursor.execute(
        """
        INSERT INTO customers (company_name, contact_name, contact_email, contact_phone)
        VALUES (?, ?, ?, ?);
        """,
        (clean_company, payload.contact_name, payload.contact_email, payload.contact_phone),
    )
    customer_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return CustomerOut(
        customer_id=customer_id,
        company_name=clean_company,
        contact_name=payload.contact_name,
        contact_email=payload.contact_email,
        contact_phone=payload.contact_phone,
        total_deals=0,
        total_pipeline=0.0,
    )


@app.put("/api/customers/{customer_id}", response_model=CustomerOut, tags=["Customers"])
def update_customer(customer_id: int, payload: CustomerUpdate):
    """
    Updates an existing customer account.
    If company_name is updated, checks for collisions and keeps tasks.db cached customer_name in sync.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT customer_id, company_name, contact_name, contact_email, contact_phone FROM customers WHERE customer_id = ?;", (customer_id,))
    existing = cursor.fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer #{customer_id} not found")

    old_company_name = existing["company_name"]
    updates = []
    params = []

    new_company_name = old_company_name
    if payload.company_name is not None:
        clean_name = payload.company_name.strip()
        if clean_name.lower() != old_company_name.lower():
            cursor.execute("SELECT customer_id FROM customers WHERE LOWER(company_name) = LOWER(?) AND customer_id != ?;", (clean_name, customer_id))
            if cursor.fetchone():
                conn.close()
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Another customer with company name '{clean_name}' already exists.")
        updates.append("company_name = ?")
        params.append(clean_name)
        new_company_name = clean_name

    if payload.contact_name is not None:
        updates.append("contact_name = ?")
        params.append(payload.contact_name.strip() if payload.contact_name else None)
    if payload.contact_email is not None:
        updates.append("contact_email = ?")
        params.append(payload.contact_email.strip() if payload.contact_email else None)
    if payload.contact_phone is not None:
        updates.append("contact_phone = ?")
        params.append(payload.contact_phone.strip() if payload.contact_phone else None)

    if updates:
        params.append(customer_id)
        cursor.execute(f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?;", params)
        conn.commit()

    # Sync cached customer_name in tasks.db if company_name changed
    if new_company_name != old_company_name:
        try:
            tasks_db_path = Path(__file__).resolve().parent / "tasks.db"
            if tasks_db_path.exists():
                t_conn = sqlite3.connect(tasks_db_path)
                t_cursor = t_conn.cursor()
                t_cursor.execute(
                    "UPDATE tasks SET customer_name = ? WHERE customer_id = ?;",
                    (new_company_name, customer_id),
                )
                t_conn.commit()
                t_conn.close()
        except Exception as e:
            print(f"Warning: could not sync customer name to tasks.db: {e}")

    # Fetch updated aggregated record
    cursor.execute("""
        SELECT 
            c.customer_id,
            c.company_name,
            c.contact_name,
            c.contact_email,
            c.contact_phone,
            COUNT(d.deal_id) AS total_deals,
            COALESCE(SUM(d.estimated_value), 0.0) AS total_pipeline
        FROM customers c
        LEFT JOIN deals d ON c.customer_id = d.customer_id
        WHERE c.customer_id = ?
        GROUP BY c.customer_id;
    """, (customer_id,))
    row = cursor.fetchone()
    conn.close()

    return CustomerOut(
        customer_id=row["customer_id"],
        company_name=row["company_name"],
        contact_name=row["contact_name"],
        contact_email=row["contact_email"],
        contact_phone=row["contact_phone"],
        total_deals=row["total_deals"],
        total_pipeline=row["total_pipeline"],
    )


@app.get("/api/deals", response_model=List[DealOut], tags=["Deals"])
def get_deals():
    """Returns all deals with joined customer information."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            d.deal_id,
            d.customer_id,
            c.company_name,
            c.contact_name,
            c.contact_email,
            c.contact_phone,
            d.deal_name,
            d.primary_vendors,
            d.stage,
            d.estimated_value,
            d.assigned_presales,
            d.vendor_notes,
            d.created_at,
            d.updated_at
        FROM deals d
        JOIN customers c ON d.customer_id = c.customer_id
        ORDER BY d.updated_at DESC, d.deal_id DESC;
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        DealOut(
            deal_id=row["deal_id"],
            customer_id=row["customer_id"],
            company_name=row["company_name"],
            contact_name=row["contact_name"],
            contact_email=row["contact_email"],
            contact_phone=row["contact_phone"],
            deal_name=row["deal_name"],
            primary_vendors=row["primary_vendors"] or "",
            stage=row["stage"],
            estimated_value=float(row["estimated_value"] or 0.0),
            assigned_presales=row["assigned_presales"],
            vendor_notes=row["vendor_notes"] or "",
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]


@app.post("/api/deals", response_model=DealOut, status_code=status.HTTP_201_CREATED, tags=["Deals"])
def create_deal(payload: DealCreate):
    """
    Creates a new deal.
    Auto-checks if customer exists by company_name; if missing, creates a new customer entry.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Customer deduplication/lookup
    clean_company = payload.company_name.strip()
    cursor.execute(
        "SELECT customer_id, contact_name, contact_email, contact_phone FROM customers WHERE LOWER(company_name) = LOWER(?);",
        (clean_company,),
    )
    existing_customer = cursor.fetchone()

    if existing_customer:
        customer_id = existing_customer["customer_id"]
        # Update missing contact details if provided in new payload
        update_fields = []
        params = []
        if payload.contact_name and not existing_customer["contact_name"]:
            update_fields.append("contact_name = ?")
            params.append(payload.contact_name.strip())
        if payload.contact_email and not existing_customer["contact_email"]:
            update_fields.append("contact_email = ?")
            params.append(payload.contact_email.strip())
        if payload.contact_phone and not existing_customer["contact_phone"]:
            update_fields.append("contact_phone = ?")
            params.append(payload.contact_phone.strip())
        if update_fields:
            params.append(customer_id)
            cursor.execute(f"UPDATE customers SET {', '.join(update_fields)} WHERE customer_id = ?;", params)
    else:
        cursor.execute(
            """
            INSERT INTO customers (company_name, contact_name, contact_email, contact_phone)
            VALUES (?, ?, ?, ?);
            """,
            (
                clean_company,
                payload.contact_name.strip() if payload.contact_name else None,
                payload.contact_email.strip() if payload.contact_email else None,
                payload.contact_phone.strip() if payload.contact_phone else None,
            ),
        )
        customer_id = cursor.lastrowid

    # 2. Insert Deal
    vendors_str = normalize_vendors(payload.primary_vendors)
    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO deals (
            customer_id, deal_name, primary_vendors, stage, estimated_value, assigned_presales, vendor_notes, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            customer_id,
            payload.deal_name.strip(),
            vendors_str,
            payload.stage.value,
            payload.estimated_value,
            payload.assigned_presales.value if payload.assigned_presales else None,
            payload.vendor_notes.strip() if payload.vendor_notes else "",
            now_iso,
            now_iso,
        ),
    )
    deal_id = cursor.lastrowid
    conn.commit()

    # Fetch newly created deal with customer join
    cursor.execute(
        """
        SELECT 
            d.deal_id, d.customer_id, c.company_name, c.contact_name, c.contact_email, c.contact_phone,
            d.deal_name, d.primary_vendors, d.stage, d.estimated_value, d.assigned_presales,
            d.vendor_notes, d.created_at, d.updated_at
        FROM deals d
        JOIN customers c ON d.customer_id = c.customer_id
        WHERE d.deal_id = ?;
        """,
        (deal_id,),
    )
    row = cursor.fetchone()
    conn.close()

    return DealOut(
        deal_id=row["deal_id"],
        customer_id=row["customer_id"],
        company_name=row["company_name"],
        contact_name=row["contact_name"],
        contact_email=row["contact_email"],
        contact_phone=row["contact_phone"],
        deal_name=row["deal_name"],
        primary_vendors=row["primary_vendors"] or "",
        stage=row["stage"],
        estimated_value=float(row["estimated_value"]),
        assigned_presales=row["assigned_presales"],
        vendor_notes=row["vendor_notes"] or "",
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


@app.put("/api/deals/{deal_id}", response_model=DealOut, tags=["Deals"])
def update_deal(deal_id: int, payload: DealUpdate):
    """
    Updates deal stage, value, vendor notes, assigned presales, vendors, or deal name.
    Updates the updated_at timestamp.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT deal_id FROM deals WHERE deal_id = ?;", (deal_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Deal with ID {deal_id} not found.")

    update_clauses = []
    params = []

    if payload.deal_name is not None:
        update_clauses.append("deal_name = ?")
        params.append(payload.deal_name.strip())

    if payload.primary_vendors is not None:
        update_clauses.append("primary_vendors = ?")
        params.append(normalize_vendors(payload.primary_vendors))

    if payload.stage is not None:
        update_clauses.append("stage = ?")
        params.append(payload.stage.value)

    if payload.estimated_value is not None:
        update_clauses.append("estimated_value = ?")
        params.append(payload.estimated_value)

    if payload.assigned_presales is not None:
        update_clauses.append("assigned_presales = ?")
        params.append(payload.assigned_presales.value)

    if payload.vendor_notes is not None:
        update_clauses.append("vendor_notes = ?")
        params.append(payload.vendor_notes.strip())

    now_iso = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    update_clauses.append("updated_at = ?")
    params.append(now_iso)

    params.append(deal_id)
    cursor.execute(
        f"UPDATE deals SET {', '.join(update_clauses)} WHERE deal_id = ?;",
        params,
    )
    conn.commit()

    # Retrieve updated deal
    cursor.execute(
        """
        SELECT 
            d.deal_id, d.customer_id, c.company_name, c.contact_name, c.contact_email, c.contact_phone,
            d.deal_name, d.primary_vendors, d.stage, d.estimated_value, d.assigned_presales,
            d.vendor_notes, d.created_at, d.updated_at
        FROM deals d
        JOIN customers c ON d.customer_id = c.customer_id
        WHERE d.deal_id = ?;
        """,
        (deal_id,),
    )
    row = cursor.fetchone()
    conn.close()

    return DealOut(
        deal_id=row["deal_id"],
        customer_id=row["customer_id"],
        company_name=row["company_name"],
        contact_name=row["contact_name"],
        contact_email=row["contact_email"],
        contact_phone=row["contact_phone"],
        deal_name=row["deal_name"],
        primary_vendors=row["primary_vendors"] or "",
        stage=row["stage"],
        estimated_value=float(row["estimated_value"]),
        assigned_presales=row["assigned_presales"],
        vendor_notes=row["vendor_notes"] or "",
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


# -----------------------------------------------------------------------------
# Analytics & Metrics Aggregation Endpoints
# -----------------------------------------------------------------------------
TASKS_DB_FILE = Path(__file__).resolve().parent / "tasks.db"


@app.get("/api/metrics/dashboard", tags=["Metrics"])
def get_dashboard_metrics(presales: Optional[str] = None) -> Dict[str, Any]:
    """Calculates comprehensive executive metrics and visual chart data across CRM deals and tasks."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            d.deal_id, d.customer_id, c.company_name, d.deal_name, d.primary_vendors,
            d.stage, d.estimated_value, d.assigned_presales, d.vendor_notes, d.created_at, d.updated_at
        FROM deals d
        JOIN customers c ON d.customer_id = c.customer_id
    """
    params = []
    if presales:
        query += " WHERE d.assigned_presales = ?"
        params.append(presales)

    query += " ORDER BY d.estimated_value DESC;"
    cursor.execute(query, params)
    deals = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM customers;")
    customer_count = cursor.fetchone()[0]
    conn.close()

    total_pipeline = 0.0
    active_pipeline = 0.0
    won_value = 0.0
    lost_value = 0.0
    won_count = 0
    lost_count = 0
    active_count = 0

    stage_order = ["Discovery", "Gathering Requirements", "RFP / Tender", "PoC", "Proposal", "Closed-Won", "Closed-Lost"]
    stage_breakdown = {s: {"count": 0, "value": 0.0} for s in stage_order}
    vendor_breakdown: Dict[str, Dict[str, Any]] = {}
    presales_breakdown = {
        "Presales 1": {"pipeline": 0.0, "won_value": 0.0, "active_count": 0, "total_count": 0},
        "Presales 2": {"pipeline": 0.0, "won_value": 0.0, "active_count": 0, "total_count": 0},
    }

    for d in deals:
        val = float(d["estimated_value"] or 0.0)
        stg = d["stage"]
        rep = d["assigned_presales"]

        total_pipeline += val

        if stg in stage_breakdown:
            stage_breakdown[stg]["count"] += 1
            stage_breakdown[stg]["value"] += val

        if stg == "Closed-Won":
            won_value += val
            won_count += 1
        elif stg == "Closed-Lost":
            lost_value += val
            lost_count += 1
        else:
            active_pipeline += val
            active_count += 1

        if rep in presales_breakdown:
            presales_breakdown[rep]["total_count"] += 1
            presales_breakdown[rep]["pipeline"] += val
            if stg == "Closed-Won":
                presales_breakdown[rep]["won_value"] += val
            elif stg not in ("Closed-Won", "Closed-Lost"):
                presales_breakdown[rep]["active_count"] += 1

        vendors_str = d["primary_vendors"] or "General"
        for v in [x.strip() for x in vendors_str.split(",") if x.strip()]:
            if v not in vendor_breakdown:
                vendor_breakdown[v] = {"count": 0, "value": 0.0}
            vendor_breakdown[v]["count"] += 1
            vendor_breakdown[v]["value"] += val

    closed_total = won_count + lost_count
    win_rate = round((won_count / closed_total * 100.0), 1) if closed_total > 0 else 0.0
    avg_deal_value = round((total_pipeline / len(deals)), 2) if deals else 0.0

    task_metrics = {
        "total_tasks": 0,
        "active_tasks": 0,
        "completed_tasks": 0,
        "status_breakdown": {},
        "category_breakdown": {},
        "blockers": [],
    }

    if TASKS_DB_FILE.exists():
        try:
            t_conn = sqlite3.connect(TASKS_DB_FILE)
            t_conn.row_factory = sqlite3.Row
            t_cur = t_conn.cursor()

            t_query = "SELECT task_id, task_title, category, assigned_to, vendor_domain, status, priority, due_date, management_blockers FROM tasks"
            t_params = []
            if presales:
                t_query += " WHERE assigned_to = ?"
                t_params.append(presales)
            t_cur.execute(t_query, t_params)
            t_rows = [dict(r) for r in t_cur.fetchall()]

            task_metrics["total_tasks"] = len(t_rows)
            for t in t_rows:
                st = t["status"]
                cat = t["category"]
                if st == "Completed":
                    task_metrics["completed_tasks"] += 1
                else:
                    task_metrics["active_tasks"] += 1

                task_metrics["status_breakdown"][st] = task_metrics["status_breakdown"].get(st, 0) + 1
                task_metrics["category_breakdown"][cat] = task_metrics["category_breakdown"].get(cat, 0) + 1

                if t.get("management_blockers") and str(t["management_blockers"]).strip():
                    task_metrics["blockers"].append({
                        "task_id": t["task_id"],
                        "task_title": t["task_title"],
                        "assigned_to": t["assigned_to"],
                        "vendor_domain": t["vendor_domain"],
                        "blocker": t["management_blockers"],
                    })
            t_conn.close()
        except Exception as ex:
            print("Warning: Could not fetch task metrics for dashboard:", ex)

    return {
        "total_pipeline": total_pipeline,
        "active_pipeline": active_pipeline,
        "won_value": won_value,
        "lost_value": lost_value,
        "total_deals_count": len(deals),
        "active_deals_count": active_count,
        "won_count": won_count,
        "lost_count": lost_count,
        "win_rate": win_rate,
        "avg_deal_value": avg_deal_value,
        "customer_count": customer_count,
        "stage_breakdown": stage_breakdown,
        "vendor_breakdown": vendor_breakdown,
        "presales_breakdown": presales_breakdown,
        "top_deals": deals[:5],
        "task_metrics": task_metrics,
    }


# -----------------------------------------------------------------------------
# Embedded Single-File HTML / Bootstrap Dashboard
# -----------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Presales CRM Dashboard</title>
    <!-- Bootstrap 5 CSS & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <!-- Chart.js for Executive Visual Analytics -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <style>
        body {
            background-color: #f1f5f9;
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
        }
        .navbar-brand {
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .stat-card {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
        }
        .table-card {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
        }
        .stage-badge {
            font-weight: 600;
            padding: 0.35em 0.65em;
            border-radius: 6px;
            font-size: 0.8rem;
        }
        .badge-Discovery { background-color: #e0e7ff; color: #3730a3; }
        .badge-Gathering-Requirements { background-color: #e0f2fe; color: #0369a1; }
        .badge-RFP---Tender, .badge-RFP-Tender { background-color: #fef08a; color: #854d0e; border: 1px solid #fde047; }
        .badge-PoC { background-color: #fef3c7; color: #b45309; }
        .badge-Proposal { background-color: #ede9fe; color: #6d28d9; }
        .badge-Closed-Won { background-color: #dcfce7; color: #15803d; }
        .badge-Closed-Lost { background-color: #fee2e2; color: #b91c1c; }
        .vendor-chip {
            display: inline-block;
            background: #e2e8f0;
            color: #334155;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 4px;
            margin: 1px;
        }
        .modal-content {
            border-radius: 14px;
            border: none;
            box-shadow: 0 20px 25px -5px rgba(0,0,0,0.2);
        }
        .table > tbody > tr:hover {
            background-color: #f8fafc;
        }
        .filter-toolbar {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
            padding: 1.1rem 1.25rem;
            margin-bottom: 1.25rem;
        }
        .filter-btn {
            font-size: 0.82rem;
            font-weight: 500;
            border-radius: 6px;
            transition: all 0.15s ease-in-out;
            padding: 0.35rem 0.75rem;
        }
        .filter-btn.active {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
            font-weight: 600;
            box-shadow: 0 2px 5px rgba(13, 110, 253, 0.3);
        }
        .filter-btn:hover:not(.active) {
            background-color: #e2e8f0;
            color: #0f172a;
        }
        .clickable-stat {
            cursor: pointer;
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out, border-color 0.15s ease;
        }
        .clickable-stat:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
            border-color: #0d6efd !important;
        }
    </style>
</head>
<body>

    <!-- Top Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top shadow-sm py-2">
        <div class="container-fluid px-4">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <i class="bi bi-diagram-3-fill text-primary"></i>
                <span>Enterprise Presales CRM</span>
            </a>
            <div class="d-flex flex-wrap align-items-center gap-2">
                <a href="http://127.0.0.1:8001" target="_blank" class="btn btn-sm btn-outline-secondary text-light d-flex align-items-center gap-1">
                    <i class="bi bi-kanban text-info"></i> Task Board (8001)
                </a>
                <a href="http://127.0.0.1:8002" target="_blank" class="btn btn-sm btn-outline-secondary text-light d-flex align-items-center gap-1">
                    <i class="bi bi-soundwave text-warning"></i> Voice Agent (8002)
                </a>
                <span class="badge bg-secondary-subtle text-dark border px-2 py-1">
                    <i class="bi bi-database me-1"></i>SQLite (crm.db)
                </span>
                <a href="/docs" target="_blank" class="btn btn-sm btn-outline-light">
                    <i class="bi bi-code-slash me-1"></i>Swagger API
                </a>
                <button class="btn btn-sm btn-primary d-flex align-items-center gap-1" onclick="openNewDealModal()">
                    <i class="bi bi-plus-circle"></i> New Deal
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid px-4 py-4">

        <!-- Top Metrics Cards -->
        <div class="row g-3 mb-4">
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card clickable-stat p-3 bg-white" onclick="setQuickFilter('ALL')" title="Click to view all deals">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Total Pipeline</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statTotalPipeline">$0</h3>
                            <div class="text-muted" style="font-size: 0.72rem;"><i class="bi bi-eye me-1"></i>Click to show all</div>
                        </div>
                        <div class="bg-primary-subtle text-primary p-3 rounded-3">
                            <i class="bi bi-cash-stack fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card clickable-stat p-3 bg-white" onclick="setQuickFilter('ACTIVE')" title="Click to filter active opportunities">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Active Opportunities</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statActiveDeals">0</h3>
                            <div class="text-muted" style="font-size: 0.72rem;"><i class="bi bi-funnel me-1"></i>Click to filter active</div>
                        </div>
                        <div class="bg-warning-subtle text-warning p-3 rounded-3">
                            <i class="bi bi-lightning-charge-fill fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card clickable-stat p-3 bg-white" onclick="setQuickFilter('Closed-Won')" title="Click to filter Closed-Won deals">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Closed-Won Value</div>
                            <h3 class="fw-bold mb-0 text-success" id="statWonValue">$0</h3>
                            <div class="text-muted" style="font-size: 0.72rem;"><i class="bi bi-trophy me-1"></i>Click to filter won</div>
                        </div>
                        <div class="bg-success-subtle text-success p-3 rounded-3">
                            <i class="bi bi-trophy-fill fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card clickable-stat p-3 bg-white" onclick="switchToCustomersTab()" title="Click to view accounts">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Accounts Tracked</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statCustomerCount">0</h3>
                            <div class="text-muted" style="font-size: 0.72rem;"><i class="bi bi-building me-1"></i>Click to view accounts</div>
                        </div>
                        <div class="bg-info-subtle text-info p-3 rounded-3">
                            <i class="bi bi-building fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Navigation Tabs Card -->
        <div class="card table-card bg-white p-3 mb-3">
            <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
                <ul class="nav nav-pills" id="crmTabs" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link active fw-semibold" id="dashboard-tab" data-bs-toggle="pill" data-bs-target="#dashboard-pane" type="button" onclick="loadDashboardMetrics()">
                            <i class="bi bi-graph-up-arrow me-1 text-primary"></i>Visual Dashboard
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link fw-semibold" id="deals-tab" data-bs-toggle="pill" data-bs-target="#deals-pane" type="button">
                            <i class="bi bi-kanban me-1"></i>Deals Pipeline
                            <span class="badge bg-primary ms-1 rounded-pill" id="dealsTabBadge">0</span>
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link fw-semibold" id="customers-tab" data-bs-toggle="pill" data-bs-target="#customers-pane" type="button">
                            <i class="bi bi-people me-1"></i>Customers & Accounts
                            <span class="badge bg-secondary ms-1 rounded-pill" id="customersTabBadge">0</span>
                        </button>
                    </li>
                </ul>
                <div class="d-flex align-items-center gap-2">
                    <button class="btn btn-sm btn-outline-primary" onclick="openNewCustomerModal()">
                        <i class="bi bi-person-plus me-1"></i>New Customer
                    </button>
                    <button class="btn btn-sm btn-primary" onclick="openNewDealModal()">
                        <i class="bi bi-plus-circle me-1"></i>New Deal
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="loadAllData()" title="Reload All Data">
                        <i class="bi bi-arrow-clockwise me-1"></i>Refresh
                    </button>
                </div>
            </div>
        </div>

        <!-- Tab Panes -->
        <div class="tab-content" id="crmTabContent">

            <!-- VISUAL DASHBOARD TAB PANE -->
            <div class="tab-pane fade show active" id="dashboard-pane" role="tabpanel">

                <!-- Dashboard Header Toolbar -->
                <div class="filter-toolbar mb-3">
                    <div class="d-flex flex-wrap align-items-center justify-content-between gap-3">
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-primary text-white p-2 rounded-2">
                                <i class="bi bi-speedometer2 fs-6"></i>
                            </span>
                            <div>
                                <h6 class="fw-bold mb-0 text-dark">Executive Analytics & Visual Metrics</h6>
                                <div class="text-muted small">Real-time pipeline analytics, vendor market share, presales workload & task health</div>
                            </div>
                        </div>

                        <!-- Presales Lead Filter Controls -->
                        <div class="d-flex flex-wrap align-items-center gap-2">
                            <span class="small text-muted fw-semibold"><i class="bi bi-person-fill me-1"></i>Filter by Lead:</span>
                            <div class="btn-group btn-group-sm" role="group" id="dashPresalesFilterGroup">
                                <button type="button" class="btn btn-outline-secondary filter-btn active" onclick="setDashboardPresalesFilter('', this)">All Leads</button>
                                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setDashboardPresalesFilter('Presales 1', this)">Presales 1</button>
                                <button type="button" class="btn btn-outline-secondary filter-btn" onclick="setDashboardPresalesFilter('Presales 2', this)">Presales 2</button>
                            </div>
                            <button class="btn btn-sm btn-outline-secondary ms-1" onclick="loadDashboardMetrics()" title="Refresh Dashboard">
                                <i class="bi bi-arrow-clockwise me-1"></i>Sync
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Executive Scorecards Ribbon -->
                <div class="row g-3 mb-4">
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Total Pipeline</div>
                            <h4 class="fw-bold mb-0 text-dark" id="dashKpiTotalPipeline">$0</h4>
                            <div class="small text-muted mt-1" id="dashKpiTotalDeals">0 Deals</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Active Pipeline</div>
                            <h4 class="fw-bold mb-0 text-primary" id="dashKpiActivePipeline">$0</h4>
                            <div class="small text-muted mt-1" id="dashKpiActiveDeals">0 In Progress</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Won Revenue</div>
                            <h4 class="fw-bold mb-0 text-success" id="dashKpiWonValue">$0</h4>
                            <div class="small text-success mt-1" id="dashKpiWinRate">0% Win Rate</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Avg Deal Size</div>
                            <h4 class="fw-bold mb-0 text-info" id="dashKpiAvgDeal">$0</h4>
                            <div class="small text-muted mt-1">Per Opportunity</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Tracked Accounts</div>
                            <h4 class="fw-bold mb-0 text-secondary" id="dashKpiAccounts">0</h4>
                            <div class="small text-muted mt-1">Clients & Orgs</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4 col-xl-2">
                        <div class="card stat-card p-3 bg-white h-100">
                            <div class="text-muted text-uppercase fw-semibold" style="font-size: 0.72rem;">Task Execution</div>
                            <h4 class="fw-bold mb-0 text-dark" id="dashKpiActiveTasks">0 Active</h4>
                            <div class="small mt-1" id="dashKpiBlockers">0 Blockers</div>
                        </div>
                    </div>
                </div>

                <!-- Charts Grid Row 1 -->
                <div class="row g-3 mb-4">
                    <!-- Stage Pipeline Bar Chart -->
                    <div class="col-12 col-xl-7">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <div>
                                    <h6 class="fw-bold mb-0 text-dark"><i class="bi bi-bar-chart-fill text-primary me-2"></i>Opportunity Pipeline by Stage</h6>
                                    <div class="text-muted small">Dollar volume & opportunity distribution across sales stages</div>
                                </div>
                                <span class="badge bg-light text-dark border">Volume ($)</span>
                            </div>
                            <div style="position: relative; height: 270px;">
                                <canvas id="stageChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Vendor Market Share Doughnut Chart -->
                    <div class="col-12 col-xl-5">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <div>
                                    <h6 class="fw-bold mb-0 text-dark"><i class="bi bi-pie-chart-fill text-info me-2"></i>Primary Vendor Share</h6>
                                    <div class="text-muted small">Pipeline value allocation across partner technologies</div>
                                </div>
                                <span class="badge bg-light text-dark border">By Vendor</span>
                            </div>
                            <div style="position: relative; height: 270px;" class="d-flex align-items-center justify-content-center">
                                <canvas id="vendorChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Charts Grid Row 2 -->
                <div class="row g-3 mb-4">
                    <!-- Presales Workload & Revenue Comparison -->
                    <div class="col-12 col-xl-6">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <div>
                                    <h6 class="fw-bold mb-0 text-dark"><i class="bi bi-people-fill text-warning me-2"></i>Presales Workload & Won Revenue</h6>
                                    <div class="text-muted small">Head-to-head comparison of managed pipeline & closed revenue</div>
                                </div>
                                <span class="badge bg-light text-dark border">P1 vs P2</span>
                            </div>
                            <div style="position: relative; height: 270px;">
                                <canvas id="presalesChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Task Execution & RFP Status -->
                    <div class="col-12 col-xl-6">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <div>
                                    <h6 class="fw-bold mb-0 text-dark"><i class="bi bi-kanban-fill text-success me-2"></i>Task Execution & Deliverable Status</h6>
                                    <div class="text-muted small">Breakdown of operational tasks and RFP deliverables</div>
                                </div>
                                <a href="http://127.0.0.1:8001" target="_blank" class="btn btn-sm btn-outline-secondary">Open Board <i class="bi bi-box-arrow-up-right ms-1"></i></a>
                            </div>
                            <div style="position: relative; height: 270px;" class="d-flex align-items-center justify-content-center">
                                <canvas id="taskStatusChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Deep-Dive Spotlight Row (Top Deals & Active Blockers) -->
                <div class="row g-3 mb-4">
                    <!-- Top 5 Opportunities Leaderboard -->
                    <div class="col-12 col-xl-7">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <div>
                                    <h6 class="fw-bold mb-0 text-dark"><i class="bi bi-trophy-fill text-warning me-2"></i>Top High-Value Opportunities</h6>
                                    <div class="text-muted small">Highest estimated value opportunities currently in pipeline</div>
                                </div>
                                <button class="btn btn-sm btn-outline-primary" onclick="switchToDealsTab()">View All <i class="bi bi-arrow-right ms-1"></i></button>
                            </div>
                            <div class="table-responsive">
                                <table class="table table-sm table-hover align-middle mb-0">
                                    <thead class="table-light small">
                                        <tr>
                                            <th>Deal (ID & Title)</th>
                                            <th>Customer (ID & Name)</th>
                                            <th>Vendors</th>
                                            <th>Stage</th>
                                            <th>Est. Value</th>
                                            <th>Lead</th>
                                        </tr>
                                    </thead>
                                    <tbody id="dashTopDealsBody">
                                        <tr><td colspan="6" class="text-center py-4 text-muted">No deals registered yet.</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Active Management Blockers Warning Feed -->
                    <div class="col-12 col-xl-5">
                        <div class="card table-card bg-white p-3 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <div>
                                    <h6 class="fw-bold mb-0 text-danger"><i class="bi bi-exclamation-octagon-fill me-2"></i>Active Management Blockers</h6>
                                    <div class="text-muted small">Vendor dependencies, sizing hurdles, or approval bottlenecks</div>
                                </div>
                                <span class="badge bg-danger-subtle text-danger border" id="dashBlockersCountBadge">0 Blockers</span>
                            </div>
                            <div id="dashBlockersList" style="max-height: 240px; overflow-y: auto;" class="d-flex flex-column gap-2">
                                <div class="text-center py-4 text-muted small">
                                    <i class="bi bi-check2-circle text-success fs-3 d-block mb-1"></i>
                                    All clear! No active management blockers reported.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

            <!-- DEALS TAB PANE -->
            <div class="tab-pane fade" id="deals-pane" role="tabpanel">

                <!-- DEDICATED HIGH-VISIBILITY DEALS FILTER TOOLBAR -->
                <div class="filter-toolbar mb-3">
                    <!-- Toolbar Header & Status Badge -->
                    <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3 pb-2 border-bottom">
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-primary text-white p-2 rounded-2">
                                <i class="bi bi-funnel-fill fs-6"></i>
                            </span>
                            <div>
                                <span class="fw-bold text-dark fs-6">Pipeline Filters</span>
                                <span class="text-muted small ms-2 d-none d-md-inline">Filter deals by quick view, stage, presales owner, vendor or keyword</span>
                            </div>
                            <span class="badge bg-primary-subtle text-primary border ms-2" id="dealsFilterSummaryBadge">Showing all deals</span>
                        </div>
                        <div class="d-flex align-items-center gap-2">
                            <button type="button" class="btn btn-sm btn-outline-danger" id="resetDealsFilterBtn" onclick="resetDealsFilters()" style="display: none;">
                                <i class="bi bi-x-circle me-1"></i>Clear Filters
                            </button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="loadAllData()" title="Reload Data">
                                <i class="bi bi-arrow-clockwise me-1"></i>Reload
                            </button>
                        </div>
                    </div>

                    <!-- Quick Filter Buttons Row (Monday.com / Kanban Style) -->
                    <div class="d-flex flex-wrap align-items-center gap-2 mb-3">
                        <span class="small text-muted fw-bold me-1"><i class="bi bi-sliders me-1"></i>Quick Views:</span>
                        <div class="btn-group btn-group-sm flex-wrap" role="group" id="quickFilterGroup">
                            <button type="button" class="btn btn-outline-secondary filter-btn active" id="qf-ALL" onclick="setQuickFilter('ALL', this)">All Deals</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-ACTIVE" onclick="setQuickFilter('ACTIVE', this)"><i class="bi bi-lightning-charge me-1 text-warning"></i>Active Only</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-Discovery" onclick="setQuickFilter('Discovery', this)">Discovery</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-Req" onclick="setQuickFilter('Gathering Requirements', this)">Gathering Req.</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-RFP" onclick="setQuickFilter('RFP / Tender', this)"><i class="bi bi-file-earmark-text me-1"></i>RFP / Tender</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-PoC" onclick="setQuickFilter('PoC', this)">PoC</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-Proposal" onclick="setQuickFilter('Proposal', this)">Proposal</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn text-success" id="qf-Won" onclick="setQuickFilter('Closed-Won', this)"><i class="bi bi-trophy-fill me-1"></i>Closed-Won</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn text-danger" id="qf-Lost" onclick="setQuickFilter('Closed-Lost', this)"><i class="bi bi-x-octagon me-1"></i>Closed-Lost</button>
                        </div>
                        <div class="btn-group btn-group-sm ms-md-2" role="group">
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-P1" onclick="setQuickFilter('PRESALES_1', this)"><i class="bi bi-person me-1 text-primary"></i>Presales 1</button>
                            <button type="button" class="btn btn-outline-secondary filter-btn" id="qf-P2" onclick="setQuickFilter('PRESALES_2', this)"><i class="bi bi-person me-1 text-info"></i>Presales 2</button>
                        </div>
                    </div>

                    <!-- Search & Dropdown Select Controls Row -->
                    <div class="row g-2 align-items-center">
                        <div class="col-12 col-lg-4">
                            <div class="input-group input-group-sm">
                                <span class="input-group-text bg-light text-muted"><i class="bi bi-search"></i></span>
                                <input type="text" id="searchInput" class="form-control" placeholder="Search deals, company, notes, vendors..." oninput="applyFilters()">
                                <button class="btn btn-outline-secondary" type="button" onclick="clearDealSearch()" title="Clear search">
                                    <i class="bi bi-x-lg"></i>
                                </button>
                            </div>
                        </div>
                        <div class="col-6 col-sm-4 col-lg-3">
                            <div class="input-group input-group-sm">
                                <span class="input-group-text bg-light small text-muted"><i class="bi bi-kanban me-1"></i>Stage</span>
                                <select id="stageFilter" class="form-select" onchange="onDropdownFilterChange()">
                                    <option value="">All Stages</option>
                                    <option value="Discovery">Discovery</option>
                                    <option value="Gathering Requirements">Gathering Requirements</option>
                                    <option value="RFP / Tender">RFP / Tender (Direct Pricing)</option>
                                    <option value="PoC">PoC</option>
                                    <option value="Proposal">Proposal</option>
                                    <option value="Closed-Won">Closed-Won</option>
                                    <option value="Closed-Lost">Closed-Lost</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-6 col-sm-4 col-lg-3">
                            <div class="input-group input-group-sm">
                                <span class="input-group-text bg-light small text-muted"><i class="bi bi-person-badge me-1"></i>Presales</span>
                                <select id="presalesFilter" class="form-select" onchange="onDropdownFilterChange()">
                                    <option value="">All Presales</option>
                                    <option value="Presales 1">Presales 1</option>
                                    <option value="Presales 2">Presales 2</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-sm-4 col-lg-2">
                            <div class="input-group input-group-sm">
                                <span class="input-group-text bg-light small text-muted"><i class="bi bi-cpu me-1"></i>Vendor</span>
                                <select id="vendorFilter" class="form-select" onchange="onDropdownFilterChange()">
                                    <option value="">All Vendors</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- DEALS TABLE -->
                <div class="card table-card bg-white">
                    <div class="table-responsive">
                        <table class="table table-hover align-middle mb-0" id="dealsTable">
                            <thead class="table-light">
                                <tr>
                                    <th style="width: 70px;">Deal ID</th>
                                    <th>Deal / Opportunity</th>
                                    <th>Customer Account</th>
                                    <th>Vendors</th>
                                    <th>Stage</th>
                                    <th>Est. Value</th>
                                    <th>Presales Lead</th>
                                    <th>Notes</th>
                                    <th>Updated</th>
                                    <th class="text-end">Actions</th>
                                </tr>
                            </thead>
                            <tbody id="dealsTableBody">
                                <tr>
                                    <td colspan="10" class="text-center py-4 text-muted">
                                        <div class="spinner-border spinner-border-sm me-2 text-primary"></div>Loading deals...
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- CUSTOMERS TAB PANE -->
            <div class="tab-pane fade" id="customers-pane" role="tabpanel">

                <!-- DEDICATED CUSTOMERS FILTER & SEARCH TOOLBAR -->
                <div class="filter-toolbar mb-3">
                    <div class="d-flex flex-wrap align-items-center justify-content-between gap-2">
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-info text-white p-2 rounded-2">
                                <i class="bi bi-building fs-6"></i>
                            </span>
                            <div>
                                <span class="fw-bold text-dark fs-6">Customer Accounts Search</span>
                                <span class="text-muted small ms-2 d-none d-md-inline">Find accounts by company name, primary contact, email or phone</span>
                            </div>
                            <span class="badge bg-info-subtle text-info border ms-2" id="customersFilterSummaryBadge">Showing all accounts</span>
                        </div>
                        <div class="d-flex align-items-center gap-2">
                            <div class="input-group input-group-sm" style="width: 320px;">
                                <span class="input-group-text bg-light text-muted"><i class="bi bi-search"></i></span>
                                <input type="text" id="customerSearchInput" class="form-control" placeholder="Search company, contact, email..." oninput="applyCustomerFilters()">
                                <button class="btn btn-outline-secondary" type="button" onclick="clearCustomerSearch()" title="Clear search">
                                    <i class="bi bi-x-lg"></i>
                                </button>
                            </div>
                            <button class="btn btn-sm btn-primary" onclick="openNewCustomerModal()">
                                <i class="bi bi-person-plus me-1"></i>New Customer
                            </button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="loadAllData()" title="Reload Data">
                                <i class="bi bi-arrow-clockwise"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- CUSTOMERS TABLE -->
                <div class="card table-card bg-white">
                    <div class="table-responsive">
                        <table class="table table-hover align-middle mb-0">
                            <thead class="table-light">
                                <tr>
                                    <th>ID</th>
                                    <th>Company</th>
                                    <th>Primary Contact</th>
                                    <th>Email</th>
                                    <th>Phone</th>
                                    <th>Total Deals</th>
                                    <th>Pipeline Value</th>
                                    <th class="text-end">Actions</th>
                                </tr>
                            </thead>
                            <tbody id="customersTableBody">
                                <tr>
                                    <td colspan="8" class="text-center py-4 text-muted">
                                        <div class="spinner-border spinner-border-sm me-2 text-primary"></div>Loading accounts...
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>

    </div>

    <!-- Create Deal Modal -->
    <div class="modal fade" id="newDealModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <form id="newDealForm" onsubmit="submitNewDeal(event)">
                    <div class="modal-header">
                        <h5 class="modal-title fw-bold"><i class="bi bi-folder-plus text-primary me-2"></i>Create New Opportunity</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <div class="row g-3">
                            <!-- Customer Info -->
                            <div class="col-12">
                                <h6 class="text-muted text-uppercase fw-semibold small mb-2 border-bottom pb-1">Customer Account (Auto-deduplicated)</h6>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Company Name *</label>
                                <input type="text" list="customerDatalist" id="newCompanyName" class="form-control" placeholder="e.g. Acme Corp" required>
                                <datalist id="customerDatalist"></datalist>
                                <div class="form-text small">Select existing account or type new to create.</div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Contact Name</label>
                                <input type="text" id="newContactName" class="form-control" placeholder="Jane Doe">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Contact Email</label>
                                <input type="email" id="newContactEmail" class="form-control" placeholder="jane@acme.com">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Contact Phone</label>
                                <input type="text" id="newContactPhone" class="form-control" placeholder="+1-555-0100">
                            </div>

                            <!-- Opportunity Details -->
                            <div class="col-12 mt-4">
                                <h6 class="text-muted text-uppercase fw-semibold small mb-2 border-bottom pb-1">Opportunity Parameters</h6>
                            </div>
                            <div class="col-md-8">
                                <label class="form-label small fw-semibold">Deal / Project Name *</label>
                                <input type="text" id="newDealName" class="form-control" placeholder="e.g. SAN Storage Refresh 2026" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Estimated Value ($) *</label>
                                <div class="input-group">
                                    <span class="input-group-text">$</span>
                                    <input type="number" step="0.01" min="0" id="newEstimatedValue" class="form-control" placeholder="0.00" required>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Stage *</label>
                                <select id="newStage" class="form-select" required>
                                    <option value="Discovery">Discovery</option>
                                    <option value="Gathering Requirements">Gathering Requirements</option>
                                    <option value="RFP / Tender">RFP / Tender (Direct Pricing)</option>
                                    <option value="PoC">PoC</option>
                                    <option value="Proposal">Proposal</option>
                                    <option value="Closed-Won">Closed-Won</option>
                                    <option value="Closed-Lost">Closed-Lost</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Assigned Presales *</label>
                                <select id="newAssignedPresales" class="form-select" required>
                                    <option value="Presales 1">Presales 1</option>
                                    <option value="Presales 2">Presales 2</option>
                                </select>
                            </div>

                            <!-- Primary Vendors -->
                            <div class="col-12">
                                <label class="form-label small fw-semibold d-block">Primary Vendors Involved</label>
                                <div class="d-flex flex-wrap gap-3">
                                    <div class="form-check">
                                        <input class="form-check-input vendor-check" type="checkbox" value="HPE" id="vendorHPE">
                                        <label class="form-check-label" for="vendorHPE">HPE</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input vendor-check" type="checkbox" value="Veeam" id="vendorVeeam">
                                        <label class="form-check-label" for="vendorVeeam">Veeam</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input vendor-check" type="checkbox" value="Dell" id="vendorDell">
                                        <label class="form-check-label" for="vendorDell">Dell</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input vendor-check" type="checkbox" value="Nutanix" id="vendorNutanix">
                                        <label class="form-check-label" for="vendorNutanix">Nutanix</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input vendor-check" type="checkbox" value="VMware" id="vendorVMware">
                                        <label class="form-check-label" for="vendorVMware">VMware</label>
                                    </div>
                                </div>
                            </div>

                            <div class="col-12">
                                <label class="form-label small fw-semibold">Vendor & Technical Notes</label>
                                <textarea id="newVendorNotes" class="form-control" rows="3" placeholder="Sizing details, competitor analysis, architecture specifications..."></textarea>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary px-4"><i class="bi bi-check-lg me-1"></i>Create Deal</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Edit Deal Modal -->
    <div class="modal fade" id="editDealModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <form id="editDealForm" onsubmit="submitEditDeal(event)">
                    <div class="modal-header">
                        <h5 class="modal-title fw-bold"><i class="bi bi-pencil-square text-primary me-2"></i>Update Deal</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <input type="hidden" id="editDealId">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Company Name</label>
                                <input type="text" id="editCompanyName" class="form-control bg-light" readonly>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Deal Name *</label>
                                <input type="text" id="editDealName" class="form-control" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Stage *</label>
                                <select id="editStage" class="form-select" required>
                                    <option value="Discovery">Discovery</option>
                                    <option value="Gathering Requirements">Gathering Requirements</option>
                                    <option value="RFP / Tender">RFP / Tender (Direct Pricing)</option>
                                    <option value="PoC">PoC</option>
                                    <option value="Proposal">Proposal</option>
                                    <option value="Closed-Won">Closed-Won</option>
                                    <option value="Closed-Lost">Closed-Lost</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Estimated Value ($)</label>
                                <div class="input-group">
                                    <span class="input-group-text">$</span>
                                    <input type="number" step="0.01" min="0" id="editEstimatedValue" class="form-control" required>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label small fw-semibold">Assigned Presales</label>
                                <select id="editAssignedPresales" class="form-select" required>
                                    <option value="Presales 1">Presales 1</option>
                                    <option value="Presales 2">Presales 2</option>
                                </select>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold d-block">Primary Vendors</label>
                                <div class="d-flex flex-wrap gap-3">
                                    <div class="form-check">
                                        <input class="form-check-input edit-vendor-check" type="checkbox" value="HPE" id="editVendorHPE">
                                        <label class="form-check-label" for="editVendorHPE">HPE</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input edit-vendor-check" type="checkbox" value="Veeam" id="editVendorVeeam">
                                        <label class="form-check-label" for="editVendorVeeam">Veeam</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input edit-vendor-check" type="checkbox" value="Dell" id="editVendorDell">
                                        <label class="form-check-label" for="editVendorDell">Dell</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input edit-vendor-check" type="checkbox" value="Nutanix" id="editVendorNutanix">
                                        <label class="form-check-label" for="editVendorNutanix">Nutanix</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input edit-vendor-check" type="checkbox" value="VMware" id="editVendorVMware">
                                        <label class="form-check-label" for="editVendorVMware">VMware</label>
                                    </div>
                                </div>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Vendor Notes & BOM</label>
                                <textarea id="editVendorNotes" class="form-control" rows="4"></textarea>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary"><i class="bi bi-save me-1"></i>Save Changes</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Create Customer Modal -->
    <div class="modal fade" id="newCustomerModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-md">
            <div class="modal-content">
                <form id="newCustomerForm" onsubmit="submitNewCustomer(event)">
                    <div class="modal-header">
                        <h5 class="modal-title fw-bold"><i class="bi bi-building-add text-primary me-2"></i>Add New Customer Account</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <div class="row g-3">
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Company / Organization Name *</label>
                                <input type="text" id="newCustCompanyName" class="form-control" placeholder="e.g. Almarai, Aramco, Acme Corp..." required>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Primary Contact Name</label>
                                <input type="text" id="newCustContactName" class="form-control" placeholder="e.g. Mohammed Al-Otaibi">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Email Address</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-envelope"></i></span>
                                    <input type="email" id="newCustContactEmail" class="form-control" placeholder="contact@company.com">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Phone Number</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-telephone"></i></span>
                                    <input type="text" id="newCustContactPhone" class="form-control" placeholder="+966 5x xxx xxxx">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary px-4"><i class="bi bi-check-lg me-1"></i>Create Customer</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Edit Customer Modal -->
    <div class="modal fade" id="editCustomerModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-md">
            <div class="modal-content">
                <form id="editCustomerForm" onsubmit="submitEditCustomer(event)">
                    <div class="modal-header">
                        <h5 class="modal-title fw-bold"><i class="bi bi-pencil-square text-primary me-2"></i>Edit Customer Account</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body p-4">
                        <input type="hidden" id="editCustId">
                        <div class="d-flex align-items-center justify-content-between mb-3 pb-2 border-bottom">
                            <span class="small text-muted">Editing Customer Record</span>
                            <span class="badge bg-primary-subtle text-primary border" id="editCustIdBadge">#0</span>
                        </div>
                        <div class="row g-3">
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Company / Organization Name *</label>
                                <input type="text" id="editCustCompanyName" class="form-control" required>
                            </div>
                            <div class="col-12">
                                <label class="form-label small fw-semibold">Primary Contact Name</label>
                                <input type="text" id="editCustContactName" class="form-control">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Email Address</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-envelope"></i></span>
                                    <input type="email" id="editCustContactEmail" class="form-control">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label small fw-semibold">Phone Number</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-telephone"></i></span>
                                    <input type="text" id="editCustContactPhone" class="form-control">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-light" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary px-4"><i class="bi bi-check-lg me-1"></i>Save Changes</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let allDeals = [];
        let allCustomers = [];
        let currentQuickFilter = 'ALL';
        let currentDashboardPresales = '';

        let stageChartInstance = null;
        let vendorChartInstance = null;
        let presalesChartInstance = null;
        let taskChartInstance = null;

        const newModal = new bootstrap.Modal(document.getElementById('newDealModal'));
        const editModal = new bootstrap.Modal(document.getElementById('editDealModal'));
        const newCustomerModal = new bootstrap.Modal(document.getElementById('newCustomerModal'));
        const editCustomerModal = new bootstrap.Modal(document.getElementById('editCustomerModal'));

        document.addEventListener('DOMContentLoaded', () => {
            loadAllData();
            loadDashboardMetrics();
        });

        async function loadAllData() {
            await Promise.all([fetchDeals(), fetchCustomers()]);
            renderMetrics();
            populateVendorFilter();
            applyFilters();
            applyCustomerFilters();
            populateCustomerDatalist();
            loadDashboardMetrics();
        }

        async function fetchDeals() {
            try {
                const res = await fetch('/api/deals');
                allDeals = await res.json();
            } catch (err) {
                console.error("Error fetching deals:", err);
            }
        }

        async function fetchCustomers() {
            try {
                const res = await fetch('/api/customers');
                allCustomers = await res.json();
            } catch (err) {
                console.error("Error fetching customers:", err);
            }
        }

        function formatCurrency(num) {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(num || 0);
        }

        function renderMetrics() {
            const totalPipeline = allDeals
                .filter(d => d.stage !== 'Closed-Lost')
                .reduce((acc, d) => acc + (d.estimated_value || 0), 0);
            
            const activeDeals = allDeals.filter(d => d.stage !== 'Closed-Won' && d.stage !== 'Closed-Lost').length;
            
            const wonValue = allDeals
                .filter(d => d.stage === 'Closed-Won')
                .reduce((acc, d) => acc + (d.estimated_value || 0), 0);

            document.getElementById('statTotalPipeline').textContent = formatCurrency(totalPipeline);
            document.getElementById('statActiveDeals').textContent = activeDeals;
            document.getElementById('statWonValue').textContent = formatCurrency(wonValue);
            document.getElementById('statCustomerCount').textContent = allCustomers.length;

            const dealsBadge = document.getElementById('dealsTabBadge');
            if (dealsBadge) dealsBadge.textContent = allDeals.length;
            const custBadge = document.getElementById('customersTabBadge');
            if (custBadge) custBadge.textContent = allCustomers.length;
        }

        async function loadDashboardMetrics() {
            try {
                const url = currentDashboardPresales 
                    ? `/api/metrics/dashboard?presales=${encodeURIComponent(currentDashboardPresales)}`
                    : '/api/metrics/dashboard';
                const res = await fetch(url);
                const data = await res.json();
                renderDashboardData(data);
            } catch (err) {
                console.error("Failed to load dashboard metrics:", err);
            }
        }

        function setDashboardPresalesFilter(lead, btn) {
            currentDashboardPresales = lead;
            document.querySelectorAll('#dashPresalesFilterGroup .filter-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            loadDashboardMetrics();
        }

        function renderDashboardData(data) {
            // 1. Executive KPIs
            document.getElementById('dashKpiTotalPipeline').textContent = formatCurrency(data.total_pipeline);
            document.getElementById('dashKpiTotalDeals').textContent = `${data.total_deals_count} Total Deals`;
            
            document.getElementById('dashKpiActivePipeline').textContent = formatCurrency(data.active_pipeline);
            document.getElementById('dashKpiActiveDeals').textContent = `${data.active_deals_count} In Progress`;
            
            document.getElementById('dashKpiWonValue').textContent = formatCurrency(data.won_value);
            document.getElementById('dashKpiWinRate').textContent = `${data.win_rate}% Win Rate (${data.won_count} Won)`;
            
            document.getElementById('dashKpiAvgDeal').textContent = formatCurrency(data.avg_deal_value);
            document.getElementById('dashKpiAccounts').textContent = data.customer_count;
            
            const taskData = data.task_metrics || {};
            document.getElementById('dashKpiActiveTasks').textContent = `${taskData.active_tasks || 0} Active`;
            const blockersCount = (taskData.blockers || []).length;
            const blockersEl = document.getElementById('dashKpiBlockers');
            if (blockersCount > 0) {
                blockersEl.innerHTML = `<span class="text-danger fw-bold"><i class="bi bi-exclamation-triangle-fill me-1"></i>${blockersCount} Blocker${blockersCount !== 1 ? 's' : ''}</span>`;
            } else {
                blockersEl.innerHTML = `<span class="text-success"><i class="bi bi-check-circle me-1"></i>0 Blockers</span>`;
            }

            // 2. Charts
            renderStageChart(data.stage_breakdown || {});
            renderVendorChart(data.vendor_breakdown || {});
            renderPresalesChart(data.presales_breakdown || {});
            renderTaskChart(taskData);

            // 3. Top Deals Table
            renderTopDealsTable(data.top_deals || []);

            // 4. Blockers List
            renderBlockersList(taskData.blockers || []);
        }

        function renderStageChart(stages) {
            const canvas = document.getElementById('stageChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (stageChartInstance) stageChartInstance.destroy();

            const labels = Object.keys(stages);
            const values = labels.map(k => stages[k].value);
            const counts = labels.map(k => stages[k].count);

            const colors = [
                '#6366f1', // Discovery
                '#0284c7', // Gathering Requirements
                '#eab308', // RFP / Tender
                '#f59e0b', // PoC
                '#8b5cf6', // Proposal
                '#10b981', // Closed-Won
                '#ef4444', // Closed-Lost
            ];

            stageChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Pipeline Value ($)',
                        data: values,
                        backgroundColor: colors,
                        borderRadius: 6,
                        borderSkipped: false,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const val = formatCurrency(context.raw);
                                    const count = counts[context.dataIndex];
                                    return ` Value: ${val} (${count} deal${count !== 1 ? 's' : ''})`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(v) {
                                    if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
                                    if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K';
                                    return '$' + v;
                                }
                            }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        function renderVendorChart(vendors) {
            const canvas = document.getElementById('vendorChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (vendorChartInstance) vendorChartInstance.destroy();

            const labels = Object.keys(vendors);
            if (!labels.length) {
                vendorChartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['No Vendor Data'],
                        datasets: [{ data: [1], backgroundColor: ['#e2e8f0'] }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '65%',
                        plugins: { legend: { position: 'bottom' } }
                    }
                });
                return;
            }

            const values = labels.map(k => vendors[k].value);
            const counts = labels.map(k => vendors[k].count);
            const palette = ['#0d6efd', '#20c997', '#ffc107', '#0dcaf0', '#6f42c1', '#fd7e14', '#e83e8c', '#6c757d'];

            vendorChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: palette.slice(0, labels.length),
                        borderWidth: 2,
                        borderColor: '#ffffff',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { boxWidth: 12, font: { size: 11 } }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const val = formatCurrency(context.raw);
                                    const count = counts[context.dataIndex];
                                    return ` ${context.label}: ${val} (${count} deal${count !== 1 ? 's' : ''})`;
                                }
                            }
                        }
                    }
                }
            });
        }

        function renderPresalesChart(presales) {
            const canvas = document.getElementById('presalesChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (presalesChartInstance) presalesChartInstance.destroy();

            const labels = ['Presales 1', 'Presales 2'];
            const pipelineVals = labels.map(k => presales[k]?.pipeline || 0);
            const wonVals = labels.map(k => presales[k]?.won_value || 0);

            presalesChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Pipeline Value ($)',
                            data: pipelineVals,
                            backgroundColor: '#0d6efd',
                            borderRadius: 6,
                        },
                        {
                            label: 'Won Revenue ($)',
                            data: wonVals,
                            backgroundColor: '#10b981',
                            borderRadius: 6,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', labels: { boxWidth: 12 } },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => ` ${ctx.dataset.label}: ${formatCurrency(ctx.raw)}`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(v) {
                                    if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
                                    if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K';
                                    return '$' + v;
                                }
                            }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        function renderTaskChart(taskData) {
            const canvas = document.getElementById('taskStatusChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (taskChartInstance) taskChartInstance.destroy();

            const statusMap = taskData.status_breakdown || {};
            const labels = Object.keys(statusMap);

            if (!labels.length) {
                taskChartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['No Tasks Recorded'],
                        datasets: [{ data: [1], backgroundColor: ['#e2e8f0'] }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '60%',
                        plugins: { legend: { position: 'bottom' } }
                    }
                });
                return;
            }

            const counts = labels.map(k => statusMap[k]);
            const colorMap = {
                'Completed': '#10b981',
                'In Progress': '#0d6efd',
                'Waiting on Vendor': '#f59e0b',
                'Pending Review': '#8b5cf6',
                'Not Started': '#94a3b8',
            };
            const bgColors = labels.map(k => colorMap[k] || '#64748b');

            taskChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: counts,
                        backgroundColor: bgColors,
                        borderWidth: 2,
                        borderColor: '#ffffff',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '60%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { boxWidth: 12, font: { size: 11 } }
                        }
                    }
                }
            });
        }

        function renderTopDealsTable(deals) {
            const tbody = document.getElementById('dashTopDealsBody');
            if (!tbody) return;
            if (!deals.length) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">No deals registered in pipeline yet.</td></tr>';
                return;
            }

            tbody.innerHTML = deals.map(d => {
                const stageClass = 'badge-' + (d.stage || '').replace(/[\\s\\/]+/g, '-');
                const vendorChips = d.primary_vendors 
                    ? d.primary_vendors.split(',').map(v => `<span class="vendor-chip">${v.trim()}</span>`).join(' ')
                    : '<span class="text-muted small">-</span>';

                return `
                    <tr>
                        <td>
                            <span class="badge bg-primary-subtle text-primary border me-1">#${d.deal_id}</span>
                            <span class="fw-bold text-dark">${d.deal_name}</span>
                        </td>
                        <td>
                            <span class="badge bg-secondary-subtle text-dark border me-1">#${d.customer_id}</span>
                            <span class="fw-semibold text-dark">${d.company_name}</span>
                        </td>
                        <td>${vendorChips}</td>
                        <td><span class="stage-badge ${stageClass}">${d.stage}</span></td>
                        <td class="fw-bold text-primary">${formatCurrency(d.estimated_value)}</td>
                        <td><span class="badge bg-light text-dark border">${d.assigned_presales || 'Unassigned'}</span></td>
                    </tr>
                `;
            }).join('');
        }

        function renderBlockersList(blockers) {
            const container = document.getElementById('dashBlockersList');
            const badge = document.getElementById('dashBlockersCountBadge');
            if (!container) return;

            if (!blockers || !blockers.length) {
                if (badge) {
                    badge.textContent = '0 Blockers';
                    badge.className = 'badge bg-success-subtle text-success border';
                }
                container.innerHTML = `
                    <div class="text-center py-4 text-muted small">
                        <i class="bi bi-check2-circle text-success fs-3 d-block mb-1"></i>
                        All clear! No active management blockers reported.
                    </div>
                `;
                return;
            }

            if (badge) {
                badge.textContent = `${blockers.length} Active Blocker${blockers.length !== 1 ? 's' : ''}`;
                badge.className = 'badge bg-danger text-white border';
            }

            container.innerHTML = blockers.map(b => `
                <div class="border border-danger border-opacity-25 rounded-3 p-2 bg-danger-subtle bg-opacity-25">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <span class="fw-bold text-dark small"><i class="bi bi-exclamation-circle-fill text-danger me-1"></i>${b.task_title}</span>
                        <span class="badge bg-danger-subtle text-danger border" style="font-size: 0.7rem;">${b.assigned_to || 'Unassigned'}</span>
                    </div>
                    <div class="small text-danger fw-semibold"><i class="bi bi-arrow-right-short"></i> ${b.blocker}</div>
                </div>
            `).join('');
        }

        function switchToDealsTab() {
            const dealsTab = document.getElementById('deals-tab');
            if (dealsTab) {
                bootstrap.Tab.getOrCreateInstance(dealsTab).show();
            }
        }

        function populateVendorFilter() {
            const select = document.getElementById('vendorFilter');
            if (!select) return;
            const currentVal = select.value;
            const vendorSet = new Set();
            allDeals.forEach(d => {
                if (d.primary_vendors) {
                    d.primary_vendors.split(',').forEach(v => {
                        const trimmed = v.trim();
                        if (trimmed) vendorSet.add(trimmed);
                    });
                }
            });
            const sortedVendors = Array.from(vendorSet).sort();
            select.innerHTML = '<option value="">All Vendors</option>' + 
                sortedVendors.map(v => `<option value="${v}">${v}</option>`).join('');
            if (currentVal && sortedVendors.includes(currentVal)) {
                select.value = currentVal;
            }
        }

        function setQuickFilter(filterType, btn) {
            currentQuickFilter = filterType;

            // Reset and update button styles
            document.querySelectorAll('#quickFilterGroup .filter-btn, .filter-btn').forEach(b => {
                b.classList.remove('active');
            });

            if (btn) {
                btn.classList.add('active');
            } else {
                let targetId = 'qf-' + filterType;
                if (filterType === 'Gathering Requirements') targetId = 'qf-Req';
                else if (filterType === 'RFP / Tender') targetId = 'qf-RFP';
                else if (filterType === 'Closed-Won') targetId = 'qf-Won';
                else if (filterType === 'Closed-Lost') targetId = 'qf-Lost';
                else if (filterType === 'PRESALES_1') targetId = 'qf-P1';
                else if (filterType === 'PRESALES_2') targetId = 'qf-P2';

                const targetBtn = document.getElementById(targetId);
                if (targetBtn) targetBtn.classList.add('active');
            }

            // Sync with dropdowns where applicable
            const stageFilter = document.getElementById('stageFilter');
            const presalesFilter = document.getElementById('presalesFilter');

            if (['Discovery', 'Gathering Requirements', 'RFP / Tender', 'PoC', 'Proposal', 'Closed-Won', 'Closed-Lost'].includes(filterType)) {
                if (stageFilter) stageFilter.value = filterType;
            } else if (filterType === 'ALL' || filterType === 'ACTIVE') {
                if (stageFilter) stageFilter.value = '';
            }

            if (filterType === 'PRESALES_1') {
                if (presalesFilter) presalesFilter.value = 'Presales 1';
            } else if (filterType === 'PRESALES_2') {
                if (presalesFilter) presalesFilter.value = 'Presales 2';
            } else if (filterType === 'ALL') {
                if (presalesFilter) presalesFilter.value = '';
            }

            // Auto-switch to Deals tab if called from a stat card
            const dealsTabEl = document.getElementById('deals-tab');
            if (dealsTabEl && !dealsTabEl.classList.contains('active')) {
                const tabInstance = bootstrap.Tab.getOrCreateInstance(dealsTabEl);
                tabInstance.show();
            }

            applyFilters();
        }

        function onDropdownFilterChange() {
            const stage = document.getElementById('stageFilter')?.value || '';
            const presales = document.getElementById('presalesFilter')?.value || '';

            document.querySelectorAll('#quickFilterGroup .filter-btn').forEach(b => b.classList.remove('active'));

            if (!stage && !presales) {
                currentQuickFilter = 'ALL';
                const b = document.getElementById('qf-ALL');
                if (b) b.classList.add('active');
            } else if (stage && !presales) {
                currentQuickFilter = stage;
                let targetId = 'qf-' + stage;
                if (stage === 'Gathering Requirements') targetId = 'qf-Req';
                else if (stage === 'RFP / Tender') targetId = 'qf-RFP';
                else if (stage === 'Closed-Won') targetId = 'qf-Won';
                else if (stage === 'Closed-Lost') targetId = 'qf-Lost';
                const b = document.getElementById(targetId);
                if (b) b.classList.add('active');
            } else if (!stage && presales) {
                currentQuickFilter = presales === 'Presales 1' ? 'PRESALES_1' : 'PRESALES_2';
                const b = document.getElementById(presales === 'Presales 1' ? 'qf-P1' : 'qf-P2');
                if (b) b.classList.add('active');
            } else {
                currentQuickFilter = 'CUSTOM';
            }

            applyFilters();
        }

        function resetDealsFilters() {
            currentQuickFilter = 'ALL';
            document.getElementById('searchInput').value = '';
            document.getElementById('stageFilter').value = '';
            document.getElementById('presalesFilter').value = '';
            const vendorFilter = document.getElementById('vendorFilter');
            if (vendorFilter) vendorFilter.value = '';

            document.querySelectorAll('#quickFilterGroup .filter-btn').forEach(b => b.classList.remove('active'));
            const allBtn = document.getElementById('qf-ALL');
            if (allBtn) allBtn.classList.add('active');

            applyFilters();
        }

        function clearDealSearch() {
            document.getElementById('searchInput').value = '';
            applyFilters();
        }

        function applyFilters() {
            const search = (document.getElementById('searchInput')?.value || '').toLowerCase().trim();
            const stage = document.getElementById('stageFilter')?.value || '';
            const presales = document.getElementById('presalesFilter')?.value || '';
            const vendor = (document.getElementById('vendorFilter')?.value || '').toLowerCase().trim();

            const isFiltered = search || stage || presales || vendor || (currentQuickFilter !== 'ALL');
            const resetBtn = document.getElementById('resetDealsFilterBtn');
            if (resetBtn) {
                resetBtn.style.display = isFiltered ? 'inline-block' : 'none';
            }

            const filtered = allDeals.filter(d => {
                // Quick filter condition
                if (currentQuickFilter === 'ACTIVE') {
                    if (d.stage === 'Closed-Won' || d.stage === 'Closed-Lost') return false;
                } else if (currentQuickFilter === 'PRESALES_1') {
                    if (d.assigned_presales !== 'Presales 1') return false;
                } else if (currentQuickFilter === 'PRESALES_2') {
                    if (d.assigned_presales !== 'Presales 2') return false;
                } else if (['Discovery', 'Gathering Requirements', 'RFP / Tender', 'PoC', 'Proposal', 'Closed-Won', 'Closed-Lost'].includes(currentQuickFilter)) {
                    if (d.stage !== currentQuickFilter) return false;
                }

                // Dropdown stage filter
                if (stage && d.stage !== stage) return false;

                // Dropdown presales filter
                if (presales && d.assigned_presales !== presales) return false;

                // Dropdown vendor filter
                if (vendor) {
                    if (!d.primary_vendors || !d.primary_vendors.toLowerCase().includes(vendor)) {
                        return false;
                    }
                }

                // Keyword search
                if (search) {
                    const match = 
                        (d.deal_name && d.deal_name.toLowerCase().includes(search)) || 
                        (d.company_name && d.company_name.toLowerCase().includes(search)) ||
                        (d.primary_vendors && d.primary_vendors.toLowerCase().includes(search)) ||
                        (d.vendor_notes && d.vendor_notes.toLowerCase().includes(search)) ||
                        (d.contact_name && d.contact_name.toLowerCase().includes(search));
                    if (!match) return false;
                }

                return true;
            });

            // Update badge summary
            const summaryBadge = document.getElementById('dealsFilterSummaryBadge');
            if (summaryBadge) {
                if (filtered.length === allDeals.length) {
                    summaryBadge.textContent = `Showing all ${allDeals.length} deals`;
                    summaryBadge.className = 'badge bg-secondary-subtle text-dark border ms-2';
                } else {
                    summaryBadge.textContent = `Showing ${filtered.length} of ${allDeals.length} deals`;
                    summaryBadge.className = 'badge bg-primary text-white border ms-2';
                }
            }

            renderDealsTable(filtered);
        }

        function switchToCustomersTab() {
            const custTabEl = document.getElementById('customers-tab');
            if (custTabEl) {
                const tabInstance = bootstrap.Tab.getOrCreateInstance(custTabEl);
                tabInstance.show();
            }
        }

        function clearCustomerSearch() {
            const input = document.getElementById('customerSearchInput');
            if (input) {
                input.value = '';
                applyCustomerFilters();
            }
        }

        function applyCustomerFilters() {
            const search = (document.getElementById('customerSearchInput')?.value || '').toLowerCase().trim();
            const filtered = allCustomers.filter(c => {
                if (!search) return true;
                return (
                    (c.company_name && c.company_name.toLowerCase().includes(search)) ||
                    (c.contact_name && c.contact_name.toLowerCase().includes(search)) ||
                    (c.contact_email && c.contact_email.toLowerCase().includes(search)) ||
                    (c.contact_phone && c.contact_phone.toLowerCase().includes(search))
                );
            });

            const badge = document.getElementById('customersFilterSummaryBadge');
            if (badge) {
                if (filtered.length === allCustomers.length) {
                    badge.textContent = `Showing all ${allCustomers.length} accounts`;
                    badge.className = 'badge bg-info-subtle text-info border ms-2';
                } else {
                    badge.textContent = `Showing ${filtered.length} of ${allCustomers.length} accounts`;
                    badge.className = 'badge bg-info text-white border ms-2';
                }
            }

            renderCustomerTable(filtered);
        }

        function populateCustomerDatalist() {
            const datalist = document.getElementById('customerDatalist');
            datalist.innerHTML = '';
            allCustomers.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.company_name;
                datalist.appendChild(opt);
            });
        }

        function renderCustomerTable(customers = allCustomers) {
            const tbody = document.getElementById('customersTableBody');
            if (!customers.length) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-muted">No customer accounts match criteria.</td></tr>';
                return;
            }
            tbody.innerHTML = customers.map(c => `
                <tr>
                    <td class="text-muted small">#${c.customer_id}</td>
                    <td class="fw-bold">${c.company_name}</td>
                    <td>${c.contact_name || '<span class="text-muted">-</span>'}</td>
                    <td>${c.contact_email ? `<a href="mailto:${c.contact_email}" class="text-decoration-none">${c.contact_email}</a>` : '<span class="text-muted">-</span>'}</td>
                    <td>${c.contact_phone || '<span class="text-muted">-</span>'}</td>
                    <td><span class="badge bg-light text-dark border">${c.total_deals}</span></td>
                    <td class="fw-semibold text-primary">${formatCurrency(c.total_pipeline)}</td>
                    <td class="text-end">
                        <button class="btn btn-sm btn-outline-primary" onclick="openEditCustomerModal(${c.customer_id})" title="Edit Account Details">
                            <i class="bi bi-pencil-square me-1"></i>Edit
                        </button>
                    </td>
                </tr>
            `).join('');
        }

        function renderDealsTable(deals) {
            const tbody = document.getElementById('dealsTableBody');
            if (!deals.length) {
                tbody.innerHTML = '<tr><td colspan="10" class="text-center py-4 text-muted">No deals match the selected criteria.</td></tr>';
                return;
            }

            tbody.innerHTML = deals.map(d => {
                const stageClass = 'badge-' + d.stage.replace(/[\\s\\/]+/g, '-');
                const vendorChips = d.primary_vendors 
                    ? d.primary_vendors.split(',').map(v => `<span class="vendor-chip">${v.trim()}</span>`).join(' ')
                    : '<span class="text-muted small">-</span>';

                return `
                    <tr>
                        <td class="text-center">
                            <span class="badge bg-primary-subtle text-primary border fw-bold">#${d.deal_id}</span>
                        </td>
                        <td>
                            <div class="fw-bold text-dark">${d.deal_name}</div>
                        </td>
                        <td>
                            <div class="d-flex align-items-center gap-1">
                                <span class="badge bg-secondary-subtle text-dark border">#${d.customer_id}</span>
                                <span class="fw-semibold text-dark">${d.company_name}</span>
                            </div>
                            ${d.contact_name ? `<div class="small text-muted ps-1"><i class="bi bi-person me-1"></i>${d.contact_name}</div>` : ''}
                        </td>
                        <td>${vendorChips}</td>
                        <td><span class="stage-badge ${stageClass}">${d.stage}</span></td>
                        <td class="fw-bold text-dark">${formatCurrency(d.estimated_value)}</td>
                        <td>
                            <span class="badge ${d.assigned_presales === 'Presales 1' ? 'bg-primary-subtle text-primary' : 'bg-info-subtle text-info'} border">
                                ${d.assigned_presales || 'Unassigned'}
                            </span>
                        </td>
                        <td style="max-width: 200px;" class="text-truncate small text-secondary" title="${d.vendor_notes || ''}">
                            ${d.vendor_notes || '<span class="text-muted fst-italic">None</span>'}
                        </td>
                        <td class="small text-muted">${d.updated_at.split(' ')[0]}</td>
                        <td class="text-end">
                            <div class="btn-group btn-group-sm">
                                <button class="btn btn-outline-secondary" onclick="openEditModal(${d.deal_id})" title="Edit Deal">
                                    <i class="bi bi-pencil"></i>
                                </button>
                                <button class="btn btn-outline-secondary dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown"></button>
                                <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                    <li><h6 class="dropdown-header">Fast Advance Stage</h6></li>
                                    <li><a class="dropdown-item" href="#" onclick="quickSetStage(${d.deal_id}, 'Discovery')">Discovery</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="quickSetStage(${d.deal_id}, 'Gathering Requirements')">Gathering Req.</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="quickSetStage(${d.deal_id}, 'RFP / Tender')">RFP / Tender</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="quickSetStage(${d.deal_id}, 'PoC')">PoC</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="quickSetStage(${d.deal_id}, 'Proposal')">Proposal</a></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><a class="dropdown-item text-success fw-semibold" href="#" onclick="quickSetStage(${d.deal_id}, 'Closed-Won')"><i class="bi bi-check2-circle me-1"></i>Closed-Won</a></li>
                                    <li><a class="dropdown-item text-danger" href="#" onclick="quickSetStage(${d.deal_id}, 'Closed-Lost')"><i class="bi bi-x-circle me-1"></i>Closed-Lost</a></li>
                                </ul>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function openNewDealModal() {
            document.getElementById('newDealForm').reset();
            newModal.show();
        }

        async function submitNewDeal(e) {
            e.preventDefault();
            const checkedVendors = Array.from(document.querySelectorAll('.vendor-check:checked')).map(cb => cb.value);

            const payload = {
                company_name: document.getElementById('newCompanyName').value,
                contact_name: document.getElementById('newContactName').value || null,
                contact_email: document.getElementById('newContactEmail').value || null,
                contact_phone: document.getElementById('newContactPhone').value || null,
                deal_name: document.getElementById('newDealName').value,
                estimated_value: parseFloat(document.getElementById('newEstimatedValue').value) || 0.0,
                stage: document.getElementById('newStage').value,
                assigned_presales: document.getElementById('newAssignedPresales').value,
                primary_vendors: checkedVendors,
                vendor_notes: document.getElementById('newVendorNotes').value || ""
            };

            try {
                const res = await fetch('/api/deals', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert('Error creating deal: ' + JSON.stringify(err.detail || err));
                    return;
                }
                newModal.hide();
                await loadAllData();
            } catch (err) {
                console.error("Failed to create deal:", err);
                alert('Connection error occurred while saving.');
            }
        }

        function openEditModal(dealId) {
            const deal = allDeals.find(d => d.deal_id === dealId);
            if (!deal) return;

            document.getElementById('editDealId').value = deal.deal_id;
            document.getElementById('editCompanyName').value = deal.company_name;
            document.getElementById('editDealName').value = deal.deal_name;
            document.getElementById('editEstimatedValue').value = deal.estimated_value;
            document.getElementById('editStage').value = deal.stage;
            document.getElementById('editAssignedPresales').value = deal.assigned_presales || 'Presales 1';
            document.getElementById('editVendorNotes').value = deal.vendor_notes || '';

            const currentVendors = (deal.primary_vendors || '').split(',').map(v => v.trim());
            document.querySelectorAll('.edit-vendor-check').forEach(cb => {
                cb.checked = currentVendors.includes(cb.value);
            });

            editModal.show();
        }

        async function submitEditDeal(e) {
            e.preventDefault();
            const dealId = document.getElementById('editDealId').value;
            const checkedVendors = Array.from(document.querySelectorAll('.edit-vendor-check:checked')).map(cb => cb.value);

            const payload = {
                deal_name: document.getElementById('editDealName').value,
                stage: document.getElementById('editStage').value,
                estimated_value: parseFloat(document.getElementById('editEstimatedValue').value) || 0.0,
                assigned_presales: document.getElementById('editAssignedPresales').value,
                primary_vendors: checkedVendors,
                vendor_notes: document.getElementById('editVendorNotes').value
            };

            try {
                const res = await fetch(`/api/deals/${dealId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert('Error updating deal: ' + JSON.stringify(err.detail || err));
                    return;
                }
                editModal.hide();
                await loadAllData();
            } catch (err) {
                console.error("Failed to update deal:", err);
                alert('Connection error occurred while updating.');
            }
        }

        async function quickSetStage(dealId, newStage) {
            try {
                const res = await fetch(`/api/deals/${dealId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stage: newStage })
                });
                if (res.ok) {
                    await loadAllData();
                } else {
                    alert('Failed to update deal stage');
                }
            } catch (err) {
                console.error("Error setting stage:", err);
            }
        }

        function openNewCustomerModal() {
            document.getElementById('newCustomerForm').reset();
            newCustomerModal.show();
        }

        async function submitNewCustomer(e) {
            e.preventDefault();
            const companyName = document.getElementById('newCustCompanyName').value.trim();
            if (!companyName) {
                alert("Company name is required.");
                return;
            }

            const payload = {
                company_name: companyName,
                contact_name: document.getElementById('newCustContactName').value.trim() || null,
                contact_email: document.getElementById('newCustContactEmail').value.trim() || null,
                contact_phone: document.getElementById('newCustContactPhone').value.trim() || null
            };

            try {
                const res = await fetch('/api/customers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert('Error creating customer: ' + (err.detail || JSON.stringify(err)));
                    return;
                }
                newCustomerModal.hide();
                await loadAllData();
                switchToCustomersTab();
            } catch (err) {
                console.error("Failed to create customer:", err);
                alert('Connection error occurred while saving customer.');
            }
        }

        function openEditCustomerModal(customerId) {
            const cust = allCustomers.find(c => c.customer_id === customerId);
            if (!cust) return;

            document.getElementById('editCustId').value = cust.customer_id;
            document.getElementById('editCustIdBadge').textContent = `Account #${cust.customer_id}`;
            document.getElementById('editCustCompanyName').value = cust.company_name;
            document.getElementById('editCustContactName').value = cust.contact_name || '';
            document.getElementById('editCustContactEmail').value = cust.contact_email || '';
            document.getElementById('editCustContactPhone').value = cust.contact_phone || '';

            editCustomerModal.show();
        }

        async function submitEditCustomer(e) {
            e.preventDefault();
            const customerId = document.getElementById('editCustId').value;
            const companyName = document.getElementById('editCustCompanyName').value.trim();
            if (!companyName) {
                alert("Company name cannot be empty.");
                return;
            }

            const payload = {
                company_name: companyName,
                contact_name: document.getElementById('editCustContactName').value.trim() || null,
                contact_email: document.getElementById('editCustContactEmail').value.trim() || null,
                contact_phone: document.getElementById('editCustContactPhone').value.trim() || null
            };

            try {
                const res = await fetch(`/api/customers/${customerId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const err = await res.json();
                    alert('Error updating customer: ' + (err.detail || JSON.stringify(err)));
                    return;
                }
                editCustomerModal.hide();
                await loadAllData();
            } catch (err) {
                console.error("Failed to update customer:", err);
                alert('Connection error occurred while updating customer.');
            }
        }
    </script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    """Serves the self-contained single-page HTML/Bootstrap CRM dashboard."""
    return HTML_DASHBOARD


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
