import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DB_FILE = Path(__file__).resolve().parent / "crm.db"

VALID_VENDORS = ["HPE", "Veeam", "Dell", "Nutanix", "VMware"]


class DealStage(str, Enum):
    DISCOVERY = "Discovery"
    GATHERING_REQUIREMENTS = "Gathering Requirements"
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS deals (
        deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        deal_name TEXT NOT NULL,
        primary_vendors TEXT,
        stage TEXT NOT NULL CHECK(stage IN (
            'Discovery', 'Gathering Requirements', 'PoC', 'Proposal', 'Closed-Won', 'Closed-Lost'
        )),
        estimated_value REAL NOT NULL DEFAULT 0.0,
        assigned_presales TEXT CHECK(assigned_presales IN ('Presales 1', 'Presales 2')),
        vendor_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
    );
    """)

    # Check if sample data exists; if empty, insert illustrative seed data
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
        clean = re.sub(r"[^\d.]", "", str(v))
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
        clean = re.sub(r"[^\d.]", "", str(v))
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


@app.get("/api/deals", response_model=List[DealOut], tags=["Deals"])
def get_deals(vendor: Optional[str] = Query(None, description="Filter deals by primary vendor (e.g. HPE, Dell)")):
    """Returns all deals with joined customer information, optionally filtered by vendor."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
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
    """
    params = []
    if vendor:
        query += " WHERE LOWER(d.primary_vendors) LIKE ? "
        params.append(f"%{vendor.lower()}%")

    query += " ORDER BY d.updated_at DESC, d.deal_id DESC;"
    cursor.execute(query, params)
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
            <div class="d-flex align-items-center gap-2">
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
                <div class="card stat-card p-3 bg-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Total Pipeline</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statTotalPipeline">$0</h3>
                        </div>
                        <div class="bg-primary-subtle text-primary p-3 rounded-3">
                            <i class="bi bi-cash-stack fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card p-3 bg-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Active Opportunities</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statActiveDeals">0</h3>
                        </div>
                        <div class="bg-warning-subtle text-warning p-3 rounded-3">
                            <i class="bi bi-lightning-charge-fill fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card p-3 bg-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Closed-Won Value</div>
                            <h3 class="fw-bold mb-0 text-success" id="statWonValue">$0</h3>
                        </div>
                        <div class="bg-success-subtle text-success p-3 rounded-3">
                            <i class="bi bi-trophy-fill fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
                <div class="card stat-card p-3 bg-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="text-muted small text-uppercase fw-semibold">Accounts Tracked</div>
                            <h3 class="fw-bold mb-0 text-dark" id="statCustomerCount">0</h3>
                        </div>
                        <div class="bg-info-subtle text-info p-3 rounded-3">
                            <i class="bi bi-building fs-4"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Navigation Tabs & Actions -->
        <div class="card table-card bg-white p-3 mb-4">
            <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
                <ul class="nav nav-pills" id="crmTabs" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link active fw-semibold" id="deals-tab" data-bs-toggle="pill" data-bs-target="#deals-pane" type="button">
                            <i class="bi bi-kanban me-1"></i>Deals Pipeline
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link fw-semibold" id="customers-tab" data-bs-toggle="pill" data-bs-target="#customers-pane" type="button">
                            <i class="bi bi-people me-1"></i>Customers & Accounts
                        </button>
                    </li>
                </ul>

                <!-- Filter & Search Controls -->
                <div class="d-flex flex-wrap align-items-center gap-2" id="filterControls">
                    <div class="input-group input-group-sm" style="width: 210px;">
                        <span class="input-group-text bg-light"><i class="bi bi-search"></i></span>
                        <input type="text" id="searchInput" class="form-control" placeholder="Search deals or accounts..." oninput="applyFilters()">
                    </div>
                    <select id="stageFilter" class="form-select form-select-sm" style="width: 150px;" onchange="applyFilters()">
                        <option value="">All Stages</option>
                        <option value="Discovery">Discovery</option>
                        <option value="Gathering Requirements">Gathering Req.</option>
                        <option value="PoC">PoC</option>
                        <option value="Proposal">Proposal</option>
                        <option value="Closed-Won">Closed-Won</option>
                        <option value="Closed-Lost">Closed-Lost</option>
                    </select>
                    <select id="vendorFilter" class="form-select form-select-sm" style="width: 140px;" onchange="applyFilters()">
                        <option value="">All Vendors</option>
                        <option value="HPE">HPE</option>
                        <option value="Dell">Dell</option>
                        <option value="Veeam">Veeam</option>
                        <option value="Nutanix">Nutanix</option>
                        <option value="VMware">VMware</option>
                    </select>
                    <select id="presalesFilter" class="form-select form-select-sm" style="width: 130px;" onchange="applyFilters()">
                        <option value="">All Presales</option>
                        <option value="Presales 1">Presales 1</option>
                        <option value="Presales 2">Presales 2</option>
                    </select>
                    <button class="btn btn-sm btn-outline-secondary" onclick="loadAllData()" title="Refresh Data">
                        <i class="bi bi-arrow-clockwise"></i>
                    </button>
                </div>
            </div>
        </div>

        <!-- Tab Panes -->
        <div class="tab-content" id="crmTabContent">
            
            <!-- DEALS TAB PANE -->
            <div class="tab-pane fade show active" id="deals-pane" role="tabpanel">
                <div class="card table-card bg-white">
                    <div class="table-responsive">
                        <table class="table table-hover align-middle mb-0" id="dealsTable">
                            <thead class="table-light">
                                <tr>
                                    <th>ID</th>
                                    <th>Deal & Customer</th>
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
                                    <td colspan="9" class="text-center py-4 text-muted">
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
                                </tr>
                            </thead>
                            <tbody id="customersTableBody">
                                <tr>
                                    <td colspan="7" class="text-center py-4 text-muted">
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

    <!-- Bootstrap JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        let allDeals = [];
        let allCustomers = [];

        const newModal = new bootstrap.Modal(document.getElementById('newDealModal'));
        const editModal = new bootstrap.Modal(document.getElementById('editDealModal'));

        document.addEventListener('DOMContentLoaded', () => {
            loadAllData();
        });

        async function loadAllData() {
            await Promise.all([fetchDeals(), fetchCustomers()]);
            renderMetrics();
            applyFilters();
            renderCustomerTable();
            populateCustomerDatalist();
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

        function renderCustomerTable() {
            const tbody = document.getElementById('customersTableBody');
            if (!allCustomers.length) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted">No customer accounts registered yet.</td></tr>';
                return;
            }
            tbody.innerHTML = allCustomers.map(c => `
                <tr>
                    <td class="text-muted small">#${c.customer_id}</td>
                    <td class="fw-bold">${c.company_name}</td>
                    <td>${c.contact_name || '<span class="text-muted">-</span>'}</td>
                    <td>${c.contact_email ? `<a href="mailto:${c.contact_email}" class="text-decoration-none">${c.contact_email}</a>` : '<span class="text-muted">-</span>'}</td>
                    <td>${c.contact_phone || '<span class="text-muted">-</span>'}</td>
                    <td><span class="badge bg-light text-dark border">${c.total_deals}</span></td>
                    <td class="fw-semibold text-primary">${formatCurrency(c.total_pipeline)}</td>
                </tr>
            `).join('');
        }

        function applyFilters() {
            const search = document.getElementById('searchInput').value.toLowerCase().trim();
            const stage = document.getElementById('stageFilter').value;
            const vendor = document.getElementById('vendorFilter').value.toLowerCase();
            const presales = document.getElementById('presalesFilter').value;

            const filtered = allDeals.filter(d => {
                const matchSearch = !search || 
                    d.deal_name.toLowerCase().includes(search) || 
                    d.company_name.toLowerCase().includes(search) ||
                    (d.primary_vendors && d.primary_vendors.toLowerCase().includes(search)) ||
                    (d.vendor_notes && d.vendor_notes.toLowerCase().includes(search));
                
                const matchStage = !stage || d.stage === stage;
                const matchVendor = !vendor || (d.primary_vendors && d.primary_vendors.toLowerCase().includes(vendor));
                const matchPresales = !presales || d.assigned_presales === presales;

                return matchSearch && matchStage && matchVendor && matchPresales;
            });

            renderDealsTable(filtered);
        }

        function renderDealsTable(deals) {
            const tbody = document.getElementById('dealsTableBody');
            if (!deals.length) {
                tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-muted">No deals match the selected criteria.</td></tr>';
                return;
            }

            tbody.innerHTML = deals.map(d => {
                const stageClass = 'badge-' + d.stage.replace(/\\s+/g, '-');
                const vendorChips = d.primary_vendors 
                    ? d.primary_vendors.split(',').map(v => `<span class="vendor-chip">${v.trim()}</span>`).join(' ')
                    : '<span class="text-muted small">-</span>';

                return `
                    <tr>
                        <td class="text-muted small">#${d.deal_id}</td>
                        <td>
                            <div class="fw-bold text-dark">${d.deal_name}</div>
                            <div class="small text-muted d-flex align-items-center gap-1">
                                <i class="bi bi-building"></i> ${d.company_name}
                                ${d.contact_name ? `&bull; <span>${d.contact_name}</span>` : ''}
                            </div>
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
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    """Serves the self-contained single-page HTML/Bootstrap CRM dashboard."""
    return HTML_DASHBOARD


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
