"""
Predefined Task Catalogs for Antigravity Presales Operations Pipeline.

Enforces strict standardization across Pre-RFP and RFP opportunities,
eliminating open-ended, free-form tasks.
"""

from typing import List, Optional, Tuple
import re

# Exact Pre-RFP Task Catalog (Opportunity Efforts & PO / Category 3)
PRE_RFP_TASKS: List[str] = [
    "Discovery & Technical Requirements Gathering",
    "High-Level Architecture (HLA) Design",
    "Preliminary BoQ & Budgetary Sizing",
    "RFP Specifications Shaping & Advisory",
    "Technical Proposal Draft & Client Review",
]

# Exact RFP Task Catalog (RFP Ownership & Distributed Scope / Categories 1 & 2)
RFP_TASKS: List[str] = [
    "Bid / No-Bid Qualification & Owner Assignment",
    "RFP Decomposition & Scope Breakdown",
    "Clarification Questions Submission",
    "Low-Level Architecture & Technical Write-up",
    "Final BoQ & Vendor Quotations",
    "Technical Compliance Matrix",
    "Proposal Integration & Master Compliance Audit",
    "Final Technical Review & Commercial Handover",
]

# Complete set of all allowed tasks
ALL_ALLOWED_TASKS: List[str] = PRE_RFP_TASKS + RFP_TASKS
ALL_ALLOWED_TASKS_SET = set(ALL_ALLOWED_TASKS)
ALL_ALLOWED_TASKS_LOWER = {t.lower(): t for t in ALL_ALLOWED_TASKS}


# Semantic keyword mapping for resilient matching of Arabic and English speech to canonical catalog items
CATALOG_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    # --- Pre-RFP Tasks ---
    (
        "Discovery & Technical Requirements Gathering",
        [
            "discovery", "gathering requirements", "technical requirements", "site survey", "site visit",
            "زيارة موقع", "زيارة ميدانية", "جمع المتطلبات", "متطلبات فنية", "استكشاف", "مسح ميداني", "جلسة متطلبات",
            "assessment", "customer visit", "زيارة العميل"
        ]
    ),
    (
        "High-Level Architecture (HLA) Design",
        [
            "hla", "high level", "high-level", "high level architecture", "conceptual design", "macro architecture",
            "معمارية مبدئية", "تصميم مبدئي", "معمارية أولية", "تصميم رفيع المستوى", "مخطط مبدئي", "مخطط أولي",
            "تصميم معماري مبدئي", "تصميم معماري", "معماري مبدئي", "معماري أولي", "معمارية"
        ]
    ),
    (
        "Preliminary BoQ & Budgetary Sizing",
        [
            "preliminary boq", "budgetary", "budgetary sizing", "sizing", "rough estimate", "cost estimate",
            "تسعير مبدئي", "جدول كميات مبدئي", "ميزانية تقديرية", "ميزانية مبدئية", "حساب السعات المبدئي",
            "حساب السعات", "حساب سعات", "سعات", "boq مبدئي", "boq تقديري", "حجم تقديري"
        ]
    ),
    (
        "RFP Specifications Shaping & Advisory",
        [
            "shaping", "specifications shaping", "advisory", "tender shaping", "specs advisory",
            "صياغة الكراسة", "توجيه المواصفات", "استشارة فنية", "صياغة الشروط والمواصفات", "تجهيز الكراسة للعميل",
            "طبخ الكراسة", "cook the rfp", "shaping rfp"
        ]
    ),
    (
        "Technical Proposal Draft & Client Review",
        [
            "proposal draft", "client review", "draft review", "draft proposal", "technical draft",
            "مسودة العرض", "مراجعة العميل", "مسودة المقترح", "عرض فني مبدئي", "مراجعة المسودة الفنية مع العميل"
        ]
    ),

    # --- RFP Tasks ---
    (
        "Bid / No-Bid Qualification & Owner Assignment",
        [
            "bid / no-bid", "bid no-bid", "bid no bid", "bid qualification", "owner assignment", "qualification",
            "قرار المشاركة", "تأهيل المناقصة", "تعيين مسؤول المناقصة", "تسمية المسؤول", "تأهيل ودراسة الجدوى",
            "bid decision", "tender qualification"
        ]
    ),
    (
        "RFP Decomposition & Scope Breakdown",
        [
            "decomposition", "scope breakdown", "scope distribution", "breakdown", "distributing scope", "scope decomposition",
            "تفكيك الكراسة", "توزيع النطاق", "توزيع نطاق العمل", "تجزئة الكراسة", "تحليل نطاق المناقصة", "توزيع المهام للشركاء",
            "نطاق الشركاء"
        ]
    ),
    (
        "Clarification Questions Submission",
        [
            "clarification", "clarifications", "inquiries", "questions submission", "tender questions",
            "استفسارات", "أسئلة المناقصة", "تقديم الاستفسارات", "إرسال الاستفسارات", "استفسارات فنية", "توضيحات"
        ]
    ),
    (
        "Low-Level Architecture & Technical Write-up",
        [
            "low-level", "lla", "low level architecture", "technical write-up", "detailed architecture", "detailed design",
            "معمارية تفصيلية", "كتابة العرض الفني", "المعمارية التفصيلية", "التصميم التفصيلي", "صياغة الحل الفني التفصيلي"
        ]
    ),
    (
        "Final BoQ & Vendor Quotations",
        [
            "final boq", "vendor quotations", "quotations", "vendor pricing", "partner quotes", "distributor pricing",
            "عروض الموردين", "تسعير الموردين", "جدول الكميات النهائي", "عروض الأسعار من الموزع", "تسعير الشركاء",
            "boq نهائي", "تسعير نهائي", "عروض أسعار الموردين"
        ]
    ),
    (
        "Technical Compliance Matrix",
        [
            "compliance matrix", "technical compliance", "compliance sheet", "compliance table",
            "جدول المطابقة", "مصفوفة المطابقة", "مطابقة المواصفات", "جدول الامتثال الفني", "مطابقة الشروط الفنية"
        ]
    ),
    (
        "Proposal Integration & Master Compliance Audit",
        [
            "proposal integration", "compliance audit", "master compliance", "integration audit", "final assembly",
            "تجميع العرض", "دمج المقترح", "تدقيق المطابقة الشامل", "المراجعة النهائية للامتثال", "دمج المستندات الفنية"
        ]
    ),
    (
        "Final Technical Review & Commercial Handover",
        [
            "commercial handover", "technical review", "handover", "final sign-off", "submission handover", "commercial",
            "تسليم العرض التجاري", "المراجعة الفنية النهائية", "تسليم الفريق التجاري", "التسليم للاعتماد", "تسليم التسعير",
            "الفريق التجاري", "للفريق التجاري", "تسليم العرض", "تسليم فني", "تسليم نهائي", "تسليم", "اعتماد تجاري"
        ]
    ),
]


def is_pre_rfp_category(deal_category: Optional[str]) -> bool:
    """Checks whether the deal category corresponds to Opportunity Efforts (Pre-RFP)."""
    if not deal_category:
        return False
    c = str(deal_category).strip().lower()
    # Explicit RFP ownership or distributed scope are strictly RFP
    if "1-" in c or "owner" in c or "prime" in c or "2-" in c or "distribut" in c or "tender" in c or "rfp" in c:
        if not ("3-" in c or "opp" in c or "effort" in c):
            return False
    return "3-" in c or "opp" in c or "effort" in c or bool(re.search(r"\bpo\b", c)) or "general" in c or c == "general_action"


def normalize_to_catalog(raw_title: Optional[str], deal_category: Optional[str] = None) -> str:
    """
    Normalizes any candidate task title strictly to one of the predefined catalog items.
    If already a valid catalog item, returns it directly.
    Otherwise, applies keyword/semantic matching against spoken Arabic & English terminology.
    If no rule matches, deterministically falls back to the canonical entry step based on deal category.
    """
    if not raw_title:
        return PRE_RFP_TASKS[0] if is_pre_rfp_category(deal_category) else RFP_TASKS[0]

    title_clean = str(raw_title).strip()

    # 1. Exact match
    if title_clean in ALL_ALLOWED_TASKS_SET:
        return title_clean

    # 2. Case-insensitive exact match
    title_lower = title_clean.lower()
    if title_lower in ALL_ALLOWED_TASKS_LOWER:
        return ALL_ALLOWED_TASKS_LOWER[title_lower]

    # 3. Keyword / Semantic matching
    best_match: Optional[str] = None
    best_score: int = 0

    for canonical_task, keywords in CATALOG_KEYWORD_RULES:
        # Check if canonical name is partially contained
        if canonical_task.lower() in title_lower:
            return canonical_task

        for kw in keywords:
            if kw.lower() in title_lower:
                # Give higher weight to longer, more specific keywords
                score = len(kw)
                # Boost match if it aligns with deal category domain
                if is_pre_rfp_category(deal_category) and canonical_task in PRE_RFP_TASKS:
                    score += 10
                elif not is_pre_rfp_category(deal_category) and canonical_task in RFP_TASKS:
                    score += 10

                if score > best_score:
                    best_score = score
                    best_match = canonical_task

    if best_match:
        return best_match

    # 4. Semantic category-aware fallbacks for ambiguous terms
    if "boq" in title_lower or "جدول كميات" in title_lower or "تسعير" in title_lower:
        if is_pre_rfp_category(deal_category) or "مبدئي" in title_lower or "تقديري" in title_lower or "preliminary" in title_lower:
            return "Preliminary BoQ & Budgetary Sizing"
        else:
            return "Final BoQ & Vendor Quotations"

    if any(w in title_lower for w in ["معمار", "architecture", "تصميم", "hla", "lla", "مخطط"]):
        if is_pre_rfp_category(deal_category) or any(w in title_lower for w in ["مبدئي", "أولي", "اولى", "high"]):
            return "High-Level Architecture (HLA) Design"
        else:
            return "Low-Level Architecture & Technical Write-up"

    # 5. Fallback default based on opportunity category
    if is_pre_rfp_category(deal_category):
        return "Discovery & Technical Requirements Gathering"
    else:
        return "RFP Decomposition & Scope Breakdown"
