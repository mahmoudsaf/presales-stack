import difflib
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text by removing tashkeel, tatweel, and normalizing
    various alef, taa marbuta, and yaa forms.
    """
    if not text:
        return ""
    t = str(text).strip().lower()
    # Remove Arabic diacritics (tashkeel & tatweel/kashida)
    t = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", t)
    # Normalize Alefs
    t = re.sub(r"[أإآٱ]", "ا", t)
    # Normalize Taa Marbuta
    t = re.sub(r"[ة]", "ه", t)
    # Normalize Alif Maqsura to Yaa
    t = re.sub(r"[ى]", "ي", t)
    # Replace punctuation and special characters with spaces
    t = re.sub(r"[^\w\s]", " ", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


NOISE_WORDS = {
    # Arabic legal/entity corporate descriptors
    "شركة", "مؤسسة", "مؤسسه", "مجموعة", "مجموعه", "هيئة", "هيئه", "وزارة", "وزاره",
    "جامعة", "جامعه", "بنك", "مصرف", "مركز", "عمادة", "عماده", "وكالة", "وكاله",
    "امانة", "أمانة", "امانه", "صندوق", "المحدودة", "المحدوده", "مساهمة", "مساهمه",
    "القابضة", "القابضه", "للتجارة", "للتجاره", "للمقاولات", "لتقنية", "للاتصالات",
    "العامة", "العامه", "الوطنية", "الوطنيه", "السعودية", "السعوديه",
    # English corporate descriptors and common prepositions
    "company", "co", "corp", "corporation", "ltd", "limited", "llc", "inc",
    "group", "holding", "holdings", "ministry", "authority", "university",
    "bank", "center", "the", "of", "for", "and", "saudi"
}

GENERIC_DEAL_WORDS = {
    "مشروع", "مناقصة", "مناقصه", "منافسة", "منافسه", "كراسة", "كراسه", "فرصة", "فرصه",
    "طلب", "توريد", "تقديم", "عقد", "عملية", "عمليه", "rfp", "tender", "deal",
    "project", "opportunity", "client", "general", "department", "لصالح", "جديدة", "جديده",
    "باسم", "اسم", "خاصة", "خاصه", "خدمات", "اعمال", "أعمال"
}

CUSTOMER_ALIASES: Dict[str, List[str]] = {
    "kacst": [
        "مدينة الملك عبد العزيز للعلوم والتقنية",
        "مدينة الملك عبدالعزيز للعلوم والتقنية",
        "كاست", "kacst"
    ],
    "kaust": [
        "جامعة الملك عبد الله للعلوم والتقنية",
        "جامعة الملك عبدالله للعلوم والتقنية",
        "كاوست", "kaust"
    ],
    "ksu": [
        "جامعة الملك سعود", "ksu", "king saud university"
    ],
    "kau": [
        "جامعة الملك عبد العزيز", "جامعة الملك عبدالعزيز", "kau", "king abdulaziz university"
    ],
    "moi": [
        "وزارة الداخلية", "وزارة الداخليه", "moi", "ministry of interior"
    ],
    "moh": [
        "وزارة الصحة", "وزارة الصحه", "moh", "ministry of health"
    ],
    "mod": [
        "وزارة الدفاع", "mod", "ministry of defense"
    ],
    "moe": [
        "وزارة التعليم", "moe", "ministry of education"
    ],
    "mof": [
        "وزارة المالية", "وزارة الماليه", "mof", "ministry of finance"
    ],
    "mofa": [
        "وزارة الخارجية", "وزارة الخارجيه", "mofa", "ministry of foreign affairs"
    ],
    "mhrsd": [
        "وزارة الموارد البشرية", "وزارة الموارد البشريه",
        "وزارة الموارد البشرية والتنمية الاجتماعية",
        "ministry of human resources", "mhrsd"
    ],
    "stc": [
        "شركة الاتصالات السعودية", "الاتصالات السعودية", "stc", "saudi telecom"
    ],
    "aramco": [
        "أرامكو", "ارامكو", "أرامكو السعودية", "ارامكو السعوديه",
        "aramco", "saudi aramco"
    ],
    "sama": [
        "البنك المركزي", "البنك المركزي السعودي", "مؤسسة النقد",
        "مؤسسة النقد العربي السعودي", "ساما", "sama"
    ],
    "almarai": [
        "المراعي", "شركة المراعي", "almarai", "al marai"
    ],
    "albilad": [
        "بنك البلاد", "البلاد", "albilad", "bank albilad"
    ],
    "riyad_bank": [
        "بنك الرياض", "riyad bank", "riyadh bank"
    ],
    "rajhi": [
        "مصرف الراجحي", "الراجحي", "alrajhi", "al rajhi bank"
    ],
    "tvtc": [
        "المؤسسة العامة للتدريب التقني والمهني", "tvtc"
    ],
    "ndha": [
        "الهيئة الوطنية للأمن السيبراني", "ndha"
    ],
    "weqaa": [
        "وقاء", "مركز وقاء", "weqaa"
    ]
}


def get_alias_cluster(text: str) -> Optional[str]:
    t_norm = normalize_arabic(text)
    for cluster_id, aliases in CUSTOMER_ALIASES.items():
        for a in aliases:
            a_norm = normalize_arabic(a)
            if t_norm == a_norm or (len(t_norm) >= 3 and t_norm == a.lower()):
                return cluster_id
    return None


def get_customer_tokens(text: str) -> Tuple[str, List[str], List[str]]:
    norm = normalize_arabic(text)
    raw_tokens = norm.split()
    core_tokens = [w for w in raw_tokens if w not in NOISE_WORDS]
    stripped_tokens = []
    for w in core_tokens:
        if w.startswith("ال") and len(w) > 3:
            stripped_tokens.append(w[2:])
        else:
            stripped_tokens.append(w)
    return norm, core_tokens, stripped_tokens


def extract_deal_scope_tokens(deal_name: str, company_name: str = "") -> set:
    """
    Extracts core project scope tokens from a deal name by removing customer name tokens,
    noise words, and generic deal words (like مناقصة, كراسة, مشروع, rfp, tender).
    """
    if not deal_name:
        return set()
    norm = normalize_arabic(deal_name)
    tokens = set(re.split(r"[\s\-:،,]+", norm))
    comp_tokens = set(re.split(r"[\s\-:،,]+", normalize_arabic(company_name))) if company_name else set()
    stripped_comp = set()
    for ct in comp_tokens:
        if ct.startswith("ال") and len(ct) > 3:
            stripped_comp.add(ct[2:])
        stripped_comp.add(ct)
    exclude = stripped_comp | comp_tokens | NOISE_WORDS | GENERIC_DEAL_WORDS
    
    result = set()
    for t in tokens:
        clean = t
        if clean.startswith("ال") and len(clean) > 3:
            clean = clean[2:]
        if len(clean) >= 3 and clean not in exclude and t not in exclude:
            result.add(clean)
    return result


def compute_customer_similarity(name1: str, name2: str) -> float:
    """
    Computes a normalized similarity score between 0.0 and 1.0 between two customer/company names.
    Supports Arabic/English transliterations, noise reduction, and enterprise acronyms.
    """
    if not name1 or not name2:
        return 0.0

    # 1. Exact raw match (case insensitive)
    if name1.strip().lower() == name2.strip().lower():
        return 1.0

    norm1, core1, strip1 = get_customer_tokens(name1)
    norm2, core2, strip2 = get_customer_tokens(name2)

    # 2. Exact normalized match
    if norm1 == norm2:
        return 0.99

    # 3. Compact match (ignoring whitespace differences like "عبد العزيز" vs "عبدالعزيز")
    if norm1.replace(" ", "") == norm2.replace(" ", ""):
        return 0.98

    # 4. Alias cluster match (bilingual / abbreviations)
    c1 = get_alias_cluster(name1)
    c2 = get_alias_cluster(name2)
    if c1 and c2 and c1 == c2:
        return 0.96

    if c1:
        for a in CUSTOMER_ALIASES[c1]:
            a_norm = normalize_arabic(a)
            if a_norm == norm2 or (len(a_norm) > 3 and a_norm in norm2):
                return 0.95
    if c2:
        for a in CUSTOMER_ALIASES[c2]:
            a_norm = normalize_arabic(a)
            if a_norm == norm1 or (len(a_norm) > 3 and a_norm in norm1):
                return 0.95

    # 5. Core stripped tokens comparison (excluding corporate noise words)
    str1 = " ".join(strip1)
    str2 = " ".join(strip2)

    if str1 and str2:
        if str1 == str2:
            return 0.93

        set1, set2 = set(strip1), set(strip2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Substring containment of significant entity name
        if (len(str1) >= 4 and str1 in str2) or (len(str2) >= 4 and str2 in str1):
            ratio_len = min(len(str1), len(str2)) / max(len(str1), len(str2))
            if ratio_len >= 0.5:
                return 0.85 + (0.10 * ratio_len)

        if jaccard >= 0.6:
            return 0.80 + (0.15 * jaccard)

        seq_core = difflib.SequenceMatcher(None, str1, str2).ratio()
        if seq_core >= 0.82:
            return 0.80 + (0.15 * seq_core)

    seq_full = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return seq_full


def find_similar_customer(
    company_name: str,
    cursor: sqlite3.Cursor,
    threshold: float = 0.80
) -> Optional[sqlite3.Row]:
    """
    Searches all existing customers in the database for the highest similarity match.
    Returns the customer sqlite3.Row if max score >= threshold, otherwise None.
    """
    if not company_name:
        return None
    clean_target = company_name.strip()

    cursor.execute("SELECT customer_id, company_name, contact_name, contact_email, contact_phone FROM customers;")
    all_customers = cursor.fetchall()
    if not all_customers:
        return None

    best_match = None
    best_score = 0.0

    for cust in all_customers:
        score = compute_customer_similarity(clean_target, cust["company_name"])
        if score > best_score:
            best_score = score
            best_match = cust

    if best_score >= threshold and best_match:
        return best_match
    return None
