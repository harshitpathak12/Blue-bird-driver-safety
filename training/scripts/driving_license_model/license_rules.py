"""
Rule-based validation for Indian driving licence (based on OCR text).

Supports standard format used across Indian states, e.g.:
- "Indian Union Driving Licence" (Bihar, etc.)
- "Issued by Government of [State]"
- License number: State code (2 letters) + RTO (2 digits) + Year (4) + 7 digits
  e.g. BR22 20250006557, MH02 20191234567, DL-06-20190001234
- Fields: Name, Date Of Birth, Blood Group, Validity (NT/TR), Issue Date, Address, etc.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

from utils.logger import get_logger

_rules_log = get_logger(__name__)


# Keywords that suggest an Indian driving licence (Union/state format)
LICENCE_KEYWORDS = [
    # Standard title (all states use this or similar)
    "indian union driving licence",
    "indian union driving license",
    "union driving licence",
    "union driving license",
    "driving licence",
    "driving license",
    "driving licen",
    "driving lic",
    # Issuing authority
    "issued by government",
    "government of",
    "government of india",
    "republic of india",
    # Common field labels
    "licence no",
    "license no",
    "licence no.",
    "license no.",
    "licence number",
    "license number",
    "issue date",
    "date of issue",
    "date of first issue",
    "validity (nt)",
    "validity (tr)",
    "valid from",
    "valid till",
    "valid to",
    "date of birth",
    "date of birth:",
    "blood group",
    "blood group :",
    "name:",
    "address:",
    "address",
    "son/daughter/wife",
    "father's name",
    "father name",
    "holder's signature",
    "organ donor",
    "issuing authority",
    "motor vehicle",
    "rto",
    "regional transport",
    "transport office",
    "authorisation to drive",
    "categories",
    "permanent address",
]

# Indian DL number: 2 letter state + 2 digit RTO + (space/hyphen) + 4 digit year + 7 digits
# e.g. BR22 20250006557, MH02 20191234567, DL-06-20190001234, HR0619850034761
DL_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}\s*\d{7}\b",
    re.IGNORECASE,
)
# With spaces between parts: BR 22 2025 0006557
DL_NUMBER_SPACED = re.compile(
    r"\b[A-Z]{2}\s+\d{2}\s+(?:19|20)\d{2}\s*\d{5,7}\b",
    re.IGNORECASE,
)
# Hyphen form at bottom of card: BR-D2217017627
DL_NUMBER_HYPHEN = re.compile(
    r"\b[A-Z]{2}\s*-\s*[A-Z]?\d{10,15}\b",
    re.IGNORECASE,
)
# Hyphen-separated: DL-06-2019-0001234, MH-12-2017-0001234
DL_NUMBER_HYPHEN_SEP = re.compile(
    r"\b[A-Z]{2}[\s\-]+\d{2}[\s\-]+(?:19|20)\d{2}[\s\-]*\d{5,7}\b",
    re.IGNORECASE,
)
# Preceded by label: "Licence No" / "DL No" / "No." followed by colon/space then the number
DL_NUMBER_LABELLED = re.compile(
    r"(?:licen[cs]e\s*no\.?|dl\s*no\.?|no\.?)\s*[:\s]\s*([A-Z]{2}[\s\-/]*\d[\d\s\-/]{8,20})",
    re.IGNORECASE,
)
# Relaxed: 2 letters + digits (catches OCR errors / variants)
DL_NUMBER_RELAXED = re.compile(
    r"\b[A-Z]{2}[\s\-]?\d{2,}[\s\-]?\d{4,}\b",
    re.IGNORECASE,
)
# Very relaxed: 2 letters followed by any combination of digits/spaces/hyphens totaling 10+
DL_NUMBER_VERY_RELAXED = re.compile(
    r"\b[A-Z]{2}[\s\-/]?\d[\d\s\-/]{9,18}\d\b",
    re.IGNORECASE,
)

# Dates near validity labels (Indian DL: DD/MM/YYYY or DD-MM-YYYY)
_DATE_DMY = re.compile(
    r"\b(\d{1,2})[/-](\d{1,2})[/-]((19|20)\d{2})\b",
    re.IGNORECASE,
)
_VALIDITY_LINE = re.compile(
    r"(?is).{0,120}?(?:valid(?:ity)?\s*\(?\s*(?:tr|nt)\s*\)?|valid\s*till|valid\s*to|valid\s*until|valid\s*from)"
    r".{0,120}?(\d{1,2})[/-](\d{1,2})[/-]((19|20)\d{2})",
)


def _parse_date(d: int, m: int, y: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def extract_validity_expiry_dates(raw: str) -> list[date]:
    """
    Collect candidate expiry dates from OCR text (validity / valid till / TR / NT lines).
    Returns parsed dates; caller may take the latest as licence end date.
    """
    if not raw or not raw.strip():
        return []
    found: list[date] = []
    lines = raw.splitlines()
    # Same-line or next-line dates after Validity / Valid till / TR / NT (common OCR layout)
    for i, line in enumerate(lines):
        ln = line.lower()
        if not any(
            k in ln
            for k in (
                "valid",
                "till",
                "validity",
                "(tr)",
                "(nt)",
                "non-transport",
                "transport",
            )
        ):
            continue
        block = line
        if i + 1 < len(lines):
            block = line + " " + lines[i + 1]
        for m in _DATE_DMY.finditer(block):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pd = _parse_date(d, mo, y)
            if pd and 1990 <= y <= 2100:
                found.append(pd)
    # Block match across wrapped text
    for m in _VALIDITY_LINE.finditer(raw):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        pd = _parse_date(d, mo, y)
        if pd:
            found.append(pd)
    return found


def _latest_expiry(dates: list[date]) -> date | None:
    if not dates:
        return None
    return max(dates)


def _all_dmy_dates_in_text(raw: str) -> list[date]:
    """Collect every DD/MM/YYYY (or -) date in OCR text — fallback when validity lines are garbled."""
    found: list[date] = []
    for m in _DATE_DMY.finditer(raw):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        pd = _parse_date(d, mo, y)
        if pd and 1990 <= y <= 2100:
            found.append(pd)
    return found


def _merge_expiry_candidates(raw: str) -> list[date]:
    """Targeted validity-line dates plus any dates found in the full transcription."""
    targeted = extract_validity_expiry_dates(raw)
    loose = _all_dmy_dates_in_text(raw)
    merged: list[date] = []
    seen: set[date] = set()
    for d in targeted + loose:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return merged


def _has_confident_dl_number(
    numbers: list[str],
    has_strict: bool,
    has_spaced: bool,
    has_hyphen: bool,
    has_relaxed: bool,
) -> bool:
    """Require a DL number that matches a strong pattern (not keyword-only)."""
    if not numbers:
        return False
    return bool(has_strict or has_spaced or has_hyphen or has_relaxed)


def validate_indian_dl(ocr_text: str) -> dict[str, Any]:
    """
    Rule-based validation for Indian driving licence using OCR text.
    Matches standard format: Indian Union Driving Licence, Issued by Government of [State],
    licence number (e.g. BR22 20250006557), Validity, Name, DOB, Blood Group, Address, etc.

    Returns:
        {
            "label": "valid" | "invalid" | "unknown",
            "confidence": float (0-1),
            "reason": str (short reason for display),
        }
    """
    if not ocr_text or not ocr_text.strip():
        return {
            "label": "unknown",
            "confidence": 0.0,
            "reason": "No text from licence",
            "dl_numbers": [],
            "validity_end": None,
        }

    text = ocr_text.strip().lower()
    raw = ocr_text.strip()
    reasons_ok = []
    reasons_fail = []

    # 1) Must have at least one licence-related keyword
    found_keyword = any(kw in text for kw in LICENCE_KEYWORDS)
    if not found_keyword:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "reason": "Not a driving licence",
            "dl_numbers": [],
            "validity_end": None,
        }
    reasons_ok.append("Licence keyword found")

    # 2) Look for licence number pattern (any supported format)
    numbers: list[str] = []
    for pat in (DL_NUMBER_PATTERN, DL_NUMBER_SPACED, DL_NUMBER_HYPHEN,
                DL_NUMBER_HYPHEN_SEP, DL_NUMBER_RELAXED, DL_NUMBER_VERY_RELAXED):
        for m in pat.finditer(raw):
            num = m.group(0).strip()
            if num not in numbers:
                numbers.append(num)
    # Also try labelled patterns (capture group 1)
    for m in DL_NUMBER_LABELLED.finditer(raw):
        num = m.group(1).strip()
        if num and num not in numbers:
            numbers.append(num)

    has_strict = any(DL_NUMBER_PATTERN.search(n) for n in numbers)
    has_spaced = any(DL_NUMBER_SPACED.search(n) for n in numbers)
    has_hyphen = any(DL_NUMBER_HYPHEN.search(n) or DL_NUMBER_HYPHEN_SEP.search(n) for n in numbers)
    has_relaxed = any(DL_NUMBER_RELAXED.search(n) or DL_NUMBER_VERY_RELAXED.search(n) for n in numbers)

    if has_strict or has_spaced or has_hyphen:
        reasons_ok.append("DL number format OK")
    elif has_relaxed:
        reasons_ok.append("DL number (relaxed) OK")
    else:
        reasons_fail.append("No licence number pattern")

    # 3) Reject if too short (likely not a full licence)
    if len(text) < 15:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "reason": "Too little text",
            "dl_numbers": numbers,
            "validity_end": None,
        }

    if reasons_fail:
        return {
            "label": "invalid",
            "confidence": 0.3,
            "reason": "; ".join(reasons_fail),
            "dl_numbers": numbers,
            "validity_end": None,
        }

    today = date.today()
    has_dl = _has_confident_dl_number(numbers, has_strict, has_spaced, has_hyphen, has_relaxed)
    merged_dates = _merge_expiry_candidates(raw)
    latest = _latest_expiry(merged_dates)
    validity_end_iso = latest.isoformat() if latest else None

    if not has_dl:
        return {
            "label": "invalid",
            "confidence": 0.0,
            "reason": "No valid DL number extracted",
            "dl_numbers": numbers,
            "validity_end": validity_end_iso,
        }

    # Qwen often transcribes dates on different lines than our narrow "validity" regexes expect.
    # If no date anywhere in the text but DL number + keywords look good, accept as valid with lower confidence
    # (avoids false "invalid" when OCR layout is readable but dates are not regex-matched).
    if latest is None and has_dl:
        if os.environ.get("DL_DEBUG_RULES", "").strip().lower() in ("1", "true", "yes"):
            _rules_log.info(
                "validate_indian_dl: no parsed dates; text_len=%s preview=%r",
                len(raw),
                (raw[:240] + "…") if len(raw) > 240 else raw,
            )
        return {
            "label": "valid",
            "confidence": 0.72,
            "reason": "DL number and licence text OK; validity date not parsed from OCR",
            "dl_numbers": numbers,
            "validity_end": None,
        }

    # Expired if validity end is strictly before today
    if latest < today:
        return {
            "label": "invalid",
            "confidence": 0.85,
            "reason": f"Licence expired (valid till {latest.strftime('%d-%m-%Y')})",
            "dl_numbers": numbers,
            "validity_end": validity_end_iso,
        }

    # VALID only with DL number + non-expired validity
    confidence = 0.9 if (has_strict or has_spaced) else 0.85 if has_hyphen else 0.8
    return {
        "label": "valid",
        "confidence": confidence,
        "reason": f"Valid till {latest.strftime('%d-%m-%Y')}",
        "dl_numbers": numbers,
        "validity_end": validity_end_iso,
    }
