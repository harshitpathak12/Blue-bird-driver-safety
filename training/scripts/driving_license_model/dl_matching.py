"""
Match registration name/age to OCR text from Indian driving licence.
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_NAME_INLINE = re.compile(
    r"(?im)(?:^|\n)\s*name\s*[:.]?\s*(.+?)(?:\n|$)",
)
_DOB = re.compile(
    r"(?i)(?:date\s*of\s*birth|d\.?\s*o\.?\s*b\.?|birth)\s*[:.]?\s*(\d{1,2})[/-](\d{1,2})[/-]((19|20)\d{2})",
)
_DOB_LOOSE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-]((19|20)\d{2})\b")


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _age_from_dob(dob: date, today: date | None = None) -> int:
    today = today or date.today()
    y = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        y -= 1
    return y


def extract_holder_name(ocr_text: str) -> str | None:
    """Best-effort holder name from full OCR (Indian DL layout)."""
    if not ocr_text or not ocr_text.strip():
        return None
    raw = ocr_text.strip()
    m = _NAME_INLINE.search(raw)
    if m:
        line = m.group(1).strip()
        line = re.sub(r"\s+", " ", line)
        if 3 <= len(line) <= 80 and not re.match(r"^\d", line):
            return line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.match(r"(?i)^name\s*[:]?\s*$", ln) and i + 1 < len(lines):
            cand = lines[i + 1]
            if 3 <= len(cand) <= 80:
                return cand
    return None


def extract_dob(ocr_text: str) -> date | None:
    """First plausible DOB near DOB keywords, else first DMY date in text."""
    if not ocr_text:
        return None
    m = _DOB.search(ocr_text)
    if not m:
        m = _DOB_LOOSE.search(ocr_text)
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def names_match(registration_name: str, ocr_text: str, extracted_name: str | None) -> bool:
    reg = normalize_name(registration_name)
    if not reg:
        return False
    if extracted_name:
        ext = normalize_name(extracted_name)
        if not ext:
            return False
        ratio = SequenceMatcher(None, reg, ext).ratio()
        if ratio >= 0.72:
            return True
        if reg in ext or ext in reg:
            return True
        # token overlap (handles middle name omissions)
        rt = set(reg.split())
        et = set(ext.split())
        if rt and et and len(rt & et) >= max(1, min(len(rt), len(et)) - 1):
            return True
    # fallback: registration tokens appear in OCR
    full = normalize_name(ocr_text)
    for tok in reg.split():
        if len(tok) >= 3 and tok in full:
            return True
    return False


def age_matches_registration(reg_age: int | None, ocr_text: str) -> tuple[bool, bool]:
    """
    Returns (age_check_ok, had_age_to_compare).
    If registration has no age, returns (True, False) — skip age check.
    """
    if reg_age is None:
        return True, False
    dob = extract_dob(ocr_text)
    if dob is None:
        return False, True
    computed = _age_from_dob(dob)
    return abs(computed - reg_age) <= 1, True


def registration_matches_dl(
    registration_name: str,
    registration_age: int | None,
    ocr_text: str,
) -> dict[str, Any]:
    """
    Returns { ok, name_ok, age_ok, holder_name, reason }.
    """
    holder = extract_holder_name(ocr_text)
    name_ok = names_match(registration_name, ocr_text, holder)
    age_ok, age_checked = age_matches_registration(registration_age, ocr_text)
    if not name_ok:
        logger.debug(
            "registration_matches_dl: name_mismatch holder=%r",
            holder,
        )
        return {
            "ok": False,
            "name_ok": False,
            "age_ok": age_ok,
            "holder_name": holder,
            "reason": "name_mismatch",
        }
    if age_checked and not age_ok:
        logger.debug("registration_matches_dl: age_mismatch")
        return {
            "ok": False,
            "name_ok": True,
            "age_ok": False,
            "holder_name": holder,
            "reason": "age_mismatch",
        }
    return {
        "ok": True,
        "name_ok": True,
        "age_ok": age_ok if age_checked else True,
        "holder_name": holder,
        "reason": "ok",
    }
