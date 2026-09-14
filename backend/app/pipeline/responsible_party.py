"""
LAYER 4 (NOVELTY #3) -- RESPONSIBLE-PARTY LEGAL RESOLUTION ENGINE

Per the architecture: "Rule-based decision tree encoding the manufacturer
vs. packer vs. importer vs. brand-owner distinction from the PC Rules --
e.g. if address present but role-keyword ('Mktd by,' 'Packed by,' 'Mfd
by') ambiguous, flag for manual review rather than false-passing."

A label that only says "Marketed by BrandCo" with no manufacturer, packer,
or importer named anywhere is a DIFFERENT and more serious violation than
one with a clearly named manufacturer -- a naive keyword-presence checker
cannot distinguish these because BOTH labels "have an address." This
module encodes that distinction explicitly.
"""

import re
from typing import Optional
from app.pipeline.types import ResponsiblePartyResult

ROLE_PATTERNS = [
    ("MANUFACTURER", [r"manufactured\s*by", r"mfd\.?\s*by", r"manufacturer\s*:"], True),
    ("PACKER", [r"packed\s*by", r"pkd\.?\s*by", r"packer\s*:"], True),
    ("IMPORTER", [r"imported\s*by", r"importer\s*:"], True),
    ("MARKETER", [r"marketed\s*by", r"mktd\.?\s*by", r"brand\s*owner", r"distributed\s*by"], False),
]

PIN_CODE_PATTERN = re.compile(r"\b\d{6}\b")
ADDRESS_PATTERN = re.compile(r"[A-Za-z0-9,.\-/\s]{15,120}\b\d{6}\b")


def resolve_responsible_party(ocr_raw_text: str) -> ResponsiblePartyResult:
    found_roles = []
    for role, patterns, is_responsible in ROLE_PATTERNS:
        for pattern in patterns:
            match = re.search(pattern, ocr_raw_text, re.IGNORECASE)
            if match:
                found_roles.append((role, match.group(0), is_responsible))
                break

    has_pin = bool(PIN_CODE_PATTERN.search(ocr_raw_text))
    address_match = ADDRESS_PATTERN.search(ocr_raw_text)
    address = address_match.group(0).strip() if address_match else None

    if not found_roles:
        return ResponsiblePartyResult(
            role="NOT_FOUND",
            matched_phrase=None,
            address=address,
            is_legally_responsible=False,
            reasoning=(
                "No manufacturer/packer/importer/marketer declaration phrase was detected anywhere in the "
                "extracted text. Rule 6 of the PC Rules requires this declaration on every principal display "
                "panel — this is a CRITICAL violation, not a minor omission."
            ),
        )

    has_responsible_role = any(r[2] for r in found_roles)
    if not has_responsible_role:
        marketer = found_roles[0]
        return ResponsiblePartyResult(
            role="MARKETER",
            matched_phrase=marketer[1],
            address=address,
            is_legally_responsible=False,
            reasoning=(
                f'Only a "{marketer[1]}" declaration was found. The PC Rules require the actual manufacturer, '
                "packer, or importer to be identified — a marketer/brand-owner declaration alone does not "
                "satisfy Rule 6 in the general case. This should be flagged for manual review rather than "
                "auto-passed."
            ),
        )

    if len(found_roles) > 1:
        responsible = next(r for r in found_roles if r[2])
        phrases = ", ".join(r[1] for r in found_roles)
        pin_note = "" if has_pin else " Note: no 6-digit PIN code detected near the address — address may be incomplete."
        return ResponsiblePartyResult(
            role=responsible[0],
            matched_phrase=responsible[1],
            address=address,
            is_legally_responsible=True,
            reasoning=(
                f'Multiple role declarations found ({phrases}). Primary legally responsible party identified as '
                f'{responsible[0]} via "{responsible[1]}".{pin_note}'
            ),
        )

    single = found_roles[0]
    pin_note = (
        " A complete address with PIN code was found nearby."
        if has_pin
        else " Warning: no 6-digit PIN code detected — address may be incomplete per Rule 6(1)(a)."
    )
    return ResponsiblePartyResult(
        role=single[0],
        matched_phrase=single[1],
        address=address,
        is_legally_responsible=True,
        reasoning=f'Identified as {single[0]} via "{single[1]}".{pin_note}',
    )
