"""
PII filter: masks personal data BEFORE it reaches any LLM, log, database or tracing service.

    mask_pii("Email me at anu@gmail.com or call +91 98765 43210")
    -> ("Email me at [EMAIL] or call [PHONE]", {"email": 1, "phone": 1})

What it detects (and how it avoids false alarms):
    [EMAIL]     email addresses
    [CARD]      13-19 digit card numbers that pass the Luhn checksum (the check banks use)
    [AADHAAR]   12-digit Aadhaar numbers that pass the Verhoeff checksum (the check UIDAI uses)
    [PAN]       PAN cards: ABCDE1234F, where the 4th letter is a real PAN holder type
    [PASSPORT]  Indian passports (K1234567), or any code with a digit right after the word "passport"
    [PHONE]     Indian mobiles (+91 / 0 / 10 digits starting 6-9) and international "+" numbers

Not covered: names and street addresses (no fixed pattern; detecting them would need
another model, which would mean sending the personal data to it first).

Only the COUNTS are ever stored or logged ({"email": 1}), never the values.

Try it:  python pii_filter.py
"""

import re
import sys

# =========================================================
# Checksums
# =========================================================

# Verhoeff tables (used by Aadhaar's last digit)
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    """True if the number's last digit is a correct Verhoeff check digit."""
    check = 0
    for i, digit in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(digit)]]
    return check == 0


def luhn_valid(digits: str) -> bool:
    """True if the number passes the Luhn checksum (every card number does)."""
    total = 0
    for i, digit in enumerate(reversed(digits)):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# =========================================================
# Patterns
# =========================================================

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 13-19 digits, spaces or dashes allowed between them
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

# 1234 5678 9012 (first digit is never 0 or 1)
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")

# ABCDE1234F - the 4th letter is the holder type (P = person, C = company, H, F, A, T, B, L, J, G)
PAN = re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b", re.IGNORECASE)

# Indian passport: one letter + 7 digits (K1234567)
PASSPORT_INDIA = re.compile(r"\b[A-PR-WY][1-9]\d{6}\b", re.IGNORECASE)
# Any passport number written right after the word "passport" (must contain a digit)
PASSPORT_AFTER_WORD = re.compile(
    r"(\bpassport\s*(?:no\.?|number|num|#)?\s*(?:is|:|-)?\s*)((?=[A-Z0-9]*\d)[A-Z0-9]{6,9})\b",
    re.IGNORECASE,
)

# Indian mobile: +91 98765 43210 / 098765 43210 / 9876543210
PHONE_INDIA = re.compile(r"(?<![\d+])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# International: starts with + and a country code, e.g. +1 (415) 555-0132, +44 20 7946 0958
PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s-]?\(?\d{1,4}\)?){2,5}(?!\d)")


# =========================================================
# Main function
# =========================================================

def mask_pii(text: str) -> tuple[str, dict]:
    """Returns (masked text, counts of what was masked). The order matters:
    checked numbers (cards, Aadhaar) go first so the phone pattern can't take their digits."""
    if not text:
        return text, {}
    counts = {}

    def replace(kind, label, check=None):
        def sub(match):
            if check and not check(match):
                return match.group(0)  # looks similar but isn't real PII: leave it
            counts[kind] = counts.get(kind, 0) + 1
            return label
        return sub

    def digits(match):
        return re.sub(r"\D", "", match.group(0))

    text = EMAIL.sub(replace("email", "[EMAIL]"), text)
    text = CARD.sub(replace("card", "[CARD]", lambda m: 13 <= len(digits(m)) <= 19 and luhn_valid(digits(m))), text)
    text = AADHAAR.sub(replace("aadhaar", "[AADHAAR]", lambda m: verhoeff_valid(digits(m))), text)
    text = PAN.sub(replace("pan", "[PAN]"), text)

    def passport_after_word(match):
        counts["passport"] = counts.get("passport", 0) + 1
        return match.group(1) + "[PASSPORT]"
    text = PASSPORT_AFTER_WORD.sub(passport_after_word, text)
    text = PASSPORT_INDIA.sub(replace("passport", "[PASSPORT]"), text)

    text = PHONE_INDIA.sub(replace("phone", "[PHONE]"), text)
    text = PHONE_INTL.sub(replace("phone", "[PHONE]", lambda m: 8 <= len(digits(m)) <= 15), text)
    return text, counts


def describe(counts: dict) -> str:
    """{"email": 1, "phone": 2} -> '1 email, 2 phone numbers'"""
    names = {
        "email": ("email", "emails"), "phone": ("phone number", "phone numbers"),
        "aadhaar": ("Aadhaar number", "Aadhaar numbers"), "pan": ("PAN", "PANs"),
        "passport": ("passport number", "passport numbers"), "card": ("card number", "card numbers"),
    }
    parts = [f"{n} {names[k][0] if n == 1 else names[k][1]}" for k, n in counts.items()]
    return ", ".join(parts)


# =========================================================
# Demo
# =========================================================

TESTS = [
    # (text, what should be masked)
    ("Plan a trip to Goa, email me at anu.maram@gmail.com", {"email": 1}),
    ("Call me on +91 98765 43210 or 9123456789", {"phone": 2}),
    ("My US number is +1 (415) 555-0132", {"phone": 1}),
    ("Aadhaar 2345 6789 0124 for the hotel check-in", {"aadhaar": 1}),
    ("My PAN is ABCPE1234F", {"pan": 1}),
    ("Passport K1234567, flying to Dubai", {"passport": 1}),
    ("UK passport number: 533401372", {"passport": 1}),
    ("Book it on my card 4111 1111 1111 1111", {"card": 1}),
    ("Card 4111-1111-1111-1111 expires 12/28", {"card": 1}),
    # Things that must NOT be masked
    ("Plan a 5 day trip from Delhi to Goa under ₹45,000", {}),
    ("Flight AI9540 departs 2026-10-02 at 05:00", {}),
    ("Budget ₹1,50,000 for 2 people, 7 nights", {}),
    ("Aadhaar-like but wrong checksum: 2345 6789 0123", {}),
    ("Card-like but fails Luhn: 4111 1111 1111 1112", {}),
    ("Do Indian passport holders need a visa for Thailand?", {}),
    ("Hotel booking ref 8842 for 3 nights in Jaipur", {}),
    ("DEL to GOI on 12 Dec, PNR 4521367890", {}),
    ("Trip for 2 adults and 1 child, ages 34, 31 and 6", {}),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    correct = 0
    for text, expected in TESTS:
        masked, counts = mask_pii(text)
        ok = counts == expected
        correct += ok
        print(f"{'✓' if ok else '✗'} {text}")
        print(f"    -> {masked}" + ("" if ok else f"   (expected {expected}, got {counts})"))
    print(f"\n{correct}/{len(TESTS)} as expected")
