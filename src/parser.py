"""
parser.py — Extract a multiple-choice answer (A/B/C/D) from Qwen's
free-text response.

Qwen doesn't reliably answer with just a letter, it might say "The
answer is B", "I think B because...", "Option C", or occasionally
nothing parseable at all. This handles the common patterns and
returns a clear failure signal rather than guessing.
"""

import re

# Ordered most to least specific. First match wins.
PATTERNS = [
    r'(?i)\banswer\s*(?:is|:)?\s*\(?([A-D])\)?\b',
    r'(?i)\b(?:option|choice)\s*\(?([A-D])\)?\b',
    r'(?i)^\s*\(?([A-D])\)?[\.\):]?\s*$',   # just "B", "B.", "(B)"
    r'(?i)^\s*\(?([A-D])\)?\s',              # response starts with the letter
]

FALLBACK_PATTERN = r'\b([A-D])\b'  # last resort: first standalone letter anywhere


def extract_answer(raw_response: str | None) -> tuple[str | None, bool]:
    """Returns (extracted_letter, succeeded)."""
    if not raw_response or not raw_response.strip():
        return None, False

    text = raw_response.strip()

    for pattern in PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper(), True

    match = re.search(FALLBACK_PATTERN, text)
    if match:
        return match.group(1).upper(), True

    return None, False


if __name__ == "__main__":
    # Quick local check against realistic Qwen response shapes.
    cases = [
        ("The answer is B", "B"),
        ("I believe B is correct because it accounts for...", "B"),
        ("Option C", "C"),
        ("(A)", "A"),
        ("D.", "D"),
        ("This is a complex question with no clear single answer.", None),
    ]
    for text, expected in cases:
        got, ok = extract_answer(text)
        status = "PASS" if got == expected else "FAIL"
        print(f"{status}: {text!r} -> {got} (expected {expected})")