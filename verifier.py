"""
Dyck Language Verifier

Validation Rules:
1. Balanced: Every opening bracket has a matching closing bracket
2. Proper Nesting: Brackets are properly nested (no crossing)
3. Unique Completion: Only one valid way to complete the sequence

Scoring:
- Correct: 1.0 - Exact match of complete valid sequence
- Incorrect: 0.0 - Any other output
"""

import re
from typing import Optional, Tuple

# Bracket pairs (opening, closing) - same as generator
BRACKET_PAIRS = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("⟨", "⟩"),
    ("⟦", "⟧"),
    ("⦃", "⦄"),
    ("⦅", "⦆"),
]

OPEN_BRACKETS = {p[0] for p in BRACKET_PAIRS}
CLOSE_BRACKETS = {p[1] for p in BRACKET_PAIRS}
OPEN_TO_CLOSE = {p[0]: p[1] for p in BRACKET_PAIRS}
CLOSE_TO_OPEN = {p[1]: p[0] for p in BRACKET_PAIRS}
ALL_BRACKETS = OPEN_BRACKETS | CLOSE_BRACKETS


def is_valid_dyck(sequence: str) -> bool:
    """
    Check if a sequence is valid Dyck language.
    Validation: Balanced + Proper nesting (no crossing).
    """
    if not sequence:
        return True
    stack = []
    for char in sequence:
        if char in OPEN_BRACKETS:
            stack.append(char)
        elif char in CLOSE_BRACKETS:
            if not stack or stack[-1] != CLOSE_TO_OPEN[char]:
                return False
            stack.pop()
        else:
            return False  # non-bracket character
    return len(stack) == 0


def get_required_closing(prefix: str) -> str:
    """
    Compute the unique minimal closing sequence for a prefix (LIFO).
    Unique completion: only one valid way to complete.
    """
    stack = []
    for char in prefix:
        if char in OPEN_BRACKETS:
            stack.append(char)
        elif char in CLOSE_BRACKETS:
            if stack and stack[-1] == CLOSE_TO_OPEN[char]:
                stack.pop()
    required = []
    while stack:
        open_char = stack.pop()
        required.append(OPEN_TO_CLOSE[open_char])
    return "".join(required)


def extract_longest_valid_bracket_sequence(text: str) -> str:
    """
    Extract the longest contiguous valid Dyck sequence from model output.
    Scans substrings and returns the longest that passes is_valid_dyck.
    """
    text = text.strip()
    # First try: take only bracket characters in order
    bracket_chars = "".join(c for c in text if c in ALL_BRACKETS)
    if not bracket_chars:
        return ""

    # Find longest valid Dyck substring (O(n^2) check)
    n = len(bracket_chars)
    best = ""
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = bracket_chars[i:j]
            if is_valid_dyck(sub) and len(sub) > len(best):
                best = sub
    return best


def extract_answer(model_output: str) -> str:
    """
    Extract answer from model output.
    Prefer text after 'FINAL ANSWER:', then longest valid bracket sequence.
    """
    text = (model_output or "").strip()
    # Prefer content after FINAL ANSWER:
    if "FINAL ANSWER:" in text:
        part = text.split("FINAL ANSWER:")[-1].strip()
        part = part.split("\n")[0].strip()
        # Return longest valid bracket sequence in that part, or whole part if valid
        longest = extract_longest_valid_bracket_sequence(part)
        if longest:
            return longest
        # If no valid bracket sequence, use part (might be just brackets without spaces)
        only_brackets = "".join(c for c in part if c in ALL_BRACKETS)
        if is_valid_dyck(only_brackets):
            return only_brackets
        return part
    return extract_longest_valid_bracket_sequence(text)


def verify(
    model_output: str,
    correct_full_sequence: str,
    question_prefix: Optional[str] = None,
) -> float:
    """
    Compare extracted answer with correct sequence.
    Returns 1.0 if correct (exact match of complete valid sequence), 0.0 otherwise.

    Args:
        model_output: Raw model output (may contain reasoning + FINAL ANSWER: ...).
        correct_full_sequence: Ground truth full Dyck sequence (prefix + closing).
        question_prefix: If provided, model answer may be only the closing part;
            then we compare question_prefix + extracted == correct_full_sequence.

    Returns:
        1.0 for exact match, 0.0 otherwise.
    """
    correct = (correct_full_sequence or "").strip()
    extracted = extract_answer(model_output)

    # Case 1: Extracted is the full sequence
    if extracted and extracted == correct:
        return 1.0

    # Case 2: Extracted is only the closing part; prepend question prefix
    if question_prefix is not None:
        prefix = question_prefix.strip()
        if extracted and prefix + extracted == correct:
            return 1.0
        # Model may output only closing brackets (no valid Dyck substring by itself)
        raw_brackets = "".join(c for c in (model_output or "") if c in ALL_BRACKETS)
        if raw_brackets and prefix + raw_brackets == correct:
            return 1.0

    return 0.0


def verify_with_data(data, model_output: str) -> float:
    """
    Compare extracted answer with correct sequence using task data.

    Use when you have a Data instance (or dict) with metadata containing
    "full_sequence" and optionally "question_sequence". Extracts longest
    valid bracket sequence from model output and compares with ground truth.

    Args:
        data: Task data with .metadata["full_sequence"] (and optionally
              .metadata["question_sequence"]). Can be generator.Data or dict.
        model_output: Raw model response (may include reasoning, extra text).

    Returns:
        1.0 if exact match of complete valid sequence, 0.0 otherwise.
    """
    if hasattr(data, "metadata"):
        meta = data.metadata
    elif isinstance(data, dict):
        meta = data.get("metadata", {})
    else:
        return 0.0
    full = meta.get("full_sequence", "")
    prefix = meta.get("question_sequence", "")
    if not full:
        return 0.0
    return verify(model_output, full, question_prefix=prefix or None)


def verify_with_prefix(
    model_output: str,
    question_prefix: str,
    correct_closing: str,
) -> Tuple[float, str]:
    """
    Verify when you have question prefix and correct closing part separately.
    Returns (score, extracted_answer).
    """
    correct_full = question_prefix + correct_closing
    score = verify(model_output, correct_full, question_prefix=question_prefix)
    extracted = extract_answer(model_output)
    return score, extracted


# ---------------------------------------------------------------------------
# Examples (Easy / Medium / Hard)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Easy: n_types=2, length=8. Question: "([(", Answer: "([()]))"
    easy_prefix = "([("
    easy_closing = get_required_closing(easy_prefix)  # ")]))"
    easy_full = easy_prefix + easy_closing
    assert is_valid_dyck(easy_full)
    assert verify(easy_full, easy_full) == 1.0
    assert verify("FINAL ANSWER: " + easy_closing, easy_full, question_prefix=easy_prefix) == 1.0

    # Medium: question "([{[{(", unique completion from get_required_closing
    medium_prefix = "([{[{("
    medium_closing = get_required_closing(medium_prefix)
    medium_full = medium_prefix + medium_closing
    assert is_valid_dyck(medium_full)
    assert verify(medium_full, medium_full) == 1.0
    # Note: model output of only closing part has no valid Dyck substring, so we test full-sequence match

    # Hard: question "([{<[{<(", unique completion
    hard_prefix = "([{<[{<("
    hard_closing = get_required_closing(hard_prefix)
    hard_full = hard_prefix + hard_closing
    assert is_valid_dyck(hard_full)
    assert verify(hard_full, hard_full) == 1.0

    # Incorrect
    assert verify("([()]", easy_full) == 0.0
    assert verify("wrong", easy_full) == 0.0
    assert verify("", easy_full) == 0.0

    print("All verifier checks passed.")
