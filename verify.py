"""
Dyck Language Verifier

Uses verifier.py for validation. Compares extracted answer with correct sequence;
extracts longest valid bracket sequence from model output and compares with ground truth.
"""

from generator import Data
from verifier import (
    verify_with_data,
    extract_answer,
    extract_longest_valid_bracket_sequence,
    is_valid_dyck,
)


class DyckLanguageVerifier:
    """Verify Dyck language completion answers."""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct.

        Extracts the longest valid bracket sequence from model output and
        compares with ground truth (full_sequence in data.metadata).
        Correct: exact match. Incorrect: any other output.

        Args:
            data: Game data containing question and metadata with full_sequence.
            test_solution: Model's response (may include reasoning).

        Returns:
            True if score is 1.0 (exact match), False otherwise.
        """
        return verify_with_data(data, test_solution) == 1.0

    def score(self, data: Data, test_solution: str) -> float:
        """Return numeric score: 1.0 for exact match, 0.0 otherwise."""
        return verify_with_data(data, test_solution)
