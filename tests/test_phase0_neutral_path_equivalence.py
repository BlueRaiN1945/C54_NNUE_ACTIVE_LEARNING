"""Behavioral-equivalence regression test between the Phase 1 neutral-path
implementation (medium_pc_audit.paths) and the frozen Phase 0
'relative-neutral-path' format checker (medium_pc_audit.schemas.registry).

Phase 0 is NOT modified by this test -- it only reads registry's private
format-checker function to compare behavior against Phase 1's independent
implementation. The two are intentionally separate modules (see paths.py's
docstring); this test is the regression guard that keeps them from
silently drifting apart on the rules they are both meant to enforce.

Control-character rejection is a Phase-1-only addition (see
tests/test_paths.py::ControlCharacterRejectionTests) and is deliberately
excluded from this shared corpus, since Phase 0 was never specified to
reject control characters and must not be modified to do so now.
"""

import unittest

from medium_pc_audit.paths import require_neutral_path
from medium_pc_audit.schemas.registry import _is_relative_neutral_path

# (value, expected_acceptance) -- covers only the rule set both
# implementations were designed to share: relativity, forward slashes, no
# drive letters, no backslashes, no '..'/empty segments. Every case here
# must be judged identically by both implementations.
SHARED_CORPUS = [
    ("a", True),
    ("a/b/c.txt", True),
    ("a.b_c-d/e.f", True),
    ("nested/deeply/nested/path.bin", True),
    ("", False),
    ("/a/b", False),
    ("//a/b", False),
    ("a\\b", False),
    ("C:/a/b", False),
    ("c:/a/b", False),
    ("a/../b", False),
    ("../a", False),
    ("a/..", False),
    ("a//b", False),
    ("a/", False),
    ("/", False),
]


def _phase1_accepts(value) -> bool:
    try:
        require_neutral_path(value)
        return True
    except (ValueError, TypeError):
        return False


class Phase0Phase1NeutralPathEquivalenceTests(unittest.TestCase):
    def test_shared_corpus_judged_identically(self):
        for value, expected in SHARED_CORPUS:
            with self.subTest(value=value):
                phase0_result = _is_relative_neutral_path(value)
                phase1_result = _phase1_accepts(value)

                self.assertEqual(
                    phase0_result, expected,
                    f"Phase 0 diverged from expected corpus value for {value!r}",
                )
                self.assertEqual(
                    phase1_result, expected,
                    f"Phase 1 diverged from expected corpus value for {value!r}",
                )
                self.assertEqual(
                    phase0_result, phase1_result,
                    f"Phase 0/Phase 1 disagree on {value!r}: phase0={phase0_result} phase1={phase1_result}",
                )


if __name__ == "__main__":
    unittest.main()
