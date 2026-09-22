"""Tests for medium_pc_audit.paths -- Phase 1."""

import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.paths import from_neutral_path, require_neutral_path, to_neutral_path


class RequireNeutralPathTests(unittest.TestCase):
    def test_accepts_simple_relative_path(self):
        self.assertEqual(require_neutral_path("a/b/c.txt"), "a/b/c.txt")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            require_neutral_path("")

    def test_rejects_backslash(self):
        with self.assertRaises(ValueError):
            require_neutral_path("a\\b")

    def test_rejects_leading_slash(self):
        with self.assertRaises(ValueError):
            require_neutral_path("/a/b")

    def test_rejects_windows_drive_prefix(self):
        with self.assertRaises(ValueError):
            require_neutral_path("C:/a/b")

    def test_rejects_dotdot_segment(self):
        with self.assertRaises(ValueError):
            require_neutral_path("a/../b")

    def test_rejects_empty_segment(self):
        with self.assertRaises(ValueError):
            require_neutral_path("a//b")


class ControlCharacterRejectionTests(unittest.TestCase):
    """Phase-1-only hardening: SHA256SUMS.txt is one 'hash  path' line per
    file, so a path containing CR/LF could inject a fake extra line, and a
    NUL could truncate some readers -- all must be rejected outright.
    """

    def test_rejects_cr_lf_nul_and_other_control_characters(self):
        control_chars = ["\r", "\n", "\x00", "\x01", "\x1f", "\x7f"]
        for ch in control_chars:
            with self.subTest(char=repr(ch)):
                with self.assertRaises(ValueError):
                    require_neutral_path(f"a{ch}b")

    def test_accepts_ordinary_printable_characters(self):
        # Sanity check: the control-character rule must not overreach.
        self.assertEqual(require_neutral_path("a b/c-d_e.f"), "a b/c-d_e.f")


class ToFromNeutralPathTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            nested = base / "a" / "b.txt"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("x", encoding="utf-8")

            neutral = to_neutral_path(nested, base)
            self.assertEqual(neutral, "a/b.txt")

            resolved_back = from_neutral_path(neutral, base)
            self.assertEqual(resolved_back.resolve(), nested.resolve())


if __name__ == "__main__":
    unittest.main()
