"""Tests for medium_pc_audit.identity -- Phase 0.

Run with: python -m unittest -v tests.test_identity
(stdlib-only; no pytest, no third-party dependency of any kind.)
"""

import unittest

from medium_pc_audit.identity import (
    InvalidIdentityComponent,
    canonicalize,
    compute_audit_id,
    compute_identity_sha256,
    compute_matchup_key,
    compute_run_id,
)


class ComputeAuditIdTests(unittest.TestCase):
    def test_zero_padded_4_digits(self):
        self.assertEqual(compute_audit_id(11), "AUDIT_V0011")
        self.assertEqual(compute_audit_id(0), "AUDIT_V0000")
        self.assertEqual(compute_audit_id(12345), "AUDIT_V12345")

    def test_rejects_bool(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_audit_id(True)
        with self.assertRaises(InvalidIdentityComponent):
            compute_audit_id(False)

    def test_rejects_negative(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_audit_id(-1)

    def test_rejects_non_int(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_audit_id("11")
        with self.assertRaises(InvalidIdentityComponent):
            compute_audit_id(11.0)


class ComputeMatchupKeyTests(unittest.TestCase):
    def test_composition(self):
        self.assertEqual(compute_matchup_key("AUDIT_V0011", "TEACHER_V0"), "AUDIT_V0011__TEACHER_V0")

    def test_rejects_empty_audit_id(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key("", "TEACHER_V0")

    def test_rejects_empty_opponent_id(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key("AUDIT_V0011", "")

    def test_rejects_invalid_characters(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key("AUDIT_V0011", "bad id with spaces")
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key("AUDIT_V0011", "bad/slash")
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key("AUDIT_V0011", "bad\\backslash")

    def test_rejects_non_string(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_matchup_key(None, "TEACHER_V0")


class ComputeRunIdTests(unittest.TestCase):
    def test_attempt_padded_3_digits(self):
        self.assertEqual(compute_run_id("AUDIT_V0011__TEACHER_V0", 1), "AUDIT_V0011__TEACHER_V0__run001")
        self.assertEqual(compute_run_id("AUDIT_V0011__TEACHER_V0", 12), "AUDIT_V0011__TEACHER_V0__run012")

    def test_rejects_attempt_zero(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_run_id("AUDIT_V0011__TEACHER_V0", 0)

    def test_rejects_negative_attempt(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_run_id("AUDIT_V0011__TEACHER_V0", -1)

    def test_rejects_bool_attempt(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_run_id("AUDIT_V0011__TEACHER_V0", True)

    def test_rejects_empty_matchup_key(self):
        with self.assertRaises(InvalidIdentityComponent):
            compute_run_id("", 1)


class CanonicalizeTests(unittest.TestCase):
    def test_key_order_independent(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        self.assertEqual(canonicalize(a), canonicalize(b))

    def test_exact_byte_format(self):
        result = canonicalize({"b": 1, "a": "x"})
        self.assertEqual(result, b'{"a":"x","b":1}\n')

    def test_ends_with_exactly_one_newline(self):
        result = canonicalize({"a": 1})
        self.assertTrue(result.endswith(b"\n"))
        self.assertFalse(result.endswith(b"\n\n"))

    def test_non_ascii_not_escaped(self):
        result = canonicalize({"name": "\u00e6\u00f8\u00e5"})
        self.assertIn("\u00e6\u00f8\u00e5".encode("utf-8"), result)
        self.assertNotIn(b"\\u", result)

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            canonicalize({"x": float("nan")})

    def test_rejects_infinity(self):
        with self.assertRaises(ValueError):
            canonicalize({"x": float("inf")})

    def test_exclude_fields_top_level_only(self):
        obj = {"a": 1, "nested": {"a": 2}}
        result = canonicalize(obj, exclude_fields=frozenset({"a"}))
        self.assertIn(b'"a":2', result)
        self.assertNotIn(b'"a":1', result)

    def test_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            canonicalize(["not", "a", "mapping"])


class IdentitySha256Tests(unittest.TestCase):
    """Strengthened per explicit requirement: every hash-relevant top-level field
    must be shown to change identity_sha256 when mutated independently, and every
    excluded field must be shown NOT to change it.
    """

    def _base_match_config(self):
        return {
            "schema_version": "v1",
            "config_id": "AUDIT_V0011__TEACHER_V0__run001",
            "audit_id": "AUDIT_V0011",
            "matchup_key": "AUDIT_V0011__TEACHER_V0",
            "matchup_type": "vs_teacher",
            "candidate": {"artifact_id": "AUDIT_CHAMPION_V0011", "sha256": "a" * 64},
            "opponent": {"artifact_id": "TEACHER_V0", "sha256": "b" * 64},
            "engine": {"pinned_id": "SF_AUDIT_V1", "expected_sha256": "c" * 64},
            "cli": {"pinned_id": "CCC_AUDIT_V1", "expected_sha256": "d" * 64},
            "uci_options": {"Threads": 1, "Hash": 64},
            "time_control": {"nodes": 50000},
            "threads": 1,
            "hash_mb": 64,
            "concurrency": 8,
            "opening_suite": {"artifact_id": "AUDIT_OPENINGS_V1", "sha256": "e" * 64},
            "repeat": True,
            "seed": 12345,
            "sprt": {"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": 4000},
            "created_at": "2026-09-22T12:00:00Z",
            "created_by": "test-fixture",
        }

    def test_deterministic(self):
        obj = self._base_match_config()
        exclude = frozenset({"created_at"})
        self.assertEqual(
            compute_identity_sha256(obj, exclude),
            compute_identity_sha256(obj, exclude),
        )

    def test_every_hash_relevant_field_changes_identity(self):
        base = self._base_match_config()
        exclude = frozenset({"created_at"})
        baseline_hash = compute_identity_sha256(base, exclude)

        mutations = {
            "config_id": "DIFFERENT_CONFIG_ID",
            "audit_id": "AUDIT_V9999",
            "matchup_key": "AUDIT_V0011__CHAMPION_V1",
            "matchup_type": "vs_champion_v1",
            "candidate": {"artifact_id": "OTHER", "sha256": "f" * 64},
            "opponent": {"artifact_id": "OTHER", "sha256": "1" * 64},
            "engine": {"pinned_id": "OTHER", "expected_sha256": "2" * 64},
            "cli": {"pinned_id": "OTHER", "expected_sha256": "3" * 64},
            "uci_options": {"Threads": 2, "Hash": 64},
            "time_control": {"nodes": 100000},
            "threads": 2,
            "hash_mb": 128,
            "concurrency": 4,
            "opening_suite": {"artifact_id": "OTHER", "sha256": "4" * 64},
            "repeat": False,
            "seed": 99999,
            "sprt": {"elo0": 0, "elo1": 10, "alpha": 0.05, "beta": 0.05, "max_games": 4000},
            "created_by": "someone-else",
        }

        for field, new_value in mutations.items():
            with self.subTest(field=field):
                mutated = dict(base)
                mutated[field] = new_value
                mutated_hash = compute_identity_sha256(mutated, exclude)
                self.assertNotEqual(
                    baseline_hash,
                    mutated_hash,
                    f"mutating top-level field {field!r} did not change identity_sha256",
                )

    def test_excluded_field_does_not_change_identity(self):
        base = self._base_match_config()
        exclude = frozenset({"created_at"})
        baseline_hash = compute_identity_sha256(base, exclude)

        mutated = dict(base)
        mutated["created_at"] = "2099-01-01T00:00:00Z"
        mutated_hash = compute_identity_sha256(mutated, exclude)

        self.assertEqual(baseline_hash, mutated_hash, "excluded field 'created_at' changed identity_sha256")

    def test_exclusion_is_top_level_only(self):
        obj = {"created_at": "2026-09-22T12:00:00Z", "nested": {"created_at": "2026-09-22T12:00:00Z"}}
        exclude = frozenset({"created_at"})
        base_hash = compute_identity_sha256(obj, exclude)

        mutated = {"created_at": "1999-01-01T00:00:00Z", "nested": {"created_at": "2099-01-01T00:00:00Z"}}
        mutated_hash = compute_identity_sha256(mutated, exclude)

        self.assertNotEqual(
            base_hash,
            mutated_hash,
            "nested field with same name as excluded top-level field must still affect identity",
        )


if __name__ == "__main__":
    unittest.main()
