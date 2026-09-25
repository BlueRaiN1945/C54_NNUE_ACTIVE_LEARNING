"""Tests for opening-suite contracts, capacity, and collision avoidance."""

import unittest

from medium_pc_audit.opening_suite import (
    InsufficientOpenings,
    OpeningSuite,
    OpeningSuiteError,
    build_suite_contract,
    collision_probability,
    load_suite_from_text,
    required_distinct_openings,
    select_openings,
    validate_capacity,
)

SHA = "e" * 64


def epd_text(n=10):
    return "\n".join(f'board{i} w KQkq - bm Nf3; id "p{i}";' for i in range(n))


def make_suite(n=10, **kw):
    suite, _ = load_suite_from_text(
        epd_text(n),
        suite_id=kw.pop("suite_id", "SUITE"),
        sha256=kw.pop("sha256", SHA),
        **kw,
    )
    return suite


class LoadTests(unittest.TestCase):
    def test_parses_and_counts(self):
        suite, report = load_suite_from_text(epd_text(5), suite_id="S", sha256=SHA)

        self.assertEqual(suite.distinct_count, 5)
        self.assertEqual(report.total_lines, 5)
        self.assertEqual(report.duplicate_lines, 0)
        self.assertEqual(report.malformed_lines, 0)

    def test_deduplicates_and_reports(self):
        text = "a b c d\na b c d\ne f g h"
        suite, report = load_suite_from_text(text, suite_id="S", sha256=SHA)

        self.assertEqual(suite.distinct_count, 2)
        self.assertEqual(report.total_lines, 3)
        self.assertEqual(report.duplicate_lines, 1)

    def test_malformed_lines_counted(self):
        text = "a b c d\nshort\ne f g h"
        suite, report = load_suite_from_text(text, suite_id="S", sha256=SHA)

        self.assertEqual(report.malformed_lines, 1)
        self.assertEqual(suite.distinct_count, 2)

    def test_blank_lines_ignored(self):
        text = "a b c d\n\n\n  \ne f g h"
        _, report = load_suite_from_text(text, suite_id="S", sha256=SHA)

        self.assertEqual(report.total_lines, 2)

    def test_all_malformed_raises(self):
        with self.assertRaises(OpeningSuiteError):
            load_suite_from_text("short\nalso short", suite_id="S", sha256=SHA)

    def test_clock_fields_ignored_so_same_position_dedupes(self):
        text = "a b c d 0 1\na b c d 9 42"
        suite, report = load_suite_from_text(text, suite_id="S", sha256=SHA)

        self.assertEqual(suite.distinct_count, 1)
        self.assertEqual(report.duplicate_lines, 1)


class DeclarationTests(unittest.TestCase):
    def test_unbalanced_defaults_false(self):
        self.assertFalse(make_suite().unbalanced_declared)

    def test_unbalanced_requires_justification(self):
        with self.assertRaises(OpeningSuiteError):
            make_suite(unbalanced_declared=True)

    def test_unbalanced_with_justification_accepted(self):
        suite = make_suite(
            unbalanced_declared=True,
            justification="selected in the +0.90..+1.19 band",
        )

        self.assertTrue(suite.unbalanced_declared)

    def test_bad_sha_rejected(self):
        with self.assertRaises(OpeningSuiteError):
            OpeningSuite(suite_id="S", sha256="nope", openings=("a b c d",))

    def test_duplicate_openings_rejected_at_construction(self):
        with self.assertRaises(OpeningSuiteError):
            OpeningSuite(
                suite_id="S",
                sha256=SHA,
                openings=("a b c d", "a b c d"),
            )

    def test_empty_suite_rejected(self):
        with self.assertRaises(OpeningSuiteError):
            OpeningSuite(suite_id="S", sha256=SHA, openings=())


class CapacityTests(unittest.TestCase):
    def test_one_opening_per_pair(self):
        self.assertEqual(required_distinct_openings(200), 200)

    def test_capacity_ok(self):
        validate_capacity(make_suite(10), pairs=10)

    def test_capacity_insufficient(self):
        with self.assertRaises(InsufficientOpenings) as ctx:
            validate_capacity(make_suite(5), pairs=6)

        self.assertIn("required", str(ctx.exception))

    def test_max_pairs(self):
        self.assertEqual(make_suite(7).max_pairs(), 7)

    def test_invalid_pairs_rejected(self):
        for bad in (0, -1, True, "5", None):
            with self.assertRaises(OpeningSuiteError):
                required_distinct_openings(bad)


class CollisionTests(unittest.TestCase):
    """Quantifies the trap that motivates without-replacement selection."""

    def test_matches_documented_estimate_for_observed_book(self):
        # 200 pairs drawn with replacement from noob_3moves.epd's 150,932.
        p = collision_probability(200, 150932)
        self.assertGreater(p, 0.10)
        self.assertLess(p, 0.15)

    def test_exceeds_half_at_500_pairs(self):
        self.assertGreater(collision_probability(500, 150932), 0.50)

    def test_single_pair_cannot_collide(self):
        self.assertAlmostEqual(collision_probability(1, 100), 0.0)

    def test_monotonic_in_pairs(self):
        a = collision_probability(50, 10000)
        b = collision_probability(100, 10000)
        self.assertGreater(b, a)


class SelectionTests(unittest.TestCase):
    def test_selection_is_distinct(self):
        selected = select_openings(make_suite(50), pairs=20, seed="S")

        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)

    def test_selection_is_deterministic(self):
        a = select_openings(make_suite(50), pairs=20, seed="S")
        b = select_openings(make_suite(50), pairs=20, seed="S")

        self.assertEqual(a, b)

    def test_seed_changes_selection(self):
        a = select_openings(make_suite(50), pairs=20, seed="S1")
        b = select_openings(make_suite(50), pairs=20, seed="S2")

        self.assertNotEqual(a, b)

    def test_selection_is_a_subset_of_the_suite(self):
        suite = make_suite(30)
        selected = select_openings(suite, pairs=10, seed="S")

        self.assertTrue(set(selected).issubset(set(suite.openings)))

    def test_oversized_request_refused(self):
        with self.assertRaises(InsufficientOpenings):
            select_openings(make_suite(5), pairs=9, seed="S")

    def test_empty_seed_rejected(self):
        with self.assertRaises(OpeningSuiteError):
            select_openings(make_suite(5), pairs=2, seed="")


class ContractTests(unittest.TestCase):
    def test_contract_records_selection_and_risk(self):
        contract = build_suite_contract(make_suite(50), pairs=20, seed="S")

        self.assertEqual(contract["pairs"], 20)
        self.assertEqual(contract["games"], 40)
        self.assertEqual(contract["selected_count"], 20)
        self.assertEqual(contract["selection"], "without_replacement_deterministic")
        self.assertGreater(contract["with_replacement_collision_probability"], 0.0)

    def test_contract_is_deterministic(self):
        a = build_suite_contract(make_suite(50), pairs=20, seed="S")
        b = build_suite_contract(make_suite(50), pairs=20, seed="S")

        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
