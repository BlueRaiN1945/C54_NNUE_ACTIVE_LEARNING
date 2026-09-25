"""Tests for heritage split, exact-FEN dedupe, and family isolation."""

import unittest

from medium_pc_audit.split import (
    SPLIT_HOLDOUT,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    DedupeReport,
    FamilyIdentityUnavailable,
    SplitError,
    SplitResult,
    SplitWeights,
    assign_split,
    build_split,
    exact_fen_dedupe,
    normalize_fen_key,
    verify_no_cross_split_overlap,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_OTHER_CLOCK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 9 42"
E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


class NormalizeTests(unittest.TestCase):
    def test_keeps_first_four_fields(self):
        self.assertEqual(
            normalize_fen_key(START),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        )

    def test_clock_fields_are_excluded(self):
        self.assertEqual(
            normalize_fen_key(START),
            normalize_fen_key(START_OTHER_CLOCK),
        )

    def test_exactly_four_fields_is_enough(self):
        self.assertEqual(normalize_fen_key("a b c d"), "a b c d")

    def test_too_few_fields_returns_none(self):
        self.assertIsNone(normalize_fen_key("a b c"))

    def test_non_string_returns_none(self):
        for value in (None, 42, [], {}):
            self.assertIsNone(normalize_fen_key(value))

    def test_extra_whitespace_is_collapsed(self):
        self.assertEqual(normalize_fen_key("a   b  c    d  e"), "a b c d")


class DedupeTests(unittest.TestCase):
    def test_counts_duplicates_and_malformed(self):
        keys, report = exact_fen_dedupe([START, START_OTHER_CLOCK, E4, "bad", None])

        self.assertEqual(len(keys), 2)
        self.assertIsInstance(report, DedupeReport)
        self.assertEqual(report.total_input, 5)
        self.assertEqual(report.unique_keys, 2)
        self.assertEqual(report.exact_duplicates_removed, 1)
        self.assertEqual(report.malformed_skipped, 2)

    def test_accounting_is_complete(self):
        _, report = exact_fen_dedupe([START, START_OTHER_CLOCK, E4, "bad"])

        self.assertEqual(
            report.total_input,
            report.unique_keys
            + report.exact_duplicates_removed
            + report.malformed_skipped,
        )

    def test_first_seen_order_preserved(self):
        keys, _ = exact_fen_dedupe([E4, START])
        self.assertEqual(keys[0], normalize_fen_key(E4))


class WeightsTests(unittest.TestCase):
    def test_default_total(self):
        self.assertEqual(SplitWeights().total, 20)

    def test_zero_train_rejected(self):
        with self.assertRaises(SplitError):
            SplitWeights(train=0, validation=1, holdout=1)

    def test_negative_rejected(self):
        with self.assertRaises(SplitError):
            SplitWeights(train=5, validation=-1, holdout=1)

    def test_bool_rejected(self):
        with self.assertRaises(SplitError):
            SplitWeights(train=True, validation=1, holdout=1)


class AssignSplitTests(unittest.TestCase):
    def test_deterministic(self):
        w = SplitWeights()
        first = assign_split("family-1", seed="S", weights=w)
        second = assign_split("family-1", seed="S", weights=w)

        self.assertEqual(first, second)
        self.assertIn(first, SPLIT_NAMES)

    def test_seed_changes_assignment_distribution(self):
        w = SplitWeights()
        keys = [f"family-{i}" for i in range(200)]

        a = [assign_split(k, seed="S1", weights=w) for k in keys]
        b = [assign_split(k, seed="S2", weights=w) for k in keys]

        self.assertNotEqual(a, b)

    def test_all_train_when_only_train_weighted(self):
        w = SplitWeights(train=1, validation=0, holdout=0)

        for i in range(50):
            self.assertEqual(assign_split(f"k{i}", seed="S", weights=w), SPLIT_TRAIN)

    def test_all_three_splits_reachable(self):
        w = SplitWeights()
        seen = {assign_split(f"k{i}", seed="S", weights=w) for i in range(500)}

        self.assertEqual(seen, set(SPLIT_NAMES))

    def test_empty_key_rejected(self):
        with self.assertRaises(SplitError):
            assign_split("", seed="S", weights=SplitWeights())

    def test_empty_seed_rejected(self):
        with self.assertRaises(SplitError):
            assign_split("k", seed="", weights=SplitWeights())


class BuildSplitTests(unittest.TestCase):
    def rows(self, n=40):
        return [
            {"fen": f"board{i} w KQkq - 0 1", "game": f"g{i // 4}"} for i in range(n)
        ]

    def test_missing_family_provider_fails_closed(self):
        """The central guard: no family identity means no split, not a fallback."""

        with self.assertRaises(FamilyIdentityUnavailable) as ctx:
            build_split(self.rows(), seed="S")

        self.assertIn("family_of", str(ctx.exception))

    def test_unsafe_opt_in_is_marked_not_isolated(self):
        result = build_split(
            self.rows(),
            seed="S",
            allow_position_level_split_unsafe=True,
        )

        self.assertFalse(result.family_isolation)
        self.assertFalse(result.leakage_controlled)
        self.assertEqual(result.unit, "position")
        self.assertTrue(result.residual_leakage)
        self.assertTrue(
            any("not a game/line/family" in r for r in result.residual_leakage)
        )

    def test_family_split_is_isolated(self):
        result = build_split(
            self.rows(),
            seed="S",
            family_of=lambda row: row["game"],
        )

        self.assertTrue(result.family_isolation)
        self.assertTrue(result.leakage_controlled)
        self.assertEqual(result.unit, "family")
        self.assertEqual(len(result.assignments), 10)

    def test_each_family_has_exactly_one_assignment(self):
        rows = self.rows()
        result = build_split(rows, seed="S", family_of=lambda row: row["game"])

        # The split unit is the family, so every row of a family necessarily
        # resolves to that family's single assignment -- no row can straddle.
        for row in rows:
            self.assertIn(row["game"], result.assignments)
            self.assertIn(result.assignments[row["game"]], SPLIT_NAMES)

        self.assertEqual(len(result.assignments), len({r["game"] for r in rows}))

    def test_counts_match_assignments(self):
        result = build_split(
            self.rows(),
            seed="S",
            family_of=lambda row: row["game"],
        )

        self.assertEqual(sum(result.counts.values()), len(result.assignments))
        verify_no_cross_split_overlap(result)

    def test_deterministic_across_calls(self):
        a = build_split(self.rows(), seed="S", family_of=lambda r: r["game"])
        b = build_split(self.rows(), seed="S", family_of=lambda r: r["game"])

        self.assertEqual(a.assignments, b.assignments)

    def test_malformed_rows_excluded_from_families(self):
        rows = [{"fen": "bad", "game": "g0"}, {"fen": "a b c d", "game": "g1"}]
        result = build_split(rows, seed="S", family_of=lambda r: r["game"])

        self.assertEqual(set(result.assignments), {"g1"})

    def test_non_mapping_row_rejected(self):
        with self.assertRaises(SplitError):
            build_split(["not-a-dict"], seed="S", family_of=lambda r: "g")

    def test_bad_family_id_rejected(self):
        with self.assertRaises(SplitError):
            build_split(self.rows(4), seed="S", family_of=lambda r: None)

    def test_non_callable_family_of_rejected(self):
        with self.assertRaises(SplitError):
            build_split(self.rows(4), seed="S", family_of="not-callable")

    def test_as_dict_reports_posture(self):
        result = build_split(
            self.rows(),
            seed="S",
            allow_position_level_split_unsafe=True,
        )
        payload = result.as_dict()

        self.assertFalse(payload["family_isolation"])
        self.assertIn("residual_leakage", payload)
        self.assertEqual(payload["unit"], "position")


class VerifyOverlapTests(unittest.TestCase):
    def test_rejects_unknown_split_name(self):
        bad = SplitResult(
            assignments={"k": "nowhere"},
            counts={SPLIT_TRAIN: 1, SPLIT_VALIDATION: 0, SPLIT_HOLDOUT: 0},
            family_isolation=True,
            leakage_controlled=True,
            residual_leakage=(),
            dedupe=DedupeReport(1, 1, 0, 0),
            seed="S",
            weights=SplitWeights(),
            unit="family",
        )

        with self.assertRaises(SplitError):
            verify_no_cross_split_overlap(bad)

    def test_rejects_count_mismatch(self):
        bad = SplitResult(
            assignments={"k": SPLIT_TRAIN},
            counts={SPLIT_TRAIN: 5, SPLIT_VALIDATION: 0, SPLIT_HOLDOUT: 0},
            family_isolation=True,
            leakage_controlled=True,
            residual_leakage=(),
            dedupe=DedupeReport(1, 1, 0, 0),
            seed="S",
            weights=SplitWeights(),
            unit="family",
        )

        with self.assertRaises(SplitError):
            verify_no_cross_split_overlap(bad)


class FenNormalizationEquivalenceTests(unittest.TestCase):
    """split.normalize_fen_key must agree with pgn_parse's implementation.

    They are separate implementations on purpose: pgn_parse owns the only
    third-party dependency in this repository, so split.py cannot import it.
    This guard follows the precedent of
    tests/test_phase0_neutral_path_equivalence.py.
    """

    def test_agrees_with_pgn_parse(self):
        try:
            from medium_pc_audit.pgn_parse import normalize_opening_ref_from_fen
        except ImportError:  # pragma: no cover - depends on optional `chess`
            self.skipTest("python-chess not installed")

        cases = [
            START,
            START_OTHER_CLOCK,
            E4,
            "a b c d",
            "a b c d e f g",
            "a   b  c    d",
            "a b c",
            "",
            None,
            42,
        ]

        for case in cases:
            self.assertEqual(
                normalize_fen_key(case),
                normalize_opening_ref_from_fen(case),
                f"divergence for {case!r}",
            )


if __name__ == "__main__":
    unittest.main()
