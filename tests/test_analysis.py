from dataclasses import dataclass
import math
import unittest

from medium_pc_audit.analysis import (
    AnalysisError,
    candidate_points,
    estimate_pentanomial_elo,
    logistic_elo,
    pentanomial_counts,
    recomputed_wdl_for_schema,
    regularize_pentanomial,
    score_from_wdl,
)


@dataclass(frozen=True)
class FakeGame:
    opening_ref: str
    candidate_role: str
    result: str


def pair(ref, white_result, black_result):
    return [
        FakeGame(ref, "white", white_result),
        FakeGame(ref, "black", black_result),
    ]


class CandidatePointsTests(unittest.TestCase):

    def test_candidate_white_mapping(self):
        self.assertEqual(
            candidate_points(FakeGame("a", "white", "1-0")),
            1.0,
        )
        self.assertEqual(
            candidate_points(FakeGame("a", "white", "0-1")),
            0.0,
        )
        self.assertEqual(
            candidate_points(FakeGame("a", "white", "1/2-1/2")),
            0.5,
        )

    def test_candidate_black_mapping(self):
        self.assertEqual(
            candidate_points(FakeGame("a", "black", "0-1")),
            1.0,
        )
        self.assertEqual(
            candidate_points(FakeGame("a", "black", "1-0")),
            0.0,
        )
        self.assertEqual(
            candidate_points(FakeGame("a", "black", "1/2-1/2")),
            0.5,
        )

    def test_unknown_role_or_result_rejected(self):
        with self.assertRaises(AnalysisError):
            candidate_points(FakeGame("a", "candidate", "1-0"))

        with self.assertRaises(AnalysisError):
            candidate_points(FakeGame("a", "white", "*"))


class PentanomialTests(unittest.TestCase):

    def test_all_five_pair_cells(self):
        games = []

        # LL
        games += pair("ll", "0-1", "1-0")

        # LD
        games += pair("ld", "0-1", "1/2-1/2")

        # 1.0-point middle cell: two draws
        games += pair(
            "mid",
            "1/2-1/2",
            "1/2-1/2",
        )

        # WD
        games += pair("wd", "1-0", "1/2-1/2")

        # WW
        games += pair("ww", "1-0", "0-1")

        self.assertEqual(
            pentanomial_counts(games),
            (1, 1, 1, 1, 1),
        )

    def test_win_loss_also_maps_to_middle_cell(self):
        games = pair("x", "1-0", "1-0")

        self.assertEqual(
            pentanomial_counts(games),
            (0, 0, 1, 0, 0),
        )

    def test_missing_opening_ref_rejected(self):
        games = [
            FakeGame("", "white", "1-0"),
            FakeGame("", "black", "0-1"),
        ]

        with self.assertRaises(AnalysisError):
            pentanomial_counts(games)

    def test_broken_pair_size_rejected(self):
        with self.assertRaises(AnalysisError):
            pentanomial_counts(
                [FakeGame("x", "white", "1-0")]
            )

    def test_broken_color_pair_rejected(self):
        games = [
            FakeGame("x", "white", "1-0"),
            FakeGame("x", "white", "0-1"),
        ]

        with self.assertRaises(AnalysisError):
            pentanomial_counts(games)


class WdlTests(unittest.TestCase):

    def test_known_endpoint_scores(self):
        self.assertAlmostEqual(
            score_from_wdl(31, 42, 327),
            0.48625,
            places=12,
        )

        self.assertAlmostEqual(
            score_from_wdl(36, 33, 331),
            0.50375,
            places=12,
        )

    def test_schema_payload(self):
        result = recomputed_wdl_for_schema(
            31,
            42,
            327,
        )

        self.assertEqual(
            result,
            {
                "wins": 31,
                "losses": 42,
                "draws": 327,
                "decisive_games": 73,
                "score": 0.48625,
            },
        )

    def test_zero_games_and_bad_counts_rejected(self):
        with self.assertRaises(AnalysisError):
            score_from_wdl(0, 0, 0)

        with self.assertRaises(AnalysisError):
            score_from_wdl(-1, 0, 0)

        with self.assertRaises(AnalysisError):
            score_from_wdl(True, 0, 0)


class EloTests(unittest.TestCase):

    def test_logistic_half_score_is_zero_elo(self):
        self.assertAlmostEqual(
            logistic_elo(0.5),
            0.0,
            places=12,
        )

    def test_logistic_symmetry(self):
        self.assertAlmostEqual(
            logistic_elo(0.6),
            -logistic_elo(0.4),
            places=12,
        )

    def test_regularization_only_replaces_zero_cells(self):
        result = regularize_pentanomial(
            [2, 28, 149, 21, 0]
        )

        self.assertEqual(
            result,
            (2.0, 28.0, 149.0, 21.0, 0.001),
        )

    def test_official_real_endpoint_reference(self):
        stats = estimate_pentanomial_elo(
            [2, 28, 149, 21, 0]
        )

        self.assertEqual(stats["pairs"], 200)

        self.assertAlmostEqual(
            stats["regularized_score"],
            0.48625257,
            places=8,
        )

        self.assertAlmostEqual(
            stats["elo"],
            -9.5551,
            places=4,
        )

        self.assertAlmostEqual(
            stats["elo_ci_low"],
            -22.3555,
            places=4,
        )

        self.assertAlmostEqual(
            stats["elo_ci_high"],
            3.2453,
            places=4,
        )

        self.assertLessEqual(
            stats["elo_ci_low"],
            0.0,
        )
        self.assertGreaterEqual(
            stats["elo_ci_high"],
            0.0,
        )

    def test_v80_real_endpoint_reference(self):
        stats = estimate_pentanomial_elo(
            [0, 27, 144, 28, 1]
        )

        self.assertEqual(stats["pairs"], 200)

        self.assertAlmostEqual(
            stats["regularized_score"],
            0.50374748,
            places=8,
        )

        self.assertAlmostEqual(
            stats["elo"],
            2.6041,
            places=4,
        )

        self.assertAlmostEqual(
            stats["elo_ci_low"],
            -10.4746,
            places=4,
        )

        self.assertAlmostEqual(
            stats["elo_ci_high"],
            15.6828,
            places=4,
        )

        self.assertLessEqual(
            stats["elo_ci_low"],
            0.0,
        )
        self.assertGreaterEqual(
            stats["elo_ci_high"],
            0.0,
        )

    def test_invalid_pentanomial_rejected(self):
        bad_cases = [
            [1, 2, 3, 4],
            [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1],
            [True, 0, 0, 0, 1],
        ]

        for value in bad_cases:
            with self.subTest(value=value):
                with self.assertRaises(AnalysisError):
                    estimate_pentanomial_elo(value)

    def test_invalid_confidence_rejected(self):
        for confidence in (0, 1, -0.5, 1.5):
            with self.subTest(confidence=confidence):
                with self.assertRaises(AnalysisError):
                    estimate_pentanomial_elo(
                        [1, 1, 1, 1, 1],
                        confidence=confidence,
                    )


if __name__ == "__main__":
    unittest.main()
