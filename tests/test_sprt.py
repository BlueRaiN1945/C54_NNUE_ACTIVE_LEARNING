import math
import unittest

from medium_pc_audit.schemas import registry
from medium_pc_audit.sprt import (
    SprtError,
    build_analysis_result_document,
    evaluate_logistic_gsprt,
    logistic_gsprt_llr,
    logistic_score,
    sprt_boundaries,
    validate_wdl_pentanomial_consistency,
)


class BoundaryTests(unittest.TestCase):

    def test_standard_five_percent_boundaries(self):
        lower, upper = sprt_boundaries(
            alpha=0.05,
            beta=0.05,
        )

        self.assertAlmostEqual(
            lower,
            math.log(0.05 / 0.95),
            places=12,
        )

        self.assertAlmostEqual(
            upper,
            math.log(0.95 / 0.05),
            places=12,
        )

        self.assertAlmostEqual(
            lower,
            -upper,
            places=12,
        )

    def test_logistic_zero_elo_is_half_score(self):
        self.assertAlmostEqual(
            logistic_score(0.0),
            0.5,
            places=12,
        )


class LlrTests(unittest.TestCase):

    def test_positive_results_produce_positive_llr(self):
        result = logistic_gsprt_llr(
            [0, 5, 40, 100, 55],
            elo0=0,
            elo1=5,
        )

        self.assertGreater(result, 0.0)

    def test_negative_results_produce_negative_llr(self):
        result = logistic_gsprt_llr(
            [55, 100, 40, 5, 0],
            elo0=0,
            elo1=5,
        )

        self.assertLess(result, 0.0)

    def test_real_endpoint_llrs_are_finite(self):
        for penta in (
            [2, 28, 149, 21, 0],
            [0, 27, 144, 28, 1],
        ):
            with self.subTest(penta=penta):
                value = logistic_gsprt_llr(
                    penta,
                    elo0=0,
                    elo1=5,
                )

                self.assertTrue(math.isfinite(value))


class StateTests(unittest.TestCase):

    def test_strong_positive_crosses_h1(self):
        result = evaluate_logistic_gsprt(
            [0, 20, 160, 400, 220],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=4000,
        )

        self.assertEqual(
            result["state"],
            "H1_ACCEPTED",
        )

        self.assertGreater(
            result["llr"],
            result["upper_bound"],
        )

    def test_strong_negative_crosses_h0(self):
        result = evaluate_logistic_gsprt(
            [220, 400, 160, 20, 0],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=4000,
        )

        self.assertEqual(
            result["state"],
            "H0_ACCEPTED",
        )

        self.assertLess(
            result["llr"],
            result["lower_bound"],
        )

    def test_real_official_endpoint_reaches_cap(self):
        result = evaluate_logistic_gsprt(
            [2, 28, 149, 21, 0],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=400,
        )

        self.assertEqual(
            result["state"],
            "CAP_REACHED",
        )
        self.assertEqual(
            result["games_played"],
            400,
        )

    def test_real_v80_endpoint_reaches_cap(self):
        result = evaluate_logistic_gsprt(
            [0, 27, 144, 28, 1],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=400,
        )

        self.assertEqual(
            result["state"],
            "CAP_REACHED",
        )
        self.assertEqual(
            result["games_played"],
            400,
        )

    def test_same_endpoint_continues_before_larger_cap(self):
        result = evaluate_logistic_gsprt(
            [0, 27, 144, 28, 1],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=4000,
        )

        self.assertEqual(
            result["state"],
            "CONTINUE",
        )


class ConsistencyTests(unittest.TestCase):

    def test_real_endpoint_pairs_match_wdl(self):
        validate_wdl_pentanomial_consistency(
            wins=31,
            losses=42,
            draws=327,
            pentanomial=[2, 28, 149, 21, 0],
        )

        validate_wdl_pentanomial_consistency(
            wins=36,
            losses=33,
            draws=331,
            pentanomial=[0, 27, 144, 28, 1],
        )

    def test_bad_total_rejected(self):
        with self.assertRaises(SprtError):
            validate_wdl_pentanomial_consistency(
                wins=36,
                losses=33,
                draws=330,
                pentanomial=[0, 27, 144, 28, 1],
            )

    def test_bad_net_wins_minus_losses_rejected(self):
        with self.assertRaises(SprtError):
            validate_wdl_pentanomial_consistency(
                wins=35,
                losses=34,
                draws=331,
                pentanomial=[0, 27, 144, 28, 1],
            )


class AnalysisResultTests(unittest.TestCase):

    def test_v80_cap_builds_schema_valid_inconclusive_document(self):
        document = build_analysis_result_document(
            run_id="AUDIT_V0001__V80__run001",
            analyzer_version="phase4b-test",
            analyzed_at="2026-09-23T00:00:00Z",
            wins=36,
            losses=33,
            draws=331,
            pentanomial=[0, 27, 144, 28, 1],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=400,
            regression_tested=True,
        )

        registry.validate(
            "analysis_result",
            "v1",
            document,
        )

        self.assertEqual(
            document["sprt_result"]["outcome"],
            "CAP_REACHED",
        )

        self.assertEqual(
            document["verdict"],
            "INCONCLUSIVE",
        )

        self.assertFalse(
            document["regression_subtest"]["flagged"]
        )

    def test_official_cap_is_also_inconclusive_not_regression(self):
        document = build_analysis_result_document(
            run_id="AUDIT_V0001__OFFICIAL__run001",
            analyzer_version="phase4b-test",
            analyzed_at="2026-09-23T00:00:00Z",
            wins=31,
            losses=42,
            draws=327,
            pentanomial=[2, 28, 149, 21, 0],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=400,
            regression_tested=True,
        )

        self.assertEqual(
            document["sprt_result"]["outcome"],
            "CAP_REACHED",
        )

        self.assertEqual(
            document["verdict"],
            "INCONCLUSIVE",
        )

        self.assertFalse(
            document["regression_subtest"]["flagged"]
        )

    def test_finalization_refuses_continue_state(self):
        with self.assertRaises(SprtError):
            build_analysis_result_document(
                run_id="AUDIT_V0001__V80__run001",
                analyzer_version="phase4b-test",
                analyzed_at="2026-09-23T00:00:00Z",
                wins=36,
                losses=33,
                draws=331,
                pentanomial=[0, 27, 144, 28, 1],
                elo0=0,
                elo1=5,
                alpha=0.05,
                beta=0.05,
                max_games=4000,
                regression_tested=True,
            )

    def test_clear_negative_result_flags_regression(self):
        document = build_analysis_result_document(
            run_id="AUDIT_V0001__NEG__run001",
            analyzer_version="phase4b-test",
            analyzed_at="2026-09-23T00:00:00Z",
            wins=20,
            losses=840,
            draws=740,
            pentanomial=[220, 400, 160, 20, 0],
            elo0=0,
            elo1=5,
            alpha=0.05,
            beta=0.05,
            max_games=4000,
            regression_tested=True,
        )

        self.assertEqual(
            document["sprt_result"]["outcome"],
            "H0_ACCEPTED",
        )

        self.assertEqual(
            document["verdict"],
            "PROGRESS_NOT_SUPPORTED",
        )

        self.assertTrue(
            document["regression_subtest"]["flagged"]
        )


class ValidationTests(unittest.TestCase):

    def test_invalid_sprt_parameters_rejected(self):
        bad = [
            dict(elo0=5, elo1=0, alpha=.05, beta=.05),
            dict(elo0=0, elo1=5, alpha=0, beta=.05),
            dict(elo0=0, elo1=5, alpha=.05, beta=1),
        ]

        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(SprtError):
                    evaluate_logistic_gsprt(
                        [1, 1, 10, 1, 1],
                        max_games=100,
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
