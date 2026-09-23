import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.ledger import (
    append_ledger_record,
    build_ledger_record,
    rebuild_index,
)
from medium_pc_audit.sprt import build_analysis_result_document
from medium_pc_audit.trend import (
    TrendError,
    compare_experiment_arms,
    load_index_records,
    summarize_series,
)


SHA_CONFIG = "1" * 64
SHA_CANDIDATE = "2" * 64
SHA_OPPONENT = "3" * 64


def make_analysis(
    *,
    run_id,
    analyzed_at,
    wins,
    losses,
    draws,
    pentanomial,
    max_games,
):
    return build_analysis_result_document(
        run_id=run_id,
        analyzer_version="phase6b-test",
        analyzed_at=analyzed_at,
        wins=wins,
        losses=losses,
        draws=draws,
        pentanomial=pentanomial,
        elo0=0,
        elo1=5,
        alpha=0.05,
        beta=0.05,
        max_games=max_games,
        regression_tested=True,
    )


def make_inconclusive_analysis(
    run_id,
    *,
    analyzed_at="2026-09-23T00:00:00Z",
    stronger=False,
):
    if stronger:
        return make_analysis(
            run_id=run_id,
            analyzed_at=analyzed_at,
            wins=37,
            losses=32,
            draws=331,
            pentanomial=[0, 26, 144, 29, 1],
            max_games=400,
        )

    return make_analysis(
        run_id=run_id,
        analyzed_at=analyzed_at,
        wins=36,
        losses=33,
        draws=331,
        pentanomial=[0, 27, 144, 28, 1],
        max_games=400,
    )


def make_supported_analysis(
    run_id,
    *,
    analyzed_at="2026-09-23T00:00:00Z",
):
    return make_analysis(
        run_id=run_id,
        analyzed_at=analyzed_at,
        wins=840,
        losses=20,
        draws=740,
        pentanomial=[0, 20, 160, 400, 220],
        max_games=4000,
    )


def make_regression_analysis(
    run_id,
    *,
    analyzed_at="2026-09-23T00:00:00Z",
):
    return make_analysis(
        run_id=run_id,
        analyzed_at=analyzed_at,
        wins=20,
        losses=840,
        draws=740,
        pentanomial=[220, 400, 160, 20, 0],
        max_games=4000,
    )


def make_record(
    analysis,
    *,
    recorded_at,
    experiment_id,
    arm_id,
    opponent_id="V80",
):
    return build_ledger_record(
        analysis_result=analysis,
        recorded_at=recorded_at,
        config_identity_sha256=SHA_CONFIG,
        candidate_sha256=SHA_CANDIDATE,
        opponent_sha256=SHA_OPPONENT,
        opponent_id=opponent_id,
        experiment_id=experiment_id,
        arm_id=arm_id,
    )


class TrendTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _append_and_rebuild(self, *records):
        for record in records:
            append_ledger_record(
                self.root,
                record,
            )

        return rebuild_index(
            self.root
        )

    def test_increasing_elo_estimates_remain_inconclusive_when_source_verdicts_are_inconclusive(self):
        first = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__V80__run001",
            ),
            recorded_at="2026-09-23T00:05:00Z",
            experiment_id="A1",
            arm_id="broad-replay",
        )

        second = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__V80__run002",
                analyzed_at="2026-09-23T00:10:00Z",
                stronger=True,
            ),
            recorded_at="2026-09-23T00:15:00Z",
            experiment_id="A1",
            arm_id="broad-replay",
        )

        database = self._append_and_rebuild(
            first,
            second,
        )

        summary = summarize_series(
            database,
            opponent_id="V80",
            experiment_id="A1",
            arm_id="broad-replay",
        )

        self.assertEqual(
            summary["run_count"],
            2,
        )
        self.assertEqual(
            summary["evidence_state"],
            "INCONCLUSIVE_ONLY",
        )
        self.assertEqual(
            summary["verdict_counts"]["INCONCLUSIVE"],
            2,
        )
        self.assertEqual(
            summary["observed_elo_direction"],
            "NONDECREASING",
        )
        self.assertGreater(
            summary["observed_elo_delta"],
            0.0,
        )

        self.assertTrue(
            all(
                run["verdict"] == "INCONCLUSIVE"
                for run in summary["runs"]
            )
        )

    def test_supported_progress_state_comes_from_source_verdict(self):
        record = make_record(
            make_supported_analysis(
                "AUDIT_V0001__V80__run010",
            ),
            recorded_at="2026-09-23T01:00:00Z",
            experiment_id="A2",
            arm_id="weak-hard",
        )

        database = self._append_and_rebuild(
            record
        )

        summary = summarize_series(
            database,
            experiment_id="A2",
            arm_id="weak-hard",
        )

        self.assertEqual(
            summary["evidence_state"],
            "SUPPORTED_PROGRESS_ONLY",
        )
        self.assertEqual(
            summary["verdict_counts"]["SUPPORTED_PROGRESS"],
            1,
        )
        self.assertEqual(
            summary["observed_elo_direction"],
            "INSUFFICIENT_POINTS",
        )

    def test_regression_evidence_is_preserved_separately(self):
        record = make_record(
            make_regression_analysis(
                "AUDIT_V0001__V80__run020",
            ),
            recorded_at="2026-09-23T02:00:00Z",
            experiment_id="A3",
            arm_id="aggressive-hard",
        )

        database = self._append_and_rebuild(
            record
        )

        summary = summarize_series(
            database,
            experiment_id="A3",
            arm_id="aggressive-hard",
        )

        self.assertEqual(
            summary["evidence_state"],
            "REGRESSION_EVIDENCE_ONLY",
        )
        self.assertEqual(
            summary["regression_flagged_count"],
            1,
        )
        self.assertEqual(
            summary["runs"][0]["verdict"],
            "PROGRESS_NOT_SUPPORTED",
        )
        self.assertTrue(
            summary["runs"][0]["regression_flagged"]
        )

    def test_compare_experiment_arms_is_descriptive_and_does_not_rank_winner(self):
        control = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__V80__run030",
            ),
            recorded_at="2026-09-23T03:00:00Z",
            experiment_id="ABLATION-1",
            arm_id="A0-control",
        )

        broad = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__V80__run031",
                analyzed_at="2026-09-23T03:05:00Z",
                stronger=True,
            ),
            recorded_at="2026-09-23T03:10:00Z",
            experiment_id="ABLATION-1",
            arm_id="A1-broad",
        )

        database = self._append_and_rebuild(
            control,
            broad,
        )

        comparison = compare_experiment_arms(
            database,
            experiment_id="ABLATION-1",
            opponent_id="V80",
        )

        self.assertEqual(
            comparison["comparison_state"],
            "DESCRIPTIVE_ONLY",
        )
        self.assertEqual(
            comparison["run_count"],
            2,
        )
        self.assertEqual(
            set(comparison["arms"]),
            {
                "A0-control",
                "A1-broad",
            },
        )
        self.assertNotIn(
            "winner",
            comparison,
        )

        self.assertEqual(
            comparison["arms"]["A0-control"]["evidence_state"],
            "INCONCLUSIVE_ONLY",
        )
        self.assertEqual(
            comparison["arms"]["A1-broad"]["evidence_state"],
            "INCONCLUSIVE_ONLY",
        )

    def test_exact_filters_are_applied(self):
        first = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__V80__run040",
            ),
            recorded_at="2026-09-23T04:00:00Z",
            experiment_id="FILTER",
            arm_id="same-arm",
            opponent_id="V80",
        )

        second = make_record(
            make_inconclusive_analysis(
                "AUDIT_V0001__OFFICIAL__run041",
            ),
            recorded_at="2026-09-23T04:10:00Z",
            experiment_id="FILTER",
            arm_id="same-arm",
            opponent_id="OFFICIAL",
        )

        database = self._append_and_rebuild(
            first,
            second,
        )

        rows = load_index_records(
            database,
            opponent_id="OFFICIAL",
            experiment_id="FILTER",
            arm_id="same-arm",
        )

        self.assertEqual(
            len(rows),
            1,
        )
        self.assertEqual(
            rows[0]["run_id"],
            "AUDIT_V0001__OFFICIAL__run041",
        )

    def test_missing_index_is_rejected(self):
        with self.assertRaises(TrendError):
            summarize_series(
                self.root / "missing.sqlite3"
            )


if __name__ == "__main__":
    unittest.main()
