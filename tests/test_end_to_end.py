"""Synthetic end-to-end integration test for the complete Medium-PC panel.

No real Stockfish or c-chess-cli process is executed here.

The synthetic execution host only creates deterministic evidence files.
Every Medium-PC stage after that uses the real production implementation:

    config generation
    -> result package ingestion
    -> immutable inbox
    -> independent PGN parsing
    -> AnalysisResultV1
    -> append-only ledger
    -> rebuildable SQLite index
    -> trend/verdict query
"""

import json
import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.analysis import pentanomial_counts
from medium_pc_audit.ledger import (
    append_ledger_record,
    build_ledger_record,
    read_ledger,
    rebuild_index,
)
from medium_pc_audit.pgn_parse import (
    build_raw_result_document,
    parse_pgn_file,
)
from medium_pc_audit.result_package import (
    ingest_result_package,
    verify_published_result,
)
from medium_pc_audit.sprt import build_analysis_result_document
from medium_pc_audit.trend import summarize_series

from tests.test_config_gen import ConfigGenTestCase
from tests.test_result_package import (
    create_package,
    reseal_package,
)


EXPERIMENT_ID = "PHASE7-SYNTHETIC"
ARM_ID = "synthetic-control"


def synthetic_four_draw_pgn(
    *,
    candidate_label: str,
    opponent_label: str,
) -> str:
    """Two color-reversed opening pairs, four draws total."""

    opening_a = (
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    opening_b = (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/"
        "PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    )

    games = []

    pairings = [
        (candidate_label, opponent_label, opening_a),
        (opponent_label, candidate_label, opening_a),
        (candidate_label, opponent_label, opening_b),
        (opponent_label, candidate_label, opening_b),
    ]

    for index, (white, black, fen) in enumerate(
        pairings,
        start=1,
    ):
        games.append(
            "\n".join(
                [
                    f'[Event "phase7-synthetic-{index}"]',
                    f'[White "{white}"]',
                    f'[Black "{black}"]',
                    '[Result "1/2-1/2"]',
                    '[SetUp "1"]',
                    f'[FEN "{fen}"]',
                    "",
                    "1/2-1/2",
                    "",
                ]
            )
        )

    return "\n".join(games)


class SyntheticEndToEndTests(ConfigGenTestCase):

    def test_config_to_ingest_to_analysis_to_ledger_to_trend(self):
        # ------------------------------------------------------------------
        # 1. Real config generation
        # ------------------------------------------------------------------
        config = self.build_default(
            sprt={
                "elo0": 0,
                "elo1": 5,
                "alpha": 0.05,
                "beta": 0.05,
                "max_games": 4,
            }
        )

        run_id = config["config_id"]

        candidate_label = config[
            "candidate"
        ]["artifact_id"]

        opponent_label = config[
            "opponent"
        ]["artifact_id"]

        # ------------------------------------------------------------------
        # 2. Synthetic execution host
        #
        # No engine subprocess is launched.  We only manufacture deterministic
        # host evidence in the same transport format a real BigPC run returns.
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            source_package = (
                root
                / "simulated_execution_host"
            )

            inbox_root = (
                root
                / "RESULTS_INBOX"
            )

            ledger_root = (
                root
                / "LEDGER"
            )

            # Reuse the established transport fixture builder for manifest,
            # artifact binding, log, and checksum layout.
            create_package(
                source_package,
                config,
            )

            pgn_path = (
                source_package
                / "games.pgn"
            )

            pgn_path.write_text(
                synthetic_four_draw_pgn(
                    candidate_label=candidate_label,
                    opponent_label=opponent_label,
                ),
                encoding="utf-8",
                newline="\n",
            )

            # ------------------------------------------------------------------
            # 3. Real independent PGN parsing on the host evidence
            # ------------------------------------------------------------------
            source_parse = parse_pgn_file(
                pgn_path,
                candidate_label=candidate_label,
                opponent_label=opponent_label,
            )

            self.assertEqual(
                source_parse.recomputed_wdl,
                {
                    "wins": 0,
                    "losses": 0,
                    "draws": 4,
                    "unresolved": 0,
                    "total_games": 4,
                },
            )

            self.assertEqual(
                len(source_parse.raw_result_games),
                4,
            )

            self.assertEqual(
                source_parse.excluded,
                [],
            )

            # Replace the fixture raw result with one produced by the real
            # Phase-3 builder from independently parsed PGN evidence.
            raw_result = build_raw_result_document(
                run_id=run_id,
                claimed_wins=0,
                claimed_losses=0,
                claimed_draws=4,
                engine_stdout_log_ref="match.log",
                pgn_ref="games.pgn",
                raw_result_games=source_parse.raw_result_games,
            )

            (
                source_package
                / "raw_result.json"
            ).write_text(
                json.dumps(
                    raw_result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

            # Host recomputes SHA256SUMS.txt after final evidence is complete.
            reseal_package(
                source_package
            )

            # ------------------------------------------------------------------
            # 4. Real Phase-5B immutable ingestion
            # ------------------------------------------------------------------
            ingestion = ingest_result_package(
                source_package,
                inbox_root,
                match_config=config,
                received_at="2026-09-23T10:30:00Z",
            )

            self.assertEqual(
                ingestion["state"],
                "ACCEPTED",
                ingestion,
            )

            self.assertTrue(
                ingestion["published"]
            )

            published_dir = (
                inbox_root
                / run_id
            )

            self.assertTrue(
                published_dir.is_dir()
            )

            verification = verify_published_result(
                inbox_root,
                run_id,
            )

            self.assertTrue(
                verification["ok"],
                verification,
            )

            # ------------------------------------------------------------------
            # 5. Parse immutable published evidence again.
            #
            # Analysis is based on the accepted inbox copy, not the temporary
            # simulated-host source.
            # ------------------------------------------------------------------
            published_parse = parse_pgn_file(
                published_dir / "games.pgn",
                candidate_label=candidate_label,
                opponent_label=opponent_label,
            )

            self.assertEqual(
                published_parse.recomputed_wdl,
                source_parse.recomputed_wdl,
            )

            self.assertEqual(
                published_parse.excluded,
                [],
            )

            pentanomial = pentanomial_counts(
                published_parse.parsed_games
            )

            self.assertEqual(
                pentanomial,
                (0, 0, 2, 0, 0),
            )

            # ------------------------------------------------------------------
            # 6. Real Phase-4B final analysis
            # ------------------------------------------------------------------
            sprt = config["sprt"]
            wdl = published_parse.recomputed_wdl

            analysis_result = (
                build_analysis_result_document(
                    run_id=run_id,
                    analyzer_version="phase7-synthetic",
                    analyzed_at="2026-09-23T10:31:00Z",
                    wins=wdl["wins"],
                    losses=wdl["losses"],
                    draws=wdl["draws"],
                    pentanomial=pentanomial,
                    elo0=sprt["elo0"],
                    elo1=sprt["elo1"],
                    alpha=sprt["alpha"],
                    beta=sprt["beta"],
                    max_games=sprt["max_games"],
                    regression_tested=True,
                )
            )

            self.assertEqual(
                analysis_result[
                    "sprt_result"
                ]["outcome"],
                "CAP_REACHED",
            )

            self.assertEqual(
                analysis_result["verdict"],
                "INCONCLUSIVE",
            )

            self.assertFalse(
                analysis_result[
                    "regression_subtest"
                ]["flagged"]
            )

            # ------------------------------------------------------------------
            # 7. Bind actual Phase-5B artifact provenance into Phase-6 ledger.
            # ------------------------------------------------------------------
            binding = json.loads(
                (
                    published_dir
                    / "artifact_binding.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            record = build_ledger_record(
                analysis_result=analysis_result,
                recorded_at="2026-09-23T10:32:00Z",
                config_identity_sha256=config[
                    "identity_sha256"
                ],
                candidate_sha256=binding[
                    "actual_candidate_sha256"
                ],
                opponent_sha256=binding[
                    "actual_opponent_sha256"
                ],
                opponent_id=opponent_label,
                experiment_id=EXPERIMENT_ID,
                arm_id=ARM_ID,
            )

            appended = append_ledger_record(
                ledger_root,
                record,
            )

            self.assertTrue(
                appended["appended"]
            )

            ledger_records = read_ledger(
                ledger_root
            )

            self.assertEqual(
                len(ledger_records),
                1,
            )

            self.assertEqual(
                ledger_records[0]["run_id"],
                run_id,
            )

            # ------------------------------------------------------------------
            # 8. Rebuild derived SQLite solely from canonical JSONL.
            # ------------------------------------------------------------------
            database = rebuild_index(
                ledger_root
            )

            self.assertTrue(
                database.is_file()
            )

            # ------------------------------------------------------------------
            # 9. Real Phase-6 trend/verdict query
            # ------------------------------------------------------------------
            summary = summarize_series(
                database,
                opponent_id=opponent_label,
                experiment_id=EXPERIMENT_ID,
                arm_id=ARM_ID,
            )

            self.assertEqual(
                summary["run_count"],
                1,
            )

            self.assertEqual(
                summary["evidence_state"],
                "INCONCLUSIVE_ONLY",
            )

            self.assertEqual(
                summary[
                    "verdict_counts"
                ]["INCONCLUSIVE"],
                1,
            )

            self.assertEqual(
                summary[
                    "observed_elo_direction"
                ],
                "INSUFFICIENT_POINTS",
            )

            self.assertEqual(
                summary[
                    "runs"
                ][0]["run_id"],
                run_id,
            )

            self.assertEqual(
                summary[
                    "runs"
                ][0]["sprt_outcome"],
                "CAP_REACHED",
            )

            self.assertEqual(
                summary[
                    "runs"
                ][0]["verdict"],
                "INCONCLUSIVE",
            )


if __name__ == "__main__":
    unittest.main()

