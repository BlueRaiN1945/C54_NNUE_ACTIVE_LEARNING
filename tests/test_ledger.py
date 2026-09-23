import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from medium_pc_audit.ledger import (
    LedgerConflictError,
    LedgerError,
    append_ledger_record,
    build_ledger_record,
    ledger_path,
    read_ledger,
    rebuild_index,
    sqlite_index_path,
)
from medium_pc_audit.sprt import build_analysis_result_document


SHA_CONFIG = "1" * 64
SHA_CANDIDATE = "2" * 64
SHA_OPPONENT = "3" * 64


def make_analysis(
    run_id="AUDIT_V0001__V80__run001",
    *,
    analyzed_at="2026-09-23T00:00:00Z",
):
    return build_analysis_result_document(
        run_id=run_id,
        analyzer_version="phase6-test",
        analyzed_at=analyzed_at,
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


def make_record(
    run_id="AUDIT_V0001__V80__run001",
    *,
    recorded_at="2026-09-23T00:05:00Z",
    experiment_id="A0",
    arm_id="control",
):
    return build_ledger_record(
        analysis_result=make_analysis(
            run_id
        ),
        recorded_at=recorded_at,
        config_identity_sha256=SHA_CONFIG,
        candidate_sha256=SHA_CANDIDATE,
        opponent_sha256=SHA_OPPONENT,
        opponent_id="V80",
        experiment_id=experiment_id,
        arm_id=arm_id,
    )


class LedgerRecordTests(unittest.TestCase):

    def test_record_preserves_complete_analysis_result(self):
        analysis = make_analysis()

        record = build_ledger_record(
            analysis_result=analysis,
            recorded_at="2026-09-23T00:05:00Z",
            config_identity_sha256=SHA_CONFIG,
            candidate_sha256=SHA_CANDIDATE,
            opponent_sha256=SHA_OPPONENT,
            opponent_id="V80",
            experiment_id="A0",
            arm_id="control",
        )

        self.assertEqual(
            record["analysis_result"],
            analysis,
        )
        self.assertEqual(
            record["source_state"],
            "ACCEPTED",
        )
        self.assertEqual(
            record["schema_version"],
            "v1",
        )
        self.assertEqual(
            len(record["record_id"]),
            64,
        )

    def test_record_identity_is_deterministic(self):
        first = make_record()
        second = make_record()

        self.assertEqual(
            first,
            second,
        )

    def test_invalid_sha256_is_rejected(self):
        with self.assertRaises(LedgerError):
            build_ledger_record(
                analysis_result=make_analysis(),
                recorded_at="2026-09-23T00:05:00Z",
                config_identity_sha256="not-a-hash",
                candidate_sha256=SHA_CANDIDATE,
                opponent_sha256=SHA_OPPONENT,
                opponent_id="V80",
            )


class AppendOnlyLedgerTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_append_and_read_roundtrip(self):
        record = make_record()

        result = append_ledger_record(
            self.root,
            record,
        )

        self.assertTrue(
            result["appended"]
        )

        self.assertEqual(
            read_ledger(self.root),
            [record],
        )

        raw = ledger_path(
            self.root
        ).read_bytes()

        self.assertTrue(
            raw.endswith(b"\n")
        )
        self.assertEqual(
            len(raw.splitlines()),
            1,
        )

    def test_same_record_is_idempotent(self):
        record = make_record()

        first = append_ledger_record(
            self.root,
            record,
        )
        second = append_ledger_record(
            self.root,
            record,
        )

        self.assertTrue(
            first["appended"]
        )
        self.assertFalse(
            second["appended"]
        )

        self.assertEqual(
            len(read_ledger(self.root)),
            1,
        )

    def test_same_run_id_with_different_content_is_conflict(self):
        first = make_record()
        second = deepcopy(
            first
        )

        second["opponent_id"] = "DIFFERENT"

        from medium_pc_audit.identity import compute_identity_sha256

        second["record_id"] = compute_identity_sha256(
            {
                key: value
                for key, value in second.items()
                if key != "record_id"
            }
        )

        append_ledger_record(
            self.root,
            first,
        )

        with self.assertRaises(LedgerConflictError):
            append_ledger_record(
                self.root,
                second,
            )

    def test_partial_final_line_is_rejected(self):
        path = ledger_path(
            self.root
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(
            b'{"schema_version":"v1"'
        )

        with self.assertRaises(LedgerError):
            read_ledger(
                self.root
            )


class RebuildIndexTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(
            self.temporary.name
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_sqlite_index_is_fully_rebuildable(self):
        first = make_record(
            "AUDIT_V0001__V80__run001",
            experiment_id="A0",
            arm_id="control",
        )
        second = make_record(
            "AUDIT_V0001__V80__run002",
            recorded_at="2026-09-23T00:06:00Z",
            experiment_id="A1",
            arm_id="broad-replay",
        )

        append_ledger_record(
            self.root,
            first,
        )
        append_ledger_record(
            self.root,
            second,
        )

        database = rebuild_index(
            self.root
        )

        self.assertEqual(
            database,
            sqlite_index_path(self.root),
        )
        self.assertTrue(
            database.exists()
        )

        with closing(sqlite3.connect(database)) as connection:
            rows_before = connection.execute(
                """
                SELECT
                    run_id,
                    opponent_id,
                    experiment_id,
                    arm_id,
                    verdict,
                    sprt_outcome,
                    games_played
                FROM records
                ORDER BY run_id
                """
            ).fetchall()

            metadata_before = dict(
                connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            )

        database.unlink()

        self.assertFalse(
            database.exists()
        )

        rebuilt = rebuild_index(
            self.root
        )

        with closing(sqlite3.connect(rebuilt)) as connection:
            rows_after = connection.execute(
                """
                SELECT
                    run_id,
                    opponent_id,
                    experiment_id,
                    arm_id,
                    verdict,
                    sprt_outcome,
                    games_played
                FROM records
                ORDER BY run_id
                """
            ).fetchall()

            metadata_after = dict(
                connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            )

        self.assertEqual(
            rows_before,
            rows_after,
        )
        self.assertEqual(
            metadata_before,
            metadata_after,
        )
        self.assertEqual(
            metadata_after["record_count"],
            "2",
        )

    def test_index_preserves_canonical_record_json(self):
        record = make_record()

        append_ledger_record(
            self.root,
            record,
        )

        database = rebuild_index(
            self.root
        )

        with closing(sqlite3.connect(database)) as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM records
                WHERE run_id = ?
                """,
                (record["run_id"],),
            ).fetchone()

        self.assertIsNotNone(
            row
        )
        self.assertEqual(
            json.loads(row[0]),
            record,
        )


if __name__ == "__main__":
    unittest.main()

