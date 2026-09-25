import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.analysis import pentanomial_counts
from medium_pc_audit.execution_package import (
    ExecutionPackageError,
    ObservedExecution,
    build_execution_package,
)
from medium_pc_audit.package_lint import (
    lint_execution_package,
)
from medium_pc_audit.pgn_parse import parse_pgn_file
from medium_pc_audit.result_package import (
    compute_config_identity_sha256,
    ingest_result_package,
    verify_published_result,
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def synthetic_pgn():
    fen = (
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    return (
        '[Event "pair-1-a"]\n'
        '[White "CANDIDATE"]\n'
        '[Black "TEACHER"]\n'
        '[Result "1/2-1/2"]\n'
        '[SetUp "1"]\n'
        f'[FEN "{fen}"]\n'
        "\n"
        "1. Nf3 Nf6 2. g3 g6 1/2-1/2\n"
        "\n"
        '[Event "pair-1-b"]\n'
        '[White "TEACHER"]\n'
        '[Black "CANDIDATE"]\n'
        '[Result "1/2-1/2"]\n'
        '[SetUp "1"]\n'
        f'[FEN "{fen}"]\n'
        "\n"
        "1. Nf3 Nf6 2. g3 g6 1/2-1/2\n"
    )


def create_source_files(root):
    files = {
        "engine": root / "stockfish.exe",
        "cli": root / "c-chess-cli.exe",
        "candidate": root / "candidate.nnue",
        "opponent": root / "opponent.nnue",
        "opening_suite": root / "openings.epd",
        "pgn": root / "source-games.pgn",
        "log": root / "source-match.log",
    }

    files["engine"].write_bytes(
        b"synthetic-engine-bytes\n"
    )
    files["cli"].write_bytes(
        b"synthetic-cli-bytes\n"
    )
    files["candidate"].write_bytes(
        b"synthetic-candidate-network\n"
    )
    files["opponent"].write_bytes(
        b"synthetic-opponent-network\n"
    )
    files["opening_suite"].write_bytes(
        b"synthetic-opening-suite\n"
    )
    files["pgn"].write_text(
        synthetic_pgn(),
        encoding="utf-8",
        newline="\n",
    )
    files["log"].write_bytes(
        b"synthetic execution log\n"
    )

    return files


def file_sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def make_config(files):
    cfg = {
        "schema_version": "v1",
        "config_id": "AUDIT_V0001__TEST__run001",
        "audit_id": "AUDIT_V0001",
        "matchup_key": "AUDIT_V0001__TEST",
        "matchup_type": "vs_teacher",
        "candidate": {
            "artifact_id": "CANDIDATE",
            "sha256": file_sha(
                files["candidate"]
            ),
        },
        "opponent": {
            "artifact_id": "TEACHER",
            "sha256": file_sha(
                files["opponent"]
            ),
        },
        "engine": {
            "pinned_id": "stockfish-test",
            "expected_sha256": file_sha(
                files["engine"]
            ),
        },
        "cli": {
            "pinned_id": "cchess-test",
            "expected_sha256": file_sha(
                files["cli"]
            ),
        },
        "uci_options": {
            "Threads": 1,
            "Hash": 64,
        },
        "time_control": {
            "nodes": 50000,
        },
        "threads": 1,
        "hash_mb": 64,
        "concurrency": 1,
        "opening_suite": {
            "artifact_id": "OPENINGS",
            "sha256": file_sha(
                files["opening_suite"]
            ),
        },
        "repeat": True,
        "seed": 12345,
        "sprt": {
            "elo0": 0,
            "elo1": 5,
            "alpha": 0.05,
            "beta": 0.05,
            "max_games": 2,
        },
        "created_at": "2026-09-24T09:00:00Z",
        "created_by": "execution-package-test",
    }

    cfg["identity_sha256"] = (
        compute_config_identity_sha256(
            cfg
        )
    )

    return cfg


def make_observation(files, cwd):
    return ObservedExecution(
        engine_path=files["engine"],
        cli_path=files["cli"],
        candidate_path=files["candidate"],
        opponent_path=files["opponent"],
        opening_suite_path=files[
            "opening_suite"
        ],
        pgn_path=files["pgn"],
        log_path=files["log"],
        argv=(
            "c-chess-cli",
            "-games",
            "2",
            "-repeat",
            "-openings",
            "file=openings.epd",
            "order=sequential",
            "-pgn",
            "games.pgn",
            "2",
        ),
        invocation_cwd=cwd,
        actual_uci_options={
            "Threads": 1,
            "Hash": 64,
        },
        actual_time_control={
            "nodes": 50000,
        },
        actual_threads=1,
        actual_hash_mb=64,
        actual_concurrency=1,
        actual_seed=12345,
        host_identity_label="synthetic-bigpc",
        executed_at_utc=(
            "2026-09-24T09:01:00Z"
        ),
        executed_by="synthetic-worker",
        runtime_seconds=1.25,
    )


class ExecutionPackageTests(
    unittest.TestCase
):
    def test_observed_files_build_clean_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()

            files = create_source_files(
                source
            )
            cfg = make_config(files)
            observation = make_observation(
                files,
                source,
            )

            package = root / "package"

            result = build_execution_package(
                package,
                match_config=cfg,
                observation=observation,
            )

            self.assertEqual(
                result["recomputed_wdl"],
                {
                    "wins": 0,
                    "losses": 0,
                    "draws": 2,
                    "unresolved": 0,
                    "total_games": 2,
                },
            )

            self.assertTrue(
                (
                    package
                    / "execution_observation.json"
                ).is_file()
            )

            observation_doc = json.loads(
                (
                    package
                    / "execution_observation.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                observation_doc["argv"],
                list(observation.argv),
            )

            self.assertEqual(
                observation_doc[
                    "files"
                ]["engine"]["sha256"],
                file_sha(files["engine"]),
            )

            checksum_text = (
                package
                / "SHA256SUMS.txt"
            ).read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "execution_observation.json",
                checksum_text,
            )

            lint = lint_execution_package(
                package,
                match_config=cfg,
            )

            self.assertTrue(
                lint["ok"],
                lint,
            )

    def test_actual_hashes_are_not_copied_from_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()

            files = create_source_files(
                source
            )
            cfg = make_config(files)

            cfg["engine"][
                "expected_sha256"
            ] = "0" * 64

            cfg["candidate"][
                "sha256"
            ] = "1" * 64

            cfg.pop(
                "identity_sha256",
                None,
            )

            cfg["identity_sha256"] = (
                compute_config_identity_sha256(
                    cfg
                )
            )

            package = root / "package"

            build_execution_package(
                package,
                match_config=cfg,
                observation=make_observation(
                    files,
                    source,
                ),
            )

            manifest = json.loads(
                (
                    package
                    / "execution_manifest.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            binding = json.loads(
                (
                    package
                    / "artifact_binding.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest[
                    "actual_engine_sha256"
                ],
                file_sha(files["engine"]),
            )

            self.assertNotEqual(
                manifest[
                    "actual_engine_sha256"
                ],
                cfg["engine"][
                    "expected_sha256"
                ],
            )

            self.assertEqual(
                binding[
                    "actual_candidate_sha256"
                ],
                file_sha(
                    files["candidate"]
                ),
            )

            self.assertNotEqual(
                binding[
                    "actual_candidate_sha256"
                ],
                cfg["candidate"][
                    "sha256"
                ],
            )

            lint = lint_execution_package(
                package,
                match_config=cfg,
            )

            self.assertFalse(
                lint["ok"],
                lint,
            )

    def test_missing_observed_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()

            files = create_source_files(
                source
            )
            cfg = make_config(files)

            observation = ObservedExecution(
                engine_path=(
                    source
                    / "missing-engine.exe"
                ),
                cli_path=files["cli"],
                candidate_path=files[
                    "candidate"
                ],
                opponent_path=files[
                    "opponent"
                ],
                opening_suite_path=files[
                    "opening_suite"
                ],
                pgn_path=files["pgn"],
                log_path=files["log"],
                argv=("c-chess-cli",),
                invocation_cwd=source,
                actual_uci_options={
                    "Threads": 1,
                    "Hash": 64,
                },
                actual_time_control={
                    "nodes": 50000,
                },
                actual_threads=1,
                actual_hash_mb=64,
                actual_concurrency=1,
                actual_seed=12345,
                host_identity_label=(
                    "synthetic-bigpc"
                ),
                executed_at_utc=(
                    "2026-09-24T09:01:00Z"
                ),
                executed_by=(
                    "synthetic-worker"
                ),
            )

            package = root / "package"

            with self.assertRaises(
                ExecutionPackageError
            ):
                build_execution_package(
                    package,
                    match_config=cfg,
                    observation=observation,
                )

            self.assertFalse(
                package.exists()
            )

    def test_synthetic_producer_to_pentanomial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()

            files = create_source_files(
                source
            )
            cfg = make_config(files)
            package = root / "package"

            build_execution_package(
                package,
                match_config=cfg,
                observation=make_observation(
                    files,
                    source,
                ),
            )

            lint = lint_execution_package(
                package,
                match_config=cfg,
            )

            self.assertTrue(
                lint["ok"],
                lint,
            )

            inbox = root / "RESULTS_INBOX"

            ingestion = ingest_result_package(
                package,
                inbox,
                match_config=cfg,
                received_at=(
                    "2026-09-24T09:02:00Z"
                ),
            )

            self.assertEqual(
                ingestion["state"],
                "ACCEPTED",
                ingestion,
            )

            verification = (
                verify_published_result(
                    inbox,
                    cfg["config_id"],
                )
            )

            self.assertTrue(
                verification["ok"],
                verification,
            )

            published = (
                inbox
                / cfg["config_id"]
            )

            parsed = parse_pgn_file(
                published / "games.pgn",
                candidate_label=(
                    cfg["candidate"][
                        "artifact_id"
                    ]
                ),
                opponent_label=(
                    cfg["opponent"][
                        "artifact_id"
                    ]
                ),
            )

            self.assertEqual(
                parsed.excluded,
                [],
            )

            self.assertEqual(
                pentanomial_counts(
                    parsed.parsed_games
                ),
                (0, 0, 1, 0, 0),
            )


if __name__ == "__main__":
    unittest.main()