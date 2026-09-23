import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.config_gen import (
    compute_config_identity_sha256,
)
from medium_pc_audit.inbox_validate import (
    ACCEPTED,
    INCOMPLETE,
    QUARANTINED,
)
from medium_pc_audit.result_package import (
    REJECTED_DUPLICATE,
    ingest_result_package,
    verify_artifact_binding,
    verify_published_result,
    verify_transport_integrity,
)
from medium_pc_audit.state import (
    RECEIVED,
    read_state_history,
)


H64 = {
    "candidate": "a" * 64,
    "opponent": "b" * 64,
    "engine": "c" * 64,
    "cli": "d" * 64,
    "openings": "e" * 64,
}


def make_config(max_games=4):
    cfg = {
        "schema_version": "v1",
        "config_id": "AUDIT_V0001__TEST__run001",
        "audit_id": "AUDIT_V0001",
        "matchup_key": "AUDIT_V0001__TEST",
        "matchup_type": "vs_teacher",
        "candidate": {
            "artifact_id": "CANDIDATE",
            "sha256": H64["candidate"],
        },
        "opponent": {
            "artifact_id": "TEACHER",
            "sha256": H64["opponent"],
        },
        "engine": {
            "pinned_id": "stockfish-test",
            "expected_sha256": H64["engine"],
        },
        "cli": {
            "pinned_id": "cchess-test",
            "expected_sha256": H64["cli"],
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
        "concurrency": 8,
        "opening_suite": {
            "artifact_id": "OPENINGS",
            "sha256": H64["openings"],
        },
        "repeat": True,
        "seed": 12345,
        "sprt": {
            "elo0": 0,
            "elo1": 5,
            "alpha": 0.05,
            "beta": 0.05,
            "max_games": max_games,
        },
        "created_at": "2026-09-23T09:00:00Z",
        "created_by": "phase5b-test",
    }

    cfg["identity_sha256"] = (
        compute_config_identity_sha256(
            cfg
        )
    )

    return cfg


def make_manifest(cfg):
    return {
        "schema_version": "v1",
        "run_id": cfg["config_id"],
        "actual_engine_sha256": cfg["engine"]["expected_sha256"],
        "actual_cli_sha256": cfg["cli"]["expected_sha256"],
        "actual_uci_options": copy.deepcopy(
            cfg["uci_options"]
        ),
        "actual_time_control": copy.deepcopy(
            cfg["time_control"]
        ),
        "actual_threads": cfg["threads"],
        "actual_hash_mb": cfg["hash_mb"],
        "actual_concurrency": cfg["concurrency"],
        "actual_seed": cfg["seed"],
        "actual_opening_suite_sha256": (
            cfg["opening_suite"]["sha256"]
        ),
        "host_identity_label": "synthetic-host",
        "executed_at_utc": "2026-09-23T09:01:00Z",
        "executed_by": "synthetic-worker",
    }


def make_binding(cfg):
    return {
        "sidecar_version": "v1",
        "run_id": cfg["config_id"],
        "config_identity_sha256": cfg["identity_sha256"],
        "actual_candidate_sha256": cfg["candidate"]["sha256"],
        "actual_opponent_sha256": cfg["opponent"]["sha256"],
    }


def game(
    index,
    opening,
    candidate_color,
    result,
):
    if candidate_color == "white":
        white_role = "candidate"
        black_role = "opponent"
    else:
        white_role = "opponent"
        black_role = "candidate"

    return {
        "game_index": index,
        "white_role": white_role,
        "black_role": black_role,
        "result": result,
        "opening_ref": opening,
    }


def complete_games():
    return [
        game(
            1,
            "opening-1",
            "white",
            "1/2-1/2",
        ),
        game(
            2,
            "opening-1",
            "black",
            "1/2-1/2",
        ),
        game(
            3,
            "opening-2",
            "white",
            "1/2-1/2",
        ),
        game(
            4,
            "opening-2",
            "black",
            "1/2-1/2",
        ),
    ]


def partial_games():
    return [
        game(
            1,
            "opening-1",
            "white",
            "1/2-1/2",
        ),
        game(
            2,
            "opening-1",
            "black",
            "1/2-1/2",
        ),
    ]


def make_raw(cfg, games):
    return {
        "schema_version": "v1",
        "run_id": cfg["config_id"],
        "games": games,
        "claimed_wins": 999,
        "claimed_losses": 0,
        "claimed_draws": 0,
        "engine_stdout_log_ref": "match.log",
        "pgn_ref": "games.pgn",
    }


def write_json(path, obj):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            obj,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        handle.write("\n")


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def reseal_package(package_dir):
    package_dir = Path(
        package_dir
    )

    checksum = (
        package_dir
        / "SHA256SUMS.txt"
    )

    if checksum.exists():
        checksum.unlink()

    files = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file()
    )

    lines = []

    for path in files:
        relative = "/".join(
            path.relative_to(
                package_dir
            ).parts
        )

        lines.append(
            f"{sha256(path)}  {relative}\n"
        )

    with open(
        checksum,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.writelines(
            lines
        )


def create_package(
    package_dir,
    cfg,
    *,
    games=None,
):
    package_dir = Path(
        package_dir
    )

    package_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if games is None:
        games = complete_games()

    write_json(
        package_dir
        / "execution_manifest.json",
        make_manifest(cfg),
    )

    write_json(
        package_dir
        / "raw_result.json",
        make_raw(
            cfg,
            games,
        ),
    )

    write_json(
        package_dir
        / "artifact_binding.json",
        make_binding(cfg),
    )

    (
        package_dir
        / "games.pgn"
    ).write_bytes(
        b"[Event \"synthetic\"]\n\n1/2-1/2\n"
    )

    (
        package_dir
        / "match.log"
    ).write_bytes(
        b"synthetic match log\n"
    )

    reseal_package(
        package_dir
    )


class ArtifactBindingTests(unittest.TestCase):

    def test_matching_binding_passes(self):
        cfg = make_config()

        result = verify_artifact_binding(
            cfg,
            make_binding(cfg),
        )

        self.assertTrue(
            result["ok"]
        )
        self.assertEqual(
            result["mismatches"],
            [],
        )

    def test_each_identity_mismatch_is_detected(self):
        cfg = make_config()

        mutations = {
            "run_id": "AUDIT_V0001__OTHER__run001",
            "config_identity_sha256": "0" * 64,
            "actual_candidate_sha256": "1" * 64,
            "actual_opponent_sha256": "2" * 64,
        }

        for field, value in mutations.items():
            with self.subTest(
                field=field
            ):
                binding = make_binding(
                    cfg
                )
                binding[field] = value

                result = (
                    verify_artifact_binding(
                        cfg,
                        binding,
                    )
                )

                self.assertFalse(
                    result["ok"]
                )

                mismatch_fields = {
                    item["field"]
                    for item in result[
                        "mismatches"
                    ]
                }

                self.assertIn(
                    field,
                    mismatch_fields,
                )

    def test_extra_or_missing_binding_fields_rejected(self):
        cfg = make_config()

        extra = make_binding(cfg)
        extra["surprise"] = True

        self.assertFalse(
            verify_artifact_binding(
                cfg,
                extra,
            )["ok"]
        )

        missing = make_binding(cfg)
        del missing[
            "actual_candidate_sha256"
        ]

        self.assertFalse(
            verify_artifact_binding(
                cfg,
                missing,
            )["ok"]
        )


class TransportIntegrityTests(unittest.TestCase):

    def test_valid_package_passes(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config()
            package = Path(td) / "package"

            create_package(
                package,
                cfg,
            )

            result = (
                verify_transport_integrity(
                    package
                )
            )

            self.assertTrue(
                result["ok"],
                result["errors"],
            )

    def test_tampered_payload_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config()
            package = Path(td) / "package"

            create_package(
                package,
                cfg,
            )

            (
                package
                / "games.pgn"
            ).write_bytes(
                b"TAMPERED\n"
            )

            result = (
                verify_transport_integrity(
                    package
                )
            )

            self.assertFalse(
                result["ok"]
            )

    def test_unlisted_extra_file_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config()
            package = Path(td) / "package"

            create_package(
                package,
                cfg,
            )

            (
                package
                / "extra.bin"
            ).write_bytes(
                b"unexpected"
            )

            result = (
                verify_transport_integrity(
                    package
                )
            )

            self.assertFalse(
                result["ok"]
            )

    def test_unsorted_checksum_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config()
            package = Path(td) / "package"

            create_package(
                package,
                cfg,
            )

            checksum = (
                package
                / "SHA256SUMS.txt"
            )

            lines = checksum.read_text(
                encoding="utf-8"
            ).splitlines(
                keepends=True
            )

            checksum.write_text(
                "".join(
                    reversed(lines)
                ),
                encoding="utf-8",
                newline="\n",
            )

            result = (
                verify_transport_integrity(
                    package
                )
            )

            self.assertFalse(
                result["ok"]
            )

    def test_missing_required_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config()
            package = Path(td) / "package"

            create_package(
                package,
                cfg,
            )

            (
                package
                / "artifact_binding.json"
            ).unlink()

            reseal_package(
                package
            )

            result = (
                verify_transport_integrity(
                    package
                )
            )

            self.assertFalse(
                result["ok"]
            )


class IngestTests(unittest.TestCase):

    def test_complete_package_is_published_and_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config(
                max_games=4
            )

            source = (
                root
                / "source"
            )

            inbox = (
                root
                / "inbox"
            )

            create_package(
                source,
                cfg,
            )

            source_checksum = (
                source
                / "SHA256SUMS.txt"
            ).read_bytes()

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                ACCEPTED,
            )

            self.assertTrue(
                result["published"]
            )

            final = (
                inbox
                / cfg["config_id"]
            )

            self.assertTrue(
                final.is_dir()
            )

            self.assertEqual(
                (
                    final
                    / "SHA256SUMS.txt"
                ).read_bytes(),
                source_checksum,
            )

            history = read_state_history(
                inbox,
                cfg["config_id"],
            )

            self.assertEqual(
                [
                    event["state"]
                    for event in history
                ],
                [
                    RECEIVED,
                    ACCEPTED,
                ],
            )

    def test_partial_run_is_published_but_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config(
                max_games=400
            )

            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
                games=partial_games(),
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                INCOMPLETE,
            )

            self.assertTrue(
                result["published"]
            )

    def test_binding_mismatch_is_preserved_but_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            binding_path = (
                source
                / "artifact_binding.json"
            )

            binding = json.loads(
                binding_path.read_text(
                    encoding="utf-8"
                )
            )

            binding[
                "actual_candidate_sha256"
            ] = "0" * 64

            write_json(
                binding_path,
                binding,
            )

            reseal_package(
                source
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                QUARANTINED,
            )

            self.assertTrue(
                result["published"]
            )

            self.assertIn(
                "artifact_binding_mismatch",
                result["reason_codes"],
            )

    def test_invalid_json_is_preserved_but_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            (
                source
                / "raw_result.json"
            ).write_bytes(
                b"{ definitely-not-json }\n"
            )

            reseal_package(
                source
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                QUARANTINED,
            )

            self.assertTrue(
                result["published"]
            )

            self.assertIn(
                "required_json_unreadable",
                result["reason_codes"],
            )

    def test_missing_referenced_evidence_is_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            raw_path = (
                source
                / "raw_result.json"
            )

            raw = json.loads(
                raw_path.read_text(
                    encoding="utf-8"
                )
            )

            raw["pgn_ref"] = (
                "missing-games.pgn"
            )

            write_json(
                raw_path,
                raw,
            )

            reseal_package(
                source
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                QUARANTINED,
            )

            self.assertIn(
                "evidence_reference_invalid",
                result["reason_codes"],
            )

    def test_duplicate_run_id_never_overwrites_first_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            first = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                first["state"],
                ACCEPTED,
            )

            final_pgn = (
                inbox
                / cfg["config_id"]
                / "games.pgn"
            )

            before = final_pgn.read_bytes()

            (
                source
                / "games.pgn"
            ).write_bytes(
                b"DIFFERENT SECOND ATTEMPT\n"
            )

            reseal_package(
                source
            )

            second = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:11:00Z",
            )

            self.assertEqual(
                second["state"],
                REJECTED_DUPLICATE,
            )

            self.assertEqual(
                final_pgn.read_bytes(),
                before,
            )

            history = read_state_history(
                inbox,
                cfg["config_id"],
            )

            self.assertEqual(
                history[-1]["state"],
                REJECTED_DUPLICATE,
            )

    def test_transport_failure_is_not_published(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            (
                source
                / "games.pgn"
            ).write_bytes(
                b"TAMPER AFTER CHECKSUM\n"
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                QUARANTINED,
            )

            self.assertFalse(
                result["published"]
            )

            self.assertFalse(
                (
                    inbox
                    / cfg["config_id"]
                ).exists()
            )

            self.assertIn(
                "transport_integrity_failed",
                result["reason_codes"],
            )

    def test_post_publication_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = make_config()
            source = root / "source"
            inbox = root / "inbox"

            create_package(
                source,
                cfg,
            )

            result = ingest_result_package(
                source,
                inbox,
                match_config=cfg,
                received_at="2026-09-23T09:10:00Z",
            )

            self.assertEqual(
                result["state"],
                ACCEPTED,
            )

            (
                inbox
                / cfg["config_id"]
                / "match.log"
            ).write_bytes(
                b"TAMPERED AFTER PUBLICATION\n"
            )

            verification = (
                verify_published_result(
                    inbox,
                    cfg["config_id"],
                )
            )

            self.assertFalse(
                verification["ok"]
            )


if __name__ == "__main__":
    unittest.main()
