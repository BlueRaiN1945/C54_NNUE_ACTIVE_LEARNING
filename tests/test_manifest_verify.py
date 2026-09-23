import copy
import unittest

from medium_pc_audit.manifest_verify import (
    V1_LIMITATIONS,
    verify_execution_manifest,
)


H64 = {
    "candidate": "a" * 64,
    "opponent": "b" * 64,
    "engine": "c" * 64,
    "cli": "d" * 64,
    "openings": "e" * 64,
    "identity": "f" * 64,
}


def config():
    return {
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
            "max_games": 400,
        },
        "created_at": "2026-09-23T00:00:00Z",
        "created_by": "phase5-test",
        "identity_sha256": H64["identity"],
    }


def manifest(cfg=None):
    cfg = cfg or config()

    return {
        "schema_version": "v1",
        "run_id": cfg["config_id"],
        "actual_engine_sha256": cfg["engine"]["expected_sha256"],
        "actual_cli_sha256": cfg["cli"]["expected_sha256"],
        "actual_uci_options": copy.deepcopy(cfg["uci_options"]),
        "actual_time_control": copy.deepcopy(cfg["time_control"]),
        "actual_threads": cfg["threads"],
        "actual_hash_mb": cfg["hash_mb"],
        "actual_concurrency": cfg["concurrency"],
        "actual_seed": cfg["seed"],
        "actual_opening_suite_sha256": cfg["opening_suite"]["sha256"],
        "host_identity_label": "synthetic-host",
        "executed_at_utc": "2026-09-23T00:01:00Z",
        "executed_by": "synthetic-worker",
    }


class ManifestVerifyTests(unittest.TestCase):

    def test_matching_manifest_passes(self):
        cfg = config()
        result = verify_execution_manifest(
            cfg,
            manifest(cfg),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(
            len(result["verified_fields"]),
            10,
        )

    def test_every_bound_field_mismatch_is_detected(self):
        cfg = config()

        mutations = {
            "run_id": lambda m: m.__setitem__(
                "run_id",
                "AUDIT_V0001__OTHER__run001",
            ),
            "engine.sha256": lambda m: m.__setitem__(
                "actual_engine_sha256",
                "0" * 64,
            ),
            "cli.sha256": lambda m: m.__setitem__(
                "actual_cli_sha256",
                "1" * 64,
            ),
            "uci_options": lambda m: m.__setitem__(
                "actual_uci_options",
                {"Threads": 2, "Hash": 64},
            ),
            "time_control": lambda m: m.__setitem__(
                "actual_time_control",
                {"nodes": 60000},
            ),
            "threads": lambda m: m.__setitem__(
                "actual_threads",
                2,
            ),
            "hash_mb": lambda m: m.__setitem__(
                "actual_hash_mb",
                128,
            ),
            "concurrency": lambda m: m.__setitem__(
                "actual_concurrency",
                4,
            ),
            "seed": lambda m: m.__setitem__(
                "actual_seed",
                999,
            ),
            "opening_suite.sha256": lambda m: m.__setitem__(
                "actual_opening_suite_sha256",
                "2" * 64,
            ),
        }

        for expected_field, mutate in mutations.items():
            with self.subTest(field=expected_field):
                m = manifest(cfg)
                mutate(m)

                result = verify_execution_manifest(
                    cfg,
                    m,
                )

                self.assertFalse(result["ok"])

                fields = {
                    item["field"]
                    for item in result["mismatches"]
                }

                self.assertIn(
                    expected_field,
                    fields,
                )

    def test_invalid_config_schema_is_structured_failure(self):
        cfg = config()
        del cfg["seed"]

        result = verify_execution_manifest(
            cfg,
            manifest(),
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["schema_errors"])

    def test_v1_artifact_hash_limitation_is_explicit(self):
        result = verify_execution_manifest(
            config(),
            manifest(),
        )

        self.assertEqual(
            tuple(result["limitations"]),
            V1_LIMITATIONS,
        )

        self.assertEqual(
            len(result["limitations"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
