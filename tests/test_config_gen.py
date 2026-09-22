"""Tests for medium_pc_audit.config_gen -- Phase 2.

All artifacts used here are synthetic, sealed via Phase 1's real seal(),
never real BigPC/Stockfish/NNUE data.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medium_pc_audit import config_gen, identity
from medium_pc_audit.schemas import registry
from tests.fixtures.synthetic_sealed_artifacts import seal_synthetic_artifact

FIXED_NOW = "2026-09-22T12:00:00Z"


def _valid_manifest(files):
    return {
        "schema_version": "v1",
        "artifact_id": "X",
        "artifact_type": "opening_suite",
        "files": files,
        "source_description": "d",
        "imported_at": "2026-09-22T00:00:00Z",
        "imported_by": "tester",
        "sealed": True,
    }


def _with_recomputed_identity(config, **overrides):
    """Mutate config and recompute identity_sha256 to match, so a test can
    isolate a single invariant (e.g. config_id semantics) without also
    tripping the (separate) identity_sha256-mismatch check.
    """
    mutated = dict(config)
    mutated.update(overrides)
    mutated["identity_sha256"] = config_gen.compute_config_identity_sha256(mutated)
    return mutated


class ArtifactContentHashTests(unittest.TestCase):
    def test_file_order_does_not_change_hash(self):
        files_a = [
            {"path": "a.txt", "sha256": "1" * 64, "size_bytes": 1},
            {"path": "b.txt", "sha256": "2" * 64, "size_bytes": 2},
        ]
        files_b = list(reversed(files_a))

        hash_a = config_gen.compute_artifact_content_hash(_valid_manifest(files_a))
        hash_b = config_gen.compute_artifact_content_hash(_valid_manifest(files_b))
        self.assertEqual(hash_a, hash_b)

    def test_import_metadata_does_not_change_hash(self):
        files = [{"path": "a.txt", "sha256": "1" * 64, "size_bytes": 1}]
        manifest_a = _valid_manifest(files)
        manifest_b = dict(manifest_a)
        manifest_b.update(
            {
                "artifact_id": "DIFFERENT_ID",
                "artifact_type": "champion_v1_anchor",
                "source_description": "totally different description",
                "imported_at": "2099-01-01T00:00:00Z",
                "imported_by": "someone-else",
            }
        )

        hash_a = config_gen.compute_artifact_content_hash(manifest_a)
        hash_b = config_gen.compute_artifact_content_hash(manifest_b)
        self.assertEqual(hash_a, hash_b)

    def test_changing_path_changes_hash(self):
        base_files = [{"path": "a.txt", "sha256": "1" * 64, "size_bytes": 1}]
        changed_files = [{"path": "b.txt", "sha256": "1" * 64, "size_bytes": 1}]
        self.assertNotEqual(
            config_gen.compute_artifact_content_hash(_valid_manifest(base_files)),
            config_gen.compute_artifact_content_hash(_valid_manifest(changed_files)),
        )

    def test_changing_sha256_changes_hash(self):
        base_files = [{"path": "a.txt", "sha256": "1" * 64, "size_bytes": 1}]
        changed_files = [{"path": "a.txt", "sha256": "2" * 64, "size_bytes": 1}]
        self.assertNotEqual(
            config_gen.compute_artifact_content_hash(_valid_manifest(base_files)),
            config_gen.compute_artifact_content_hash(_valid_manifest(changed_files)),
        )

    def test_changing_size_changes_hash(self):
        base_files = [{"path": "a.txt", "sha256": "1" * 64, "size_bytes": 1}]
        changed_files = [{"path": "a.txt", "sha256": "1" * 64, "size_bytes": 999}]
        self.assertNotEqual(
            config_gen.compute_artifact_content_hash(_valid_manifest(base_files)),
            config_gen.compute_artifact_content_hash(_valid_manifest(changed_files)),
        )


class ConfigGenTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.artifacts_root = self.base / "artifacts"
        self.artifacts_root.mkdir()
        self.configs_root = self.base / "configs"
        self.configs_root.mkdir()

        self.candidate_dir = seal_synthetic_artifact(
            self.base, artifact_type="audit_champion", artifact_id="AUDIT_CHAMPION_V0011",
            dest_root=self.artifacts_root,
        )
        self.teacher_dir = seal_synthetic_artifact(
            self.base, artifact_type="teacher_anchor", artifact_id="TEACHER_V0",
            dest_root=self.artifacts_root,
        )
        self.champion_v1_dir = seal_synthetic_artifact(
            self.base, artifact_type="champion_v1_anchor", artifact_id="CHAMPION_V1",
            dest_root=self.artifacts_root,
        )
        self.prev_audit_dir = seal_synthetic_artifact(
            self.base, artifact_type="audit_champion", artifact_id="AUDIT_CHAMPION_V0001",
            dest_root=self.artifacts_root,
        )
        self.opening_suite_dir = seal_synthetic_artifact(
            self.base, artifact_type="opening_suite", artifact_id="OPENINGS_V1",
            dest_root=self.artifacts_root,
        )

    def build_default(self, **overrides):
        kwargs = dict(
            audit_id="AUDIT_V0011",
            matchup_type="vs_teacher",
            candidate_artifact_dir=self.candidate_dir,
            opponent_artifact_dir=self.teacher_dir,
            engine_pinned_id="ENGINE_V1",
            engine_expected_sha256="a" * 64,
            cli_pinned_id="CLI_V1",
            cli_expected_sha256="b" * 64,
            opening_suite_artifact_dir=self.opening_suite_dir,
            uci_options={"Threads": 1, "Hash": 64},
            time_control={"nodes": 50000},
            threads=1,
            hash_mb=64,
            concurrency=8,
            repeat=True,
            seed=12345,
            sprt={"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": 4000},
            attempt=1,
            created_by="tester",
            now_utc=FIXED_NOW,
        )
        kwargs.update(overrides)
        return config_gen.build_match_config(**kwargs)


class BuildMatchConfigHappyPathTests(ConfigGenTestCase):
    def test_build_produces_schema_valid_config(self):
        config = self.build_default()
        registry.validate("match_config", "v1", config)

    def test_config_id_equals_canonical_run_id(self):
        config = self.build_default()

        expected_matchup_key = identity.compute_matchup_key("AUDIT_V0011", "TEACHER_V0")
        expected_run_id = identity.compute_run_id(expected_matchup_key, 1)
        self.assertEqual(config["matchup_key"], expected_matchup_key)
        self.assertEqual(config["config_id"], expected_run_id)

    def test_identity_sha256_deterministic(self):
        config_a = self.build_default()
        config_b = self.build_default()
        self.assertEqual(config_a["identity_sha256"], config_b["identity_sha256"])

    def test_created_at_excluded_created_by_included(self):
        config_a = self.build_default(now_utc=FIXED_NOW)
        config_b = self.build_default(now_utc="2099-01-01T00:00:00Z")
        self.assertEqual(config_a["identity_sha256"], config_b["identity_sha256"])

        config_c = self.build_default(created_by="different-operator")
        self.assertNotEqual(config_a["identity_sha256"], config_c["identity_sha256"])

    def test_identity_sha256_changes_when_hash_relevant_field_changes(self):
        baseline = self.build_default()
        mutations = {
            "seed": 99999,
            "threads": 2,
            "hash_mb": 128,
            "concurrency": 4,
            "repeat": False,
            "uci_options": {"Threads": 2, "Hash": 64},
            "time_control": {"movetime_ms": 1000},
            "sprt": {"elo0": 0, "elo1": 10, "alpha": 0.05, "beta": 0.05, "max_games": 4000},
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                mutated = self.build_default(**{field: value})
                self.assertNotEqual(baseline["identity_sha256"], mutated["identity_sha256"])


class MatchupRoleValidationTests(ConfigGenTestCase):
    def test_vs_teacher_requires_teacher_anchor_opponent(self):
        self.build_default(matchup_type="vs_teacher", opponent_artifact_dir=self.teacher_dir)
        with self.assertRaises(config_gen.InvalidMatchupRole):
            self.build_default(matchup_type="vs_teacher", opponent_artifact_dir=self.champion_v1_dir)

    def test_vs_champion_v1_requires_champion_v1_anchor_opponent(self):
        self.build_default(matchup_type="vs_champion_v1", opponent_artifact_dir=self.champion_v1_dir)
        with self.assertRaises(config_gen.InvalidMatchupRole):
            self.build_default(matchup_type="vs_champion_v1", opponent_artifact_dir=self.teacher_dir)

    def test_vs_prev_audit_champion_requires_audit_champion_opponent(self):
        self.build_default(matchup_type="vs_prev_audit_champion", opponent_artifact_dir=self.prev_audit_dir)
        with self.assertRaises(config_gen.InvalidMatchupRole):
            self.build_default(matchup_type="vs_prev_audit_champion", opponent_artifact_dir=self.teacher_dir)

    def test_opening_suite_must_have_opening_suite_type(self):
        with self.assertRaises(config_gen.InvalidMatchupRole):
            self.build_default(opening_suite_artifact_dir=self.teacher_dir)

    def test_candidate_must_be_allowed_type(self):
        self.assertEqual(config_gen.ALLOWED_CANDIDATE_ARTIFACT_TYPES, frozenset({"audit_champion"}))
        with self.assertRaises(config_gen.InvalidMatchupRole):
            self.build_default(candidate_artifact_dir=self.teacher_dir)


class UnsealedArtifactTests(ConfigGenTestCase):
    def test_rejects_when_candidate_fails_verify_seal(self):
        (self.candidate_dir / "files" / "data.bin").write_bytes(b"TAMPERED")
        with self.assertRaises(config_gen.UnsealedArtifactReference):
            self.build_default()


class TimeControlValidationTests(ConfigGenTestCase):
    def test_accepts_nodes(self):
        self.build_default(time_control={"nodes": 1})

    def test_accepts_movetime_ms(self):
        self.build_default(time_control={"movetime_ms": 1})

    def test_accepts_tc_string(self):
        self.build_default(time_control={"tc": "10+0.1"})

    def test_accepts_punctuation_heavy_tc_strings(self):
        # Representative real chess time-control syntax: '/', ':', '+', '.'
        # must all be accepted -- none of these are filesystem paths.
        representative_tc_strings = [
            "40/9000:10+0.1",
            "moves/nodes=100000",
            "1:00+1",
            "10+0.1",
            "inf",
            "40/60+0.4",
        ]
        for tc in representative_tc_strings:
            with self.subTest(tc=tc):
                config = self.build_default(time_control={"tc": tc})
                self.assertEqual(config["time_control"], {"tc": tc})

    def test_rejects_zero_keys(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={})

    def test_rejects_multiple_keys(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"nodes": 1, "movetime_ms": 1})

    def test_rejects_unknown_key(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"depth": 10})

    def test_rejects_bool_for_nodes(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"nodes": True})

    def test_rejects_non_positive_nodes(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"nodes": 0})

    def test_rejects_empty_tc_string(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"tc": ""})

    def test_rejects_tc_with_nul(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"tc": "10+0.1\x00"})

    def test_rejects_tc_with_cr(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"tc": "10+0.1\r"})

    def test_rejects_tc_with_lf(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"tc": "10+0.1\n"})

    def test_rejects_tc_with_other_control_character(self):
        with self.assertRaises(config_gen.InvalidTimeControl):
            self.build_default(time_control={"tc": "10+0.1\x01"})


class SprtValidationTests(ConfigGenTestCase):
    def test_rejects_elo0_not_less_than_elo1(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": 5, "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": 100})

    def test_rejects_alpha_out_of_range(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": 0, "elo1": 5, "alpha": 1.5, "beta": 0.05, "max_games": 100})

    def test_rejects_beta_out_of_range(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.0, "max_games": 100})

    def test_rejects_max_games_less_than_one(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": 0})

    def test_rejects_bool_max_games(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": True})

    def test_rejects_non_finite_elo(self):
        with self.assertRaises(config_gen.InvalidSprtParameters):
            self.build_default(sprt={"elo0": float("nan"), "elo1": 5, "alpha": 0.05, "beta": 0.05, "max_games": 100})


class UciOptionsValidationTests(ConfigGenTestCase):
    def test_rejects_empty_key(self):
        with self.assertRaises(config_gen.InvalidUciOptions):
            self.build_default(uci_options={"": 1})

    def test_rejects_non_primitive_value(self):
        with self.assertRaises(config_gen.InvalidUciOptions):
            self.build_default(uci_options={"Threads": [1, 2]})

    def test_rejects_nan_float(self):
        with self.assertRaises(config_gen.InvalidUciOptions):
            self.build_default(uci_options={"Contempt": float("nan")})

    def test_rejects_infinite_float(self):
        with self.assertRaises(config_gen.InvalidUciOptions):
            self.build_default(uci_options={"Contempt": float("inf")})

    def test_accepts_bool_int_float_str_values(self):
        self.build_default(uci_options={"Ponder": False, "Threads": 1, "Contempt": 1.5, "Variant": "standard"})

    def test_rejects_path_like_string_value(self):
        with self.assertRaises(config_gen.InvalidConfigContent):
            self.build_default(uci_options={"EvalFile": "C:\\some\\path.nnue"})


class PathLeakageTests(ConfigGenTestCase):
    def test_rejects_path_like_created_by(self):
        with self.assertRaises(config_gen.InvalidConfigContent):
            self.build_default(created_by="C:\\Users\\someone")


class PublishConfigTests(ConfigGenTestCase):
    def test_publish_creates_expected_layout(self):
        config = self.build_default()
        final_dir = config_gen.publish_config(config, configs_root=self.configs_root)

        self.assertEqual(final_dir, self.configs_root / config["matchup_key"] / config["config_id"])
        self.assertTrue((final_dir / "match_config.json").is_file())
        self.assertTrue((final_dir / "SHA256SUMS.txt").is_file())
        self.assertEqual({p.name for p in final_dir.iterdir()}, {"match_config.json", "SHA256SUMS.txt"})

    def test_sha256sums_matches_exact_published_bytes(self):
        import hashlib

        config = self.build_default()
        final_dir = config_gen.publish_config(config, configs_root=self.configs_root)

        published_bytes = (final_dir / "match_config.json").read_bytes()
        expected_digest = hashlib.sha256(published_bytes).hexdigest()
        sums_text = (final_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
        self.assertEqual(sums_text, f"{expected_digest}  match_config.json\n")

    def test_published_bytes_are_lf_utf8_single_trailing_newline(self):
        config = self.build_default()
        final_dir = config_gen.publish_config(config, configs_root=self.configs_root)

        raw = (final_dir / "match_config.json").read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        text = raw.decode("utf-8")
        self.assertEqual(json.loads(text), config)

    def test_publish_rejects_when_final_already_exists(self):
        config = self.build_default()
        config_gen.publish_config(config, configs_root=self.configs_root)

        with self.assertRaises(config_gen.ConfigAlreadyExists):
            config_gen.publish_config(config, configs_root=self.configs_root)

    def test_simulated_publication_failure_leaves_no_final_package(self):
        config = self.build_default()

        real_write = config_gen._atomic_write_bytes
        state = {"calls": 0}

        def failing_write(path, data):
            state["calls"] += 1
            if state["calls"] == 2:
                raise RuntimeError("simulated publication failure")
            real_write(path, data)

        with mock.patch.object(config_gen, "_atomic_write_bytes", side_effect=failing_write):
            with self.assertRaises(RuntimeError):
                config_gen.publish_config(config, configs_root=self.configs_root)

        final_dir = self.configs_root / config["matchup_key"] / config["config_id"]
        self.assertFalse(final_dir.exists())

        type_root = self.configs_root / config["matchup_key"]
        leftover = [p for p in type_root.iterdir() if p.name.startswith(".staging-")]
        self.assertEqual(leftover, [], "failed publish must not leave a staging directory behind")

    def test_publish_detects_tampered_identity_sha256(self):
        config = self.build_default()
        tampered = copy.deepcopy(config)
        tampered["identity_sha256"] = "0" * 64

        with self.assertRaises(config_gen.ConfigIdentityMismatch):
            config_gen.publish_config(tampered, configs_root=self.configs_root)

        final_dir = self.configs_root / tampered["matchup_key"] / tampered["config_id"]
        self.assertFalse(final_dir.exists())

    def test_regeneration_requires_new_attempt(self):
        config_1 = self.build_default(attempt=1)
        dir_1 = config_gen.publish_config(config_1, configs_root=self.configs_root)

        config_2 = self.build_default(attempt=2)
        dir_2 = config_gen.publish_config(config_2, configs_root=self.configs_root)

        self.assertNotEqual(dir_1, dir_2)
        self.assertTrue(dir_1.is_dir())
        self.assertTrue(dir_2.is_dir())

        with self.assertRaises(config_gen.ConfigAlreadyExists):
            config_gen.publish_config(config_1, configs_root=self.configs_root)


class PublishConfigIdSemanticsTests(ConfigGenTestCase):
    """Direct publish_config() tests using manually constructed
    otherwise-valid configs (schema-valid, identity_sha256 recomputed to
    match) proving semantic config_id/matchup_key mismatches cannot be
    published -- publish_config() must not rely on construction-time
    correctness.
    """

    def _assert_rejected_and_unpublished(self, bad_config):
        with self.assertRaises(config_gen.ConfigIdentityMismatch):
            config_gen.publish_config(bad_config, configs_root=self.configs_root)

        final_dir = self.configs_root / bad_config["matchup_key"] / bad_config["config_id"]
        self.assertFalse(final_dir.exists())

    def test_rejects_malformed_config_id(self):
        config = self.build_default()
        bad = _with_recomputed_identity(config, config_id=config["matchup_key"] + "_bad_suffix")
        self._assert_rejected_and_unpublished(bad)

    def test_rejects_wrong_matchup_config_id(self):
        config = self.build_default()
        bad = _with_recomputed_identity(config, config_id="SOME_OTHER_MATCHUP__run001")
        self._assert_rejected_and_unpublished(bad)

    def test_rejects_zero_attempt_config_id(self):
        config = self.build_default()
        bad = _with_recomputed_identity(config, config_id=config["matchup_key"] + "__run000")
        self._assert_rejected_and_unpublished(bad)

    def test_rejects_mismatched_padding_config_id(self):
        config = self.build_default()
        # "run1" (unpadded) does not equal the canonical "run001" -- the
        # round-trip through identity.compute_run_id() must catch this.
        bad = _with_recomputed_identity(config, config_id=config["matchup_key"] + "__run1")
        self._assert_rejected_and_unpublished(bad)

    def test_correctly_formed_config_id_still_publishes(self):
        # Sanity check: the new check does not reject a genuinely valid config.
        config = self.build_default()
        final_dir = config_gen.publish_config(config, configs_root=self.configs_root)
        self.assertTrue(final_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
