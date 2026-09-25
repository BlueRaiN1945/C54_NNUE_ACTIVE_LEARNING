"""Tests for M0 corpus provenance: independent hash verification + evidence."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.corpus_provenance import (
    CorpusProvenanceError,
    build_corpus_data_source,
)
from medium_pc_audit.experiment import (
    CLAIM_CORPUS_HASH_VERIFIED,
    CLAIM_CORPUS_IDENTIFIED,
    CLAIM_HERITAGE_FREE,
)
from medium_pc_audit.frozen_seed import SEED_SHA256

WRONG_SHA = "0" * 64


def write_artifact(directory, content=b"synthetic modern corpus bytes"):
    path = Path(directory) / "modern_pool.binpack"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def base_record(artifact_sha256, **overrides):
    record = {
        "schema_version": "v1",
        "artifact_sha256": artifact_sha256,
        "seed_sha256": SEED_SHA256,
        "engine_sha256": "1" * 64,
        "cli_sha256": "2" * 64,
        "opening_suite_sha256": "3" * 64,
        "trainer_commit": "9f72946529c4187d3679014036cd22c3be419716",
        "self_play_games": 100,
        "self_play_nodes": 50000,
        "rescore_nodes": 200000,
        "extraction_skip_first_n_plies": 6,
        "created_utc": "2026-09-25T00:00:00Z",
    }
    record.update(overrides)
    return record


class BuildCorpusDataSourceTests(unittest.TestCase):
    def test_matching_hash_and_seed_establishes_heritage_free(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="M0_SELFPLAY_POOL_V1",
                generation_record=base_record(real_sha),
                justification="self-play from frozen seed, non-heritage book",
            )

            self.assertEqual(spec.sha256, real_sha)
            self.assertTrue(spec.is_heritage_free_established)
            self.assertTrue(spec.establishes(CLAIM_CORPUS_IDENTIFIED))
            self.assertTrue(spec.establishes(CLAIM_CORPUS_HASH_VERIFIED))
            self.assertTrue(spec.establishes(CLAIM_HERITAGE_FREE))

    def test_spec_sha256_is_independently_computed_not_the_records_claim(self):
        """The DataSourceSpec's hash must come from OUR hashing, never the
        producer's claim copied through -- otherwise a lying record would
        pass by construction."""

        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="X",
                generation_record=base_record(real_sha),
                justification="j",
            )

            recomputed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(spec.sha256, recomputed)

    def test_artifact_hash_mismatch_blocks_heritage_free(self):
        """Fail-closed: producer's claimed hash disagrees with the actual
        bytes -- corroboration must fail, not silently pass."""

        with tempfile.TemporaryDirectory() as td:
            path, _real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="X",
                generation_record=base_record(WRONG_SHA),
                justification="j",
            )

            self.assertFalse(spec.establishes(CLAIM_CORPUS_HASH_VERIFIED))
            self.assertFalse(spec.is_heritage_free_established)

    def test_seed_hash_mismatch_raises(self):
        """A generation record built on a different seed than the frozen,
        approved one must be rejected outright, not merely warned about."""

        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            with self.assertRaises(CorpusProvenanceError) as ctx:
                build_corpus_data_source(
                    artifact_path=path,
                    source_id="X",
                    generation_record=base_record(real_sha, seed_sha256="9" * 64),
                    justification="j",
                )

            self.assertIn("seed", str(ctx.exception).lower())

    def test_missing_artifact_sha256_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)
            record = base_record(real_sha)
            del record["artifact_sha256"]

            with self.assertRaises(CorpusProvenanceError):
                build_corpus_data_source(
                    artifact_path=path,
                    source_id="X",
                    generation_record=record,
                    justification="j",
                )

    def test_missing_seed_sha256_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)
            record = base_record(real_sha)
            del record["seed_sha256"]

            with self.assertRaises(CorpusProvenanceError):
                build_corpus_data_source(
                    artifact_path=path,
                    source_id="X",
                    generation_record=record,
                    justification="j",
                )

    def test_artifact_path_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist.binpack"

            with self.assertRaises(CorpusProvenanceError):
                build_corpus_data_source(
                    artifact_path=missing,
                    source_id="X",
                    generation_record=base_record("a" * 64),
                    justification="j",
                )

    def test_non_dict_generation_record_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path, _real_sha = write_artifact(td)

            with self.assertRaises(CorpusProvenanceError):
                build_corpus_data_source(
                    artifact_path=path,
                    source_id="X",
                    generation_record="not-a-dict",
                    justification="j",
                )

    def test_evidence_result_and_expected_are_independent_on_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="X",
                generation_record=base_record(WRONG_SHA),
                justification="j",
            )

            hash_evidence = next(
                e for e in spec.evidence if e.kind == "artifact_hash_observation"
            )

            self.assertEqual(hash_evidence.result_sha256, real_sha)
            self.assertEqual(hash_evidence.expected_sha256, WRONG_SHA)
            self.assertNotEqual(
                hash_evidence.result_sha256, hash_evidence.expected_sha256
            )

    def test_detail_captures_recipe_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="X",
                generation_record=base_record(real_sha, self_play_games=12345),
                justification="j",
            )

            generation_evidence = next(
                e for e in spec.evidence if e.kind == "corpus_generation_record"
            )

            self.assertIn("12345", generation_evidence.detail)

    def test_source_id_and_justification_propagate(self):
        with tempfile.TemporaryDirectory() as td:
            path, real_sha = write_artifact(td)

            spec = build_corpus_data_source(
                artifact_path=path,
                source_id="M0_POOL_V7",
                generation_record=base_record(real_sha),
                justification="a specific stated reason",
            )

            self.assertEqual(spec.source_id, "M0_POOL_V7")
            self.assertEqual(spec.justification, "a specific stated reason")


if __name__ == "__main__":
    unittest.main()
