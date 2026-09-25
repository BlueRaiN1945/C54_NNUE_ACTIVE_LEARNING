"""Tests for experiment/arm provenance contracts."""

import unittest

from medium_pc_audit.experiment import (
    CLAIM_HISTORICAL_ORIGIN,
    CLAIM_SERIALIZATION_EQUIVALENCE,
    CLAIM_TRAINABILITY,
    EVIDENCE_CREATION_LOG,
    EVIDENCE_SERIALIZATION_ROUNDTRIP,
    EVIDENCE_TRAINING_SMOKE_RUN,
    HERITAGE_CURATED,
    HERITAGE_FREE_DECLARED,
    HERITAGE_UNKNOWN,
    ROLE_CONTROL_M0,
    ROLE_TREATMENT_HERITAGE,
    ArmSpec,
    DataSourceSpec,
    EvidenceRef,
    ExperimentError,
    SeedSpec,
    TrainingBudget,
    build_experiment,
)

SEED_A = "a" * 64
SEED_B = "b" * 64
SRC_A = "c" * 64
SRC_B = "d" * 64
NNUE_SHA = "9" * 64


def roundtrip_evidence(subject=SEED_A, result=NNUE_SHA, expected=NNUE_SHA):
    return EvidenceRef(
        kind=EVIDENCE_SERIALIZATION_ROUNDTRIP,
        record_id="/tmp/c54_seed_roundtrip_verify.nnue",
        subject_sha256=subject,
        verified_at="2026-09-24T08:40:00Z",
        result_sha256=result,
        expected_sha256=expected,
        detail="serialize.py roundtrip",
    )


def creation_evidence(subject=SEED_A):
    return EvidenceRef(
        kind=EVIDENCE_CREATION_LOG,
        record_id="/data/logs/official_base_creation.log",
        subject_sha256=subject,
        verified_at="2026-09-24T08:40:00Z",
    )


def training_evidence(subject=SEED_A):
    return EvidenceRef(
        kind=EVIDENCE_TRAINING_SMOKE_RUN,
        record_id="SMOKE_RUN_001",
        subject_sha256=subject,
        verified_at="2026-09-24T08:40:00Z",
    )


def seed(sha=SEED_A, **kw):
    return SeedSpec(artifact_id="OFFICIAL_BASE", sha256=sha, **kw)


def budget(**kw):
    defaults = {
        "lr": 5e-6,
        "epoch_size": 65536,
        "batch_size": 4096,
        "blocks": 1,
    }
    defaults.update(kw)
    return TrainingBudget(**defaults)


def free_source(source_id="MODERN_POOL", sha=SRC_A):
    return DataSourceSpec(
        source_id=source_id,
        sha256=sha,
        heritage_status=HERITAGE_FREE_DECLARED,
        justification="synthetic corpus, no OLA/GEN2 curation",
    )


def heritage_source(source_id="OLA_POOL", sha=SRC_B):
    return DataSourceSpec(
        source_id=source_id,
        sha256=sha,
        heritage_status=HERITAGE_CURATED,
    )


def control_arm(**kw):
    return ArmSpec(
        arm_id=kw.get("arm_id", "M0"),
        role=ROLE_CONTROL_M0,
        seed=kw.get("seed", seed()),
        data_source=kw.get("data_source", free_source()),
        budget=kw.get("budget", budget()),
    )


def treatment_arm(**kw):
    return ArmSpec(
        arm_id=kw.get("arm_id", "HERITAGE"),
        role=ROLE_TREATMENT_HERITAGE,
        seed=kw.get("seed", seed()),
        data_source=kw.get("data_source", heritage_source()),
        budget=kw.get("budget", budget()),
    )


class EvidenceRefTests(unittest.TestCase):
    def test_unknown_kind_rejected(self):
        with self.assertRaises(ExperimentError):
            EvidenceRef(
                kind="vibes",
                record_id="somewhere",
                subject_sha256=SEED_A,
                verified_at="2026-09-24T08:40:00Z",
            )

    def test_roundtrip_requires_both_hashes(self):
        with self.assertRaises(ExperimentError):
            EvidenceRef(
                kind=EVIDENCE_SERIALIZATION_ROUNDTRIP,
                record_id="out.nnue",
                subject_sha256=SEED_A,
                verified_at="2026-09-24T08:40:00Z",
            )

    def test_record_id_required(self):
        with self.assertRaises(ExperimentError):
            creation_evidence().__class__(
                kind=EVIDENCE_CREATION_LOG,
                record_id="   ",
                subject_sha256=SEED_A,
                verified_at="2026-09-24T08:40:00Z",
            )

    def test_matching_hashes_corroborate(self):
        self.assertTrue(roundtrip_evidence().corroborates)

    def test_mismatched_hashes_do_not_corroborate(self):
        """A failed roundtrip is recordable but establishes nothing."""

        evidence = roundtrip_evidence(result="1" * 64, expected="2" * 64)

        self.assertFalse(evidence.corroborates)
        self.assertEqual(evidence.establishes, frozenset())

    def test_roundtrip_cannot_establish_historical_origin(self):
        """The interpretation rule, enforced structurally rather than by prose."""

        self.assertEqual(
            roundtrip_evidence().establishes,
            frozenset({CLAIM_SERIALIZATION_EQUIVALENCE}),
        )
        self.assertNotIn(
            CLAIM_HISTORICAL_ORIGIN,
            roundtrip_evidence().establishes,
        )


class SeedSpecTests(unittest.TestCase):
    def test_defaults_establish_nothing(self):
        self.assertEqual(seed().established_claims, frozenset())

    def test_roundtrip_evidence_establishes_only_equivalence(self):
        s = seed(evidence=(roundtrip_evidence(),))

        self.assertTrue(s.establishes(CLAIM_SERIALIZATION_EQUIVALENCE))
        self.assertFalse(s.establishes(CLAIM_HISTORICAL_ORIGIN))
        self.assertFalse(s.establishes(CLAIM_TRAINABILITY))

    def test_creation_log_establishes_origin(self):
        s = seed(evidence=(creation_evidence(),))

        self.assertTrue(s.establishes(CLAIM_HISTORICAL_ORIGIN))

    def test_evidence_about_another_artifact_is_ignored(self):
        """Evidence cannot be transplanted from one checkpoint to another."""

        s = seed(sha=SEED_A, evidence=(roundtrip_evidence(subject=SEED_B),))

        self.assertEqual(s.bound_evidence, ())
        self.assertFalse(s.establishes(CLAIM_SERIALIZATION_EQUIVALENCE))

    def test_failed_roundtrip_establishes_nothing(self):
        s = seed(evidence=(roundtrip_evidence(result="1" * 64, expected="2" * 64),))

        self.assertFalse(s.establishes(CLAIM_SERIALIZATION_EQUIVALENCE))

    def test_multiple_evidence_accumulates(self):
        s = seed(
            evidence=(
                roundtrip_evidence(),
                creation_evidence(),
                training_evidence(),
            )
        )

        self.assertEqual(
            s.established_claims,
            frozenset(
                {
                    CLAIM_SERIALIZATION_EQUIVALENCE,
                    CLAIM_HISTORICAL_ORIGIN,
                    CLAIM_TRAINABILITY,
                }
            ),
        )

    def test_unknown_claim_rejected(self):
        with self.assertRaises(ExperimentError):
            seed().establishes("wishful_thinking")

    def test_non_evidence_entry_rejected(self):
        with self.assertRaises(ExperimentError):
            seed(evidence=("just a string",))

    def test_bad_hash_rejected(self):
        with self.assertRaises(ExperimentError):
            SeedSpec(artifact_id="X", sha256="not-a-hash")

    def test_uppercase_hash_rejected(self):
        with self.assertRaises(ExperimentError):
            SeedSpec(artifact_id="X", sha256="A" * 64)


class DataSourceTests(unittest.TestCase):
    def test_default_status_is_unknown(self):
        ds = DataSourceSpec(source_id="X", sha256=SRC_A)
        self.assertEqual(ds.heritage_status, HERITAGE_UNKNOWN)
        self.assertFalse(ds.is_heritage_free_declared)
        self.assertFalse(ds.is_heritage_free_established)

    def test_heritage_free_requires_justification(self):
        with self.assertRaises(ExperimentError):
            DataSourceSpec(
                source_id="X",
                sha256=SRC_A,
                heritage_status=HERITAGE_FREE_DECLARED,
            )

    def test_unknown_status_needs_no_justification(self):
        DataSourceSpec(source_id="X", sha256=SRC_A)

    def test_invalid_status_rejected(self):
        with self.assertRaises(ExperimentError):
            DataSourceSpec(
                source_id="X",
                sha256=SRC_A,
                heritage_status="probably_fine",
            )


class ArmTests(unittest.TestCase):
    def test_control_must_use_heritage_free_corpus(self):
        with self.assertRaises(ExperimentError) as ctx:
            ArmSpec(
                arm_id="M0",
                role=ROLE_CONTROL_M0,
                seed=seed(),
                data_source=heritage_source(),
                budget=budget(),
            )

        self.assertIn("heritage", str(ctx.exception))

    def test_control_with_unknown_corpus_rejected(self):
        with self.assertRaises(ExperimentError):
            ArmSpec(
                arm_id="M0",
                role=ROLE_CONTROL_M0,
                seed=seed(),
                data_source=DataSourceSpec(source_id="X", sha256=SRC_A),
                budget=budget(),
            )

    def test_treatment_may_use_heritage_corpus(self):
        treatment_arm()


class ExperimentTests(unittest.TestCase):
    def test_valid_experiment(self):
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[control_arm(), treatment_arm()],
        )

        self.assertEqual(exp.control.arm_id, "M0")
        self.assertEqual(len(exp.treatments), 1)

    def test_divergent_seeds_rejected(self):
        """The exact failure mode of the existing V80 vs official pairing."""

        with self.assertRaises(ExperimentError) as ctx:
            build_experiment(
                experiment_id="EXP1",
                arms=[
                    control_arm(),
                    treatment_arm(seed=seed(sha=SEED_B)),
                ],
            )

        self.assertIn("byte-identical seed", str(ctx.exception))

    def test_divergent_budgets_rejected(self):
        with self.assertRaises(ExperimentError) as ctx:
            build_experiment(
                experiment_id="EXP1",
                arms=[
                    control_arm(),
                    treatment_arm(budget=budget(lr=1e-5)),
                ],
            )

        self.assertIn("identical TrainingBudget", str(ctx.exception))

    def test_requires_exactly_one_control(self):
        with self.assertRaises(ExperimentError):
            build_experiment(
                experiment_id="EXP1",
                arms=[control_arm(), control_arm(arm_id="M0B"), treatment_arm()],
            )

    def test_requires_at_least_one_treatment(self):
        with self.assertRaises(ExperimentError):
            build_experiment(experiment_id="EXP1", arms=[control_arm()])

    def test_duplicate_arm_ids_rejected(self):
        with self.assertRaises(ExperimentError):
            build_experiment(
                experiment_id="EXP1",
                arms=[
                    control_arm(arm_id="SAME"),
                    treatment_arm(arm_id="SAME"),
                ],
            )

    def test_identity_is_deterministic_and_order_independent(self):
        a = build_experiment(
            experiment_id="EXP1",
            arms=[control_arm(), treatment_arm()],
        )
        b = build_experiment(
            experiment_id="EXP1",
            arms=[treatment_arm(), control_arm()],
        )

        self.assertEqual(a.identity_sha256, b.identity_sha256)

    def test_identity_changes_with_evidence(self):
        base = build_experiment(
            experiment_id="EXP1",
            arms=[control_arm(), treatment_arm()],
        )

        evidenced = seed(evidence=(roundtrip_evidence(),))

        other = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(seed=evidenced),
                treatment_arm(seed=evidenced),
            ],
        )

        self.assertNotEqual(base.identity_sha256, other.identity_sha256)

    def test_shared_seed_accessor(self):
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[control_arm(), treatment_arm()],
        )

        self.assertEqual(exp.shared_seed.sha256, SEED_A)


if __name__ == "__main__":
    unittest.main()
