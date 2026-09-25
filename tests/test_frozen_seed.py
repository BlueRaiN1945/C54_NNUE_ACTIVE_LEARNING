"""Tests for the frozen common seed and the acknowledged-unknown waiver."""

import unittest

from medium_pc_audit.experiment import (
    CLAIM_HISTORICAL_ORIGIN,
    CLAIM_SERIALIZATION_EQUIVALENCE,
    CLAIM_TRAINABILITY,
    ExperimentError,
    build_experiment,
)
from medium_pc_audit.frozen_seed import (
    ACCEPTED_UNKNOWNS,
    OFFICIAL_NNUE_SHA256,
    SEED_SHA256,
    frozen_seed,
    serialization_evidence,
    trainability_evidence,
)
from medium_pc_audit.preflight import (
    ACKNOWLEDGED_UNKNOWN,
    BLOCKED,
    READY,
    Precondition,
    PreflightError,
    evaluate_preflight,
    require_authorized,
)
from tests.test_experiment import control_arm, treatment_arm
from tests.test_preflight import evidenced_free_source, good_suite, isolated_split


def statuses(report):
    return {p.code: p.status for p in report.preconditions}


def experiment_on_frozen_seed(accepted_unknowns=()):
    s = frozen_seed()

    return build_experiment(
        experiment_id="C54HERITAGE1",
        arms=[
            control_arm(
                seed=s,
                data_source=evidenced_free_source(),
            ),
            treatment_arm(seed=s),
        ],
        accepted_unknowns=accepted_unknowns,
    )


def full_report(accepted_unknowns=()):
    return evaluate_preflight(
        experiment_on_frozen_seed(accepted_unknowns),
        opening_suite=good_suite(50),
        split_result=isolated_split(),
        pairs=20,
    )


class FrozenSeedEvidenceTests(unittest.TestCase):
    def test_roundtrip_evidence_corroborates(self):
        evidence = serialization_evidence()

        self.assertTrue(evidence.corroborates)
        self.assertEqual(evidence.result_sha256, OFFICIAL_NNUE_SHA256)
        self.assertEqual(evidence.expected_sha256, OFFICIAL_NNUE_SHA256)

    def test_all_evidence_is_bound_to_the_seed_hash(self):
        seed = frozen_seed()

        self.assertEqual(len(seed.bound_evidence), 2)

        for item in seed.evidence:
            self.assertEqual(item.subject_sha256, SEED_SHA256)

    def test_establishes_equivalence_and_trainability(self):
        seed = frozen_seed()

        self.assertTrue(seed.establishes(CLAIM_SERIALIZATION_EQUIVALENCE))
        self.assertTrue(seed.establishes(CLAIM_TRAINABILITY))

    def test_does_not_establish_historical_origin(self):
        """Neither verification speaks to where the checkpoint came from."""

        self.assertFalse(frozen_seed().establishes(CLAIM_HISTORICAL_ORIGIN))

    def test_trainability_evidence_records_no_strength_claim(self):
        detail = trainability_evidence().detail

        self.assertIn("no strength", detail.lower())


class WaiverContractTests(unittest.TestCase):
    def test_only_historical_origin_is_waivable(self):
        self.assertEqual(ACCEPTED_UNKNOWNS, (CLAIM_HISTORICAL_ORIGIN,))

    def test_technical_claims_cannot_be_waived(self):
        """Waiving a testable fact would mean declining to run a cheap check."""

        for claim in (CLAIM_SERIALIZATION_EQUIVALENCE, CLAIM_TRAINABILITY):
            with self.assertRaises(ExperimentError) as ctx:
                experiment_on_frozen_seed(accepted_unknowns=(claim,))

            self.assertIn("not waivable", str(ctx.exception))

    def test_unknown_claim_rejected(self):
        with self.assertRaises(ExperimentError):
            experiment_on_frozen_seed(accepted_unknowns=("wishful_thinking",))

    def test_duplicate_waivers_rejected(self):
        with self.assertRaises(ExperimentError):
            experiment_on_frozen_seed(
                accepted_unknowns=(
                    CLAIM_HISTORICAL_ORIGIN,
                    CLAIM_HISTORICAL_ORIGIN,
                )
            )

    def test_waiver_changes_experiment_identity(self):
        """An experiment that waived something is never confusable with one
        that satisfied it."""

        without = experiment_on_frozen_seed()
        with_waiver = experiment_on_frozen_seed(ACCEPTED_UNKNOWNS)

        self.assertNotEqual(
            without.identity_sha256,
            with_waiver.identity_sha256,
        )

    def test_non_blocking_status_must_be_acknowledged(self):
        """Nothing else may quietly declare itself non-blocking."""

        with self.assertRaises(PreflightError):
            Precondition(
                code="sneaky",
                status=BLOCKED,
                detail="",
                blocking=False,
            )


class FrozenSeedPreflightTests(unittest.TestCase):
    def test_without_waiver_provenance_blocks(self):
        report = full_report()

        self.assertEqual(statuses(report)["seed_historical_provenance"], BLOCKED)
        self.assertFalse(report.authorized)

    def test_blocked_detail_explains_the_remedy(self):
        report = full_report()

        detail = next(
            p.detail
            for p in report.preconditions
            if p.code == "seed_historical_provenance"
        )

        self.assertIn("accepted_unknowns", detail)

    def test_with_waiver_provenance_is_acknowledged_not_ready(self):
        report = full_report(ACCEPTED_UNKNOWNS)
        s = statuses(report)

        self.assertEqual(s["seed_historical_provenance"], ACKNOWLEDGED_UNKNOWN)
        self.assertNotEqual(s["seed_historical_provenance"], READY)

    def test_with_waiver_the_experiment_is_authorized(self):
        report = full_report(ACCEPTED_UNKNOWNS)

        self.assertTrue(report.authorized, report.as_dict()["blockers"])
        require_authorized(report)

    def test_acknowledged_gap_remains_visible(self):
        """A waiver records the gap; it never deletes it."""

        report = full_report(ACCEPTED_UNKNOWNS)

        self.assertEqual(len(report.acknowledged), 1)
        self.assertEqual(
            report.acknowledged[0].code,
            "seed_historical_provenance",
        )
        self.assertIn("UNKNOWN", report.acknowledged[0].detail)
        self.assertIn("acknowledged", report.as_dict())

    def test_technical_gates_are_ready_from_real_evidence(self):
        s = statuses(full_report(ACCEPTED_UNKNOWNS))

        self.assertEqual(s["seed_serialization_equivalence"], READY)
        self.assertEqual(s["seed_trainability_demonstrated"], READY)
        self.assertEqual(s["shared_seed_across_arms"], READY)

    def test_waiver_does_not_excuse_other_blockers(self):
        """The waiver is scoped to provenance and nothing else."""

        report = evaluate_preflight(
            experiment_on_frozen_seed(ACCEPTED_UNKNOWNS),
            opening_suite=None,
            split_result=isolated_split(),
            pairs=20,
        )

        self.assertFalse(report.authorized)
        self.assertIn(
            "opening_suite_pinned",
            [b.code for b in report.blockers],
        )


if __name__ == "__main__":
    unittest.main()
