"""Tests for the fail-closed preflight authorization gate."""

import unittest

from medium_pc_audit.experiment import (
    EVIDENCE_ARTIFACT_HASH_OBSERVATION,
    EVIDENCE_CORPUS_PROVENANCE_AUDIT,
    HERITAGE_FREE_DECLARED,
    ROLE_TREATMENT_HERITAGE,
    ArmSpec,
    DataSourceSpec,
    EvidenceRef,
    build_experiment,
)
from medium_pc_audit.opening_suite import load_suite_from_text
from medium_pc_audit.preflight import (
    BLOCKED,
    NOT_YET_IMPLEMENTED,
    READY,
    USER_DECISION,
    NotAuthorized,
    PreflightError,
    evaluate_preflight,
    require_authorized,
)
from medium_pc_audit.split import build_split
from tests.test_experiment import (
    SEED_A,
    budget,
    control_arm,
    creation_evidence,
    roundtrip_evidence,
    seed,
    training_evidence,
    treatment_arm,
)

SHA = "f" * 64


def proven_seed():
    """A seed with corroborating evidence for all three claims."""

    return seed(
        evidence=(
            roundtrip_evidence(),
            creation_evidence(),
            training_evidence(),
        )
    )


def roundtrip_only_seed():
    """Tonight's actual evidence state: equivalence shown, origin not."""

    return seed(evidence=(roundtrip_evidence(),))


def evidenced_free_source(
    source_id="MODERN_POOL",
    sha=SHA,
    evidence_subject=None,
):
    """A declared corpus whose own bytes and heritage-freedom are evidenced."""

    subject = sha if evidence_subject is None else evidence_subject

    return DataSourceSpec(
        source_id=source_id,
        sha256=sha,
        heritage_status=HERITAGE_FREE_DECLARED,
        justification="synthetic corpus, no OLA/GEN2 curation",
        evidence=(
            EvidenceRef(
                kind=EVIDENCE_ARTIFACT_HASH_OBSERVATION,
                record_id="modern_pool.sha256",
                subject_sha256=subject,
                verified_at="2026-09-24T10:00:00Z",
                result_sha256=sha,
                expected_sha256=sha,
                detail="observed corpus bytes",
            ),
            EvidenceRef(
                kind=EVIDENCE_CORPUS_PROVENANCE_AUDIT,
                record_id="modern_pool_provenance_audit.json",
                subject_sha256=subject,
                verified_at="2026-09-24T10:01:00Z",
                detail="corpus audited as heritage-free",
            ),
        ),
    )

def experiment(seed_spec=None):
    s = seed_spec or seed()
    return build_experiment(
        experiment_id="EXP1",
        arms=[control_arm(seed=s), treatment_arm(seed=s)],
    )


def good_suite(n=50):
    suite, _ = load_suite_from_text(
        "\n".join(f"board{i} w KQkq -" for i in range(n)),
        suite_id="UHO",
        sha256=SHA,
        unbalanced_declared=True,
        justification="selected in a deliberately imbalanced evaluation band",
    )
    return suite


def isolated_split():
    rows = [{"fen": f"b{i} w KQkq - 0 1", "g": f"g{i // 2}"} for i in range(20)]
    return build_split(rows, seed="S", family_of=lambda r: r["g"])


def position_split():
    rows = [{"fen": f"b{i} w KQkq - 0 1"} for i in range(20)]
    return build_split(rows, seed="S", allow_position_level_split_unsafe=True)


def statuses(report):
    return {p.code: p.status for p in report.preconditions}


class DefaultPostureTests(unittest.TestCase):
    """With no evidence supplied, everything that can block must block."""

    def test_bare_experiment_is_not_authorized(self):
        report = evaluate_preflight(experiment())

        self.assertFalse(report.authorized)
        self.assertTrue(report.blockers)

    def test_missing_evidence_blocks_each_gate(self):
        s = statuses(evaluate_preflight(experiment()))

        self.assertEqual(s["seed_serialization_equivalence"], BLOCKED)
        self.assertEqual(s["seed_historical_provenance"], BLOCKED)
        self.assertEqual(s["seed_trainability_demonstrated"], NOT_YET_IMPLEMENTED)
        self.assertEqual(s["heritage_split_family_isolated"], BLOCKED)
        self.assertEqual(s["opening_suite_pinned"], BLOCKED)
        self.assertEqual(s["opening_suite_unbalanced"], BLOCKED)
        self.assertEqual(s["opening_suite_capacity"], BLOCKED)

    def test_shared_seed_and_budget_pass_by_construction(self):
        s = statuses(evaluate_preflight(experiment()))

        self.assertEqual(s["shared_seed_across_arms"], READY)
        self.assertEqual(s["budget_parity"], READY)

    def test_report_identity_binds_to_experiment(self):
        exp = experiment()
        report = evaluate_preflight(exp)

        self.assertEqual(report.experiment_identity_sha256, exp.identity_sha256)


class SeedGateTests(unittest.TestCase):
    def test_fully_evidenced_seed_is_ready(self):
        s = statuses(evaluate_preflight(experiment(proven_seed())))

        self.assertEqual(s["seed_serialization_equivalence"], READY)
        self.assertEqual(s["seed_historical_provenance"], READY)
        self.assertEqual(s["seed_trainability_demonstrated"], READY)

    def test_hash_alone_does_not_satisfy_provenance(self):
        """A stable SHA256 establishes identity, not origin."""

        s = statuses(evaluate_preflight(experiment(seed(sha=SEED_A))))

        self.assertEqual(s["seed_historical_provenance"], BLOCKED)

    def test_roundtrip_alone_does_not_satisfy_historical_provenance(self):
        """Tonight's result: equivalence is READY, origin stays BLOCKED.

        This is the interpretation rule as an executable assertion -- a
        roundtrip shows what a checkpoint decodes to, never where it came
        from, and no amount of it can move the provenance gate.
        """

        s = statuses(evaluate_preflight(experiment(roundtrip_only_seed())))

        self.assertEqual(s["seed_serialization_equivalence"], READY)
        self.assertEqual(s["seed_historical_provenance"], BLOCKED)

    def test_roundtrip_ready_detail_cites_its_record(self):
        report = evaluate_preflight(experiment(roundtrip_only_seed()))

        detail = next(
            p.detail
            for p in report.preconditions
            if p.code == "seed_serialization_equivalence"
        )

        self.assertIn("serialization_roundtrip", detail)
        self.assertIn("c54_seed_roundtrip_verify.nnue", detail)


class DataGateTests(unittest.TestCase):
    def test_unknown_corpus_raises_user_decision(self):
        s = seed()
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(seed=s),
                ArmSpec(
                    arm_id="TREAT",
                    role=ROLE_TREATMENT_HERITAGE,
                    seed=s,
                    data_source=DataSourceSpec(source_id="MYSTERY", sha256=SHA),
                    budget=budget(),
                ),
            ],
        )

        self.assertEqual(
            statuses(evaluate_preflight(exp))["all_corpora_classified"],
            USER_DECISION,
        )

    def test_classified_corpora_do_not_establish_control_heritage_freedom(self):
        s = statuses(evaluate_preflight(experiment()))

        self.assertEqual(s["all_corpora_classified"], READY)
        self.assertEqual(s["control_corpus_heritage_free"], BLOCKED)

    def test_evidenced_control_corpus_is_ready(self):
        s = seed()
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(
                    seed=s,
                    data_source=evidenced_free_source(),
                ),
                treatment_arm(seed=s),
            ],
        )

        self.assertEqual(
            statuses(evaluate_preflight(exp))["control_corpus_heritage_free"],
            READY,
        )

    def test_control_evidence_for_another_corpus_does_not_satisfy_gate(self):
        s = seed()
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(
                    seed=s,
                    data_source=evidenced_free_source(
                        evidence_subject="e" * 64,
                    ),
                ),
                treatment_arm(seed=s),
            ],
        )

        self.assertEqual(
            statuses(evaluate_preflight(exp))["control_corpus_heritage_free"],
            BLOCKED,
        )


class SplitGateTests(unittest.TestCase):
    def test_family_isolated_split_is_ready(self):
        report = evaluate_preflight(experiment(), split_result=isolated_split())

        self.assertEqual(statuses(report)["heritage_split_family_isolated"], READY)

    def test_position_level_split_is_blocked(self):
        report = evaluate_preflight(experiment(), split_result=position_split())

        self.assertEqual(statuses(report)["heritage_split_family_isolated"], BLOCKED)

    def test_wrong_type_rejected(self):
        with self.assertRaises(PreflightError):
            evaluate_preflight(experiment(), split_result="not-a-split")


class OpeningGateTests(unittest.TestCase):
    def test_declared_suite_with_capacity_is_ready(self):
        report = evaluate_preflight(
            experiment(),
            opening_suite=good_suite(50),
            pairs=20,
        )
        s = statuses(report)

        self.assertEqual(s["opening_suite_pinned"], READY)
        self.assertEqual(s["opening_suite_unbalanced"], READY)
        self.assertEqual(s["opening_suite_capacity"], READY)

    def test_undeclared_suite_blocks(self):
        suite, _ = load_suite_from_text(
            "\n".join(f"b{i} w KQkq -" for i in range(20)),
            suite_id="BALANCED",
            sha256=SHA,
        )

        s = statuses(evaluate_preflight(experiment(), opening_suite=suite, pairs=5))

        self.assertEqual(s["opening_suite_unbalanced"], BLOCKED)

    def test_insufficient_capacity_blocks(self):
        s = statuses(
            evaluate_preflight(experiment(), opening_suite=good_suite(5), pairs=9)
        )

        self.assertEqual(s["opening_suite_capacity"], BLOCKED)

    def test_missing_pairs_blocks_capacity(self):
        s = statuses(evaluate_preflight(experiment(), opening_suite=good_suite()))

        self.assertEqual(s["opening_suite_capacity"], BLOCKED)

    def test_invalid_pairs_rejected(self):
        with self.assertRaises(PreflightError):
            evaluate_preflight(experiment(), opening_suite=good_suite(), pairs=0)


class FullAuthorizationTests(unittest.TestCase):
    def fully_satisfied(self):
        s = proven_seed()
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(
                    seed=s,
                    data_source=evidenced_free_source(),
                ),
                treatment_arm(seed=s),
            ],
        )

        return evaluate_preflight(
            exp,
            opening_suite=good_suite(50),
            split_result=isolated_split(),
            pairs=20,
        )

    def test_everything_supplied_authorizes(self):
        report = self.fully_satisfied()

        self.assertTrue(report.authorized, report.as_dict()["blockers"])
        self.assertEqual(report.blockers, ())

    def test_require_authorized_passes(self):
        require_authorized(self.fully_satisfied())

    def test_require_authorized_raises_when_blocked(self):
        with self.assertRaises(NotAuthorized) as ctx:
            require_authorized(evaluate_preflight(experiment()))

        self.assertIn("not authorized", str(ctx.exception))

    def test_one_missing_gate_withholds_authorization(self):
        """Authorization is a conjunction: a single gap is enough to stop it."""

        s = proven_seed()
        exp = build_experiment(
            experiment_id="EXP1",
            arms=[
                control_arm(
                    seed=s,
                    data_source=evidenced_free_source(),
                ),
                treatment_arm(seed=s),
            ],
        )

        report = evaluate_preflight(
            exp,
            opening_suite=good_suite(50),
            split_result=position_split(),
            pairs=20,
        )

        self.assertFalse(report.authorized)
        self.assertEqual(
            [b.code for b in report.blockers],
            ["heritage_split_family_isolated"],
        )

    def test_as_dict_round_trip(self):
        payload = self.fully_satisfied().as_dict()

        self.assertTrue(payload["authorized"])
        self.assertEqual(payload["blockers"], [])
        self.assertTrue(payload["preconditions"])

    def test_by_status_filter(self):
        report = evaluate_preflight(experiment())

        self.assertTrue(report.by_status(BLOCKED))

        with self.assertRaises(PreflightError):
            report.by_status("NONSENSE")


class InputValidationTests(unittest.TestCase):
    def test_non_experiment_rejected(self):
        with self.assertRaises(PreflightError):
            evaluate_preflight("not-an-experiment")

    def test_non_suite_rejected(self):
        with self.assertRaises(PreflightError):
            evaluate_preflight(experiment(), opening_suite="nope", pairs=2)

    def test_require_authorized_type_checked(self):
        with self.assertRaises(PreflightError):
            require_authorized("not-a-report")


if __name__ == "__main__":
    unittest.main()
