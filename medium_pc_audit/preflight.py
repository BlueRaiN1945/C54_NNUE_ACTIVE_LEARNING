"""Fail-closed preflight guards for experiment authorization.

Stdlib-only and deterministic: no filesystem, network, engine, training, or
wall-clock access.

This module turns the Phase 1.5 precondition checklist into executable gates.
Its governing rule is **fail closed**: a precondition is READY only when
positive evidence is supplied. Absent, unknown, or unverifiable evidence
yields BLOCKED -- never READY, and never a warning that execution proceeds
past. `authorized` is the conjunction of every gate, so a single non-READY
precondition withholds authorization.

Two properties are re-verified here even though the constructors in
medium_pc_audit.experiment already enforce them (shared seed, budget parity).
That duplication is deliberate and matches this repository's established
practice -- config_gen.publish_config() likewise re-validates and re-derives
everything rather than trusting that an object was built by the blessed path.
A caller can hand-construct an ExperimentSpec-shaped object; the gate must
not assume it did not.

This module decides nothing about the world. It reports on the claims it is
given. Setting `provenance_documented=True` on a seed that has no production
record will pass the gate -- and that is a falsified input, not a gate
failure. The gate's job is to make the claim explicit and attributable.
"""

from __future__ import annotations

from dataclasses import dataclass

from medium_pc_audit.experiment import (
    CLAIM_HISTORICAL_ORIGIN,
    CLAIM_SERIALIZATION_EQUIVALENCE,
    CLAIM_TRAINABILITY,
    HERITAGE_FREE_DECLARED,
    HERITAGE_UNKNOWN,
    ExperimentSpec,
)
from medium_pc_audit.identity import compute_identity_sha256
from medium_pc_audit.opening_suite import OpeningSuite
from medium_pc_audit.split import SplitResult

READY = "READY"
BLOCKED = "BLOCKED"
USER_DECISION = "USER-DECISION"
NOT_YET_IMPLEMENTED = "NOT-YET-IMPLEMENTED"

# A gap that is real, unresolved, and explicitly accepted by a recorded
# decision. It is never a silent pass: the finding stays in the report, and the
# acceptance is bound into the experiment's identity hash, so an experiment
# that waived something can never be confused with one that satisfied it.
ACKNOWLEDGED_UNKNOWN = "ACKNOWLEDGED-UNKNOWN"

STATUSES = frozenset(
    {
        READY,
        BLOCKED,
        USER_DECISION,
        NOT_YET_IMPLEMENTED,
        ACKNOWLEDGED_UNKNOWN,
    }
)

# -- readiness profiles -------------------------------------------------------
#
# Three different questions, deliberately not collapsed into one:
#
#   TECHNICAL_PILOT -- does the pipeline work end to end? A tiny non-scientific
#       fixture corpus is fine, and family/line splitting is irrelevant,
#       because nothing is being concluded about chess. A real pinned opening
#       artifact IS still required: the pairing/pentanomial machinery is
#       exactly what a technical pilot exists to exercise.
#
#   M0 -- is this a valid control arm? Now the corpus must genuinely be a
#       verified heritage-free modern corpus, because that property IS the
#       control.
#
#   HERITAGE -- is this a valid heritage-transfer experiment? Everything M0
#       requires, plus leakage-safe family/line splitting, because a heritage
#       claim contaminated by split leakage is not a claim.
PROFILE_TECHNICAL_PILOT = "TECHNICAL_PILOT"
PROFILE_M0 = "M0"
PROFILE_HERITAGE = "HERITAGE"

PROFILES = (
    PROFILE_TECHNICAL_PILOT,
    PROFILE_M0,
    PROFILE_HERITAGE,
)

_CORPUS_GATES = (
    "control_corpus_identified",
    "control_corpus_hash_verified",
    "control_corpus_heritage_free",
)

_SPLIT_GATES = ("heritage_split_family_isolated",)

# Gates NOT required by each profile, with the reason recorded on the gate.
_NOT_REQUIRED_BY_PROFILE = {
    PROFILE_TECHNICAL_PILOT: {
        **{
            gate: (
                "a technical pilot proves the pipeline, not a scientific "
                "result; a tiny non-scientific fixture corpus is sufficient "
                "and no heritage-free property is being relied upon"
            )
            for gate in _CORPUS_GATES
        },
        **{
            gate: (
                "a technical pilot draws no conclusion from the data, so "
                "split leakage cannot contaminate anything"
            )
            for gate in _SPLIT_GATES
        },
    },
    PROFILE_M0: {
        gate: (
            "M0 validity rests on the corpus being heritage-free, not on how "
            "heritage material is split; family isolation gates the heritage "
            "treatment arm instead"
        )
        for gate in _SPLIT_GATES
    },
    PROFILE_HERITAGE: {},
}


class PreflightError(ValueError):
    """Raised when preflight input is malformed."""


class NotAuthorized(PreflightError):
    """Raised when execution is attempted without a fully READY preflight."""


@dataclass(frozen=True)
class Precondition:
    code: str
    status: str
    detail: str
    blocking: bool = True

    def __post_init__(self):
        if not isinstance(self.code, str) or not self.code:
            raise PreflightError("precondition code must be a non-empty string")

        if self.status not in STATUSES:
            raise PreflightError(
                f"status must be one of {sorted(STATUSES)}, got {self.status!r}"
            )

        if not isinstance(self.blocking, bool):
            raise PreflightError("precondition blocking must be a bool")

        # Only an explicitly acknowledged unknown may be non-blocking. Any
        # other non-READY status withholding its block would be a silent
        # weakening of the gate.
        if not self.blocking and self.status not in {READY, ACKNOWLEDGED_UNKNOWN}:
            raise PreflightError(
                f"precondition {self.code!r} with status {self.status!r} cannot "
                "be non-blocking; only an acknowledged unknown may be waived"
            )

    @property
    def is_ready(self) -> bool:
        return self.status == READY

    @property
    def withholds_authorization(self) -> bool:
        return self.blocking and not self.is_ready

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "status": self.status,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass(frozen=True)
class PreflightReport:
    experiment_id: str
    experiment_identity_sha256: str
    preconditions: tuple

    @property
    def authorized(self) -> bool:
        """True only when no blocking precondition is outstanding.

        Acknowledged unknowns do not withhold authorization, but they remain
        in `preconditions` and in `acknowledged` -- a waiver is recorded, not
        erased.
        """
        return not any(p.withholds_authorization for p in self.preconditions)

    @property
    def blockers(self) -> tuple:
        return tuple(p for p in self.preconditions if p.withholds_authorization)

    @property
    def acknowledged(self) -> tuple:
        """Real gaps the experiment is proceeding despite, by explicit decision."""

        return tuple(
            p for p in self.preconditions if p.status == ACKNOWLEDGED_UNKNOWN
        )

    def by_status(self, status: str) -> tuple:
        if status not in STATUSES:
            raise PreflightError(f"unknown status {status!r}")

        return tuple(p for p in self.preconditions if p.status == status)

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "experiment_identity_sha256": self.experiment_identity_sha256,
            "authorized": self.authorized,
            "preconditions": [p.as_dict() for p in self.preconditions],
            "blockers": [p.as_dict() for p in self.blockers],
            "acknowledged": [p.as_dict() for p in self.acknowledged],
        }


def _cite(seed, claim) -> str:
    """Name the records establishing a claim, so a READY stays auditable."""

    return ", ".join(
        f"{e.kind}:{e.record_id}"
        for e in seed.bound_evidence
        if claim in e.establishes
    )


def _seed_preconditions(experiment) -> list:
    seed = experiment.shared_seed
    out = []

    # Serialization equivalence and historical origin are separate gates
    # because they are separate facts. A roundtrip can satisfy the first and
    # is structurally incapable of satisfying the second.
    if seed.establishes(CLAIM_SERIALIZATION_EQUIVALENCE):
        out.append(
            Precondition(
                code="seed_serialization_equivalence",
                status=READY,
                detail=(
                    f"seed {seed.artifact_id!r} ({seed.sha256[:12]}...) verified "
                    "serialization-equivalent to its pinned network via "
                    f"{_cite(seed, CLAIM_SERIALIZATION_EQUIVALENCE)}"
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="seed_serialization_equivalence",
                status=BLOCKED,
                detail=(
                    f"seed {seed.artifact_id!r} ({seed.sha256[:12]}...) has no "
                    "corroborating serialization evidence bound to this hash"
                ),
            )
        )

    if seed.establishes(CLAIM_HISTORICAL_ORIGIN):
        out.append(
            Precondition(
                code="seed_historical_provenance",
                status=READY,
                detail=(
                    f"seed {seed.artifact_id!r} origin documented by "
                    f"{_cite(seed, CLAIM_HISTORICAL_ORIGIN)}"
                ),
            )
        )
    elif CLAIM_HISTORICAL_ORIGIN in experiment.accepted_unknowns:
        out.append(
            Precondition(
                code="seed_historical_provenance",
                status=ACKNOWLEDGED_UNKNOWN,
                blocking=False,
                detail=(
                    f"seed {seed.artifact_id!r} ({seed.sha256[:12]}...) has no "
                    "creation or import record, and this experiment explicitly "
                    "accepts that gap. Origin remains UNKNOWN and is recorded "
                    "here rather than resolved: results carry it as a stated "
                    "limitation, and this acceptance is bound into the "
                    f"experiment identity ({experiment.identity_sha256[:12]}...)."
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="seed_historical_provenance",
                status=BLOCKED,
                detail=(
                    f"seed {seed.artifact_id!r} has no creation or import record. "
                    "Serialization equivalence does not establish origin: it "
                    "shows what the checkpoint decodes to, not where it came "
                    "from. A stable hash establishes identity, not history. "
                    f"To proceed regardless, declare {CLAIM_HISTORICAL_ORIGIN!r} "
                    "in the experiment's accepted_unknowns."
                ),
            )
        )

    if seed.establishes(CLAIM_TRAINABILITY):
        out.append(
            Precondition(
                code="seed_trainability_demonstrated",
                status=READY,
                detail=(
                    f"seed {seed.artifact_id!r} demonstrated trainable by "
                    f"{_cite(seed, CLAIM_TRAINABILITY)}"
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="seed_trainability_demonstrated",
                status=NOT_YET_IMPLEMENTED,
                detail=(
                    f"seed {seed.artifact_id!r} has never been demonstrated as a "
                    "trainable starting checkpoint; format compatibility is not "
                    "the same as a completed training step"
                ),
            )
        )

    # Re-verified rather than trusted: a hand-constructed spec can bypass the
    # constructor that normally guarantees this.
    seed_hashes = {arm.seed.sha256 for arm in experiment.arms}

    if len(seed_hashes) == 1:
        out.append(
            Precondition(
                code="shared_seed_across_arms",
                status=READY,
                detail=(
                    f"all {len(experiment.arms)} arms share seed "
                    f"{seed.sha256[:12]}..."
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="shared_seed_across_arms",
                status=BLOCKED,
                detail=(
                    f"{len(seed_hashes)} distinct seed hashes across arms; a "
                    "strength difference would confound the data under test "
                    "with the differing starting weights"
                ),
            )
        )

    return out


def _data_preconditions(experiment) -> list:
    out = []
    control = experiment.control

    if control.data_source.is_heritage_free_established:
        out.append(
            Precondition(
                code="control_corpus_heritage_free",
                status=READY,
                detail=(
                    f"control corpus {control.data_source.source_id!r} has "
                    "hash-bound evidence establishing heritage-freedom"
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="control_corpus_heritage_free",
                status=BLOCKED,
                detail=(
                    f"control corpus {control.data_source.source_id!r} does not "
                    "have hash-bound evidence establishing heritage-freedom; "
                    "an M0 control requires evidence, not declaration alone"
                ),
            )
        )

    unknown = [
        arm.arm_id
        for arm in experiment.arms
        if arm.data_source.heritage_status == HERITAGE_UNKNOWN
    ]

    if unknown:
        out.append(
            Precondition(
                code="all_corpora_classified",
                status=USER_DECISION,
                detail=(
                    f"arms {sorted(unknown)} carry an unclassified corpus; each "
                    "must be declared heritage-curated or heritage-free before "
                    "its results can be interpreted"
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="all_corpora_classified",
                status=READY,
                detail="every arm's corpus carries an explicit heritage status",
            )
        )

    budgets = {compute_identity_sha256(arm.budget.as_dict()) for arm in experiment.arms}

    out.append(
        Precondition(
            code="budget_parity",
            status=READY if len(budgets) == 1 else BLOCKED,
            detail=(
                "all arms share one training budget"
                if len(budgets) == 1
                else f"{len(budgets)} distinct training budgets across arms"
            ),
        )
    )

    return out


def _split_preconditions(split_result) -> list:
    if split_result is None:
        return [
            Precondition(
                code="heritage_split_family_isolated",
                status=BLOCKED,
                detail=(
                    "no split was supplied. A heritage experiment requires a "
                    "family/line-isolated split; absence of evidence is treated "
                    "as failure, not as permission"
                ),
            )
        ]

    if not isinstance(split_result, SplitResult):
        raise PreflightError("split_result must be a SplitResult or None")

    if split_result.family_isolation:
        return [
            Precondition(
                code="heritage_split_family_isolated",
                status=READY,
                detail=(
                    f"split is family-isolated over {len(split_result.assignments)} "
                    f"families; {len(split_result.residual_leakage)} residual "
                    "risks documented"
                ),
            )
        ]

    return [
        Precondition(
            code="heritage_split_family_isolated",
            status=BLOCKED,
            detail=(
                f"split unit is {split_result.unit!r}, not a game/line/family. "
                "Near-adjacent positions from one game can straddle the split "
                "and the magnitude is unmeasurable without line identity"
            ),
        )
    ]


def _opening_preconditions(suite, pairs) -> list:
    if suite is None:
        return [
            Precondition(
                code="opening_suite_pinned",
                status=BLOCKED,
                detail="no opening suite supplied",
            ),
            Precondition(
                code="opening_suite_unbalanced",
                status=BLOCKED,
                detail="no opening suite supplied, so unbalancedness is unverified",
            ),
            Precondition(
                code="opening_suite_capacity",
                status=BLOCKED,
                detail="no opening suite supplied, so pair capacity is unverified",
            ),
        ]

    if not isinstance(suite, OpeningSuite):
        raise PreflightError("opening_suite must be an OpeningSuite or None")

    out = [
        Precondition(
            code="opening_suite_pinned",
            status=READY,
            detail=f"suite {suite.suite_id!r} pinned at {suite.sha256[:12]}...",
        )
    ]

    if suite.unbalanced_declared:
        out.append(
            Precondition(
                code="opening_suite_unbalanced",
                status=READY,
                detail=f"declared unbalanced: {suite.justification}",
            )
        )
    else:
        out.append(
            Precondition(
                code="opening_suite_unbalanced",
                status=BLOCKED,
                detail=(
                    f"suite {suite.suite_id!r} is not declared deliberately "
                    "unbalanced; balanced books produced 78.75-84.0% draw rates "
                    "in the largest observed sample"
                ),
            )
        )

    if pairs is None:
        out.append(
            Precondition(
                code="opening_suite_capacity",
                status=BLOCKED,
                detail="pair count not supplied, so capacity cannot be checked",
            )
        )
    elif isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise PreflightError(f"pairs must be an int >= 1 or None, got {pairs!r}")
    elif suite.distinct_count >= pairs:
        out.append(
            Precondition(
                code="opening_suite_capacity",
                status=READY,
                detail=(
                    f"{suite.distinct_count} distinct openings cover {pairs} pairs "
                    f"({pairs * 2} games) without replacement"
                ),
            )
        )
    else:
        out.append(
            Precondition(
                code="opening_suite_capacity",
                status=BLOCKED,
                detail=(
                    f"{suite.distinct_count} distinct openings cannot cover "
                    f"{pairs} pairs; sampling would have to repeat an opening, "
                    "which quarantines the run as malformed_opening_pair"
                ),
            )
        )

    return out


def evaluate_preflight(
    experiment,
    *,
    opening_suite=None,
    split_result=None,
    pairs=None,
) -> PreflightReport:
    """Evaluate every precondition and report whether execution is authorized.

    Every argument except `experiment` defaults to None, and every None yields
    BLOCKED. Calling this with an experiment alone therefore reports a fully
    blocked state, which is the correct default posture.
    """

    if not isinstance(experiment, ExperimentSpec):
        raise PreflightError(
            f"experiment must be an ExperimentSpec, got {type(experiment).__name__}"
        )

    preconditions = []
    preconditions.extend(_seed_preconditions(experiment))
    preconditions.extend(_data_preconditions(experiment))
    preconditions.extend(_split_preconditions(split_result))
    preconditions.extend(_opening_preconditions(opening_suite, pairs))

    return PreflightReport(
        experiment_id=experiment.experiment_id,
        experiment_identity_sha256=experiment.identity_sha256,
        preconditions=tuple(preconditions),
    )


def require_authorized(report: PreflightReport) -> None:
    """Raise NotAuthorized unless every precondition is READY."""

    if not isinstance(report, PreflightReport):
        raise PreflightError("report must be a PreflightReport")

    if report.authorized:
        return

    detail = "; ".join(f"{p.code}={p.status}" for p in report.blockers)

    raise NotAuthorized(
        f"experiment {report.experiment_id!r} is not authorized to execute: {detail}"
    )
