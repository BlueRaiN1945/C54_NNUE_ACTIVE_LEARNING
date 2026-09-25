"""Experiment/arm specification and provenance contracts.

Stdlib-only and deterministic: no filesystem, network, engine, training, or
wall-clock access anywhere in this module.

This module encodes, as executable guards, the locked methodology decisions
recorded in C54_HERITAGE_TRANSFER/M0_CONTROL_PROPOSAL.md and
PHASE1_5_PRECONDITIONS.md. Three of those decisions are enforced here rather
than left to convention, because each has already been violated in practice
by the existing BigPC lines:

  - **Shared seed.** An M0 control arm and every heritage treatment arm it is
    compared against must start from a byte-identical checkpoint.
    C54_V80_RESEARCH_A seeds from champion generation 80 -- 80 generations of
    heritage fine-tuning downstream of the official base -- so it cannot form
    a valid pair with a control seeded from that base. Enforced by
    _require_shared_seed().

  - **Heritage-freedom is declared, never inferred.** A corpus is only
    heritage-free if a caller explicitly says so AND supplies a justification.
    The default is HERITAGE_UNKNOWN, and UNKNOWN fails closed. This exists
    because OFFICIAL_LINE_A was treated as a "modern-only" control while
    mining from the same heritage-curated TRAIN_95 pool as the heritage arms.

  - **Budget parity.** All arms share one TrainingBudget, so an observed
    strength difference is attributable to the data rather than the schedule.

Seed provenance/trainability are carried as explicit booleans that default to
False. This module does not decide whether they are satisfied -- it only
records the claim; medium_pc_audit.preflight is what refuses to authorize
execution when they are False.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medium_pc_audit.identity import compute_identity_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# -- heritage status of a training corpus -------------------------------------
HERITAGE_CURATED = "heritage_curated"
HERITAGE_FREE_DECLARED = "heritage_free_declared"
HERITAGE_UNKNOWN = "heritage_unknown"

HERITAGE_STATUSES = frozenset(
    {
        HERITAGE_CURATED,
        HERITAGE_FREE_DECLARED,
        HERITAGE_UNKNOWN,
    }
)

# -- arm roles ----------------------------------------------------------------
ROLE_CONTROL_M0 = "control_m0"
ROLE_TREATMENT_HERITAGE = "treatment_heritage"

ARM_ROLES = frozenset(
    {
        ROLE_CONTROL_M0,
        ROLE_TREATMENT_HERITAGE,
    }
)

OPTIMIZER_NAMES = frozenset(
    {
        "schedulefree",
        "ranger21",
        "rangerlite",
    }
)

# -- what a piece of evidence can and cannot establish -------------------------
#
# These claims are deliberately distinct. A serialization roundtrip proves that
# a checkpoint DECODES to a known network; it says nothing about where the
# checkpoint came from. Conflating the two is the specific error this taxonomy
# exists to make impossible: EVIDENCE_KIND_ESTABLISHES contains no mapping from
# a roundtrip to CLAIM_HISTORICAL_ORIGIN, so no quantity of roundtrip evidence
# can ever satisfy a provenance gate.
CLAIM_SERIALIZATION_EQUIVALENCE = "serialization_equivalence"
CLAIM_HISTORICAL_ORIGIN = "historical_origin"
CLAIM_TRAINABILITY = "trainability"

# -- corpus claims, deliberately separated ------------------------------------
#
# A corpus passes through four distinct states, and collapsing them is how a
# placeholder gets mistaken for data:
#
#   1. declared            -- a DataSourceSpec exists saying what is intended
#   2. identified          -- a real artifact was located
#   3. hash verified       -- that artifact's bytes were hashed by observation
#   4. heritage-free       -- a record establishes how it was built/audited
#
# Only (1) is expressible without evidence, and (1) alone establishes nothing.
# Heritage-freedom additionally requires (3): a claim about an artifact whose
# bytes were never observed is a claim about nothing.
CLAIM_CORPUS_IDENTIFIED = "corpus_identified"
CLAIM_CORPUS_HASH_VERIFIED = "corpus_hash_verified"
CLAIM_HERITAGE_FREE = "heritage_free"

CLAIMS = frozenset(
    {
        CLAIM_SERIALIZATION_EQUIVALENCE,
        CLAIM_HISTORICAL_ORIGIN,
        CLAIM_TRAINABILITY,
        CLAIM_CORPUS_IDENTIFIED,
        CLAIM_CORPUS_HASH_VERIFIED,
        CLAIM_HERITAGE_FREE,
    }
)

EVIDENCE_SERIALIZATION_ROUNDTRIP = "serialization_roundtrip"
EVIDENCE_CREATION_LOG = "creation_log"
EVIDENCE_IMPORT_RECORD = "import_record"
EVIDENCE_TRAINING_SMOKE_RUN = "training_smoke_run"

# Hashing a file that actually exists. Hash-compared, so a placeholder cannot
# produce one: there is nothing to hash.
EVIDENCE_ARTIFACT_HASH_OBSERVATION = "artifact_hash_observation"

# How the corpus was produced, or an audit of it against heritage sources.
EVIDENCE_CORPUS_GENERATION_RECORD = "corpus_generation_record"
EVIDENCE_CORPUS_PROVENANCE_AUDIT = "corpus_provenance_audit"

EVIDENCE_KIND_ESTABLISHES = {
    EVIDENCE_SERIALIZATION_ROUNDTRIP: frozenset({CLAIM_SERIALIZATION_EQUIVALENCE}),
    EVIDENCE_CREATION_LOG: frozenset({CLAIM_HISTORICAL_ORIGIN}),
    EVIDENCE_IMPORT_RECORD: frozenset({CLAIM_HISTORICAL_ORIGIN}),
    EVIDENCE_TRAINING_SMOKE_RUN: frozenset({CLAIM_TRAINABILITY}),
    EVIDENCE_ARTIFACT_HASH_OBSERVATION: frozenset(
        {CLAIM_CORPUS_IDENTIFIED, CLAIM_CORPUS_HASH_VERIFIED}
    ),
    EVIDENCE_CORPUS_GENERATION_RECORD: frozenset({CLAIM_HERITAGE_FREE}),
    EVIDENCE_CORPUS_PROVENANCE_AUDIT: frozenset({CLAIM_HERITAGE_FREE}),
}

# Kinds whose corroboration is decided by comparing two recorded hashes rather
# than by a caller asserting it.
_HASH_COMPARED_KINDS = frozenset(
    {
        EVIDENCE_SERIALIZATION_ROUNDTRIP,
        EVIDENCE_ARTIFACT_HASH_OBSERVATION,
    }
)

# Claims an experiment may proceed WITHOUT, provided the gap is declared.
#
# Only historical origin is waivable, and deliberately so. Serialization
# equivalence and trainability are technical facts that can simply be tested --
# waiving them would mean declining to run a cheap check, which is never a
# legitimate experimental trade-off. Origin is different: it depends on records
# that may genuinely not exist and cannot be manufactured after the fact.
#
# A waiver never deletes the finding. It converts a blocking gate into a
# recorded acknowledgement that is bound into the experiment's identity hash,
# so an experiment that waived something can never be mistaken for one that
# satisfied it.
WAIVABLE_CLAIMS = frozenset({CLAIM_HISTORICAL_ORIGIN})


class ExperimentError(ValueError):
    """Raised when an experiment/arm invariant is violated."""


def _require_identifier(value, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"{name} must be a non-empty string, got {value!r}")

    if not _IDENTIFIER_RE.fullmatch(value):
        raise ExperimentError(
            f"{name} must match {_IDENTIFIER_RE.pattern!r}, got {value!r}"
        )

    return value


def _require_sha256(value, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ExperimentError(
            f"{name} must be 64 lowercase hexadecimal characters, got {value!r}"
        )

    return value


def _require_text(value, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{name} must be a non-empty string, got {value!r}")

    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ExperimentError(f"{name} must not contain control characters")

    return value


def _require_bool(value, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentError(f"{name} must be a bool, got {value!r}")

    return value


def _require_positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentError(f"{name} must be an int >= 1, got {value!r}")

    return value


def _require_positive_float(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentError(f"{name} must be a number, got {value!r}")

    value = float(value)

    if not value > 0.0:
        raise ExperimentError(f"{name} must be > 0, got {value}")

    return value


@dataclass(frozen=True)
class EvidenceRef:
    """A checkable reference to evidence about a specific artifact.

    This replaces free boolean assertions like `provenance_documented=True`.
    Three properties make it verifiable rather than merely declared:

      1. **It names a record.** record_id must identify where the evidence
         lives (a path, log name, or run identifier).
      2. **It is bound to a subject.** subject_sha256 is the hash of the
         artifact the evidence is about, so evidence for one file cannot be
         silently transplanted onto another.
      3. **Hash-compared kinds decide their own corroboration.** For a
         serialization roundtrip the caller records result_sha256 and
         expected_sha256; `corroborates` is computed by comparing them, not
         asserted. A roundtrip whose hashes disagree can still be recorded --
         honest negative results matter -- but it will never satisfy a gate.

    What a kind can establish is fixed by EVIDENCE_KIND_ESTABLISHES. A
    roundtrip establishes serialization equivalence only; it can never
    establish historical origin.
    """

    kind: str
    record_id: str
    subject_sha256: str
    verified_at: str
    result_sha256: str = ""
    expected_sha256: str = ""
    detail: str = ""

    def __post_init__(self):
        if self.kind not in EVIDENCE_KIND_ESTABLISHES:
            raise ExperimentError(
                f"evidence.kind must be one of "
                f"{sorted(EVIDENCE_KIND_ESTABLISHES)}, got {self.kind!r}"
            )

        _require_text(self.record_id, name="evidence.record_id")
        _require_text(self.verified_at, name="evidence.verified_at")
        _require_sha256(self.subject_sha256, name="evidence.subject_sha256")

        if self.kind in _HASH_COMPARED_KINDS:
            _require_sha256(self.result_sha256, name="evidence.result_sha256")
            _require_sha256(self.expected_sha256, name="evidence.expected_sha256")
        else:
            for name in ("result_sha256", "expected_sha256"):
                value = getattr(self, name)

                if value:
                    _require_sha256(value, name=f"evidence.{name}")

    @property
    def corroborates(self) -> bool:
        """Whether this evidence actually supports its claim.

        For hash-compared kinds this is a computed comparison, never a
        caller's assertion.
        """

        if self.kind in _HASH_COMPARED_KINDS:
            return self.result_sha256 == self.expected_sha256

        return True

    @property
    def establishes(self) -> frozenset:
        if not self.corroborates:
            return frozenset()

        return EVIDENCE_KIND_ESTABLISHES[self.kind]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "record_id": self.record_id,
            "subject_sha256": self.subject_sha256,
            "verified_at": self.verified_at,
            "result_sha256": self.result_sha256,
            "expected_sha256": self.expected_sha256,
            "detail": self.detail,
            "corroborates": self.corroborates,
            "establishes": sorted(self.establishes),
        }


@dataclass(frozen=True)
class SeedSpec:
    """A trainable starting checkpoint and the evidence held about it.

    There are no boolean provenance flags. What is known about this seed is
    derived from `evidence`, and only from evidence whose subject_sha256
    matches this seed's own hash -- so evidence gathered about a different
    checkpoint contributes nothing here.
    """

    artifact_id: str
    sha256: str
    evidence: tuple = ()

    def __post_init__(self):
        _require_identifier(self.artifact_id, name="seed.artifact_id")
        _require_sha256(self.sha256, name="seed.sha256")

        evidence = tuple(self.evidence)

        for item in evidence:
            if not isinstance(item, EvidenceRef):
                raise ExperimentError(
                    f"seed.evidence entries must be EvidenceRef, "
                    f"got {type(item).__name__}"
                )

        object.__setattr__(self, "evidence", evidence)

    @property
    def bound_evidence(self) -> tuple:
        """Evidence whose subject is this exact checkpoint."""

        return tuple(e for e in self.evidence if e.subject_sha256 == self.sha256)

    @property
    def established_claims(self) -> frozenset:
        claims = frozenset()

        for item in self.bound_evidence:
            claims |= item.establishes

        return claims

    def establishes(self, claim: str) -> bool:
        if claim not in CLAIMS:
            raise ExperimentError(
                f"claim must be one of {sorted(CLAIMS)}, got {claim!r}"
            )

        return claim in self.established_claims

    def as_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "evidence": [e.as_dict() for e in self.evidence],
            "established_claims": sorted(self.established_claims),
        }


@dataclass(frozen=True)
class DataSourceSpec:
    """A training corpus, its declared intent, and the evidence held about it.

    **heritage_status is an intent, not a finding.** Setting it to
    HERITAGE_FREE_DECLARED says what the corpus is meant to be; it establishes
    nothing. The heritage-free *claim* comes only from `evidence`, and only
    from evidence bound to this corpus's own hash.

    That separation exists because the earlier model let a placeholder --
    a name, a made-up hash, a justification string -- turn the control-corpus
    gate green while no corpus existed at all. A declaration is not evidence.
    """

    source_id: str
    sha256: str
    heritage_status: str = HERITAGE_UNKNOWN
    justification: str = ""
    row_count: int | None = None
    evidence: tuple = ()

    @property
    def bound_evidence(self) -> tuple:
        """Evidence whose subject is this exact corpus."""

        return tuple(e for e in self.evidence if e.subject_sha256 == self.sha256)

    @property
    def established_claims(self) -> frozenset:
        claims = frozenset()

        for item in self.bound_evidence:
            claims |= item.establishes

        # A heritage-free claim about an artifact whose bytes were never
        # observed is a claim about nothing, so it is withheld until the
        # corpus itself has been hash-verified.
        if CLAIM_CORPUS_HASH_VERIFIED not in claims:
            claims -= {CLAIM_HERITAGE_FREE}

        return claims

    def establishes(self, claim: str) -> bool:
        if claim not in CLAIMS:
            raise ExperimentError(
                f"claim must be one of {sorted(CLAIMS)}, got {claim!r}"
            )

        return claim in self.established_claims

    def __post_init__(self):
        _require_identifier(self.source_id, name="data_source.source_id")
        _require_sha256(self.sha256, name="data_source.sha256")

        if self.heritage_status not in HERITAGE_STATUSES:
            raise ExperimentError(
                f"data_source.heritage_status must be one of "
                f"{sorted(HERITAGE_STATUSES)}, got {self.heritage_status!r}"
            )

        if not isinstance(self.justification, str):
            raise ExperimentError("data_source.justification must be a string")

        if self.heritage_status == HERITAGE_FREE_DECLARED:
            _require_text(
                self.justification,
                name="data_source.justification",
            )

        if self.row_count is not None:
            _require_positive_int(self.row_count, name="data_source.row_count")

        evidence = tuple(self.evidence)

        for item in evidence:
            if not isinstance(item, EvidenceRef):
                raise ExperimentError(
                    f"data_source.evidence entries must be EvidenceRef, "
                    f"got {type(item).__name__}"
                )

        object.__setattr__(self, "evidence", evidence)

    @property
    def is_heritage_free_declared(self) -> bool:
        """What the corpus is INTENDED to be. Establishes nothing on its own."""

        return self.heritage_status == HERITAGE_FREE_DECLARED

    @property
    def is_heritage_free_established(self) -> bool:
        """What the EVIDENCE supports. This is the one that counts."""

        return self.establishes(CLAIM_HERITAGE_FREE)

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "sha256": self.sha256,
            "heritage_status": self.heritage_status,
            "justification": self.justification,
            "row_count": self.row_count,
            "evidence": [e.as_dict() for e in self.evidence],
            "established_claims": sorted(self.established_claims),
        }


@dataclass(frozen=True)
class TrainingBudget:
    """Compute/schedule parameters. Shared identically across all arms."""

    lr: float
    epoch_size: int
    batch_size: int
    blocks: int
    optimizer_name: str = "rangerlite"
    lambda_: float = 1.0

    def __post_init__(self):
        _require_positive_float(self.lr, name="budget.lr")
        _require_positive_int(self.epoch_size, name="budget.epoch_size")
        _require_positive_int(self.batch_size, name="budget.batch_size")
        _require_positive_int(self.blocks, name="budget.blocks")

        if self.optimizer_name not in OPTIMIZER_NAMES:
            raise ExperimentError(
                f"budget.optimizer_name must be one of "
                f"{sorted(OPTIMIZER_NAMES)}, got {self.optimizer_name!r}"
            )

        if (
            isinstance(self.lambda_, bool)
            or not isinstance(self.lambda_, (int, float))
            or not 0.0 <= float(self.lambda_) <= 1.0
        ):
            raise ExperimentError(
                f"budget.lambda_ must be a number in [0, 1], got {self.lambda_!r}"
            )

    def as_dict(self) -> dict:
        return {
            "lr": float(self.lr),
            "epoch_size": self.epoch_size,
            "batch_size": self.batch_size,
            "blocks": self.blocks,
            "optimizer_name": self.optimizer_name,
            "lambda": float(self.lambda_),
        }


@dataclass(frozen=True)
class ArmSpec:
    """One experimental arm: a seed, a corpus, and a budget."""

    arm_id: str
    role: str
    seed: SeedSpec
    data_source: DataSourceSpec
    budget: TrainingBudget

    def __post_init__(self):
        _require_identifier(self.arm_id, name="arm.arm_id")

        if self.role not in ARM_ROLES:
            raise ExperimentError(
                f"arm.role must be one of {sorted(ARM_ROLES)}, got {self.role!r}"
            )

        for name, value, expected in (
            ("seed", self.seed, SeedSpec),
            ("data_source", self.data_source, DataSourceSpec),
            ("budget", self.budget, TrainingBudget),
        ):
            if not isinstance(value, expected):
                raise ExperimentError(
                    f"arm.{name} must be a {expected.__name__}, "
                    f"got {type(value).__name__}"
                )

        # The control arm's whole purpose is 'no heritage positions/targets'.
        # A control whose corpus is not even *declared* heritage-free is not a
        # control, so this is rejected at construction time. Note this is only
        # the declaration check -- whether the claim is ESTABLISHED is decided
        # by evidence in preflight, and construction succeeding here says
        # nothing about that.
        if (
            self.role == ROLE_CONTROL_M0
            and not self.data_source.is_heritage_free_declared
        ):
            raise ExperimentError(
                f"arm {self.arm_id!r} has role {ROLE_CONTROL_M0!r} but its data "
                f"source {self.data_source.source_id!r} has heritage_status "
                f"{self.data_source.heritage_status!r}; an M0 control requires "
                f"{HERITAGE_FREE_DECLARED!r}"
            )

    def as_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "role": self.role,
            "seed": self.seed.as_dict(),
            "data_source": self.data_source.as_dict(),
            "budget": self.budget.as_dict(),
        }


def _require_shared_seed(arms) -> None:
    """Every arm must start from the byte-identical checkpoint.

    This is the guard that the existing BigPC pairing fails: a treatment arm
    continued from champion generation 80 and a control seeded from the
    official base do not share a starting point, so any measured difference
    confounds 'heritage data' with '80 generations of prior fine-tuning'.
    """

    seen = {}

    for arm in arms:
        seen.setdefault(arm.seed.sha256, []).append(arm.arm_id)

    if len(seen) > 1:
        detail = "; ".join(
            f"{sha256}: {sorted(arm_ids)}" for sha256, arm_ids in sorted(seen.items())
        )

        raise ExperimentError(
            "all arms must start from a byte-identical seed checkpoint, but "
            f"{len(seen)} distinct seed hashes were supplied -- {detail}"
        )


def _require_shared_budget(arms) -> None:
    seen = {}

    for arm in arms:
        key = compute_identity_sha256(arm.budget.as_dict())
        seen.setdefault(key, []).append(arm.arm_id)

    if len(seen) > 1:
        raise ExperimentError(
            "all arms must share an identical TrainingBudget so that an "
            "observed difference is attributable to the data rather than the "
            f"schedule; {len(seen)} distinct budgets were supplied: "
            + "; ".join(sorted(str(sorted(v)) for v in seen.values()))
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """A control arm plus one or more heritage treatment arms."""

    experiment_id: str
    arms: tuple
    description: str = ""
    accepted_unknowns: tuple = ()
    _identity: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self):
        _require_identifier(self.experiment_id, name="experiment_id")

        accepted = tuple(self.accepted_unknowns)

        for claim in accepted:
            if claim not in CLAIMS:
                raise ExperimentError(
                    f"accepted_unknowns entries must be claims from "
                    f"{sorted(CLAIMS)}, got {claim!r}"
                )

            if claim not in WAIVABLE_CLAIMS:
                raise ExperimentError(
                    f"claim {claim!r} is not waivable; it is a technical fact "
                    "that can be established by running the corresponding "
                    f"check. Waivable claims: {sorted(WAIVABLE_CLAIMS)}"
                )

        if len(set(accepted)) != len(accepted):
            raise ExperimentError("accepted_unknowns must not contain duplicates")

        object.__setattr__(self, "accepted_unknowns", tuple(sorted(accepted)))

        arms = tuple(self.arms)

        if not arms:
            raise ExperimentError("experiment must contain at least one arm")

        for arm in arms:
            if not isinstance(arm, ArmSpec):
                raise ExperimentError(
                    f"every arm must be an ArmSpec, got {type(arm).__name__}"
                )

        arm_ids = [arm.arm_id for arm in arms]
        duplicates = sorted({a for a in arm_ids if arm_ids.count(a) > 1})

        if duplicates:
            raise ExperimentError(f"duplicate arm_id values: {duplicates}")

        controls = [a for a in arms if a.role == ROLE_CONTROL_M0]
        treatments = [a for a in arms if a.role == ROLE_TREATMENT_HERITAGE]

        if len(controls) != 1:
            raise ExperimentError(
                f"experiment must contain exactly one {ROLE_CONTROL_M0!r} arm, "
                f"got {len(controls)}"
            )

        if not treatments:
            raise ExperimentError(
                f"experiment must contain at least one "
                f"{ROLE_TREATMENT_HERITAGE!r} arm"
            )

        _require_shared_seed(arms)
        _require_shared_budget(arms)

        object.__setattr__(self, "arms", arms)
        object.__setattr__(
            self,
            "_identity",
            compute_identity_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "accepted_unknowns": list(self.accepted_unknowns),
            "arms": [
                arm.as_dict() for arm in sorted(self.arms, key=lambda a: a.arm_id)
            ],
        }

    @property
    def identity_sha256(self) -> str:
        """Deterministic identity over the full arm specification.

        Excludes nothing: two experiments differing in any declared field --
        including a seed's provenance claims -- are different experiments.
        """
        return self._identity

    @property
    def control(self) -> ArmSpec:
        return next(a for a in self.arms if a.role == ROLE_CONTROL_M0)

    @property
    def treatments(self) -> tuple:
        return tuple(a for a in self.arms if a.role == ROLE_TREATMENT_HERITAGE)

    @property
    def shared_seed(self) -> SeedSpec:
        return self.arms[0].seed

    def as_dict(self) -> dict:
        payload = self._identity_payload()
        payload["identity_sha256"] = self.identity_sha256
        return payload


def build_experiment(
    *,
    experiment_id: str,
    arms,
    description: str = "",
    accepted_unknowns=(),
) -> ExperimentSpec:
    """Construct and fully validate an ExperimentSpec."""

    return ExperimentSpec(
        experiment_id=experiment_id,
        arms=tuple(arms),
        description=description,
        accepted_unknowns=tuple(accepted_unknowns),
    )
