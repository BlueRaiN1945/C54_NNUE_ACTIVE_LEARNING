"""The frozen common experimental seed for M0 and all treatment arms.

Stdlib-only, no I/O. This module records -- as constants and one factory --
the seed that was verified on BigPC on 2026-09-24 and then frozen by explicit
user decision.

**What is established, and what is not.**

  - VERIFIED: serialization equivalence. Running the pinned serializer on this
    exact checkpoint reproduced the official network byte-for-byte. Observed
    directly, not inferred from a pre-existing file.
  - VERIFIED: trainability. The pinned trainer accepted it through
    `--resume-from-model`, executed eight real optimizer updates, and the
    result serialized to an NNUE that Stockfish loaded and searched with.
  - **UNKNOWN: historical origin.** No creation log, import record, or
    download receipt exists for the `.pt` file. Neither verification touches
    this question. One shows what the checkpoint decodes *to*; the other shows
    the trainer *accepts* it. Neither shows where it came *from*.

The unknown is not an execution blocker for the new controlled experiment --
that is a recorded user decision -- but it is never silently dropped. An
experiment using this seed must declare CLAIM_HISTORICAL_ORIGIN in its
`accepted_unknowns`, which binds the acknowledgement into the experiment's
identity hash. See ACCEPTED_UNKNOWNS below.

Hashes here are the evidence itself, so they are written as literals rather
than loaded from anywhere: a pin that a config file could edit is not a pin.
"""

from __future__ import annotations

from medium_pc_audit.experiment import (
    CLAIM_HISTORICAL_ORIGIN,
    EVIDENCE_SERIALIZATION_ROUNDTRIP,
    EVIDENCE_TRAINING_SMOKE_RUN,
    EvidenceRef,
    SeedSpec,
)

# -- the pinned artifacts -----------------------------------------------------
SEED_ARTIFACT_ID = "OFFICIAL_BASE_PT"
SEED_PATH = "/data/official_base.pt"
SEED_SHA256 = "dc41943b473c82e878e6c967a4489dd26fd58619362b85bb27a0fbeccee91c82"

OFFICIAL_NNUE_PATH = "/data/STOCKFISH_LINUX_17a6c8f1/src/nn-134a887f4c8f.nnue"
OFFICIAL_NNUE_SHA256 = (
    "134a887f4c8ff7bf7284177a3b3fc6ff9cef95ba89eb8db3079a8e507f7126af"
)

SERIALIZER_PATH = "/workspace/nnue-pytorch/serialize.py"
SERIALIZER_SHA256 = "25da9e6530c6c3142a513c49b6bb0e9a65150cb4e856be96a8dc52bf3fb17abf"
TRAINER_COMMIT = "9f72946529c4187d3679014036cd22c3be419716"

# -- the trainability smoke (disposable artifacts, hashes preserved) ----------
SMOKE_DATASET_PATH = "/workspace/nnue-pytorch/.pgo/small.binpack"
SMOKE_DATASET_SHA256 = (
    "61b894937b24e031147a6734f9f176507a9b4f62225eb202d6852dbb054faf71"
)
SMOKE_CHECKPOINT_SHA256 = (
    "a3404bf89288166c3c731ea3c87db925253d61d40a4aecf023c7222773792326"
)
SMOKE_NNUE_SHA256 = "4e129d545c5c68e9e6a1e2201e53efd02e7bdaa8b50b38808cd827b1e86dfa2a"
SMOKE_UPDATES = 8

VERIFIED_AT = "2026-09-24T08:52:50Z"

# Historical origin is unresolved and accepted as such. Any experiment built
# on this seed must pass this through to ExperimentSpec.accepted_unknowns.
ACCEPTED_UNKNOWNS = (CLAIM_HISTORICAL_ORIGIN,)


def serialization_evidence() -> EvidenceRef:
    """The observed roundtrip: seed -> serializer -> official network."""

    return EvidenceRef(
        kind=EVIDENCE_SERIALIZATION_ROUNDTRIP,
        record_id="roundtrip 2026-09-24 via /tmp/c54_seed_roundtrip_verify.nnue",
        subject_sha256=SEED_SHA256,
        verified_at="2026-09-24T08:40:00Z",
        result_sha256=OFFICIAL_NNUE_SHA256,
        expected_sha256=OFFICIAL_NNUE_SHA256,
        detail=(
            f"serialize.py ({SERIALIZER_SHA256[:12]}...) at trainer commit "
            f"{TRAINER_COMMIT[:12]}... reproduced {OFFICIAL_NNUE_PATH} "
            "byte-for-byte (98,961,994 bytes); temporary output removed"
        ),
    )


def trainability_evidence() -> EvidenceRef:
    """The observed smoke: seed -> trainer -> updated net that Stockfish runs."""

    return EvidenceRef(
        kind=EVIDENCE_TRAINING_SMOKE_RUN,
        record_id="trainability smoke 2026-09-24 (/tmp/c54_trainability_smoke)",
        subject_sha256=SEED_SHA256,
        verified_at=VERIFIED_AT,
        detail=(
            f"--resume-from-model accepted; {SMOKE_UPDATES} optimizer updates "
            f"on {SMOKE_DATASET_PATH} ({SMOKE_DATASET_SHA256[:12]}...); "
            f"checkpoint {SMOKE_CHECKPOINT_SHA256[:12]}... serialized to "
            f"{SMOKE_NNUE_SHA256[:12]}..., which differs from the untrained "
            f"{OFFICIAL_NNUE_SHA256[:12]}... proving weights changed; Stockfish "
            "bench 102,681 nodes. Technical pipeline result only -- no strength "
            "implication. Artifacts disposed."
        ),
    )


def frozen_seed() -> SeedSpec:
    """The frozen common seed, carrying its verified evidence.

    Every arm of a controlled experiment must use this exact seed, so that a
    measured difference is attributable to the variable under test rather than
    to differing starting weights.
    """

    return SeedSpec(
        artifact_id=SEED_ARTIFACT_ID,
        sha256=SEED_SHA256,
        evidence=(
            serialization_evidence(),
            trainability_evidence(),
        ),
    )
