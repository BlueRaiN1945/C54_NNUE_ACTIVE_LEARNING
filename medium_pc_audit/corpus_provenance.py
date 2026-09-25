"""M0 corpus provenance: independent hash verification + evidence assembly.

M0 corpus pipeline step 5 (Medium-PC side). Mirrors medium_pc_audit.
frozen_seed's pattern -- literal, independently-checked evidence, not a
trusted declaration -- but for a freshly generated artifact rather than a
pre-existing one, so there is no pre-existing ground-truth hash to compare
against.

**The producer/consumer split, applied to data instead of match evidence.**
This applies the same producer/consumer separation used for match-result
evidence: the execution host is the producer -- it generates the corpus and
records its own claimed artifact_sha256 in a generation record.
This module is the consumer: it independently re-hashes the actual
transferred artifact file and treats the producer's claim as `expected`,
never as the truth. Only an independently-recomputed hash that agrees with
the claim corroborates -- this is verify_artifact_binding()'s and
manifest_verify.py's "never trust a manifest, always re-verify from disk"
discipline, applied here for the first time to a training corpus rather than
a match package.

**Why the seed hash is checked, not just the artifact hash.** The two are
different failure modes. An artifact-hash mismatch means "the file changed
in transit." A seed-hash mismatch means "this corpus was generated from a
DIFFERENT starting checkpoint than the one this project froze and verified" --
exactly the class of substitution error that motivated building this evidence
system in the first place. It is therefore a hard error (raised), not a
downgraded claim: a corpus generated from the wrong seed is not evidence of
anything about the frozen seed's heritage-freedom, it is evidence about a
different pipeline run entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

from medium_pc_audit.artifact_import import compute_file_sha256
from medium_pc_audit.experiment import (
    EVIDENCE_ARTIFACT_HASH_OBSERVATION,
    EVIDENCE_CORPUS_GENERATION_RECORD,
    HERITAGE_FREE_DECLARED,
    DataSourceSpec,
    EvidenceRef,
)
from medium_pc_audit.frozen_seed import SEED_SHA256
from medium_pc_audit.schemas import registry

REQUIRED_RECORD_FIELDS = ("artifact_sha256", "seed_sha256")


class CorpusProvenanceError(ValueError):
    """Raised for malformed input or an anchor mismatch (e.g. wrong seed).

    Never raised for an artifact-hash mismatch -- that case is reported as a
    non-corroborating EvidenceRef instead, per this module's fail-closed
    contract: it must be visible in the resulting DataSourceSpec, not hidden
    behind an exception the caller might not construct the spec to see.
    """


def _require_dict(value, *, name: str) -> dict:
    if not isinstance(value, dict):
        raise CorpusProvenanceError(f"{name} must be a dict, got {type(value).__name__}")

    return value


def _require_field(record: dict, field: str) -> str:
    if field not in record:
        raise CorpusProvenanceError(
            f"generation_record is missing required field {field!r}"
        )

    return record[field]


def _validate_seed_anchor(record: dict) -> None:
    declared = _require_field(record, "seed_sha256")

    if declared != SEED_SHA256:
        raise CorpusProvenanceError(
            f"generation_record.seed_sha256 ({declared[:12]}...) does not match "
            f"the frozen approved seed ({SEED_SHA256[:12]}...). A corpus "
            "generated from a different seed cannot be evidence about this "
            "project's frozen seed's heritage-freedom."
        )


def build_corpus_data_source(
    *,
    artifact_path,
    source_id: str,
    generation_record: dict,
    justification: str,
) -> DataSourceSpec:
    """Build a DataSourceSpec for a generated M0 corpus artifact.

    artifact_path is hashed HERE, independently -- this is the only source of
    truth for the resulting DataSourceSpec.sha256. generation_record's own
    artifact_sha256 field is used only as the `expected` side of a comparison,
    never copied through as the spec's identity.

    Raises CorpusProvenanceError for malformed input or a seed-anchor
    mismatch. Returns normally (with a non-corroborating, non-heritage-free
    spec) for an artifact-hash mismatch -- see CorpusProvenanceError's
    docstring for why that distinction is deliberate.
    """

    _require_dict(generation_record, name="generation_record")

    try:
        registry.validate("corpus_generation_record", "v1", generation_record)
    except registry.SchemaValidationError as exc:
        raise CorpusProvenanceError(
            f"generation_record failed schema validation: {exc}"
        ) from exc

    _require_field(generation_record, "artifact_sha256")
    _validate_seed_anchor(generation_record)

    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise CorpusProvenanceError(f"artifact_path does not exist: {artifact_path}")

    observed_sha256 = compute_file_sha256(artifact_path)
    claimed_sha256 = generation_record["artifact_sha256"]

    hash_evidence = EvidenceRef(
        kind=EVIDENCE_ARTIFACT_HASH_OBSERVATION,
        record_id=f"sha256({artifact_path.name})",
        subject_sha256=observed_sha256,
        verified_at=generation_record.get("created_utc", "1970-01-01T00:00:00Z"),
        result_sha256=observed_sha256,
        expected_sha256=claimed_sha256,
        detail=(
            f"independently hashed {artifact_path}; producer claimed "
            f"{claimed_sha256[:12]}..."
        ),
    )

    generation_evidence = EvidenceRef(
        kind=EVIDENCE_CORPUS_GENERATION_RECORD,
        record_id=f"generation_record({source_id})",
        subject_sha256=observed_sha256,
        verified_at=generation_record.get("created_utc", "1970-01-01T00:00:00Z"),
        detail=json.dumps(generation_record, sort_keys=True),
    )

    return DataSourceSpec(
        source_id=source_id,
        sha256=observed_sha256,
        heritage_status=HERITAGE_FREE_DECLARED,
        justification=justification,
        evidence=(hash_evidence, generation_evidence),
    )
