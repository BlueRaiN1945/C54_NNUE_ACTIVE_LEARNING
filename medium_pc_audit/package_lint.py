"""Read-only, side-effect-free validation of a candidate execution package.

This is the Medium-PC half of the evidence bridge. The execution host produces
a package; this module tells it, before anything is shipped or ingested,
whether that package would be accepted and exactly what is wrong if not.

**Why this exists separately from ingest_result_package().** Ingestion
*commits*: it copies the package into the immutable inbox and appends
permanent events to the append-only state history. Those are the right
semantics for real evidence and the wrong semantics for iterating on a
packager. A failed ingest leaves a QUARANTINED state record that can never be
retracted, so using ingestion as a development check would pollute the
research record with the packager's teething problems. This module performs
the same checks and writes nothing, anywhere.

**What it deliberately does NOT do.** It never produces or repairs a package.
The `actual_*` fields in ExecutionManifestV1 must be observed on the execution
host; anything that filled them in from the MatchConfig would make
`verify_execution_manifest()` compare a config against a copy of itself and
pass trivially. See EXECUTION_PACKAGE_CONTRACT.md §0. Medium-PC verifies; the
execution host attests.

All real checking is delegated to the already-committed implementations
(verify_transport_integrity, verify_artifact_binding, classify_result_bundle)
-- this module composes and reports, it does not reimplement.
"""

from __future__ import annotations

import json
from pathlib import Path

from medium_pc_audit.inbox_validate import (
    ACCEPTED,
    INCOMPLETE,
    QUARANTINED,
    classify_result_bundle,
)
from medium_pc_audit.paths import require_neutral_path
from medium_pc_audit.result_package import (
    REQUIRED_PACKAGE_FILES,
    SHA256SUMS_NAME,
    verify_artifact_binding,
    verify_transport_integrity,
)

WOULD_BE_ACCEPTED = "WOULD_BE_ACCEPTED"
WOULD_BE_INCOMPLETE = "WOULD_BE_INCOMPLETE"
WOULD_BE_QUARANTINED = "WOULD_BE_QUARANTINED"

_PREDICTED_FROM_STATE = {
    ACCEPTED: WOULD_BE_ACCEPTED,
    INCOMPLETE: WOULD_BE_INCOMPLETE,
    QUARANTINED: WOULD_BE_QUARANTINED,
}


class PackageLintError(ValueError):
    """Raised for lint API misuse, never for a package simply being wrong."""


def _load_json(path: Path):
    """Load a package JSON under the contract's stricter portability rules."""

    raw = path.read_bytes()

    if b"\r" in raw:
        raise ValueError(f"{path.name} must use LF line endings, not CRLF")

    return json.loads(raw.decode("utf-8"))


def _finding(code: str, severity: str, detail, fix: str) -> dict:
    return {
        "code": code,
        "severity": severity,
        "detail": detail,
        "fix": fix,
    }


_FIX_HINTS = {
    "unpaired_run_not_supported_v1": (
        "Set repeat=true in the MatchConfig and run paired, colour-swapped "
        "games; v1 intake only supports paired runs."
    ),
    "paired_max_games_must_be_even": (
        "sprt.max_games must be even, since every opening yields two games."
    ),
    "game_indices_not_exactly_sequential": (
        "game_index must run 1..N with no gaps, duplicates, or reordering."
    ),
    "odd_game_count_for_paired_run": (
        "An odd game count means a pair is unfinished. Complete or drop it."
    ),
    "incomplete_opening_pair": (
        "Every opening_ref needs exactly two games, one per colour. A missing "
        "second game usually means the run was cut short."
    ),
    "malformed_opening_pair": (
        "More than two games share an opening_ref -- usually an opening drawn "
        "twice. Sample openings WITHOUT replacement "
        "(opening_suite.select_openings) or use order=sequential."
    ),
    "game_count_exceeds_max_games": (
        "More games were played than sprt.max_games allows."
    ),
    "sprt_continue_before_cap": (
        "The GSPRT has not reached a boundary or the game cap, so the run is "
        "genuinely unfinished. Continue it rather than reporting it."
    ),
    "execution_manifest_mismatch": (
        "An actual_* field disagrees with the MatchConfig. The host did not do "
        "what was asked -- investigate the run. Never 'fix' this by copying "
        "config values into the manifest; that defeats the whole check."
    ),
    "raw_result_run_id_mismatch": (
        "raw_result.run_id must equal match_config.config_id."
    ),
    "no_games": "The package contains no games.",
    "schema_invalid": (
        "A document failed its frozen v1 schema. See the detail for the field."
    ),
}


def lint_execution_package(package_dir, *, match_config: dict) -> dict:
    """Report whether a candidate package would be accepted, and why not.

    Writes nothing and touches no inbox or state history. Returns a dict with
    `ok`, a predicted intake outcome, and an ordered `findings` list carrying
    an actionable `fix` for each problem.

    Checks run in dependency order and stop at the first layer that makes
    later layers meaningless: a package whose bytes do not match its own
    checksums cannot be meaningfully classified, so piling downstream semantic
    errors on top of that would be noise rather than diagnosis.
    """

    if not isinstance(match_config, dict):
        raise PackageLintError("match_config must be a dict")

    package_dir = Path(package_dir)
    findings = []

    # -- layer 1: transport integrity ------------------------------------
    transport = verify_transport_integrity(package_dir)

    if not transport["ok"]:
        for error in transport["errors"]:
            findings.append(
                _finding(
                    code="transport_integrity",
                    severity="error",
                    detail=error,
                    fix=(
                        f"Regenerate {SHA256SUMS_NAME} after all evidence files "
                        "are final. It must cover every file except itself, use "
                        "LF endings, end with LF, and list paths sorted. "
                        f"Required files: {sorted(REQUIRED_PACKAGE_FILES)}"
                    ),
                )
            )

        return {
            "ok": False,
            "predicted_intake": WOULD_BE_QUARANTINED,
            "stopped_at": "transport_integrity",
            "findings": findings,
            "transport": transport,
        }

    # -- layer 2: required documents parse --------------------------------
    documents = {}

    for name in sorted(REQUIRED_PACKAGE_FILES):
        try:
            documents[name] = _load_json(package_dir / name)
        except Exception as exc:
            findings.append(
                _finding(
                    code="document_unreadable",
                    severity="error",
                    detail=f"{name}: {exc}",
                    fix=(
                        f"Emit {name} as UTF-8 JSON with LF line endings. "
                        "CRLF is rejected for evidence portability."
                    ),
                )
            )

    if findings:
        return {
            "ok": False,
            "predicted_intake": WOULD_BE_QUARANTINED,
            "stopped_at": "document_parse",
            "findings": findings,
            "transport": transport,
        }

    # -- layer 3: artifact binding ----------------------------------------
    binding = verify_artifact_binding(
        match_config,
        documents["artifact_binding.json"],
    )

    for error in binding["errors"]:
        findings.append(
            _finding(
                code="artifact_binding",
                severity="error",
                detail=error,
                fix=(
                    "artifact_binding.json must carry exactly sidecar_version, "
                    "run_id, config_identity_sha256, actual_candidate_sha256 "
                    "and actual_opponent_sha256, all lowercase hex."
                ),
            )
        )

    for mismatch in binding["mismatches"]:
        findings.append(
            _finding(
                code="artifact_binding_mismatch",
                severity="error",
                detail=mismatch,
                fix=(
                    f"Field {mismatch['field']!r} must be computed from the "
                    "artifact the engine actually loaded. A mismatch means the "
                    "run did not use the configured artifact -- investigate the "
                    "run before changing the sidecar."
                ),
            )
        )

    # -- layer 4: evidence references -------------------------------------
    raw_result = documents["raw_result.json"]
    present = set(transport["files"])

    if isinstance(raw_result, dict):
        for field in ("engine_stdout_log_ref", "pgn_ref"):
            value = raw_result.get(field)

            try:
                neutral = require_neutral_path(value)
            except Exception as exc:
                findings.append(
                    _finding(
                        code="evidence_reference_invalid",
                        severity="error",
                        detail=f"{field}: {exc}",
                        fix=(
                            f"{field} must be a relative forward-slash path "
                            "inside the package, e.g. 'games.pgn'."
                        ),
                    )
                )
                continue

            if neutral not in present:
                findings.append(
                    _finding(
                        code="evidence_reference_missing",
                        severity="error",
                        detail=f"{field} points at {neutral!r}, not in the package",
                        fix=(
                            f"Include {neutral} in the package and in "
                            f"{SHA256SUMS_NAME}."
                        ),
                    )
                )

    # -- layer 5: semantic classification ---------------------------------
    try:
        classification = classify_result_bundle(
            match_config=match_config,
            execution_manifest=documents["execution_manifest.json"],
            raw_result=raw_result,
        )
    except Exception as exc:
        findings.append(
            _finding(
                code="classification_exception",
                severity="error",
                detail=str(exc),
                fix="Fix the errors above; the bundle could not be classified.",
            )
        )

        return {
            "ok": False,
            "predicted_intake": WOULD_BE_QUARANTINED,
            "stopped_at": "classification",
            "findings": findings,
            "transport": transport,
            "artifact_binding": binding,
        }

    for reason in classification.get("reason_codes", []):
        findings.append(
            _finding(
                code=reason,
                severity=(
                    "error" if classification["state"] == QUARANTINED else "incomplete"
                ),
                detail=classification.get("detail", reason),
                fix=_FIX_HINTS.get(
                    reason,
                    "See EXECUTION_PACKAGE_CONTRACT.md for this requirement.",
                ),
            )
        )

    predicted = _PREDICTED_FROM_STATE[classification["state"]]

    return {
        "ok": predicted == WOULD_BE_ACCEPTED and not findings,
        "predicted_intake": predicted,
        "stopped_at": None,
        "findings": findings,
        "transport": transport,
        "artifact_binding": binding,
        "classification": classification,
    }
