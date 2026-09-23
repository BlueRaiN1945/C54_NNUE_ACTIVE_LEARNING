"""Immutable RESULTS_INBOX package ingestion.

A returned execution package is manually transferred to Medium-PC and must
already contain SHA256SUMS.txt created by the execution side.  Medium-PC
copies the package verbatim to a staging directory, re-verifies every byte
against that checksum file, performs provenance/semantic checks, and then
publishes the directory with one rename.

Final raw evidence:

    RESULTS_INBOX/<run_id>/

is create-only and never edited by this module.

Workflow state is kept separately in:

    RESULTS_INBOX/_STATE/<run_id>.jsonl

The sidecar artifact_binding.json closes a frozen-v1 limitation:
ExecutionManifestV1 has no actual candidate/opponent SHA256 fields.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

from medium_pc_audit.config_gen import (
    compute_config_identity_sha256,
)
from medium_pc_audit.inbox_validate import (
    ACCEPTED,
    INCOMPLETE,
    QUARANTINED,
    classify_result_bundle,
)
from medium_pc_audit.paths import (
    require_neutral_path,
    to_neutral_path,
)
from medium_pc_audit.schemas import registry
from medium_pc_audit.state import (
    RECEIVED,
    REJECTED_DUPLICATE,
    append_state_event,
    require_safe_run_id,
    run_id_seen,
)


SHA256SUMS_NAME = "SHA256SUMS.txt"

REQUIRED_PACKAGE_FILES = frozenset(
    {
        "execution_manifest.json",
        "raw_result.json",
        "artifact_binding.json",
    }
)

ARTIFACT_BINDING_KEYS = frozenset(
    {
        "sidecar_version",
        "run_id",
        "config_identity_sha256",
        "actual_candidate_sha256",
        "actual_opponent_sha256",
    }
)

_SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)

_CHECKSUM_LINE_RE = re.compile(
    r"^([0-9a-f]{64})  (.+)$"
)


class ResultPackageError(ValueError):
    """Programming/API misuse rather than routine intake rejection."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def _enumerate_payload_files(
    package_dir: Path,
) -> tuple[list[str], list[str]]:
    """Return neutral file paths plus structural errors."""

    files = []
    errors = []

    if package_dir.is_symlink():
        errors.append(
            "package directory itself must not be a symlink"
        )
        return files, errors

    try:
        entries = list(
            package_dir.rglob("*")
        )
    except OSError as exc:
        return files, [
            f"cannot enumerate package: {exc}"
        ]

    for entry in entries:

        if entry.is_symlink():
            try:
                neutral = to_neutral_path(
                    entry,
                    package_dir,
                )
            except Exception:
                neutral = str(entry)

            errors.append(
                f"symlink not allowed: {neutral}"
            )
            continue

        if entry.is_file():
            try:
                neutral = to_neutral_path(
                    entry,
                    package_dir,
                )
            except Exception as exc:
                errors.append(
                    f"invalid package path {entry}: {exc}"
                )
                continue

            if neutral != SHA256SUMS_NAME:
                files.append(neutral)

    files.sort()

    return files, errors


def _parse_sha256sums(
    checksum_path: Path,
) -> tuple[dict[str, str], list[str]]:

    errors = []

    try:
        data = checksum_path.read_bytes()
    except OSError as exc:
        return {}, [
            f"cannot read {SHA256SUMS_NAME}: {exc}"
        ]

    if not data:
        return {}, [
            f"{SHA256SUMS_NAME} is empty"
        ]

    if b"\r" in data:
        errors.append(
            f"{SHA256SUMS_NAME} must use LF, not CRLF"
        )

    if not data.endswith(b"\n"):
        errors.append(
            f"{SHA256SUMS_NAME} must end with LF"
        )

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {}, errors + [
            f"{SHA256SUMS_NAME} is not valid UTF-8"
        ]

    lines = text.splitlines()

    mapping = {}
    paths_in_order = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        match = _CHECKSUM_LINE_RE.fullmatch(
            line
        )

        if match is None:
            errors.append(
                f"invalid checksum line {line_number}"
            )
            continue

        digest, neutral = match.groups()

        try:
            neutral = require_neutral_path(
                neutral
            )
        except Exception as exc:
            errors.append(
                f"invalid checksum path at line {line_number}: {exc}"
            )
            continue

        if neutral == SHA256SUMS_NAME:
            errors.append(
                f"{SHA256SUMS_NAME} must not hash itself"
            )
            continue

        if neutral in mapping:
            errors.append(
                f"duplicate checksum path: {neutral}"
            )
            continue

        mapping[neutral] = digest
        paths_in_order.append(neutral)

    if paths_in_order != sorted(
        paths_in_order
    ):
        errors.append(
            f"{SHA256SUMS_NAME} paths must be sorted"
        )

    return mapping, errors


def verify_transport_integrity(
    package_dir,
) -> dict:
    """Verify host-provided checksum coverage and bytes."""

    package_dir = Path(
        package_dir
    )

    errors = []

    if (
        not package_dir.exists()
        or not package_dir.is_dir()
    ):
        return {
            "ok": False,
            "errors": [
                f"package directory not found: {package_dir}"
            ],
            "files": [],
        }

    checksum_path = (
        package_dir
        / SHA256SUMS_NAME
    )

    if (
        not checksum_path.exists()
        or not checksum_path.is_file()
    ):
        return {
            "ok": False,
            "errors": [
                f"missing {SHA256SUMS_NAME}"
            ],
            "files": [],
        }

    actual_files, structure_errors = (
        _enumerate_payload_files(
            package_dir
        )
    )

    errors.extend(
        structure_errors
    )

    checksums, checksum_errors = (
        _parse_sha256sums(
            checksum_path
        )
    )

    errors.extend(
        checksum_errors
    )

    checksum_files = sorted(
        checksums
    )

    missing_from_checksum = sorted(
        set(actual_files)
        - set(checksum_files)
    )

    missing_from_package = sorted(
        set(checksum_files)
        - set(actual_files)
    )

    if missing_from_checksum:
        errors.append(
            "files missing from checksum manifest: "
            + ", ".join(
                missing_from_checksum
            )
        )

    if missing_from_package:
        errors.append(
            "checksum entries missing from package: "
            + ", ".join(
                missing_from_package
            )
        )

    missing_required = sorted(
        REQUIRED_PACKAGE_FILES
        - set(actual_files)
    )

    if missing_required:
        errors.append(
            "missing required package files: "
            + ", ".join(
                missing_required
            )
        )

    if not errors:
        for neutral in actual_files:
            path = (
                package_dir
                / Path(
                    *neutral.split("/")
                )
            )

            try:
                actual_sha = _sha256_file(
                    path
                )
            except OSError as exc:
                errors.append(
                    f"cannot hash {neutral}: {exc}"
                )
                continue

            expected_sha = checksums[
                neutral
            ]

            if actual_sha != expected_sha:
                errors.append(
                    f"SHA256 mismatch for {neutral}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

    return {
        "ok": not errors,
        "errors": errors,
        "files": actual_files,
    }


def verify_artifact_binding(
    match_config: dict,
    binding: dict,
) -> dict:
    """Verify the candidate/opponent sidecar against MatchConfigV1."""

    errors = []
    mismatches = []

    try:
        registry.validate(
            "match_config",
            "v1",
            match_config,
        )
    except Exception as exc:
        return {
            "ok": False,
            "errors": [
                f"invalid match config: {exc}"
            ],
            "mismatches": [],
        }

    try:
        computed_identity = (
            compute_config_identity_sha256(
                match_config
            )
        )
    except Exception as exc:
        return {
            "ok": False,
            "errors": [
                f"cannot recompute config identity: {exc}"
            ],
            "mismatches": [],
        }

    if (
        match_config["identity_sha256"]
        != computed_identity
    ):
        errors.append(
            "MatchConfigV1 identity_sha256 does not match recomputed identity"
        )

    if not isinstance(binding, dict):
        return {
            "ok": False,
            "errors": errors + [
                "artifact_binding.json must contain an object"
            ],
            "mismatches": [],
        }

    actual_keys = set(
        binding
    )

    if actual_keys != ARTIFACT_BINDING_KEYS:
        missing = sorted(
            ARTIFACT_BINDING_KEYS
            - actual_keys
        )

        extra = sorted(
            actual_keys
            - ARTIFACT_BINDING_KEYS
        )

        errors.append(
            f"artifact binding fields differ; missing={missing}, extra={extra}"
        )

        return {
            "ok": False,
            "errors": errors,
            "mismatches": [],
        }

    if binding["sidecar_version"] != "v1":
        errors.append(
            "artifact binding sidecar_version must be 'v1'"
        )

    for field in (
        "config_identity_sha256",
        "actual_candidate_sha256",
        "actual_opponent_sha256",
    ):
        value = binding[field]

        if (
            not isinstance(value, str)
            or not _SHA256_RE.fullmatch(
                value
            )
        ):
            errors.append(
                f"{field} must be lowercase SHA256 hex"
            )

    expected = {
        "run_id": match_config[
            "config_id"
        ],
        "config_identity_sha256": (
            computed_identity
        ),
        "actual_candidate_sha256": (
            match_config[
                "candidate"
            ]["sha256"]
        ),
        "actual_opponent_sha256": (
            match_config[
                "opponent"
            ]["sha256"]
        ),
    }

    for field, wanted in expected.items():
        actual = binding[field]

        if actual != wanted:
            mismatches.append(
                {
                    "field": field,
                    "expected": wanted,
                    "actual": actual,
                }
            )

    return {
        "ok": (
            not errors
            and not mismatches
        ),
        "errors": errors,
        "mismatches": mismatches,
    }


def _load_json(
    path: Path,
):
    raw = path.read_bytes()

    if b"\r" in raw:
        # JSON itself may technically use CRLF, but evidence portability is
        # deliberately stricter in this control plane.
        raise ValueError(
            f"{path.name} must use LF line endings"
        )

    return json.loads(
        raw.decode("utf-8")
    )


def _inspect_staged_package(
    staging: Path,
    *,
    match_config: dict,
    transport: dict,
) -> dict:
    """Semantic inspection after byte-level transport verification."""

    try:
        execution_manifest = _load_json(
            staging
            / "execution_manifest.json"
        )

        raw_result = _load_json(
            staging
            / "raw_result.json"
        )

        binding = _load_json(
            staging
            / "artifact_binding.json"
        )

    except Exception as exc:
        return {
            "state": QUARANTINED,
            "reason_codes": [
                "required_json_unreadable"
            ],
            "detail": str(exc),
            "artifact_binding": None,
            "classification": None,
        }

    binding_result = (
        verify_artifact_binding(
            match_config,
            binding,
        )
    )

    if not binding_result["ok"]:
        return {
            "state": QUARANTINED,
            "reason_codes": [
                "artifact_binding_mismatch"
            ],
            "artifact_binding": binding_result,
            "classification": None,
        }

    reference_errors = []

    if isinstance(raw_result, dict):
        for field in (
            "engine_stdout_log_ref",
            "pgn_ref",
        ):
            value = raw_result.get(
                field
            )

            try:
                neutral = require_neutral_path(
                    value
                )
            except Exception as exc:
                reference_errors.append(
                    f"{field}: {exc}"
                )
                continue

            if neutral not in set(
                transport["files"]
            ):
                reference_errors.append(
                    f"{field} references missing or unhashed file: {neutral}"
                )

    if reference_errors:
        return {
            "state": QUARANTINED,
            "reason_codes": [
                "evidence_reference_invalid"
            ],
            "detail": reference_errors,
            "artifact_binding": binding_result,
            "classification": None,
        }

    try:
        classification = (
            classify_result_bundle(
                match_config=match_config,
                execution_manifest=execution_manifest,
                raw_result=raw_result,
            )
        )
    except Exception as exc:
        return {
            "state": QUARANTINED,
            "reason_codes": [
                "classification_exception"
            ],
            "detail": str(exc),
            "artifact_binding": binding_result,
            "classification": None,
        }

    return {
        "state": classification[
            "state"
        ],
        "reason_codes": list(
            classification.get(
                "reason_codes",
                [],
            )
        ),
        "artifact_binding": binding_result,
        "classification": classification,
    }


def ingest_result_package(
    source_dir,
    inbox_root,
    *,
    match_config: dict,
    received_at: str,
) -> dict:
    """Copy, verify, classify, and atomically publish one run package."""

    run_id = require_safe_run_id(
        match_config.get(
            "config_id"
        )
    )

    source_dir = Path(
        source_dir
    ).resolve()

    inbox_root = Path(
        inbox_root
    ).resolve()

    inbox_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dir = (
        inbox_root
        / run_id
    )

    if run_id_seen(
        inbox_root,
        run_id,
    ):
        append_state_event(
            inbox_root,
            run_id=run_id,
            state=REJECTED_DUPLICATE,
            recorded_at=received_at,
            reason_codes=[
                "run_id_already_seen"
            ],
        )

        return {
            "state": REJECTED_DUPLICATE,
            "reason_codes": [
                "run_id_already_seen"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": (
                str(final_dir)
                if final_dir.exists()
                else None
            ),
        }

    if (
        source_dir == inbox_root
        or inbox_root in source_dir.parents
        or source_dir in inbox_root.parents
    ):
        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "source_destination_overlap"
            ],
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "source_destination_overlap"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
        }

    if (
        not source_dir.exists()
        or not source_dir.is_dir()
    ):
        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "source_package_missing"
            ],
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "source_package_missing"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
        }

    staging = (
        inbox_root
        / (
            ".staging-"
            + run_id
            + "-"
            + uuid.uuid4().hex
        )
    )

    try:
        shutil.copytree(
            source_dir,
            staging,
        )

    except Exception as exc:
        if staging.exists():
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "package_copy_failed"
            ],
            details=str(exc),
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "package_copy_failed"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
        }

    transport = (
        verify_transport_integrity(
            staging
        )
    )

    if not transport["ok"]:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )

        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "transport_integrity_failed"
            ],
            details=transport["errors"],
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "transport_integrity_failed"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
            "transport": transport,
        }

    inspection = (
        _inspect_staged_package(
            staging,
            match_config=match_config,
            transport=transport,
        )
    )

    # Recheck immediately before publication so a copied/staged file cannot
    # be silently modified between inspection and the final rename.
    final_transport = (
        verify_transport_integrity(
            staging
        )
    )

    if not final_transport["ok"]:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )

        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "staging_integrity_changed"
            ],
            details=final_transport[
                "errors"
            ],
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "staging_integrity_changed"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
        }

    try:
        staging.rename(
            final_dir
        )

    except OSError as exc:

        shutil.rmtree(
            staging,
            ignore_errors=True,
        )

        if final_dir.exists():
            append_state_event(
                inbox_root,
                run_id=run_id,
                state=REJECTED_DUPLICATE,
                recorded_at=received_at,
                reason_codes=[
                    "publication_race_duplicate"
                ],
            )

            return {
                "state": REJECTED_DUPLICATE,
                "reason_codes": [
                    "publication_race_duplicate"
                ],
                "published": False,
                "run_id": run_id,
                "package_dir": str(
                    final_dir
                ),
            }

        append_state_event(
            inbox_root,
            run_id=run_id,
            state=QUARANTINED,
            recorded_at=received_at,
            reason_codes=[
                "publication_failed"
            ],
            details=str(exc),
        )

        return {
            "state": QUARANTINED,
            "reason_codes": [
                "publication_failed"
            ],
            "published": False,
            "run_id": run_id,
            "package_dir": None,
        }

    append_state_event(
        inbox_root,
        run_id=run_id,
        state=RECEIVED,
        recorded_at=received_at,
        reason_codes=[],
    )

    append_state_event(
        inbox_root,
        run_id=run_id,
        state=inspection[
            "state"
        ],
        recorded_at=received_at,
        reason_codes=inspection[
            "reason_codes"
        ],
        details={
            "artifact_binding": inspection.get(
                "artifact_binding"
            ),
            "classification": inspection.get(
                "classification"
            ),
            "detail": inspection.get(
                "detail"
            ),
        },
    )

    return {
        "state": inspection[
            "state"
        ],
        "reason_codes": inspection[
            "reason_codes"
        ],
        "published": True,
        "run_id": run_id,
        "package_dir": str(
            final_dir
        ),
        "transport": final_transport,
        "artifact_binding": inspection.get(
            "artifact_binding"
        ),
        "classification": inspection.get(
            "classification"
        ),
    }


def verify_published_result(
    inbox_root,
    run_id: str,
) -> dict:
    """Re-hash immutable raw evidence after publication."""

    run_id = require_safe_run_id(
        run_id
    )

    package_dir = (
        Path(inbox_root)
        / run_id
    )

    return verify_transport_integrity(
        package_dir
    )
