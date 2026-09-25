"""MatchConfigV1 generation and publication for the research evidence
plane (Phase 2).

Builds on the frozen Phase 0 schemas/identity helpers and the committed
Phase 1 artifact-sealing module. Neither is modified here.

Locked design invariants:
  - Artifact content identity (v1): sha256 of the canonical form of
    {"artifact_content_hash_version": "v1", "files": sorted(files, key=path)}
    -- import metadata (imported_at/imported_by/source_description/
    artifact_id/artifact_type) never participates.
  - engine/cli identity is asserted (pinned_id + expected_sha256), never
    looked up or verified locally -- that belongs to a later
    ExecutionManifest/result-ingestion boundary.
  - identity_sha256 uses the already-frozen Phase 0 contract exactly:
    exclude_fields = {"created_at"} -- created_by remains hash-relevant.
  - publish_config() is strictly create-only, built in a sibling staging
    directory and published via a single atomic rename, exactly like
    Phase 1's seal(). It never trusts construction-time correctness: it
    independently re-validates the schema, re-derives/verifies config_id
    against '<matchup_key>__runNNN', and recomputes identity_sha256.
  - time_control.tc rejects only control characters, never ordinary chess
    time-control punctuation ('/', ':', '+', '.', etc).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from medium_pc_audit import artifact_import, identity
from medium_pc_audit.schemas import registry

_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:")
_RUN_ID_SUFFIX_RE = re.compile(r"^__run(\d+)$")

MATCHUP_OPPONENT_ARTIFACT_TYPE = {
    "vs_teacher": "teacher_anchor",
    "vs_champion_v1": "champion_v1_anchor",
    "vs_prev_audit_champion": "audit_champion",
}

# The explicit, restrictive v1 set of artifact_type values allowed to serve
# as the audited candidate. Deliberately NOT "anything registered as an
# artifact" -- a candidate is always the specific champion under audit.
ALLOWED_CANDIDATE_ARTIFACT_TYPES = frozenset({"audit_champion"})

OPENING_SUITE_ARTIFACT_TYPE = "opening_suite"

_TIME_CONTROL_KEYS = ("nodes", "movetime_ms", "tc")


class ConfigGenError(ValueError):
    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class UnsealedArtifactReference(ConfigGenError):
    pass


class InvalidMatchupRole(ConfigGenError):
    pass


class InvalidTimeControl(ConfigGenError):
    pass


class InvalidSprtParameters(ConfigGenError):
    pass


class InvalidUciOptions(ConfigGenError):
    pass


class InvalidConfigContent(ConfigGenError):
    pass


class ConfigAlreadyExists(ConfigGenError):
    pass


class ConfigIdentityMismatch(ConfigGenError):
    pass


def compute_artifact_content_hash(manifest: dict) -> str:
    """Frozen v1 artifact-content identity: sha256 of the canonical form of
    {"artifact_content_hash_version": "v1", "files": sorted(files, key=path)}.

    Import metadata (imported_at/imported_by/source_description/artifact_id/
    artifact_type) never participates -- only the sealed content tree does.
    """
    payload = {
        "artifact_content_hash_version": "v1",
        "files": sorted(manifest["files"], key=lambda entry: entry["path"]),
    }
    return identity.compute_identity_sha256(payload)


def require_sealed_artifact(artifact_dir) -> dict:
    """Verify artifact_dir is a currently-valid sealed Phase 1 artifact and
    return its manifest. Never trusts a manifest that hasn't just been
    re-verified from disk.
    """
    artifact_dir = Path(artifact_dir)
    try:
        artifact_import.verify_seal(artifact_dir)
    except artifact_import.ArtifactIntegrityError as exc:
        raise UnsealedArtifactReference(
            [f"referenced artifact at {artifact_dir} failed verify_seal(): {e}" for e in exc.errors]
        ) from exc

    manifest_path = artifact_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _reject_path_like_string(value: str, *, field_name: str) -> None:
    if "\\" in value or "/" in value or _DRIVE_LETTER_RE.match(value):
        raise InvalidConfigContent(
            f"{field_name} must not look like a filesystem path, got {value!r}"
        )


def _reject_control_characters(value: str, *, field_name: str) -> None:
    """Rejects NUL/CR/LF/other C0/DEL bytes only -- never ordinary
    punctuation. Used for time_control.tc, which may legitimately contain
    '/', ':', '+', '.', etc as valid chess time-control syntax.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise InvalidTimeControl(f"{field_name} must not contain control characters, got {value!r}")


def _validate_time_control(time_control: dict) -> dict:
    if not isinstance(time_control, dict):
        raise InvalidTimeControl(f"time_control must be an object, got {type(time_control).__name__}")

    unexpected = sorted(k for k in time_control if k not in _TIME_CONTROL_KEYS)
    if unexpected:
        raise InvalidTimeControl(f"time_control contains unsupported keys: {unexpected}")

    if len(time_control) != 1:
        raise InvalidTimeControl(
            f"time_control must contain exactly one key from {_TIME_CONTROL_KEYS}, got {sorted(time_control)}"
        )

    key = next(iter(time_control))
    value = time_control[key]

    if key in ("nodes", "movetime_ms"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidTimeControl(f"time_control.{key} must be a plain int, got {value!r}")
        if value < 1:
            raise InvalidTimeControl(f"time_control.{key} must be >= 1, got {value}")
    elif key == "tc":
        if not isinstance(value, str) or not value:
            raise InvalidTimeControl(f"time_control.tc must be a non-empty string, got {value!r}")
        # Deliberately NOT a path-like-string check: '/', ':', '+', '.' are
        # legitimate chess time-control syntax (e.g. "40/9000:10+0.1").
        # Only control characters (NUL/CR/LF/etc) are rejected.
        _reject_control_characters(value, field_name="time_control.tc")

    return time_control


def _validate_uci_options(uci_options: dict) -> dict:
    if not isinstance(uci_options, dict):
        raise InvalidUciOptions(f"uci_options must be an object, got {type(uci_options).__name__}")

    for key, value in uci_options.items():
        if not isinstance(key, str) or not key:
            raise InvalidUciOptions(f"uci_options keys must be non-empty strings, got {key!r}")

        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise InvalidUciOptions(f"uci_options[{key!r}] must be finite, got {value!r}")
            continue
        if isinstance(value, int):
            continue
        if isinstance(value, str):
            _reject_path_like_string(value, field_name=f"uci_options[{key!r}]")
            continue

        raise InvalidUciOptions(
            f"uci_options[{key!r}] has unsupported value type {type(value).__name__} "
            f"(only bool/int/float/str are allowed)"
        )

    return uci_options


def _validate_sprt_params(sprt: dict) -> dict:
    if not isinstance(sprt, dict):
        raise InvalidSprtParameters(f"sprt must be an object, got {type(sprt).__name__}")

    required = ("elo0", "elo1", "alpha", "beta", "max_games")
    missing = [k for k in required if k not in sprt]
    if missing:
        raise InvalidSprtParameters(f"sprt missing required fields: {missing}")

    for name in ("elo0", "elo1", "alpha", "beta"):
        value = sprt[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidSprtParameters(f"sprt.{name} must be a number, got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidSprtParameters(f"sprt.{name} must be finite, got {value!r}")

    elo0, elo1 = sprt["elo0"], sprt["elo1"]
    alpha, beta = sprt["alpha"], sprt["beta"]

    if not (elo0 < elo1):
        raise InvalidSprtParameters(f"sprt.elo0 must be < sprt.elo1, got elo0={elo0} elo1={elo1}")
    if not (0.0 < alpha < 1.0):
        raise InvalidSprtParameters(f"sprt.alpha must be in (0, 1), got {alpha}")
    if not (0.0 < beta < 1.0):
        raise InvalidSprtParameters(f"sprt.beta must be in (0, 1), got {beta}")

    max_games = sprt["max_games"]
    if isinstance(max_games, bool) or not isinstance(max_games, int) or max_games < 1:
        raise InvalidSprtParameters(f"sprt.max_games must be an int >= 1, got {max_games!r}")

    return sprt


def _validate_matchup_roles(
    *,
    matchup_type: str,
    candidate_manifest: dict,
    opponent_manifest: dict,
    opening_suite_manifest: dict,
) -> None:
    if candidate_manifest["artifact_type"] not in ALLOWED_CANDIDATE_ARTIFACT_TYPES:
        raise InvalidMatchupRole(
            f"candidate artifact_type {candidate_manifest['artifact_type']!r} is not an allowed "
            f"v1 candidate type {sorted(ALLOWED_CANDIDATE_ARTIFACT_TYPES)!r}"
        )

    expected_opponent_type = MATCHUP_OPPONENT_ARTIFACT_TYPE[matchup_type]
    if opponent_manifest["artifact_type"] != expected_opponent_type:
        raise InvalidMatchupRole(
            f"matchup_type {matchup_type!r} requires opponent artifact_type "
            f"{expected_opponent_type!r}, got {opponent_manifest['artifact_type']!r}"
        )

    if opening_suite_manifest["artifact_type"] != OPENING_SUITE_ARTIFACT_TYPE:
        raise InvalidMatchupRole(
            f"opening suite reference must have artifact_type {OPENING_SUITE_ARTIFACT_TYPE!r}, "
            f"got {opening_suite_manifest['artifact_type']!r}"
        )


def compute_config_identity_sha256(config: dict) -> str:
    """identity_sha256 per the already-frozen Phase 0 contract: excludes
    only 'created_at'; every other top-level field (including config_id and
    created_by) is hash-relevant. config's own 'identity_sha256' key (if
    present) is stripped before hashing to avoid self-reference.
    """
    without_identity = {k: v for k, v in config.items() if k != "identity_sha256"}
    return identity.compute_identity_sha256(without_identity, exclude_fields=frozenset({"created_at"}))


def _verify_config_id_semantics(config: dict) -> int:
    """Independently parse and verify config_id against the canonical
    '<matchup_key>__runNNN' form, without trusting that the config was
    built via build_match_config(). Returns the parsed attempt integer.

    This does not touch the frozen Phase 0 schema -- config_id/matchup_key
    are already schema-validated as safe identifier strings by the time
    this runs; this adds the *semantic* relationship check between them.
    """
    matchup_key = config["matchup_key"]
    config_id = config["config_id"]

    if not config_id.startswith(matchup_key):
        raise ConfigIdentityMismatch(
            f"config_id {config_id!r} does not start with matchup_key {matchup_key!r}"
        )

    suffix = config_id[len(matchup_key):]
    match = _RUN_ID_SUFFIX_RE.match(suffix)
    if match is None:
        raise ConfigIdentityMismatch(
            f"config_id {config_id!r} does not match the canonical form "
            f"'<matchup_key>__runNNN' (suffix was {suffix!r})"
        )

    attempt = int(match.group(1))
    if attempt < 1:
        raise ConfigIdentityMismatch(f"config_id {config_id!r} implies attempt {attempt} < 1")

    expected_run_id = identity.compute_run_id(matchup_key, attempt)
    if expected_run_id != config_id:
        raise ConfigIdentityMismatch(
            f"config_id {config_id!r} does not match the canonical run_id "
            f"{expected_run_id!r} for matchup_key {matchup_key!r} attempt {attempt}"
        )

    return attempt


def build_match_config(
    *,
    audit_id: str,
    matchup_type: str,
    candidate_artifact_dir,
    opponent_artifact_dir,
    engine_pinned_id: str,
    engine_expected_sha256: str,
    cli_pinned_id: str,
    cli_expected_sha256: str,
    opening_suite_artifact_dir,
    uci_options: dict,
    time_control: dict,
    threads: int,
    hash_mb: int,
    concurrency: int,
    repeat: bool,
    seed: int,
    sprt: dict,
    attempt: int,
    created_by: str,
    now_utc: str,
) -> dict:
    """Build a validated, schema-conformant MatchConfigV1 dict.

    Every referenced artifact (candidate/opponent/opening_suite) must
    already be sealed and pass Phase 1's verify_seal() -- this function
    never builds a config referencing broken or unverified evidence.
    """
    if matchup_type not in MATCHUP_OPPONENT_ARTIFACT_TYPE:
        raise InvalidMatchupRole(
            f"matchup_type must be one of {sorted(MATCHUP_OPPONENT_ARTIFACT_TYPE)}, got {matchup_type!r}"
        )

    _validate_time_control(time_control)
    _validate_uci_options(uci_options)
    _validate_sprt_params(sprt)
    _reject_path_like_string(created_by, field_name="created_by")

    candidate_manifest = require_sealed_artifact(candidate_artifact_dir)
    opponent_manifest = require_sealed_artifact(opponent_artifact_dir)
    opening_suite_manifest = require_sealed_artifact(opening_suite_artifact_dir)

    _validate_matchup_roles(
        matchup_type=matchup_type,
        candidate_manifest=candidate_manifest,
        opponent_manifest=opponent_manifest,
        opening_suite_manifest=opening_suite_manifest,
    )

    opponent_id = opponent_manifest["artifact_id"]
    matchup_key = identity.compute_matchup_key(audit_id, opponent_id)
    run_id = identity.compute_run_id(matchup_key, attempt)

    config = {
        "schema_version": "v1",
        "config_id": run_id,
        "audit_id": audit_id,
        "matchup_key": matchup_key,
        "matchup_type": matchup_type,
        "candidate": {
            "artifact_id": candidate_manifest["artifact_id"],
            "sha256": compute_artifact_content_hash(candidate_manifest),
        },
        "opponent": {
            "artifact_id": opponent_manifest["artifact_id"],
            "sha256": compute_artifact_content_hash(opponent_manifest),
        },
        "engine": {"pinned_id": engine_pinned_id, "expected_sha256": engine_expected_sha256},
        "cli": {"pinned_id": cli_pinned_id, "expected_sha256": cli_expected_sha256},
        "uci_options": uci_options,
        "time_control": time_control,
        "threads": threads,
        "hash_mb": hash_mb,
        "concurrency": concurrency,
        "opening_suite": {
            "artifact_id": opening_suite_manifest["artifact_id"],
            "sha256": compute_artifact_content_hash(opening_suite_manifest),
        },
        "repeat": repeat,
        "seed": seed,
        "sprt": sprt,
        "created_at": now_utc,
        "created_by": created_by,
    }

    config["identity_sha256"] = compute_config_identity_sha256(config)

    registry.validate("match_config", "v1", config)

    return config


def _serialize_config_bytes(config: dict) -> bytes:
    text = json.dumps(
        config,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _render_config_sha256sums(config_bytes: bytes) -> bytes:
    digest = hashlib.sha256(config_bytes).hexdigest()
    return f"{digest}  match_config.json\n".encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def publish_config(config: dict, *, configs_root) -> Path:
    """Publish config to configs_root/<matchup_key>/<config_id>/, strictly
    create-only.

    Never trusts construction-time correctness (this is a public function
    that may be called on a manually constructed config): independently
    re-validates the schema, verifies config_id against the canonical
    '<matchup_key>__runNNN' form, and recomputes/verifies identity_sha256,
    all before touching disk. A failed publish never leaves a final
    directory that could be mistaken for a real config package.
    """
    registry.validate("match_config", "v1", config)

    _verify_config_id_semantics(config)

    recomputed = compute_config_identity_sha256(config)
    if recomputed != config["identity_sha256"]:
        raise ConfigIdentityMismatch(
            f"identity_sha256 mismatch: stored={config['identity_sha256']!r} recomputed={recomputed!r}"
        )

    configs_root = Path(configs_root)
    matchup_key = config["matchup_key"]
    run_id = config["config_id"]

    type_root = configs_root / matchup_key
    final_dir = type_root / run_id

    if final_dir.exists():
        raise ConfigAlreadyExists(f"final config directory already exists: {final_dir}")

    type_root.mkdir(parents=True, exist_ok=True)
    staging_dir = type_root / f".staging-{run_id}-{uuid.uuid4().hex}"
    published = False

    try:
        staging_dir.mkdir(parents=True)
        config_bytes = _serialize_config_bytes(config)
        _atomic_write_bytes(staging_dir / "match_config.json", config_bytes)
        _atomic_write_bytes(staging_dir / "SHA256SUMS.txt", _render_config_sha256sums(config_bytes))

        if final_dir.exists():
            raise ConfigAlreadyExists(f"final config directory already exists: {final_dir}")

        os.replace(staging_dir, final_dir)
        published = True
        return final_dir
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def generate_and_publish_config(*, configs_root, **build_kwargs):
    """Convenience: build_match_config(**build_kwargs) then publish_config()."""
    config = build_match_config(**build_kwargs)
    final_dir = publish_config(config, configs_root=configs_root)
    return config, final_dir
