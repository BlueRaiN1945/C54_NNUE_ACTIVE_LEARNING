"""Artifact import / immutable artifact sealing for the research evidence
plane (Phase 1).

Builds on the frozen Phase 0 schemas (medium_pc_audit.schemas.registry) and
the Phase 1 OS-neutral path helpers. This module never modifies Phase 0
files.

Locked design invariants:
  - seal() is strictly create-only: it never replaces or modifies an
    existing final artifact directory. Duplicate/idempotent handling lives
    ONLY in import_artifact().
  - The entire artifact package (files/, manifest.json, SHA256SUMS.txt) is
    built in a sibling staging directory and published via a single atomic
    rename -- a failed import never leaves a final directory that could be
    mistaken for sealed evidence.
  - Filesystem read-only permissions are NOT used as an integrity
    mechanism; verify_seal()'s hash recomputation is the sole source of
    truth.
  - verify_seal() binds manifest identity to filesystem location, rejects
    duplicate neutral paths, and enforces the exact top-level package
    layout ({files/, manifest.json, SHA256SUMS.txt} and nothing else).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medium_pc_audit.paths import to_neutral_path
from medium_pc_audit.schemas import registry

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ALLOWED_TOP_LEVEL_ENTRIES = frozenset({"files", "manifest.json", "SHA256SUMS.txt"})


class ArtifactImportError(ValueError):
    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class InvalidArtifactType(ArtifactImportError):
    pass


class UnsafeArtifactId(ArtifactImportError):
    pass


class EmptyArtifactSource(ArtifactImportError):
    pass


class UnsafeSourceEntry(ArtifactImportError):
    pass


class OverlappingArtifactPaths(ArtifactImportError):
    pass


class ArtifactAlreadySealed(ArtifactImportError):
    pass


class StagedCopyMismatch(ArtifactImportError):
    pass


class DuplicateArtifactConflict(ArtifactImportError):
    pass


class ArtifactIntegrityError(ArtifactImportError):
    pass


def require_safe_artifact_id(value: Any) -> str:
    """Validate value as one safe filesystem path component.

    Rejects empty values, '.', '..', path separators, drive-like/colon
    forms, and anything outside [A-Za-z0-9_-] -- i.e. anything capable of
    changing directory structure when joined onto a base path.
    """
    if not isinstance(value, str) or not value:
        raise UnsafeArtifactId(f"artifact_id must be a non-empty string, got {value!r}")
    if value in (".", ".."):
        raise UnsafeArtifactId(f"artifact_id must not be '.' or '..', got {value!r}")
    if "/" in value or "\\" in value:
        raise UnsafeArtifactId(f"artifact_id must not contain path separators, got {value!r}")
    if ":" in value:
        raise UnsafeArtifactId(f"artifact_id must not contain ':' (drive-like form), got {value!r}")
    if not _SAFE_COMPONENT_RE.fullmatch(value):
        raise UnsafeArtifactId(f"artifact_id must match {_SAFE_COMPONENT_RE.pattern!r}, got {value!r}")
    return value


def validate_artifact_type(artifact_type: Any) -> str:
    """Validate artifact_type against the frozen Phase 0 ArtifactManifestV1 enum.

    Reads the enum directly from the committed schema file (never a
    duplicated literal list here), so this always stays in sync with
    Phase 0 without risking drift.
    """
    schema = registry.get_schema("artifact_manifest", "v1")
    allowed = schema["properties"]["artifact_type"]["enum"]
    if not isinstance(artifact_type, str) or artifact_type not in allowed:
        raise InvalidArtifactType(f"artifact_type must be one of {allowed!r}, got {artifact_type!r}")
    return artifact_type


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now_string() -> str:
    """Convenience helper producing a value satisfying the frozen Phase 0
    'utc-iso8601' format. seal()/import_artifact() never call this
    internally -- now_utc is always supplied by the caller, so the core
    logic stays pure and testable with fixed timestamps.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _walk_regular_files(directory: Path) -> list:
    """Enumerate regular files under directory, rejecting symlinks/junctions/specials.

    Returns an empty list if directory does not exist. Callers that require
    non-emptiness (sealing a new artifact) must check that separately --
    this function is also used by verify_seal(), where a missing/empty
    files/ directory is itself the finding to report, not an error to raise
    here.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if directory.is_symlink() or directory.is_junction():
        raise UnsafeSourceEntry(f"source directory itself is a symlink or junction: {directory}")

    results = []
    for dirpath_str, dirnames, filenames in os.walk(directory, followlinks=False):
        dirpath = Path(dirpath_str)

        for dname in list(dirnames):
            dpath = dirpath / dname
            if dpath.is_symlink() or dpath.is_junction():
                raise UnsafeSourceEntry(
                    f"symlink or junction directory not allowed in artifact source: {dpath}"
                )

        for fname in filenames:
            fpath = dirpath / fname
            if fpath.is_symlink():
                raise UnsafeSourceEntry(f"symlink not allowed in artifact source: {fpath}")
            if fpath.is_junction():
                raise UnsafeSourceEntry(f"junction not allowed in artifact source: {fpath}")
            if not fpath.is_file():
                raise UnsafeSourceEntry(f"non-regular file not allowed in artifact source: {fpath}")
            results.append(fpath)

    return results


def enumerate_source_files(source_dir: Path) -> list:
    """Strict enumeration used at seal()-time: source_dir must exist, be a
    real (non-symlink) directory, and contain at least one regular file.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise EmptyArtifactSource(f"source_dir does not exist or is not a directory: {source_dir}")

    files = _walk_regular_files(source_dir)
    if not files:
        raise EmptyArtifactSource(f"source_dir contains no files: {source_dir}")

    files.sort(key=lambda p: to_neutral_path(p, source_dir))
    return files


def _hash_entries(files: list, base: Path) -> list:
    entries = [
        {
            "path": to_neutral_path(p, base),
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
        }
        for p in files
    ]
    entries.sort(key=lambda e: e["path"])
    return entries


def _find_duplicate_paths(files: list) -> list:
    """Return sorted neutral paths that appear more than once in files,
    even if their hash/size are identical -- a duplicate path is always a
    structural defect in a manifest, never a harmless coincidence.
    """
    seen = set()
    duplicates = set()
    for entry in files:
        path = entry["path"]
        if path in seen:
            duplicates.add(path)
        seen.add(path)
    return sorted(duplicates)


def _paths_overlap(a: Path, b: Path) -> bool:
    a = Path(a).resolve()
    b = Path(b).resolve()
    if a == b:
        return True
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return True
    except ValueError:
        pass
    return False


def _diff_entries(expected: list, actual: list) -> list:
    expected_map = {e["path"]: e for e in expected}
    actual_map = {e["path"]: e for e in actual}
    diffs = []
    for path in sorted(set(expected_map) | set(actual_map)):
        e = expected_map.get(path)
        a = actual_map.get(path)
        if e is None:
            diffs.append("added: " + path + " (sha256=" + a["sha256"] + ")")
        elif a is None:
            diffs.append("removed: " + path + " (sha256=" + e["sha256"] + ")")
        elif e["sha256"] != a["sha256"]:
            diffs.append(
                "changed: " + path + " (expected sha256=" + e["sha256"] + ", actual sha256=" + a["sha256"] + ")"
            )
    return diffs


def _render_sha256sums(files: list) -> str:
    ordered = sorted(files, key=lambda e: e["path"])
    return "".join(entry["sha256"] + "  " + entry["path"] + "\n" for entry in ordered)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)


def seal(
    *,
    artifact_type: str,
    artifact_id: str,
    source_dir,
    dest_root,
    source_description: str,
    imported_by: str,
    now_utc: str,
) -> dict:
    """Seal source_dir into dest_root/artifact_type/artifact_id/, strictly create-only.

    Raises ArtifactAlreadySealed if the final artifact directory already
    exists -- this function never replaces or modifies existing sealed
    content. Duplicate-aware (idempotent-or-conflict) handling is the job
    of import_artifact(), not this function.
    """
    validate_artifact_type(artifact_type)
    require_safe_artifact_id(artifact_id)

    source_dir = Path(source_dir)
    dest_root = Path(dest_root)

    type_root = dest_root / artifact_type
    final_dir = type_root / artifact_id

    if final_dir.exists():
        raise ArtifactAlreadySealed(f"final artifact directory already exists: {final_dir}")

    if _paths_overlap(source_dir, dest_root):
        raise OverlappingArtifactPaths(
            f"source_dir and dest_root must not overlap: {source_dir} vs {dest_root}"
        )

    source_files = enumerate_source_files(source_dir)
    source_entries = _hash_entries(source_files, source_dir)

    type_root.mkdir(parents=True, exist_ok=True)
    staging_dir = type_root / f".staging-{artifact_id}-{uuid.uuid4().hex}"
    published = False

    try:
        files_dir = staging_dir / "files"
        for entry in source_entries:
            src_path = source_dir.joinpath(*entry["path"].split("/"))
            dst_path = files_dir.joinpath(*entry["path"].split("/"))
            _atomic_copy(src_path, dst_path)

        staged_files = _walk_regular_files(files_dir)
        staged_entries = _hash_entries(staged_files, files_dir)

        if staged_entries != source_entries:
            raise StagedCopyMismatch(_diff_entries(source_entries, staged_entries))

        manifest = {
            "schema_version": "v1",
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "files": source_entries,
            "source_description": source_description,
            "imported_at": now_utc,
            "imported_by": imported_by,
            "sealed": True,
        }

        try:
            registry.validate("artifact_manifest", "v1", manifest)
        except registry.SchemaValidationError as exc:
            raise ArtifactIntegrityError(
                [f"constructed manifest failed schema validation: {e}" for e in exc.errors]
            ) from exc

        _atomic_write_text(
            staging_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        _atomic_write_text(staging_dir / "SHA256SUMS.txt", _render_sha256sums(source_entries))

        if final_dir.exists():
            raise ArtifactAlreadySealed(f"final artifact directory already exists: {final_dir}")

        os.replace(staging_dir, final_dir)
        published = True
        return manifest
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def verify_seal(artifact_dir) -> None:
    """Recompute every file's hash from disk and compare against
    manifest.json and SHA256SUMS.txt.

    Raises ArtifactIntegrityError listing every mismatch: identity not
    bound to filesystem location (artifact_id/artifact_type/sealed),
    duplicate neutral paths in the manifest, unexpected top-level package
    entries, changed/missing/unexpected-extra files, manifest.json
    corruption or schema violations, and a SHA256SUMS.txt that disagrees
    with manifest.json.
    """
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / "manifest.json"

    if not manifest_path.is_file():
        raise ArtifactIntegrityError([f"manifest.json missing: {manifest_path}"])

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError([f"manifest.json is not valid JSON: {exc}"]) from exc

    try:
        registry.validate("artifact_manifest", "v1", manifest)
    except registry.SchemaValidationError as exc:
        raise ArtifactIntegrityError(
            [f"manifest.json failed schema validation: {e}" for e in exc.errors]
        ) from exc

    errors = []

    # -- package identity must be bound to filesystem location --
    if manifest["artifact_id"] != artifact_dir.name:
        errors.append(
            f"manifest artifact_id {manifest['artifact_id']!r} does not match "
            f"directory name {artifact_dir.name!r}"
        )
    if manifest["artifact_type"] != artifact_dir.parent.name:
        errors.append(
            f"manifest artifact_type {manifest['artifact_type']!r} does not match "
            f"parent directory name {artifact_dir.parent.name!r}"
        )
    if manifest["sealed"] is not True:
        errors.append(f"manifest 'sealed' field must be True, got {manifest['sealed']!r}")

    # -- no duplicate neutral paths, even with identical hash/size --
    duplicate_paths = _find_duplicate_paths(manifest["files"])
    if duplicate_paths:
        errors.append(f"manifest.json contains duplicate file paths: {duplicate_paths}")

    # -- exact top-level package layout --
    try:
        actual_top_level = {p.name for p in artifact_dir.iterdir()}
    except FileNotFoundError:
        actual_top_level = set()
    unexpected_top_level = sorted(actual_top_level - _ALLOWED_TOP_LEVEL_ENTRIES)
    for name in unexpected_top_level:
        errors.append(f"unexpected top-level entry in sealed artifact directory: {name}")

    files_dir = artifact_dir / "files"
    on_disk_files = _walk_regular_files(files_dir)
    on_disk_entries = {
        to_neutral_path(p, files_dir): {
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
        }
        for p in on_disk_files
    }

    manifest_entries = {
        entry["path"]: {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
        for entry in manifest["files"]
    }

    for path, expected in manifest_entries.items():
        actual = on_disk_entries.get(path)
        if actual is None:
            errors.append(f"missing file (in manifest, absent on disk): {path}")
        elif actual["sha256"] != expected["sha256"]:
            errors.append(
                "sha256 mismatch for " + path + ": manifest=" + expected["sha256"] + " actual=" + actual["sha256"]
            )
        elif actual["size_bytes"] != expected["size_bytes"]:
            errors.append(
                "size mismatch for "
                + path
                + ": manifest="
                + str(expected["size_bytes"])
                + " actual="
                + str(actual["size_bytes"])
            )

    for path in sorted(set(on_disk_entries) - set(manifest_entries)):
        errors.append(f"unexpected extra file (on disk, absent from manifest): {path}")

    sums_path = artifact_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        errors.append(f"SHA256SUMS.txt missing: {sums_path}")
    else:
        expected_sums = _render_sha256sums(manifest["files"])
        actual_sums = sums_path.read_text(encoding="utf-8")
        if actual_sums != expected_sums:
            errors.append("SHA256SUMS.txt content does not match manifest.json")

    if errors:
        raise ArtifactIntegrityError(errors)


def import_artifact(
    *,
    artifact_type: str,
    artifact_id: str,
    source_dir,
    dest_root,
    source_description: str,
    imported_by: str,
    now_utc: str,
) -> dict:
    """The only duplicate-aware entry point: seals a new artifact, or, if
    one is already sealed under this artifact_id, treats byte-identical
    content as an idempotent no-op and raises DuplicateArtifactConflict
    otherwise.
    """
    validate_artifact_type(artifact_type)
    require_safe_artifact_id(artifact_id)

    final_dir = Path(dest_root) / artifact_type / artifact_id

    if not final_dir.exists():
        return seal(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            source_dir=source_dir,
            dest_root=dest_root,
            source_description=source_description,
            imported_by=imported_by,
            now_utc=now_utc,
        )

    verify_seal(final_dir)
    existing_manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))

    source_files = enumerate_source_files(Path(source_dir))
    source_entries = _hash_entries(source_files, Path(source_dir))

    if source_entries == existing_manifest["files"]:
        return existing_manifest

    raise DuplicateArtifactConflict(_diff_entries(existing_manifest["files"], source_entries))
