"""Execution-side evidence package producer.

This module does not execute engines or matches. It packages evidence that
already exists on an execution host.

The central rule is deliberate: actual hashes come from observed files, never
from MatchConfig expected values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .pgn_parse import build_raw_result_document, parse_pgn_file
from .result_package import compute_config_identity_sha256
from .schemas import registry


class ExecutionPackageError(ValueError):
    """Raised when execution evidence cannot be packaged safely."""


@dataclass(frozen=True)
class ObservedExecution:
    """Evidence captured by the execution-side wrapper."""

    engine_path: str | Path
    cli_path: str | Path
    candidate_path: str | Path
    opponent_path: str | Path
    opening_suite_path: str | Path
    pgn_path: str | Path
    log_path: str | Path

    argv: tuple
    invocation_cwd: str | Path

    actual_uci_options: dict
    actual_time_control: dict
    actual_threads: int
    actual_hash_mb: int
    actual_concurrency: int
    actual_seed: int

    host_identity_label: str
    executed_at_utc: str
    executed_by: str
    runtime_seconds: float | int | None = None

    def __post_init__(self):
        argv = tuple(self.argv)

        if not argv:
            raise ExecutionPackageError("argv must not be empty")

        for index, value in enumerate(argv):
            if not isinstance(value, str) or not value:
                raise ExecutionPackageError(
                    f"argv[{index}] must be a non-empty string"
                )

        object.__setattr__(self, "argv", argv)

        for name in (
            "host_identity_label",
            "executed_at_utc",
            "executed_by",
        ):
            value = getattr(self, name)

            if not isinstance(value, str) or not value:
                raise ExecutionPackageError(
                    f"{name} must be a non-empty string"
                )

        for name in (
            "actual_uci_options",
            "actual_time_control",
        ):
            value = getattr(self, name)

            if not isinstance(value, dict):
                raise ExecutionPackageError(
                    f"{name} must be a dict"
                )

            object.__setattr__(
                self,
                name,
                copy.deepcopy(value),
            )

        for name in (
            "actual_threads",
            "actual_hash_mb",
            "actual_concurrency",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ExecutionPackageError(
                    f"{name} must be an int >= 1"
                )

        if (
            isinstance(self.actual_seed, bool)
            or not isinstance(self.actual_seed, int)
            or self.actual_seed < 0
        ):
            raise ExecutionPackageError(
                "actual_seed must be an int >= 0"
            )

        if self.runtime_seconds is not None:
            value = self.runtime_seconds

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ExecutionPackageError(
                    "runtime_seconds must be finite and >= 0"
                )


def _sha256_file(path: Path) -> str:
    """Hash a file and reject an obvious change during observation."""

    before = path.stat()
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    after = path.stat()

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ExecutionPackageError(
            f"file changed while hashing: {path}"
        )

    return digest.hexdigest()


def _resolve_file(value, *, name: str) -> Path:
    path = Path(value).expanduser()

    if path.is_symlink():
        raise ExecutionPackageError(
            f"{name} must not be a symlink: {path}"
        )

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExecutionPackageError(
            f"{name} cannot be resolved: {path}: {exc}"
        ) from exc

    if not resolved.is_file():
        raise ExecutionPackageError(
            f"{name} is not a regular file: {resolved}"
        )

    return resolved


def _resolve_directory(value, *, name: str) -> Path:
    path = Path(value).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExecutionPackageError(
            f"{name} cannot be resolved: {path}: {exc}"
        ) from exc

    if not resolved.is_dir():
        raise ExecutionPackageError(
            f"{name} is not a directory: {resolved}"
        )

    return resolved


def _observe_file(value, *, name: str) -> tuple[Path, dict]:
    path = _resolve_file(
        value,
        name=name,
    )

    return (
        path,
        {
            "resolved_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        },
    )


def _write_json(path: Path, obj) -> None:
    text = (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(text)


def _seal_package(package_dir: Path) -> None:
    checksum_path = package_dir / "SHA256SUMS.txt"

    if checksum_path.exists():
        checksum_path.unlink()

    files = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file()
    )

    lines = []

    for path in files:
        relative = "/".join(
            path.relative_to(package_dir).parts
        )

        lines.append(
            f"{_sha256_file(path)}  {relative}\n"
        )

    with checksum_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.writelines(lines)


def build_execution_package(
    output_dir,
    *,
    match_config: dict,
    observation: ObservedExecution,
) -> dict:
    """Build one immutable-ready execution evidence package.

    Expected values in MatchConfig are never substituted for observations.
    Mismatches are intentionally preserved for downstream verification.
    """

    if not isinstance(match_config, dict):
        raise ExecutionPackageError(
            "match_config must be a dict"
        )

    if not isinstance(observation, ObservedExecution):
        raise ExecutionPackageError(
            "observation must be an ObservedExecution"
        )

    registry.validate(
        "match_config",
        "v1",
        match_config,
    )

    computed_identity = compute_config_identity_sha256(
        match_config
    )

    if (
        match_config["identity_sha256"]
        != computed_identity
    ):
        raise ExecutionPackageError(
            "match_config identity_sha256 does not match "
            "its recomputed identity"
        )

    run_id = match_config["config_id"]

    paths = {}
    evidence = {}

    observed_inputs = {
        "engine": observation.engine_path,
        "cli": observation.cli_path,
        "candidate": observation.candidate_path,
        "opponent": observation.opponent_path,
        "opening_suite": observation.opening_suite_path,
        "pgn": observation.pgn_path,
        "log": observation.log_path,
    }

    for name, value in observed_inputs.items():
        path, item = _observe_file(
            value,
            name=f"{name}_path",
        )

        paths[name] = path
        evidence[name] = item

    invocation_cwd = _resolve_directory(
        observation.invocation_cwd,
        name="invocation_cwd",
    )

    output = Path(output_dir)

    if output.exists():
        raise ExecutionPackageError(
            f"output_dir already exists: {output}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.building-",
            dir=output.parent,
        )
    )

    try:
        staged_pgn = stage / "games.pgn"
        staged_log = stage / "match.log"

        shutil.copyfile(
            paths["pgn"],
            staged_pgn,
        )

        shutil.copyfile(
            paths["log"],
            staged_log,
        )

        staged_pgn_sha = _sha256_file(
            staged_pgn
        )

        staged_log_sha = _sha256_file(
            staged_log
        )

        if staged_pgn_sha != evidence["pgn"]["sha256"]:
            raise ExecutionPackageError(
                "games.pgn changed or copied bytes differ"
            )

        if staged_log_sha != evidence["log"]["sha256"]:
            raise ExecutionPackageError(
                "match.log changed or copied bytes differ"
            )

        parse = parse_pgn_file(
            staged_pgn,
            candidate_label=match_config[
                "candidate"
            ]["artifact_id"],
            opponent_label=match_config[
                "opponent"
            ]["artifact_id"],
        )

        wdl = parse.recomputed_wdl

        raw_result = build_raw_result_document(
            run_id=run_id,
            claimed_wins=wdl["wins"],
            claimed_losses=wdl["losses"],
            claimed_draws=wdl["draws"],
            engine_stdout_log_ref="match.log",
            pgn_ref="games.pgn",
            raw_result_games=parse.raw_result_games,
        )

        manifest = {
            "schema_version": "v1",
            "run_id": run_id,
            "actual_engine_sha256": evidence[
                "engine"
            ]["sha256"],
            "actual_cli_sha256": evidence[
                "cli"
            ]["sha256"],
            "actual_uci_options": copy.deepcopy(
                observation.actual_uci_options
            ),
            "actual_time_control": copy.deepcopy(
                observation.actual_time_control
            ),
            "actual_threads": observation.actual_threads,
            "actual_hash_mb": observation.actual_hash_mb,
            "actual_concurrency": (
                observation.actual_concurrency
            ),
            "actual_seed": observation.actual_seed,
            "actual_opening_suite_sha256": evidence[
                "opening_suite"
            ]["sha256"],
            "host_identity_label": (
                observation.host_identity_label
            ),
            "executed_at_utc": (
                observation.executed_at_utc
            ),
            "executed_by": observation.executed_by,
        }

        registry.validate(
            "execution_manifest",
            "v1",
            manifest,
        )

        binding = {
            "sidecar_version": "v1",
            "run_id": run_id,
            "config_identity_sha256": (
                computed_identity
            ),
            "actual_candidate_sha256": evidence[
                "candidate"
            ]["sha256"],
            "actual_opponent_sha256": evidence[
                "opponent"
            ]["sha256"],
        }

        evidence["pgn"]["package_ref"] = (
            "games.pgn"
        )
        evidence["log"]["package_ref"] = (
            "match.log"
        )

        observation_document = {
            "observation_version": "v1",
            "run_id": run_id,
            "argv": list(observation.argv),
            "invocation_cwd": str(
                invocation_cwd
            ),
            "host_identity_label": (
                observation.host_identity_label
            ),
            "executed_at_utc": (
                observation.executed_at_utc
            ),
            "executed_by": observation.executed_by,
            "runtime_seconds": (
                observation.runtime_seconds
            ),
            "files": evidence,
            "settings": {
                "uci_options": copy.deepcopy(
                    observation.actual_uci_options
                ),
                "time_control": copy.deepcopy(
                    observation.actual_time_control
                ),
                "threads": (
                    observation.actual_threads
                ),
                "hash_mb": (
                    observation.actual_hash_mb
                ),
                "concurrency": (
                    observation.actual_concurrency
                ),
                "seed": observation.actual_seed,
            },
            "pgn_reconstruction": {
                "recomputed_wdl": copy.deepcopy(
                    parse.recomputed_wdl
                ),
                "raw_result_game_count": len(
                    parse.raw_result_games
                ),
                "excluded_count": len(
                    parse.excluded
                ),
                "excluded": copy.deepcopy(
                    parse.excluded
                ),
            },
        }

        _write_json(
            stage / "execution_manifest.json",
            manifest,
        )

        _write_json(
            stage / "raw_result.json",
            raw_result,
        )

        _write_json(
            stage / "artifact_binding.json",
            binding,
        )

        _write_json(
            stage / "execution_observation.json",
            observation_document,
        )

        _seal_package(stage)

        if output.exists():
            raise ExecutionPackageError(
                f"output_dir appeared during build: {output}"
            )

        stage.replace(output)

    except Exception:
        if stage.exists():
            shutil.rmtree(
                stage,
                ignore_errors=True,
            )

        raise

    payload_files = sorted(
        path.name
        for path in output.iterdir()
        if path.is_file()
    )

    return {
        "run_id": run_id,
        "package_dir": str(output),
        "files": payload_files,
        "recomputed_wdl": copy.deepcopy(
            parse.recomputed_wdl
        ),
        "excluded_count": len(
            parse.excluded
        ),
    }