"""Canonical append-only research ledger and rebuildable SQLite index.

The JSONL ledger is the authoritative research history.

    LEDGER/research.jsonl

The SQLite database is derived state only and may be deleted and rebuilt from
the JSONL ledger at any time.

This module deliberately exposes no update/delete/truncate API for the ledger.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from medium_pc_audit.identity import (
    canonicalize,
    compute_identity_sha256,
)
from medium_pc_audit.schemas import registry
from medium_pc_audit.state import require_safe_run_id


LEDGER_FILENAME = "research.jsonl"
INDEX_FILENAME = "research.sqlite3"

LEDGER_SCHEMA_VERSION = "v1"
ACCEPTED_SOURCE_STATE = "ACCEPTED"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LedgerError(ValueError):
    """Raised when a research-ledger invariant is violated."""


class LedgerConflictError(LedgerError):
    """Raised when the same run_id is presented with different content."""


def ledger_path(ledger_root) -> Path:
    return Path(ledger_root) / LEDGER_FILENAME


def sqlite_index_path(ledger_root) -> Path:
    return Path(ledger_root) / INDEX_FILENAME


def _validate_utc(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LedgerError(
            f"{name} must be an ISO-8601 UTC string ending in Z"
        )

    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00"
        )
    except ValueError as exc:
        raise LedgerError(
            f"invalid {name} timestamp: {value!r}"
        ) from exc

    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LedgerError(
            f"{name} must represent UTC"
        )

    return value


def _validate_sha256(value, *, name: str) -> str:
    if not isinstance(value, str):
        raise LedgerError(
            f"{name} must be a SHA256 hex string"
        )

    normalized = value.lower()

    if not _SHA256_RE.fullmatch(normalized):
        raise LedgerError(
            f"{name} must contain exactly 64 hexadecimal characters"
        )

    return normalized


def _validate_label(value, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(ch) < 0x20 or ord(ch) == 0x7F
            for ch in value
        )
    ):
        raise LedgerError(
            f"invalid {name}: {value!r}"
        )

    return value


def _payload_for_identity(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key != "record_id"
    }


def _record_identity(record: dict) -> str:
    return compute_identity_sha256(
        _payload_for_identity(record)
    )


def build_ledger_record(
    *,
    analysis_result: dict,
    recorded_at: str,
    config_identity_sha256: str,
    candidate_sha256: str,
    opponent_sha256: str,
    opponent_id: str,
    experiment_id: str | None = None,
    arm_id: str | None = None,
) -> dict:
    """Build one canonical ledger record from an accepted final analysis.

    Phase 6 does not recompute PGN, Elo, CI, or GSPRT.  The complete original
    AnalysisResultV1 document is retained as evidence.
    """

    if not isinstance(analysis_result, dict):
        raise LedgerError(
            "analysis_result must be an object"
        )

    try:
        registry.validate(
            "analysis_result",
            "v1",
            analysis_result,
        )
    except Exception as exc:
        raise LedgerError(
            "analysis_result is not a valid AnalysisResultV1 document"
        ) from exc

    run_id = require_safe_run_id(
        analysis_result.get("run_id")
    )

    recorded_at = _validate_utc(
        recorded_at,
        name="recorded_at",
    )

    config_identity_sha256 = _validate_sha256(
        config_identity_sha256,
        name="config_identity_sha256",
    )
    candidate_sha256 = _validate_sha256(
        candidate_sha256,
        name="candidate_sha256",
    )
    opponent_sha256 = _validate_sha256(
        opponent_sha256,
        name="opponent_sha256",
    )

    opponent_id = _validate_label(
        opponent_id,
        name="opponent_id",
    )

    if experiment_id is not None:
        experiment_id = _validate_label(
            experiment_id,
            name="experiment_id",
        )

    if arm_id is not None:
        arm_id = _validate_label(
            arm_id,
            name="arm_id",
        )

    record = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "source_state": ACCEPTED_SOURCE_STATE,
        "recorded_at": recorded_at,
        "run_id": run_id,
        "config_identity_sha256": config_identity_sha256,
        "candidate_sha256": candidate_sha256,
        "opponent_sha256": opponent_sha256,
        "opponent_id": opponent_id,
        "analysis_result": deepcopy(
            analysis_result
        ),
    }

    if experiment_id is not None:
        record["experiment_id"] = experiment_id

    if arm_id is not None:
        record["arm_id"] = arm_id

    record["record_id"] = _record_identity(
        record
    )

    _validate_ledger_record(record)

    return record


def _validate_ledger_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise LedgerError(
            "ledger record must be an object"
        )

    required = {
        "schema_version",
        "source_state",
        "recorded_at",
        "run_id",
        "config_identity_sha256",
        "candidate_sha256",
        "opponent_sha256",
        "opponent_id",
        "analysis_result",
        "record_id",
    }
    optional = {
        "experiment_id",
        "arm_id",
    }

    keys = set(record)

    missing = required - keys
    extra = keys - required - optional

    if missing:
        raise LedgerError(
            "ledger record missing fields: "
            + ", ".join(sorted(missing))
        )

    if extra:
        raise LedgerError(
            "ledger record has unknown fields: "
            + ", ".join(sorted(extra))
        )

    if record["schema_version"] != LEDGER_SCHEMA_VERSION:
        raise LedgerError(
            f"unsupported ledger schema_version: "
            f"{record['schema_version']!r}"
        )

    if record["source_state"] != ACCEPTED_SOURCE_STATE:
        raise LedgerError(
            "only ACCEPTED source results may enter the research ledger"
        )

    run_id = require_safe_run_id(
        record["run_id"]
    )

    _validate_utc(
        record["recorded_at"],
        name="recorded_at",
    )

    for field_name in (
        "config_identity_sha256",
        "candidate_sha256",
        "opponent_sha256",
    ):
        normalized = _validate_sha256(
            record[field_name],
            name=field_name,
        )
        if record[field_name] != normalized:
            raise LedgerError(
                f"{field_name} must use lowercase hexadecimal"
            )

    _validate_label(
        record["opponent_id"],
        name="opponent_id",
    )

    if "experiment_id" in record:
        _validate_label(
            record["experiment_id"],
            name="experiment_id",
        )

    if "arm_id" in record:
        _validate_label(
            record["arm_id"],
            name="arm_id",
        )

    analysis_result = record["analysis_result"]

    if not isinstance(analysis_result, dict):
        raise LedgerError(
            "analysis_result must be an object"
        )

    try:
        registry.validate(
            "analysis_result",
            "v1",
            analysis_result,
        )
    except Exception as exc:
        raise LedgerError(
            "embedded analysis_result is invalid"
        ) from exc

    if analysis_result.get("run_id") != run_id:
        raise LedgerError(
            "ledger run_id does not match analysis_result run_id"
        )

    expected_record_id = _record_identity(
        record
    )

    if record["record_id"] != expected_record_id:
        raise LedgerError(
            "ledger record_id does not match canonical record content"
        )

    return record


def read_ledger(ledger_root) -> list[dict]:
    """Read and fully validate the authoritative JSONL ledger."""

    path = ledger_path(
        ledger_root
    )

    if not path.exists():
        return []

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LedgerError(
            f"cannot read research ledger: {exc}"
        ) from exc

    if not raw:
        return []

    if not raw.endswith(b"\n"):
        raise LedgerError(
            "research ledger is missing its final newline; "
            "possible partial append"
        )

    raw_lines = raw.splitlines()

    records = []
    by_run_id = {}
    by_record_id = {}

    for line_number, raw_line in enumerate(
        raw_lines,
        start=1,
    ):
        if not raw_line:
            raise LedgerError(
                f"blank ledger line at {line_number}"
            )

        try:
            record = json.loads(
                raw_line.decode("utf-8")
            )
        except Exception as exc:
            raise LedgerError(
                f"invalid ledger JSON at line {line_number}"
            ) from exc

        try:
            _validate_ledger_record(
                record
            )
        except Exception as exc:
            if isinstance(exc, LedgerError):
                raise LedgerError(
                    f"invalid ledger record at line {line_number}: {exc}"
                ) from exc

            raise

        run_id = record["run_id"]
        record_id = record["record_id"]

        if run_id in by_run_id:
            raise LedgerError(
                f"duplicate run_id in ledger at line {line_number}: "
                f"{run_id}"
            )

        if record_id in by_record_id:
            raise LedgerError(
                f"duplicate record_id in ledger at line {line_number}: "
                f"{record_id}"
            )

        by_run_id[run_id] = record
        by_record_id[record_id] = record

        records.append(
            record
        )

    return records


def append_ledger_record(
    ledger_root,
    record: dict,
) -> dict:
    """Idempotently append exactly one canonical research record."""

    _validate_ledger_record(
        record
    )

    root = Path(
        ledger_root
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_records = read_ledger(
        root
    )

    for existing in existing_records:
        if existing["run_id"] != record["run_id"]:
            continue

        if existing["record_id"] == record["record_id"]:
            return {
                "appended": False,
                "record": existing,
            }

        raise LedgerConflictError(
            "run_id already exists with different canonical content: "
            + record["run_id"]
        )

    encoded = canonicalize(
        record
    )

    if not encoded.endswith(b"\n"):
        raise LedgerError(
            "canonical ledger serialization must end with a newline"
        )

    path = ledger_path(
        root
    )

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_BINARY", 0)
    )

    fd = os.open(
        path,
        flags,
        0o600,
    )

    try:
        written = os.write(
            fd,
            encoded,
        )

        if written != len(encoded):
            raise LedgerError(
                "short write while appending research ledger"
            )

        os.fsync(fd)

    finally:
        os.close(fd)

    return {
        "appended": True,
        "record": record,
    }


def _canonical_json_text(value: dict) -> str:
    encoded = canonicalize(
        value
    )
    return encoded.decode("utf-8").rstrip("\n")


def _create_index_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            recorded_at TEXT NOT NULL,
            analyzed_at TEXT NOT NULL,
            analyzer_version TEXT NOT NULL,

            config_identity_sha256 TEXT NOT NULL,
            candidate_sha256 TEXT NOT NULL,
            opponent_sha256 TEXT NOT NULL,
            opponent_id TEXT NOT NULL,

            experiment_id TEXT,
            arm_id TEXT,

            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            draws INTEGER NOT NULL,

            games_played INTEGER NOT NULL,
            sprt_outcome TEXT NOT NULL,
            verdict TEXT NOT NULL,

            elo_value REAL NOT NULL,
            elo_ci_low REAL NOT NULL,
            elo_ci_high REAL NOT NULL,

            regression_tested INTEGER NOT NULL,
            regression_flagged INTEGER NOT NULL,

            record_json TEXT NOT NULL
        );

        CREATE INDEX idx_records_recorded_at
            ON records(recorded_at);

        CREATE INDEX idx_records_opponent_id
            ON records(opponent_id);

        CREATE INDEX idx_records_candidate_sha256
            ON records(candidate_sha256);

        CREATE INDEX idx_records_config_identity_sha256
            ON records(config_identity_sha256);

        CREATE INDEX idx_records_verdict
            ON records(verdict);

        CREATE INDEX idx_records_experiment_arm
            ON records(experiment_id, arm_id);
        """
    )


def rebuild_index(
    ledger_root,
    *,
    sqlite_path=None,
) -> Path:
    """Rebuild the derived SQLite index entirely from canonical JSONL."""

    root = Path(
        ledger_root
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = read_ledger(
        root
    )

    target = (
        Path(sqlite_path)
        if sqlite_path is not None
        else sqlite_index_path(root)
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_name(
        target.name
        + ".tmp-"
        + uuid.uuid4().hex
    )

    connection = None

    try:
        connection = sqlite3.connect(
            temporary
        )

        _create_index_schema(
            connection
        )

        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                "ledger_schema_version",
                LEDGER_SCHEMA_VERSION,
            ),
        )

        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                "record_count",
                str(len(records)),
            ),
        )

        for record in records:
            analysis = record["analysis_result"]
            wdl = analysis["recomputed_wdl"]
            sprt = analysis["sprt_result"]
            elo = analysis["elo_estimate"]
            regression = analysis["regression_subtest"]

            connection.execute(
                """
                INSERT INTO records (
                    record_id,
                    run_id,
                    recorded_at,
                    analyzed_at,
                    analyzer_version,

                    config_identity_sha256,
                    candidate_sha256,
                    opponent_sha256,
                    opponent_id,

                    experiment_id,
                    arm_id,

                    wins,
                    losses,
                    draws,

                    games_played,
                    sprt_outcome,
                    verdict,

                    elo_value,
                    elo_ci_low,
                    elo_ci_high,

                    regression_tested,
                    regression_flagged,

                    record_json
                )
                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?
                )
                """,
                (
                    record["record_id"],
                    record["run_id"],
                    record["recorded_at"],
                    analysis["analyzed_at"],
                    analysis["analyzer_version"],

                    record["config_identity_sha256"],
                    record["candidate_sha256"],
                    record["opponent_sha256"],
                    record["opponent_id"],

                    record.get("experiment_id"),
                    record.get("arm_id"),

                    wdl["wins"],
                    wdl["losses"],
                    wdl["draws"],

                    sprt["games_played"],
                    sprt["outcome"],
                    analysis["verdict"],

                    elo["value"],
                    elo["ci_low"],
                    elo["ci_high"],

                    int(regression["tested"]),
                    int(regression["flagged"]),

                    _canonical_json_text(record),
                ),
            )

        connection.commit()
        connection.close()
        connection = None

        fd = os.open(
            temporary,
            os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        os.replace(
            temporary,
            target,
        )

    except Exception:
        if connection is not None:
            connection.close()

        if temporary.exists():
            temporary.unlink()

        raise

    return target

