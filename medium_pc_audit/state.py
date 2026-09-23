"""Append-only per-run state history for RESULTS_INBOX.

Raw execution evidence lives at:

    RESULTS_INBOX/<run_id>/

and is never modified after publication.

Mutable workflow knowledge is NOT stored inside that raw package.  Instead,
each state transition is appended as one canonical JSON line to:

    RESULTS_INBOX/_STATE/<run_id>.jsonl

This module deliberately exposes no update/delete/truncate API.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from medium_pc_audit.identity import canonicalize


RECEIVED = "RECEIVED"
ACCEPTED = "ACCEPTED"
INCOMPLETE = "INCOMPLETE"
QUARANTINED = "QUARANTINED"
REJECTED_DUPLICATE = "REJECTED_DUPLICATE"

ALLOWED_STATES = frozenset(
    {
        RECEIVED,
        ACCEPTED,
        INCOMPLETE,
        QUARANTINED,
        REJECTED_DUPLICATE,
    }
)

_SAFE_RUN_ID_RE = re.compile(
    r"^[A-Za-z0-9_-]+$"
)


class StateHistoryError(ValueError):
    """Raised when append-only state history invariants are violated."""


def require_safe_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or not run_id
        or not _SAFE_RUN_ID_RE.fullmatch(run_id)
    ):
        raise StateHistoryError(
            f"unsafe run_id: {run_id!r}"
        )

    return run_id


def _validate_utc(timestamp: str) -> str:
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise StateHistoryError(
            "recorded_at must be an ISO-8601 UTC string ending in Z"
        )

    try:
        parsed = datetime.fromisoformat(
            timestamp[:-1] + "+00:00"
        )
    except ValueError as exc:
        raise StateHistoryError(
            f"invalid recorded_at timestamp: {timestamp!r}"
        ) from exc

    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise StateHistoryError(
            "recorded_at must represent UTC"
        )

    return timestamp


def _validate_reason_codes(reason_codes) -> list[str]:
    if isinstance(reason_codes, str):
        raise StateHistoryError(
            "reason_codes must be a sequence, not a bare string"
        )

    try:
        values = list(reason_codes)
    except TypeError as exc:
        raise StateHistoryError(
            "reason_codes must be iterable"
        ) from exc

    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or any(
                ord(ch) < 0x20 or ord(ch) == 0x7F
                for ch in value
            )
        ):
            raise StateHistoryError(
                f"invalid reason code: {value!r}"
            )

    return values


def state_history_path(
    inbox_root,
    run_id: str,
) -> Path:
    run_id = require_safe_run_id(run_id)

    return (
        Path(inbox_root)
        / "_STATE"
        / f"{run_id}.jsonl"
    )


def read_state_history(
    inbox_root,
    run_id: str,
) -> list[dict]:
    path = state_history_path(
        inbox_root,
        run_id,
    )

    if not path.exists():
        return []

    events = []

    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise StateHistoryError(
            f"cannot read state history: {exc}"
        ) from exc

    for line_number, raw_line in enumerate(
        raw_lines,
        start=1,
    ):
        if not raw_line:
            raise StateHistoryError(
                f"blank state-history line at {line_number}"
            )

        try:
            event = json.loads(
                raw_line.decode("utf-8")
            )
        except Exception as exc:
            raise StateHistoryError(
                f"invalid state-history JSON at line {line_number}"
            ) from exc

        if not isinstance(event, dict):
            raise StateHistoryError(
                f"state-history line {line_number} is not an object"
            )

        if event.get("run_id") != run_id:
            raise StateHistoryError(
                f"state-history run_id mismatch at line {line_number}"
            )

        if event.get("state") not in ALLOWED_STATES:
            raise StateHistoryError(
                f"unknown state at line {line_number}"
            )

        events.append(event)

    return events


def _transition_allowed(
    history: list[dict],
    new_state: str,
) -> bool:

    if not history:
        return new_state in {
            RECEIVED,
            QUARANTINED,
            REJECTED_DUPLICATE,
        }

    previous = history[-1]["state"]

    if previous == RECEIVED:
        return new_state in {
            ACCEPTED,
            INCOMPLETE,
            QUARANTINED,
            REJECTED_DUPLICATE,
        }

    # One execution attempt is terminal after ACCEPTED / INCOMPLETE /
    # QUARANTINED.  Re-use of the same run_id is never another execution;
    # it is recorded only as a duplicate attempt.
    return new_state == REJECTED_DUPLICATE


def append_state_event(
    inbox_root,
    *,
    run_id: str,
    state: str,
    recorded_at: str,
    reason_codes=(),
    details=None,
) -> dict:
    """Append exactly one canonical JSON event."""

    run_id = require_safe_run_id(run_id)

    if state not in ALLOWED_STATES:
        raise StateHistoryError(
            f"unsupported state: {state!r}"
        )

    recorded_at = _validate_utc(
        recorded_at
    )

    reasons = _validate_reason_codes(
        reason_codes
    )

    history = read_state_history(
        inbox_root,
        run_id,
    )

    if not _transition_allowed(
        history,
        state,
    ):
        previous = (
            history[-1]["state"]
            if history
            else None
        )

        raise StateHistoryError(
            f"invalid state transition: {previous!r} -> {state!r}"
        )

    event = {
        "recorded_at": recorded_at,
        "reason_codes": reasons,
        "run_id": run_id,
        "state": state,
    }

    if details is not None:
        event["details"] = details

    encoded = canonicalize(event)

    path = state_history_path(
        inbox_root,
        run_id,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
            raise StateHistoryError(
                "short write while appending state history"
            )

        os.fsync(fd)

    finally:
        os.close(fd)

    return event


def run_id_seen(
    inbox_root,
    run_id: str,
) -> bool:
    run_id = require_safe_run_id(run_id)

    root = Path(inbox_root)

    return (
        (root / run_id).exists()
        or state_history_path(
            root,
            run_id,
        ).exists()
    )
