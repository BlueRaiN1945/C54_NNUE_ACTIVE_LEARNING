"""Read-only trend and experiment comparison over the derived SQLite index.

This layer is deliberately descriptive.

It does not recompute PGN, Elo, confidence intervals, GSPRT, or verdicts.
It does not rank experiment arms from Elo point estimates.

Statistical conclusions are inherited from the canonical AnalysisResultV1
documents represented in the rebuildable SQLite index.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from medium_pc_audit.ledger import LEDGER_SCHEMA_VERSION


class TrendError(ValueError):
    """Raised when trend/index invariants are violated."""


def _validate_label(value, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(ch) < 0x20 or ord(ch) == 0x7F
            for ch in value
        )
    ):
        raise TrendError(
            f"invalid {name}: {value!r}"
        )

    return value


def _validate_index(connection: sqlite3.Connection) -> None:
    try:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM metadata"
            ).fetchall()
        )

        actual_count = connection.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]

    except sqlite3.DatabaseError as exc:
        raise TrendError(
            "invalid or incompatible research SQLite index"
        ) from exc

    if metadata.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
        raise TrendError(
            "SQLite index ledger schema version mismatch"
        )

    try:
        expected_count = int(
            metadata["record_count"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrendError(
            "SQLite index has invalid record_count metadata"
        ) from exc

    if expected_count != actual_count:
        raise TrendError(
            "SQLite index record_count metadata does not match records table"
        )


def load_index_records(
    sqlite_path,
    *,
    opponent_id: str | None = None,
    experiment_id: str | None = None,
    arm_id: str | None = None,
) -> list[dict]:
    """Load ordered derived records with optional exact-match filters."""

    path = Path(
        sqlite_path
    )

    if not path.exists() or not path.is_file():
        raise TrendError(
            f"SQLite index does not exist: {path}"
        )

    filters = []
    parameters = []

    if opponent_id is not None:
        opponent_id = _validate_label(
            opponent_id,
            name="opponent_id",
        )
        filters.append(
            "opponent_id = ?"
        )
        parameters.append(
            opponent_id
        )

    if experiment_id is not None:
        experiment_id = _validate_label(
            experiment_id,
            name="experiment_id",
        )
        filters.append(
            "experiment_id = ?"
        )
        parameters.append(
            experiment_id
        )

    if arm_id is not None:
        arm_id = _validate_label(
            arm_id,
            name="arm_id",
        )
        filters.append(
            "arm_id = ?"
        )
        parameters.append(
            arm_id
        )

    where = (
        " WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    query = (
        """
        SELECT
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
            regression_flagged

        FROM records
        """
        + where
        + """
        ORDER BY recorded_at, run_id
        """
    )

    try:
        with closing(
            sqlite3.connect(path)
        ) as connection:
            connection.row_factory = sqlite3.Row

            _validate_index(
                connection
            )

            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

    except TrendError:
        raise

    except sqlite3.DatabaseError as exc:
        raise TrendError(
            "cannot query research SQLite index"
        ) from exc

    return [
        dict(row)
        for row in rows
    ]


def _verdict_counts(records: list[dict]) -> dict:
    counter = Counter(
        record["verdict"]
        for record in records
    )

    return {
        "SUPPORTED_PROGRESS": counter.get(
            "SUPPORTED_PROGRESS",
            0,
        ),
        "PROGRESS_NOT_SUPPORTED": counter.get(
            "PROGRESS_NOT_SUPPORTED",
            0,
        ),
        "INCONCLUSIVE": counter.get(
            "INCONCLUSIVE",
            0,
        ),
    }


def _evidence_state(
    records: list[dict],
    verdict_counts: dict,
) -> str:
    """Summarize source verdict composition without inventing a new test."""

    if not records:
        return "NO_DATA"

    regression_count = sum(
        int(record["regression_flagged"])
        for record in records
    )

    supported = verdict_counts[
        "SUPPORTED_PROGRESS"
    ]
    unsupported = verdict_counts[
        "PROGRESS_NOT_SUPPORTED"
    ]
    inconclusive = verdict_counts[
        "INCONCLUSIVE"
    ]
    total = len(records)

    if regression_count:
        if (
            regression_count == total
            and supported == 0
            and inconclusive == 0
        ):
            return "REGRESSION_EVIDENCE_ONLY"

        return "MIXED_EVIDENCE_WITH_REGRESSION"

    if supported == total:
        return "SUPPORTED_PROGRESS_ONLY"

    if unsupported == total:
        return "PROGRESS_NOT_SUPPORTED_ONLY"

    if inconclusive == total:
        return "INCONCLUSIVE_ONLY"

    return "MIXED_EVIDENCE"


def _observed_elo_direction(
    records: list[dict],
) -> str:
    """Describe point-estimate movement only; this is not significance."""

    if len(records) < 2:
        return "INSUFFICIENT_POINTS"

    values = [
        float(record["elo_value"])
        for record in records
    ]

    differences = [
        right - left
        for left, right in zip(
            values,
            values[1:],
        )
    ]

    tolerance = 1e-12

    positive = any(
        difference > tolerance
        for difference in differences
    )
    negative = any(
        difference < -tolerance
        for difference in differences
    )

    if positive and not negative:
        return "NONDECREASING"

    if negative and not positive:
        return "NONINCREASING"

    if not positive and not negative:
        return "FLAT"

    return "MIXED"


def _ci_relation_to_zero(record: dict) -> str:
    low = float(
        record["elo_ci_low"]
    )
    high = float(
        record["elo_ci_high"]
    )

    if low > 0.0:
        return "ENTIRELY_ABOVE_ZERO"

    if high < 0.0:
        return "ENTIRELY_BELOW_ZERO"

    return "CROSSES_ZERO"


def summarize_series(
    sqlite_path,
    *,
    opponent_id: str | None = None,
    experiment_id: str | None = None,
    arm_id: str | None = None,
) -> dict:
    """Build a descriptive series summary from indexed final results."""

    records = load_index_records(
        sqlite_path,
        opponent_id=opponent_id,
        experiment_id=experiment_id,
        arm_id=arm_id,
    )

    counts = _verdict_counts(
        records
    )

    run_summaries = [
        {
            "run_id": record["run_id"],
            "recorded_at": record["recorded_at"],
            "candidate_sha256": record["candidate_sha256"],
            "opponent_id": record["opponent_id"],
            "experiment_id": record["experiment_id"],
            "arm_id": record["arm_id"],
            "games_played": record["games_played"],
            "sprt_outcome": record["sprt_outcome"],
            "verdict": record["verdict"],
            "elo_value": record["elo_value"],
            "elo_ci_low": record["elo_ci_low"],
            "elo_ci_high": record["elo_ci_high"],
            "ci_relation_to_zero": _ci_relation_to_zero(
                record
            ),
            "regression_flagged": bool(
                record["regression_flagged"]
            ),
        }
        for record in records
    ]

    if records:
        first_elo = float(
            records[0]["elo_value"]
        )
        latest_elo = float(
            records[-1]["elo_value"]
        )
        observed_delta = (
            latest_elo - first_elo
        )
    else:
        first_elo = None
        latest_elo = None
        observed_delta = None

    return {
        "filters": {
            "opponent_id": opponent_id,
            "experiment_id": experiment_id,
            "arm_id": arm_id,
        },
        "run_count": len(records),
        "verdict_counts": counts,
        "regression_flagged_count": sum(
            int(record["regression_flagged"])
            for record in records
        ),
        "evidence_state": _evidence_state(
            records,
            counts,
        ),
        "observed_elo_direction": _observed_elo_direction(
            records
        ),
        "first_elo": first_elo,
        "latest_elo": latest_elo,
        "observed_elo_delta": observed_delta,
        "runs": run_summaries,
    }


def compare_experiment_arms(
    sqlite_path,
    *,
    experiment_id: str,
    opponent_id: str | None = None,
) -> dict:
    """Compare experiment arms descriptively without ranking a winner."""

    experiment_id = _validate_label(
        experiment_id,
        name="experiment_id",
    )

    records = load_index_records(
        sqlite_path,
        opponent_id=opponent_id,
        experiment_id=experiment_id,
    )

    arm_ids = sorted(
        {
            record["arm_id"]
            for record in records
            if record["arm_id"] is not None
        }
    )

    arms = {
        arm_id: summarize_series(
            sqlite_path,
            opponent_id=opponent_id,
            experiment_id=experiment_id,
            arm_id=arm_id,
        )
        for arm_id in arm_ids
    }

    unassigned_count = sum(
        record["arm_id"] is None
        for record in records
    )

    if not records:
        comparison_state = "NO_DATA"
    elif len(arm_ids) < 2:
        comparison_state = "INSUFFICIENT_ARMS"
    else:
        comparison_state = "DESCRIPTIVE_ONLY"

    return {
        "experiment_id": experiment_id,
        "opponent_id": opponent_id,
        "comparison_state": comparison_state,
        "run_count": len(records),
        "unassigned_run_count": unassigned_count,
        "arms": arms,
    }
