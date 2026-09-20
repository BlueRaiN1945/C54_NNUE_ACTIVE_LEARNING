from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass


METRICS_VERSION = "V1"
CLASSIFIER_VERSION = "V1"


@dataclass
class ComputedMetrics:
    run_id: int
    position_id: int

    candidate_bestmove: str | None
    teacher_bestmove: str | None

    candidate_margin_internal: float | None

    teacher_regret_internal: float | None
    teacher_regret_wdl_expected: float | None

    candidate_static_preference: float | None
    official_static_preference: float | None
    deep_teacher_preference: float | None

    static_disagreement: float | None
    search_amplification: float | None

    rank_error: float | None
    topk_overlap: float | None

    structural_persistence: float | None
    break_score: float | None

    primary_class: str | None


def wdl_expected(row) -> float | None:
    if row is None:
        return None

    w = row["wdl_w"]
    d = row["wdl_d"]
    l = row["wdl_l"]

    if w is None or d is None or l is None:
        return None

    total = w + d + l

    if total <= 0:
        return None

    return (w + 0.5 * d) / total


def get_job(
    db,
    *,
    run_id: int,
    position_id: int,
    engine_role: str,
    search_kind: str,
    restricted_move: str,
):
    return db.execute(
        """
        SELECT *
        FROM search_jobs
        WHERE run_id = ?
          AND position_id = ?
          AND engine_role = ?
          AND search_kind = ?
          AND restricted_move = ?
        """,
        (
            run_id,
            position_id,
            engine_role,
            search_kind,
            restricted_move,
        ),
    ).fetchone()


def get_results(db, job_id: int):
    return db.execute(
        """
        SELECT *
        FROM search_results
        WHERE job_id = ?
        ORDER BY rank
        """,
        (job_id,),
    ).fetchall()


def get_teacher_searchmove(
    db,
    *,
    run_id: int,
    position_id: int,
    move: str,
):
    return db.execute(
        """
        SELECT
            r.rank,
            r.move,
            r.score_type,
            r.score_value,
            r.bound,
            r.wdl_w,
            r.wdl_d,
            r.wdl_l,
            r.depth,
            r.seldepth,
            r.nodes,
            r.time_ms,
            r.nps,
            r.pv,
            j.requested_nodes
        FROM search_jobs AS j
        JOIN search_results AS r
          ON r.job_id = j.job_id
        WHERE j.run_id = ?
          AND j.position_id = ?
          AND j.engine_role = 'teacher'
          AND j.search_kind = 'searchmove'
          AND j.restricted_move = ?
          AND r.rank = 1
        """,
        (
            run_id,
            position_id,
            move,
        ),
    ).fetchone()


def exact_cp(row) -> float | None:
    if row is None:
        return None

    if row["bound"] != "exact":
        return None

    if row["score_type"] != "cp":
        return None

    return float(row["score_value"])


def load_static_scores(
    db,
    *,
    run_id: int,
    position_id: int,
):
    rows = db.execute(
        """
        SELECT
            network_role,
            move,
            status,
            score_root_pov
        FROM static_evals
        WHERE run_id = ?
          AND position_id = ?
        """,
        (
            run_id,
            position_id,
        ),
    ).fetchall()

    result = {}

    for row in rows:
        key = (
            row["network_role"],
            row["move"],
        )

        if (
            row["status"] == "ok"
            and row["score_root_pov"] is not None
        ):
            result[key] = float(
                row["score_root_pov"]
            )
        else:
            result[key] = None

    return result


def compute_one(
    db,
    *,
    run_id: int,
    position_id: int,
):
    candidate_job = get_job(
        db,
        run_id=run_id,
        position_id=position_id,
        engine_role="candidate",
        search_kind="multipv",
        restricted_move="ALL",
    )

    teacher_job = get_job(
        db,
        run_id=run_id,
        position_id=position_id,
        engine_role="teacher",
        search_kind="singlepv",
        restricted_move="ALL",
    )

    if candidate_job is None:
        raise RuntimeError(
            f"No candidate MultiPV job for position {position_id}"
        )

    if teacher_job is None:
        raise RuntimeError(
            f"No unrestricted teacher job for position {position_id}"
        )

    candidate_best = candidate_job["bestmove"]
    teacher_best = teacher_job["bestmove"]

    candidate_results = get_results(
        db,
        int(candidate_job["job_id"]),
    )

    candidate_by_rank = {
        int(row["rank"]): row
        for row in candidate_results
    }

    # --------------------------------------------------------
    # Candidate confidence margin
    #
    # Positive means candidate rank #1 is ahead of rank #2.
    # Only defined for exact CP scores.
    # --------------------------------------------------------

    candidate_margin = None

    r1 = candidate_by_rank.get(1)
    r2 = candidate_by_rank.get(2)

    r1_cp = exact_cp(r1)
    r2_cp = exact_cp(r2)

    if r1_cp is not None and r2_cp is not None:
        candidate_margin = r1_cp - r2_cp

    # --------------------------------------------------------
    # Candidate rank of teacher bestmove.
    #
    # rank_error = 0 means candidate also ranked it #1.
    # rank_error = 1 means candidate ranked it #2, etc.
    #
    # If teacher move lies outside candidate Top-K we leave the
    # numeric value NULL rather than pretending to know its rank.
    # --------------------------------------------------------

    teacher_candidate_rank = None

    if teacher_best is not None:
        for row in candidate_results:
            if row["move"] == teacher_best:
                teacher_candidate_rank = int(
                    row["rank"]
                )
                break

    rank_error = None

    if teacher_candidate_rank is not None:
        rank_error = float(
            teacher_candidate_rank - 1
        )

    # --------------------------------------------------------
    # Teacher constrained comparisons.
    #
    # These compare the candidate move and teacher move using
    # separate searchmoves jobs with the same requested budget.
    #
    # We require selected snapshots to be exact before using
    # CP/WDL differences.
    # --------------------------------------------------------

    candidate_teacher_probe = None
    teacher_teacher_probe = None

    if candidate_best is not None:
        candidate_teacher_probe = (
            get_teacher_searchmove(
                db,
                run_id=run_id,
                position_id=position_id,
                move=candidate_best,
            )
        )

    if teacher_best is not None:
        teacher_teacher_probe = (
            get_teacher_searchmove(
                db,
                run_id=run_id,
                position_id=position_id,
                move=teacher_best,
            )
        )

    deep_teacher_preference = None
    teacher_regret_internal = None
    teacher_regret_wdl = None

    if (
        candidate_best is not None
        and teacher_best is not None
        and candidate_best == teacher_best
    ):
        # Comparing a move against itself is exactly zero.
        deep_teacher_preference = 0.0
        teacher_regret_internal = 0.0
        teacher_regret_wdl = 0.0

    else:
        candidate_teacher_cp = exact_cp(
            candidate_teacher_probe
        )

        teacher_teacher_cp = exact_cp(
            teacher_teacher_probe
        )

        if (
            candidate_teacher_cp is not None
            and teacher_teacher_cp is not None
        ):
            deep_teacher_preference = (
                teacher_teacher_cp
                - candidate_teacher_cp
            )

            teacher_regret_internal = max(
                0.0,
                deep_teacher_preference,
            )

        if (
            candidate_teacher_probe is not None
            and teacher_teacher_probe is not None
            and candidate_teacher_probe["bound"] == "exact"
            and teacher_teacher_probe["bound"] == "exact"
        ):
            candidate_e = wdl_expected(
                candidate_teacher_probe
            )

            teacher_e = wdl_expected(
                teacher_teacher_probe
            )

            if (
                candidate_e is not None
                and teacher_e is not None
            ):
                teacher_regret_wdl = max(
                    0.0,
                    teacher_e - candidate_e,
                )

    # --------------------------------------------------------
    # Static move preference.
    #
    # Positive candidate_static_preference means candidate
    # NNUE statically prefers the candidate-selected child.
    #
    # Positive official_static_preference means official NNUE
    # statically prefers the candidate-selected child.
    #
    # Thus an official negative value while candidate is
    # positive is one structural-disagreement signal.
    # --------------------------------------------------------

    static_scores = load_static_scores(
        db,
        run_id=run_id,
        position_id=position_id,
    )

    candidate_static_preference = None
    official_static_preference = None

    if (
        candidate_best is not None
        and teacher_best is not None
    ):
        if candidate_best == teacher_best:
            c = static_scores.get(
                ("candidate", candidate_best)
            )

            o = static_scores.get(
                ("official", candidate_best)
            )

            if c is not None:
                candidate_static_preference = 0.0

            if o is not None:
                official_static_preference = 0.0

        else:
            c_candidate = static_scores.get(
                ("candidate", candidate_best)
            )

            c_teacher = static_scores.get(
                ("candidate", teacher_best)
            )

            o_candidate = static_scores.get(
                ("official", candidate_best)
            )

            o_teacher = static_scores.get(
                ("official", teacher_best)
            )

            if (
                c_candidate is not None
                and c_teacher is not None
            ):
                candidate_static_preference = (
                    c_candidate - c_teacher
                )

            if (
                o_candidate is not None
                and o_teacher is not None
            ):
                official_static_preference = (
                    o_candidate - o_teacher
                )

    static_disagreement = None

    if (
        candidate_static_preference is not None
        and official_static_preference is not None
    ):
        static_disagreement = abs(
            candidate_static_preference
            - official_static_preference
        )

    # --------------------------------------------------------
    # Intentionally NOT defined in conservative V1 yet.
    #
    # topk_overlap requires a genuine teacher Top-K set.
    # search_amplification / structural_persistence / break_score
    # need definitions validated on more than one disagreement.
    # --------------------------------------------------------

    topk_overlap = None
    search_amplification = None
    structural_persistence = None
    break_score = None

    # --------------------------------------------------------
    # Conservative classes: no thresholds.
    # --------------------------------------------------------

    classes = []

    if candidate_best is None or teacher_best is None:
        primary_class = "EVIDENCE_INCOMPLETE"

        classes.append(
            (
                "EVIDENCE_INCOMPLETE",
                1.0,
            )
        )

    elif candidate_best == teacher_best:
        primary_class = "AGREEMENT"

        classes.append(
            (
                "AGREEMENT",
                1.0,
            )
        )

    else:
        primary_class = "MOVE_DISAGREEMENT"

        classes.append(
            (
                "MOVE_DISAGREEMENT",
                1.0,
            )
        )

        if teacher_candidate_rank is None:
            classes.append(
                (
                    "TEACHER_OUTSIDE_CANDIDATE_TOPK",
                    1.0,
                )
            )

    metrics = ComputedMetrics(
        run_id=run_id,
        position_id=position_id,

        candidate_bestmove=candidate_best,
        teacher_bestmove=teacher_best,

        candidate_margin_internal=candidate_margin,

        teacher_regret_internal=(
            teacher_regret_internal
        ),
        teacher_regret_wdl_expected=(
            teacher_regret_wdl
        ),

        candidate_static_preference=(
            candidate_static_preference
        ),
        official_static_preference=(
            official_static_preference
        ),
        deep_teacher_preference=(
            deep_teacher_preference
        ),

        static_disagreement=(
            static_disagreement
        ),
        search_amplification=(
            search_amplification
        ),

        rank_error=rank_error,
        topk_overlap=topk_overlap,

        structural_persistence=(
            structural_persistence
        ),
        break_score=break_score,

        primary_class=primary_class,
    )

    return metrics, classes


def store_one(
    db,
    metrics: ComputedMetrics,
    classes,
):
    db.execute(
        """
        INSERT INTO mining_metrics (
            run_id,
            position_id,

            candidate_bestmove,
            teacher_bestmove,

            candidate_margin_internal,

            teacher_regret_internal,
            teacher_regret_wdl_expected,

            candidate_static_preference,
            official_static_preference,
            deep_teacher_preference,

            static_disagreement,
            search_amplification,

            rank_error,
            topk_overlap,

            structural_persistence,
            break_score,

            primary_class,
            metrics_version
        )
        VALUES (
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?
        )

        ON CONFLICT (run_id, position_id)
        DO UPDATE SET
            candidate_bestmove =
                excluded.candidate_bestmove,

            teacher_bestmove =
                excluded.teacher_bestmove,

            candidate_margin_internal =
                excluded.candidate_margin_internal,

            teacher_regret_internal =
                excluded.teacher_regret_internal,

            teacher_regret_wdl_expected =
                excluded.teacher_regret_wdl_expected,

            candidate_static_preference =
                excluded.candidate_static_preference,

            official_static_preference =
                excluded.official_static_preference,

            deep_teacher_preference =
                excluded.deep_teacher_preference,

            static_disagreement =
                excluded.static_disagreement,

            search_amplification =
                excluded.search_amplification,

            rank_error =
                excluded.rank_error,

            topk_overlap =
                excluded.topk_overlap,

            structural_persistence =
                excluded.structural_persistence,

            break_score =
                excluded.break_score,

            primary_class =
                excluded.primary_class,

            metrics_version =
                excluded.metrics_version
        """,
        (
            metrics.run_id,
            metrics.position_id,

            metrics.candidate_bestmove,
            metrics.teacher_bestmove,

            metrics.candidate_margin_internal,

            metrics.teacher_regret_internal,
            metrics.teacher_regret_wdl_expected,

            metrics.candidate_static_preference,
            metrics.official_static_preference,
            metrics.deep_teacher_preference,

            metrics.static_disagreement,
            metrics.search_amplification,

            metrics.rank_error,
            metrics.topk_overlap,

            metrics.structural_persistence,
            metrics.break_score,

            metrics.primary_class,
            METRICS_VERSION,
        ),
    )

    db.execute(
        """
        DELETE FROM position_classes
        WHERE run_id = ?
          AND position_id = ?
        """,
        (
            metrics.run_id,
            metrics.position_id,
        ),
    )

    for class_name, class_score in classes:
        db.execute(
            """
            INSERT INTO position_classes (
                run_id,
                position_id,
                class_name,
                class_score,
                classifier_version
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                metrics.run_id,
                metrics.position_id,
                class_name,
                class_score,
                CLASSIFIER_VERSION,
            ),
        )


def process_run(
    db,
    *,
    run_tag: str,
    write: bool,
):
    run = db.execute(
        """
        SELECT run_id
        FROM mining_runs
        WHERE run_tag = ?
        """,
        (run_tag,),
    ).fetchone()

    if run is None:
        raise RuntimeError(
            f"Unknown run_tag: {run_tag}"
        )

    run_id = int(run["run_id"])

    positions = db.execute(
        """
        SELECT DISTINCT position_id
        FROM search_jobs
        WHERE run_id = ?
        ORDER BY position_id
        """,
        (run_id,),
    ).fetchall()

    if not positions:
        raise RuntimeError(
            f"No positions found for run {run_tag}"
        )

    for row in positions:
        position_id = int(
            row["position_id"]
        )

        metrics, classes = compute_one(
            db,
            run_id=run_id,
            position_id=position_id,
        )

        print()
        print("=" * 70)
        print(
            f"run={run_tag} "
            f"position_id={position_id}"
        )
        print("=" * 70)

        for field, value in vars(metrics).items():
            print(
                f"{field:30s}: {value}"
            )

        print(
            f"{'classes':30s}: "
            f"{classes}"
        )

        if write:
            store_one(
                db,
                metrics,
                classes,
            )

    if write:
        db.commit()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True,
    )

    parser.add_argument(
        "--run-tag",
        required=True,
    )

    parser.add_argument(
        "--write",
        action="store_true",
    )

    args = parser.parse_args()

    db = sqlite3.connect(
        args.db
    )

    db.row_factory = sqlite3.Row
    db.execute(
        "PRAGMA foreign_keys = ON"
    )

    try:
        process_run(
            db,
            run_tag=args.run_tag,
            write=args.write,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
