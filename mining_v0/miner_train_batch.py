from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from db_writer import MiningDB, sha256_file
from stockfish_session import StockfishSession


DEFAULT_STOCKFISH_COMMIT = (
    "17a6c8f1eb0da45c2ca405321919519bf4e211ba"
)

CANDIDATE_NODES = 50_000
CANDIDATE_MULTIPV = 3

TEACHER_NODES = 425_000
TEACHER_SEARCHMOVE_NODES = 100_000

THREADS = 1
HASH_MB = 32


def int_or_none(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return int(value)


def stm(fen: str) -> str:
    return fen.split()[1]


def score_to_root_pov(
    raw_stm_score: int | None,
    root_fen: str,
    evaluated_fen: str,
) -> int | None:
    if raw_stm_score is None:
        return None

    if stm(root_fen) == stm(evaluated_fen):
        return raw_stm_score

    return -raw_stm_score


def load_manifest(path: str):
    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )

    if not rows:
        raise RuntimeError(
            "Manifest contains no positions"
        )

    required = {
        "batch_index",
        "selection_sha256",
        "train_row_number",
        "fen",
        "source",
        "sf_score_type",
        "sf_score",
        "sf_wdl_w",
        "sf_wdl_d",
        "sf_wdl_l",
        "sf_bestmove",
        "sf_error",
    }

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    return rows


def build_probe_moves(
    candidate_search,
    teacher_search,
):
    """
    Build the deterministic teacher probe union:

    candidate UCI bestmove
    UNION coherent candidate MultiPV moves
    UNION teacher UCI bestmove

    Order is preserved and duplicates are removed.
    """
    probe_moves = []

    if candidate_search.bestmove is not None:
        probe_moves.append(
            candidate_search.bestmove
        )

    for rank in sorted(
        candidate_search.infos
    ):
        move = (
            candidate_search
            .infos[rank]
            .move
        )

        if (
            move is not None
            and move not in probe_moves
        ):
            probe_moves.append(move)

    if (
        teacher_search.bestmove is not None
        and teacher_search.bestmove
        not in probe_moves
    ):
        probe_moves.append(
            teacher_search.bestmove
        )

    return probe_moves


def collect_position_evidence(
    *,
    row,
    candidate,
    teacher,
):
    fen = row["fen"]

    # --------------------------------------------------------
    # Root static
    # --------------------------------------------------------

    candidate_root_static, _ = (
        candidate.static_eval(fen)
    )

    official_root_static, _ = (
        teacher.static_eval(fen)
    )

    # --------------------------------------------------------
    # Candidate MultiPV
    # --------------------------------------------------------

    candidate.new_game()

    candidate_search, _ = candidate.search(
        fen,
        nodes=CANDIDATE_NODES,
        multipv=CANDIDATE_MULTIPV,
    )

    if candidate_search.bestmove is None:
        raise RuntimeError(
            "Candidate returned no bestmove"
        )

    # --------------------------------------------------------
    # Teacher unrestricted
    # --------------------------------------------------------

    teacher.new_game()

    teacher_search, _ = teacher.search(
        fen,
        nodes=TEACHER_NODES,
        multipv=1,
    )

    if teacher_search.bestmove is None:
        raise RuntimeError(
            "Teacher returned no bestmove"
        )

    # --------------------------------------------------------
    # Candidate Top-K UNION teacher bestmove
    # --------------------------------------------------------

    probe_moves = build_probe_moves(
        candidate_search,
        teacher_search,
    )

    if not probe_moves:
        raise RuntimeError(
            "No moves available for teacher probes"
        )

    # --------------------------------------------------------
    # Equal-budget constrained teacher probes
    # --------------------------------------------------------

    teacher_searchmoves = {}

    for move in probe_moves:
        teacher.new_game()

        result, _ = teacher.search(
            fen,
            nodes=TEACHER_SEARCHMOVE_NODES,
            multipv=1,
            searchmoves=[move],
        )

        if (
            result.bestmove is not None
            and result.bestmove != move
        ):
            raise RuntimeError(
                f"searchmoves {move} returned "
                f"bestmove {result.bestmove}"
            )

        teacher_searchmoves[move] = result

    # --------------------------------------------------------
    # Static child matrix:
    #
    # candidate best child
    # teacher best child
    #
    # evaluated by candidate and official NNUE.
    # --------------------------------------------------------

    child_moves = []

    for move in (
        candidate_search.bestmove,
        teacher_search.bestmove,
    ):
        if move not in child_moves:
            child_moves.append(move)

    child_static = {}

    for move in child_moves:
        child_fen = (
            candidate.fen_after_moves(
                fen,
                [move],
            )
        )

        cand_static, _ = (
            candidate.static_eval(
                child_fen
            )
        )

        off_static, _ = (
            teacher.static_eval(
                child_fen
            )
        )

        child_static[move] = {
            "fen": child_fen,

            "candidate": cand_static,
            "candidate_root_pov":
                score_to_root_pov(
                    cand_static.score_raw_stm,
                    fen,
                    child_fen,
                ),

            "official": off_static,
            "official_root_pov":
                score_to_root_pov(
                    off_static.score_raw_stm,
                    fen,
                    child_fen,
                ),
        }

    return {
        "candidate_root_static":
            candidate_root_static,

        "official_root_static":
            official_root_static,

        "candidate_search":
            candidate_search,

        "teacher_search":
            teacher_search,

        "teacher_searchmoves":
            teacher_searchmoves,

        "child_static":
            child_static,
    }


def persist_position(
    *,
    db,
    row,
    run_id,
    engine_id,
    candidate_id,
    official_id,
    evidence,
):
    fen = row["fen"]

    position_id = db.register_position(
        fen=fen,
        source=row["source"],
        split="train",

        original_score_type=(
            row["sf_score_type"]
        ),

        original_score_value=(
            int_or_none(
                row["sf_score"]
            )
        ),

        original_wdl_w=(
            int_or_none(
                row["sf_wdl_w"]
            )
        ),

        original_wdl_d=(
            int_or_none(
                row["sf_wdl_d"]
            )
        ),

        original_wdl_l=(
            int_or_none(
                row["sf_wdl_l"]
            )
        ),
    )

    candidate_root_static = (
        evidence[
            "candidate_root_static"
        ]
    )

    official_root_static = (
        evidence[
            "official_root_static"
        ]
    )

    candidate_search = (
        evidence[
            "candidate_search"
        ]
    )

    teacher_search = (
        evidence[
            "teacher_search"
        ]
    )

    teacher_searchmoves = (
        evidence[
            "teacher_searchmoves"
        ]
    )

    child_static = (
        evidence[
            "child_static"
        ]
    )

    # --------------------------------------------------------
    # ROOT static
    # --------------------------------------------------------

    db.store_static_eval(
        run_id=run_id,
        position_id=position_id,

        network_role="candidate",
        network_id=candidate_id,

        move="ROOT",
        child_fen=None,

        result=candidate_root_static,

        score_root_pov=(
            candidate_root_static
            .score_raw_stm

            if (
                candidate_root_static
                .status
                == "ok"
            )

            else None
        ),
    )

    db.store_static_eval(
        run_id=run_id,
        position_id=position_id,

        network_role="official",
        network_id=official_id,

        move="ROOT",
        child_fen=None,

        result=official_root_static,

        score_root_pov=(
            official_root_static
            .score_raw_stm

            if (
                official_root_static
                .status
                == "ok"
            )

            else None
        ),
    )

    # --------------------------------------------------------
    # Candidate MultiPV
    # --------------------------------------------------------

    candidate_job = (
        db.begin_search_job(
            run_id=run_id,
            position_id=position_id,

            engine_role="candidate",
            engine_id=engine_id,
            network_id=candidate_id,

            search_kind="multipv",
            restricted_move="ALL",

            requested_nodes=(
                CANDIDATE_NODES
            ),

            requested_multipv=(
                CANDIDATE_MULTIPV
            ),
        )
    )

    db.finish_search_job(
        job_id=candidate_job,
        result=candidate_search,
    )

    # --------------------------------------------------------
    # Teacher unrestricted
    # --------------------------------------------------------

    teacher_job = (
        db.begin_search_job(
            run_id=run_id,
            position_id=position_id,

            engine_role="teacher",
            engine_id=engine_id,
            network_id=official_id,

            search_kind="singlepv",
            restricted_move="ALL",

            requested_nodes=(
                TEACHER_NODES
            ),

            requested_multipv=1,
        )
    )

    db.finish_search_job(
        job_id=teacher_job,
        result=teacher_search,
    )

    # --------------------------------------------------------
    # Teacher constrained
    # --------------------------------------------------------

    for move, result in (
        teacher_searchmoves.items()
    ):
        job_id = (
            db.begin_search_job(
                run_id=run_id,
                position_id=position_id,

                engine_role="teacher",
                engine_id=engine_id,
                network_id=official_id,

                search_kind="searchmove",
                restricted_move=move,

                requested_nodes=(
                    TEACHER_SEARCHMOVE_NODES
                ),

                requested_multipv=1,
            )
        )

        db.finish_search_job(
            job_id=job_id,
            result=result,
        )

    # --------------------------------------------------------
    # Child static
    # --------------------------------------------------------

    for move, child in (
        child_static.items()
    ):
        db.store_static_eval(
            run_id=run_id,
            position_id=position_id,

            network_role="candidate",
            network_id=candidate_id,

            move=move,
            child_fen=child["fen"],

            result=child["candidate"],

            score_root_pov=(
                child[
                    "candidate_root_pov"
                ]
            ),
        )

        db.store_static_eval(
            run_id=run_id,
            position_id=position_id,

            network_role="official",
            network_id=official_id,

            move=move,
            child_fen=child["fen"],

            result=child["official"],

            score_root_pov=(
                child[
                    "official_root_pov"
                ]
            ),
        )

    return position_id


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stockfish",
        required=True,
        help="Path to the Stockfish executable.",
    )

    parser.add_argument(
        "--official-net",
        required=True,
        help="Path to the official/reference NNUE network.",
    )

    parser.add_argument(
        "--candidate-net",
        required=True,
        help="Path to the candidate NNUE network.",
    )

    parser.add_argument(
        "--stockfish-commit",
        default=DEFAULT_STOCKFISH_COMMIT,
        help="Stockfish source commit recorded in research metadata.",
    )

    parser.add_argument(
        "--db",
        required=True,
    )

    parser.add_argument(
        "--batch",
        required=True,
    )

    parser.add_argument(
        "--run-tag",
        required=True,
    )

    parser.add_argument(
        "--expected-batch-sha",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    stockfish = str(
        Path(args.stockfish).expanduser().resolve()
    )
    official = str(
        Path(args.official_net).expanduser().resolve()
    )
    candidate = str(
        Path(args.candidate_net).expanduser().resolve()
    )

    for label, artifact in (
        ("Stockfish executable", stockfish),
        ("official NNUE", official),
        ("candidate NNUE", candidate),
    ):
        if not Path(artifact).is_file():
            parser.error(
                f"{label} does not exist: {artifact}"
            )

    actual_batch_sha = (
        sha256_file(args.batch)
    )

    if (
        actual_batch_sha
        != args.expected_batch_sha
    ):
        raise RuntimeError(
            "Batch SHA256 mismatch:\n"
            f" expected="
            f"{args.expected_batch_sha}\n"
            f" actual={actual_batch_sha}"
        )

    rows = load_manifest(
        args.batch
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be > 0"
            )

        rows = rows[:args.limit]

    engine_sha = sha256_file(stockfish)
    official_sha = sha256_file(
        official
    )
    candidate_sha = sha256_file(
        candidate
    )

    print()
    print("=" * 72)
    print(" MINER TRAIN BATCH V0")
    print("=" * 72)

    print(
        "Run tag            :",
        args.run_tag,
    )

    print(
        "Manifest SHA256     :",
        actual_batch_sha,
    )

    print(
        "Positions this run  :",
        len(rows),
    )

    print(
        "Candidate nodes     :",
        CANDIDATE_NODES,
    )

    print(
        "Candidate MultiPV   :",
        CANDIDATE_MULTIPV,
    )

    print(
        "Teacher nodes       :",
        TEACHER_NODES,
    )

    print(
        "Searchmove nodes    :",
        TEACHER_SEARCHMOVE_NODES,
    )

    # --------------------------------------------------------
    # Register immutable identities/run first.
    # --------------------------------------------------------

    db = MiningDB(
        args.db
    )

    try:
        engine_id = db.register_engine(
            name=(
                f"Stockfish "
                f"{args.stockfish_commit[:8]}"
            ),

            sha256=engine_sha,

            stockfish_commit=(
                args.stockfish_commit
            ),

            executable_path=stockfish,
        )

        official_id = (
            db.register_network(
                role_hint="official",

                name=Path(official).name,

                sha256=official_sha,
                path=official,

                description=(
                    "Official Stockfish "
                    "reference/teacher"
                ),
            )
        )

        candidate_id = (
            db.register_network(
                role_hint="candidate",

                name=Path(candidate).name,

                sha256=candidate_sha,
                path=candidate,

                description=(
                    "C54 probe candidate; "
                    "131072 training "
                    "positions; LR=2.5e-5"
                ),
            )
        )

        run_id = db.create_run(
            run_tag=args.run_tag,

            engine_id=engine_id,

            candidate_network_id=(
                candidate_id
            ),

            official_network_id=(
                official_id
            ),

            teacher_network_id=(
                official_id
            ),

            candidate_nodes=(
                CANDIDATE_NODES
            ),

            candidate_multipv=(
                CANDIDATE_MULTIPV
            ),

            teacher_nodes=(
                TEACHER_NODES
            ),

            threads=THREADS,
            hash_mb=HASH_MB,

            description=(
                "TRAIN batch miner V0; "
                f"manifest_sha256="
                f"{actual_batch_sha}; "
                f"teacher_searchmove_nodes="
                f"{TEACHER_SEARCHMOVE_NODES}"
            ),
        )

        # Freeze run identity separately from
        # position transactions.
        db.commit()

        # ----------------------------------------------------
        # Keep engines alive across positions.
        # Clear Hash before independent searches.
        # ----------------------------------------------------

        with StockfishSession(
            stockfish,
            candidate,
            threads=THREADS,
            hash_mb=HASH_MB,
            multipv=CANDIDATE_MULTIPV,
        ) as candidate, StockfishSession(
            stockfish,
            official,
            threads=THREADS,
            hash_mb=HASH_MB,
            multipv=1,
        ) as teacher:

            total = len(rows)

            for n, row in enumerate(
                rows,
                start=1,
            ):
                fen = row["fen"]

                print()
                print("=" * 72)

                print(
                    f"[{n}/{total}] "
                    f"batch_index="
                    f"{row['batch_index']} "
                    f"source={row['source']}"
                )

                print("=" * 72)

                print("FEN:", fen)

                print(
                    "Frozen:",
                    row["sf_score_type"],
                    row["sf_score"],
                    "move=",
                    row["sf_bestmove"],
                )

                try:
                    evidence = (
                        collect_position_evidence(
                            row=row,
                            candidate=candidate,
                            teacher=teacher,
                        )
                    )

                    candidate_best = (
                        evidence[
                            "candidate_search"
                        ].bestmove
                    )

                    teacher_best = (
                        evidence[
                            "teacher_search"
                        ].bestmove
                    )

                    print(
                        "Candidate best:",
                        candidate_best,
                    )

                    print(
                        "Teacher best  :",
                        teacher_best,
                    )

                    print(
                        "Agreement     :",
                        (
                            "YES"
                            if (
                                candidate_best
                                == teacher_best
                            )
                            else "NO"
                        ),
                    )

                    print(
                        "Teacher probes:",
                        ", ".join(
                            evidence[
                                "teacher_searchmoves"
                            ].keys()
                        ),
                    )

                    position_id = (
                        persist_position(
                            db=db,
                            row=row,
                            run_id=run_id,
                            engine_id=engine_id,
                            candidate_id=(
                                candidate_id
                            ),
                            official_id=(
                                official_id
                            ),
                            evidence=evidence,
                        )
                    )

                    # One position = one durable transaction.
                    db.commit()

                    print(
                        "Committed position_id:",
                        position_id,
                    )

                except Exception:
                    db.rollback()
                    raise

        # ----------------------------------------------------
        # Run-specific read-back
        # ----------------------------------------------------

        con = sqlite3.connect(
            args.db
        )

        con.row_factory = sqlite3.Row
        con.execute(
            "PRAGMA foreign_keys = ON"
        )

        position_count = con.execute(
            """
            SELECT COUNT(
                DISTINCT position_id
            )
            FROM search_jobs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

        job_count = con.execute(
            """
            SELECT COUNT(*)
            FROM search_jobs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

        result_count = con.execute(
            """
            SELECT COUNT(*)
            FROM search_results
            WHERE job_id IN (
                SELECT job_id
                FROM search_jobs
                WHERE run_id = ?
            )
            """,
            (run_id,),
        ).fetchone()[0]

        static_count = con.execute(
            """
            SELECT COUNT(*)
            FROM static_evals
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]

        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        fk_errors = con.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        print()
        print("=" * 72)
        print(" BATCH READ-BACK")
        print("=" * 72)

        print(
            "positions      :",
            position_count,
        )

        print(
            "search_jobs    :",
            job_count,
        )

        print(
            "search_results :",
            result_count,
        )

        print(
            "static_evals   :",
            static_count,
        )

        print(
            "integrity      :",
            integrity,
        )

        print(
            "foreign keys   :",
            len(fk_errors),
            "errors",
        )

        assert integrity == "ok"
        assert fk_errors == []

        con.close()

    finally:
        db.close()

    print()
    print(
        "TRAIN BATCH MINER COMPLETE ✅"
    )


if __name__ == "__main__":
    main()
