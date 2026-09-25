from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from db_writer import MiningDB, sha256_file
from stockfish_session import StockfishSession


# ============================================================
# Fixed experiment identity
# ============================================================

DEFAULT_STOCKFISH_COMMIT = (
    "17a6c8f1eb0da45c2ca405321919519bf4e211ba"
)

DEFAULT_RUN_TAG = "V0_ONE_FEN_001"

CANDIDATE_NODES = 50_000
CANDIDATE_MULTIPV = 3

TEACHER_NODES = 425_000

# All candidate Top-K moves + teacher bestmove are re-evaluated
# independently with the SAME constrained teacher budget.
TEACHER_SEARCHMOVE_NODES = 100_000

THREADS = 1
HASH_MB = 32


# ============================================================
# Helpers
# ============================================================

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


def read_first_validation_row():
    with open(
        VAL_TSV,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        return next(reader)


def int_or_none(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return int(value)


def show_search(label, result):
    print()
    print(label)
    print("-" * len(label))

    print("bestmove:", result.bestmove)
    print("ponder  :", result.ponder)

    for rank in sorted(result.infos):
        info = result.infos[rank]

        print(
            f"rank={rank} "
            f"move={info.move} "
            f"score={info.score_type} {info.score_value} "
            f"bound={info.bound} "
            f"wdl={info.wdl_w}/{info.wdl_d}/{info.wdl_l} "
            f"depth={info.depth} "
            f"nodes={info.nodes}"
        )


# ============================================================
# Input position
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Legacy V0 one-FEN integration fixture. "
        "Validation input is evaluation-only and must not "
        "become active-learning training data."
    )
)

parser.add_argument(
    "--stockfish",
    required=True,
)
parser.add_argument(
    "--official-net",
    required=True,
)
parser.add_argument(
    "--candidate-net",
    required=True,
)
parser.add_argument(
    "--db",
    required=True,
)
parser.add_argument(
    "--validation-tsv",
    required=True,
)
parser.add_argument(
    "--run-tag",
    default=DEFAULT_RUN_TAG,
)
parser.add_argument(
    "--stockfish-commit",
    default=DEFAULT_STOCKFISH_COMMIT,
)

args = parser.parse_args()

SF = str(
    Path(args.stockfish).expanduser().resolve()
)
OFFICIAL = str(
    Path(args.official_net).expanduser().resolve()
)
CANDIDATE = str(
    Path(args.candidate_net).expanduser().resolve()
)
DB_PATH = str(
    Path(args.db).expanduser().resolve()
)
VAL_TSV = str(
    Path(args.validation_tsv).expanduser().resolve()
)
RUN_TAG = args.run_tag
STOCKFISH_COMMIT = args.stockfish_commit

for label, artifact in (
    ("Stockfish executable", SF),
    ("official NNUE", OFFICIAL),
    ("candidate NNUE", CANDIDATE),
    ("mining database", DB_PATH),
    ("validation TSV", VAL_TSV),
):
    if not Path(artifact).is_file():
        parser.error(
            f"{label} does not exist: {artifact}"
        )

row = read_first_validation_row()

fen = row["fen"]

print()
print("============================================================")
print(" MINER V0 — FIRST PERSISTENT ONE-FEN RUN")
print("============================================================")

print()
print("FEN:")
print(fen)

print()
print("Frozen source :", row["source"])
print(
    "Frozen score  :",
    row["sf_score_type"],
    row["sf_score"],
)
print("Frozen move   :", row["sf_bestmove"])
print(
    "Frozen WDL    :",
    row["sf_wdl_w"],
    row["sf_wdl_d"],
    row["sf_wdl_l"],
)


# ============================================================
# Hash identities
# ============================================================

engine_sha = sha256_file(SF)
official_sha = sha256_file(OFFICIAL)
candidate_sha = sha256_file(CANDIDATE)

print()
print("=== IDENTITIES ===")
print("Engine SHA256   :", engine_sha)
print("Official SHA256 :", official_sha)
print("Candidate SHA256:", candidate_sha)


# ============================================================
# Run Stockfish first.
#
# We collect all raw evidence in memory before touching SQLite.
# If Stockfish fails, the live DB remains untouched.
# ============================================================

with StockfishSession(
    SF,
    CANDIDATE,
    threads=THREADS,
    hash_mb=HASH_MB,
    multipv=CANDIDATE_MULTIPV,
) as candidate, StockfishSession(
    SF,
    OFFICIAL,
    threads=THREADS,
    hash_mb=HASH_MB,
    multipv=1,
) as teacher:

    # --------------------------------------------------------
    # STATIC root evidence
    # --------------------------------------------------------

    candidate_root_static, _ = candidate.static_eval(
        fen
    )

    official_root_static, _ = teacher.static_eval(
        fen
    )

    print()
    print("=== ROOT STATIC ===")
    print(
        "Candidate:",
        candidate_root_static.status,
        candidate_root_static.score_raw_stm,
    )
    print(
        "Official :",
        official_root_static.status,
        official_root_static.score_raw_stm,
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

    show_search(
        "Candidate MultiPV",
        candidate_search,
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

    show_search(
        "Teacher unrestricted",
        teacher_search,
    )

    if candidate_search.bestmove is None:
        raise RuntimeError(
            "Candidate returned no bestmove"
        )

    if teacher_search.bestmove is None:
        raise RuntimeError(
            "Teacher returned no bestmove"
        )

    # --------------------------------------------------------
    # Build constrained teacher move set:
    #
    # candidate Top-K
    #      UNION
    # teacher bestmove
    #
    # preserving order and removing duplicates.
    # --------------------------------------------------------

    probe_moves = []

    for rank in sorted(candidate_search.infos):
        move = candidate_search.infos[rank].move

        if move and move not in probe_moves:
            probe_moves.append(move)

    if teacher_search.bestmove not in probe_moves:
        probe_moves.append(
            teacher_search.bestmove
        )

    print()
    print("=== TEACHER SEARCHMOVES SET ===")

    for move in probe_moves:
        print(" ", move)

    teacher_searchmoves = {}

    for move in probe_moves:
        teacher.new_game()

        result, _ = teacher.search(
            fen,
            nodes=TEACHER_SEARCHMOVE_NODES,
            multipv=1,
            searchmoves=[move],
        )

        teacher_searchmoves[move] = result

        show_search(
            f"Teacher searchmoves {move}",
            result,
        )

    # --------------------------------------------------------
    # Child STATIC evidence
    #
    # For V0 we evaluate the two strategically relevant
    # children:
    #
    # candidate bestmove
    # teacher bestmove
    #
    # If they are identical, only one child is stored.
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
        child_fen = candidate.fen_after_moves(
            fen,
            [move],
        )

        cand_static, _ = candidate.static_eval(
            child_fen
        )

        off_static, _ = teacher.static_eval(
            child_fen
        )

        cand_root_pov = score_to_root_pov(
            cand_static.score_raw_stm,
            fen,
            child_fen,
        )

        off_root_pov = score_to_root_pov(
            off_static.score_raw_stm,
            fen,
            child_fen,
        )

        child_static[move] = {
            "fen": child_fen,
            "candidate": cand_static,
            "candidate_root_pov": cand_root_pov,
            "official": off_static,
            "official_root_pov": off_root_pov,
        }

        print()
        print(
            f"=== CHILD STATIC {move} ==="
        )
        print("Child FEN:")
        print(child_fen)

        print(
            "Candidate:",
            cand_static.status,
            "raw STM=",
            cand_static.score_raw_stm,
            "root POV=",
            cand_root_pov,
        )

        print(
            "Official :",
            off_static.status,
            "raw STM=",
            off_static.score_raw_stm,
            "root POV=",
            off_root_pov,
        )


# ============================================================
# Persist atomically to MINING_DB_V1
# ============================================================

print()
print("=== WRITING LIVE SQLITE DATABASE ===")

with MiningDB(DB_PATH) as db:

    engine_id = db.register_engine(
        name=(
            f"Stockfish "
            f"{STOCKFISH_COMMIT[:8]}"
        ),
        sha256=engine_sha,
        stockfish_commit=STOCKFISH_COMMIT,
        executable_path=SF,
    )

    official_id = db.register_network(
        role_hint="official",
        name=Path(OFFICIAL).name,
        sha256=official_sha,
        path=OFFICIAL,
        description=(
            "Official Stockfish network used as "
            "reference and teacher"
        ),
    )

    candidate_id = db.register_network(
        role_hint="candidate",
        name=Path(CANDIDATE).name,
        sha256=candidate_sha,
        path=CANDIDATE,
        description=(
            "C54 probe candidate, 131072 training "
            "positions, LR=2.5e-5"
        ),
    )

    position_id = db.register_position(
        fen=fen,
        source=row["source"],
        split="val",
        original_score_type=row["sf_score_type"],
        original_score_value=int_or_none(
            row["sf_score"]
        ),
        original_wdl_w=int_or_none(
            row["sf_wdl_w"]
        ),
        original_wdl_d=int_or_none(
            row["sf_wdl_d"]
        ),
        original_wdl_l=int_or_none(
            row["sf_wdl_l"]
        ),
    )

    run_id = db.create_run(
        run_tag=RUN_TAG,
        engine_id=engine_id,
        candidate_network_id=candidate_id,
        official_network_id=official_id,
        teacher_network_id=official_id,
        candidate_nodes=CANDIDATE_NODES,
        candidate_multipv=CANDIDATE_MULTIPV,
        teacher_nodes=TEACHER_NODES,
        threads=THREADS,
        hash_mb=HASH_MB,
        description=(
            "MINING_DB_V1 V0 one-FEN integration run. "
            f"Teacher constrained searchmoves="
            f"{TEACHER_SEARCHMOVE_NODES} nodes."
        ),
    )

    # --------------------------------------------------------
    # Root static evidence
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
            candidate_root_static.score_raw_stm
            if candidate_root_static.status == "ok"
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
            official_root_static.score_raw_stm
            if official_root_static.status == "ok"
            else None
        ),
    )

    # --------------------------------------------------------
    # Candidate MultiPV search
    # --------------------------------------------------------

    candidate_job = db.begin_search_job(
        run_id=run_id,
        position_id=position_id,
        engine_role="candidate",
        engine_id=engine_id,
        network_id=candidate_id,
        search_kind="multipv",
        restricted_move="ALL",
        requested_nodes=CANDIDATE_NODES,
        requested_multipv=CANDIDATE_MULTIPV,
    )

    db.finish_search_job(
        job_id=candidate_job,
        result=candidate_search,
    )

    # --------------------------------------------------------
    # Teacher unrestricted
    # --------------------------------------------------------

    teacher_job = db.begin_search_job(
        run_id=run_id,
        position_id=position_id,
        engine_role="teacher",
        engine_id=engine_id,
        network_id=official_id,
        search_kind="singlepv",
        restricted_move="ALL",
        requested_nodes=TEACHER_NODES,
        requested_multipv=1,
    )

    db.finish_search_job(
        job_id=teacher_job,
        result=teacher_search,
    )

    # --------------------------------------------------------
    # Teacher constrained searchmoves
    # --------------------------------------------------------

    for move, result in teacher_searchmoves.items():

        job_id = db.begin_search_job(
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

        db.finish_search_job(
            job_id=job_id,
            result=result,
        )

    # --------------------------------------------------------
    # Child static evidence
    # --------------------------------------------------------

    for move, evidence in child_static.items():

        db.store_static_eval(
            run_id=run_id,
            position_id=position_id,
            network_role="candidate",
            network_id=candidate_id,
            move=move,
            child_fen=evidence["fen"],
            result=evidence["candidate"],
            score_root_pov=(
                evidence["candidate_root_pov"]
            ),
        )

        db.store_static_eval(
            run_id=run_id,
            position_id=position_id,
            network_role="official",
            network_id=official_id,
            move=move,
            child_fen=evidence["fen"],
            result=evidence["official"],
            score_root_pov=(
                evidence["official_root_pov"]
            ),
        )


# ============================================================
# Independent read-back verification
# ============================================================

print()
print("=== READ-BACK VERIFICATION ===")

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
con.execute("PRAGMA foreign_keys = ON")

counts = {}

for table in (
    "engines",
    "networks",
    "positions",
    "mining_runs",
    "search_jobs",
    "search_results",
    "static_evals",
    "mining_metrics",
    "position_classes",
):
    counts[table] = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

for table, count in counts.items():
    print(
        f"{table:18s}: {count}"
    )

print()
print("=== SEARCH JOBS ===")

jobs = con.execute(
    """
    SELECT
        job_id,
        engine_role,
        search_kind,
        restricted_move,
        requested_nodes,
        requested_multipv,
        status,
        bestmove
    FROM search_jobs
    WHERE run_id = (
        SELECT run_id
        FROM mining_runs
        WHERE run_tag = ?
    )
    ORDER BY job_id
    """,
    (RUN_TAG,),
).fetchall()

for j in jobs:
    print(
        f"job={j['job_id']} "
        f"role={j['engine_role']} "
        f"kind={j['search_kind']} "
        f"move={j['restricted_move']} "
        f"requested={j['requested_nodes']} "
        f"multipv={j['requested_multipv']} "
        f"status={j['status']} "
        f"bestmove={j['bestmove']}"
    )

print()
print("=== SEARCH RESULTS ===")

results = con.execute(
    """
    SELECT
        j.engine_role,
        j.search_kind,
        j.restricted_move,
        r.rank,
        r.move,
        r.score_type,
        r.score_value,
        r.bound,
        r.wdl_w,
        r.wdl_d,
        r.wdl_l,
        r.nodes
    FROM search_jobs AS j
    JOIN search_results AS r
      ON r.job_id = j.job_id
    WHERE j.run_id = (
        SELECT run_id
        FROM mining_runs
        WHERE run_tag = ?
    )
    ORDER BY
        j.job_id,
        r.rank
    """,
    (RUN_TAG,),
).fetchall()

for r in results:
    print(
        f"{r['engine_role']:9s} "
        f"{r['search_kind']:10s} "
        f"{r['restricted_move']:5s} "
        f"rank={r['rank']} "
        f"move={r['move']} "
        f"score={r['score_type']} "
        f"{r['score_value']} "
        f"bound={r['bound']} "
        f"wdl={r['wdl_w']}/"
        f"{r['wdl_d']}/"
        f"{r['wdl_l']} "
        f"snapshot_nodes={r['nodes']}"
    )

print()
print("=== STATIC EVIDENCE ===")

statics = con.execute(
    """
    SELECT
        network_role,
        move,
        status,
        score_raw_stm,
        score_root_pov,
        child_fen
    FROM static_evals
    WHERE run_id = (
        SELECT run_id
        FROM mining_runs
        WHERE run_tag = ?
    )
    ORDER BY
        move,
        network_role
    """,
    (RUN_TAG,),
).fetchall()

for s in statics:
    print(
        f"{s['network_role']:9s} "
        f"move={s['move']:5s} "
        f"status={s['status']:8s} "
        f"raw_stm={s['score_raw_stm']} "
        f"root_pov={s['score_root_pov']}"
    )

con.close()

print()
print("============================================================")
print(" FIRST PERSISTENT MINING POSITION COMPLETE ✅")
print("============================================================")
