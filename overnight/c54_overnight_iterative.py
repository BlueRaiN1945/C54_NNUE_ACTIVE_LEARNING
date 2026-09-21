#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# C54 OVERNIGHT ITERATIVE TRAINER V1
# ============================================================

ROOT = Path(
    "/data/BASELINE_2026/NNUE_TUNING/"
    "C54_ITERATIVE_OVERNIGHT_V1"
)

REPO = Path("/workspace/nnue-pytorch")

AL = Path(
    "/data/BASELINE_2026/TRAINING_425K/"
    "MINING_V1/C54_NNUE_ACTIVE_LEARNING"
)

SPLIT = Path(
    "/data/BASELINE_2026/TRAINING_425K/SPLIT_V1"
)

TRAIN_TSV = SPLIT / "TRAIN_95.tsv"
VAL_BIN = SPLIT / "VAL_05.binpack"
CONVERTER = SPLIT / "tsv_to_binpack"
EVAL = SPLIT / "eval_official_base.py"

SF = Path(
    "/data/STOCKFISH_LINUX_17a6c8f1/src/stockfish"
)

OFFICIAL = Path(
    "/data/STOCKFISH_LINUX_17a6c8f1/src/"
    "nn-134a887f4c8f.nnue"
)

INITIAL_PT = Path(
    "/data/BASELINE_2026/NNUE_TUNING/"
    "C54_LR_SWEEP_BS4096_131K_V1/"
    "LR_1e-5/candidate_LR_1e-5.pt"
)

INITIAL_NNUE = Path(
    "/data/BASELINE_2026/NNUE_TUNING/"
    "C54_LR_SWEEP_BS4096_131K_V1/"
    "LR_1e-5/candidate_LR_1e-5.nnue"
)

CCHESS = Path(
    "/data/C54_MATCH_TOOLS/c-chess-cli/c-chess-cli"
)

BOOK = Path(
    "/data/C54_MATCH_TOOLS/books/noob_3moves.epd"
)

SCHEMA = AL / "schema/MINING_DB_V1.sql"
MINER = AL / "mining_v0/miner_train_batch.py"
METRICS = AL / "mining_v0/metrics.py"

STATE = ROOT / "STATE.json"
STOP_FILE = ROOT / "STOP"
ORDER_FILE = ROOT / "ELIGIBLE_ORDER.txt"

HOURS = 8.0

# One new research region per generation.
POOL_SIZE = 512

# Broad replay protects general strength.
REPLAY_SIZE = 8192

# Conservative update.
LR = 5e-6
EPOCH_SIZE = 65536
BATCH_SIZE = 4096

# Promotion test.
MATCH_GAMES = 100
MATCH_NODES = 50000
PROMOTE_SCORE = 0.52

# Candidate may not damage broad fixed DEV by more than 0.5%.
MAX_DEV_RELATIVE = 1.005

ORDER_SEED = "C54_ITERATIVE_ORDER_V1"


# ============================================================
# HELPERS
# ============================================================

def log(*x):
    print(
        time.strftime("[%Y-%m-%d %H:%M:%S]"),
        *x,
        flush=True,
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def position_key(fen):
    p = fen.split()
    return " ".join(p[:4])


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def stop_requested():
    return STOP_FILE.exists()


class CleanStop(Exception):
    pass


def check_stop():
    if stop_requested():
        raise CleanStop()


def run_logged(cmd, cwd, logfile, env=None):
    logfile = Path(logfile)
    logfile.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("RUN:", " ".join(map(str, cmd)))
    log("LOG:", logfile)

    with open(
        logfile,
        "a",
        encoding="utf-8",
        buffering=1,
    ) as out:
        out.write(
            "\nCOMMAND:\n"
            + " ".join(map(str, cmd))
            + "\n\n"
        )

        rc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode

    if rc != 0:
        raise RuntimeError(
            f"Command failed rc={rc}: "
            + " ".join(map(str, cmd))
        )

    check_stop()


def run_capture(
    cmd,
    cwd,
    logfile,
    env=None,
    timeout=None,
):
    log("RUN:", " ".join(map(str, cmd)))

    try:
        p = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""

        if isinstance(output, bytes):
            output = output.decode(
                "utf-8",
                errors="replace",
            )

        Path(logfile).write_text(
            output,
            encoding="utf-8",
        )

        raise RuntimeError(
            f"Command timed out after {timeout}s: "
            + " ".join(map(str, cmd))
        ) from exc

    Path(logfile).write_text(
        p.stdout,
        encoding="utf-8",
    )

    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={p.returncode}: "
            + " ".join(map(str, cmd))
        )

    check_stop()
    return p.stdout


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs():
    paths = [
        REPO / "train.py",
        REPO / "serialize.py",
        TRAIN_TSV,
        VAL_BIN,
        CONVERTER,
        EVAL,
        SF,
        OFFICIAL,
        INITIAL_PT,
        INITIAL_NNUE,
        CCHESS,
        BOOK,
        SCHEMA,
        MINER,
        METRICS,
    ]

    missing = [
        str(p)
        for p in paths
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required files:\n"
            + "\n".join(missing)
        )

    log("All required files exist.")


# ============================================================
# INITIAL CHAMPION / STATE
# ============================================================

def load_or_create_state():
    ROOT.mkdir(parents=True, exist_ok=True)

    (ROOT / "CHAMPIONS").mkdir(
        exist_ok=True
    )

    (ROOT / "GENERATIONS").mkdir(
        exist_ok=True
    )

    if STATE.exists():
        state = json.loads(
            STATE.read_text(
                encoding="utf-8"
            )
        )

        log(
            "RESUME:",
            "champion V"
            + str(state["champion_version"]),
            "| next GEN",
            state["next_gen"],
            "| cursor",
            state["cursor"],
        )

        return state

    champdir = ROOT / "CHAMPIONS/V1"
    champdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    champ_pt = (
        champdir
        / "C54_CHAMPION_V1.pt"
    )

    champ_nnue = (
        champdir
        / "C54_CHAMPION_V1.nnue"
    )

    if not champ_pt.exists():
        shutil.copy2(
            INITIAL_PT,
            champ_pt,
        )

    if not champ_nnue.exists():
        shutil.copy2(
            INITIAL_NNUE,
            champ_nnue,
        )

    state = {
        "format": 1,
        "status": "READY",
        "champion_version": 1,
        "champion_pt": str(champ_pt),
        "champion_nnue": str(champ_nnue),
        "champion_dev": None,
        "next_gen": 1,
        "cursor": 0,
        "cycle": 0,
        "avg_gen_seconds": 1800.0,
        "history": [],
    }

    atomic_json(STATE, state)

    (champdir / "SHA256SUMS.txt").write_text(
        sha256_file(champ_pt)
        + "  "
        + str(champ_pt)
        + "\n"
        + sha256_file(champ_nnue)
        + "  "
        + str(champ_nnue)
        + "\n",
        encoding="utf-8",
    )

    log("Created frozen C54_CHAMPION_V1.")

    return state


# ============================================================
# FIXED DEV
# ============================================================

def eval_pt(pt, logfile):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    logfile = Path(logfile)
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        check_stop()

        attempt_log = logfile.with_name(
            logfile.stem
            + f".attempt_{attempt}"
            + logfile.suffix
        )

        log(
            "DEV attempt",
            attempt,
            "/",
            max_attempts,
            "| model:",
            pt,
        )

        try:
            text = run_capture(
                [
                    sys.executable,
                    EVAL,
                    pt,
                    VAL_BIN,
                ],
                REPO,
                attempt_log,
                env=env,
                timeout=90,
            )

        except RuntimeError as exc:
            if stop_requested():
                raise CleanStop() from exc

            if "timed out after" not in str(exc):
                raise

            log(
                "DEV attempt",
                attempt,
                "timed out after 90s.",
            )

            if attempt == max_attempts:
                raise RuntimeError(
                    "Fixed DEV evaluation failed after "
                    f"{max_attempts} timeout attempts: {pt}"
                ) from exc

            time.sleep(2)
            continue

        matches = re.findall(
            r"Mean validation loss:\s*"
            r"([0-9.eE+\-]+)",
            text,
        )

        if not matches:
            log(
                "DEV attempt",
                attempt,
                "returned without parseable loss.",
            )

            if attempt == max_attempts:
                raise RuntimeError(
                    "Could not parse fixed DEV loss after "
                    f"{max_attempts} attempts."
                )

            time.sleep(2)
            continue

        logfile.write_text(
            text,
            encoding="utf-8",
        )

        value = float(matches[-1])

        log(
            "DEV attempt",
            attempt,
            "succeeded | loss:",
            value,
        )

        return value


# ============================================================
# DETERMINISTIC GLOBAL TRAIN ORDER
# ============================================================

def build_or_load_order():
    if ORDER_FILE.exists():
        log("Loading eligible order:", ORDER_FILE)

        return [
            int(x)
            for x in ORDER_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

    log(
        "Building deterministic eligible TRAIN order."
    )

    items = []

    with open(
        TRAIN_TSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for rownum, row in enumerate(
            reader,
            start=1,
        ):
            if (
                row.get("sf_score_type") != "cp"
                or row.get("sf_bound") != "exact"
                or row.get("sf_error", "").strip()
                or not row.get(
                    "sf_score", ""
                ).strip()
                or row.get(
                    "sf_bestmove", ""
                ).strip()
                in ("", "(none)", "0000")
            ):
                continue

            key = position_key(
                row["fen"]
            )

            digest = hashlib.sha256(
                (
                    ORDER_SEED
                    + "\0"
                    + key
                ).encode("utf-8")
            ).digest()

            items.append(
                (digest, rownum)
            )

    items.sort(
        key=lambda x: x[0]
    )

    order = [
        rownum
        for _, rownum in items
    ]

    ORDER_FILE.write_text(
        "".join(
            f"{n}\n"
            for n in order
        ),
        encoding="utf-8",
    )

    log(
        "Eligible positions:",
        len(order),
    )

    return order


def take_circular(order, start, count, exclude=None):
    exclude = exclude or set()

    out = []
    i = start % len(order)

    while len(out) < count:
        rn = order[i]

        if rn not in exclude:
            out.append(rn)

        i = (i + 1) % len(order)

    return out


def read_train_rows(rownums):
    needed = set(rownums)
    out = {}

    with open(
        TRAIN_TSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        fields = list(
            reader.fieldnames
        )

        for rownum, row in enumerate(
            reader,
            start=1,
        ):
            if rownum in needed:
                out[rownum] = row

                if len(out) == len(needed):
                    break

    missing = needed - set(out)

    if missing:
        raise RuntimeError(
            "Missing TRAIN rows: "
            + repr(
                sorted(missing)[:20]
            )
        )

    return fields, out


# ============================================================
# MINER MANIFEST
# ============================================================

def make_manifest(
    path,
    pool_rownums,
    rows,
    original_fields,
    generation,
):
    fields = [
        "batch_index",
        "selection_sha256",
        "train_row_number",
    ] + original_fields

    seed = (
        f"C54_GEN_{generation:04d}"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        for i, rn in enumerate(
            pool_rownums,
            start=1,
        ):
            row = dict(rows[rn])

            key = position_key(
                row["fen"]
            )

            selection_hash = (
                hashlib.sha256(
                    (
                        seed
                        + "\0"
                        + key
                    ).encode("utf-8")
                ).hexdigest()
            )

            out = {
                "batch_index": i,
                "selection_sha256":
                    selection_hash,
                "train_row_number": rn,
            }

            out.update(row)
            writer.writerow(out)


def create_db(path):
    if path.exists():
        path.unlink()

    schema = SCHEMA.read_text(
        encoding="utf-8"
    )

    con = sqlite3.connect(path)

    try:
        con.executescript(schema)
        con.commit()

        integrity = con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                "SQLite integrity failed."
            )
    finally:
        con.close()


# ============================================================
# HARD POSITION EXTRACTION
# ============================================================

def classify_pool(
    db_path,
    run_tag,
    pool_rows,
):
    by_fen = {
        row["fen"]: rn
        for rn, row
        in pool_rows.items()
    }

    by_key = {
        position_key(row["fen"]): rn
        for rn, row
        in pool_rows.items()
    }

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        run = con.execute(
            """
            SELECT run_id
            FROM mining_runs
            WHERE run_tag=?
            """,
            (run_tag,),
        ).fetchone()

        if run is None:
            raise RuntimeError(
                "Mining run not found."
            )

        run_id = run["run_id"]

        metrics = con.execute(
            """
            SELECT *
            FROM mining_metrics
            WHERE run_id=?
            """,
            (run_id,),
        ).fetchall()

        pos_cols = [
            x[1]
            for x in con.execute(
                "PRAGMA table_info(positions)"
            )
        ]

        if "fen" in pos_cols:
            fen_col = "fen"
        else:
            candidates = [
                c for c in pos_cols
                if "fen" in c.lower()
            ]

            if not candidates:
                raise RuntimeError(
                    "No FEN column found "
                    "in positions table."
                )

            fen_col = candidates[0]

        positions = con.execute(
            f"""
            SELECT position_id,
                   {fen_col} AS fen
            FROM positions
            """
        ).fetchall()

        pid_to_fen = {
            r["position_id"]: r["fen"]
            for r in positions
        }

        hard = {}
        seen = set()

        for m in metrics:
            fen = pid_to_fen.get(
                m["position_id"]
            )

            if fen is None:
                continue

            rn = by_fen.get(fen)

            if rn is None:
                rn = by_key.get(
                    position_key(fen)
                )

            if rn is None:
                continue

            seen.add(rn)

            cls = (
                m["primary_class"]
                if "primary_class"
                in m.keys()
                else None
            )

            regret = (
                m["teacher_regret_internal"]
                if "teacher_regret_internal"
                in m.keys()
                else None
            )

            margin = (
                m["candidate_margin_internal"]
                if "candidate_margin_internal"
                in m.keys()
                else None
            )

            static_diff = (
                m["static_disagreement"]
                if "static_disagreement"
                in m.keys()
                else None
            )

            regret = (
                float(regret)
                if regret is not None
                else 0.0
            )

            margin = (
                float(margin)
                if margin is not None
                else 999999.0
            )

            static_diff = (
                abs(float(static_diff))
                if static_diff is not None
                else 0.0
            )

            is_hard = (
                cls != "AGREEMENT"
                or regret > 0
                or margin <= 8
                or static_diff >= 8
            )

            if is_hard:
                repeats = 32

                if regret >= 16:
                    repeats = 48

                if regret >= 32:
                    repeats = 64

                hard[rn] = repeats

        anchors = sorted(
            set(pool_rows)
            - set(hard)
        )

        return hard, anchors, len(metrics)

    finally:
        con.close()


# ============================================================
# TRAINING DATASET
# ============================================================

def make_training_tsv(
    path,
    original_fields,
    pool_rows,
    replay_rows,
    hard,
    anchors,
    generation,
):
    training = []

    # Hard candidate-specific positions.
    for rn, repeats in hard.items():
        for _ in range(repeats):
            training.append(
                dict(pool_rows[rn])
            )

    # Positions already understood become anchors,
    # but receive much less weight.
    for rn in anchors:
        for _ in range(4):
            training.append(
                dict(pool_rows[rn])
            )

    # Broad replay protects general Stockfish strength.
    for rn in sorted(replay_rows):
        training.append(
            dict(replay_rows[rn])
        )

    rng = random.Random(
        540000 + generation
    )

    rng.shuffle(training)

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=original_fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(training)

    return len(training)


# ============================================================
# MATCH
# ============================================================

def match_candidate(
    champion_nnue,
    challenger_nnue,
    generation,
    outdir,
):
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = (
        outdir / "match.txt"
    )

    pgn = (
        outdir / "games.pgn"
    )

    seed = (
        500000
        + generation * 7919
    )

    cmd = [
        CCHESS,

        "-each",
        f"nodes={MATCH_NODES}",
        "option.Hash=64",
        "option.Threads=1",

        "-engine",
        f"cmd={SF}",
        "name=CHAMPION",
        f"option.EvalFile={champion_nnue}",

        "-engine",
        f"cmd={SF}",
        "name=CHALLENGER",
        f"option.EvalFile={challenger_nnue}",

        "-games",
        str(MATCH_GAMES),

        "-concurrency",
        "8",

        "-openings",
        f"file={BOOK}",
        "order=random",
        f"srand={seed}",

        "-repeat",

        "-pgn",
        str(pgn),
        "2",
    ]

    run_logged(
        cmd,
        REPO,
        logfile,
    )

    text = logfile.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = re.findall(
        r"Score of CHAMPION vs CHALLENGER:"
        r"\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)",
        text,
    )

    if not matches:
        raise RuntimeError(
            "Could not parse c-chess result."
        )

    champ_wins, chall_wins, draws = (
        map(int, matches[-1])
    )

    total = (
        champ_wins
        + chall_wins
        + draws
    )

    challenger_score = (
        chall_wins
        + 0.5 * draws
    ) / total

    return {
        "champion_wins": champ_wins,
        "challenger_wins": chall_wins,
        "draws": draws,
        "games": total,
        "challenger_score":
            challenger_score,
        "seed": seed,
    }


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    validate_inputs()

    state = load_or_create_state()

    # A previous clean STOP request must not prevent resume.
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    order = build_or_load_order()

    if len(order) < (
        POOL_SIZE
        + REPLAY_SIZE
    ):
        raise RuntimeError(
            "Eligible TRAIN order too small."
        )

    # Establish fixed DEV for the current parent once.
    if state.get(
        "champion_dev"
    ) is None:
        log(
            "Evaluating initial champion "
            "on frozen DEV."
        )

        state["champion_dev"] = eval_pt(
            state["champion_pt"],
            ROOT / "INITIAL_CHAMPION_DEV.txt",
        )

        atomic_json(
            STATE,
            state,
        )

        log(
            "Champion DEV:",
            state["champion_dev"],
        )

    state["status"] = "RUNNING"
    state["session_started"] = (
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    atomic_json(
        STATE,
        state,
    )

    log(
        "Continuous mode active: "
        "running until manual STOP."
    )

    try:
        while True:
            check_stop()

            generation = int(
                state["next_gen"]
            )

            gen_name = (
                f"GEN_{generation:04d}"
            )

            gen_dir = (
                ROOT
                / "GENERATIONS"
                / gen_name
            )

            # If power/process died during this same
            # generation, preserve the partial evidence
            # and rerun cleanly from the last committed state.
            if gen_dir.exists():
                stamp = time.strftime(
                    "%Y%m%d_%H%M%S"
                )

                log(
                    "Existing incomplete generation kept:",
                    gen_dir,
                )

                gen_dir = Path(
                    str(gen_dir)
                    + "_RETRY_"
                    + stamp
                )

                log(
                    "Retry generation directory:",
                    gen_dir,
                )

            gen_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            gen_start = time.time()

            log(
                "================================================"
            )
            log(
                gen_name,
                "| champion V",
                state["champion_version"],
                "| cursor",
                state["cursor"],
                "| continuous mode",
            )
            log(
                "================================================"
            )

            cursor = int(
                state["cursor"]
            )

            pool_nums = take_circular(
                order,
                cursor,
                POOL_SIZE,
            )

            pool_set = set(
                pool_nums
            )

            replay_start = (
                cursor
                + len(order) // 3
                + generation * REPLAY_SIZE
            ) % len(order)

            replay_nums = take_circular(
                order,
                replay_start,
                REPLAY_SIZE,
                exclude=pool_set,
            )

            all_needed = (
                pool_nums
                + replay_nums
            )

            fields, rows = (
                read_train_rows(
                    all_needed
                )
            )

            pool_rows = {
                rn: rows[rn]
                for rn in pool_nums
            }

            replay_rows = {
                rn: rows[rn]
                for rn in replay_nums
            }

            manifest = (
                gen_dir / "MINING_POOL.tsv"
            )

            make_manifest(
                manifest,
                pool_nums,
                pool_rows,
                fields,
                generation,
            )

            db_path = (
                gen_dir / "mining.sqlite"
            )

            create_db(db_path)

            run_tag = (
                f"C54_{gen_name}"
            )

            manifest_sha = (
                sha256_file(manifest)
            )

            log(
                "Mining",
                POOL_SIZE,
                "fresh positions against "
                "current champion."
            )

            run_logged(
                [
                    sys.executable,
                    MINER,
                    "--stockfish",
                    SF,
                    "--official-net",
                    OFFICIAL,
                    "--candidate-net",
                    state["champion_nnue"],
                    "--db",
                    db_path,
                    "--batch",
                    manifest,
                    "--run-tag",
                    run_tag,
                    "--expected-batch-sha",
                    manifest_sha,
                ],
                AL,
                gen_dir / "mining.log",
            )

            run_logged(
                [
                    sys.executable,
                    METRICS,
                    "--db",
                    db_path,
                    "--run-tag",
                    run_tag,
                    "--write",
                ],
                AL,
                gen_dir / "metrics.log",
            )

            hard, anchors, metric_count = (
                classify_pool(
                    db_path,
                    run_tag,
                    pool_rows,
                )
            )

            log(
                "Metrics:",
                metric_count,
                "| hard:",
                len(hard),
                "| anchors:",
                len(anchors),
            )

            # No useful frontier discovered:
            # move on rather than perturbing weights.
            if len(hard) < 8:
                log(
                    "Too few hard positions; "
                    "skip weight update."
                )

                elapsed = (
                    time.time()
                    - gen_start
                )

                state["history"].append(
                    {
                        "generation":
                            generation,
                        "status":
                            "NO_TRAIN_LOW_INFORMATION",
                        "hard":
                            len(hard),
                        "anchors":
                            len(anchors),
                        "seconds":
                            elapsed,
                    }
                )

                new_cursor = (
                    cursor
                    + POOL_SIZE
                )

                if new_cursor >= len(order):
                    state["cycle"] += 1

                state["cursor"] = (
                    new_cursor
                    % len(order)
                )

                state["next_gen"] = (
                    generation + 1
                )

                state["avg_gen_seconds"] = (
                    0.8
                    * float(
                        state.get(
                            "avg_gen_seconds",
                            elapsed,
                        )
                    )
                    + 0.2 * elapsed
                )

                atomic_json(
                    STATE,
                    state,
                )

                (
                    gen_dir / "DONE"
                ).write_text(
                    "LOW_INFORMATION\n",
                    encoding="utf-8",
                )

                continue

            train_tsv = (
                gen_dir
                / "TARGETED_TRAIN.tsv"
            )

            training_rows = (
                make_training_tsv(
                    train_tsv,
                    fields,
                    pool_rows,
                    replay_rows,
                    hard,
                    anchors,
                    generation,
                )
            )

            log(
                "Targeted training rows:",
                training_rows,
            )

            train_bin = (
                gen_dir
                / "TARGETED_TRAIN.binpack"
            )

            run_logged(
                [
                    CONVERTER,
                    train_tsv,
                    train_bin,
                ],
                gen_dir,
                gen_dir
                / "binpack.log",
            )

            train_root = (
                gen_dir
                / "TRAIN_RUN"
            )

            seed = (
                42000
                + generation
            )

            log(
                "Fine-tuning from champion:",
                state["champion_pt"],
            )

            run_logged(
                [
                    sys.executable,
                    REPO / "train.py",
                    train_bin,

                    "--resume-from-model",
                    state["champion_pt"],

                    "--default-root-dir",
                    train_root,

                    "--accelerator",
                    "cuda",

                    "--gpus",
                    "0",

                    "--max-epochs",
                    "1",

                    "--epoch-size",
                    str(EPOCH_SIZE),

                    "--batch-size",
                    str(BATCH_SIZE),

                    "--seed",
                    str(seed),

                    "--lr",
                    str(LR),

                    "--lambda",
                    "1.0",

                    "--optimizer-name",
                    "rangerlite",

                    "--validation-datasets",
                    VAL_BIN,

                    "--validation-size",
                    "32768",

                    "--check-val-every-n-epoch",
                    "1",
                ],
                REPO,
                gen_dir / "train.log",
            )

            ckpts = list(
                train_root.rglob(
                    "last.ckpt"
                )
            )

            if not ckpts:
                raise RuntimeError(
                    "No last.ckpt produced."
                )

            ckpt = max(
                ckpts,
                key=lambda p:
                    p.stat().st_mtime,
            )

            challenger_pt = (
                gen_dir
                / "challenger.pt"
            )

            challenger_nnue = (
                gen_dir
                / "challenger.nnue"
            )

            run_logged(
                [
                    sys.executable,
                    REPO / "serialize.py",
                    ckpt,
                    challenger_pt,
                ],
                REPO,
                gen_dir
                / "serialize_pt.log",
            )

            run_logged(
                [
                    sys.executable,
                    REPO / "serialize.py",
                    ckpt,
                    challenger_nnue,
                    "--description",
                    (
                        "C54 iterative "
                        + gen_name
                        + " LR=5e-6"
                    ),
                ],
                REPO,
                gen_dir
                / "serialize_nnue.log",
            )

            candidate_dev = eval_pt(
                challenger_pt,
                gen_dir
                / "fixed_dev.txt",
            )

            parent_dev = float(
                state["champion_dev"]
            )

            log(
                "DEV parent:",
                parent_dev,
                "| challenger:",
                candidate_dev,
            )

            result = match_candidate(
                state["champion_nnue"],
                challenger_nnue,
                generation,
                gen_dir / "MATCH",
            )

            score = float(
                result[
                    "challenger_score"
                ]
            )

            dev_ok = (
                candidate_dev
                <= parent_dev
                * MAX_DEV_RELATIVE
            )

            match_ok = (
                score
                >= PROMOTE_SCORE
            )

            promote = (
                dev_ok
                and match_ok
            )

            result.update(
                {
                    "generation":
                        generation,
                    "hard_positions":
                        len(hard),
                    "anchors":
                        len(anchors),
                    "training_rows":
                        training_rows,
                    "lr":
                        LR,
                    "epoch_size":
                        EPOCH_SIZE,
                    "parent_dev":
                        parent_dev,
                    "challenger_dev":
                        candidate_dev,
                    "dev_ok":
                        dev_ok,
                    "match_ok":
                        match_ok,
                    "promoted":
                        promote,
                    "challenger_pt_sha256":
                        sha256_file(
                            challenger_pt
                        ),
                    "challenger_nnue_sha256":
                        sha256_file(
                            challenger_nnue
                        ),
                }
            )

            (
                gen_dir
                / "RESULT.json"
            ).write_text(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            if promote:
                state[
                    "champion_version"
                ] += 1

                state[
                    "champion_pt"
                ] = str(
                    challenger_pt
                )

                state[
                    "champion_nnue"
                ] = str(
                    challenger_nnue
                )

                state[
                    "champion_dev"
                ] = candidate_dev

                log(
                    "🏆 PROMOTED → champion V",
                    state[
                        "champion_version"
                    ],
                    "| observed score",
                    round(
                        score * 100,
                        2,
                    ),
                    "%",
                )

            else:
                log(
                    "REJECTED / ROLLBACK",
                    "| observed challenger score",
                    round(
                        score * 100,
                        2,
                    ),
                    "%",
                    "| dev_ok=",
                    dev_ok,
                )

            elapsed = (
                time.time()
                - gen_start
            )

            result[
                "seconds"
            ] = elapsed

            state[
                "history"
            ].append(result)

            new_cursor = (
                cursor
                + POOL_SIZE
            )

            if new_cursor >= len(order):
                state["cycle"] += 1

            state["cursor"] = (
                new_cursor
                % len(order)
            )

            state["next_gen"] = (
                generation + 1
            )

            state[
                "avg_gen_seconds"
            ] = (
                0.8
                * float(
                    state.get(
                        "avg_gen_seconds",
                        elapsed,
                    )
                )
                + 0.2 * elapsed
            )

            state["status"] = (
                "RUNNING"
            )

            atomic_json(
                STATE,
                state,
            )

            (
                gen_dir / "DONE"
            ).write_text(
                "PROMOTED\n"
                if promote
                else "REJECTED\n",
                encoding="utf-8",
            )

            log(
                gen_name,
                "complete in",
                round(
                    elapsed / 60,
                    1,
                ),
                "minutes.",
            )

    except CleanStop:
        log(
            "Clean STOP requested."
        )

        state[
            "status"
        ] = "STOPPED_CLEANLY"

        atomic_json(
            STATE,
            state,
        )

        return

    except KeyboardInterrupt:
        log(
            "Interrupted by user."
        )

        state[
            "status"
        ] = "INTERRUPTED"

        atomic_json(
            STATE,
            state,
        )

        return

    except Exception as exc:
        state["status"] = "ERROR"
        state["error"] = repr(exc)

        atomic_json(
            STATE,
            state,
        )

        log(
            "FATAL ERROR:",
            repr(exc),
        )

        raise

    state["status"] = (
        "TIME_BUDGET_COMPLETE"
    )

    state[
        "session_finished"
    ] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    atomic_json(
        STATE,
        state,
    )

    log(
        "================================================"
    )
    log(
        "OVERNIGHT SESSION COMPLETE"
    )
    log(
        "Current champion V",
        state["champion_version"],
    )
    log(
        "Next generation:",
        state["next_gen"],
    )
    log(
        "TRAIN cursor:",
        state["cursor"],
        "/",
        len(order),
    )
    log(
        "================================================"
    )


if __name__ == "__main__":
    main()
