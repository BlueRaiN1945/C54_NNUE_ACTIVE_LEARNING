#!/usr/bin/env python3
"""Rescore arbitrary FEN positions with Stockfish over UCI.

M0 corpus pipeline step 3. A generic variant of BigPC's existing
rescore_c54_exact.py, with exactly one schema change: INPUT_REQUIRED is
["fen"] instead of the seven OLA-specific columns
(fen, source, weight, gen2_class, gen2_frequency, ola_games, ola_moves).
Everything else -- UCI engine interaction, fixed node budget, per-row error
recording, meta.json provenance -- follows the same pattern, because that
pattern is already correct and already proven across hundreds of thousands
of real rows; only the schema coupling was OLA-specific, so only that changes
here.

THIS SCRIPT IS FOR THE EXECUTION HOST (BigPC), NOT MEDIUM-PC.
Importing it does nothing but define pure functions and classes -- no engine
is spawned, no file is touched -- until __main__ runs (see the bottom of this
file) or UCIEngine() is explicitly constructed. Medium-PC's own tests
(tests/test_rescore_generic.py) import ONLY parse_info_line() and
validate_input_header(), both pure and side-effect-free, and never construct
a UCIEngine. Deploying this file to BigPC and running it there requires
separate, explicit authorization; writing or testing it here does not
constitute deployment.

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# The one schema change relative to rescore_c54_exact.py: only "fen" is
# required. A generic modern-corpus TSV carries no OLA/GEN2 metadata columns,
# and it must not be made to invent any just to satisfy this script.
INPUT_REQUIRED = ["fen"]

ADDED_COLUMNS = [
    "sf_score_type",  # cp | mate
    "sf_score",  # raw UCI score from side-to-move perspective
    "sf_bound",  # exact | lowerbound | upperbound
    "sf_wdl_w",  # Stockfish model WDL, side-to-move perspective
    "sf_wdl_d",
    "sf_wdl_l",
    "sf_depth",
    "sf_seldepth",
    "sf_nodes",
    "sf_time_ms",
    "sf_nps",
    "sf_bestmove",
    "sf_ponder",
    "sf_pv",
    "sf_error",
]


class RescoreGenericError(ValueError):
    """Raised for malformed input (e.g. a missing 'fen' column)."""


@dataclass
class Analysis:
    sf_score_type: str = ""
    sf_score: str = ""
    sf_bound: str = ""
    sf_wdl_w: str = ""
    sf_wdl_d: str = ""
    sf_wdl_l: str = ""
    sf_depth: str = ""
    sf_seldepth: str = ""
    sf_nodes: str = ""
    sf_time_ms: str = ""
    sf_nps: str = ""
    sf_bestmove: str = ""
    sf_ponder: str = ""
    sf_pv: str = ""
    sf_error: str = ""


def parse_info_line(line: str) -> Analysis | None:
    """Parse ONE complete UCI "info ..." search line into an Analysis.

    Pure and side-effect-free: no engine, no I/O. All fields are taken from
    this single line only -- nothing is carried over from earlier or later
    info lines. Returns None for any line that is not a scored info line
    (info string, no score token, or malformed tail).
    """

    if not line.startswith("info ") or line.startswith("info string"):
        return None

    t = line.split()

    if "score" not in t:
        return None

    try:
        i = t.index("score")

        if i + 2 >= len(t) or t[i + 1] not in ("cp", "mate"):
            return None

        a = Analysis()
        a.sf_score_type = t[i + 1]
        a.sf_score = t[i + 2]

        a.sf_bound = "exact"
        score_tail = t[i + 3 :]

        if "lowerbound" in score_tail:
            a.sf_bound = "lowerbound"
        elif "upperbound" in score_tail:
            a.sf_bound = "upperbound"

        if "depth" in t:
            a.sf_depth = t[t.index("depth") + 1]
        if "seldepth" in t:
            a.sf_seldepth = t[t.index("seldepth") + 1]
        if "nodes" in t:
            a.sf_nodes = t[t.index("nodes") + 1]
        if "time" in t:
            a.sf_time_ms = t[t.index("time") + 1]
        if "nps" in t:
            a.sf_nps = t[t.index("nps") + 1]

        if "wdl" in t:
            j = t.index("wdl")
            if j + 3 < len(t):
                a.sf_wdl_w, a.sf_wdl_d, a.sf_wdl_l = t[j + 1 : j + 4]

        if "pv" in t:
            j = t.index("pv")
            a.sf_pv = " ".join(t[j + 1 :])

        return a

    except (ValueError, IndexError):
        return None


def validate_input_header(header: Sequence[str]) -> None:
    """Raise RescoreGenericError unless every column in INPUT_REQUIRED is present.

    Pure and side-effect-free: takes an already-parsed header list, does not
    open any file itself.
    """

    missing = [c for c in INPUT_REQUIRED if c not in header]

    if missing:
        raise RescoreGenericError(
            f"input is missing required columns: {', '.join(missing)}"
        )


class UCIEngine:
    """A persistent UCI engine subprocess. Constructing one launches a process.

    Never constructed by tests/test_rescore_generic.py -- see this module's
    top docstring. Structurally identical to rescore_c54_exact.py's engine
    wrapper on BigPC; reproduced here rather than imported because that
    script lives on BigPC, outside this repository.
    """

    def __init__(self, path: Path, hash_mb: int):
        import subprocess

        self.path = path
        self.hash_mb = hash_mb
        self.proc: subprocess.Popen | None = None
        self.engine_name = ""
        self.start()

    def start(self) -> None:
        import subprocess

        self.close()
        self.proc = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._send("uci")
        deadline = time.time() + 30
        while True:
            if time.time() > deadline:
                raise RuntimeError("Timeout waiting for uciok")
            line = self._readline()
            if line.startswith("id name "):
                self.engine_name = line[8:].strip()
            if line == "uciok":
                break
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("setoption name MultiPV value 1")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("isready")
        self._wait_for("readyok", 30)

    def close(self) -> None:
        p = self.proc
        self.proc = None
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    if p.stdin is not None:
                        p.stdin.write("quit\n")
                        p.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    try:
                        p.kill()
                        p.wait(timeout=2)
                    except Exception:
                        pass
        finally:
            for stream in (p.stdin, p.stdout, p.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    def _send(self, cmd: str) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Engine is not running")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("Engine is not running")
        line = self.proc.stdout.readline()
        if line == "":
            rc = self.proc.poll()
            raise RuntimeError(f"Engine terminated unexpectedly (exit={rc})")
        return line.rstrip("\r\n")

    def _wait_for(self, exact: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise RuntimeError(f"Timeout waiting for {exact}")
            if self._readline() == exact:
                return

    def analyse(self, fen: str, nodes: int) -> Analysis:
        last_any: Analysis | None = None
        last_exact: Analysis | None = None

        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")

        while True:
            line = self._readline()

            if line.startswith("info "):
                candidate = parse_info_line(line)
                if candidate is not None:
                    last_any = candidate
                    if candidate.sf_bound == "exact":
                        last_exact = candidate

            elif line.startswith("bestmove "):
                a = last_exact or last_any or Analysis()
                t = line.split()
                final_bestmove = t[1] if len(t) >= 2 else ""
                final_ponder = t[3] if len(t) >= 4 and t[2] == "ponder" else ""

                pv_moves = a.sf_pv.split()
                if pv_moves:
                    a.sf_bestmove = pv_moves[0]
                    a.sf_ponder = pv_moves[1] if len(pv_moves) >= 2 else ""
                else:
                    a.sf_bestmove = final_bestmove
                    a.sf_ponder = final_ponder

                return a


_thread_local = threading.local()
_all_engines: list[UCIEngine] = []
_all_engines_lock = threading.Lock()
_engine_path: Path
_hash_mb: int
_nodes: int


def get_engine() -> UCIEngine:
    eng = getattr(_thread_local, "engine", None)
    if eng is None:
        eng = UCIEngine(_engine_path, _hash_mb)
        _thread_local.engine = eng
        with _all_engines_lock:
            _all_engines.append(eng)
    return eng


def analyse_row(row: dict[str, str]) -> dict[str, str]:
    last_err = ""
    for attempt in range(2):
        try:
            a = get_engine().analyse(row["fen"], _nodes)
            out = dict(row)
            out.update(asdict(a))
            return out
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            eng = getattr(_thread_local, "engine", None)
            if eng is not None:
                try:
                    eng.close()
                    eng.start()
                except Exception as restart_exc:
                    last_err += f" | restart: {type(restart_exc).__name__}: {restart_exc}"
            if attempt == 0:
                continue
    out = dict(row)
    a = Analysis(sf_error=last_err[:1000])
    out.update(asdict(a))
    return out


def iter_input_rows(
    path: Path, skip_rows: int, limit: int | None
) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for _ in range(skip_rows):
            try:
                next(r)
            except StopIteration:
                return
        for n, row in enumerate(r):
            if limit is not None and n >= limit:
                break
            yield row


def write_meta(
    path: Path, args: argparse.Namespace, engine_name: str, total_rows: int
) -> None:
    meta = {
        "engine_path": str(args.engine.resolve()),
        "engine_name": engine_name,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "nodes_per_position": args.nodes,
        "workers": args.workers,
        "hash_mb_per_worker": args.hash,
        "input_rows": total_rows,
        "score_perspective": "side-to-move (raw UCI score)",
        "wdl_perspective": "side-to-move (Stockfish UCI_ShowWDL model)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Rescore arbitrary FEN positions with Stockfish (generic, non-OLA)."
    )
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("rescored.tsv"))
    p.add_argument("--nodes", type=int, default=25000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hash", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()

    if args.nodes < 1 or args.workers < 1 or args.hash < 1 or args.batch < 1:
        p.error("--nodes, --workers, --hash and --batch must be positive")
    if not args.engine.exists():
        p.error(f"Engine not found: {args.engine}")
    if not args.input.exists():
        p.error(f"Input not found: {args.input}")

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f, delimiter="\t"))
    validate_input_header(header)

    global _engine_path, _hash_mb, _nodes
    _engine_path = args.engine.resolve()
    _hash_mb = args.hash
    _nodes = args.nodes

    probe = UCIEngine(args.engine, args.hash)
    engine_name = probe.engine_name or "Stockfish"
    probe.close()

    output_header = header + ADDED_COLUMNS
    rows_iter = iter_input_rows(args.input, 0, args.limit)

    with args.output.open(
        "w", encoding="utf-8", newline=""
    ) as outf, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        w = csv.DictWriter(
            outf,
            fieldnames=output_header,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()

        while True:
            batch: list[dict[str, str]] = []
            try:
                for _ in range(args.batch):
                    batch.append(next(rows_iter))
            except StopIteration:
                pass
            if not batch:
                break
            for result in ex.map(analyse_row, batch):
                w.writerow(result)
            outf.flush()

    write_meta(
        args.output.with_suffix(args.output.suffix + ".meta.json"),
        args,
        engine_name,
        0,
    )

    with _all_engines_lock:
        engines = list(_all_engines)
    for eng in engines:
        try:
            eng.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
