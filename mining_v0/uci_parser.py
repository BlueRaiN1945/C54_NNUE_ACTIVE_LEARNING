from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class StaticEvalResult:
    status: str
    score_raw_stm: int | None
    error_text: str | None = None


@dataclass(frozen=True)
class SearchInfo:
    rank: int
    score_type: str
    score_value: int
    bound: str

    wdl_w: int | None
    wdl_d: int | None
    wdl_l: int | None

    depth: int | None
    seldepth: int | None
    nodes: int | None
    time_ms: int | None
    nps: int | None

    move: str | None
    pv: str


@dataclass(frozen=True)
class SearchResult:
    infos: dict[int, SearchInfo]
    bestmove: str | None
    ponder: str | None


_STATIC_RE = re.compile(
    r"^NNUE evaluation\s+([+-]?\d+)"
    r"\s+\(side to move, internal units\)\s*$"
)


def parse_static_eval(lines: Iterable[str]) -> StaticEvalResult:
    score = None
    in_check = False

    for raw in lines:
        line = raw.strip()

        m = _STATIC_RE.match(line)
        if m:
            score = int(m.group(1))

        if line.startswith("Final evaluation: none"):
            in_check = True

    if in_check:
        return StaticEvalResult(
            status="in_check",
            score_raw_stm=None,
        )

    if score is None:
        return StaticEvalResult(
            status="parse_error",
            score_raw_stm=None,
            error_text="Raw NNUE internal-unit score not found",
        )

    return StaticEvalResult(
        status="ok",
        score_raw_stm=score,
    )


def _int_after(tokens: list[str], key: str) -> int | None:
    try:
        i = tokens.index(key)
        return int(tokens[i + 1])
    except (ValueError, IndexError):
        return None


def parse_info_line(line: str) -> SearchInfo | None:
    tokens = line.strip().split()

    if not tokens or tokens[0] != "info":
        return None

    if "score" not in tokens or "pv" not in tokens:
        return None

    rank = _int_after(tokens, "multipv") or 1

    try:
        si = tokens.index("score")
        score_type = tokens[si + 1]
        score_value = int(tokens[si + 2])
    except (ValueError, IndexError):
        return None

    if score_type not in ("cp", "mate"):
        return None

    tail = tokens[si + 3:]

    if "lowerbound" in tail:
        bound = "lowerbound"
    elif "upperbound" in tail:
        bound = "upperbound"
    else:
        bound = "exact"

    w = d = l = None

    if "wdl" in tokens:
        try:
            wi = tokens.index("wdl")
            w = int(tokens[wi + 1])
            d = int(tokens[wi + 2])
            l = int(tokens[wi + 3])
        except (ValueError, IndexError):
            return None

    try:
        pi = tokens.index("pv")
        pv_tokens = tokens[pi + 1:]
    except ValueError:
        pv_tokens = []

    return SearchInfo(
        rank=rank,
        score_type=score_type,
        score_value=score_value,
        bound=bound,
        wdl_w=w,
        wdl_d=d,
        wdl_l=l,
        depth=_int_after(tokens, "depth"),
        seldepth=_int_after(tokens, "seldepth"),
        nodes=_int_after(tokens, "nodes"),
        time_ms=_int_after(tokens, "time"),
        nps=_int_after(tokens, "nps"),
        move=pv_tokens[0] if pv_tokens else None,
        pv=" ".join(pv_tokens),
    )


def parse_bestmove_line(line: str):
    tokens = line.strip().split()

    if not tokens or tokens[0] != "bestmove":
        return None, None

    bestmove = tokens[1] if len(tokens) >= 2 else None

    if bestmove in ("(none)", "0000"):
        bestmove = None

    ponder = None

    if len(tokens) >= 4 and tokens[2] == "ponder":
        ponder = tokens[3]

        if ponder in ("(none)", "0000"):
            ponder = None

    return bestmove, ponder


def parse_search_output(
    lines: Iterable[str],
    expected_multipv: int = 1,
) -> SearchResult:
    if expected_multipv < 1:
        raise ValueError(
            "expected_multipv must be >= 1"
        )

    bestmove = None
    ponder = None

    # Fallback evidence if no complete coherent exact snapshot
    # exists. This preserves V0's last-exact-per-rank behavior,
    # but it is used only as a fallback.
    latest_any: dict[int, SearchInfo] = {}
    latest_exact: dict[int, SearchInfo] = {}

    # Coherent MultiPV round currently being assembled.
    #
    # Stockfish normally emits:
    #
    #   multipv 1 ... depth D
    #   multipv 2 ... depth D
    #   ...
    #   multipv N ... depth D
    #
    # A new rank-1 line starts a new candidate round.
    current_depth: int | None = None
    current_round: dict[int, SearchInfo] = {}

    # Last complete exact round seen in the stream.
    coherent_snapshot: dict[int, SearchInfo] | None = None

    for raw in lines:
        line = raw.strip()

        if line.startswith("info "):
            info = parse_info_line(line)

            if info is None:
                continue

            latest_any[info.rank] = info

            if info.bound == "exact":
                latest_exact[info.rank] = info

            # ------------------------------------------------
            # SinglePV
            # ------------------------------------------------

            if expected_multipv == 1:
                if (
                    info.rank == 1
                    and info.bound == "exact"
                ):
                    coherent_snapshot = {
                        1: info
                    }

                continue

            # ------------------------------------------------
            # MultiPV coherent-round assembly
            # ------------------------------------------------

            if info.rank == 1:
                current_depth = info.depth
                current_round = {}

                if info.bound == "exact":
                    current_round[1] = info

            else:
                # Rank > 1 only belongs to the currently-open
                # round if it has the same depth.
                if (
                    current_depth is not None
                    and info.depth == current_depth
                    and info.bound == "exact"
                ):
                    current_round[
                        info.rank
                    ] = info

            # A valid coherent snapshot must contain every
            # expected rank and every row must be exact and at
            # the same depth.
            expected_ranks = set(
                range(
                    1,
                    expected_multipv + 1,
                )
            )

            if (
                set(current_round)
                == expected_ranks
            ):
                depths = {
                    i.depth
                    for i in (
                        current_round.values()
                    )
                }

                if len(depths) == 1:
                    coherent_snapshot = dict(
                        current_round
                    )

        elif line.startswith("bestmove "):
            bestmove, ponder = (
                parse_bestmove_line(line)
            )

    # --------------------------------------------------------
    # Preferred result:
    # last complete exact coherent snapshot.
    # --------------------------------------------------------

    if coherent_snapshot is not None:
        infos = {
            rank: coherent_snapshot[rank]
            for rank in sorted(
                coherent_snapshot
            )
        }

        return SearchResult(
            infos=infos,
            bestmove=bestmove,
            ponder=ponder,
        )

    # --------------------------------------------------------
    # Conservative fallback:
    # last exact per rank, else last any.
    #
    # Caller can detect incompleteness by checking rank count
    # and/or bestmove-vs-rank1 consistency.
    # --------------------------------------------------------

    infos = {}

    for rank in sorted(latest_any):
        infos[rank] = latest_exact.get(
            rank,
            latest_any[rank],
        )

    return SearchResult(
        infos=infos,
        bestmove=bestmove,
        ponder=ponder,
    )


# ============================================================
# POSITION DISPLAY parsing
# ============================================================

def parse_display_fen(lines: Iterable[str]) -> str | None:
    """
    Parse the exact FEN emitted by Stockfish's `d` command.

    Example:
        Fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    """
    for raw in lines:
        line = raw.strip()

        if line.startswith("Fen: "):
            return line[5:].strip()

    return None
