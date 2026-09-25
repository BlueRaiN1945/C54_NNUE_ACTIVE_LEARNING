"""PGN parsing for the research evidence plane (Phase 3).

Builds on the frozen Phase 0 schemas (medium_pc_audit.schemas.registry).
Phase 0, Phase 1, and Phase 2 modules are not modified here.

Dependency boundary (locked, with two documented exceptions): this module is
allowed to import the third-party `chess` package (PyPI name 'chess', the
current name for what used to be distributed as 'python-chess'; pinned
exactly, see medium_pc_audit/third_party_deps/chess.lock.json). All other production modules under medium_pc_audit remain stdlib-only,
with exactly two deliberate exceptions:
medium_pc_audit.position_extract, which needs full
movetext/move-legality parsing for M0 corpus position extraction -- a
different concern from this module's header-only, tolerant-of-malformed-PGN
audit parsing, and one this module's own invariants (below) rule out serving;
and medium_pc_audit.clean_room_openings, which enumerates legal moves from
the standard starting position to build a from-first-principles opening
artifact and parses no PGN of any kind, so it belongs in neither this module
nor position_extract. None of the three modules may be merged: this one must
keep skipping movetext (robustness to any engine's real-world PGN quirks
matters more here than position fidelity), position_extract must keep
requiring fully legal, parseable movetext (position fidelity matters more
there than tolerance), and clean_room_openings must keep reading no PGN at
all (it starts only from the standard initial position).

This module parses real c-chess-cli PGN output and independently
reconstructs candidate-perspective win/loss/draw results. It never trusts
an engine/harness's self-reported claimed_wins/claimed_losses/
claimed_draws -- recompute_wdl_from_games() takes no claimed_* arguments at
all, so it is structurally impossible for a claimed count to leak into the
recomputed count.

Locked design invariants:
  - Games are delimited by calling chess.pgn.read_game() repeatedly until
    EOF (it returns None at end of file). [Event] is NEVER used as a game
    delimiter -- real c-chess-cli PGNs may omit it entirely, and headers
    other than White/Black/Result/FEN/Termination/Round/PlyCount are not
    required.
  - Only headers are parsed, via chess.pgn.HeadersBuilder, which causes the
    parser to skip movetext entirely (no SAN tokenizing, no board legality
    checks). This is deliberate: this is an audit parser for W/L/D
    outcomes, not a move-legality validator, and must never raise or
    misbehave on an engine-vs-engine game whose moves the parser's own
    rules engine would reject.
  - candidate_label/opponent_label are supplied explicitly by the caller
    (read off the real BigPC screen/run), never hard-coded here -- a real
    audit pairs one candidate against many different frozen opponents
    across separate runs.
  - Candidate-perspective win/loss/draw is derived from the White/Black
    engine-label headers together with the raw PGN Result header, never
    from color alone (the candidate plays both colors across a match).
  - opening_ref, required by the frozen Phase 0 RawResultV1 schema, is
    derived from the normalized first four FEN header fields (piece
    placement, side-to-move, castling rights, en-passant square) when a
    usable FEN header is present. A game with no usable FEN has no
    opening_ref and is EXCLUDED from the RawResultV1 'games' list --
    explicitly, via `excluded`, never silently dropped.
  - recompute_wdl_from_games() does not depend on opening_ref availability
    at all, so a future schema revision could retain valid W/L/D even when
    pairing/opening identity is unavailable. This module does not change
    the frozen Phase 0 schema.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TextIO, Union

import chess.pgn

from medium_pc_audit.schemas import registry

_VALID_PGN_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})

REASON_MISSING_WHITE_OR_BLACK = "missing_white_or_black_header"
REASON_MISSING_RESULT = "missing_result_header"
REASON_UNRESOLVED_OR_INVALID_RESULT = "unresolved_or_invalid_result"
REASON_UNRECOGNIZED_ENGINE_LABELS = "unrecognized_engine_labels"
REASON_MISSING_OPENING_REF = "missing_or_invalid_fen_for_opening_ref"


class PgnParseError(ValueError):
    """Raised for invalid call-level arguments (e.g. bad labels).

    Never raised for a malformed/unexpected individual game inside a PGN
    stream -- those are reported explicitly via ParsedGame.exclusion_reason
    and PgnParseOutcome.excluded instead of raising.
    """


@dataclass(frozen=True)
class ParsedGame:
    """One game's parsed header state, independent of whether it is usable
    for RawResultV1 construction.

    candidate_outcome is deliberately computed WITHOUT regard to
    opening_ref availability, so that recompute_wdl_from_games() can stay
    correct even for games that lack a usable FEN/opening_ref -- this is
    what lets a future schema revision retain valid W/L/D independent of
    pairing/opening identity, without touching the frozen Phase 0 schema
    now.
    """

    game_index: int
    white: Optional[str]
    black: Optional[str]
    result: Optional[str]
    termination: Optional[str]
    fen: Optional[str]
    opening_ref: Optional[str]
    candidate_role: Optional[str]  # "white" | "black" | None (unrecognized labels)
    candidate_outcome: Optional[str]  # "win" | "loss" | "draw" | None (undetermined)
    exclusion_reason: Optional[str]  # None means usable for RawResultV1's 'games' list


@dataclass(frozen=True)
class PgnParseOutcome:
    parsed_games: list
    raw_result_games: list
    excluded: list
    recomputed_wdl: dict


def _require_label(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PgnParseError(f"{name} must be a non-empty string, got {value!r}")
    return value


def normalize_opening_ref_from_fen(fen: Any) -> Optional[str]:
    """Derive opening_ref from the normalized first four FEN fields (piece
    placement, side-to-move, castling rights, en-passant square).

    Deliberately excludes the halfmove clock and fullmove number, which
    vary with when in the game the FEN happened to be recorded and are not
    part of opening/position identity. Returns None for anything that is
    not a string with at least four whitespace-separated fields (missing,
    empty, or malformed FEN).
    """
    if not isinstance(fen, str):
        return None
    fields = fen.split()
    if len(fields) < 4:
        return None
    return " ".join(fields[:4])


def _determine_candidate_role(
    *, white: Optional[str], black: Optional[str], candidate_label: str, opponent_label: str
) -> Optional[str]:
    if white == candidate_label and black == opponent_label:
        return "white"
    if white == opponent_label and black == candidate_label:
        return "black"
    return None


def _determine_candidate_outcome(
    *, candidate_role: Optional[str], result: Optional[str]
) -> Optional[str]:
    if candidate_role is None or result not in _VALID_PGN_RESULTS:
        return None
    if result == "1/2-1/2":
        return "draw"
    white_won = result == "1-0"
    if (candidate_role == "white" and white_won) or (candidate_role == "black" and not white_won):
        return "win"
    return "loss"


def _determine_exclusion_reason(
    *,
    white: Optional[str],
    black: Optional[str],
    result: Optional[str],
    candidate_role: Optional[str],
    opening_ref: Optional[str],
) -> Optional[str]:
    if white is None or black is None:
        return REASON_MISSING_WHITE_OR_BLACK
    if result is None:
        return REASON_MISSING_RESULT
    if result not in _VALID_PGN_RESULTS:
        return f"{REASON_UNRESOLVED_OR_INVALID_RESULT}: Result header was {result!r}"
    if candidate_role is None:
        return (
            f"{REASON_UNRECOGNIZED_ENGINE_LABELS}: White={white!r} Black={black!r} "
            "matched neither (candidate, opponent) nor (opponent, candidate)"
        )
    if opening_ref is None:
        return REASON_MISSING_OPENING_REF
    return None


def _classify_game(
    game_index: int, headers, candidate_label: str, opponent_label: str
) -> ParsedGame:
    white = headers.get("White")
    black = headers.get("Black")
    result = headers.get("Result")
    termination = headers.get("Termination")
    fen = headers.get("FEN")

    candidate_role = _determine_candidate_role(
        white=white, black=black, candidate_label=candidate_label, opponent_label=opponent_label
    )
    opening_ref = normalize_opening_ref_from_fen(fen)
    candidate_outcome = _determine_candidate_outcome(candidate_role=candidate_role, result=result)
    exclusion_reason = _determine_exclusion_reason(
        white=white, black=black, result=result, candidate_role=candidate_role, opening_ref=opening_ref
    )

    return ParsedGame(
        game_index=game_index,
        white=white,
        black=black,
        result=result,
        termination=termination,
        fen=fen,
        opening_ref=opening_ref,
        candidate_role=candidate_role,
        candidate_outcome=candidate_outcome,
        exclusion_reason=exclusion_reason,
    )


def _iter_pgn_headers(handle: TextIO):
    while True:
        headers = chess.pgn.read_game(handle, Visitor=chess.pgn.HeadersBuilder)
        if headers is None:
            return
        yield headers


def _build_raw_result_games(parsed_games: Iterable[ParsedGame]):
    raw_games = []
    excluded = []
    for pg in parsed_games:
        if pg.exclusion_reason is not None:
            excluded.append({"game_index": pg.game_index, "reason": pg.exclusion_reason})
            continue

        entry = {
            "game_index": pg.game_index,
            "white_role": "candidate" if pg.candidate_role == "white" else "opponent",
            "black_role": "candidate" if pg.candidate_role == "black" else "opponent",
            "result": pg.result,
            "opening_ref": pg.opening_ref,
        }
        if pg.termination is not None:
            entry["termination"] = pg.termination
        raw_games.append(entry)
    return raw_games, excluded


def recompute_wdl_from_games(games: Iterable[ParsedGame]) -> dict:
    """Tally candidate-perspective wins/losses/draws purely from parsed
    game records.

    This function accepts ONLY `games` -- there is no claimed_wins/
    claimed_losses/claimed_draws parameter anywhere in this signature, so
    it is structurally impossible for a claimed count to affect the
    recomputed result. Games whose candidate_outcome could not be
    determined (unrecognized engine labels or an unresolved/invalid
    result) are counted in 'unresolved', not silently dropped from the
    total.
    """
    games = list(games)
    wins = losses = draws = unresolved = 0
    for g in games:
        outcome = g.candidate_outcome
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "draw":
            draws += 1
        else:
            unresolved += 1
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unresolved": unresolved,
        "total_games": len(games),
    }


def parse_pgn_games(
    pgn_source: Union[str, TextIO], *, candidate_label: str, opponent_label: str
) -> PgnParseOutcome:
    """Parse every game in pgn_source and classify it for both RawResultV1
    construction and independent WDL recomputation.

    pgn_source may be a PGN text string, or an already-open text-mode
    handle (caller-controlled encoding, e.g. `open(path, encoding="utf-8")`
    for real c-chess-cli output -- see parse_pgn_file() for the common
    case).

    candidate_label/opponent_label are the exact engine identity strings
    that appear in this PGN's White/Black headers (e.g. "RESEARCH_HEAD" /
    "FROZEN_V80"); they are never hard-coded and must be supplied by the
    caller for every run.

    game_index is assigned sequentially starting at 1, in PGN stream order,
    for every game encountered -- including games later excluded from the
    RawResultV1 'games' list -- so indices stay stable and no game is ever
    silently skipped from the accounting.
    """
    _require_label(candidate_label, name="candidate_label")
    _require_label(opponent_label, name="opponent_label")
    if candidate_label == opponent_label:
        raise PgnParseError(
            f"candidate_label and opponent_label must differ, both were {candidate_label!r}"
        )

    handle = io.StringIO(pgn_source) if isinstance(pgn_source, str) else pgn_source

    parsed_games = [
        _classify_game(game_index, headers, candidate_label, opponent_label)
        for game_index, headers in enumerate(_iter_pgn_headers(handle), start=1)
    ]

    raw_result_games, excluded = _build_raw_result_games(parsed_games)
    recomputed_wdl = recompute_wdl_from_games(parsed_games)

    return PgnParseOutcome(
        parsed_games=parsed_games,
        raw_result_games=raw_result_games,
        excluded=excluded,
        recomputed_wdl=recomputed_wdl,
    )


def parse_pgn_file(path, *, candidate_label: str, opponent_label: str) -> PgnParseOutcome:
    """Convenience: opens path as UTF-8 text (real c-chess-cli PGN output is
    UTF-8, per BigPC observation; python-chess's parser ignores a leading
    BOM on its own) and parses it with parse_pgn_games().
    """
    with open(path, "r", encoding="utf-8") as handle:
        return parse_pgn_games(handle, candidate_label=candidate_label, opponent_label=opponent_label)


def build_raw_result_document(
    *,
    run_id: str,
    claimed_wins: int,
    claimed_losses: int,
    claimed_draws: int,
    engine_stdout_log_ref: str,
    pgn_ref: str,
    raw_result_games: list,
) -> dict:
    """Assemble a RawResultV1 document from parser output and validate it
    against the frozen Phase 0 schema (medium_pc_audit.schemas.registry).

    claimed_wins/claimed_losses/claimed_draws are passed straight through
    as the schema's own 'claimed_*' fields (whatever the harness/log
    reported) -- they are never read by recompute_wdl_from_games() and
    never influence it; this function does not recompute or cross-check
    them, it only carries them into the schema-required document shape.
    """
    document = {
        "schema_version": "v1",
        "run_id": run_id,
        "games": raw_result_games,
        "claimed_wins": claimed_wins,
        "claimed_losses": claimed_losses,
        "claimed_draws": claimed_draws,
        "engine_stdout_log_ref": engine_stdout_log_ref,
        "pgn_ref": pgn_ref,
    }
    registry.validate("raw_result", "v1", document)
    return document
