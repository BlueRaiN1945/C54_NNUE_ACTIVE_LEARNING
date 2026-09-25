"""Deterministic FEN position extraction from full-movetext PGN.

M0 corpus pipeline step 2 (Medium-PC side). Consumes a self-play PGN produced
on the execution host (step 1: pinned mainline Stockfish + pinned c-chess-cli,
sequential draw from a non-heritage opening book) and emits one FEN per ply,
in game order, for step 3 (rescoring on the execution host).

Dependency boundary: this is the SECOND and ONLY OTHER module in
medium_pc_audit allowed to import the third-party `chess` package, alongside
medium_pc_audit.pgn_parse -- see that module's docstring for why the two are
deliberately not merged. This module requires full, legal movetext: unlike
pgn_parse's audit parsing (which must tolerate any real-world engine PGN
without raising), a self-play PGN with an illegal or unparseable move is a
genuine defect in the generation pipeline that must fail loudly, not be
silently tolerated into training data.

Determinism: this module contributes no randomness of its own. Given the same
PGN text, extract_positions() always returns the same result -- ply-selection
is a pure, deterministic function of ply index (skip_first_n_plies), with no
RNG anywhere in this module. Determinism of the underlying corpus depends on
step 1's own determinism (sequential book order, fixed concurrency), which
this module does not control and does not need to.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TextIO

import chess
import chess.pgn


class PositionExtractError(ValueError):
    """Raised for invalid arguments or a self-play PGN that fails to parse.

    Raised loudly and specifically (never silently skipped), because an
    illegal move in a self-play PGN indicates a defect in the generation
    pipeline (steps 1 or the transfer between them), not routine messiness to
    tolerate.
    """


@dataclass(frozen=True)
class ExtractedPosition:
    """One extracted position: which game, which ply, and its full FEN.

    ply is 1-indexed and counted from the start of ITS OWN game (resets for
    each new game in the stream) -- it is a count of halfmoves applied so
    far, not the ply field encoded in any FEN.
    """

    game_index: int
    ply: int
    fen: str


def _require_non_negative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PositionExtractError(f"{name} must be a non-negative int, got {value!r}")

    return value


def extract_positions(
    pgn_source: str | TextIO,
    *,
    skip_first_n_plies: int = 0,
) -> list:
    """Extract one ExtractedPosition per ply from every game in pgn_source.

    pgn_source may be a PGN text string or an already-open text-mode handle.

    skip_first_n_plies: positions at ply <= skip_first_n_plies are omitted,
    per game. The caller chooses this value (there is no built-in default
    beyond 0, i.e. no filtering) and is responsible for recording that choice
    in the corpus generation record -- see medium_pc_audit.corpus_provenance.

    Games are read via chess.pgn.read_game() with python-chess's DEFAULT
    visitor (full SAN parsing and legality checking), repeated until EOF
    returns None -- the same delimiter-free iteration pattern pgn_parse.py
    uses for headers, applied here to full games instead.
    """

    skip_first_n_plies = _require_non_negative_int(
        skip_first_n_plies, name="skip_first_n_plies"
    )

    handle = io.StringIO(pgn_source) if isinstance(pgn_source, str) else pgn_source

    positions = []
    game_index = 0

    while True:
        try:
            game = chess.pgn.read_game(handle)
        except ValueError as exc:
            raise PositionExtractError(
                f"game {game_index + 1}: failed to parse movetext: {exc}"
            ) from exc

        if game is None:
            break

        game_index += 1

        # python-chess's default parser is LENIENT: an illegal/unparseable
        # SAN token does not raise -- it is recorded in game.errors and the
        # mainline is silently truncated at that point. For audit parsing
        # (pgn_parse.py) that leniency is correct. Here it is not: a
        # truncated mainline would silently produce a shorter, wrong set of
        # positions instead of failing, so game.errors must be checked
        # explicitly rather than trusted to surface as an exception.
        if game.errors:
            raise PositionExtractError(
                f"game {game_index}: {len(game.errors)} parse error(s), "
                f"first: {game.errors[0]}"
            )

        board = game.board()

        for ply, move in enumerate(game.mainline_moves(), start=1):
            if move not in board.legal_moves:
                raise PositionExtractError(
                    f"game {game_index}: illegal move {move} at ply {ply}"
                )

            board.push(move)

            if ply > skip_first_n_plies:
                positions.append(
                    ExtractedPosition(
                        game_index=game_index,
                        ply=ply,
                        fen=board.fen(),
                    )
                )

    return positions
