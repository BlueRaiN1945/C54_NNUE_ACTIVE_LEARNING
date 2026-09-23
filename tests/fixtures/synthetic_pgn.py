"""Synthetic PGN text builders for Phase 3 pgn_parse tests.

Every PGN blob here is fabricated to mirror real c-chess-cli output as
observed on BigPC -- no [Event] header, explicit White/Black engine-label
headers, FEN/Termination/PlyCount/Round headers, UTF-8 content -- but the
header values themselves are always synthetic labels, never real
BigPC/Stockfish/NNUE identities or game records.
"""

from __future__ import annotations

CANDIDATE = "RESEARCH_HEAD"
OPPONENT = "FROZEN_V80"
OTHER_OPPONENT = "FROZEN_OFFICIAL"

STANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STANDARD_OPENING_REF = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def render_game(headers: dict, *, movetext: str = None) -> str:
    """Render one PGN game block: header tag lines in the given order, a
    blank line, then a movetext/result line.

    No [Event] header is ever added unless the caller puts one in
    `headers` explicitly -- this mirrors real c-chess-cli PGNs, which may
    omit [Event] entirely, and is what rules out using [Event] as a game
    delimiter in the parser under test.

    `movetext` defaults to headers.get("Result", "*") so a well-formed
    result token terminates the game even when a test deliberately omits
    the [Result] header tag itself (to simulate a missing Result header).
    """
    lines = [f'[{key} "{value}"]' for key, value in headers.items()]
    line = movetext if movetext is not None else headers.get("Result", "*")
    return "\n".join(lines) + "\n\n" + line + "\n"


def join_games(*games: str) -> str:
    return "\n".join(games)


def multi_game_no_event_pgn() -> str:
    """Three consecutive games with no [Event] header anywhere: candidate
    plays White and wins, candidate plays Black and wins, then a draw.
    """
    return join_games(
        render_game(
            {
                "White": CANDIDATE,
                "Black": OPPONENT,
                "Result": "1-0",
                "Round": "1",
                "FEN": STANDARD_FEN,
                "Termination": "normal",
                "PlyCount": "40",
            }
        ),
        render_game(
            {
                "White": OPPONENT,
                "Black": CANDIDATE,
                "Result": "0-1",
                "Round": "2",
                "FEN": STANDARD_FEN,
                "Termination": "normal",
                "PlyCount": "55",
            }
        ),
        render_game(
            {
                "White": CANDIDATE,
                "Black": OPPONENT,
                "Result": "1/2-1/2",
                "Round": "3",
                "FEN": STANDARD_FEN,
                "Termination": "normal",
                "PlyCount": "80",
            }
        ),
    )


def color_reversed_pair_pgn() -> str:
    """Two games between the same pair of engines with colors swapped and
    an IDENTICAL raw Result token ('1-0') in both -- proving the parser
    derives candidate perspective from White/Black labels together with
    Result, never from color (or Result token) alone.

    Game 1: candidate is White, Result "1-0" -> candidate WIN.
    Game 2: candidate is Black, Result "1-0" -> candidate LOSS (opponent,
    now White, is the one who won).
    """
    return join_games(
        render_game(
            {"White": CANDIDATE, "Black": OPPONENT, "Result": "1-0", "Round": "1", "FEN": STANDARD_FEN}
        ),
        render_game(
            {"White": OPPONENT, "Black": CANDIDATE, "Result": "1-0", "Round": "2", "FEN": STANDARD_FEN}
        ),
    )


def mostly_draws_pgn(*, n_draws: int = 8, n_candidate_wins: int = 1, n_candidate_losses: int = 1) -> str:
    """A run dominated by draws, with a small number of decisive games on
    both sides so win/loss/draw tallying is exercised together.
    """
    games = []
    round_no = 1
    for _ in range(n_draws):
        games.append(
            render_game(
                {
                    "White": CANDIDATE if round_no % 2 else OPPONENT,
                    "Black": OPPONENT if round_no % 2 else CANDIDATE,
                    "Result": "1/2-1/2",
                    "Round": str(round_no),
                    "FEN": STANDARD_FEN,
                }
            )
        )
        round_no += 1
    for _ in range(n_candidate_wins):
        games.append(
            render_game(
                {"White": CANDIDATE, "Black": OPPONENT, "Result": "1-0", "Round": str(round_no), "FEN": STANDARD_FEN}
            )
        )
        round_no += 1
    for _ in range(n_candidate_losses):
        games.append(
            render_game(
                {"White": CANDIDATE, "Black": OPPONENT, "Result": "0-1", "Round": str(round_no), "FEN": STANDARD_FEN}
            )
        )
        round_no += 1
    return join_games(*games)


def malformed_missing_headers_pgn() -> str:
    """Three games, each missing one of the three required identity/result
    headers: no White, no Black, no Result (header tag entirely absent,
    not merely empty).
    """
    return join_games(
        render_game({"Black": OPPONENT, "Result": "1-0", "FEN": STANDARD_FEN}),
        render_game({"White": CANDIDATE, "Result": "0-1", "FEN": STANDARD_FEN}),
        render_game({"White": CANDIDATE, "Black": OPPONENT, "FEN": STANDARD_FEN}, movetext="*"),
    )


def unresolved_star_result_pgn() -> str:
    """A game with an explicit, unresolved '*' Result -- otherwise
    complete and well-formed.
    """
    return render_game(
        {"White": CANDIDATE, "Black": OPPONENT, "Result": "*", "FEN": STANDARD_FEN, "Round": "1"}
    )


def truncated_input_pgn() -> str:
    """One well-formed game followed by an abruptly truncated second game
    (cut off mid-header-value, no closing quote, no blank line, no
    movetext at all) -- simulating a c-chess-cli PGN file cut off by a
    crashed/killed process.
    """
    complete_game = render_game(
        {"White": CANDIDATE, "Black": OPPONENT, "Result": "1-0", "FEN": STANDARD_FEN, "Round": "1"}
    )
    truncated_fragment = '[White "' + OPPONENT
    return complete_game + "\n" + truncated_fragment


def missing_fen_pgn() -> str:
    """A fully valid, decisive game with no [FEN] header at all -- usable
    for WDL recomputation but must be excluded from the RawResultV1
    'games' list for lack of an opening_ref.
    """
    return render_game({"White": CANDIDATE, "Black": OPPONENT, "Result": "1-0", "Round": "1"})


def malformed_fen_pgn() -> str:
    """A fully valid, decisive game whose [FEN] header is present but has
    fewer than the four fields needed to derive opening_ref.
    """
    return render_game(
        {"White": CANDIDATE, "Black": OPPONENT, "Result": "0-1", "FEN": "not-a-real-fen", "Round": "1"}
    )


UTF8_CANDIDATE_LABEL = "ИССЛЕДОВАТЕЛЬ"
UTF8_OPPONENT_LABEL = "ЗАМОРОЖЕННЫЙ_V80"


def utf8_headers_pgn() -> str:
    """Non-ASCII header content: Cyrillic engine labels used consistently
    as both the PGN White/Black tag values and the candidate/opponent
    labels supplied to the parser, plus a non-ASCII Termination string.
    """
    return render_game(
        {
            "White": UTF8_CANDIDATE_LABEL,
            "Black": UTF8_OPPONENT_LABEL,
            "Result": "1-0",
            "FEN": STANDARD_FEN,
            "Termination": "время истекло",
            "Round": "1",
        }
    )


def garbage_binary_pgn() -> str:
    """Content that is not PGN at all -- control characters and stray
    bracket-like text with no valid header block or result token.
    """
    return "\x00\x01\x02 this is not a pgn file at all \xff [Not [[a header"


def unrecognized_engine_labels_pgn() -> str:
    """A well-formed, decisive, FEN-bearing game between two engines
    neither of which is the candidate or the configured opponent -- e.g. a
    leftover game from a different matchup pairing in the same PGN.
    """
    return render_game(
        {"White": OTHER_OPPONENT, "Black": "SOME_THIRD_ENGINE", "Result": "1-0", "FEN": STANDARD_FEN, "Round": "1"}
    )


def empty_pgn() -> str:
    return ""
