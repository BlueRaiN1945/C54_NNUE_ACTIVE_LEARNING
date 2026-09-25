"""Clean-room, first-principles M0 self-play opening artifact.

Built to resolve a specific gap: the only opening-book-shaped artifact found
on the execution host (`noob_3moves.epd`) lives under a match-tooling
directory with zero provenance documentation, so it cannot be called
heritage-free and cannot be used as the M0 self-play opening source. This
module generates a replacement from first principles instead of using any
external material.

**Why this is eligible to be called heritage-independent.** Every position
in the resulting artifact is reachable from `chess.Board()`'s standard
initial position by python-chess's own legal-move generator alone. No PGN,
opening book, engine evaluation, or external corpus of any kind is read.
The only external input is the pinned `chess==1.11.2` package's rules
engine -- the same dependency medium_pc_audit.pgn_parse and
medium_pc_audit.position_extract already rely on for board handling.

**Scope.** This is the M0 SELF-PLAY CORPUS GENERATION opening source. It is
NOT the UHO / deliberately-unbalanced suite used later for strength
testing -- that remains a separate, still-open item, and
medium_pc_audit.opening_suite.OpeningSuite.unbalanced_declared is never set
True here.

Dependency boundary: this is now a THIRD module in medium_pc_audit allowed
to import the third-party `chess` package, alongside
medium_pc_audit.pgn_parse and medium_pc_audit.position_extract (see that
module's docstring, which is updated alongside this one to say "third"
rather than "second and only other"). chess is used here solely for legal
move generation from the standard starting position.

Recipe (locked by the calling contract that requested this module, not
invented here):
  1. enumerate every legal move sequence of exactly `plies` half-moves from
     the standard initial position;
  2. take the resulting position's FEN after each such sequence;
  3. normalize to the first four FEN fields and exact-deduplicate, via the
     same medium_pc_audit.split.exact_fen_dedupe() every other position
     source in this project already uses;
  4. rank the unique keys by SHA256(UTF8(namespace + NUL + key)), ascending
     -- the same seeded-digest-ranking idea already used by
     medium_pc_audit.opening_suite.select_openings() and
     medium_pc_audit.split.assign_split(), applied to a population this
     module itself generates rather than one read from a file;
  5. take the first `count` ranked keys.

Determinism: given the same (namespace, plies, count), this module always
returns the same tuple -- no RNG, no wall-clock, no filesystem read beyond
this module's own imports.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import chess

from medium_pc_audit.artifact_import import compute_file_sha256
from medium_pc_audit.split import exact_fen_dedupe

NAMESPACE = "M0_OPENINGS_V1"
ENUMERATION_PLIES = 4
SELECTION_COUNT = 20000
ARTIFACT_FILENAME = "m0_openings_v1.epd"
RECIPE_VERSION = "clean_room_openings_v1"


class CleanRoomOpeningsError(ValueError):
    """Raised for invalid arguments or an under-populated candidate pool."""


def _require_positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CleanRoomOpeningsError(f"{name} must be an int >= 1, got {value!r}")

    return value


def enumerate_leaf_fens(plies: int = ENUMERATION_PLIES) -> list:
    """Every position reached by exactly `plies` legal half-moves from the
    standard initial position, one FEN per distinct legal move sequence.

    Two different move orders reaching the same position (a transposition)
    each contribute their own entry here -- collapsing those is
    exact_fen_dedupe()'s job, not this function's, exactly as raw self-play
    PGN positions are deduplicated elsewhere in this project rather than
    de-transposed at extraction time.

    Touches no file, network, or engine subprocess: the only external input
    is the `chess` package's own legal-move generator over a board this
    function constructs itself.
    """

    plies = _require_positive_int(plies, name="plies")

    leaves: list = []

    def _walk(board: chess.Board, remaining: int) -> None:
        if remaining == 0:
            leaves.append(board.fen())
            return

        for move in board.legal_moves:
            board.push(move)
            _walk(board, remaining - 1)
            board.pop()

    _walk(chess.Board(), plies)

    return leaves


def _rank_digest(namespace: str, key: str) -> bytes:
    return hashlib.sha256(
        namespace.encode("utf-8") + b"\x00" + key.encode("utf-8")
    ).digest()


def build_m0_openings(
    *,
    namespace: str = NAMESPACE,
    plies: int = ENUMERATION_PLIES,
    count: int = SELECTION_COUNT,
) -> tuple:
    """Build the deterministic candidate M0 self-play opening set.

    Raises CleanRoomOpeningsError if fewer than `count` unique positions
    exist at this depth -- this must never silently return a short list,
    matching medium_pc_audit.opening_suite.InsufficientOpenings's fail-loud
    posture for the same class of shortfall.
    """

    if not isinstance(namespace, str) or not namespace:
        raise CleanRoomOpeningsError("namespace must be a non-empty string")

    plies = _require_positive_int(plies, name="plies")
    count = _require_positive_int(count, name="count")

    raw_fens = enumerate_leaf_fens(plies)
    unique_keys, dedupe = exact_fen_dedupe(raw_fens)

    if dedupe.malformed_skipped:
        # enumerate_leaf_fens() only ever emits real chess.Board().fen()
        # output, which always carries >= 4 whitespace-separated fields, so
        # a malformed key here means this module's own enumerator
        # regressed -- not that upstream data is dirty.
        raise CleanRoomOpeningsError(
            f"{dedupe.malformed_skipped} enumerated position(s) failed FEN "
            "normalization; this indicates a defect in enumerate_leaf_fens()"
        )

    if len(unique_keys) < count:
        raise CleanRoomOpeningsError(
            f"only {len(unique_keys)} unique position(s) exist at depth "
            f"{plies} plies, fewer than the requested {count}"
        )

    ranked = sorted(unique_keys, key=lambda key: (_rank_digest(namespace, key), key))

    return tuple(ranked[:count])


def render_epd_text(openings) -> str:
    """One normalized FEN per line, in the exact order given.

    No re-sorting here: the persisted order IS the selection order, so the
    file itself is evidence that the ranking rule was applied, not just a
    set of positions that happens to satisfy the count.
    """

    openings = tuple(openings)

    if not openings:
        raise CleanRoomOpeningsError("openings must not be empty")

    return "\n".join(openings) + "\n"


def write_artifact(path, openings) -> Path:
    """Render and write the candidate artifact. The only file I/O in this
    module; enumeration/ranking/rendering above remain pure functions."""

    path = Path(path)
    text = render_epd_text(openings)
    path.write_text(text, encoding="utf-8", newline="\n")

    return path


def build_provenance_record(
    *,
    artifact_path,
    namespace: str = NAMESPACE,
    plies: int = ENUMERATION_PLIES,
    count: int = SELECTION_COUNT,
    candidate_pool_size: int,
) -> dict:
    """The provenance record for a generated candidate artifact.

    artifact_sha256/artifact_size_bytes are observed by hashing/stat-ing the
    actual file on disk -- never copied through from any caller claim --
    matching medium_pc_audit.corpus_provenance's "re-hash, never trust"
    discipline.
    """

    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise CleanRoomOpeningsError(f"artifact_path does not exist: {artifact_path}")

    return {
        "artifact_filename": artifact_path.name,
        "artifact_sha256": compute_file_sha256(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "unique_opening_count": count,
        "candidate_pool_size": candidate_pool_size,
        "recipe": RECIPE_VERSION,
        "python_version": platform.python_version(),
        "chess_package_version": getattr(chess, "__version__", "unknown"),
        "namespace": namespace,
        "enumeration_depth_plies": plies,
        "dedup_rule": (
            "exact_fen_dedupe: normalized first-four-FEN-field exact match "
            "(medium_pc_audit.split)"
        ),
        "selection_rule": (
            "ascending SHA256(UTF8(namespace + NUL + normalized_fen)), "
            f"first {count} taken"
        ),
        "source": (
            "generated solely from chess.Board() standard initial position "
            "and python-chess legal move enumeration; no external opening "
            "book, PGN, engine evaluation, or corpus used"
        ),
    }
