"""Opening suite (UHO) contracts, capacity checks, and deterministic selection.

Stdlib-only and deterministic: no filesystem, network, or wall-clock access.
Callers read files themselves and pass text in.

Two locked requirements are enforced here:

  - **Unbalancedness is declared, never assumed** (locked decision #7: "a
    pinned deliberately-unbalanced/UHO-style opening source is required
    before strength testing"). A suite carries unbalanced_declared plus a
    justification; the default is False, and preflight refuses a suite that
    has not been positively declared.

  - **Pair capacity and distinctness.** This addresses a concrete operational
    trap enforced by the match-result intake rules:

    `opening_ref` is derived from FEN fields 1-4, so two independently drawn
    occurrences of the same opening collide into a single ref. Intake
    (inbox_validate.pentanomial_from_raw_games) requires every opening_ref to
    appear EXACTLY twice; a collision yields four games under one ref and
    quarantines the whole package as `malformed_opening_pair`.

    BigPC was observed running `-openings ... order=random -repeat`. Drawing
    200 pairs with replacement from 150,932 openings carries roughly a 12%
    chance of at least one collision; at 500 pairs it exceeds 50%. An entire
    run would be discarded, and it would look like a Medium-PC bug when the
    strictness is in fact correct.

    select_openings() therefore samples WITHOUT replacement, deterministically
    from a seed, which makes the collision structurally impossible rather
    than merely unlikely.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from medium_pc_audit.split import normalize_fen_key

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OpeningSuiteError(ValueError):
    """Raised when an opening-suite invariant is violated."""


class InsufficientOpenings(OpeningSuiteError):
    """Raised when a suite cannot supply enough distinct openings."""


@dataclass(frozen=True)
class SuiteReport:
    total_lines: int
    distinct_openings: int
    duplicate_lines: int
    malformed_lines: int

    def as_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "distinct_openings": self.distinct_openings,
            "duplicate_lines": self.duplicate_lines,
            "malformed_lines": self.malformed_lines,
        }


@dataclass(frozen=True)
class OpeningSuite:
    """A pinned set of distinct opening positions.

    openings is always deduplicated and order-stable (first-seen order from
    the source text), so a suite's identity does not depend on dict ordering.
    """

    suite_id: str
    sha256: str
    openings: tuple
    unbalanced_declared: bool = False
    justification: str = ""
    source_note: str = ""

    def __post_init__(self):
        if not isinstance(self.suite_id, str) or not _IDENTIFIER_RE.fullmatch(
            self.suite_id
        ):
            raise OpeningSuiteError(
                f"suite_id must match {_IDENTIFIER_RE.pattern!r}, "
                f"got {self.suite_id!r}"
            )

        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise OpeningSuiteError(
                "suite sha256 must be 64 lowercase hexadecimal characters"
            )

        if not isinstance(self.unbalanced_declared, bool):
            raise OpeningSuiteError("unbalanced_declared must be a bool")

        openings = tuple(self.openings)

        if not openings:
            raise OpeningSuiteError("suite must contain at least one opening")

        if len(set(openings)) != len(openings):
            raise OpeningSuiteError(
                "suite openings must already be deduplicated; use "
                "load_suite_from_text() which deduplicates and reports"
            )

        # Declaring a suite deliberately unbalanced without saying on what
        # basis is the same hand-wave the heritage-status contract forbids.
        if self.unbalanced_declared and not str(self.justification).strip():
            raise OpeningSuiteError(
                "unbalanced_declared=True requires a non-empty justification "
                "(e.g. the evaluation band the positions were selected for)"
            )

        object.__setattr__(self, "openings", openings)

    @property
    def distinct_count(self) -> int:
        return len(self.openings)

    def max_pairs(self) -> int:
        """Each pair consumes one distinct opening, played twice (color swap)."""
        return self.distinct_count

    def as_dict(self) -> dict:
        return {
            "suite_id": self.suite_id,
            "sha256": self.sha256,
            "distinct_openings": self.distinct_count,
            "unbalanced_declared": self.unbalanced_declared,
            "justification": self.justification,
            "source_note": self.source_note,
        }


def load_suite_from_text(
    text: str,
    *,
    suite_id: str,
    sha256: str,
    unbalanced_declared: bool = False,
    justification: str = "",
    source_note: str = "",
):
    """Parse EPD/FEN-per-line text into a deduplicated OpeningSuite.

    An EPD line begins with the same four fields as a FEN (placement, side to
    move, castling, en-passant) followed by operations, so the shared
    normalize_fen_key() applies directly. Blank lines are ignored; lines
    without four usable fields are counted as malformed and reported, never
    silently discarded.

    Returns (OpeningSuite, SuiteReport).
    """

    if not isinstance(text, str):
        raise OpeningSuiteError("text must be a string")

    seen = set()
    ordered = []
    total = 0
    malformed = 0
    duplicates = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        total += 1
        key = normalize_fen_key(line)

        if key is None:
            malformed += 1
            continue

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        ordered.append(key)

    if not ordered:
        raise OpeningSuiteError(
            f"no usable openings parsed from {total} non-blank lines "
            f"({malformed} malformed)"
        )

    suite = OpeningSuite(
        suite_id=suite_id,
        sha256=sha256,
        openings=tuple(ordered),
        unbalanced_declared=unbalanced_declared,
        justification=justification,
        source_note=source_note,
    )

    report = SuiteReport(
        total_lines=total,
        distinct_openings=len(ordered),
        duplicate_lines=duplicates,
        malformed_lines=malformed,
    )

    return suite, report


def required_distinct_openings(pairs: int) -> int:
    """One distinct opening per pair; the pair is the two color-swapped games."""

    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise OpeningSuiteError(f"pairs must be an int >= 1, got {pairs!r}")

    return pairs


def collision_probability(pairs: int, distinct: int) -> float:
    """Approximate P(at least one repeat) when drawing WITH replacement.

    Birthday approximation: 1 - exp(-pairs*(pairs-1) / (2*distinct)).

    This exists to quantify the trap, not to endorse random draws: any
    non-zero result means a run can be quarantined for a reason that has
    nothing to do with engine strength. select_openings() avoids it entirely.
    """

    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise OpeningSuiteError(f"pairs must be an int >= 1, got {pairs!r}")

    if isinstance(distinct, bool) or not isinstance(distinct, int) or distinct < 1:
        raise OpeningSuiteError(f"distinct must be an int >= 1, got {distinct!r}")

    return 1.0 - math.exp(-pairs * (pairs - 1) / (2.0 * distinct))


def validate_capacity(suite: OpeningSuite, *, pairs: int) -> None:
    """Raise InsufficientOpenings unless the suite can cover `pairs` pairs."""

    if not isinstance(suite, OpeningSuite):
        raise OpeningSuiteError("suite must be an OpeningSuite")

    needed = required_distinct_openings(pairs)

    if suite.distinct_count < needed:
        raise InsufficientOpenings(
            f"suite {suite.suite_id!r} has {suite.distinct_count} distinct "
            f"openings but {needed} are required for {pairs} pairs "
            f"({pairs * 2} games)"
        )


def select_openings(suite: OpeningSuite, *, pairs: int, seed: str) -> tuple:
    """Deterministically select `pairs` distinct openings WITHOUT replacement.

    Ordering key is SHA256(seed + NUL + opening), ascending, with the opening
    itself as a tiebreaker so the result is total and reproducible. Sampling
    without replacement makes the duplicate-opening_ref quarantine
    structurally impossible rather than merely improbable.
    """

    validate_capacity(suite, pairs=pairs)

    if not isinstance(seed, str) or not seed:
        raise OpeningSuiteError(f"seed must be a non-empty string, got {seed!r}")

    def sort_key(opening):
        digest = hashlib.sha256(
            seed.encode("utf-8") + b"\x00" + opening.encode("utf-8")
        ).digest()
        return (digest, opening)

    return tuple(sorted(suite.openings, key=sort_key)[:pairs])


def build_suite_contract(suite: OpeningSuite, *, pairs: int, seed: str) -> dict:
    """Produce the full, auditable record of an opening selection."""

    selected = select_openings(suite, pairs=pairs, seed=seed)

    return {
        "suite": suite.as_dict(),
        "pairs": pairs,
        "games": pairs * 2,
        "seed": seed,
        "selected_count": len(selected),
        "selection": "without_replacement_deterministic",
        "with_replacement_collision_probability": collision_probability(
            pairs,
            suite.distinct_count,
        ),
        "selected_openings": list(selected),
    }
