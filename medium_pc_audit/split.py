"""Heritage split, exact-FEN deduplication, and family/line isolation.

Stdlib-only and deterministic: no filesystem, network, or wall-clock access.

Locked methodology this module enforces:

  - **Family isolation is mandatory and fails closed.** The frozen
    methodology requires splitting by game / opening line / family boundary,
    NOT by individual position. No family identifier exists in the current
    heritage source data (OLA_C54_MASTER.tsv columns are
    `fen, source, weight, gen2_class, gen2_frequency, ola_games, ola_moves` --
    live-reconfirmed, none of which is a game or line identity). Therefore
    build_split() REFUSES by default when no family_of provider is supplied,
    rather than silently degrading to a per-position split. That silent
    degradation is precisely what the existing SPLIT_V1 does, and why its
    output does not meet the leakage bar.

  - **Residual leakage is always documented**, never assumed absent. Every
    SplitResult carries an explicit residual_leakage list.

  - The bucketing algorithm deliberately mirrors the proven SPLIT_V1 scheme
    -- SHA256(seed + NUL + key), first 64 bits, modulo total weight -- so
    that behavior is familiar and auditable. It is generalized from 2-way to
    3-way (train/validation/holdout).

**Dependency boundary:** normalize_fen_key() reimplements, in pure stdlib
string operations, the same normalization as
medium_pc_audit.pgn_parse.normalize_opening_ref_from_fen(). It is not imported
from there because this module deliberately remains stdlib-only. The `chess`
dependency is confined to medium_pc_audit.pgn_parse and
medium_pc_audit.position_extract. The normalizers must agree exactly;
tests/test_split.py contains the regression guard, following the precedent of
tests/test_phase0_neutral_path_equivalence.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_HOLDOUT = "holdout"

SPLIT_NAMES = (
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_HOLDOUT,
)


class SplitError(ValueError):
    """Raised when a split invariant is violated."""


class FamilyIdentityUnavailable(SplitError):
    """Raised when a family/line-isolated split is required but impossible.

    This is the expected, correct outcome for the heritage corpus as it
    exists today. It is an error rather than a warning so that no caller can
    accidentally produce a split that looks leakage-controlled but is not.
    """


def normalize_fen_key(fen):
    """Return the first four FEN fields joined by single spaces, or None.

    Fields: piece placement, side to move, castling rights, en-passant square.
    The halfmove clock and fullmove number are deliberately excluded -- they
    vary with when a position was recorded and are not part of position
    identity. Returns None for anything that is not a string carrying at
    least four whitespace-separated fields.
    """

    if not isinstance(fen, str):
        return None

    fields = fen.split()

    if len(fields) < 4:
        return None

    return " ".join(fields[:4])


@dataclass(frozen=True)
class SplitWeights:
    """Integer bucket weights. Mirrors SPLIT_V1's 1-in-N bucket scheme."""

    train: int = 18
    validation: int = 1
    holdout: int = 1

    def __post_init__(self):
        for name in SPLIT_NAMES:
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SplitError(
                    f"weights.{name} must be a non-negative int, got {value!r}"
                )

        if self.total <= 0:
            raise SplitError("weights must sum to a positive total")

        if self.train <= 0:
            raise SplitError("weights.train must be > 0")

    @property
    def total(self) -> int:
        return self.train + self.validation + self.holdout

    def as_dict(self) -> dict:
        return {
            "train": self.train,
            "validation": self.validation,
            "holdout": self.holdout,
            "total": self.total,
        }


@dataclass(frozen=True)
class DedupeReport:
    total_input: int
    unique_keys: int
    exact_duplicates_removed: int
    malformed_skipped: int

    def as_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "unique_keys": self.unique_keys,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "malformed_skipped": self.malformed_skipped,
        }


def exact_fen_dedupe(fens):
    """Deduplicate by normalized FEN key, preserving first-seen order.

    Returns (ordered_unique_keys, DedupeReport). Malformed entries are
    counted and reported, never silently dropped without accounting.
    """

    seen = set()
    ordered = []
    total = 0
    malformed = 0

    for fen in fens:
        total += 1
        key = normalize_fen_key(fen)

        if key is None:
            malformed += 1
            continue

        if key in seen:
            continue

        seen.add(key)
        ordered.append(key)

    return ordered, DedupeReport(
        total_input=total,
        unique_keys=len(ordered),
        exact_duplicates_removed=total - len(ordered) - malformed,
        malformed_skipped=malformed,
    )


def assign_split(key: str, *, seed: str, weights: SplitWeights) -> str:
    """Deterministically assign one key to a split.

    Algorithm (mirrors SPLIT_V1, generalized to three ways):
        bucket = int(SHA256(seed + NUL + key)[:8], big-endian) % weights.total

    Bucket ranges are assigned in a fixed, documented order so the mapping is
    stable forever: validation first, then holdout, then train.
    """

    if not isinstance(key, str) or not key:
        raise SplitError(f"split key must be a non-empty string, got {key!r}")

    if not isinstance(seed, str) or not seed:
        raise SplitError(f"seed must be a non-empty string, got {seed!r}")

    if not isinstance(weights, SplitWeights):
        raise SplitError("weights must be a SplitWeights")

    digest = hashlib.sha256(
        seed.encode("utf-8") + b"\x00" + key.encode("utf-8")
    ).digest()

    bucket = int.from_bytes(digest[:8], "big") % weights.total

    if bucket < weights.validation:
        return SPLIT_VALIDATION

    if bucket < weights.validation + weights.holdout:
        return SPLIT_HOLDOUT

    return SPLIT_TRAIN


@dataclass(frozen=True)
class SplitResult:
    """The outcome of a split, including its leakage posture.

    family_isolation and leakage_controlled are separate on purpose: a split
    can be deterministic and exactly-deduplicated (controlling exact
    duplicates) while still not being family-isolated.
    """

    assignments: dict
    counts: dict
    family_isolation: bool
    leakage_controlled: bool
    residual_leakage: tuple
    dedupe: DedupeReport
    seed: str
    weights: SplitWeights
    unit: str

    def as_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "family_isolation": self.family_isolation,
            "leakage_controlled": self.leakage_controlled,
            "residual_leakage": list(self.residual_leakage),
            "dedupe": self.dedupe.as_dict(),
            "seed": self.seed,
            "weights": self.weights.as_dict(),
            "unit": self.unit,
            "unit_count": len(self.assignments),
        }


_POSITION_LEVEL_RESIDUAL_LEAKAGE = (
    (
        "Split unit is the individual position, not a game/line/family. Near-"
        "adjacent positions from the same historical game can land on opposite "
        "sides of the split; this is NOT ruled out and its magnitude is unknown."
    ),
    (
        "No game or line identifier exists in the source data, so the size of "
        "this risk cannot be measured, only acknowledged."
    ),
    (
        "This split does NOT meet the frozen methodology's leakage bar and must "
        "not be used as evidence for a heritage-transfer claim."
    ),
)

_FAMILY_LEVEL_RESIDUAL_LEAKAGE = (
    (
        "Positions are isolated at the declared family boundary. Leakage across "
        "that boundary is only as strong as the family_of provider's fidelity to "
        "true game/line identity -- verify the provider, not just this split."
    ),
    (
        "Transpositions reaching an identical position from different families "
        "are assigned per-family, so the same position key may appear in more "
        "than one split if it genuinely belongs to multiple families."
    ),
)


def build_split(
    rows,
    *,
    seed: str,
    weights: SplitWeights | None = None,
    family_of=None,
    fen_key: str = "fen",
    allow_position_level_split_unsafe: bool = False,
) -> SplitResult:
    """Build a family-isolated train/validation/holdout split.

    rows: iterable of mappings, each carrying a FEN under `fen_key`.
    family_of: callable(row) -> family identifier. REQUIRED unless
        allow_position_level_split_unsafe is explicitly True.

    Raises FamilyIdentityUnavailable when family_of is absent and the unsafe
    escape hatch was not deliberately requested. That escape hatch exists for
    diagnostic work only: it stamps the result with family_isolation=False
    and leakage_controlled=False, and medium_pc_audit.preflight refuses to
    authorize any experiment whose split reports family_isolation=False.
    """

    if weights is None:
        weights = SplitWeights()

    if not isinstance(seed, str) or not seed:
        raise SplitError(f"seed must be a non-empty string, got {seed!r}")

    if family_of is None and not allow_position_level_split_unsafe:
        raise FamilyIdentityUnavailable(
            "a family/line-isolated split requires a family_of provider. No "
            "game/line identifier exists in the current heritage source data, "
            "so this split cannot be produced. Pass "
            "allow_position_level_split_unsafe=True ONLY for diagnostic work "
            "that will not be used as heritage-transfer evidence."
        )

    if family_of is not None and not callable(family_of):
        raise SplitError("family_of must be callable")

    rows = list(rows)
    fens = []

    for row in rows:
        if not isinstance(row, dict):
            raise SplitError(f"each row must be a mapping, got {type(row).__name__}")

        fens.append(row.get(fen_key))

    unique_keys, dedupe = exact_fen_dedupe(fens)

    if family_of is None:
        unit = "position"
        keys = unique_keys
        family_isolation = False
        residual = _POSITION_LEVEL_RESIDUAL_LEAKAGE
    else:
        unit = "family"
        seen_families = []
        seen = set()

        for row in rows:
            if normalize_fen_key(row.get(fen_key)) is None:
                continue

            family_id = family_of(row)

            if not isinstance(family_id, str) or not family_id:
                raise SplitError(
                    f"family_of must return a non-empty string, got {family_id!r}"
                )

            if family_id in seen:
                continue

            seen.add(family_id)
            seen_families.append(family_id)

        keys = seen_families
        family_isolation = True
        residual = _FAMILY_LEVEL_RESIDUAL_LEAKAGE

    assignments = {key: assign_split(key, seed=seed, weights=weights) for key in keys}

    counts = {name: 0 for name in SPLIT_NAMES}

    for assigned in assignments.values():
        counts[assigned] += 1

    return SplitResult(
        assignments=assignments,
        counts=counts,
        family_isolation=family_isolation,
        leakage_controlled=family_isolation,
        residual_leakage=residual,
        dedupe=dedupe,
        seed=seed,
        weights=weights,
        unit=unit,
    )


def verify_no_cross_split_overlap(result: SplitResult) -> None:
    """Assert no unit is assigned to more than one split.

    Trivially true by construction (assignments is a dict), but asserted
    explicitly so the guarantee is tested rather than assumed, matching the
    repository's practice of re-verifying rather than trusting construction.
    """

    if not isinstance(result, SplitResult):
        raise SplitError("result must be a SplitResult")

    for key, assigned in result.assignments.items():
        if assigned not in SPLIT_NAMES:
            raise SplitError(f"unit {key!r} has unknown split {assigned!r}")

    if sum(result.counts.values()) != len(result.assignments):
        raise SplitError("split counts do not match assignment count")
