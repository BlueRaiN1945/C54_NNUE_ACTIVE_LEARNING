"""Paired match statistics for the C54 Medium-PC audit plane.

Phase 4A deliberately contains no filesystem, engine, training, wall-clock,
or network behavior.  It consumes already-parsed game records and computes
deterministic paired statistics.

The statistical unit for repeated/color-reversed engine matches is the
opening pair.  Five pair outcomes are represented in candidate-score order:

    0.0  -> LL
    0.5  -> LD
    1.0  -> DD / WL / LW
    1.5  -> WD
    2.0  -> WW

SPRT/GSPRT decision semantics are intentionally NOT implemented in Phase 4A.
"""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import NormalDist
from typing import Iterable, Sequence


class AnalysisError(ValueError):
    """Raised when analysis input violates a Phase-4A invariant."""


_PAIR_SCORE_TO_INDEX = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
    1.5: 3,
    2.0: 4,
}


def _require_non_negative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisError(f"{name} must be a non-negative integer")
    return value


def candidate_points(game) -> float:
    """Return candidate score for one resolved parsed game."""

    result = getattr(game, "result", None)
    role = getattr(game, "candidate_role", None)

    if result == "1/2-1/2":
        if role not in {"white", "black"}:
            raise AnalysisError(
                f"invalid candidate role for draw: {role!r}"
            )
        return 0.5

    if role == "white":
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return 0.0

    if role == "black":
        if result == "0-1":
            return 1.0
        if result == "1-0":
            return 0.0

    raise AnalysisError(
        f"unsupported candidate role/result: {role!r}/{result!r}"
    )


def pentanomial_counts(games: Iterable) -> tuple[int, int, int, int, int]:
    """Group games by opening_ref and return LL/LD/DDWL/WD/WW counts.

    Every opening must occur exactly twice, once with the candidate as White
    and once with the candidate as Black.  Broken pairs are rejected rather
    than silently treated as independent games.
    """

    pairs = defaultdict(list)

    for game in games:
        opening_ref = getattr(game, "opening_ref", None)

        if not isinstance(opening_ref, str) or not opening_ref:
            raise AnalysisError("every game must have a non-empty opening_ref")

        pairs[opening_ref].append(game)

    if not pairs:
        raise AnalysisError("at least one opening pair is required")

    counts = [0, 0, 0, 0, 0]

    for opening_ref, pair in pairs.items():

        if len(pair) != 2:
            raise AnalysisError(
                f"{opening_ref}: expected exactly 2 games, got {len(pair)}"
            )

        roles = {
            getattr(game, "candidate_role", None)
            for game in pair
        }

        if roles != {"white", "black"}:
            raise AnalysisError(
                f"{opening_ref}: expected one candidate-white and "
                f"one candidate-black game, got {sorted(map(str, roles))}"
            )

        pair_score = sum(candidate_points(game) for game in pair)

        try:
            index = _PAIR_SCORE_TO_INDEX[pair_score]
        except KeyError as exc:
            raise AnalysisError(
                f"{opening_ref}: unsupported pair score {pair_score!r}"
            ) from exc

        counts[index] += 1

    return tuple(counts)


def score_from_wdl(wins: int, losses: int, draws: int) -> float:
    """Return candidate score fraction from W/L/D."""

    wins = _require_non_negative_int(wins, name="wins")
    losses = _require_non_negative_int(losses, name="losses")
    draws = _require_non_negative_int(draws, name="draws")

    total = wins + losses + draws

    if total == 0:
        raise AnalysisError("W/L/D total must be greater than zero")

    return (wins + 0.5 * draws) / total


def recomputed_wdl_for_schema(
    wins: int,
    losses: int,
    draws: int,
) -> dict:
    """Build the AnalysisResultV1 recomputed_wdl payload."""

    score = score_from_wdl(wins, losses, draws)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_games": wins + losses,
        "score": score,
    }


def _validate_pentanomial(
    results: Sequence[int],
) -> tuple[int, int, int, int, int]:

    if len(results) != 5:
        raise AnalysisError(
            "pentanomial results must contain exactly 5 cells"
        )

    validated = []

    for index, value in enumerate(results):
        validated.append(
            _require_non_negative_int(
                value,
                name=f"pentanomial[{index}]",
            )
        )

    if sum(validated) == 0:
        raise AnalysisError(
            "pentanomial results must contain at least one pair"
        )

    return tuple(validated)


def regularize_pentanomial(
    results: Sequence[int],
    *,
    epsilon: float = 0.001,
) -> tuple[float, float, float, float, float]:
    """Replace empty cells with epsilon for stable Elo transformation."""

    validated = _validate_pentanomial(results)

    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
    ):
        raise AnalysisError("epsilon must be a finite positive number")

    epsilon = float(epsilon)

    return tuple(
        epsilon if value == 0 else float(value)
        for value in validated
    )


def logistic_elo(
    score: float,
    *,
    clip_epsilon: float = 0.001,
) -> float:
    """Convert score fraction to logistic Elo."""

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise AnalysisError("score must be a finite number")

    score = float(score)

    if not 0.0 <= score <= 1.0:
        raise AnalysisError("score must be between 0 and 1")

    if not 0.0 < clip_epsilon < 0.5:
        raise AnalysisError("clip_epsilon must be between 0 and 0.5")

    score = max(
        clip_epsilon,
        min(1.0 - clip_epsilon, score),
    )

    return -400.0 * math.log10(1.0 / score - 1.0)


def estimate_pentanomial_elo(
    results: Sequence[int],
    *,
    confidence: float = 0.95,
    zero_cell_epsilon: float = 0.001,
) -> dict:
    """Estimate score, logistic Elo, and paired normal-approximation CI.

    Empty pentanomial cells are regularized with a small epsilon before the
    empirical score/variance calculation.  The returned raw pair count is
    always the actual number of observed opening pairs.
    """

    raw = _validate_pentanomial(results)

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise AnalysisError("confidence must be strictly between 0 and 1")

    confidence = float(confidence)

    reg = regularize_pentanomial(
        raw,
        epsilon=zero_cell_epsilon,
    )

    cell_count = len(reg)
    regularized_pairs = sum(reg)

    # Five pentanomial cells represent scores 0, .5, 1, 1.5, 2
    # over two games.  This converts the pair distribution to the
    # equivalent per-game score scale.
    effective_games = (
        regularized_pairs * (cell_count - 1) / 2.0
    )

    score = sum(
        reg[index] * (index / 2.0)
        for index in range(cell_count)
    ) / effective_games

    pair_scale_mean = (
        (cell_count - 1) / 2.0 * score
    )

    variance = sum(
        reg[index]
        * (index / 2.0 - pair_scale_mean) ** 2
        for index in range(cell_count)
    ) / effective_games

    stdev = math.sqrt(variance)

    z = NormalDist().inv_cdf(
        0.5 + confidence / 2.0
    )

    score_error = (
        z * stdev / math.sqrt(effective_games)
    )

    score_ci_low = score - score_error
    score_ci_high = score + score_error

    elo = logistic_elo(score)

    elo_error = (
        logistic_elo(score_ci_high)
        - logistic_elo(score_ci_low)
    ) / 2.0

    return {
        "pentanomial": list(raw),
        "pairs": sum(raw),
        "confidence": confidence,
        "regularized_score": score,
        "score_ci_low": score_ci_low,
        "score_ci_high": score_ci_high,
        "elo": elo,
        "elo_ci_low": elo - elo_error,
        "elo_ci_high": elo + elo_error,
        "elo_error": elo_error,
    }
