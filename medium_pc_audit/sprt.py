"""Logistic pentanomial GSPRT and Phase-4B decision logic.

Phase 4B is deterministic and stdlib-only.

The frozen MatchConfigV1/AnalysisResultV1 schemas do not carry an
``elo_model`` field.  For schema v1, elo0/elo1 are therefore explicitly
defined here as *logistic Elo*, matching Phase 4A's Elo representation.

The GSPRT calculation mirrors the relevant Stockfish Fishtest logic:
- zero pentanomial cells receive a 0.001 regularization prior;
- pentanomial support is [0, .25, .5, .75, 1];
- constrained multinomial MLEs are computed at H0/H1 expected scores;
- LLR = log likelihood(H1) - log likelihood(H0);
- Wald boundaries are log(beta/(1-alpha)) and
  log((1-beta)/alpha).

No engine execution, filesystem writes, network access, or wall-clock reads
occur in this module.
"""

from __future__ import annotations

import math
from typing import Sequence

from medium_pc_audit.analysis import (
    estimate_pentanomial_elo,
    recomputed_wdl_for_schema,
    regularize_pentanomial,
)
from medium_pc_audit.schemas import registry


class SprtError(ValueError):
    """Raised when Phase-4B statistical invariants are violated."""


def _finite_number(value, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SprtError(f"{name} must be a finite number")

    return float(value)


def _positive_int(value, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise SprtError(f"{name} must be an integer >= 1")

    return value


def _validate_pentanomial(
    results: Sequence[int],
) -> tuple[int, int, int, int, int]:

    if len(results) != 5:
        raise SprtError(
            "pentanomial must contain exactly five cells"
        )

    out = []

    for index, value in enumerate(results):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise SprtError(
                f"pentanomial[{index}] must be a non-negative integer"
            )

        out.append(value)

    if sum(out) == 0:
        raise SprtError(
            "pentanomial must contain at least one pair"
        )

    return tuple(out)


def logistic_score(elo: float) -> float:
    """Convert logistic Elo to expected game score."""

    elo = _finite_number(elo, name="elo")
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_boundaries(
    *,
    alpha: float,
    beta: float,
) -> tuple[float, float]:
    """Return lower/upper Wald LLR boundaries."""

    alpha = _finite_number(alpha, name="alpha")
    beta = _finite_number(beta, name="beta")

    if not 0.0 < alpha < 1.0:
        raise SprtError("alpha must be strictly between 0 and 1")

    if not 0.0 < beta < 1.0:
        raise SprtError("beta must be strictly between 0 and 1")

    return (
        math.log(beta / (1.0 - alpha)),
        math.log((1.0 - beta) / alpha),
    )


def _pdf_stats(pdf):
    probability_sum = sum(prob for _, prob in pdf)

    if abs(probability_sum - 1.0) > 1e-8:
        raise SprtError(
            f"PDF probabilities do not sum to 1: {probability_sum}"
        )

    mean = sum(value * prob for value, prob in pdf)

    variance = sum(
        prob * (value - mean) ** 2
        for value, prob in pdf
    )

    return mean, variance


def _results_to_pdf(results):
    raw = _validate_pentanomial(results)

    regularized = regularize_pentanomial(
        raw,
        epsilon=0.001,
    )

    total = sum(regularized)

    pdf = [
        (
            index / (len(regularized) - 1),
            regularized[index] / total,
        )
        for index in range(len(regularized))
    ]

    return total, pdf


def _secular_root(shifted_pdf):
    """Solve sum p*a/(1+x*a) == 0 using deterministic bisection."""

    values = [value for value, _ in shifted_pdf]

    low_support = min(values)
    high_support = max(values)

    if not low_support < 0.0 < high_support:
        raise SprtError(
            "constrained MLE target must lie inside PDF support"
        )

    epsilon = 1e-9

    left = -1.0 / high_support + epsilon
    right = -1.0 / low_support - epsilon

    def f(x):
        return sum(
            prob * value / (1.0 + x * value)
            for value, prob in shifted_pdf
        )

    f_left = f(left)
    f_right = f(right)

    if not (f_left > 0.0 and f_right < 0.0):
        raise SprtError(
            "failed to bracket constrained-MLE secular root"
        )

    for _ in range(200):
        mid = (left + right) / 2.0
        f_mid = f(mid)

        if abs(f_mid) <= 1e-15:
            return mid

        if f_mid > 0.0:
            left = mid
        else:
            right = mid

    return (left + right) / 2.0


def _mle_expected(pdf_hat, target_score: float):
    target_score = _finite_number(
        target_score,
        name="target_score",
    )

    shifted = [
        (value - target_score, prob)
        for value, prob in pdf_hat
    ]

    root = _secular_root(shifted)

    mle = [
        (
            value,
            prob / (
                1.0
                + root * (value - target_score)
            ),
        )
        for value, prob in pdf_hat
    ]

    if any(prob <= 0.0 for _, prob in mle):
        raise SprtError(
            "constrained MLE produced non-positive probability"
        )

    mean, _ = _pdf_stats(mle)

    if abs(mean - target_score) > 1e-7:
        raise SprtError(
            "constrained MLE failed expectation check: "
            f"wanted {target_score}, got {mean}"
        )

    return mle


def logistic_gsprt_llr(
    results: Sequence[int],
    *,
    elo0: float,
    elo1: float,
) -> float:
    """Return pentanomial logistic GSPRT log-likelihood ratio."""

    elo0 = _finite_number(elo0, name="elo0")
    elo1 = _finite_number(elo1, name="elo1")

    if not elo0 < elo1:
        raise SprtError("elo0 must be strictly less than elo1")

    sample_count, pdf_hat = _results_to_pdf(results)

    score0 = logistic_score(elo0)
    score1 = logistic_score(elo1)

    mle0 = _mle_expected(pdf_hat, score0)
    mle1 = _mle_expected(pdf_hat, score1)

    llr_per_sample = sum(
        pdf_hat[index][1]
        * (
            math.log(mle1[index][1])
            - math.log(mle0[index][1])
        )
        for index in range(len(pdf_hat))
    )

    return sample_count * llr_per_sample


def validate_wdl_pentanomial_consistency(
    *,
    wins: int,
    losses: int,
    draws: int,
    pentanomial: Sequence[int],
) -> None:
    """Verify W/D/L and pentanomial describe the same paired run."""

    for name, value in (
        ("wins", wins),
        ("losses", losses),
        ("draws", draws),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise SprtError(
                f"{name} must be a non-negative integer"
            )

    p = _validate_pentanomial(pentanomial)

    if wins + losses + draws != 2 * sum(p):
        raise SprtError(
            "W/D/L game count does not equal two times pair count"
        )

    # Necessary exact relation between pentanomial and net W-L.
    expected_net = (
        2 * p[4]
        + p[3]
        - p[1]
        - 2 * p[0]
    )

    if wins - losses != expected_net:
        raise SprtError(
            "W/D/L net score is inconsistent with pentanomial"
        )

    # Middle cell can be DD, WL or LW, so draw count has a range.
    min_draws = p[3] + p[1]
    max_draws = p[3] + 2 * p[2] + p[1]

    if not min_draws <= draws <= max_draws:
        raise SprtError(
            "draw count is inconsistent with pentanomial"
        )


def evaluate_logistic_gsprt(
    pentanomial: Sequence[int],
    *,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_games: int,
) -> dict:
    """Evaluate current GSPRT state.

    Internal state may be CONTINUE.  AnalysisResultV1 finalization is only
    allowed after H0_ACCEPTED, H1_ACCEPTED, or CAP_REACHED.
    """

    p = _validate_pentanomial(pentanomial)

    elo0 = _finite_number(elo0, name="elo0")
    elo1 = _finite_number(elo1, name="elo1")

    if not elo0 < elo1:
        raise SprtError("elo0 must be strictly less than elo1")

    max_games = _positive_int(
        max_games,
        name="max_games",
    )

    lower, upper = sprt_boundaries(
        alpha=alpha,
        beta=beta,
    )

    llr = logistic_gsprt_llr(
        p,
        elo0=elo0,
        elo1=elo1,
    )

    games_played = 2 * sum(p)

    if llr < lower:
        state = "H0_ACCEPTED"
    elif llr > upper:
        state = "H1_ACCEPTED"
    elif games_played >= max_games:
        state = "CAP_REACHED"
    else:
        state = "CONTINUE"

    return {
        "elo0": elo0,
        "elo1": elo1,
        "alpha": float(alpha),
        "beta": float(beta),
        "max_games": max_games,
        "games_played": games_played,
        "llr": llr,
        "lower_bound": lower,
        "upper_bound": upper,
        "state": state,
    }


def build_analysis_result_document(
    *,
    run_id: str,
    analyzer_version: str,
    analyzed_at: str,
    wins: int,
    losses: int,
    draws: int,
    pentanomial: Sequence[int],
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_games: int,
    regression_tested: bool,
    regression_threshold_elo: float = 0.0,
) -> dict:
    """Build and schema-validate a final AnalysisResultV1 document."""

    if not isinstance(regression_tested, bool):
        raise SprtError(
            "regression_tested must be a bool"
        )

    validate_wdl_pentanomial_consistency(
        wins=wins,
        losses=losses,
        draws=draws,
        pentanomial=pentanomial,
    )

    stats = estimate_pentanomial_elo(
        pentanomial,
    )

    evaluation = evaluate_logistic_gsprt(
        pentanomial,
        elo0=elo0,
        elo1=elo1,
        alpha=alpha,
        beta=beta,
        max_games=max_games,
    )

    state = evaluation["state"]

    if state == "CONTINUE":
        raise SprtError(
            "cannot finalize AnalysisResultV1 while GSPRT state is CONTINUE"
        )

    regression_threshold_elo = _finite_number(
        regression_threshold_elo,
        name="regression_threshold_elo",
    )

    if regression_tested:
        flagged = (
            stats["elo_ci_high"]
            < regression_threshold_elo
        )

        regression_subtest = {
            "tested": True,
            "flagged": flagged,
            "ci_low": stats["elo_ci_low"],
            "ci_high": stats["elo_ci_high"],
        }
    else:
        flagged = False

        regression_subtest = {
            "tested": False,
            "flagged": False,
        }

    if state == "H1_ACCEPTED":
        if flagged:
            raise SprtError(
                "contradictory result: H1 accepted while regression flagged"
            )

        verdict = "SUPPORTED_PROGRESS"

    elif state == "H0_ACCEPTED":
        verdict = "PROGRESS_NOT_SUPPORTED"

    elif state == "CAP_REACHED":
        verdict = (
            "PROGRESS_NOT_SUPPORTED"
            if flagged
            else "INCONCLUSIVE"
        )

    else:
        raise SprtError(
            f"unexpected final GSPRT state: {state}"
        )

    document = {
        "schema_version": "v1",
        "run_id": run_id,
        "analyzer_version": analyzer_version,
        "analyzed_at": analyzed_at,
        "recomputed_wdl": recomputed_wdl_for_schema(
            wins,
            losses,
            draws,
        ),
        "sprt_result": {
            "elo0": evaluation["elo0"],
            "elo1": evaluation["elo1"],
            "alpha": evaluation["alpha"],
            "beta": evaluation["beta"],
            "max_games": evaluation["max_games"],
            "games_played": evaluation["games_played"],
            "outcome": state,
        },
        "elo_estimate": {
            "value": stats["elo"],
            "ci_low": stats["elo_ci_low"],
            "ci_high": stats["elo_ci_high"],
        },
        "verdict": verdict,
        "regression_subtest": regression_subtest,
    }

    registry.validate(
        "analysis_result",
        "v1",
        document,
    )

    return document
