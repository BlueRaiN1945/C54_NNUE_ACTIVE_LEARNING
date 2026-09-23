"""Pure Phase-5A result-bundle classification.

This module decides whether a returned run is structurally/provenance-ready
for analysis, incomplete, or quarantined.

No filesystem writes occur in Phase 5A.  Immutable inbox publication and
state-history persistence are Phase 5B.

Claimed aggregate W/L/D fields are intentionally never trusted here.
"""

from __future__ import annotations

from collections import defaultdict

from medium_pc_audit.manifest_verify import verify_execution_manifest
from medium_pc_audit.schemas import registry
from medium_pc_audit.sprt import evaluate_logistic_gsprt


ACCEPTED = "ACCEPTED"
INCOMPLETE = "INCOMPLETE"
QUARANTINED = "QUARANTINED"


_PAIR_SCORE_TO_INDEX = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
    1.5: 3,
    2.0: 4,
}


class _IncompletePair(Exception):
    pass


class _MalformedPair(Exception):
    pass


def _schema_error(doc_type: str, document: dict):
    try:
        registry.validate(
            doc_type,
            "v1",
            document,
        )
    except Exception as exc:
        return str(exc)

    return None


def _candidate_game_view(game: dict):
    white_role = game["white_role"]
    black_role = game["black_role"]
    result = game["result"]

    if {white_role, black_role} != {
        "candidate",
        "opponent",
    }:
        raise _MalformedPair(
            "each game must contain exactly one candidate and one opponent"
        )

    if white_role == "candidate":
        candidate_color = "white"

        if result == "1-0":
            points = 1.0
        elif result == "0-1":
            points = 0.0
        else:
            points = 0.5

    else:
        candidate_color = "black"

        if result == "0-1":
            points = 1.0
        elif result == "1-0":
            points = 0.0
        else:
            points = 0.5

    return candidate_color, points


def pentanomial_from_raw_games(
    games: list,
) -> tuple[int, int, int, int, int]:
    """Build LL/LD/DDWL/WD/WW counts from RawResultV1 games."""

    grouped = defaultdict(list)

    for game in games:
        grouped[game["opening_ref"]].append(game)

    counts = [0, 0, 0, 0, 0]

    for opening_ref, pair in grouped.items():

        if len(pair) == 1:
            raise _IncompletePair(
                f"{opening_ref}: only one game of pair is present"
            )

        if len(pair) > 2:
            raise _MalformedPair(
                f"{opening_ref}: more than two games share opening_ref"
            )

        views = [
            _candidate_game_view(game)
            for game in pair
        ]

        colors = {
            color
            for color, _ in views
        }

        if colors != {"white", "black"}:
            raise _MalformedPair(
                f"{opening_ref}: pair is not color reversed"
            )

        score = sum(
            points
            for _, points in views
        )

        try:
            index = _PAIR_SCORE_TO_INDEX[score]
        except KeyError as exc:
            raise _MalformedPair(
                f"{opening_ref}: unsupported pair score {score!r}"
            ) from exc

        counts[index] += 1

    return tuple(counts)


def classify_result_bundle(
    *,
    match_config: dict,
    execution_manifest: dict,
    raw_result: dict,
) -> dict:
    """Classify a returned run without throwing routine intake failures."""

    schema_failures = []

    for doc_type, document in (
        ("match_config", match_config),
        ("execution_manifest", execution_manifest),
        ("raw_result", raw_result),
    ):
        error = _schema_error(
            doc_type,
            document,
        )

        if error is not None:
            schema_failures.append(
                {
                    "document": doc_type,
                    "error": error,
                }
            )

    if schema_failures:
        return {
            "state": QUARANTINED,
            "reason_codes": ["schema_invalid"],
            "schema_failures": schema_failures,
            "games_played": None,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    provenance = verify_execution_manifest(
        match_config,
        execution_manifest,
    )

    if not provenance["ok"]:
        return {
            "state": QUARANTINED,
            "reason_codes": ["execution_manifest_mismatch"],
            "provenance": provenance,
            "games_played": None,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    if raw_result["run_id"] != match_config["config_id"]:
        return {
            "state": QUARANTINED,
            "reason_codes": ["raw_result_run_id_mismatch"],
            "provenance": provenance,
            "games_played": len(raw_result["games"]),
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    if match_config["repeat"] is not True:
        return {
            "state": QUARANTINED,
            "reason_codes": ["unpaired_run_not_supported_v1"],
            "provenance": provenance,
            "games_played": len(raw_result["games"]),
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    max_games = match_config["sprt"]["max_games"]

    if max_games % 2 != 0:
        return {
            "state": QUARANTINED,
            "reason_codes": ["paired_max_games_must_be_even"],
            "provenance": provenance,
            "games_played": len(raw_result["games"]),
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    games = raw_result["games"]
    games_played = len(games)

    if games_played == 0:
        return {
            "state": INCOMPLETE,
            "reason_codes": ["no_games"],
            "provenance": provenance,
            "games_played": 0,
            "pairs": 0,
            "pentanomial": None,
            "sprt_state": None,
        }

    if games_played > max_games:
        return {
            "state": QUARANTINED,
            "reason_codes": ["game_count_exceeds_max_games"],
            "provenance": provenance,
            "games_played": games_played,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    indices = [
        game["game_index"]
        for game in games
    ]

    expected_indices = list(
        range(
            1,
            games_played + 1,
        )
    )

    if indices != expected_indices:
        return {
            "state": QUARANTINED,
            "reason_codes": ["game_indices_not_exactly_sequential"],
            "provenance": provenance,
            "games_played": games_played,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    if games_played % 2 != 0:
        return {
            "state": INCOMPLETE,
            "reason_codes": ["odd_game_count_for_paired_run"],
            "provenance": provenance,
            "games_played": games_played,
            "pairs": games_played // 2,
            "pentanomial": None,
            "sprt_state": None,
        }

    try:
        pentanomial = pentanomial_from_raw_games(
            games
        )

    except _IncompletePair as exc:
        return {
            "state": INCOMPLETE,
            "reason_codes": ["incomplete_opening_pair"],
            "detail": str(exc),
            "provenance": provenance,
            "games_played": games_played,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    except _MalformedPair as exc:
        return {
            "state": QUARANTINED,
            "reason_codes": ["malformed_opening_pair"],
            "detail": str(exc),
            "provenance": provenance,
            "games_played": games_played,
            "pairs": None,
            "pentanomial": None,
            "sprt_state": None,
        }

    pairs = sum(pentanomial)

    if pairs * 2 != games_played:
        return {
            "state": QUARANTINED,
            "reason_codes": ["pair_accounting_mismatch"],
            "provenance": provenance,
            "games_played": games_played,
            "pairs": pairs,
            "pentanomial": list(pentanomial),
            "sprt_state": None,
        }

    sprt = match_config["sprt"]

    try:
        evaluation = evaluate_logistic_gsprt(
            pentanomial,
            elo0=sprt["elo0"],
            elo1=sprt["elo1"],
            alpha=sprt["alpha"],
            beta=sprt["beta"],
            max_games=sprt["max_games"],
        )
    except Exception as exc:
        return {
            "state": QUARANTINED,
            "reason_codes": ["analysis_precheck_failed"],
            "detail": str(exc),
            "provenance": provenance,
            "games_played": games_played,
            "pairs": pairs,
            "pentanomial": list(pentanomial),
            "sprt_state": None,
        }

    sprt_state = evaluation["state"]

    if sprt_state in {
        "H0_ACCEPTED",
        "H1_ACCEPTED",
        "CAP_REACHED",
    }:
        return {
            "state": ACCEPTED,
            "reason_codes": [],
            "provenance": provenance,
            "games_played": games_played,
            "pairs": pairs,
            "pentanomial": list(pentanomial),
            "sprt_state": sprt_state,
        }

    return {
        "state": INCOMPLETE,
        "reason_codes": ["sprt_continue_before_cap"],
        "provenance": provenance,
        "games_played": games_played,
        "pairs": pairs,
        "pentanomial": list(pentanomial),
        "sprt_state": sprt_state,
    }
