import copy
import unittest

from medium_pc_audit.inbox_validate import (
    ACCEPTED,
    INCOMPLETE,
    QUARANTINED,
    classify_result_bundle,
)


H64 = {
    "candidate": "a" * 64,
    "opponent": "b" * 64,
    "engine": "c" * 64,
    "cli": "d" * 64,
    "openings": "e" * 64,
    "identity": "f" * 64,
}


def config(max_games=4):
    return {
        "schema_version": "v1",
        "config_id": "AUDIT_V0001__TEST__run001",
        "audit_id": "AUDIT_V0001",
        "matchup_key": "AUDIT_V0001__TEST",
        "matchup_type": "vs_teacher",
        "candidate": {
            "artifact_id": "CANDIDATE",
            "sha256": H64["candidate"],
        },
        "opponent": {
            "artifact_id": "TEACHER",
            "sha256": H64["opponent"],
        },
        "engine": {
            "pinned_id": "stockfish-test",
            "expected_sha256": H64["engine"],
        },
        "cli": {
            "pinned_id": "cchess-test",
            "expected_sha256": H64["cli"],
        },
        "uci_options": {
            "Threads": 1,
            "Hash": 64,
        },
        "time_control": {
            "nodes": 50000,
        },
        "threads": 1,
        "hash_mb": 64,
        "concurrency": 8,
        "opening_suite": {
            "artifact_id": "OPENINGS",
            "sha256": H64["openings"],
        },
        "repeat": True,
        "seed": 12345,
        "sprt": {
            "elo0": 0,
            "elo1": 5,
            "alpha": 0.05,
            "beta": 0.05,
            "max_games": max_games,
        },
        "created_at": "2026-09-23T00:00:00Z",
        "created_by": "phase5-test",
        "identity_sha256": H64["identity"],
    }


def manifest(cfg):
    return {
        "schema_version": "v1",
        "run_id": cfg["config_id"],
        "actual_engine_sha256": cfg["engine"]["expected_sha256"],
        "actual_cli_sha256": cfg["cli"]["expected_sha256"],
        "actual_uci_options": copy.deepcopy(cfg["uci_options"]),
        "actual_time_control": copy.deepcopy(cfg["time_control"]),
        "actual_threads": cfg["threads"],
        "actual_hash_mb": cfg["hash_mb"],
        "actual_concurrency": cfg["concurrency"],
        "actual_seed": cfg["seed"],
        "actual_opening_suite_sha256": cfg["opening_suite"]["sha256"],
        "host_identity_label": "synthetic-host",
        "executed_at_utc": "2026-09-23T00:01:00Z",
        "executed_by": "synthetic-worker",
    }


def game(index, opening, candidate_color, result):
    if candidate_color == "white":
        white_role = "candidate"
        black_role = "opponent"
    else:
        white_role = "opponent"
        black_role = "candidate"

    return {
        "game_index": index,
        "white_role": white_role,
        "black_role": black_role,
        "result": result,
        "opening_ref": opening,
    }


def games_from_penta(penta):
    games = []
    index = 1
    opening_number = 1

    patterns = {
        0: ("0-1", "1-0"),
        1: ("0-1", "1/2-1/2"),
        2: ("1/2-1/2", "1/2-1/2"),
        3: ("1-0", "1/2-1/2"),
        4: ("1-0", "0-1"),
    }

    for cell, count in enumerate(penta):
        white_result, black_result = patterns[cell]

        for _ in range(count):
            opening = f"opening-{opening_number}"

            games.append(
                game(
                    index,
                    opening,
                    "white",
                    white_result,
                )
            )
            index += 1

            games.append(
                game(
                    index,
                    opening,
                    "black",
                    black_result,
                )
            )
            index += 1

            opening_number += 1

    return games


def raw_result(cfg, games):
    return {
        "schema_version": "v1",
        "run_id": cfg["config_id"],
        "games": games,
        "claimed_wins": 0,
        "claimed_losses": 0,
        "claimed_draws": 0,
        "engine_stdout_log_ref": "match.log",
        "pgn_ref": "games.pgn",
    }


def classify(cfg, games):
    return classify_result_bundle(
        match_config=cfg,
        execution_manifest=manifest(cfg),
        raw_result=raw_result(cfg, games),
    )


class InboxClassificationTests(unittest.TestCase):

    def test_complete_cap_run_is_accepted(self):
        cfg = config(max_games=4)

        result = classify(
            cfg,
            games_from_penta([0, 0, 2, 0, 0]),
        )

        self.assertEqual(result["state"], ACCEPTED)
        self.assertEqual(
            result["sprt_state"],
            "CAP_REACHED",
        )
        self.assertEqual(result["games_played"], 4)
        self.assertEqual(result["pairs"], 2)

    def test_claimed_aggregate_is_ignored(self):
        cfg = config(max_games=4)
        games = games_from_penta([0, 0, 2, 0, 0])
        raw = raw_result(cfg, games)

        raw["claimed_wins"] = 4
        raw["claimed_losses"] = 0
        raw["claimed_draws"] = 0

        result = classify_result_bundle(
            match_config=cfg,
            execution_manifest=manifest(cfg),
            raw_result=raw,
        )

        self.assertEqual(result["state"], ACCEPTED)

    def test_manifest_mismatch_is_quarantined(self):
        cfg = config()
        m = manifest(cfg)
        m["actual_seed"] += 1

        result = classify_result_bundle(
            match_config=cfg,
            execution_manifest=m,
            raw_result=raw_result(
                cfg,
                games_from_penta([0, 0, 2, 0, 0]),
            ),
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )
        self.assertIn(
            "execution_manifest_mismatch",
            result["reason_codes"],
        )

    def test_raw_result_run_id_mismatch_is_quarantined(self):
        cfg = config()
        raw = raw_result(
            cfg,
            games_from_penta([0, 0, 2, 0, 0]),
        )
        raw["run_id"] = "AUDIT_V0001__OTHER__run001"

        result = classify_result_bundle(
            match_config=cfg,
            execution_manifest=manifest(cfg),
            raw_result=raw,
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

    def test_valid_partial_run_is_incomplete(self):
        cfg = config(max_games=400)

        result = classify(
            cfg,
            games_from_penta([0, 0, 1, 0, 0]),
        )

        self.assertEqual(
            result["state"],
            INCOMPLETE,
        )
        self.assertEqual(
            result["sprt_state"],
            "CONTINUE",
        )

    def test_empty_run_is_incomplete(self):
        cfg = config(max_games=400)

        result = classify(
            cfg,
            [],
        )

        self.assertEqual(
            result["state"],
            INCOMPLETE,
        )
        self.assertIn(
            "no_games",
            result["reason_codes"],
        )

    def test_game_count_above_cap_is_quarantined(self):
        cfg = config(max_games=4)

        result = classify(
            cfg,
            games_from_penta([0, 0, 3, 0, 0]),
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

    def test_duplicate_or_nonsequential_indices_are_quarantined(self):
        cfg = config(max_games=4)
        games = games_from_penta([0, 0, 2, 0, 0])

        games[1]["game_index"] = 1

        result = classify(
            cfg,
            games,
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

        self.assertIn(
            "game_indices_not_exactly_sequential",
            result["reason_codes"],
        )

    def test_missing_half_of_pair_is_incomplete(self):
        cfg = config(max_games=400)
        games = games_from_penta([0, 0, 2, 0, 0])
        games.pop()

        result = classify(
            cfg,
            games,
        )

        self.assertEqual(
            result["state"],
            INCOMPLETE,
        )

    def test_same_candidate_color_twice_is_quarantined(self):
        cfg = config(max_games=4)
        games = games_from_penta([0, 0, 2, 0, 0])

        games[1]["white_role"] = "candidate"
        games[1]["black_role"] = "opponent"

        result = classify(
            cfg,
            games,
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

    def test_unpaired_config_is_quarantined_for_v1_analysis(self):
        cfg = config(max_games=4)
        cfg["repeat"] = False

        result = classify(
            cfg,
            games_from_penta([0, 0, 2, 0, 0]),
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

        self.assertIn(
            "unpaired_run_not_supported_v1",
            result["reason_codes"],
        )

    def test_odd_paired_cap_is_quarantined(self):
        cfg = config(max_games=5)

        result = classify(
            cfg,
            games_from_penta([0, 0, 2, 0, 0]),
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

    def test_early_h1_terminal_run_is_accepted(self):
        penta = [0, 20, 160, 400, 220]
        games = games_from_penta(penta)
        cfg = config(max_games=4000)

        result = classify(
            cfg,
            games,
        )

        self.assertEqual(
            result["state"],
            ACCEPTED,
        )
        self.assertEqual(
            result["sprt_state"],
            "H1_ACCEPTED",
        )
        self.assertLess(
            result["games_played"],
            cfg["sprt"]["max_games"],
        )

    def test_early_h0_terminal_run_is_accepted(self):
        penta = [220, 400, 160, 20, 0]
        games = games_from_penta(penta)
        cfg = config(max_games=4000)

        result = classify(
            cfg,
            games,
        )

        self.assertEqual(
            result["state"],
            ACCEPTED,
        )
        self.assertEqual(
            result["sprt_state"],
            "H0_ACCEPTED",
        )

    def test_schema_invalid_document_is_quarantined(self):
        cfg = config()
        del cfg["seed"]

        result = classify_result_bundle(
            match_config=cfg,
            execution_manifest=manifest(config()),
            raw_result=raw_result(
                config(),
                games_from_penta([0, 0, 2, 0, 0]),
            ),
        )

        self.assertEqual(
            result["state"],
            QUARANTINED,
        )

        self.assertIn(
            "schema_invalid",
            result["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
