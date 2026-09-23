"""Tests for medium_pc_audit.pgn_parse -- Phase 3.

Every PGN used here is synthetic (see tests/fixtures/synthetic_pgn.py),
modeled after real c-chess-cli output as observed on BigPC, never real
BigPC/Stockfish/NNUE data.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import chess

from medium_pc_audit import pgn_parse
from medium_pc_audit.schemas import registry
from tests.fixtures import synthetic_pgn as fx

_LOCK_FILE = Path(__file__).parent.parent / "medium_pc_audit" / "third_party_deps" / "chess.lock.json"


def _exclusion_reasons(outcome):
    return {entry["game_index"]: entry["reason"] for entry in outcome.excluded}


class NoEventDelimiterTests(unittest.TestCase):
    def test_multi_game_pgn_with_no_event_headers_parses_all_games(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.multi_game_no_event_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(len(outcome.parsed_games), 3)
        self.assertEqual(outcome.excluded, [])
        self.assertEqual(len(outcome.raw_result_games), 3)

    def test_parsing_succeeds_though_source_pgn_has_no_event_tag_anywhere(self):
        pgn_text = fx.multi_game_no_event_pgn()
        self.assertNotIn("[Event", pgn_text)
        outcome = pgn_parse.parse_pgn_games(pgn_text, candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)
        self.assertEqual(outcome.recomputed_wdl["total_games"], 3)


class CandidateColorPerspectiveTests(unittest.TestCase):
    def test_candidate_as_white_and_candidate_as_black_both_resolved(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.multi_game_no_event_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        game1, game2, game3 = outcome.parsed_games
        self.assertEqual(game1.candidate_role, "white")
        self.assertEqual(game1.candidate_outcome, "win")
        self.assertEqual(game2.candidate_role, "black")
        self.assertEqual(game2.candidate_outcome, "win")
        self.assertEqual(game3.candidate_outcome, "draw")
        self.assertEqual(outcome.recomputed_wdl, {"wins": 2, "losses": 0, "draws": 1, "unresolved": 0, "total_games": 3})


class ColorReversedPairTests(unittest.TestCase):
    def test_identical_result_token_yields_opposite_candidate_outcomes(self):
        """Both games carry the raw Result token '1-0', but the candidate's
        color is swapped -- proving outcome is derived from White/Black
        labels + Result, never from Result (or color) alone.
        """
        outcome = pgn_parse.parse_pgn_games(
            fx.color_reversed_pair_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        game1, game2 = outcome.parsed_games
        self.assertEqual(game1.result, "1-0")
        self.assertEqual(game2.result, "1-0")
        self.assertEqual(game1.candidate_outcome, "win")
        self.assertEqual(game2.candidate_outcome, "loss")
        self.assertEqual(outcome.recomputed_wdl["wins"], 1)
        self.assertEqual(outcome.recomputed_wdl["losses"], 1)


class MostlyDrawsTests(unittest.TestCase):
    def test_mostly_draw_run_tallies_correctly(self):
        pgn_text = fx.mostly_draws_pgn(n_draws=8, n_candidate_wins=1, n_candidate_losses=1)
        outcome = pgn_parse.parse_pgn_games(pgn_text, candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)
        self.assertEqual(
            outcome.recomputed_wdl,
            {"wins": 1, "losses": 1, "draws": 8, "unresolved": 0, "total_games": 10},
        )
        self.assertEqual(len(outcome.raw_result_games), 10)
        self.assertEqual(outcome.excluded, [])


class LargeScaleSyntheticRunTests(unittest.TestCase):
    """A synthetic 400-game run proving the parser scales to the same order
    of magnitude as a real c-chess-cli 400-game audit run. The exact W/L/D
    split used here is deliberately NOT the real BigPC validation target
    (31/42/327) -- that value is reserved for a later, separate real-data
    validation step and must never be hard-coded into this test.
    """

    def test_400_game_synthetic_run(self):
        pgn_text = fx.mostly_draws_pgn(n_draws=300, n_candidate_wins=50, n_candidate_losses=50)
        outcome = pgn_parse.parse_pgn_games(pgn_text, candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)
        self.assertEqual(outcome.recomputed_wdl["total_games"], 400)
        self.assertEqual(outcome.recomputed_wdl["wins"], 50)
        self.assertEqual(outcome.recomputed_wdl["losses"], 50)
        self.assertEqual(outcome.recomputed_wdl["draws"], 300)
        self.assertEqual(outcome.recomputed_wdl["unresolved"], 0)
        self.assertEqual(len(outcome.raw_result_games), 400)
        self.assertEqual(outcome.excluded, [])
        self.assertEqual([g.game_index for g in outcome.parsed_games], list(range(1, 401)))


class MalformedMissingHeadersTests(unittest.TestCase):
    def test_missing_white_black_or_result_are_all_excluded_explicitly(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.malformed_missing_headers_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(len(outcome.parsed_games), 3)
        self.assertEqual(outcome.raw_result_games, [])
        self.assertEqual(len(outcome.excluded), 3)

        reasons = _exclusion_reasons(outcome)
        self.assertEqual(reasons[1], pgn_parse.REASON_MISSING_WHITE_OR_BLACK)
        self.assertEqual(reasons[2], pgn_parse.REASON_MISSING_WHITE_OR_BLACK)
        self.assertEqual(reasons[3], pgn_parse.REASON_MISSING_RESULT)

        self.assertEqual(outcome.recomputed_wdl["unresolved"], 3)


class UnresolvedStarResultTests(unittest.TestCase):
    def test_star_result_is_excluded_and_counted_as_unresolved(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.unresolved_star_result_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(len(outcome.parsed_games), 1)
        self.assertEqual(outcome.raw_result_games, [])
        self.assertEqual(len(outcome.excluded), 1)
        self.assertIn(pgn_parse.REASON_UNRESOLVED_OR_INVALID_RESULT, outcome.excluded[0]["reason"])
        self.assertEqual(outcome.recomputed_wdl["unresolved"], 1)


class TruncatedInputTests(unittest.TestCase):
    def test_truncated_trailing_game_does_not_raise_and_is_excluded(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.truncated_input_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(len(outcome.parsed_games), 2)
        self.assertEqual(outcome.parsed_games[0].candidate_outcome, "win")
        self.assertIsNone(outcome.parsed_games[0].exclusion_reason)
        self.assertEqual(outcome.parsed_games[1].exclusion_reason, pgn_parse.REASON_MISSING_WHITE_OR_BLACK)
        self.assertEqual(len(outcome.raw_result_games), 1)
        self.assertEqual(len(outcome.excluded), 1)

    def test_garbage_non_pgn_input_does_not_raise(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.garbage_binary_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(len(outcome.parsed_games), 1)
        self.assertEqual(outcome.parsed_games[0].exclusion_reason, pgn_parse.REASON_MISSING_WHITE_OR_BLACK)
        self.assertEqual(outcome.raw_result_games, [])

    def test_empty_input_yields_zero_games(self):
        outcome = pgn_parse.parse_pgn_games(fx.empty_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)
        self.assertEqual(outcome.parsed_games, [])
        self.assertEqual(outcome.raw_result_games, [])
        self.assertEqual(outcome.excluded, [])
        self.assertEqual(
            outcome.recomputed_wdl,
            {"wins": 0, "losses": 0, "draws": 0, "unresolved": 0, "total_games": 0},
        )


class MissingOpeningRefTests(unittest.TestCase):
    def test_missing_fen_excludes_from_raw_result_but_keeps_wdl(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.missing_fen_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        game = outcome.parsed_games[0]
        self.assertIsNone(game.fen)
        self.assertIsNone(game.opening_ref)
        self.assertEqual(game.candidate_outcome, "win")  # WDL is independent of opening_ref
        self.assertEqual(game.exclusion_reason, pgn_parse.REASON_MISSING_OPENING_REF)
        self.assertEqual(outcome.raw_result_games, [])
        self.assertEqual(len(outcome.excluded), 1)
        self.assertEqual(outcome.recomputed_wdl["wins"], 1)
        self.assertEqual(outcome.recomputed_wdl["unresolved"], 0)

    def test_malformed_fen_also_excludes_but_keeps_wdl(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.malformed_fen_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        game = outcome.parsed_games[0]
        self.assertEqual(game.fen, "not-a-real-fen")
        self.assertIsNone(game.opening_ref)
        self.assertEqual(game.candidate_outcome, "loss")
        self.assertEqual(game.exclusion_reason, pgn_parse.REASON_MISSING_OPENING_REF)
        self.assertEqual(outcome.recomputed_wdl["losses"], 1)

    def test_normalize_opening_ref_from_fen_uses_first_four_fields_only(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(
            pgn_parse.normalize_opening_ref_from_fen(fen),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        )

    def test_normalize_opening_ref_from_fen_rejects_short_or_non_string(self):
        self.assertIsNone(pgn_parse.normalize_opening_ref_from_fen("only two fields"))
        self.assertIsNone(pgn_parse.normalize_opening_ref_from_fen(None))
        self.assertIsNone(pgn_parse.normalize_opening_ref_from_fen(123))

    def test_halfmove_and_fullmove_counters_do_not_affect_opening_ref(self):
        base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
        self.assertEqual(pgn_parse.normalize_opening_ref_from_fen(base + " 0 1"), base)
        self.assertEqual(pgn_parse.normalize_opening_ref_from_fen(base + " 17 42"), base)


class UnrecognizedEngineLabelsTests(unittest.TestCase):
    def test_neither_side_matching_candidate_or_opponent_is_excluded(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.unrecognized_engine_labels_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        game = outcome.parsed_games[0]
        self.assertIsNone(game.candidate_role)
        self.assertIsNone(game.candidate_outcome)
        self.assertIn(pgn_parse.REASON_UNRECOGNIZED_ENGINE_LABELS, game.exclusion_reason)
        self.assertEqual(outcome.recomputed_wdl["unresolved"], 1)


class DeliberatelyWrongClaimedWdlTests(unittest.TestCase):
    """Proves claimed_wins/claimed_losses/claimed_draws cannot influence
    recompute_wdl_from_games(), even when they are deliberately wrong.
    """

    def test_recompute_wdl_from_games_signature_has_no_claimed_parameters(self):
        signature = inspect.signature(pgn_parse.recompute_wdl_from_games)
        param_names = set(signature.parameters)
        self.assertEqual(param_names, {"games"})
        self.assertFalse(any(name.startswith("claimed") for name in param_names))

    def test_wrong_claimed_counts_do_not_change_recomputed_counts(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.multi_game_no_event_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        # Real recomputed result for this fixture is wins=2 losses=0 draws=1.
        recomputed_a = pgn_parse.recompute_wdl_from_games(outcome.parsed_games)

        document = pgn_parse.build_raw_result_document(
            run_id="AUDIT_V0011__FROZEN_V80__run001",
            claimed_wins=999,
            claimed_losses=999,
            claimed_draws=999,
            engine_stdout_log_ref="logs/engine_stdout.log",
            pgn_ref="pgn/games.pgn",
            raw_result_games=outcome.raw_result_games,
        )
        # Schema validation passes even though claims are absurd -- the
        # schema does not (and must not) cross-check claims against games.
        registry.validate("raw_result", "v1", document)
        self.assertEqual(document["claimed_wins"], 999)

        recomputed_b = pgn_parse.recompute_wdl_from_games(outcome.parsed_games)
        self.assertEqual(recomputed_a, recomputed_b)
        self.assertEqual(recomputed_b["wins"], 2)
        self.assertEqual(recomputed_b["losses"], 0)
        self.assertEqual(recomputed_b["draws"], 1)


class Utf8HeaderTests(unittest.TestCase):
    def test_non_ascii_engine_labels_and_termination_round_trip(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.utf8_headers_pgn(),
            candidate_label=fx.UTF8_CANDIDATE_LABEL,
            opponent_label=fx.UTF8_OPPONENT_LABEL,
        )
        game = outcome.parsed_games[0]
        self.assertEqual(game.white, fx.UTF8_CANDIDATE_LABEL)
        self.assertEqual(game.black, fx.UTF8_OPPONENT_LABEL)
        self.assertEqual(game.candidate_outcome, "win")
        self.assertEqual(game.termination, "время истекло")
        self.assertIsNotNone(game.opening_ref)

    def test_utf8_pgn_file_on_disk_parses_via_parse_pgn_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games.pgn"
            path.write_text(fx.utf8_headers_pgn(), encoding="utf-8")

            outcome = pgn_parse.parse_pgn_file(
                path, candidate_label=fx.UTF8_CANDIDATE_LABEL, opponent_label=fx.UTF8_OPPONENT_LABEL
            )
            self.assertEqual(len(outcome.parsed_games), 1)
            self.assertEqual(outcome.parsed_games[0].candidate_outcome, "win")
            self.assertEqual(outcome.parsed_games[0].termination, "время истекло")


class SequentialGameIndexTests(unittest.TestCase):
    def test_game_index_is_stable_and_sequential_across_mixed_valid_and_excluded_games(self):
        combined = fx.join_games(
            fx.multi_game_no_event_pgn(),
            fx.malformed_missing_headers_pgn(),
            fx.missing_fen_pgn(),
            fx.unresolved_star_result_pgn(),
        )
        outcome = pgn_parse.parse_pgn_games(combined, candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)

        indices = [g.game_index for g in outcome.parsed_games]
        self.assertEqual(indices, list(range(1, len(outcome.parsed_games) + 1)))

        raw_indices = {entry["game_index"] for entry in outcome.raw_result_games}
        excluded_indices = {entry["game_index"] for entry in outcome.excluded}
        self.assertEqual(raw_indices | excluded_indices, set(indices))
        self.assertEqual(raw_indices & excluded_indices, set())


class NoSilentExclusionsTests(unittest.TestCase):
    """Every parsed game must appear in exactly one of raw_result_games or
    excluded -- never simply vanish from the accounting.
    """

    def _assert_full_accounting(self, pgn_text):
        outcome = pgn_parse.parse_pgn_games(pgn_text, candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT)
        self.assertEqual(len(outcome.raw_result_games) + len(outcome.excluded), len(outcome.parsed_games))

        raw_indices = {entry["game_index"] for entry in outcome.raw_result_games}
        excluded_indices = {entry["game_index"] for entry in outcome.excluded}
        all_indices = {g.game_index for g in outcome.parsed_games}
        self.assertEqual(raw_indices | excluded_indices, all_indices)
        return outcome

    def test_mixed_fixture_has_full_accounting(self):
        combined = fx.join_games(
            fx.multi_game_no_event_pgn(),
            fx.malformed_missing_headers_pgn(),
            fx.missing_fen_pgn(),
            fx.malformed_fen_pgn(),
            fx.unresolved_star_result_pgn(),
            fx.unrecognized_engine_labels_pgn(),
        )
        self._assert_full_accounting(combined)

    def test_truncated_fixture_has_full_accounting(self):
        self._assert_full_accounting(fx.truncated_input_pgn())

    def test_garbage_fixture_has_full_accounting(self):
        self._assert_full_accounting(fx.garbage_binary_pgn())


class SchemaValidationTests(unittest.TestCase):
    def test_build_raw_result_document_validates_against_frozen_phase0_schema(self):
        outcome = pgn_parse.parse_pgn_games(
            fx.multi_game_no_event_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        document = pgn_parse.build_raw_result_document(
            run_id="AUDIT_V0011__FROZEN_V80__run001",
            claimed_wins=2,
            claimed_losses=0,
            claimed_draws=1,
            engine_stdout_log_ref="logs/engine_stdout.log",
            pgn_ref="pgn/games.pgn",
            raw_result_games=outcome.raw_result_games,
        )
        registry.validate("raw_result", "v1", document)  # must not raise
        self.assertEqual(document["schema_version"], "v1")
        self.assertEqual(len(document["games"]), 3)

    def test_build_raw_result_document_with_excluded_only_games_still_validates(self):
        """Games list may legitimately be empty (every game excluded) --
        the frozen schema allows an empty games array, and this must still
        validate cleanly; exclusions are recorded separately, never by
        smuggling a placeholder into 'games'.
        """
        outcome = pgn_parse.parse_pgn_games(
            fx.malformed_missing_headers_pgn(), candidate_label=fx.CANDIDATE, opponent_label=fx.OPPONENT
        )
        self.assertEqual(outcome.raw_result_games, [])
        document = pgn_parse.build_raw_result_document(
            run_id="AUDIT_V0011__FROZEN_V80__run002",
            claimed_wins=0,
            claimed_losses=0,
            claimed_draws=0,
            engine_stdout_log_ref="logs/engine_stdout.log",
            pgn_ref="pgn/games.pgn",
            raw_result_games=outcome.raw_result_games,
        )
        registry.validate("raw_result", "v1", document)
        self.assertEqual(document["games"], [])

    def test_build_raw_result_document_rejects_games_missing_opening_ref(self):
        """The frozen Phase 0 schema is never bypassed: a hand-crafted game
        entry missing opening_ref must still fail schema validation.
        """
        bad_games = [
            {
                "game_index": 1,
                "white_role": "candidate",
                "black_role": "opponent",
                "result": "1-0",
                # opening_ref deliberately omitted
            }
        ]
        with self.assertRaises(registry.SchemaValidationError):
            pgn_parse.build_raw_result_document(
                run_id="AUDIT_V0011__FROZEN_V80__run003",
                claimed_wins=1,
                claimed_losses=0,
                claimed_draws=0,
                engine_stdout_log_ref="logs/engine_stdout.log",
                pgn_ref="pgn/games.pgn",
                raw_result_games=bad_games,
            )


class CallArgumentValidationTests(unittest.TestCase):
    def test_rejects_empty_candidate_label(self):
        with self.assertRaises(pgn_parse.PgnParseError):
            pgn_parse.parse_pgn_games("", candidate_label="", opponent_label=fx.OPPONENT)

    def test_rejects_empty_opponent_label(self):
        with self.assertRaises(pgn_parse.PgnParseError):
            pgn_parse.parse_pgn_games("", candidate_label=fx.CANDIDATE, opponent_label="")

    def test_rejects_non_string_label(self):
        with self.assertRaises(pgn_parse.PgnParseError):
            pgn_parse.parse_pgn_games("", candidate_label=None, opponent_label=fx.OPPONENT)

    def test_rejects_identical_candidate_and_opponent_labels(self):
        with self.assertRaises(pgn_parse.PgnParseError):
            pgn_parse.parse_pgn_games("", candidate_label=fx.CANDIDATE, opponent_label=fx.CANDIDATE)


class DependencyProvenanceTests(unittest.TestCase):
    """Guards the Phase 3 dependency pin/provenance file against drift from
    what is actually installed and importable.
    """

    def test_lock_file_records_current_package_name_and_version(self):
        with _LOCK_FILE.open("r", encoding="utf-8") as f:
            lock = json.load(f)

        self.assertEqual(lock["package"], "chess")
        self.assertEqual(lock["obsolete_package_name"], "python-chess")
        self.assertEqual(lock["version"], "1.11.2")
        self.assertEqual(len(lock["sdist"]["sha256"]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in lock["sdist"]["sha256"]))

    def test_lock_file_version_matches_installed_chess_version(self):
        with _LOCK_FILE.open("r", encoding="utf-8") as f:
            lock = json.load(f)
        self.assertEqual(chess.__version__, lock["version"])


if __name__ == "__main__":
    unittest.main()
