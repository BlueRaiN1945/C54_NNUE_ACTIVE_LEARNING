"""Tests for deterministic PGN position extraction (M0 corpus pipeline step 2)."""

import unittest

import chess

from medium_pc_audit.position_extract import (
    ExtractedPosition,
    PositionExtractError,
    extract_positions,
)

# A short, real, legal game (Scholar's mate skeleton) -- enough plies to
# exercise skip/determinism without being large.
SHORT_GAME_MOVES = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]

SHORT_GAME_PGN = (
    '[White "SELFPLAY_A"]\n'
    '[Black "SELFPLAY_B"]\n'
    '[Result "*"]\n'
    "\n"
    "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 *\n"
)

ILLEGAL_MOVE_PGN = (
    '[White "SELFPLAY_A"]\n'
    '[Black "SELFPLAY_B"]\n'
    '[Result "*"]\n'
    "\n"
    "1. e4 e5 2. Ke2 Ke2 *\n"  # second Ke2 is illegal
)


def ground_truth_fens(moves):
    """Independently derive expected FEN after each move, not via the module under test."""

    board = chess.Board()
    fens = []
    for san in moves:
        board.push_san(san)
        fens.append(board.fen())
    return fens


class ExtractPositionsTests(unittest.TestCase):
    def test_extracts_every_ply_by_default(self):
        positions = extract_positions(SHORT_GAME_PGN)

        self.assertEqual(len(positions), len(SHORT_GAME_MOVES))
        self.assertEqual(
            [p.fen for p in positions],
            ground_truth_fens(SHORT_GAME_MOVES),
        )

    def test_ply_numbers_are_sequential_starting_at_one(self):
        positions = extract_positions(SHORT_GAME_PGN)

        self.assertEqual([p.ply for p in positions], [1, 2, 3, 4, 5, 6])

    def test_game_index_is_recorded(self):
        positions = extract_positions(SHORT_GAME_PGN)

        self.assertTrue(all(p.game_index == 1 for p in positions))

    def test_skip_first_n_plies_excludes_early_positions(self):
        full = extract_positions(SHORT_GAME_PGN)
        skipped = extract_positions(SHORT_GAME_PGN, skip_first_n_plies=2)

        self.assertEqual(len(skipped), len(full) - 2)
        self.assertEqual([p.fen for p in skipped], [p.fen for p in full[2:]])
        self.assertEqual(skipped[0].ply, 3)

    def test_skip_all_plies_yields_empty_list(self):
        positions = extract_positions(SHORT_GAME_PGN, skip_first_n_plies=100)

        self.assertEqual(positions, [])

    def test_negative_skip_rejected(self):
        with self.assertRaises(PositionExtractError):
            extract_positions(SHORT_GAME_PGN, skip_first_n_plies=-1)

    def test_deterministic_across_calls(self):
        first = extract_positions(SHORT_GAME_PGN)
        second = extract_positions(SHORT_GAME_PGN)

        self.assertEqual(first, second)

    def test_multiple_games_reset_ply_and_increment_game_index(self):
        two_games = SHORT_GAME_PGN + "\n" + SHORT_GAME_PGN

        positions = extract_positions(two_games)

        self.assertEqual(len(positions), 2 * len(SHORT_GAME_MOVES))
        self.assertEqual(positions[0].game_index, 1)
        self.assertEqual(positions[0].ply, 1)
        self.assertEqual(positions[len(SHORT_GAME_MOVES)].game_index, 2)
        self.assertEqual(positions[len(SHORT_GAME_MOVES)].ply, 1)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(extract_positions(""), [])

    def test_illegal_move_raises_with_context(self):
        with self.assertRaises(PositionExtractError) as ctx:
            extract_positions(ILLEGAL_MOVE_PGN)

        message = str(ctx.exception)
        self.assertIn("game", message.lower())

    def test_result_entries_are_frozen_dataclass(self):
        import dataclasses

        positions = extract_positions(SHORT_GAME_PGN)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            positions[0].fen = "tampered"

    def test_accepts_text_stream_as_well_as_string(self):
        import io

        stream_result = extract_positions(io.StringIO(SHORT_GAME_PGN))
        string_result = extract_positions(SHORT_GAME_PGN)

        self.assertEqual(stream_result, string_result)

    def test_extracted_position_is_exactly_expected_type(self):
        positions = extract_positions(SHORT_GAME_PGN)

        self.assertIsInstance(positions[0], ExtractedPosition)


if __name__ == "__main__":
    unittest.main()
