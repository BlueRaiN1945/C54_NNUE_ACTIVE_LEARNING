"""Tests for the pure, engine-free logic of the generic rescoring script.

This test module NEVER spawns a subprocess or a real Stockfish engine. It
imports only parse_info_line() and validate_input_header() -- both pure
functions with no side effects -- from execution_host.rescore_generic. That
module's UCIEngine class is deliberately not touched here: constructing one
launches a real process, which belongs on the execution host, not in this
repository's test suite.
"""

import unittest

from execution_host.rescore_generic import (
    INPUT_REQUIRED,
    RescoreGenericError,
    parse_info_line,
    validate_input_header,
)


class ParseInfoLineTests(unittest.TestCase):
    def test_parses_exact_cp_score(self):
        line = (
            "info depth 10 seldepth 14 multipv 1 score cp 25 nodes 12345 "
            "nps 500000 wdl 500 300 200 pv e2e4 e7e5"
        )

        a = parse_info_line(line)

        self.assertIsNotNone(a)
        self.assertEqual(a.sf_score_type, "cp")
        self.assertEqual(a.sf_score, "25")
        self.assertEqual(a.sf_bound, "exact")
        self.assertEqual(a.sf_nodes, "12345")
        self.assertEqual(a.sf_nps, "500000")
        self.assertEqual((a.sf_wdl_w, a.sf_wdl_d, a.sf_wdl_l), ("500", "300", "200"))
        self.assertEqual(a.sf_pv, "e2e4 e7e5")

    def test_parses_mate_score(self):
        line = "info depth 5 score mate 3 nodes 100 pv e2e4"

        a = parse_info_line(line)

        self.assertEqual(a.sf_score_type, "mate")
        self.assertEqual(a.sf_score, "3")

    def test_parses_lowerbound(self):
        line = "info depth 8 score cp 40 lowerbound nodes 200 pv d2d4"

        a = parse_info_line(line)

        self.assertEqual(a.sf_bound, "lowerbound")

    def test_parses_upperbound(self):
        line = "info depth 8 score cp -40 upperbound nodes 200 pv d2d4"

        a = parse_info_line(line)

        self.assertEqual(a.sf_bound, "upperbound")

    def test_defaults_to_exact_when_no_bound_token(self):
        line = "info depth 8 score cp 12 nodes 50 pv e2e4"

        a = parse_info_line(line)

        self.assertEqual(a.sf_bound, "exact")

    def test_ignores_info_string_lines(self):
        self.assertIsNone(parse_info_line("info string NNUE evaluation using ..."))

    def test_ignores_lines_without_score(self):
        self.assertIsNone(parse_info_line("info depth 5 nodes 100 pv e2e4"))

    def test_ignores_non_info_lines(self):
        self.assertIsNone(parse_info_line("bestmove e2e4 ponder e7e5"))
        self.assertIsNone(parse_info_line("uciok"))

    def test_missing_wdl_and_pv_leave_fields_empty(self):
        line = "info depth 3 score cp 5 nodes 10"

        a = parse_info_line(line)

        self.assertEqual(a.sf_wdl_w, "")
        self.assertEqual(a.sf_pv, "")

    def test_score_with_nothing_after_returns_none(self):
        self.assertIsNone(parse_info_line("info score"))

    def test_score_missing_type_token_returns_none(self):
        self.assertIsNone(parse_info_line("info score 25 nodes 10"))

    def test_seldepth_and_time_are_captured(self):
        line = "info depth 4 seldepth 9 score cp 0 time 123 nodes 10"

        a = parse_info_line(line)

        self.assertEqual(a.sf_seldepth, "9")
        self.assertEqual(a.sf_time_ms, "123")

    def test_malformed_line_does_not_raise(self):
        # Garbage after "score" that isn't cp/mate must return None, not crash.
        self.assertIsNone(parse_info_line("info score notanumber"))


class ValidateInputHeaderTests(unittest.TestCase):
    def test_only_fen_is_required(self):
        self.assertEqual(list(INPUT_REQUIRED), ["fen"])

    def test_header_with_only_fen_is_accepted(self):
        validate_input_header(["fen"])

    def test_header_with_fen_plus_extra_columns_is_accepted(self):
        validate_input_header(["fen", "anything", "else"])

    def test_header_missing_fen_is_rejected(self):
        with self.assertRaises(RescoreGenericError):
            validate_input_header(["source", "weight"])

    def test_ola_specific_columns_are_not_required(self):
        """The whole point of this variant: OLA's columns must not be needed."""

        validate_input_header(["fen", "sf_score_type", "sf_score"])


if __name__ == "__main__":
    unittest.main()
