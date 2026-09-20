import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from miner_train_batch import build_probe_moves
from stockfish_session import StockfishSession


class TestStockfishSessionContracts(unittest.TestCase):

    def test_search_passes_expected_multipv_to_parser(self):
        session = object.__new__(StockfishSession)

        session.set_multipv = Mock()
        session.set_position = Mock()
        session._send = Mock()

        session._readline = Mock(side_effect=[
            (
                "info depth 12 multipv 1 "
                "score cp 10 nodes 100 "
                "pv a2a3"
            ),
            "bestmove a2a3",
        ])

        parsed = object()

        with patch(
            "stockfish_session.parse_search_output",
            return_value=parsed,
        ) as parser:
            result, lines = session.search(
                "8/8/8/8/8/8/8/K6k w - - 0 1",
                nodes=50000,
                multipv=3,
            )

        self.assertIs(result, parsed)

        session.set_multipv.assert_called_once_with(3)

        session.set_position.assert_called_once_with(
            "8/8/8/8/8/8/8/K6k w - - 0 1"
        )

        session._send.assert_called_once_with(
            "go nodes 50000"
        )

        parser.assert_called_once_with(
            lines,
            expected_multipv=3,
        )


class TestStockfishSessionCleanup(unittest.TestCase):

    def test_close_closes_process_pipes(self):
        session = object.__new__(StockfishSession)

        stdin = Mock()
        stdin.closed = False

        stdout = Mock()
        stdout.closed = False

        proc = Mock()
        proc.poll.return_value = 0
        proc.stdin = stdin
        proc.stdout = stdout

        session.proc = proc

        session.close()

        stdin.close.assert_called_once_with()
        stdout.close.assert_called_once_with()



class TestProbeUnionContracts(unittest.TestCase):

    def test_candidate_bestmove_cannot_fall_out_of_probes(self):
        # Reproduces the important shape of the old position-7 bug:
        # final UCI bestmove is NOT coherent MultiPV rank1.
        candidate = SimpleNamespace(
            bestmove="d6e5",
            infos={
                1: SimpleNamespace(move="b6b3"),
                2: SimpleNamespace(move="d6e7"),
                3: SimpleNamespace(move="c7c5"),
            },
        )

        teacher = SimpleNamespace(
            bestmove="b6b2",
        )

        probes = build_probe_moves(
            candidate,
            teacher,
        )

        self.assertEqual(
            probes,
            [
                "d6e5",
                "b6b3",
                "d6e7",
                "c7c5",
                "b6b2",
            ],
        )

        self.assertIn(
            candidate.bestmove,
            probes,
        )

    def test_probe_union_deduplicates_moves(self):
        candidate = SimpleNamespace(
            bestmove="a2a3",
            infos={
                1: SimpleNamespace(move="a2a3"),
                2: SimpleNamespace(move="b2b3"),
                3: SimpleNamespace(move="c2c3"),
            },
        )

        teacher = SimpleNamespace(
            bestmove="b2b3",
        )

        probes = build_probe_moves(
            candidate,
            teacher,
        )

        self.assertEqual(
            probes,
            [
                "a2a3",
                "b2b3",
                "c2c3",
            ],
        )

        self.assertEqual(
            len(probes),
            len(set(probes)),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
