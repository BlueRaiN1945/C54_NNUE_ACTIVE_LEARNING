import os
import unittest
from pathlib import Path

from stockfish_session import StockfishSession


SF = os.environ.get("C54_STOCKFISH")
OFFICIAL = os.environ.get("C54_OFFICIAL_NET")
CANDIDATE = os.environ.get("C54_CANDIDATE_NET")

READY = all(
    value and Path(value).is_file()
    for value in (
        SF,
        OFFICIAL,
        CANDIDATE,
    )
)

FEN = (
    "1rb2rk1/2q1p1bp/3n1pp1/1p1pN3/"
    "2pPnPP1/2P1P2P/1P1N2B1/R2QBRK1 "
    "w - - 0 1"
)


@unittest.skipUnless(
    READY,
    (
        "Real Stockfish artifacts not configured. "
        "Set C54_STOCKFISH, C54_OFFICIAL_NET, "
        "and C54_CANDIDATE_NET."
    ),
)
class TestStockfishSessionReal(unittest.TestCase):

    def test_real_session_and_searchmoves(self):
        with StockfishSession(
            SF,
            CANDIDATE,
            threads=1,
            hash_mb=32,
            multipv=3,
        ) as candidate:

            static, _ = candidate.static_eval(FEN)

            self.assertEqual(
                static.status,
                "ok",
            )

            candidate.new_game()

            result, _ = candidate.search(
                FEN,
                nodes=50_000,
                multipv=3,
            )

            self.assertIsNotNone(
                result.bestmove
            )

            self.assertEqual(
                sorted(result.infos),
                [1, 2, 3],
            )

            self.assertEqual(
                len({
                    result.infos[1].depth,
                    result.infos[2].depth,
                    result.infos[3].depth,
                }),
                1,
            )

            candidate_bestmove = (
                result.bestmove
            )

        with StockfishSession(
            SF,
            OFFICIAL,
            threads=1,
            hash_mb=32,
            multipv=1,
        ) as teacher:

            static, _ = teacher.static_eval(FEN)

            self.assertEqual(
                static.status,
                "ok",
            )

            teacher.new_game()

            result, _ = teacher.search(
                FEN,
                nodes=425_000,
                multipv=1,
            )

            self.assertIsNotNone(
                result.bestmove
            )

            teacher.new_game()

            restricted, _ = teacher.search(
                FEN,
                nodes=100_000,
                multipv=1,
                searchmoves=[
                    candidate_bestmove
                ],
            )

            self.assertEqual(
                restricted.bestmove,
                candidate_bestmove,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
