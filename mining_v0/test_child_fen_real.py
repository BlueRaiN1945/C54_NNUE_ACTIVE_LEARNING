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

ROOT_FEN = (
    "1rb2rk1/2q1p1bp/3n1pp1/1p1pN3/"
    "2pPnPP1/2P1P2P/1P1N2B1/R2QBRK1 "
    "w - - 0 1"
)

MOVE = "e5f3"


def stm(fen: str) -> str:
    return fen.split()[1]


def to_root_pov(
    raw_stm_score: int,
    root_fen: str,
    evaluated_fen: str,
) -> int:
    if stm(root_fen) == stm(evaluated_fen):
        return raw_stm_score

    return -raw_stm_score


@unittest.skipUnless(
    READY,
    (
        "Real Stockfish artifacts not configured. "
        "Set C54_STOCKFISH, C54_OFFICIAL_NET, "
        "and C54_CANDIDATE_NET."
    ),
)
class TestChildFenReal(unittest.TestCase):

    def test_child_fen_and_sign_normalization(self):
        with StockfishSession(
            SF,
            CANDIDATE,
            threads=1,
            hash_mb=32,
        ) as candidate:

            child_fen = candidate.fen_after_moves(
                ROOT_FEN,
                [MOVE],
            )

            result, _ = candidate.static_eval(
                child_fen
            )

            self.assertNotEqual(
                stm(ROOT_FEN),
                stm(child_fen),
            )

            self.assertEqual(
                result.status,
                "ok",
            )

            root_score = to_root_pov(
                result.score_raw_stm,
                ROOT_FEN,
                child_fen,
            )

            self.assertEqual(
                root_score,
                -result.score_raw_stm,
            )

        with StockfishSession(
            SF,
            OFFICIAL,
            threads=1,
            hash_mb=32,
        ) as official:

            result, _ = official.static_eval(
                child_fen
            )

            self.assertEqual(
                result.status,
                "ok",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
