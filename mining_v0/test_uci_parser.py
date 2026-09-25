import unittest

from uci_parser import (
    parse_search_output,
    parse_static_eval,
)


class TestStaticEval(unittest.TestCase):

    def test_real_official_static(self):
        r = parse_static_eval([
            "NNUE evaluation          -183 (side to move, internal units)",
            "NNUE evaluation        -0.70 (white side)",
            "Final evaluation      -0.74 (white side) [with scaled NNUE, ...]",
        ])

        self.assertEqual(r.status, "ok")
        self.assertEqual(r.score_raw_stm, -183)


class TestLastExactSemantics(unittest.TestCase):

    def test_exact_beats_later_lowerbound(self):
        lines = [
            (
                "info depth 18 seldepth 22 "
                "score cp -83 wdl 1 671 328 "
                "nodes 350000 time 200 "
                "pv e5f3 c8e6"
            ),

            # Later line, but it is only a bound.
            (
                "info depth 20 seldepth 24 "
                "score cp -72 lowerbound "
                "wdl 1 755 244 "
                "nodes 425035 time 250 "
                "pv e5f3 c8e6"
            ),

            "bestmove e5f3 ponder c8e6",
        ]

        r = parse_search_output(lines)

        self.assertEqual(r.bestmove, "e5f3")

        info = r.infos[1]

        # Critical assertion:
        self.assertEqual(info.bound, "exact")
        self.assertEqual(info.score_value, -83)

    def test_bound_fallback_if_no_exact_exists(self):
        lines = [
            (
                "info depth 20 "
                "score cp -72 lowerbound "
                "wdl 1 755 244 "
                "nodes 425035 "
                "pv e5f3 c8e6"
            ),
            "bestmove e5f3",
        ]

        r = parse_search_output(lines)

        info = r.infos[1]

        self.assertEqual(info.bound, "lowerbound")
        self.assertEqual(info.score_value, -72)


class TestCoherentMultiPV(unittest.TestCase):

    def test_uses_last_complete_same_depth_round(self):
        lines = [
            # Complete coherent depth-12 round.
            (
                "info depth 12 multipv 1 "
                "score cp 10 "
                "nodes 30000 "
                "pv a2a3 a7a6"
            ),
            (
                "info depth 12 multipv 2 "
                "score cp 5 "
                "nodes 30000 "
                "pv b2b3 a7a6"
            ),
            (
                "info depth 12 multipv 3 "
                "score cp 0 "
                "nodes 30000 "
                "pv c2c3 a7a6"
            ),

            # Incomplete later depth-13 round:
            # only rank 1 arrived before search ended.
            (
                "info depth 13 multipv 1 "
                "score cp 20 "
                "nodes 50000 "
                "pv d2d4 d7d5"
            ),

            "bestmove d2d4 ponder d7d5",
        ]

        r = parse_search_output(
            lines,
            expected_multipv=3,
        )

        # UCI decision is still the actual final bestmove.
        self.assertEqual(
            r.bestmove,
            "d2d4",
        )

        # But stored MultiPV evidence must come from one
        # complete coherent depth-12 round.
        self.assertEqual(
            r.infos[1].move,
            "a2a3",
        )
        self.assertEqual(
            r.infos[2].move,
            "b2b3",
        )
        self.assertEqual(
            r.infos[3].move,
            "c2c3",
        )

        self.assertEqual(
            {
                r.infos[1].depth,
                r.infos[2].depth,
                r.infos[3].depth,
            },
            {12},
        )

    def test_coherent_margin_cannot_mix_depths(self):
        lines = [
            (
                "info depth 14 multipv 1 "
                "score cp 50 "
                "nodes 40000 "
                "pv a2a3"
            ),
            (
                "info depth 14 multipv 2 "
                "score cp 20 "
                "nodes 40000 "
                "pv b2b3"
            ),
            (
                "info depth 14 multipv 3 "
                "score cp 10 "
                "nodes 40000 "
                "pv c2c3"
            ),

            # A later isolated rank-2 exact must NOT replace
            # rank 2 inside the coherent depth-14 snapshot.
            (
                "info depth 15 multipv 2 "
                "score cp 100 "
                "nodes 50000 "
                "pv h2h4"
            ),

            "bestmove a2a3",
        ]

        r = parse_search_output(
            lines,
            expected_multipv=3,
        )

        margin = (
            r.infos[1].score_value
            - r.infos[2].score_value
        )

        self.assertEqual(
            margin,
            30,
        )
        self.assertGreaterEqual(
            margin,
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
