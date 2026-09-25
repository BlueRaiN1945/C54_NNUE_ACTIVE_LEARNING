import sqlite3
import tempfile
import unittest
from pathlib import Path

from db_writer import MiningDB
from uci_parser import (
    SearchInfo,
    SearchResult,
    StaticEvalResult,
)


SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "schema"
    / "MINING_DB_V1.sql"
)


def make_database(path: Path):
    schema = SCHEMA.read_text(
        encoding="utf-8"
    )

    db = sqlite3.connect(path)
    db.executescript(schema)
    db.commit()
    db.close()


class TestDBWriter(unittest.TestCase):

    def test_round_trip_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            make_database(db_path)

            with MiningDB(db_path) as db:
                engine_id = db.register_engine(
                    name="Stockfish test",
                    sha256="a" * 64,
                    stockfish_commit="17a6c8f1",
                    executable_path="stockfish-test",
                )

                engine_id_2 = db.register_engine(
                    name="Stockfish test",
                    sha256="a" * 64,
                    stockfish_commit="17a6c8f1",
                    executable_path="stockfish-test",
                )

                self.assertEqual(
                    engine_id,
                    engine_id_2,
                )

                candidate_id = db.register_network(
                    role_hint="candidate",
                    name="candidate",
                    sha256="b" * 64,
                    path="candidate.nnue",
                )

                official_id = db.register_network(
                    role_hint="official",
                    name="official",
                    sha256="c" * 64,
                    path="official.nnue",
                )

                fen = (
                    "1rb2rk1/2q1p1bp/3n1pp1/1p1pN3/"
                    "2pPnPP1/2P1P2P/1P1N2B1/R2QBRK1 "
                    "w - - 0 1"
                )

                position_id = db.register_position(
                    fen=fen,
                    source="OLA",
                    split="val",
                    original_score_type="cp",
                    original_score_value=-83,
                    original_wdl_w=1,
                    original_wdl_d=671,
                    original_wdl_l=328,
                )

                run_id = db.create_run(
                    run_tag="TEST_V0",
                    engine_id=engine_id,
                    candidate_network_id=candidate_id,
                    official_network_id=official_id,
                    teacher_network_id=official_id,
                    candidate_nodes=50_000,
                    candidate_multipv=3,
                    teacher_nodes=425_000,
                    threads=1,
                    hash_mb=32,
                    description=(
                        "Temporary DB writer test"
                    ),
                )

                job_id = db.begin_search_job(
                    run_id=run_id,
                    position_id=position_id,
                    engine_role="candidate",
                    engine_id=engine_id,
                    network_id=candidate_id,
                    search_kind="multipv",
                    restricted_move="ALL",
                    requested_nodes=50_000,
                    requested_multipv=3,
                )

                search = SearchResult(
                    infos={
                        1: SearchInfo(
                            rank=1,
                            score_type="cp",
                            score_value=-84,
                            bound="exact",
                            wdl_w=1,
                            wdl_d=661,
                            wdl_l=338,
                            depth=13,
                            seldepth=19,
                            nodes=50012,
                            time_ms=50,
                            nps=1000240,
                            move="e5f3",
                            pv="e5f3 c8b7",
                        )
                    },
                    bestmove="e5f3",
                    ponder="c8b7",
                )

                db.finish_search_job(
                    job_id=job_id,
                    result=search,
                )

                static = StaticEvalResult(
                    status="ok",
                    score_raw_stm=-183,
                    error_text=None,
                )

                db.store_static_eval(
                    run_id=run_id,
                    position_id=position_id,
                    network_role="candidate",
                    network_id=candidate_id,
                    move="ROOT",
                    child_fen=None,
                    result=static,
                    score_root_pov=-183,
                )

            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            con.execute(
                "PRAGMA foreign_keys = ON"
            )

            counts = {}

            for table in (
                "engines",
                "networks",
                "positions",
                "mining_runs",
                "search_jobs",
                "search_results",
                "static_evals",
            ):
                counts[table] = con.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]

            self.assertEqual(counts["engines"], 1)
            self.assertEqual(counts["networks"], 2)
            self.assertEqual(counts["positions"], 1)
            self.assertEqual(counts["mining_runs"], 1)
            self.assertEqual(counts["search_jobs"], 1)
            self.assertEqual(counts["search_results"], 1)
            self.assertEqual(counts["static_evals"], 1)

            job = con.execute("""
                SELECT status, requested_nodes, bestmove
                FROM search_jobs
            """).fetchone()

            evidence = con.execute("""
                SELECT score_value, bound, nodes
                FROM search_results
            """).fetchone()

            static_row = con.execute("""
                SELECT
                    move,
                    score_raw_stm,
                    score_root_pov
                FROM static_evals
            """).fetchone()

            self.assertEqual(job["status"], "ok")
            self.assertEqual(
                job["requested_nodes"],
                50000,
            )
            self.assertEqual(
                job["bestmove"],
                "e5f3",
            )

            self.assertEqual(
                evidence["score_value"],
                -84,
            )
            self.assertEqual(
                evidence["bound"],
                "exact",
            )
            self.assertEqual(
                evidence["nodes"],
                50012,
            )

            self.assertEqual(
                static_row["move"],
                "ROOT",
            )
            self.assertEqual(
                static_row["score_raw_stm"],
                -183,
            )
            self.assertEqual(
                static_row["score_root_pov"],
                -183,
            )

            fk = con.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

            self.assertEqual(fk, [])

            con.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
