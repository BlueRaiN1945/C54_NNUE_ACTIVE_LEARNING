"""End-to-end fixture test for the M0 corpus pipeline.

synthetic self-play PGN -> position_extract -> synthetic rescoring ->
synthetic corpus TSV -> corpus_provenance -> DataSourceSpec.
is_heritage_free_established

No real chess match is run and no BigPC-only binary (a real Stockfish
engine, the compiled tsv_to_binpack) is invoked. This proves that
position_extract.py's output shape and corpus_provenance.py's input contract
genuinely agree end to end -- the producer/consumer split this pipeline
depends on -- using only already-verified local components: this repo's own
code, python-chess, and the frozen seed's recorded hash. Same pattern as
tests/test_end_to_end.py, which does the equivalent for match evidence.
"""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import chess

from medium_pc_audit.corpus_provenance import build_corpus_data_source
from medium_pc_audit.frozen_seed import SEED_SHA256
from medium_pc_audit.position_extract import extract_positions

SELF_PLAY_PGN = (
    '[White "SELFPLAY_A"]\n[Black "SELFPLAY_B"]\n[Result "*"]\n\n'
    "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 *\n"
    "\n"
    '[White "SELFPLAY_A"]\n[Black "SELFPLAY_B"]\n[Result "*"]\n\n'
    "1. d4 d5 2. Nf3 Nf6 3. c4 e6 *\n"
)

CSV_FIELDS = ["fen", "sf_score_type", "sf_score", "sf_bound", "sf_bestmove", "sf_error"]


def synthetic_rescore(fen: str) -> dict:
    """Fabricate a deterministic, engine-free stand-in for step 3.

    Real rescoring must use the pinned Stockfish engine on the execution
    host; this fixture only stands in for that step so the SURROUNDING
    contract (TSV shape required by tsv_to_binpack, hashing, provenance) can
    be exercised without a real engine or BigPC access. The score is a
    deterministic function of the FEN itself (first byte of its own SHA256,
    shifted to a bounded int) -- not random -- so the whole test stays
    reproducible.
    """

    board = chess.Board(fen)
    bestmove = next(iter(board.legal_moves))
    score = hashlib.sha256(fen.encode("utf-8")).digest()[0] - 128

    return {
        "fen": fen,
        "sf_score_type": "cp",
        "sf_score": str(score),
        "sf_bound": "exact",
        "sf_bestmove": bestmove.uci(),
        "sf_error": "",
    }


class M0CorpusEndToEndTests(unittest.TestCase):
    def build_corpus_tsv(self, directory, *, skip_first_n_plies=1):
        positions = extract_positions(
            SELF_PLAY_PGN, skip_first_n_plies=skip_first_n_plies
        )
        self.assertTrue(positions, "fixture PGN must yield at least one position")

        rows = [synthetic_rescore(p.fen) for p in positions]

        tsv_path = Path(directory) / "modern_pool.tsv"
        with tsv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_FIELDS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return tsv_path, positions

    def test_full_pipeline_establishes_heritage_free(self):
        with tempfile.TemporaryDirectory() as td:
            tsv_path, positions = self.build_corpus_tsv(td)
            artifact_sha256 = hashlib.sha256(tsv_path.read_bytes()).hexdigest()

            generation_record = {
                "schema_version": "v1",
                "artifact_sha256": artifact_sha256,
                "seed_sha256": SEED_SHA256,
                "engine_sha256": "1" * 64,
                "cli_sha256": "2" * 64,
                "opening_suite_sha256": "3" * 64,
                "trainer_commit": "9f72946529c4187d3679014036cd22c3be419716",
                "self_play_games": 2,
                "extraction_skip_first_n_plies": 1,
                "position_count": len(positions),
                "created_utc": "2026-09-25T00:00:00Z",
            }

            spec = build_corpus_data_source(
                artifact_path=tsv_path,
                source_id="M0_E2E_FIXTURE",
                generation_record=generation_record,
                justification=(
                    "synthetic end-to-end fixture; no heritage data anywhere "
                    "in the chain"
                ),
            )

            self.assertTrue(spec.is_heritage_free_established)
            self.assertEqual(spec.sha256, artifact_sha256)

    def test_tampering_the_artifact_after_the_record_was_written_fails_closed(self):
        """Proves the producer/consumer split actually catches drift end to
        end, not just that an isolated unit test can construct a mismatch."""

        with tempfile.TemporaryDirectory() as td:
            tsv_path, _positions = self.build_corpus_tsv(td)
            artifact_sha256 = hashlib.sha256(tsv_path.read_bytes()).hexdigest()

            generation_record = {
                "schema_version": "v1",
                "artifact_sha256": artifact_sha256,
                "seed_sha256": SEED_SHA256,
                "opening_suite_sha256": "3" * 64,
                "created_utc": "2026-09-25T00:00:00Z",
            }

            # Simulate the file changing after the producer recorded its hash
            # (e.g. a corrupted transfer).
            with tsv_path.open("ab") as f:
                f.write(b"\n")

            spec = build_corpus_data_source(
                artifact_path=tsv_path,
                source_id="M0_E2E_FIXTURE",
                generation_record=generation_record,
                justification="j",
            )

            self.assertFalse(spec.is_heritage_free_established)

    def test_extraction_and_tsv_row_count_agree(self):
        """Producer (extraction) and consumer (TSV writer) contracts must
        agree on row count -- no silent drop between the two stages."""

        with tempfile.TemporaryDirectory() as td:
            tsv_path, positions = self.build_corpus_tsv(td)

            with tsv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), len(positions))
            self.assertEqual({r["fen"] for r in rows}, {p.fen for p in positions})

    def test_deterministic_end_to_end(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            tsv_1, _ = self.build_corpus_tsv(td1)
            tsv_2, _ = self.build_corpus_tsv(td2)

            self.assertEqual(tsv_1.read_bytes(), tsv_2.read_bytes())


if __name__ == "__main__":
    unittest.main()
