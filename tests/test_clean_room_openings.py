"""Tests for the clean-room, first-principles M0 opening artifact generator."""

import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.clean_room_openings import (
    ARTIFACT_FILENAME,
    ENUMERATION_PLIES,
    NAMESPACE,
    SELECTION_COUNT,
    CleanRoomOpeningsError,
    build_m0_openings,
    build_provenance_record,
    enumerate_leaf_fens,
    render_epd_text,
    write_artifact,
)
from medium_pc_audit.opening_suite import load_suite_from_text
from medium_pc_audit.split import normalize_fen_key


class EnumerateLeafFensTests(unittest.TestCase):
    def test_depth_1_yields_the_20_legal_first_moves(self):
        leaves = enumerate_leaf_fens(1)
        self.assertEqual(len(leaves), 20)

    def test_depth_2_matches_known_perft(self):
        # perft(2) from the standard starting position is a well-known,
        # independently documented value (20 * 20).
        leaves = enumerate_leaf_fens(2)
        self.assertEqual(len(leaves), 400)

    def test_rejects_non_positive_plies(self):
        with self.assertRaises(CleanRoomOpeningsError):
            enumerate_leaf_fens(0)


class BuildM0OpeningsTests(unittest.TestCase):
    def test_returns_exactly_the_requested_count(self):
        result = build_m0_openings(plies=2, count=50)
        self.assertEqual(len(result), 50)

    def test_result_positions_are_valid_normalized_keys(self):
        result = build_m0_openings(plies=2, count=50)
        for key in result:
            self.assertEqual(normalize_fen_key(key), key)

    def test_deterministic_across_independent_calls(self):
        first = build_m0_openings(plies=2, count=50)
        second = build_m0_openings(plies=2, count=50)
        self.assertEqual(first, second)

    def test_different_namespace_changes_the_selection_or_order(self):
        a = build_m0_openings(namespace="NS_A", plies=2, count=50)
        b = build_m0_openings(namespace="NS_B", plies=2, count=50)
        self.assertNotEqual(a, b)

    def test_no_duplicate_positions_in_result(self):
        result = build_m0_openings(plies=2, count=50)
        self.assertEqual(len(set(result)), len(result))

    def test_insufficient_pool_raises(self):
        # Depth 1 has exactly 20 unique positions; requesting more must fail
        # loudly rather than silently returning a short list.
        with self.assertRaises(CleanRoomOpeningsError):
            build_m0_openings(plies=1, count=21)

    def test_rejects_empty_namespace(self):
        with self.assertRaises(CleanRoomOpeningsError):
            build_m0_openings(namespace="", plies=2, count=10)


class RenderAndWriteTests(unittest.TestCase):
    def test_render_epd_text_one_per_line(self):
        result = build_m0_openings(plies=2, count=10)
        text = render_epd_text(result)
        lines = text.splitlines()
        self.assertEqual(len(lines), 10)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(tuple(lines), result)

    def test_render_epd_text_rejects_empty(self):
        with self.assertRaises(CleanRoomOpeningsError):
            render_epd_text(())

    def test_opening_suite_parser_accepts_the_rendered_format(self):
        """Verifies medium_pc_audit.opening_suite's existing EPD parser
        accepts this module's output with zero duplicates/malformed lines,
        since the positions are already normalized and deduplicated."""

        result = build_m0_openings(plies=2, count=100)
        text = render_epd_text(result)

        suite, report = load_suite_from_text(
            text,
            suite_id="M0_OPENINGS_TEST",
            sha256="0" * 64,
        )

        self.assertEqual(report.total_lines, 100)
        self.assertEqual(report.distinct_openings, 100)
        self.assertEqual(report.duplicate_lines, 0)
        self.assertEqual(report.malformed_lines, 0)
        self.assertEqual(suite.distinct_count, 100)

    def test_write_artifact_round_trips_and_is_reproducible(self):
        result = build_m0_openings(plies=2, count=10)

        with tempfile.TemporaryDirectory() as td:
            path1 = write_artifact(Path(td) / "a.epd", result)
            path2 = write_artifact(Path(td) / "b.epd", result)

            self.assertEqual(path1.read_bytes(), path2.read_bytes())
            self.assertEqual(path1.read_text(encoding="utf-8"), render_epd_text(result))


class BuildProvenanceRecordTests(unittest.TestCase):
    def test_record_contains_required_minimum_fields(self):
        result = build_m0_openings(plies=2, count=10)

        with tempfile.TemporaryDirectory() as td:
            path = write_artifact(Path(td) / ARTIFACT_FILENAME, result)
            record = build_provenance_record(
                artifact_path=path,
                count=10,
                candidate_pool_size=400,
            )

        for field in (
            "artifact_filename",
            "artifact_sha256",
            "artifact_size_bytes",
            "unique_opening_count",
            "candidate_pool_size",
            "recipe",
            "python_version",
            "chess_package_version",
            "namespace",
            "enumeration_depth_plies",
            "dedup_rule",
            "selection_rule",
        ):
            self.assertIn(field, record)

        self.assertEqual(record["artifact_filename"], ARTIFACT_FILENAME)
        self.assertEqual(record["namespace"], NAMESPACE)
        self.assertEqual(len(record["artifact_sha256"]), 64)

    def test_record_hash_matches_independent_recomputation(self):
        import hashlib

        result = build_m0_openings(plies=2, count=10)

        with tempfile.TemporaryDirectory() as td:
            path = write_artifact(Path(td) / ARTIFACT_FILENAME, result)
            record = build_provenance_record(
                artifact_path=path,
                count=10,
                candidate_pool_size=400,
            )
            expected = hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(record["artifact_sha256"], expected)

    def test_missing_artifact_raises(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist.epd"
            with self.assertRaises(CleanRoomOpeningsError):
                build_provenance_record(
                    artifact_path=missing,
                    count=10,
                    candidate_pool_size=400,
                )


class ProductionScaleGenerationTests(unittest.TestCase):
    """The real locked recipe: depth 4, count 20000. Slower (real legal-move
    enumeration from the standard position), run once here to prove the
    production parameters actually work end to end, not just the mechanism
    at toy scale."""

    def test_default_parameters_match_the_locked_recipe(self):
        self.assertEqual(ENUMERATION_PLIES, 4)
        self.assertEqual(SELECTION_COUNT, 20000)
        self.assertEqual(NAMESPACE, "M0_OPENINGS_V1")

    def test_production_recipe_yields_20000_unique_epd_compatible_positions(self):
        result = build_m0_openings()

        self.assertEqual(len(result), 20000)
        self.assertEqual(len(set(result)), 20000)

        text = render_epd_text(result)
        suite, report = load_suite_from_text(
            text,
            suite_id="M0_OPENINGS_V1",
            sha256="0" * 64,
        )

        self.assertEqual(report.distinct_openings, 20000)
        self.assertEqual(report.duplicate_lines, 0)
        self.assertEqual(report.malformed_lines, 0)


if __name__ == "__main__":
    unittest.main()
