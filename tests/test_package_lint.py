"""Tests for the dry-run execution-package linter (the evidence bridge)."""

import json
import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.package_lint import (
    WOULD_BE_ACCEPTED,
    WOULD_BE_QUARANTINED,
    PackageLintError,
    lint_execution_package,
)
from tests.test_config_gen import ConfigGenTestCase
from tests.test_result_package import create_package, reseal_package


def snapshot(directory):
    """Path -> (size, mtime_ns) for every file, to detect any mutation."""

    return {
        str(path.relative_to(directory)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def codes(report):
    return [f["code"] for f in report["findings"]]


class PackageLintTests(ConfigGenTestCase):
    def config(self):
        return self.build_default(
            sprt={
                "elo0": 0,
                "elo1": 5,
                "alpha": 0.05,
                "beta": 0.05,
                "max_games": 4,
            }
        )

    def test_valid_package_would_be_accepted(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            report = lint_execution_package(package, match_config=cfg)

            self.assertTrue(report["ok"], report["findings"])
            self.assertEqual(report["predicted_intake"], WOULD_BE_ACCEPTED)
            self.assertEqual(report["findings"], [])
            self.assertIsNone(report["stopped_at"])

    def test_linting_writes_nothing(self):
        """The core promise: a dry run leaves no trace, anywhere."""

        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / "run"
            create_package(package, cfg)

            before = snapshot(root)
            lint_execution_package(package, match_config=cfg)
            after = snapshot(root)

            self.assertEqual(before, after)

            # No inbox or state history may be conjured into existence.
            self.assertFalse((root / "_STATE").exists())
            self.assertFalse((root / "RESULTS_INBOX").exists())

    def test_corrupted_checksum_stops_at_transport(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            (package / "games.pgn").write_bytes(b"tampered after sealing\n")

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertEqual(report["predicted_intake"], WOULD_BE_QUARANTINED)
            self.assertEqual(report["stopped_at"], "transport_integrity")
            self.assertIn("transport_integrity", codes(report))

    def test_missing_required_document_is_reported(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            (package / "artifact_binding.json").unlink()
            reseal_package(package)

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertEqual(report["stopped_at"], "transport_integrity")

    def test_missing_checksum_file_is_reported(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            (package / "SHA256SUMS.txt").unlink()

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertIn("transport_integrity", codes(report))

    def test_crlf_document_is_rejected(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            path = package / "raw_result.json"
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            reseal_package(package)

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertEqual(report["stopped_at"], "document_parse")
            self.assertIn("document_unreadable", codes(report))

    def test_dangling_pgn_reference_is_reported(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            path = package / "raw_result.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["pgn_ref"] = "not_shipped.pgn"

            path.write_text(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            reseal_package(package)

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertIn("evidence_reference_missing", codes(report))

    def test_absolute_pgn_reference_is_reported(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)

            path = package / "raw_result.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["pgn_ref"] = "/absolute/games.pgn"

            path.write_text(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            reseal_package(package)

            report = lint_execution_package(package, match_config=cfg)

            self.assertFalse(report["ok"])
            self.assertIn("evidence_reference_invalid", codes(report))

    def test_missing_package_directory_is_reported(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            report = lint_execution_package(
                Path(td) / "nope",
                match_config=cfg,
            )

            self.assertFalse(report["ok"])
            self.assertEqual(report["predicted_intake"], WOULD_BE_QUARANTINED)

    def test_every_finding_carries_an_actionable_fix(self):
        cfg = self.config()

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "run"
            create_package(package, cfg)
            (package / "SHA256SUMS.txt").unlink()

            report = lint_execution_package(package, match_config=cfg)

            self.assertTrue(report["findings"])

            for finding in report["findings"]:
                self.assertTrue(finding["fix"].strip(), finding)
                self.assertIn("severity", finding)

    def test_non_dict_config_rejected(self):
        with self.assertRaises(PackageLintError):
            lint_execution_package("somewhere", match_config="not-a-dict")


if __name__ == "__main__":
    unittest.main()
