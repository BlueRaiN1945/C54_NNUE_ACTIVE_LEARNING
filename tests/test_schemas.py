"""Tests for medium_pc_audit.schemas.registry -- Phase 0.

Run with: python -m unittest -v tests.test_schemas
"""

import json
import unittest
from pathlib import Path

from medium_pc_audit.schemas import registry

_FIXTURES = Path(__file__).parent / "fixtures"

_DOC_TYPES = [
    "artifact_manifest",
    "match_config",
    "execution_manifest",
    "raw_result",
    "analysis_result",
    "ledger_record",
]


def _load_fixture(subdir, name):
    path = _FIXTURES / subdir / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class ValidFixturesValidateCleanly(unittest.TestCase):
    def test_all_valid_fixtures_pass(self):
        for doc_type in _DOC_TYPES:
            with self.subTest(doc_type=doc_type):
                instance = _load_fixture("valid", f"{doc_type}_v1_valid.json")
                registry.validate(doc_type, "v1", instance)  # must not raise


class InvalidFixturesFailExplicitly(unittest.TestCase):
    _CASES = [
        ("artifact_manifest", "artifact_manifest_v1_missing_field.json", "sealed"),
        ("match_config", "match_config_v1_bad_sha256.json", "sha256-hex"),
        ("execution_manifest", "execution_manifest_v1_wrong_type.json", "actual_threads"),
        ("raw_result", "raw_result_v1_bad_enum.json", "result"),
        ("analysis_result", "analysis_result_v1_bad_timestamp.json", "utc-iso8601"),
        ("ledger_record", "ledger_record_v1_absolute_path.json", "relative-neutral-path"),
    ]

    def test_each_invalid_fixture_raises_with_useful_message(self):
        for doc_type, filename, expected_substring in self._CASES:
            with self.subTest(doc_type=doc_type, filename=filename):
                instance = _load_fixture("invalid", filename)
                with self.assertRaises(registry.SchemaValidationError) as ctx:
                    registry.validate(doc_type, "v1", instance)
                message = str(ctx.exception)
                self.assertIn(expected_substring, message, message)


class MissingSchemaVersionAlwaysFails(unittest.TestCase):
    def test_missing_schema_version(self):
        for doc_type in _DOC_TYPES:
            with self.subTest(doc_type=doc_type):
                instance = _load_fixture("valid", f"{doc_type}_v1_valid.json")
                del instance["schema_version"]
                with self.assertRaises(registry.SchemaValidationError) as ctx:
                    registry.validate(doc_type, "v1", instance)
                self.assertIn("schema_version", str(ctx.exception))


class RegistryUnknownVersionTests(unittest.TestCase):
    def test_unknown_version_raises_explicit_error(self):
        with self.assertRaises(registry.UnknownSchemaError):
            registry.get_schema("match_config", "v2")

    def test_unknown_doc_type_raises_explicit_error(self):
        with self.assertRaises(registry.UnknownSchemaError):
            registry.get_schema("nonexistent_type", "v1")


class NoSchemaUsesUnsupportedKeywordOrFormat(unittest.TestCase):
    def test_all_schema_files_stay_within_supported_subset(self):
        for path in registry.all_schema_paths():
            with self.subTest(path=str(path)):
                schema = json.loads(path.read_text(encoding="utf-8"))
                keywords = set()
                formats = set()
                registry.collect_keywords_and_formats(schema, keywords, formats)

                unsupported_keywords = keywords - registry.SUPPORTED_KEYWORDS
                unsupported_formats = formats - registry.SUPPORTED_FORMATS

                self.assertFalse(unsupported_keywords, f"{path.name} uses unsupported keywords: {unsupported_keywords}")
                self.assertFalse(unsupported_formats, f"{path.name} uses unsupported formats: {unsupported_formats}")


if __name__ == "__main__":
    unittest.main()
