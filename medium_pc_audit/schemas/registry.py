"""Minimal, explicit, stdlib-only JSON-Schema-subset validator + registry for the
NNUE domain-adaptation research evidence plane.

Design constraint (locked): Phase 0 must not depend on the `jsonschema` PyPI package,
so this module implements exactly the subset of JSON Schema vocabulary the six v1
schemas actually use. That subset is declared in SUPPORTED_KEYWORDS / SUPPORTED_FORMATS
below and is *tested* (see tests/test_schemas.py ::
NoSchemaUsesUnsupportedKeywordOrFormat) so the .schema.json files and this validator
can never silently drift apart.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SCHEMA_DIR = Path(__file__).parent / "v1"

SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "required",
        "properties",
        "additionalProperties",
        "enum",
        "items",
        "pattern",
        "minimum",
        "minLength",
        "format",
    }
)

SUPPORTED_FORMATS = frozenset({"sha256-hex", "utc-iso8601", "relative-neutral-path"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_utc_iso8601(value: Any) -> bool:
    """Accepts exactly 'YYYY-MM-DDTHH:MM:SS[.ffffff]Z' -- a single canonical UTC spelling.

    Naive timestamps and any other explicit offset (e.g. '+00:00') are rejected on
    purpose: this format must reject non-UTC timestamps rather than merely accepting
    an arbitrary parseable string.
    """
    if not isinstance(value, str) or not _UTC_ISO8601_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _is_relative_neutral_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if "\\" in value:
        return False
    if value.startswith("/"):
        return False
    if _WINDOWS_ABS_RE.match(value):
        return False
    if value.startswith("//"):
        return False
    segments = value.split("/")
    if any(segment in ("", "..") for segment in segments):
        return False
    return True


_FORMAT_CHECKERS = {
    "sha256-hex": _is_sha256_hex,
    "utc-iso8601": _is_utc_iso8601,
    "relative-neutral-path": _is_relative_neutral_path,
}


class SchemaValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class UnknownSchemaError(LookupError):
    pass


def _check_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise ValueError(f"Unsupported schema 'type' value: {expected!r}")


def _matches_type(instance: Any, type_field: Any) -> bool:
    candidates = type_field if isinstance(type_field, list) else [type_field]
    return any(_check_type(instance, t) for t in candidates)


def _validate_node(instance: Any, schema: dict, path: str, errors: list) -> None:
    if "type" in schema:
        if not _matches_type(instance, schema["type"]):
            errors.append(f"{path}: expected type {schema['type']!r}, got {type(instance).__name__}")
            return

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in enum {schema['enum']!r}")

    if isinstance(instance, dict) and ("properties" in schema or schema.get("type") == "object"):
        required = schema.get("required", [])
        for req in required:
            if req not in instance:
                errors.append(f"{path}: missing required field '{req}'")

        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate_node(value, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected additional property '{key}'")

    if isinstance(instance, list) and ("items" in schema or schema.get("type") == "array"):
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                _validate_node(item, item_schema, f"{path}[{index}]", errors)

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: length {len(instance)} < minLength {schema['minLength']}")
        if "pattern" in schema and re.fullmatch(schema["pattern"], instance) is None:
            errors.append(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")
        if "format" in schema:
            fmt = schema["format"]
            checker = _FORMAT_CHECKERS.get(fmt)
            if checker is None:
                errors.append(f"{path}: schema references unsupported format {fmt!r} (validator/schema drift)")
            elif not checker(instance):
                errors.append(f"{path}: {instance!r} does not satisfy format {fmt!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")


def validate(doc_type: str, version: str, instance: dict) -> None:
    """Validate instance against the named schema. Raises SchemaValidationError with all errors found."""
    schema = get_schema(doc_type, version)
    errors: list = []
    _validate_node(instance, schema, "$", errors)
    if errors:
        raise SchemaValidationError(errors)


_SCHEMA_FILENAMES = {
    "artifact_manifest": "artifact_manifest.schema.json",
    "match_config": "match_config.schema.json",
    "execution_manifest": "execution_manifest.schema.json",
    "raw_result": "raw_result.schema.json",
    "analysis_result": "analysis_result.schema.json",
    "ledger_record": "ledger_record.schema.json",
}

_schema_cache: dict = {}


def get_schema(doc_type: str, version: str) -> dict:
    if version != "v1":
        raise UnknownSchemaError(
            f"Unknown schema version {version!r} for doc_type {doc_type!r}; only 'v1' exists."
        )
    if doc_type not in _SCHEMA_FILENAMES:
        raise UnknownSchemaError(
            f"Unknown doc_type {doc_type!r}; known types: {sorted(_SCHEMA_FILENAMES)}"
        )

    cache_key = (doc_type, version)
    if cache_key not in _schema_cache:
        path = _SCHEMA_DIR / _SCHEMA_FILENAMES[doc_type]
        with path.open("r", encoding="utf-8") as f:
            _schema_cache[cache_key] = json.load(f)
    return _schema_cache[cache_key]


def all_schema_paths() -> list:
    return [_SCHEMA_DIR / filename for filename in _SCHEMA_FILENAMES.values()]


def collect_keywords_and_formats(schema: Any, keywords: set, formats: set) -> None:
    """Recursively collect every schema keyword and format value used, for drift testing."""
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        keywords.add(key)
        if key == "format" and isinstance(value, str):
            formats.add(value)
        elif key == "properties" and isinstance(value, dict):
            for subschema in value.values():
                collect_keywords_and_formats(subschema, keywords, formats)
        elif key == "items":
            collect_keywords_and_formats(value, keywords, formats)
