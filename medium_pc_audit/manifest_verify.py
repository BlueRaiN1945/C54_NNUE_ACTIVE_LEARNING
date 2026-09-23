"""ExecutionManifestV1 verification against MatchConfigV1.

Phase 5A pure provenance verification.  No filesystem mutation occurs here.

Important frozen-v1 limitation:
ExecutionManifestV1 does not contain actual candidate/opponent artifact
hashes.  Therefore those two actual artifact identities cannot be proven
from ExecutionManifestV1 alone.  The verifier reports that limitation
explicitly rather than silently claiming full provenance.
"""

from __future__ import annotations

from medium_pc_audit.schemas import registry


V1_LIMITATIONS = (
    "ExecutionManifestV1 does not carry actual candidate artifact SHA256",
    "ExecutionManifestV1 does not carry actual opponent artifact SHA256",
)


_FIELD_BINDINGS = (
    (
        "run_id",
        lambda config: config["config_id"],
        lambda manifest: manifest["run_id"],
    ),
    (
        "engine.sha256",
        lambda config: config["engine"]["expected_sha256"],
        lambda manifest: manifest["actual_engine_sha256"],
    ),
    (
        "cli.sha256",
        lambda config: config["cli"]["expected_sha256"],
        lambda manifest: manifest["actual_cli_sha256"],
    ),
    (
        "uci_options",
        lambda config: config["uci_options"],
        lambda manifest: manifest["actual_uci_options"],
    ),
    (
        "time_control",
        lambda config: config["time_control"],
        lambda manifest: manifest["actual_time_control"],
    ),
    (
        "threads",
        lambda config: config["threads"],
        lambda manifest: manifest["actual_threads"],
    ),
    (
        "hash_mb",
        lambda config: config["hash_mb"],
        lambda manifest: manifest["actual_hash_mb"],
    ),
    (
        "concurrency",
        lambda config: config["concurrency"],
        lambda manifest: manifest["actual_concurrency"],
    ),
    (
        "seed",
        lambda config: config["seed"],
        lambda manifest: manifest["actual_seed"],
    ),
    (
        "opening_suite.sha256",
        lambda config: config["opening_suite"]["sha256"],
        lambda manifest: manifest["actual_opening_suite_sha256"],
    ),
)


def verify_execution_manifest(
    match_config: dict,
    execution_manifest: dict,
) -> dict:
    """Return structured provenance verification; never hide mismatches."""

    schema_errors = []

    try:
        registry.validate(
            "match_config",
            "v1",
            match_config,
        )
    except Exception as exc:
        schema_errors.append(
            {
                "document": "match_config",
                "error": str(exc),
            }
        )

    try:
        registry.validate(
            "execution_manifest",
            "v1",
            execution_manifest,
        )
    except Exception as exc:
        schema_errors.append(
            {
                "document": "execution_manifest",
                "error": str(exc),
            }
        )

    if schema_errors:
        return {
            "ok": False,
            "schema_errors": schema_errors,
            "mismatches": [],
            "verified_fields": [],
            "limitations": list(V1_LIMITATIONS),
        }

    mismatches = []
    verified_fields = []

    for field, expected_getter, actual_getter in _FIELD_BINDINGS:
        expected = expected_getter(match_config)
        actual = actual_getter(execution_manifest)

        if actual != expected:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                }
            )
        else:
            verified_fields.append(field)

    return {
        "ok": not mismatches,
        "schema_errors": [],
        "mismatches": mismatches,
        "verified_fields": verified_fields,
        "limitations": list(V1_LIMITATIONS),
    }
