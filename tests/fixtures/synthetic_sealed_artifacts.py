"""Synthetic sealed-artifact builders for Phase 2 config_gen tests.

Uses Phase 1's real artifact_import.seal() with fake content -- never real
BigPC/Stockfish/NNUE data.
"""

from __future__ import annotations

from pathlib import Path

from medium_pc_audit import artifact_import

FIXED_NOW = "2026-09-22T12:00:00Z"


def seal_synthetic_artifact(
    base: Path,
    *,
    artifact_type: str,
    artifact_id: str,
    dest_root: Path,
    content: bytes = b"synthetic fixture content, not real BigPC data",
) -> Path:
    source = base / f"source_{artifact_id}"
    source.mkdir(parents=True, exist_ok=True)
    (source / "data.bin").write_bytes(content)

    artifact_import.seal(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        source_dir=source,
        dest_root=dest_root,
        source_description="synthetic fixture for Phase 2 tests",
        imported_by="tester",
        now_utc=FIXED_NOW,
    )
    return dest_root / artifact_type / artifact_id
