"""Synthetic, non-real source-directory builders for Phase 1 artifact_import tests.

Every file created here is fake content -- never real BigPC/Stockfish/NNUE data.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_FILES = {
    "readme.txt": b"synthetic fixture artifact -- not real BigPC data\n",
    "sub/data.bin": bytes(range(256)) * 4,
}


def make_source_dir(base: Path, files=None) -> Path:
    """Create base/source/ populated with the given {relative_path: content} map."""
    if files is None:
        files = DEFAULT_FILES

    source = base / "source"
    source.mkdir(parents=True, exist_ok=True)
    for relpath, content in files.items():
        path = source / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source
