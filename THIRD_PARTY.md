# Third-party components

This repository keeps large external engines, neural-network artifacts, and
execution tools outside Git. Version, hash, and provenance information should
be recorded by the relevant research contract or evidence package.

## Stockfish

This project interoperates with Stockfish as an external chess engine.

Stockfish executables and NNUE network files are not included in this
repository. Stockfish is licensed under the GNU General Public License
version 3 (GPLv3).

Users are responsible for obtaining Stockfish and applicable network files
from appropriate sources and complying with their respective licenses.

## chess

The modern research/control suite uses the PyPI package `chess` for PGN
parsing and deterministic legal-move/position extraction.

The repository pins version `1.11.2`. The expected source-distribution
filename and SHA-256 are recorded in
`medium_pc_audit/third_party_deps/chess.lock.json`.

The upstream project licenses python-chess / `chess` under the GNU General
Public License version 3 or later (GPLv3+). The package is not vendored in
this repository; CI obtains the pinned source distribution and verifies its
recorded SHA-256 before installation.

## External research tools

Research protocols and execution plans may also reference tools such as
`c-chess-cli` and `nnue-pytorch`. They are external tools and are not vendored
or redistributed by this repository. Experimental evidence should identify
the exact version, commit, or artifact hash used for a run where applicable.
