# Active-Learning Subsystem Validation Record

Date: 2026-09-20

This record documents validation of the historical V0 active-learning/mining subsystem now incorporated into **Stockfish NNUE Domain Adaptation & Knowledge Transfer**.

The V0 run identifiers are preserved exactly because they are part of the evidence trail. Canonical project terminology is defined in `NOMENCLATURE.md`.

## Parser defect discovered in V0_TRAIN25_001

The original MultiPV parser independently selected the last exact line for each rank. This could mix ranks from different search depths or iterations.

Observed consequences in `V0_TRAIN25_001`:

- 19 candidate snapshots contained mixed depths;
- 1 negative candidate margin;
- 1 candidate UCI bestmove was missing from teacher constrained probes;
- 1 final UCI bestmove / stored rank-1 mismatch was observed.

The final UCI bestmove/rank-1 mismatch itself is not considered an error. The final UCI decision may come from a newer incomplete iteration while the stored MultiPV evidence intentionally comes from the last complete coherent snapshot.

## Corrective change

The parser now prefers the last complete MultiPV snapshot where:

- all expected ranks are present;
- all stored scores are exact;
- all ranks come from the same depth.

Final UCI `bestmove` remains a separate observation.

Teacher probe construction is now:

candidate UCI bestmove  
UNION coherent candidate MultiPV moves  
UNION teacher UCI bestmove

with deterministic ordering and duplicate removal.

## V0_TRAIN25_002 validation result

The exact same frozen 25-position training manifest was rerun.

Raw audit:

- candidate MultiPV jobs: 25
- bad rank sets: 0
- mixed depths: 0
- non-exact snapshots: 0
- negative raw margins: 0
- missing candidate probes: 0
- bestmove/rank-1 mismatches: 6, allowed by design

Metrics:

- rows: 25
- `AGREEMENT`: 14
- `MOVE_DISAGREEMENT`: 11
- negative candidate margins: 0
- metrics version: V1

Database:

- `PRAGMA integrity_check = ok`
- `PRAGMA foreign_key_check = 0 errors`

## Regression hardening

Permanent tests cover:

- coherent MultiPV snapshot selection;
- prevention of cross-depth candidate margins;
- exact-score preference over bound-only lines;
- forwarding of `expected_multipv` through `StockfishSession.search`;
- mandatory inclusion of candidate UCI bestmove in teacher probes;
- probe deduplication;
- explicit subprocess pipe cleanup;
- SQLite writer round-trip/idempotency;
- real Stockfish session/searchmoves behavior;
- real child-FEN/sign normalization.

## Portable release validation

Offline environment:

- 12 tests discovered;
- 10 passed;
- 2 real-engine integration tests skipped as designed.

BigPC with validated external artifacts:

- 12 tests discovered;
- 12 passed;
- 0 skipped.

Portable CLI smoke test:

- one frozen training position mined into a temporary SQLite database;
- metrics V1 written successfully;
- integrity check: ok;
- foreign-key errors: 0;
- runs: 1;
- metrics rows: 1.

## Interpretation

This record validates the V0 mining/evidence subsystem. It does not by itself establish playing-strength improvement, generalization, or successful domain transfer. Those claims require the separate training and match-evaluation protocols of the broader research program.
