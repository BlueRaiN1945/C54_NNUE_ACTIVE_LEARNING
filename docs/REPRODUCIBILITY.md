# Reproducibility Record

This document preserves the external artifacts and search parameters used for the validated historical V0 active-learning reference run.

The compact V0 and C54-era identifiers are retained where required for exact traceability. Human-facing terminology follows `NOMENCLATURE.md`.

## Validated external artifacts

### Stockfish engine

Commit:

`17a6c8f1eb0da45c2ca405321919519bf4e211ba`

Executable SHA256:

`c9eb516ac9e4ea409ed40958a5c9fdb9fbe9f9c77e8967f2bfd2e536c50b0200`

### Modern reference NNUE

SHA256:

`134a887f4c8ff7bf7284177a3b3fc6ff9cef95ba89eb8db3079a8e507f7126af`

### Historical candidate NNUE used by the V0 reference run

SHA256:

`043bf3ab2e8d71a85925e363eeb591222241811de26109af22bcbf6537e50743`

The role name above describes the artifact's function in this historical run; it does not imply that the candidate is the final architecture or deployment target of the wider domain-adaptation research program.

## V0 search parameters

Candidate search:

- nodes: 50,000
- MultiPV: 3

Teacher unrestricted search:

- nodes: 425,000

Teacher constrained probes:

- nodes: 100,000 per move
- one `searchmoves` move per independent job

Common:

- Threads: 1
- Hash: 32 MB
- UCI_ShowWDL: true

## Frozen TRAIN25 manifest

SHA256:

`26057948bb5eae9de06d768c7209fa7eb4167cb7b0dbc80e0e54a3492113e27f`

Properties at freeze:

- 25 positions
- 25 unique full FENs
- 25 unique normalized first-four-FEN keys
- all 25 from TRAIN
- zero first-four-FEN overlap with VAL

The exact manifest is kept outside the public Git tree until redistribution status and source-data policy are finalized.

## Frozen validation database

`MINING_V1_AFTER_TRAIN25_002.sqlite`

SHA256:

`9d437b75746d67a5e5b8a5e1875fc81fef6f1b05a83dd524cf46ab5cfb9f5d57`

The database itself is intentionally not stored in Git.

## Regression report

`V0_TRAIN25_001_vs_002_REPORT.txt`

SHA256:

`1550594bb8361218288d6a14830247a91017e17e17911294158d8c3b0f75b267`

## Validation-data isolation

Validation is evaluation-only.

The historical `V0_ONE_FEN_001` integration fixture originated from VAL and must never become active-learning training data.

This isolation rule remains part of the broader research methodology: training, validation, holdout, and match-evaluation material must retain explicit roles, and evidence from one role must not be silently recycled into another.
