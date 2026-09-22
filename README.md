# c54-medium-pc-audit

Control, analysis, and evidence plane for the C54 NNUE slow audit loop.

## Isolation boundaries (do not violate)

- **BigPC** is the untouched producer of champions and evidence. This repository never
  writes to BigPC, never mounts it, and never syncs with it automatically. All artifact
  transfer in or out is a manual, explicit, immutable package moved by a human operator.
- **Medium-PC** (this repository) is a control/analysis/evidence plane only. It does not
  run chess engines, does not train NNUE networks, and does not install/build
  Stockfish, c-chess-cli, or Docker.
- **Execution host** is a future, still-unchosen, replaceable machine that will actually
  run audit matches. This repository only ever produces configs for it and consumes
  results back from it, both as manual immutable packages.

## Status

Phase 0 (schemas & identities) only. See `medium_pc_audit/identity.py` and
`medium_pc_audit/schemas/` for the current scope. No later phase has been implemented.

## Running tests

Stdlib-only, no external dependencies required:

    python -m unittest -v tests.test_identity tests.test_schemas
