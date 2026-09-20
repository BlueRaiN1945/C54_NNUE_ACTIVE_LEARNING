# C54 / OLA NNUE Active Learning Miner

Research tooling for a C54/OLA-specialized NNUE active-learning pipeline built around official Stockfish.

## Current status

Validated V0 mining core.

The V0 pipeline:

1. searches positions with a candidate NNUE using fixed-node MultiPV;
2. verifies candidate decisions with an official Stockfish/NNUE teacher;
3. runs equal-budget constrained `searchmoves` probes;
4. records raw evidence in SQLite;
5. computes conservative V1 metrics/classes;
6. preserves validation data strictly for evaluation only.

The current V0 classes are intentionally conservative:

- `AGREEMENT`
- `MOVE_DISAGREEMENT`
- `TEACHER_OUTSIDE_CANDIDATE_TOPK` as an auxiliary class

More advanced active-learning classes are intentionally deferred until clean
empirical distributions are available.

## Important design rules

- fixed node budgets;
- `Threads=1` for reproducibility;
- hash cleared between independent searches;
- `UCI_ShowWDL=true`;
- Stockfish `eval` is used for static NNUE evaluation, not `go depth 1`;
- child static scores are normalized to root point-of-view;
- final UCI `bestmove` is stored separately from the last complete coherent
  MultiPV snapshot;
- teacher regret uses equal-budget constrained `searchmoves`;
- validation data must never be mined into training data.

## Requirements

- Python 3.12 or compatible Python 3.x;
- SQLite support in Python;
- a Stockfish executable;
- an official/reference NNUE network;
- a candidate NNUE network.

The Python V0 code uses only the Python standard library.

Stockfish binaries and NNUE networks are external artifacts and are not
included in this repository.

## Offline tests

From the repository root:

~~~bash
cd mining_v0
python -m unittest discover -v -p 'test_*.py'
~~~

Without external Stockfish artifacts, the real integration tests are skipped.

## Real integration tests

Configure:

~~~bash
export C54_STOCKFISH=/path/to/stockfish
export C54_OFFICIAL_NET=/path/to/official.nnue
export C54_CANDIDATE_NET=/path/to/candidate.nnue
~~~

Then run:

~~~bash
cd mining_v0
python -m unittest discover -v -p 'test_*.py'
~~~

## Batch miner

Example:

~~~bash
python mining_v0/miner_train_batch.py \
    --stockfish /path/to/stockfish \
    --official-net /path/to/official.nnue \
    --candidate-net /path/to/candidate.nnue \
    --db /path/to/MINING_V1.sqlite \
    --batch /path/to/TRAIN_BATCH.tsv \
    --run-tag MY_RUN \
    --expected-batch-sha <sha256>
~~~

Then compute V1 metrics:

~~~bash
python mining_v0/metrics.py \
    --db /path/to/MINING_V1.sqlite \
    --run-tag MY_RUN \
    --write
~~~

## Repository layout

- `mining_v0/` — V0 miner, parser, SQLite writer, metrics and tests
- `schema/` — SQLite schema
- `docs/` — validation and reproducibility evidence
- `.github/workflows/` — offline CI
- `private_validation/` — local-only validation material; ignored by Git

## Validation

See:

- `docs/VALIDATION.md`
- `docs/REPRODUCIBILITY.md`
- `docs/V0_TRAIN25_001_vs_002_REPORT.txt`
- `docs/VALIDATED_V0_BASELINE.sha256`

## Scope

This repository contains the validated mining/research core. Large engine,
network, database and training artifacts are deliberately kept outside Git.
