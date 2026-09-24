# Stockfish NNUE Domain Adaptation & Knowledge Transfer

A reproducible research framework for controlled domain adaptation and knowledge transfer in modern Stockfish NNUEs, with explicit preservation of general playing strength.

## Research objective

This project studies whether domain-specific positional knowledge derived from historical chess corpora and legacy evaluation systems can be transferred into a modern Stockfish-compatible NNUE without degrading general playing strength.

The first research domain is the **Italian Game / Giuoco Piano** family. The project is deliberately structured so that the experimental method is not tied to a single ECO code, historical dataset label, or legacy network.

The modern Stockfish NNUE remains the student model and deployment target. Historical datasets and earlier evaluation systems are treated as **knowledge sources**, not as architectures to be revived or continued.

## Experimental design

The research program separates a modern control from domain-adaptation arms while holding the starting network, compute budget, optimizer configuration, training schedule, and evaluation protocol fixed wherever methodologically possible.

### Modern Baseline

The control arm uses a modern, independently generated reference corpus with no explicit legacy-domain signal.

Historical internal identifier: `M0`.

### Italian Domain — Modern Supervision

Historical Italian/Piano positions may define the position distribution, but supervision comes from the pinned modern Stockfish teacher.

Historical internal identifier: `C54-0`.

### Italian Domain — Conservative Transfer

A matched domain-adaptation arm with a deliberately limited legacy-knowledge signal.

Historical internal identifier: `C54-1`.

### Italian Domain — Enhanced Transfer

A matched domain-adaptation arm with a stronger legacy-knowledge signal.

Historical internal identifier: `C54-2`.

The purpose of the arm structure is to isolate the effect of domain transfer rather than confounding it with changes in seed, compute, optimizer, or training duration.

## Evaluation model

Playing strength is evaluated on two separate axes:

- **General Strength Evaluation** — broad playing strength outside the target domain.
- **Domain-Specific Strength Evaluation** — playing strength within the Italian / Giuoco Piano positional domain.

Validation or training loss is diagnostic only and is never treated as evidence of playing strength.

Match methodology is designed around reproducible engine builds, pinned opening material, paired color-reversed games, explicit execution evidence, and statistical treatment appropriate to paired chess-engine results.

## Provenance and reproducibility

Scientific conclusions are tied to observed artifacts rather than filenames or informal claims.

Relevant engines, networks, datasets, opening suites, generated corpora, checkpoints, match outputs, and conversion tools are identified by cryptographic hashes and version information where available.

The project distinguishes between:

- **declared properties** — intended or asserted properties of an artifact; and
- **evidence-established properties** — properties supported by observed hashes, generation records, provenance records, or other reproducible evidence.

Where evidence is incomplete, the repository records the uncertainty rather than silently upgrading a declaration into a fact.

## Historical terminology

Earlier work used compact internal identifiers including `C54`, `OLA`, `GEN2`, `V80`, `M0`, and `UHO`. They are retained where required for artifact identity, compatibility, and provenance, but they are no longer the primary human-facing vocabulary.

See [docs/NOMENCLATURE.md](docs/NOMENCLATURE.md) for the canonical terminology and legacy-ID mapping.

In particular:

- `C54` refers to the historical Italian / Giuoco Piano research domain.
- `OLA` is retained strictly as a historical source identifier; its original acronym expansion is not treated as established.
- `GEN2` identifies a historical generated-position source.
- `V80` identifies a legacy evaluation/network lineage.
- `M0` identifies the Modern Baseline control.
- `UHO` is retained where used by opening-test tooling and historical records.

## Current published subsystem

The repository originated as a validated active-learning/mining subsystem. That subsystem remains part of the wider research framework and currently provides:

1. fixed-node candidate MultiPV search;
2. verification against a pinned Stockfish/NNUE teacher;
3. equal-budget constrained `searchmoves` probes;
4. raw evidence capture in SQLite;
5. conservative derived metrics and position classes;
6. strict isolation of validation material from active-learning training data.

The existing V0 classes are intentionally conservative:

- `AGREEMENT`
- `MOVE_DISAGREEMENT`
- `TEACHER_OUTSIDE_CANDIDATE_TOPK` as an auxiliary class

Historical V0 identifiers remain unchanged so that validation reports, hashes, and older evidence remain traceable.

## Important design rules

- fixed node budgets for comparable searches;
- `Threads=1` where deterministic execution is required;
- hash cleared between independent searches where specified by protocol;
- `UCI_ShowWDL=true` for recorded WDL evidence;
- Stockfish `eval` for static NNUE evaluation rather than `go depth 1`;
- child static scores normalized to root point-of-view;
- final UCI `bestmove` stored separately from the last complete coherent MultiPV snapshot;
- teacher regret computed using equal-budget constrained `searchmoves`;
- validation and holdout material must never be mined into training data.

## Requirements

For the published V0 mining subsystem:

- Python 3.12 or compatible Python 3.x;
- SQLite support in Python;
- a compatible Stockfish executable;
- an official/reference NNUE network;
- a candidate NNUE network.

The published V0 Python code uses only the Python standard library.

Stockfish binaries, NNUE networks, large corpora, databases, and training checkpoints are external research artifacts and are not included in this repository.

## Offline tests

From the repository root:

~~~bash
cd mining_v0
python -m unittest discover -v -p 'test_*.py'
~~~

Without external Stockfish artifacts, real-engine integration tests are skipped.

## Real integration tests

The historical V0 environment-variable interface remains supported:

~~~bash
export C54_STOCKFISH=/path/to/stockfish
export C54_OFFICIAL_NET=/path/to/official.nnue
export C54_CANDIDATE_NET=/path/to/candidate.nnue
~~~

These names are retained for compatibility and provenance; new documentation uses the canonical terminology defined in `docs/NOMENCLATURE.md`.

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

- `mining_v0/` — validated active-learning miner, parser, SQLite writer, metrics, and tests
- `schema/` — evidence-database schema
- `docs/` — terminology, validation, reproducibility, and historical evidence
- `.github/workflows/` — offline CI
- `private_validation/` — local-only validation material; ignored by Git

## Validation and evidence

See:

- `docs/NOMENCLATURE.md`
- `docs/VALIDATION.md`
- `docs/REPRODUCIBILITY.md`
- `docs/V0_TRAIN25_001_vs_002_REPORT.txt`
- `docs/VALIDATED_V0_BASELINE.sha256`

## Scope

This repository is the research code and evidence framework for controlled Stockfish NNUE domain adaptation and knowledge transfer. Large engine, network, corpus, match, and training artifacts are intentionally kept outside Git and referenced through reproducible provenance instead.
