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

## Architecture and execution boundary

The repository separates control/evidence logic from engine-facing execution.

- `medium_pc_audit/` contains deterministic experiment contracts, provenance,
  validation, artifact handling, statistical analysis, and evidence tooling.
- `execution_host/` contains scripts intended for explicit execution-host use
  when an experiment requires an engine subprocess or other host-observed work.
- `medium_pc_audit/orchestration.py` describes authorized work as data; it does
  not execute engines, training, matches, or network operations itself.
- Large engines, networks, corpora, checkpoints, databases, and match outputs
  remain external artifacts and are bound to research records through hashes
  and provenance evidence.

This separation keeps planning, authorization, execution, and evidence
verification independently inspectable.

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

## Historical V0 subsystem

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

For the repository's offline control, provenance, analysis, and test framework:

- Python 3.12 (the CI reference interpreter);
- `chess==1.11.2` for PGN parsing, deterministic position extraction, and
  clean-room opening-artifact generation;
- SQLite support from Python's standard library.

The exact `chess` package version and source-artifact SHA-256 are recorded in
`medium_pc_audit/third_party_deps/chess.lock.json`. The historical
`mining_v0/` subsystem itself remains standard-library-only.

Real engine, mining, training, or execution-host runs additionally require the
relevant pinned Stockfish executable, NNUE networks, opening material, and
protocol-specific external tools. Those large artifacts are intentionally not
stored in this repository.

## Test suites

Install the pinned Python dependency:

~~~bash
python -m pip install chess==1.11.2
~~~

Run the modern research/control suite from the repository root:

~~~bash
python -m unittest discover -s tests -p 'test_*.py' -v
~~~

Run the historical V0 suite separately:

~~~bash
cd mining_v0
python -m unittest discover -v -p 'test_*.py'
~~~

Real-engine V0 integration tests skip automatically when their external
Stockfish/network artifacts are not configured. GitHub CI runs both offline
test suites.

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

- `medium_pc_audit/` - experiment contracts, provenance, artifact handling,
  preflight gates, splitting, match analysis, result ingestion, and research
  evidence utilities
- `execution_host/` - explicitly invoked engine-facing execution-host scripts
- `tests/` - modern research/control test suite
- `mining_v0/` - validated historical active-learning/mining subsystem
- `schema/` - historical V0 mining/evidence database schema
- `docs/` - methodology, nomenclature, validation, reproducibility, and
  evidence records
- `.github/workflows/` - continuous-integration configuration
- `private_validation/` - local-only validation material; ignored by Git

## Validation and evidence

See:

- `docs/MODERN_BASELINE_TRAINING_CONTRACT.md`
- `docs/NOMENCLATURE.md`
- `docs/VALIDATION.md`
- `docs/REPRODUCIBILITY.md`
- `docs/V0_TRAIN25_001_vs_002_REPORT.txt`
- `docs/VALIDATED_V0_BASELINE.sha256`

Historical hash manifests and validation records are retained for provenance;
they are not silently regenerated when later documentation evolves.

## Scope

This repository is the research code and evidence framework for controlled Stockfish NNUE domain adaptation and knowledge transfer. Large engine, network, corpus, match, and training artifacts are intentionally kept outside Git and referenced through reproducible provenance instead.
