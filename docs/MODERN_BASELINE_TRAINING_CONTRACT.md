# Modern Baseline Training Contract

**Status:** LOCKED
**Lock date:** 2026-09-25
**Historical/internal experiment ID:** `M0`

## Purpose

This document defines the production run contract for the **Modern Baseline** control arm of the **Stockfish NNUE Domain Adaptation & Knowledge Transfer** research program.

The Modern Baseline starts from the same pinned modern Stockfish NNUE used by later matched domain-adaptation arms, but receives no explicit legacy-domain training signal.

The frozen starting model's historical origin remains explicitly:

`ACKNOWLEDGED-UNKNOWN`

## Experimental role

The Modern Baseline uses:

- the pinned modern Stockfish starting network;
- a newly generated modern reference corpus;
- the same training budget and schedule intended for matched adaptation arms;
- no C54, OLA, GEN2, V80, or other explicit legacy-domain material as corpus input.

The operational "heritage-free" property applies only to this controlled production chain. It does not resolve historical uncertainty about the frozen starting network.

## Locked training schedule

| Parameter | Value |
| --- | ---: |
| Learning rate | `5e-6` |
| Optimizer | `rangerlite` |
| Batch size | `4096` |
| Epoch size | `65536` |
| Optimizer steps per block | `16` |
| Training blocks | `5` |
| Total optimizer steps | `80` |

These values come from the previously established research plan. Smoke-test defaults are not promoted into the scientific production contract.

## Pinned starting model

Trainable `.pt` SHA256:

`dc41943b473c82e878e6c967a4489dd26fd58619362b85bb27a0fbeccee91c82`

Corresponding official serialized NNUE SHA256:

`134a887f4c8ff7bf7284177a3b3fc6ff9cef95ba89eb8db3079a8e507f7126af`

Historical seed origin:

`ACKNOWLEDGED-UNKNOWN`

## Modern reference corpus generation

**LOCKED, 2026-09-25**

| Parameter | Value |
| --- | ---: |
| Self-play search budget | `50000 nodes` |
| Concurrency | `1` |
| Opening order | `sequential` |
| Random extraction | disabled |

Concurrency `1` is deliberately chosen for the first production baseline to maximize auditability and reproducibility.

The historical active-learning miner is not a self-play generator; it selects positions from an existing pool. The production self-play protocol is therefore new methodology rather than a continuation of the mining configuration.

## Modern teacher rescoring

**LOCKED, 2026-09-25**

| Parameter | Value |
| --- | ---: |
| Rescoring budget | `25000 nodes` |
| Workers | `8` |
| Hash | `32 MB` |

The modern pinned Stockfish engine is the teacher.

## Position extraction

**LOCKED**

`skip_first_n_plies = 0`

No additional random position sampling is introduced.

The trainer's established `DataloaderSkipConfig` remains responsible for its own training-side sampling/skipping semantics.

PGN extraction must preserve deterministic game identity so dataset partitioning can occur at whole-game boundaries.

## Binpack eligibility filter

Rows are eligible only when:

```text
sf_score_type == "cp"
sf_bound == "exact"
sf_error == ""
sf_bestmove not in ("", "(none)", "0000")
```

This mirrors the existing `tsv_to_binpack` requirements.

Generation-smoke observation:

- raw rescored rows: `349`
- retained rows: `335`
- observed retention: `95.99%`

This retention rate is planning evidence only, not a production invariant.

## Target corpus size

Target usable corpus size:

approximately `817,000` positions.

## Dataset partitioning

**LOCKED, 2026-09-25**

| Partition | Ratio |
| --- | ---: |
| Training | `90%` |
| Validation | `5%` |
| Holdout | `5%` |

The split unit is the **whole self-play game**.

Assignment must be deterministic from game identity.

The holdout partition is excluded from training, mining, learning-rate selection, checkpoint selection, and other model-selection decisions.

### Critical limitation

This split contract applies only to the newly generated Modern Baseline corpus, where reliable game identity exists.

It does **not** resolve the historical Italian-domain data leakage blocker.

Reliable game / opening-line / family identity is still unavailable for the legacy material, so a scientifically defensible family-isolated heritage split remains:

`BLOCKED`

No percentage split may be presented as resolving that problem.

## Technical readiness evidence

The pre-production pipeline has been validated end-to-end:

`self-play → extraction → rescoring → binpack → provenance → training → checkpoint → serialization → Stockfish load/search`

First real smoke binpack SHA256:

`cf953e14498b398c934eef5040312d16874c40bb78540d585ecd8e41176ec887`

The quantization-crossing smoke produced a serialized NNUE distinct from the frozen starting network:

`acdb305d4f9051061932cc7e828ac17381a393a39391be90333c4939666f4d9c`

These smoke results establish technical readiness only. They are not evidence of playing-strength improvement.

## Production-scale estimate

Calibration at the locked budgets observed:

- one self-play game at `50000` nodes: `99` plies in approximately `25 s`;
- rescoring `349` positions at `25000` nodes with `8` workers: approximately `16 s`.

Current planning estimate for one full Modern Baseline corpus:

| Quantity | Estimate |
| --- | ---: |
| Usable target positions | ~`817,000` |
| Raw positions required | ~`851,150` |
| Expected games | ~`4,250–8,600` |
| Total self-play search nodes | ~`4.26 × 10^10` |
| Total rescoring nodes | ~`2.13 × 10^10` |
| Self-play wall-clock | ~`59.7 h` |
| Rescoring wall-clock | ~`10.84 h` |
| Total sequential BigPC generation time | ~`70.5 h` |
| Approximate elapsed duration | ~`2.9 days` |
| Estimated storage | ~`300–350 MB` |

The game-count range remains uncertain because only one production-budget self-play calibration game is currently available.

## Matched-arm requirement

Where methodologically applicable, later adaptation arms must preserve:

- the same frozen starting network;
- optimizer;
- learning rate;
- batch size;
- total optimizer-step budget;
- training schedule;
- checkpoint policy;
- evaluation methodology.

The experimental treatment should vary only in the explicitly defined domain-position or legacy-knowledge signal.

## Authorization boundary

This document locks the scientific run contract.

It does **not** by itself authorize:

- production-scale corpus generation;
- full training;
- strength matches;
- modification or destruction of historical research artifacts.

Heavy BigPC execution remains a separate explicit action.
