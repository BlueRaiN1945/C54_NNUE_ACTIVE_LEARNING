# Canonical Project Nomenclature

## Project identity

**Formal project name:** Stockfish NNUE Domain Adaptation & Knowledge Transfer

**Repository name:** `stockfish-nnue-domain-transfer`

The project studies controlled specialization of a modern Stockfish NNUE through domain adaptation and knowledge transfer while explicitly measuring preservation of general playing strength.

## Naming principles

Human-facing documentation uses semantic names that describe the scientific role of an object. Historical compact identifiers remain available only where they are required for compatibility, provenance, artifact identity, or direct comparison with earlier records.

A historical identifier must not silently acquire a new meaning. If the original meaning of an identifier is unknown or insufficiently documented, the repository says so.

## Canonical terminology

| Canonical term | Historical/internal identifier | Meaning |
| --- | --- | --- |
| **Modern Baseline** | `M0` | Control arm initialized from the same modern Stockfish NNUE and trained without explicit legacy-domain signal. |
| **Italian / Giuoco Piano Domain** | `C54` | The first domain-specific research family. The ECO code is retained as historical provenance, not as the primary project name. |
| **Italian Domain — Modern Supervision** | `C54-0` | Domain-position arm supervised by the modern Stockfish teacher without an additional legacy-transfer signal. |
| **Italian Domain — Conservative Transfer** | `C54-1` | Matched arm with a limited legacy-knowledge transfer signal. |
| **Italian Domain — Enhanced Transfer** | `C54-2` | Matched arm with a stronger legacy-knowledge transfer signal. |
| **Legacy Curated Position Set** | `OLA` | Historical curated source label. The original acronym expansion is not treated as established by current evidence. |
| **Legacy Generated Position Set** | `GEN2` | Historical generated-position source used in earlier research. |
| **Legacy Evaluation Network** | `V80` | Historical network/evaluation lineage used as a possible source of transferable domain knowledge. |
| **Unbalanced Opening Evaluation Suite** | `UHO` | Intentionally unbalanced opening material used where needed to improve match sensitivity. |
| **General Strength Evaluation** | General Elo | Broad playing-strength evaluation outside the target domain. |
| **Domain-Specific Strength Evaluation** | Heritage Elo | Evaluation targeted at the domain being adapted; currently Italian / Giuoco Piano. |

## Artifact naming

Historical filenames, hashes, environment variables, database fields, run tags, and validation records are not renamed merely for aesthetics. Their original identifiers are part of the evidence chain.

New documents should prefer:

- `MODERN_BASELINE_PROTOCOL.md` over `M0_CONTROL_PROPOSAL.md`;
- `MODERN_BASELINE_TRAINING_CONTRACT.md` over `M0_C54_RUN_CONTRACT.md`;
- `ITALIAN_DOMAIN_TRANSFER_STUDY_A` over `C54_V80_RESEARCH_A`;
- `COMPUTE_ENVIRONMENT_VERIFICATION.md` over machine- or phase-specific audit names where the broader description is accurate;
- `EXECUTION_EVIDENCE_PROTOCOL.md` for execution/provenance packaging contracts.

Existing historical artifacts should be referenced by both their canonical role and exact original filename when reproducibility requires it.

Example:

> **Legacy Italian Position Corpus — Modern Stockfish Rescored Edition**  
> Historical artifact: `OLA_C54_RESCORED_425K_EXACT.tsv`

## Writing convention

Prefer:

> The Italian-domain adaptation arm was trained on legacy curated positions and supervised by the pinned modern Stockfish teacher.

over:

> C54-1 used OLA/GEN2 heritage.

Prefer explicit scientific roles first, and place compact historical identifiers in parentheses only when they add provenance value.
