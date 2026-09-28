# Reproduction scope

## Saved-result verification

The main command uses frozen inputs and predictions. It does not fit models, construct new splits, or alter operational identity decisions.

| Group | Intended coverage |
|---|---|
| Canonical CPU | File bindings and exact agreement between saved repeat runs for the designated predictions, metrics and pair-similarity outputs |
| Graph models | Reaggregate all 350 saved jobs using their fixed record memberships and metric definitions |
| Tree models | Reaggregate all 420 saved jobs, retaining the saved prediction precision and the executed float32 handling for XGBoost |
| Operational-parent sensitivity | Replay the fixed-prediction analysis from frozen identity decisions, memberships and masks, with separate released-identifier and canonical-key references |

The software checks the pinned verification environment before these groups. Reports distinguish the environment precondition from the four scientific groups. Passing these checks establishes consistency of the supplied artifacts within this scope; it does not independently validate every analysis or regenerate every figure in the associated study.

Family diagnostics are supplied as descriptive saved results. They cover 184 model–seed–family records: 104 fixed-seed tree/null evaluations and 80 graph-model evaluations (two architectures, five seeds, eight families). The public verifier does not recompute these family diagnostics. Runs sharing one family holdout are not independent experimental replications.

## Scoring conventions

Scored record populations, identifier schemes and aggregation rules are part of each result's definition. A pooled held-out score and an equal-block double-cold margin answer different questions and must not be substituted for one another. Null comparisons use the same scored populations as the corresponding model comparison.

Full-precision reference comparisons use an eight-ULP rule where specified. Tree R² reaggregation retains its historical strict absolute tolerance of `1e-12`. Exact bindings, identifiers, statuses and memberships are checked exactly. Rounded producer references retain their original precision and are not promoted to full-precision evidence.

Undefined scores remain undefined. The operational-parent replay retains its guards for fewer than two observations, constant targets, near-constant predictions (standard deviation at most `1e-9`, population convention) and nonfinite values. It does not replace undefined scores by zero. Historical target and prediction dtypes are retained when they affect scoring.

## Operational-parent sensitivity

The analysis scores retained test appearances with already fitted predictions. The frozen representation rules and group decisions are operational definitions, not adjudications of experimental chemical equivalence. The primary scenario retains strict identities for uncertain groups; alternative scenarios use their separately frozen masks.

The released-identifier and canonical-key evaluations have different memberships and references. Test appearances, unique physical rows and distinct identities are different counting units. An appearance can be retained in one layer or scenario and removed in another. The results do not estimate performance after parent-disjoint retraining, and alternative scenarios are not statistical upper and lower bounds.

## New training

Training commands are separate from saved-result verification and use fresh output directories under `runs/`. They retain the frozen jobs, memberships and settings. They are protocol-compatible training entrypoints, not a guarantee of bitwise recovery of historical models.

The original graph preparation refers to an unavailable pilot preparation-script hash; one historical job used that pilot runner. GPU model, deterministic algorithms and TF32 were not all fixed historically. The frozen tree feature array is supplied because its original 3D feature preparation used a wall-clock cutoff. The public training commands do not regenerate that feature array.

Use the README commands and the corresponding pinned requirements. Full graph training requires a compatible CUDA environment. A full suite is substantially more expensive than saved-result verification; start with one complete specified job when testing an installation.
