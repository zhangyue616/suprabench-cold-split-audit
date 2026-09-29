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

## Additional saved-result audits

The materials below are public scientific addenda. They retain their own input bindings and scope rather than changing the four-group verifier.

### Canonical GNN run and model-versus-null aggregation

The frozen primary run is in `results/canonical_gnn_run_a/`; the verification-only repeat is in `results/canonical_gnn_run_b/`. The primary run contains 12,964 test-appearance predictions and 50 metric rows. Its sanitized provenance records the actual execution environment: Python 3.11.15, Torch 2.9.0+cu128, PyG 2.8.0, CUDA 12.8 and an NVIDIA GeForce RTX 5090. The scientific configuration and extracted training core are in `data/canonical_gnn/scientific_config.json` and `scripts/canonical_gnn/`.

The following command reaggregates the saved primary predictions against the same six canonical CPU nulls. It does not train a model:

```bash
python scripts/reaggregate_canonical_gnn.py \
  --gnn-predictions results/canonical_gnn_run_a/predictions_gnn.csv \
  --gnn-metrics results/canonical_gnn_run_a/metrics_gnn.tsv \
  --cpu-predictions results/canonical_cpu/run_a/predictions_cpu.csv \
  --cpu-metrics results/canonical_cpu/run_a/metrics_cpu.tsv \
  --frozen-targets data/canonical_cpu/features/tabular_features.npz \
  --output-dir runs/canonical_gnn_aggregation \
  --output-stem canonical_gnn_aggregation \
  --blocks-filename canonical_gnn_double_cold_blocks.csv
```

The frozen public results are `results/canonical_gnn_aggregation.json`, `.tsv`, and `results/canonical_gnn_double_cold_blocks.csv`. The equal-block double-cold margins are −0.2757452044 for GINE (2/12 positive blocks), −0.0235904965 for AttentiveFP (6/12), and +0.1198370303 for descriptor-RF (10/12). Appearance-weighted summaries are retained separately and are not substituted for equal-block values.

### Canonical GNN repeat comparison

`results/canonical_gnn_comparison/` retains the prespecified A/B repeat comparison as a 326-row table plus an allow-listed receipt. It reports 326 PASS, zero FAIL and zero symmetric-undefined comparisons under the frozen thresholds. The input predictions, exact historical comparator source and the narrow deprecation-warning compatibility wrapper are retained. This receipt is a repeat comparison; it is distinct from the model-versus-null reaggregation above. The exact historical comparator depends on its original canonical-build layout and is supplied for source audit rather than advertised as a standalone public rerun command.

### Diagnostic results

`results/diagnostics/` contains five saved JSON outputs exported by explicit scientific-key whitelists. Every key named `_meta` was omitted recursively; no value under such a key is used as evidence. The paired scripts under `scripts/diagnostics/` have repository-relative inputs and configurable outputs. Packaging did not execute them or fit their RF/GNN models. The marginal and leakage diagnostics require the upstream `records.parquet` with the SHA recorded in `DATA_SOURCES.md`.

The canonical RF specification is 300 estimators, `random_state=0`, `max_features="sqrt"`, and one job. The modern and diagnostic branches retain the same scientific RF parameters with `n_jobs=-1`. The modern run records scikit-learn 1.9.0. Attribution of the similarity-null run to that environment is inferred from its companion evidence because no per-run environment manifest was retained. Pregate and condition audits used scikit-learn 1.7.2. The saved identity-marginal double-cold fields report all four identity-keyed marginal branches as degenerate constant predictions; the global-mean control is constant by definition.

### Population, P1 and recurrence audits

`scripts/build_structure_valid_subset.py` reconstructs the deterministic 2,609-to-2,383 filter from the SHA-bound upstream parquet into a fresh output directory. The saved manifest, exclusions and verification are under `data/population/`. `results/record_audit/` freezes the row/pair/task-identifier semantics, family residual decomposition and all 66 leave-two-block means.

`protocols/p1_d6sc_overlap/` keeps the outcome-blind Stage-A predicate and labels separate from Stage-B fixed-prediction scoring. `results/strict_recurrence_sensitivity/` and `scripts/reaggregate_37_mask.py` retain the exact 37-appearance mask and the seven-model before/after block scores. The retained-population audit reuses saved predictions across the same 12 blocks; it does not remove training exposure or refit any model.

`scripts/parent_identity_b1/`, `protocols/parent_identity_b1/` and `results/parent_identity_b1/` retain the five v1 identity layers, parent construction, frozen membership masks and Stage-C fixed-prediction results. Private review prompts and historical execution logs are outside the public scientific package.

## Scoring conventions

Scored record populations, identifier schemes and aggregation rules are part of each result's definition. A pooled held-out score and an equal-block double-cold margin answer different questions and must not be substituted for one another. Null comparisons use the same scored populations as the corresponding model comparison.

Full-precision reference comparisons use an eight-ULP rule where specified. Tree R² reaggregation retains its historical strict absolute tolerance of `1e-12`. Exact bindings, identifiers, statuses and memberships are checked exactly. Rounded producer references retain their original precision and are not promoted to full-precision evidence.

Undefined scores remain undefined. The operational-parent replay retains its guards for fewer than two observations, constant targets, near-constant predictions (standard deviation at most `1e-9`, population convention) and nonfinite values. It does not replace undefined scores by zero. Historical target and prediction dtypes are retained when they affect scoring.

## Operational-parent sensitivity

The analysis scores retained test appearances with already fitted predictions. The frozen representation rules and group decisions are operational definitions, not adjudications of experimental chemical equivalence. The primary scenario retains strict identities for uncertain groups; alternative scenarios use their separately frozen masks.

The released-identifier and canonical-key evaluations have different memberships and references. Test appearances, unique physical rows and distinct identities are different counting units. An appearance can be retained in one layer or scenario and removed in another. The results do not estimate performance after parent-disjoint retraining, and alternative scenarios are not statistical upper and lower bounds.

## New training

Training commands are separate from saved-result verification and use fresh output directories under `runs/`. They retain the frozen jobs, memberships and settings. They are protocol-compatible training entrypoints, not a guarantee of bitwise recovery of historical models.

The original released-job graph preparation refers to an unavailable pilot preparation-script hash; one historical job used that pilot runner. GPU model, deterministic algorithms and TF32 were not all fixed for that released-job history. The later canonical v19 run has a separate hard-determinism contract and actual Torch 2.9 runtime record. The frozen tree feature array is supplied because its original 3D feature preparation used a wall-clock cutoff. The public training commands do not regenerate that feature array.

Use the README commands and the corresponding pinned requirements. Full graph training requires a compatible CUDA environment. A full suite is substantially more expensive than saved-result verification; start with one complete specified job when testing an installation.
