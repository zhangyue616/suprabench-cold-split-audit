# Public science coverage

This document states what the repository supplies for each scientific layer and what kind of verification is possible. Paths are relative to the repository root. The 55 historical Supplementary Information pointers are mapped individually in `docs/SI_PATH_CROSSWALK.tsv`.

| Scientific layer | Public materials | Verification scope |
|---|---|---|
| Structure-valid population | `data/population/`, `docs/STRUCTURE_VALID_FILTER.md`, `scripts/build_structure_valid_subset.py` | Frozen 2,609-row manifest, 226 exclusions, 2,383 retained records and deterministic filter. Rebuilding requires the upstream SHA-bound parquet. |
| Raw identifiers and recurrence | `data/folds/`, `data/identity/canonical_identity_recurrence_2026-08-01.tsv` | Released raw host/guest mappings, fold memberships and frozen canonical recurrence audit. |
| Canonical CPU models | `data/canonical_cpu/`, `results/canonical_cpu/` | Exact saved repeat comparison and the public four-group verifier. |
| Released graph jobs | `results/gnn_saved/`, `scripts/reaggregate_saved_results.py`, `scripts/run_gnn_public.py` | All 350 saved job predictions/results can be reaggregated. Per-epoch logs and historical checkpoints are not claimed as public contents. `historical_seed0_comparison.csv` preserves the reported rounded-history comparison. |
| Released tree jobs | `results/tree_saved/`, `scripts/reaggregate_saved_results.py`, `scripts/run_trees_public.py` | All 420 saved job predictions/results can be reaggregated. Frozen 3D features are supplied; their wall-clock-bounded historical preparation is not regenerated. |
| Canonical GNN primary run | `results/canonical_gnn_run_a/`, `data/canonical_gnn/scientific_config.json`, `scripts/canonical_gnn/`, `environments/canonical_gnn.txt` | Frozen run-A predictions/metrics and allow-listed provenance. The environment file is a partial historical runtime record: the retained source records Python, Torch, PyG, CUDA and GPU, but not a complete package freeze. The scientific core and determinism contract are auditable; historical outer lifecycle/review gates are excluded, so this is not advertised as a bitwise end-to-end training driver. |
| Canonical GNN repeat | `results/canonical_gnn_run_b/`, `results/canonical_gnn_comparison/` | Run-B verification input plus 326-row A/B comparison receipt, thresholds and exact historical comparator source. The original comparator remains layout-dependent. |
| Canonical GNN versus nulls | `scripts/reaggregate_canonical_gnn.py`, `results/canonical_gnn_aggregation.*`, `results/canonical_gnn_double_cold_blocks.csv` | Saved-prediction replay of pooled, equal-block and appearance-weighted summaries. No fitting. |
| Similarity and null diagnostics | `scripts/diagnostics/`, `results/diagnostics/` | Saved sim-null, pregate, condition, identity-marginal, full-anchor similarity and leakage outputs. Public JSON uses scientific-key whitelists and contains no `_meta` keys. Rerunning the scripts is a separate model-fitting action. |
| Family diagnostics | `results/family_diagnostics/`, `scripts/analyze_family_diagnostics.py` | Frozen 184 family/model/seed records and path-parameterized producer. Not part of the main saved-result verifier. |
| Record semantics and block arithmetic | `results/record_audit/`, `scripts/record_and_algebra_audit.py`, `scripts/leave_two_blocks.py` | Frozen task-identifier counting, residual decomposition and 66 leave-two-block summaries. No independent-experiment inference. |
| P1 structural neighborhood | `protocols/p1_d6sc_overlap/` | Frozen outcome-blind predicate/labels and separate fixed-prediction Stage-B results. No outcome-driven predicate tuning or retraining. |
| Strict recurrence sensitivity | `results/strict_recurrence_sensitivity/`, `scripts/reaggregate_37_mask.py` | Exact 37-appearance mask, seven-model block scores, baseline check and retained-population summary. Saved predictions only. |
| Parent identity | `scripts/parent_identity_b1/`, `protocols/parent_identity_b1/`, `results/parent_identity_b1/` | Five frozen v1 identity layers, parent construction, memberships and fixed-prediction scoring. Private review records are excluded. |

## Historical execution boundaries

Four dated pointers refer to broader historical execution workspaces rather than missing scientific tables: the canonical fold builder, model-feature builder, fold-export wrapper and formal GNN runtime adapter. The public repository supplies the frozen folds, features, schemas, SHA bindings, scientific GNN core, configuration and saved outputs. It does not claim to reproduce private lifecycle gates, unavailable upstream preparation inputs or internal review logs.

The raw `records.parquet` is not redistributed. Its upstream location, byte count and SHA-256 are in `docs/DATA_SOURCES.md`. The public filter and two full-anchor diagnostic scripts accept that file explicitly.

## Licensing

`FILE_LICENSES.tsv` is the per-file licensing authority. Original analysis software identified there as MIT is licensed under the MIT License. Scientific data, saved predictions, derived numerical results and figures, together with `docs/DATA_SOURCES.md`, `docs/REPRODUCTION.md`, `docs/SCIENCE_COVERAGE.md`, `docs/SI_PATH_CROSSWALK.tsv`, and `docs/STRUCTURE_VALID_FILTER.md`, are CC BY 4.0 with the upstream attribution recorded in `THIRD_PARTY.md`. All other documentation follows the license assigned to that file in `FILE_LICENSES.tsv`. No repository-wide MIT or CC BY 4.0 grant is implied.
