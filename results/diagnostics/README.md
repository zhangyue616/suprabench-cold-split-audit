# Saved diagnostic results

This directory contains the saved scientific outputs of five diagnostic analyses. The JSON files were exported from the archived result files by an explicit top-level key whitelist. Keys named `_meta` were removed recursively, and the public scripts omit the historical metadata-writing statements, including the non-evidentiary noise-floor placeholder. No model was refit and no diagnostic analysis was rerun during this export.

| Saved result | Public script | Historical input | Historical runtime note |
|---|---|---|---|
| `suprabench_sim_null_results.json` | `../../scripts/diagnostics/suprabench_sim_null.py` | `../../data/clean/suprabench_bap_clean.csv`; expected SHA-256 `c9301b153733e847902ac7f0535802bc00aaa0bbe962673f53bc9a99eaaa1f28` | Attribution to scikit-learn 1.9.0 is inferred from companion evidence because no per-run environment manifest survives; the source retains its fold-consistency gate. |
| `suprabench_pregate_results.json` | `../../scripts/diagnostics/suprabench_pregate.py` | Base clean CSV above plus `../../data/clean/suprabench_bap_clean_cond.csv`; condition-file SHA-256 `56af035c649ca5772675de608772918482ca7f74f12a91efcc50e44fc80ee900` | CPU-only historical run under scikit-learn 1.7.2. |
| `suprabench_cond_audit_results.json` | `../../scripts/diagnostics/suprabench_cond_audit.py` | The same base and condition-annotated clean CSVs | CPU-only historical run under scikit-learn 1.7.2; the script retains its stated fold-consistency tolerance. |
| `suprabench_marginal_killgate_results.json` | `../../scripts/diagnostics/suprabench_marginal_killgate.py` | Upstream `records.parquet`, SHA-256 `a4ba8a95d169ad852cda058243e06d321e7dc8e9cc054c8b029fe64c7d4a7c38` | CPU-only RF/descriptor and marginal-null analysis. |
| `suprabench_killgate_results.json` | `../../scripts/diagnostics/suprabench_killgate.py` | The same upstream `records.parquet` | CPU-only RF/descriptor leakage diagnostic. |

The upstream `records.parquet` is not distributed in this repository. The marginal-killgate and killgate scripts therefore require an explicit `--records PATH`. The other three scripts resolve their default CSV inputs relative to the repository root. All five scripts accept `--output PATH`; their default output is this directory.

The exported top-level scientific keys are:

- `suprabench_sim_null_results.json`: `fold_consistency`, `random`, `host_cold`, `guest_cold`, `double_cold`, `stratified_double_cold`, `gnn_double_cold_seedvar`.
- `suprabench_pregate_results.json`: `fold_consistency`, `P1_block_double_cold`, `P1_lowsim_bin`, `P2_lowsim_composition`, `P2_lowsim_per_family`, `P3_family_cold`, `P3b_struct_cluster`.
- `suprabench_cond_audit_results.json`: `fold_consistency`, `P1`, `P2`, `P3`, `P4_condition_null_double_cold`, `P4_within_solvent`.
- `suprabench_marginal_killgate_results.json`: `random`, `host_cold`, `guest_cold`, `double_cold`, `family_cold`, `similarity`.
- `suprabench_killgate_results.json`: `sha256_records`, `n_rows`, `counts`, `dup_pairs`, `parse`, `n_usable_both_parse`, `leakage`, `baselines`.

These saved outputs preserve the historical results. Running the public scripts performs the analyses again and may fit the models defined in each script; rerunning is separate from inspecting the saved JSON files.
