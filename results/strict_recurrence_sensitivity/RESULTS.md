# Fixed-prediction sensitivity to 37 recurrent test appearances

## Scope

This saved-prediction sensitivity excludes the 37 audited double-cold test appearances by exact `(block, row_index)` key and applies the same frozen mask to descriptor-RF and the six prespecified null families. It does not remove training exposure, alter folds, refit a model, add a confidence interval, or perform a significance test.

The 37 appearances correspond to 31 distinct physical rows across 9 of 12 double-cold blocks. The retained population contains 1,947 appearances and 1,356 unique physical rows. The mask was frozen from the identity recurrence audit before outcome values were loaded.

## Saved result

The baseline replay reproduced all 12 archived block margins at relative tolerance `1e-12` and absolute tolerance zero. The equal-block mean descriptor-RF-minus-best-null margin was `0.09503991115167403` before exclusion and `0.08970178332441779` after exclusion, a change of `-0.005338127827256234`. Positive blocks changed from 11/12 to 10/12. The best-null identity did not change in any block, and no newly undefined score was introduced.

The source-only RF remains undefined in every block under the frozen near-constant guard `std(y_pred, ddof=0) <= 1e-9`; undefined values are not replaced by zero and no block is dropped.

## Files

- [`mask_37_appearances.tsv`](mask_37_appearances.tsv): frozen exact appearance mask.
- [`BASELINE_CHECK.json`](BASELINE_CHECK.json): blockwise replay against archived margins.
- [`model_r2_by_block.tsv`](model_r2_by_block.tsv): seven models across 12 blocks before and after exclusion.
- [`block_comparison.tsv`](block_comparison.tsv): descriptor-RF, best-null and margin comparison by block.
- [`SUMMARY.json`](SUMMARY.json): aggregate saved result and boundary statements.
- [`INPUT_RECORD.json`](INPUT_RECORD.json): public input paths, hashes, versions and counting units.

From the repository root, replay into a fresh directory with:

```bash
SUPRABENCH_RECURRENCE_OUTPUT=runs/strict_recurrence_sensitivity \
  python -B scripts/reaggregate_37_mask.py
```
