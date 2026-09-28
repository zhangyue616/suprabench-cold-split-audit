# Data sources and transformations

## Upstream resource

The source is the [SupraBench Binding-Affinity-Prediction resource](https://github.com/Tianyi-Billy-Ma/SupraBench). Its code and curated benchmark data are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). SupraBench identifies [SupraBank](https://suprabank.org/) as a source of binding records. These attributions apply to the records and their derived scientific artifacts in this repository; see [THIRD_PARTY.md](../THIRD_PARTY.md).

The original archive is available from the [SupraBench/Binding-Affinity dataset](https://huggingface.co/datasets/SupraBench/Binding-Affinity), as `data/records.parquet`. A further upstream entry is [SupraBench/bap](https://huggingface.co/datasets/SupraBench/bap). The archive used for this analysis has:

| Property | Recorded value |
|---|---|
| File | `records.parquet` |
| Bytes | 96,671,103 |
| SHA-256 | `a4ba8a95d169ad852cda058243e06d321e7dc8e9cc054c8b029fe64c7d4a7c38` |

This large archive is not duplicated here. Download it from the upstream dataset if raw-record reconstruction is needed, then compare its SHA-256 with the recorded value. An upstream file with a different hash is not the frozen source used here. The supplied processed data and predictions are sufficient for the saved-result verification command.

## Analysis population

The source table contains 2,609 records. Requiring a usable parsed structure with at least one heavy atom for both host and guest retains 2,383 records, representing 2,260 unique standardized host–guest pairs, 189 host identities and 1,164 guest identities. Of the 2,383 structure-valid records, 432 are upstream averages of multiple measurements. Record counts, underlying measurement counts and pair counts are different quantities.

The supplied tables preserve the frozen processing used in the analysis: structural filtering, condition bucketing, identity mappings, split construction and feature generation. Saved predictions, null comparisons, aggregations and operational-parent masks are subsequent analysis outputs. They are modifications of the upstream resource and do not imply endorsement by its authors.

Prediction `row_index` values refer to zero-based physical row positions in the structure-valid condition table `data/clean/suprabench_bap_clean_cond.csv`. They must not be interpreted as positions in the original parquet archive or a manifest's `parquet_row` column.

## Scope of supplied material

This repository contains curated numerical and structural records and the scientific artifacts needed by the documented commands. It does not redistribute article full text. Molecular-coordinate archives are unnecessary for saved-prediction scoring; tree training uses the supplied frozen feature array. The original 3D feature preparation used a wall-clock cutoff, so a fresh feature build is not claimed to reproduce that historical array.
