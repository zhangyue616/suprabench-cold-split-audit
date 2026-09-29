# STRUCTURE-VALID SUBSET — definition, verification, excluded records

**Deliverables 2 & 3.** How the full BAP (2,609 pairs) maps to the structure-valid subset (2,383)
used by the baselines, the exact filter, the verification that the reconstruction is exact, and the
transparent excluded-record table.

## The filter rule (exact)

```
keep row  ⇔  RDKit Chem.MolFromSmiles(host_smiles).GetNumHeavyAtoms() >= 1
             AND Chem.MolFromSmiles(guest_smiles).GetNumHeavyAtoms() >= 1
```

Empty / whitespace SMILES produce a **0-atom degenerate Mol** in RDKit (it returns an empty `Mol`,
**not** `None`), so a naive parse-failure check (`MolFromSmiles is None`) reports **0 failures** and
silently keeps them — in the early gates they all collapsed into a single empty-string pseudo-host.
The structure-valid filter removes them.

> **Labelling (per task):** the dropped rows are **"structure-invalid for an RDKit small-molecule
> graph"** — the host/guest cannot be represented as a small-molecule SMILES graph. They are **NOT**
> labelled benchmark errors. Several are real macrocycles whose SMILES simply was not shipped in the
> benchmark; others are genuinely not small molecules (protein, zeolite, rotaxane).

Documented origin: `docs/suprabench_smoke1_2026-06-26.md` §9 ("records.parquet → (drop empty/<1-atom
SMILES) → suprabench_bap_clean.csv"). The build was interactive; `scripts/build_structure_valid_subset.py`
reconstructs it.

## Verification (the reconstruction is exact)

`scripts/build_structure_valid_subset.py` applied the rule to `records.parquet` and compared to the
canonical `suprabench_bap_clean.csv` (`structure_valid_verification.json`):

| check | result |
|---|---|
| full pairs | 2,609 |
| structure-valid (reconstructed) | **2,383** |
| clean CSV rows | 2,383 |
| `valid_count == clean_count` | **true** |
| `valid (host,guest,y) multiset == clean multiset` | **true** |

⇒ the filter rule reproduces the clean-subset **row selection exactly** (CSV byte-serialization not
compared; only the selected rows).

## Counts: full vs subset

| quantity | full BAP | structure-valid subset | Δ |
|---|---|---|---|
| pairs | 2,609 | 2,383 | −226 |
| unique host SMILES | 190 | 189 | −1 (the empty-SMILES pseudo-host) |
| unique guest SMILES | 1,216 | 1,164 | −52 |
| unique host names | 234 | 198 | −36 |
| sources | eupmc 2,392 / cb7_supplement 217 | eupmc 2,167 / cb7_supplement 216 | −225 / −1 |

## Excluded records — 226 total

Full per-record table: `excluded_records.csv` (226 rows). Summaries: `excluded_summary_by_host.csv`,
`excluded_summary_by_class.csv`.

**By reason:**

| reason | n |
|---|---|
| `empty_host_SMILES` | 223 |
| `empty_guest_SMILES` | 3 |
| **total** | **226** |

(The "223" cited in the smoke report is specifically the empty-**host** rows = 8.55% of 2,609; the
full structure-invalid set is 226 = 8.66%, adding 3 empty-**guest** rows whose host is a valid macrocycle.)

**By host class (descriptive — what the excluded host is):**

| host class | n records | n host names |
|---|---|---|
| calixarene (SMILES not shipped) | 121 | — |
| pillararene (SMILES not shipped) | 44 | — |
| functionalized cucurbituril (SMILES not shipped) | 35 | — |
| inorganic zeolite | 18 | Zeolite L3.0 / Y2.55 (FAU) / Y15 (FAU15) |
| mechanically interlocked (rotaxane) | 5 | Rotaxane / Methyl-CB8-Viologen-Rotaxane |
| protein (HSA) | 2 | Human Serum Albumin |
| cyclodextrin (SMILES not shipped) | 1 | β-Cyclodextrin (one empty-guest row) |
| **total** | **226** | (39 distinct host names) |

Most excluded rows are **sulfonated / functionalized calixarenes and pillararenes** that ARE small
molecules but were shipped **without** a SMILES string in the benchmark — a benchmark data-completeness
gap, not a curation error. A minority (zeolite, HSA, rotaxane) are genuinely outside the
small-molecule-graph representation. See per-host detail in `excluded_summary_by_host.csv`.

> Encoding note: a few host names carry mojibake (e.g. `��-Cyclodextrin` for β-Cyclodextrin) from the
> upstream parquet; the excluded table preserves the raw bytes as-is for fidelity.

## Two-track manifest

`full_bap_manifest.csv` lists all 2,609 records with a `track` column
(`structure_valid_subset` | `excluded_structure_invalid`) plus `host_heavy` / `guest_heavy` /
`structure_valid` / `exclude_reason` / `host_class`, so the full track and the modelled subset are
both reconstructable from one file.
