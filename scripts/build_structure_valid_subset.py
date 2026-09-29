#!/usr/bin/env python
"""SUPRABENCH_REPRO — Deliverables 2/3/4: structure-valid subset definition (+ verification),
excluded-record table, and the full-BAP two-track manifest.

The clean structural subset (suprabench_bap_clean.csv) was originally produced by applying this
deterministic filter to records.parquet; no standalone builder script was retained. This public
script RECONSTRUCTS the filter from records.parquet and VERIFIES that the
reconstructed row selection reproduces the canonical clean CSV exactly (row count + (host,guest,y)
multiset). It does not overwrite source data and writes only to the explicit --output-dir.

FILTER RULE (structure-valid for an RDKit small-molecule graph):
    keep row  <=>  RDKit Chem.MolFromSmiles(host_smiles).GetNumHeavyAtoms() >= 1
                   AND Chem.MolFromSmiles(guest_smiles).GetNumHeavyAtoms() >= 1
Empty / whitespace SMILES yield a 0-atom mol (RDKit returns a degenerate empty Mol, NOT None, so a
naive parse-failure check misses them). These rows are STRUCTURE-INVALID FOR THE RDKIT SMALL-MOLECULE
GRAPH — they are NOT labelled benchmark errors; the host/guest is simply not representable as a
small-molecule SMILES graph (protein, zeolite, rotaxane, or a macrocycle shipped without a SMILES).

Requires pandas with parquet support and RDKit. Sources are read-only; choose a fresh --output-dir.
"""
import argparse, os, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

REPO_ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(description="Rebuild the structure-valid SupraBench subset.")
PARSER.add_argument("--records", type=Path, required=True, help="SHA-bound upstream records.parquet")
PARSER.add_argument("--clean", type=Path, default=REPO_ROOT / "data/clean/suprabench_bap_clean.csv")
PARSER.add_argument("--output-dir", type=Path, required=True, help="fresh output directory")
ARGS = PARSER.parse_args()
PARQ = str(ARGS.records.resolve())
CLEAN = str(ARGS.clean.resolve())
OUT = str(ARGS.output_dir.resolve())
os.makedirs(OUT, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def heavy(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return -1   # NaN
    s = str(s).strip()
    if s == '': return 0                                               # empty
    m = Chem.MolFromSmiles(s)
    if m is None: return -2                                            # unparseable
    return m.GetNumHeavyAtoms()

def host_class(name):
    n = str(name).lower()
    if 'serum albumin' in n or n.strip() == 'hsa': return 'protein (HSA)'
    if 'zeolite' in n: return 'inorganic zeolite'
    if 'rotaxane' in n: return 'mechanically interlocked (rotaxane)'
    if 'pillar' in n: return 'pillararene (SMILES not shipped)'
    if 'calix' in n: return 'calixarene (SMILES not shipped)'
    if 'cucurbit' in n or 'cb[' in n or 'cb6' in n or 'cb8' in n: return 'functionalized cucurbituril (SMILES not shipped)'
    if 'cyclodext' in n: return 'cyclodextrin (SMILES not shipped)'
    return 'other (SMILES not shipped)'

rec = pd.read_parquet(PARQ).reset_index(drop=True)
rec['host_heavy'] = rec['host_smiles'].map(heavy)
rec['guest_heavy'] = rec['guest_smiles'].map(heavy)
rec['structure_valid'] = (rec['host_heavy'] >= 1) & (rec['guest_heavy'] >= 1)

def reason(r):
    if r['host_heavy'] == 0:  return 'empty_host_SMILES'
    if r['host_heavy'] == -1: return 'nan_host_SMILES'
    if r['host_heavy'] == -2: return 'unparseable_host_SMILES'
    if r['guest_heavy'] == 0:  return 'empty_guest_SMILES'
    if r['guest_heavy'] == -1: return 'nan_guest_SMILES'
    if r['guest_heavy'] == -2: return 'unparseable_guest_SMILES'
    return ''
rec['exclude_reason'] = rec.apply(reason, axis=1)
rec['host_class'] = np.where(rec['structure_valid'], '', rec['host_name'].map(host_class))

# ---------- Deliverable 4: full-BAP two-track manifest (all 2,609, incl. excluded) ----------
keep_cols = ['task_id', 'host_name', 'host_smiles', 'guest_name', 'guest_smiles',
             'binding_affinity', 'solvent', 'temperature', 'ph', 'source',
             'host_3d_basename', 'guest_3d_basename',
             'host_heavy', 'guest_heavy', 'structure_valid', 'exclude_reason', 'host_class']
full = rec[keep_cols].copy()
full.insert(0, 'parquet_row', np.arange(len(full)))
full['track'] = np.where(full['structure_valid'], 'structure_valid_subset', 'excluded_structure_invalid')
full.to_csv(os.path.join(OUT, "full_bap_manifest.csv"), index=False, encoding='utf-8')

# ---------- Deliverable 3: excluded-record table (the 226 structure-invalid rows) ----------
excl = full[~full['structure_valid']].copy()
excl_out = excl[['parquet_row', 'task_id', 'host_name', 'host_class', 'host_smiles',
                 'guest_name', 'guest_smiles', 'binding_affinity', 'source', 'solvent',
                 'temperature', 'ph', 'exclude_reason']]
excl_out.to_csv(os.path.join(OUT, "excluded_records.csv"), index=False, encoding='utf-8')

# excluded summary: by reason, by host_class, by host_name
summ_reason = excl.groupby('exclude_reason').size().rename('n_records').reset_index()
summ_class = excl.groupby('host_class').agg(n_records=('task_id', 'size'),
                                            n_host_names=('host_name', 'nunique')).reset_index()
summ_name = (excl.groupby(['host_name', 'host_class', 'exclude_reason']).size()
             .rename('n_records').reset_index().sort_values('n_records', ascending=False))
summ_name.to_csv(os.path.join(OUT, "excluded_summary_by_host.csv"), index=False, encoding='utf-8')
summ_class.to_csv(os.path.join(OUT, "excluded_summary_by_class.csv"), index=False, encoding='utf-8')

# ---------- Deliverable 2: verification that the filter reproduces the clean CSV exactly ----------
clean = pd.read_csv(CLEAN)
valid = rec[rec['structure_valid']]
def keyset(d, ycol):
    return sorted(zip(d['host_smiles'].astype(str), d['guest_smiles'].astype(str),
                      np.round(d[ycol].astype(float).values, 6)))
multiset_match = keyset(valid, 'binding_affinity') == keyset(clean, 'y')
verification = {
    'parquet': PARQ, 'parquet_sha256': sha256(PARQ),
    'clean_csv': CLEAN, 'clean_csv_sha256': sha256(CLEAN),
    'full_n': int(len(rec)),
    'structure_valid_n': int(rec['structure_valid'].sum()),
    'excluded_n': int((~rec['structure_valid']).sum()),
    'clean_csv_n': int(len(clean)),
    'valid_count_equals_clean_count': bool(int(rec['structure_valid'].sum()) == len(clean)),
    'valid_keyset_equals_clean_keyset': bool(multiset_match),
    'excluded_reason_counts': {k: int(v) for k, v in excl['exclude_reason'].value_counts().items()},
    'excluded_host_class_counts': {k: int(v) for k, v in excl['host_class'].value_counts().items()},
    'full_counts': {'pairs': int(len(rec)), 'host_smiles': int(rec['host_smiles'].nunique()),
                    'guest_smiles': int(rec['guest_smiles'].nunique()),
                    'host_name': int(rec['host_name'].nunique()), 'guest_name': int(rec['guest_name'].nunique()),
                    'source': {k: int(v) for k, v in rec['source'].value_counts().items()}},
    'subset_counts': {'pairs': int(len(clean)), 'host_smiles': int(clean['host_smiles'].nunique()),
                      'guest_smiles': int(clean['guest_smiles'].nunique()),
                      'host_name': int(clean['host_name'].nunique())},
    'filter_rule': 'keep iff RDKit MolFromSmiles(host).GetNumHeavyAtoms()>=1 AND guest>=1; '
                   'empty/whitespace SMILES -> 0-atom degenerate Mol -> dropped. '
                   'Labelled structure-invalid-for-RDKit-small-molecule-graph, NOT benchmark error.',
}
with open(os.path.join(OUT, "structure_valid_verification.json"), 'w') as f:
    json.dump(verification, f, indent=2, ensure_ascii=False)

print(json.dumps({k: verification[k] for k in
                  ['full_n', 'structure_valid_n', 'excluded_n', 'clean_csv_n',
                   'valid_count_equals_clean_count', 'valid_keyset_equals_clean_keyset',
                   'excluded_reason_counts', 'excluded_host_class_counts']}, indent=2, ensure_ascii=False))
print("WROTE repro/data/  (full_bap_manifest.csv, excluded_records.csv, excluded_summary_by_host.csv, "
      "excluded_summary_by_class.csv, structure_valid_verification.json)")
