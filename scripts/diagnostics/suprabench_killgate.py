#!/usr/bin/env python
"""SUPRABENCH_KILLGATE — bounded prior-dominance/leakage diagnostic on SupraBench BAP.
RF/descriptor baselines ONLY (no GNN). Read-only on downloaded public CC-BY data."""
import argparse
import json, hashlib, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

REPO_ROOT=Path(__file__).resolve().parents[2]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True,help='Path to the upstream records.parquet (not distributed in this repository).')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/'diagnostics'/'suprabench_killgate_results.json',help='Output JSON path.')
    return p.parse_args()
ARGS=parse_args(); REC=ARGS.records; OUTPUT=ARGS.output
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

r = pd.read_parquet(REC)
out = {"sha256_records": sha256(REC), "n_rows": int(len(r))}

# ---------- STEP 0: counts ----------
r['y'] = r['binding_affinity'].astype(float)
out["counts"] = {
  "pairs": int(len(r)),
  "unique_host_name": int(r['host_name'].nunique()),
  "unique_host_smiles": int(r['host_smiles'].nunique()),
  "unique_guest_name": int(r['guest_name'].nunique()),
  "unique_guest_smiles": int(r['guest_smiles'].nunique()),
  "source_counts": {k:int(v) for k,v in r['source'].value_counts().items()},
  "logKa": {k:round(float(v),4) for k,v in r['y'].describe().items()},
}

# ---------- duplicate (host,guest) pairs => measurement-noise floor ----------
g = r.groupby(['host_smiles','guest_smiles'])['y']
dup = g.agg(['count','min','max','std']).reset_index()
dup_multi = dup[dup['count']>1]
out["dup_pairs"] = {
  "n_unique_pairs": int(len(dup)),
  "n_pairs_with_replicates": int(len(dup_multi)),
  "n_records_in_replicate_pairs": int(dup_multi['count'].sum()) if len(dup_multi) else 0,
  "median_logKa_range_within_dup": round(float((dup_multi['max']-dup_multi['min']).median()),4) if len(dup_multi) else None,
  "mean_logKa_std_within_dup": round(float(dup_multi['std'].dropna().mean()),4) if len(dup_multi) else None,
}

# ---------- featurization ----------
def mol(s):
    try: return Chem.MolFromSmiles(s)
    except Exception: return None

r['hm'] = r['host_smiles'].map(mol)
r['gm'] = r['guest_smiles'].map(mol)
out["parse"] = {"host_fail": int(r['hm'].isna().sum()), "guest_fail": int(r['gm'].isna().sum())}
d = r[(~r['hm'].isna()) & (~r['gm'].isna())].reset_index(drop=True).copy()
out["n_usable_both_parse"] = int(len(d))

def ecfp(m, n=2048, rad=2):
    try: return np.array(AllChem.GetMorganFingerprintAsBitVect(m, rad, nBits=n), dtype=np.float32)
    except Exception: return np.zeros(n, dtype=np.float32)

DESC = [('MW',Descriptors.MolWt),('LogP',Descriptors.MolLogP),('TPSA',Descriptors.TPSA),
        ('HBD',Descriptors.NumHDonors),('HBA',Descriptors.NumHAcceptors),
        ('RotB',Descriptors.NumRotatableBonds),('Ring',rdMolDescriptors.CalcNumRings),
        ('ArRing',rdMolDescriptors.CalcNumAromaticRings),('FrCSP3',Descriptors.FractionCSP3),
        ('Heavy',lambda m:m.GetNumHeavyAtoms()),('Charge',Chem.GetFormalCharge)]
def desc(m):
    o=[]
    for _,fn in DESC:
        try: o.append(float(fn(m)))
        except Exception: o.append(0.0)
    return np.array(o, dtype=np.float32)

H_ec = np.vstack([ecfp(m) for m in d['hm']]); G_ec = np.vstack([ecfp(m) for m in d['gm']])
H_d  = np.vstack([desc(m) for m in d['hm']]); G_d  = np.vstack([desc(m) for m in d['gm']])
y = d['y'].values.astype(np.float32)

X_ecfp = np.hstack([H_ec, G_ec])                      # baseline (i)
X_desc = np.hstack([H_d, G_d])                        # baseline (ii)
# baseline (iii): physically-motivated host-guest pair descriptors (size/charge/complementarity)
def safediv(a,b): return a/np.where(np.abs(b)<1e-6,1e-6,b)
pair = np.column_stack([
  H_d[:,0]+G_d[:,0], safediv(G_d[:,0],H_d[:,0]),            # MW sum, guest/host MW ratio (size fit)
  H_d[:,9]-G_d[:,9], safediv(G_d[:,9],H_d[:,9]),            # heavy-atom diff, ratio (cavity vs guest size)
  H_d[:,10]*G_d[:,10], H_d[:,10]+G_d[:,10],                 # charge product, charge sum (electrostatics)
  np.abs(H_d[:,2]-G_d[:,2]),                                 # TPSA diff (polarity match)
  H_d[:,3]+G_d[:,4], H_d[:,4]+G_d[:,3],                      # H-bond complementarity (HBD_h+HBA_g, HBA_h+HBD_g)
  H_d[:,1]+G_d[:,1],                                         # LogP sum (hydrophobicity)
])
X_pair = pair.astype(np.float32)

groups_host = pd.factorize(d['host_smiles'])[0]
groups_guest = pd.factorize(d['guest_smiles'])[0]

# ---------- STEP 1: leakage under naive random split ----------
rng = np.random.RandomState(0)
leak = {"random_splits":[]}
for seed in range(5):
    idx = rng.permutation(len(d)); cut=int(0.8*len(d))
    tr, te = set(idx[:cut]), set(idx[cut:])
    htr=set(d['host_smiles'].values[list(tr)]); gtr=set(d['guest_smiles'].values[list(tr)])
    te_l=list(te)
    sh=np.mean([d['host_smiles'].values[i] in htr for i in te_l])
    sg=np.mean([d['guest_smiles'].values[i] in gtr for i in te_l])
    leak["random_splits"].append({"test_frac_host_seen_in_train":round(float(sh),3),
                                  "test_frac_guest_seen_in_train":round(float(sg),3)})
leak["mean_test_frac_host_seen"]=round(float(np.mean([s["test_frac_host_seen_in_train"] for s in leak["random_splits"]])),3)
leak["mean_test_frac_guest_seen"]=round(float(np.mean([s["test_frac_guest_seen_in_train"] for s in leak["random_splits"]])),3)
out["leakage"]=leak

# ---------- STEP 2: baselines x splits ----------
def evaluate(X, splitter, groups=None):
    r2s,sps,maes=[],[],[]
    for tr,te in (splitter.split(X,y,groups) if groups is not None else splitter.split(X)):
        m=RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=0,max_features='sqrt')
        m.fit(X[tr],y[tr]); p=m.predict(X[te])
        r2s.append(r2_score(y[te],p)); sps.append(spearmanr(y[te],p).correlation); maes.append(mean_absolute_error(y[te],p))
    f=lambda a:[round(float(np.mean(a)),3),round(float(np.std(a)),3)]
    return {"R2":f(r2s),"Spearman":f(sps),"MAE":f(maes)}

def eval_mean(splitter, groups=None):
    r2s,sps,maes=[],[],[]
    for tr,te in (splitter.split(np.zeros((len(y),1)),y,groups) if groups is not None else splitter.split(np.zeros((len(y),1)))):
        p=np.full(len(te), y[tr].mean())
        r2s.append(r2_score(y[te],p)); sc=spearmanr(y[te],p).correlation
        sps.append(0.0 if np.isnan(sc) else sc); maes.append(mean_absolute_error(y[te],p))
    f=lambda a:[round(float(np.mean(a)),3),round(float(np.std(a)),3)]
    return {"R2":f(r2s),"Spearman":f(sps),"MAE":f(maes)}

splits = {
  "random_5fold": (KFold(5,shuffle=True,random_state=0), None),
  "host_disjoint_5fold": (GroupKFold(5), groups_host),
  "guest_disjoint_5fold": (GroupKFold(5), groups_guest),
}
feats = {"mean_predictor":None, "ECFP_RF":X_ecfp, "RDKitDesc_RF":X_desc, "PairDesc_RF":X_pair}
res={}
for sname,(spl,grp) in splits.items():
    res[sname]={}
    for fname,X in feats.items():
        if fname=="mean_predictor": res[sname][fname]=eval_mean(spl,grp)
        else: res[sname][fname]=evaluate(X,spl,grp)
        print(f"{sname:22s} {fname:16s} {res[sname][fname]}")
out["baselines"]=res

with open(OUTPUT,"w") as f:
    json.dump(out,f,indent=2)
print("\n=== SUMMARY ===")
print(json.dumps({k:out[k] for k in ["counts","dup_pairs","parse","n_usable_both_parse","leakage"]},indent=2))
print(f"WROTE {OUTPUT}")
