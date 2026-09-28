#!/usr/bin/env python
"""SUPRABENCH_BASELINES (modern) — K3 (descriptor-RF vs XGB/LGBM vs strong 2D GNN vs light 3D)
and K2-robustness (six-null survival + block-level increment under strong models).
Reuses EXACT smoke1/pregate folds + features. Adopted env tapt_5090_modern (sklearn 1.9.0, torch 2.11 cu128).
argv: 'smoke' = fast code-path validation; 'full' = real run."""
import os, json, time, hashlib, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr, wilcoxon, linregress
import xgboost as xgb, lightgbm as lgb
import torch, torch.nn as nn
from torch_geometric.nn import GINEConv, AttentiveFP, global_mean_pool
from torch_geometric.data import Data, Batch

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
SMOKE = (MODE == 'smoke')
DATE = "2026-06-26"
SS = str(Path(__file__).resolve().parents[1] / "clean")
COND = os.path.join(SS, "suprabench_bap_clean_cond.csv")
BASE = os.path.join(SS, "suprabench_bap_clean.csv")
OUT = os.path.join(SS, "suprabench_baselines_modern_results.json") if not SMOKE else \
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines_smoke_results.json")
SMOKE_ANCH = {'random': 0.677, 'host_cold': 0.448, 'guest_cold': 0.641, 'double_cold': 0.386}
TOL = 0.02
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = np.random.RandomState(0); BRNG = np.random.RandomState(0)
MAXEP = 3 if SMOKE else 300
PAT = 2 if SMOKE else 30

def log(*a): print(*a, flush=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

t0 = time.time()
df = pd.read_csv(COND); base = pd.read_csv(BASE)
assert (df['host_smiles'].values == base['host_smiles'].values).all() and \
       (df['guest_smiles'].values == base['guest_smiles'].values).all() and \
       np.allclose(df['y'].values, base['y'].values), "JOIN_FAIL"
y = df['y'].values.astype(np.float32)
hid = pd.factorize(df['host_smiles'])[0]; gid = pd.factorize(df['guest_smiles'])[0]; fam = df['family'].values
hname = df['host_name'].values; srcv = df['source'].values; solv = df['solvent_bucket'].values
uhost = pd.factorize(df['host_smiles'])[1]; uguest = pd.factorize(df['guest_smiles'])[1]
log(f"MODE={MODE} rows={len(df)} hosts={len(uhost)} guests={len(uguest)} dev={DEV} "
    f"torch={torch.__version__} sklearn={sklearn.__version__} xgb={xgb.__version__} lgb={lgb.__version__} "
    f"cond_sha={sha256(COND)[:12]}")

# ---------------- features (EXACT pregate construction) ----------------
DESC = [('MW', Descriptors.MolWt), ('LogP', Descriptors.MolLogP), ('TPSA', Descriptors.TPSA), ('HBD', Descriptors.NumHDonors),
        ('HBA', Descriptors.NumHAcceptors), ('RotB', Descriptors.NumRotatableBonds), ('Ring', rdMolDescriptors.CalcNumRings),
        ('ArRing', rdMolDescriptors.CalcNumAromaticRings), ('FrCSP3', Descriptors.FractionCSP3),
        ('Heavy', lambda m: m.GetNumHeavyAtoms()), ('Charge', Chem.GetFormalCharge)]
def dvec(s):
    m = Chem.MolFromSmiles(s); o = []
    for _, fn in DESC:
        try: o.append(float(fn(m)))
        except Exception: o.append(0.0)
    return np.array(o, np.float32)
uh = {s: dvec(s) for s in df['host_smiles'].unique()}; ug = {s: dvec(s) for s in df['guest_smiles'].unique()}
H = np.vstack([uh[s] for s in df['host_smiles']]); G = np.vstack([ug[s] for s in df['guest_smiles']])
def sdiv(a, b): return a / np.where(np.abs(b) < 1e-6, 1e-6, b)
P = np.column_stack([H[:,0]+G[:,0], sdiv(G[:,0],H[:,0]), H[:,9]-G[:,9], sdiv(G[:,9],H[:,9]), H[:,10]*G[:,10],
                     H[:,10]+G[:,10], np.abs(H[:,2]-G[:,2]), H[:,3]+G[:,4], H[:,4]+G[:,3], H[:,1]+G[:,1]]).astype(np.float32)
X_pair = np.hstack([H, G, P]); X_host = H; X_guest = G
def ecfp(s): return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024), np.float32)
he = {s: ecfp(s) for s in df['host_smiles'].unique()}; ge = {s: ecfp(s) for s in df['guest_smiles'].unique()}
X_ecfp = np.hstack([np.vstack([he[s] for s in df['host_smiles']]), np.vstack([ge[s] for s in df['guest_smiles']])])
COND_OH = pd.get_dummies(df[['source','solvent_bucket','temp_bucket','ph_bucket']].astype(str)).values.astype(np.float32)
SRC_OH = pd.get_dummies(df['source'].astype(str)).values.astype(np.float32)
hfps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in uhost]
gfps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in uguest]
HostSim = np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f, hfps), np.float32) for f in hfps])
GuestSim = np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f, gfps), np.float32) for f in gfps])
log(f"features+sim built [{time.time()-t0:.0f}s]")

# ---------------- light 3D arm: guest ETKDG shape descriptors ----------------
def guest3d(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumHeavyAtoms() > 70: return None
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 0; p.useRandomCoords = True; p.maxIterations = 200
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
        try: AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception: pass
        return np.array([rdMolDescriptors.CalcAsphericity(m), rdMolDescriptors.CalcEccentricity(m),
                         rdMolDescriptors.CalcInertialShapeFactor(m), rdMolDescriptors.CalcNPR1(m),
                         rdMolDescriptors.CalcNPR2(m), rdMolDescriptors.CalcRadiusOfGyration(m),
                         rdMolDescriptors.CalcSpherocityIndex(m), rdMolDescriptors.CalcPMI1(m),
                         rdMolDescriptors.CalcPMI2(m), rdMolDescriptors.CalcPMI3(m)], np.float32)
    except Exception:
        return None
THREE_D = {'available': False}
g3_unique = {}
n_ok = 0
guests_to_embed = list(df['guest_smiles'].unique())
if SMOKE:
    guests_to_embed = guests_to_embed[:40]
emb_t0 = time.time(); EMB_BUDGET = 240
for s in guests_to_embed:
    if time.time() - emb_t0 > EMB_BUDGET:
        log('  3D embed budget hit; embedded ' + str(n_ok) + ', rest median-filled'); break
    v = guest3d(s); g3_unique[s] = v
    if v is not None: n_ok += 1
D3 = 10
arr = np.array([g3_unique[s] for s in g3_unique if g3_unique[s] is not None])
med = np.median(arr, axis=0) if len(arr) else np.zeros(D3, np.float32)
def g3row(s):
    v = g3_unique.get(s)
    return v if v is not None else med
G3 = np.vstack([g3row(s) for s in df['guest_smiles']]).astype(np.float32)
X_pair3d = np.hstack([X_pair, G3]).astype(np.float32)
THREE_D = {'available': True, 'arm': 'guest_ETKDG_shape(10d)+PAIR_descriptors (host 3D omitted: macrocycles)',
           'n_unique_guests': len(g3_unique), 'n_embed_ok': n_ok,
           'embed_success_rate': round(n_ok / max(1, len(g3_unique)), 3)}
log(f"3D guest-shape: {n_ok}/{len(g3_unique)} embedded [{time.time()-t0:.0f}s]")

# ---------------- molecular graphs (PyG) ----------------
ELEMS = [6,7,8,16,15,9,17,35,53,11,5,14,1,3,19,20,30,64]
def onehot(x, allowed):
    v = [0.0]*(len(allowed)+1); v[allowed.index(x) if x in allowed else -1] = 1.0; return v
def atomf(a):
    return (onehot(a.GetAtomicNum(), ELEMS) + onehot(min(a.GetDegree(),6), list(range(7))) +
            onehot(int(np.clip(a.GetFormalCharge(),-2,2)), [-2,-1,0,1,2]) +
            onehot(str(a.GetHybridization()), ['S','SP','SP2','SP3','SP3D','SP3D2']) +
            [float(a.GetIsAromatic()), float(a.IsInRing())] + onehot(min(a.GetTotalNumHs(),4), list(range(5))))
BT = {Chem.BondType.SINGLE:0, Chem.BondType.DOUBLE:1, Chem.BondType.TRIPLE:2, Chem.BondType.AROMATIC:3}
def bondf(b):
    v = [0.,0.,0.,0.]; v[BT.get(b.GetBondType(), 0)] = 1.
    return v + [float(b.GetIsConjugated()), float(b.IsInRing())]
FDIM = len(atomf(Chem.MolFromSmiles('CCO').GetAtomWithIdx(0))); EDIM = 6
def to_data(s):
    m = Chem.MolFromSmiles(s)
    x = torch.tensor([atomf(a) for a in m.GetAtoms()], dtype=torch.float)
    ei = [[], []]; ea = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bondf(b)
        ei[0] += [i, j]; ei[1] += [j, i]; ea += [f, f]
    if len(ea) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long); edge_attr = torch.zeros((0,EDIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(ei, dtype=torch.long); edge_attr = torch.tensor(ea, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
host_batch = Batch.from_data_list([to_data(s) for s in uhost]).to(DEV)
guest_batch = Batch.from_data_list([to_data(s) for s in uguest]).to(DEV)
log(f"graphs built FDIM={FDIM} EDIM={EDIM} [{time.time()-t0:.0f}s]")

class GINEEnc(nn.Module):
    def __init__(s, fdim, edim, d=128, L=3, drop=0.1):
        super().__init__(); s.lin0 = nn.Linear(fdim, d)
        s.convs = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d)), edge_dim=edim) for _ in range(L)])
        s.bns = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(L)]); s.drop = nn.Dropout(drop)
    def forward(s, data):
        h = s.lin0(data.x)
        for conv, bn in zip(s.convs, s.bns):
            h = conv(h, data.edge_index, data.edge_attr); h = bn(h); h = torch.relu(h); h = s.drop(h)
        return global_mean_pool(h, data.batch)
class AFPEnc(nn.Module):
    def __init__(s, fdim, edim, d=128, L=3, T=2, drop=0.1, out=128):
        super().__init__()
        s.afp = AttentiveFP(in_channels=fdim, hidden_channels=d, out_channels=out, edge_dim=edim,
                            num_layers=L, num_timesteps=T, dropout=drop)
    def forward(s, data):
        return s.afp(data.x, data.edge_index, data.edge_attr, data.batch)
class PairNet(nn.Module):
    def __init__(s, enc, d):
        super().__init__(); s.enc = enc
        s.head = nn.Sequential(nn.Linear(4*d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(s, hidx, gidx):
        heb = s.enc(host_batch); geb = s.enc(guest_batch)
        a = heb[hidx]; b = geb[gidx]
        return s.head(torch.cat([a, b, a*b, (a-b).abs()], 1)).squeeze(1)
def mk_gine(): return GINEEnc(FDIM, EDIM, d=128, L=3)
def mk_afp(): return AFPEnc(FDIM, EDIM, d=128, L=3, T=2, out=128)
GNN = {'GINE': (mk_gine, 128), 'AttentiveFP': (mk_afp, 128)}

def train_gnn(make_enc, d, tr, te, seed=0):
    torch.manual_seed(seed)
    tr = np.array(tr); rs = np.random.RandomState(seed); perm = rs.permutation(len(tr))
    nval = max(20, int(0.15*len(tr))); val = tr[perm[:nval]]; trn = tr[perm[nval:]]
    mu, sd = y[trn].mean(), y[trn].std() + 1e-6
    model = PairNet(make_enc(), d).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    hi = torch.tensor(hid, device=DEV); gi = torch.tensor(gid, device=DEV)
    yt = torch.tensor((y - mu)/sd, device=DEV)
    trn_t = torch.tensor(trn, device=DEV); val_t = torch.tensor(val, device=DEV); te_t = torch.tensor(te, device=DEV)
    yval = torch.tensor(y[val], device=DEV)
    best = 1e9; best_state = None; wait = 0
    for ep in range(MAXEP):
        model.train(); opt.zero_grad()
        pred = model(hi[trn_t], gi[trn_t]); loss = ((pred - yt[trn_t])**2).mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(hi[val_t], gi[val_t])*sd + mu
            vmae = torch.abs(vp - yval).mean().item()
        if vmae < best - 1e-4: best = vmae; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; wait = 0
        else: wait += 1
        if wait >= PAT: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pr = (model(hi[te_t], gi[te_t])*sd + mu).cpu().numpy()
    del model; torch.cuda.empty_cache()
    return pr

# ---------------- classical models + nulls ----------------
def rf(X, tr, te): return RandomForestRegressor(300, n_jobs=-1, random_state=0, max_features='sqrt').fit(X[tr], y[tr]).predict(X[te])
def xgbr(X, tr, te):
    return xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=1.0, n_jobs=-1, random_state=0, tree_method='hist').fit(X[tr], y[tr]).predict(X[te])
def lgbr(X, tr, te):
    return lgb.LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             subsample_freq=1, reg_lambda=1.0, n_jobs=-1, random_state=0, verbose=-1).fit(X[tr], y[tr]).predict(X[te])
def simknn(tr, te, k=5):
    htr = hid[tr]; gtr = gid[tr]; ytr = y[tr]; pr = np.zeros(len(te)); ms = np.zeros(len(te))
    for ii, i in enumerate(te):
        comb = (HostSim[hid[i], htr] + GuestSim[gid[i], gtr]) / 2.0
        idx = np.argpartition(-comb, min(k, len(comb)-1))[:k]; w = comb[idx] + 1e-6
        pr[ii] = np.sum(w*ytr[idx]) / np.sum(w); ms[ii] = comb[idx].max()
    return pr, ms

PROPER = ['sim_knn_k5', 'additive', 'host_only_RF', 'guest_only_RF', 'cond_only_RF', 'source_only_RF']
STRONG = ['PAIR_RF', 'ECFP_RF', 'XGB_pair', 'XGB_ecfp', 'LGBM_pair', 'LGBM_ecfp', '3D_pair_RF', 'GINE', 'AttentiveFP']
def predict_all(tr, te):
    """Return dict model->preds and max_sim array (from sim_knn)."""
    out = {}
    out['PAIR_RF'] = rf(X_pair, tr, te)
    out['ECFP_RF'] = rf(X_ecfp, tr, te)
    out['XGB_pair'] = xgbr(X_pair, tr, te); out['XGB_ecfp'] = xgbr(X_ecfp, tr, te)
    out['LGBM_pair'] = lgbr(X_pair, tr, te); out['LGBM_ecfp'] = lgbr(X_ecfp, tr, te)
    out['3D_pair_RF'] = rf(X_pair3d, tr, te)
    out['host_only_RF'] = rf(X_host, tr, te); out['guest_only_RF'] = rf(X_guest, tr, te)
    out['cond_only_RF'] = rf(COND_OH, tr, te); out['source_only_RF'] = rf(SRC_OH, tr, te)
    g = y[tr].mean(); out['additive'] = out['host_only_RF'] + out['guest_only_RF'] - g
    s5, ms = simknn(tr, te, 5); out['sim_knn_k5'] = s5
    for name, (mk, d) in GNN.items():
        out[name] = train_gnn(mk, d, tr, te, seed=0)
    return out, ms

def R2(yt, yp): return float(r2_score(yt, yp)) if (len(np.unique(yt)) > 1 and np.std(yp) > 1e-9) else float('nan')
def hci(yt, yp, ht, reps=400):
    Hs = np.unique(ht); r2 = []
    for _ in range(reps):
        sel = RNG.choice(Hs, len(Hs), True); m = np.concatenate([np.where(ht == h)[0] for h in sel])
        if len(np.unique(yt[m])) > 1: r2.append(r2_score(yt[m], yp[m]))
    return [round(float(np.percentile(r2, 2.5)), 3), round(float(np.percentile(r2, 97.5)), 3)] if r2 else [None, None]
def calib(yt, yp):
    try: return round(float(linregress(yp, yt).slope), 3)
    except Exception: return None

ALLM = STRONG + PROPER
R = {'_meta': {'cond_sha256': sha256(COND), 'n': int(len(df)), 'sklearn': sklearn.__version__, 'torch': torch.__version__,
               'xgboost': xgb.__version__, 'lightgbm': lgb.__version__, 'device': DEV, 'mode': MODE,
               'gnn_budget': {'max_epoch': MAXEP, 'patience': PAT, 'lr': 1e-3, 'val_frac': 0.15, 'd': 128, 'L': 3},
               'models': ALLM}, '3D_arm': THREE_D}
def save(): json.dump(R, open(OUT, 'w'), indent=2, default=float)

# ---------------- folds (EXACT) ----------------
def dcfolds():
    idx = np.arange(len(y)); h = hid; g = gid; uh_ = np.unique(h); ug_ = np.unique(g); out = []
    for seed in range(12):
        rs = np.random.RandomState(seed)
        hte = set(rs.choice(uh_, max(1, int(0.25*len(uh_))), False)); gte = set(rs.choice(ug_, max(1, int(0.25*len(ug_))), False))
        te = idx[[(h[k] in hte and g[k] in gte) for k in range(len(idx))]]
        tr = idx[[(h[k] not in hte and g[k] not in gte) for k in range(len(idx))]]
        if len(te) < 25 or len(np.unique(y[te])) < 2 or len(tr) < 50: continue
        out.append((tr, te))
    return out
FOLDS = {'random': list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(y)))),
         'host_cold': list(GroupKFold(5).split(np.arange(len(y)), y, hid)),
         'guest_cold': list(GroupKFold(5).split(np.arange(len(y)), y, gid))}
DCF = dcfolds()

# ---------- fold-consistency anchor (PAIR_RF) ----------
log("=== fold-consistency (PAIR_RF) ===")
fc = {}
for sp, fo in FOLDS.items():
    yy, pp = [], []
    for tr, te in (fo[:1] if SMOKE else fo): yy.append(y[te]); pp.append(rf(X_pair, tr, te))
    fc[sp] = round(R2(np.concatenate(yy), np.concatenate(pp)), 3)
dcf_use = DCF[:2] if SMOKE else DCF
fc['double_cold'] = round(float(np.mean([R2(y[te], rf(X_pair, tr, te)) for tr, te in dcf_use])), 3)
R['fold_consistency'] = fc
ok = all(abs(fc[k] - SMOKE_ANCH[k]) < (0.1 if SMOKE else TOL) for k in SMOKE_ANCH)
log(f"fold_consistency {fc} {'OK' if ok else 'MISMATCH'}")
if not ok and not SMOKE:
    R['_meta']['STOP'] = 'FOLD_MISMATCH'; save(); log("FOLD_MISMATCH"); sys.exit(2)

# ---------------- evaluate splits ----------------
def eval_group(splitname, folds, host_level=True):
    log(f"=== {splitname} ({len(folds)} folds) ===")
    preds = {m: [] for m in ALLM}; yy = []; hh = []
    for fi, (tr, te) in enumerate(folds):
        po, ms = predict_all(tr, te); yy.append(y[te]); hh.append(hid[te])
        for m in ALLM: preds[m].append(po[m])
        log(f"  {splitname} fold {fi+1}/{len(folds)} done [{time.time()-t0:.0f}s]")
    yy = np.concatenate(yy); hh = np.concatenate(hh)
    res = {}
    for m in ALLM:
        p = np.concatenate(preds[m])
        res[m] = {'R2': round(R2(yy, p), 3),
                  'R2_hostCI': hci(yy, p, hh) if host_level else None,
                  'Spearman': round(float(spearmanr(yy, p).correlation), 3) if len(np.unique(yy)) > 1 else None,
                  'MAE': round(float(mean_absolute_error(yy, p)), 3)}
    R[splitname] = res; save()
    for m in STRONG: log(f"    {m:12s} R2={res[m]['R2']} CI={res[m]['R2_hostCI']}")
    return res

for sp in (['random'] if SMOKE else ['random', 'host_cold', 'guest_cold']):
    eval_group(sp, FOLDS[sp][:1] if SMOKE else FOLDS[sp])

# ---------------- double-cold + block-level K2 ----------------
log("=== double_cold (blocks) + K2 block-level ===")
block_R2 = {m: [] for m in ALLM}; recs = []
use_dcf = DCF[:2] if SMOKE else DCF
for bi, (tr, te) in enumerate(use_dcf):
    po, ms = predict_all(tr, te)
    for m in ALLM: block_R2[m].append(R2(y[te], po[m]))
    for ii, i in enumerate(te):
        rec = {'block': bi, 'host': int(hid[i]), 'family': fam[i], 'y': float(y[i]), 'max_sim': float(ms[ii])}
        for m in ALLM: rec[m] = float(po[m][ii])
        recs.append(rec)
    log(f"  double_cold block {bi+1}/{len(use_dcf)} done [{time.time()-t0:.0f}s]")
bdf = pd.DataFrame({m: block_R2[m] for m in ALLM})
bdf['best_null'] = bdf[PROPER].max(axis=1)
R['double_cold_summary'] = {m: {'R2_mean': round(float(np.nanmean(block_R2[m])), 3),
                                'R2_std': round(float(np.nanstd(block_R2[m])), 3)} for m in ALLM}
# K2 block-level gain for each strong model vs best-of-six-null
k2 = {'n_blocks': int(len(bdf)), 'best_null_meanR2': round(float(bdf['best_null'].mean()), 3),
      'which_null_best_per_block': [max(PROPER, key=lambda m: bdf.iloc[r][m]) for r in range(len(bdf))]}
for m in ['PAIR_RF', 'XGB_pair', 'LGBM_pair', 'GINE', 'AttentiveFP', '3D_pair_RF']:
    g = (bdf[m] - bdf['best_null']).values
    g = g[~np.isnan(g)]
    boot = [float(np.mean(BRNG.choice(g, len(g), True))) for _ in range(3000)] if len(g) else [0]
    try: wp = float(wilcoxon(bdf[m].dropna(), bdf['best_null'][bdf[m].notna()]).pvalue)
    except Exception: wp = None
    k2[m] = {'meanR2': round(float(bdf[m].mean()), 3), 'gain_vs_bestnull_mean': round(float(np.mean(g)), 3),
             'gain_block_CI': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
             'n_blocks_model_gt_bestnull': int((g > 0).sum()), 'wilcoxon_p': wp}
R['K2_block_double_cold'] = k2; save()
log("  K2 block gains: " + json.dumps({m: k2[m]['gain_vs_bestnull_mean'] for m in ['PAIR_RF','XGB_pair','LGBM_pair','GINE','AttentiveFP','3D_pair_RF']}))

# ---------------- low-sim<0.4 bin (per-model per-family) ----------------
rdf = pd.DataFrame(recs)
binr = rdf[rdf['max_sim'] < 0.4].copy()
lowsim = {'n': int(len(binr)), 'n_families': int(binr['family'].nunique()) if len(binr) else 0,
          'per_model_R2': {m: round(R2(binr['y'].values, binr[m].values), 3) for m in STRONG} if len(binr) else {}}
lowsim['per_family'] = {}
for f, sub in (binr.groupby('family') if len(binr) else []):
    if len(sub) < 15: lowsim['per_family'][f] = {'n': int(len(sub)), 'flag': 'NO_CONCLUSION_small'}; continue
    lowsim['per_family'][f] = {'n': int(len(sub)),
                               **{m: round(R2(sub['y'].values, sub[m].values), 3) for m in ['PAIR_RF','XGB_pair','LGBM_pair','GINE','AttentiveFP','sim_knn_k5']}}
R['lowsim_bin'] = lowsim; save()

# ---------------- family-cold decomposition (per model) ----------------
log("=== family_cold (per-model R2/Spearman/calib/offset) ===")
fams = sorted(set(fam))
if SMOKE: fams = fams[:2]
R['family_cold'] = {}
for f in fams:
    te = np.where(fam == f)[0]; tr = np.where(fam != f)[0]
    nh = len(np.unique(hid[te])); small = (nh < 5 or len(te) < 80)
    po, ms = predict_all(tr, te)
    row = {'n_pair': int(len(te)), 'n_host': int(nh), 'flag': 'NO_CONCLUSION_small' if small else 'ok',
           'y_mean': round(float(y[te].mean()), 3), 'models': {}}
    for m in STRONG:
        p = po[m]
        row['models'][m] = {'R2': round(R2(y[te], p), 3),
                            'Spearman': round(float(spearmanr(y[te], p).correlation), 3) if len(np.unique(y[te])) > 1 else None,
                            'calib_slope': calib(y[te], p),
                            'offset': round(float(y[te].mean() - p.mean()), 3)}
    R['family_cold'][f] = row; save()
    log(f"  {f:13s} {row['flag']:18s} PAIR_RF R2={row['models']['PAIR_RF']['R2']} rho={row['models']['PAIR_RF']['Spearman']} "
        f"GINE R2={row['models']['GINE']['R2']} AFP R2={row['models']['AttentiveFP']['R2']} XGB R2={row['models']['XGB_pair']['R2']}")

R['_meta']['runtime_s'] = round(time.time() - t0, 1)
save()
log(f"WROTE {OUT}  total {time.time()-t0:.0f}s  sha={sha256(OUT)[:12]}")
