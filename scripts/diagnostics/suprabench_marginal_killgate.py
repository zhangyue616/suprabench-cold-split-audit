#!/usr/bin/env python
"""SUPRABENCH_MARGINAL_KILLGATE — does a pair-aware model beat marginal/additive nulls
under strict cold-start (incl. DOUBLE-COLD)? CPU only, RF/descriptor + trivial baselines, NO GNN."""
import argparse
import json, hashlib, warnings, re
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

REPO_ROOT=Path(__file__).resolve().parents[2]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True,help='Path to the upstream records.parquet (not distributed in this repository).')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/'diagnostics'/'suprabench_marginal_killgate_results.json',help='Output JSON path.')
    return p.parse_args()
ARGS=parse_args(); REC=ARGS.records; OUTPUT=ARGS.output
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
RNG = np.random.RandomState(0)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

r = pd.read_parquet(REC).reset_index(drop=True)
r['y']=r['binding_affinity'].astype(float)
def fam(name):
    n=str(name).lower()
    if 'cucurbit' in n or re.search(r'\bcb\s*\[?\d',n): return 'cucurbituril'
    if 'cyclodext' in n or re.search(r'\b[abgαβγ]-?cd\b',n): return 'cyclodextrin'
    if 'calix' in n: return 'calixarene'
    if 'pillar' in n: return 'pillararene'
    if 'cavitand' in n: return 'cavitand'
    if 'cyclophane' in n: return 'cyclophane'
    if 'crown' in n: return 'crown_ether'
    return 'other'
r['family']=r['host_name'].map(fam)
hid = pd.factorize(r['host_smiles'])[0]; gid = pd.factorize(r['guest_smiles'])[0]
r['hid']=hid; r['gid']=gid
y=r['y'].values

DESC=[('MW',Descriptors.MolWt),('LogP',Descriptors.MolLogP),('TPSA',Descriptors.TPSA),
      ('HBD',Descriptors.NumHDonors),('HBA',Descriptors.NumHAcceptors),('RotB',Descriptors.NumRotatableBonds),
      ('Ring',rdMolDescriptors.CalcNumRings),('ArRing',rdMolDescriptors.CalcNumAromaticRings),
      ('FrCSP3',Descriptors.FractionCSP3),('Heavy',lambda m:m.GetNumHeavyAtoms()),('Charge',Chem.GetFormalCharge)]
def descvec(s):
    m=Chem.MolFromSmiles(s); o=[]
    for _,fn in DESC:
        try:o.append(float(fn(m)))
        except:o.append(0.0)
    return np.array(o,np.float32)
uh={s:descvec(s) for s in r['host_smiles'].unique()}
ug={s:descvec(s) for s in r['guest_smiles'].unique()}
H_d=np.vstack([uh[s] for s in r['host_smiles']]); G_d=np.vstack([ug[s] for s in r['guest_smiles']])
def sdiv(a,b):return a/np.where(np.abs(b)<1e-6,1e-6,b)
PAIR=np.column_stack([H_d[:,0]+G_d[:,0],sdiv(G_d[:,0],H_d[:,0]),H_d[:,9]-G_d[:,9],sdiv(G_d[:,9],H_d[:,9]),
      H_d[:,10]*G_d[:,10],H_d[:,10]+G_d[:,10],np.abs(H_d[:,2]-G_d[:,2]),H_d[:,3]+G_d[:,4],H_d[:,4]+G_d[:,3],H_d[:,1]+G_d[:,1]]).astype(np.float32)
X_host=H_d; X_guest=G_d; X_pair=np.hstack([H_d,G_d,PAIR])

def fp(s):
    m=Chem.MolFromSmiles(s); return AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
hfp={s:fp(s) for s in r['host_smiles'].unique()}; gfp={s:fp(s) for s in r['guest_smiles'].unique()}

def metrics(yt,yp):
    if len(np.unique(yt))<2 or np.std(yp)<1e-9:
        return dict(R2=float('nan'),Spearman=0.0,MAE=float(mean_absolute_error(yt,yp)),n=len(yt))
    sc=spearmanr(yt,yp).correlation
    return dict(R2=float(r2_score(yt,yp)),Spearman=float(0.0 if np.isnan(sc) else sc),MAE=float(mean_absolute_error(yt,yp)),n=len(yt))

def b_global(tr,te): return np.full(len(te),y[tr].mean())
def _grpmean(tr,te,key):
    g=y[tr].mean(); d=pd.Series(y[tr]).groupby(key[tr]).mean().to_dict()
    return np.array([d.get(k,g) for k in key[te]])
def b_hostmean(tr,te): return _grpmean(tr,te,hid)
def b_guestmean(tr,te): return _grpmean(tr,te,gid)
def b_additive(tr,te,lam=5.0):
    g=y[tr].mean()
    def dev(key):
        s=pd.Series(y[tr]).groupby(key[tr]); n=s.count(); mu=s.mean(); return ((mu-g)*n/(n+lam)).to_dict()
    dh=dev(hid); dg=dev(gid)
    return np.array([g+dh.get(hid[i],0.0)+dg.get(gid[i],0.0) for i in te])
def b_te_ridge(tr,te,lam=5.0):
    g=y[tr].mean()
    def enc_full(key):
        s=pd.Series(y[tr]).groupby(key[tr]); n=s.count(); mu=s.mean(); return ((mu*n+g*lam)/(n+lam)).to_dict()
    eh=enc_full(hid); eg=enc_full(gid)
    tr=np.array(tr); oof_h=np.zeros(len(tr)); oof_g=np.zeros(len(tr)); kf=KFold(5,shuffle=True,random_state=1)
    for a,b in kf.split(tr):
        ia,ib=tr[a],tr[b]; ga=y[ia].mean()
        sh=pd.Series(y[ia]).groupby(hid[ia]); dh=((sh.mean()*sh.count()+ga*lam)/(sh.count()+lam)).to_dict()
        sg=pd.Series(y[ia]).groupby(gid[ia]); dg=((sg.mean()*sg.count()+ga*lam)/(sg.count()+lam)).to_dict()
        oof_h[b]=[dh.get(hid[i],ga) for i in ib]; oof_g[b]=[dg.get(gid[i],ga) for i in ib]
    Rg=Ridge(alpha=1.0).fit(np.c_[oof_h,oof_g],y[tr])
    return Rg.predict(np.c_[[eh.get(hid[i],g) for i in te],[eg.get(gid[i],g) for i in te]])
def b_nn1(tr,te):
    Z=StandardScaler().fit(np.hstack([X_host,X_guest])[tr]).transform(np.hstack([X_host,X_guest]))
    nn=NearestNeighbors(n_neighbors=1).fit(Z[tr]); _,idx=nn.kneighbors(Z[te]); return y[tr][idx[:,0]]
def _rf(X,tr,te):
    return RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=0,max_features='sqrt').fit(X[tr],y[tr]).predict(X[te])
def b_hostRF(tr,te): return _rf(X_host,tr,te)
def b_guestRF(tr,te): return _rf(X_guest,tr,te)
def b_pairRF(tr,te): return _rf(X_pair,tr,te)
BASELINES={'global_mean':b_global,'host_mean':b_hostmean,'guest_mean':b_guestmean,'additive_shrink':b_additive,
           'te_ridge':b_te_ridge,'nn1':b_nn1,'host_only_RF':b_hostRF,'guest_only_RF':b_guestRF,'PAIR_RF':b_pairRF}

def oof_preds(folds):
    res={b:{'y':[],'p':[],'h':[]} for b in BASELINES}
    for tr,te in folds:
        for b,fn in BASELINES.items():
            res[b]['y'].append(y[te]); res[b]['p'].append(fn(tr,te)); res[b]['h'].append(hid[te])
    return {b:(np.concatenate(res[b]['y']),np.concatenate(res[b]['p']),np.concatenate(res[b]['h'])) for b in BASELINES}
def host_boot_ci(yt,yp,ht,reps=600):
    hosts=np.unique(ht); r2=[]
    for _ in range(reps):
        sel=RNG.choice(hosts,len(hosts),replace=True); mask=np.concatenate([np.where(ht==h)[0] for h in sel])
        if len(np.unique(yt[mask]))<2: continue
        r2.append(r2_score(yt[mask],yp[mask]))
    return [round(float(np.percentile(r2,2.5)),3),round(float(np.percentile(r2,97.5)),3)] if r2 else [None,None]

def kfold_pairs(): return list(KFold(5,shuffle=True,random_state=0).split(np.arange(len(y))))
def group_folds(key): return list(GroupKFold(5).split(np.arange(len(y)),y,key))
RESULTS={}
print("=== random / host-cold / guest-cold (OOF) ===")
for sname,folds in [('random',kfold_pairs()),('host_cold',group_folds(hid)),('guest_cold',group_folds(gid))]:
    op=oof_preds(folds); RESULTS[sname]={}
    for b,(yt,yp,ht) in op.items():
        m=metrics(yt,yp); m['R2_CI']=host_boot_ci(yt,yp,ht); RESULTS[sname][b]={k:(round(v,3) if isinstance(v,float) else v) for k,v in m.items()}
        print(f"{sname:11s} {b:16s} R2={RESULTS[sname][b]['R2']} CI={m['R2_CI']} rho={RESULTS[sname][b]['Spearman']} MAE={RESULTS[sname][b]['MAE']}")

print("\n=== DOUBLE-COLD (test host AND guest both unseen; 12 seeds) ===")
uh_ids=np.unique(hid); ug_ids=np.unique(gid); dc={b:[] for b in BASELINES}; dc_n=[]
for seed in range(12):
    rs=np.random.RandomState(seed)
    h_te=set(rs.choice(uh_ids,int(0.25*len(uh_ids)),replace=False)); g_te=set(rs.choice(ug_ids,int(0.25*len(ug_ids)),replace=False))
    te=np.array([i for i in range(len(y)) if hid[i] in h_te and gid[i] in g_te])
    tr=np.array([i for i in range(len(y)) if hid[i] not in h_te and gid[i] not in g_te])
    if len(te)<25 or len(np.unique(y[te]))<2: continue
    dc_n.append(len(te))
    for b,fn in BASELINES.items(): dc[b].append(metrics(y[te],fn(tr,te))['R2'])
RESULTS['double_cold']={'mean_test_n':int(np.mean(dc_n)),'n_seeds':len(dc_n)}
for b in BASELINES:
    a=np.array([x for x in dc[b] if not np.isnan(x)])
    if len(a)==0:
        RESULTS['double_cold'][b]={'R2_mean':None,'note':'DEGENERATE constant-pred (identity null collapses under double-cold)'}
        print(f"double_cold {b:16s} DEGENERATE (identity null -> constant pred)"); continue
    RESULTS['double_cold'][b]={'R2_mean':round(float(a.mean()),3),'R2_std':round(float(a.std()),3),
        'R2_p2.5':round(float(np.percentile(a,2.5)),3),'R2_p97.5':round(float(np.percentile(a,97.5)),3),'n_valid':int(len(a))}
    print(f"double_cold {b:16s} R2={RESULTS['double_cold'][b]['R2_mean']} +/-{RESULTS['double_cold'][b]['R2_std']} (n_te~{RESULTS['double_cold']['mean_test_n']})")

print("\n=== FAMILY-COLD (leave-one-family-out) ===")
fam_counts=r.groupby('family').agg(pairs=('y','size'),hosts=('host_smiles','nunique'))
RESULTS['family_cold']={'family_sizes':{k:{'pairs':int(v['pairs']),'hosts':int(v['hosts'])} for k,v in fam_counts.to_dict('index').items()}}
for famname in fam_counts.index:
    np_,nh_=int(fam_counts.loc[famname,'pairs']),int(fam_counts.loc[famname,'hosts'])
    te=np.where(r['family'].values==famname)[0]; tr=np.where(r['family'].values!=famname)[0]
    small=(nh_<5 or np_<80); row={'pairs':np_,'hosts':nh_,'flag':'NO_CONCLUSION_small' if small else 'ok'}
    for b in ['global_mean','guest_mean','guest_only_RF','PAIR_RF']:
        row[b]=round(metrics(y[te],BASELINES[b](tr,te))['R2'],3)
    RESULTS['family_cold'][famname]=row
    print(f"family={famname:13s} pairs={np_:4d} hosts={nh_:3d} {row['flag']:18s} guest_mean={row['guest_mean']} guest_RF={row['guest_only_RF']} PAIR_RF={row['PAIR_RF']}")

print("\n=== train-test similarity (max Tanimoto to nearest train) ===")
def simdist(folds,which='host'):
    smap=hfp if which=='host' else gfp; key=r['host_smiles'].values if which=='host' else r['guest_smiles'].values; allv=[]
    for tr,te in folds:
        trf=[smap[s] for s in np.unique(key[tr])]
        for s in np.unique(key[te]):
            sims=DataStructs.BulkTanimotoSimilarity(smap[s],trf); allv.append(max(sims) if sims else 0.0)
    a=np.array(allv); return {'median':round(float(np.median(a)),3),'p90':round(float(np.percentile(a,90)),3),'frac_gt_0.9':round(float((a>0.9).mean()),3)}
SIM={}
for sname,folds in [('random',kfold_pairs()),('host_cold',group_folds(hid)),('guest_cold',group_folds(gid))]:
    SIM[sname]={'host':simdist(folds,'host'),'guest':simdist(folds,'guest')}; print(sname,SIM[sname])
RESULTS['similarity']=SIM
with open(OUTPUT,"w") as f:
    json.dump(RESULTS,f,indent=2)
print(f"\nWROTE {OUTPUT}")
