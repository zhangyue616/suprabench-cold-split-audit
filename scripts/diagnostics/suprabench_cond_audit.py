#!/usr/bin/env python
"""SUPRABENCH_COND_AUDIT — does the double-cold interaction signal survive controlling for
solvent/temp/ph/source? CPU only. Reuses the saved clean-subset folds.
Historical execution used scikit-learn 1.7.2; the fold-consistency tolerance is 0.06 to absorb
the scikit-learn 1.7.2-vs-1.9.0 RF difference (splits confirmed, not bit-identical RF)."""
import argparse
import json, hashlib, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

REPO_ROOT=Path(__file__).resolve().parents[2]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cond',type=Path,default=REPO_ROOT/'data'/'clean'/'suprabench_bap_clean_cond.csv',help='Condition-annotated clean CSV.')
    p.add_argument('--base',type=Path,default=REPO_ROOT/'data'/'clean'/'suprabench_bap_clean.csv',help='Base clean CSV used for row-identity verification.')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/'diagnostics'/'suprabench_cond_audit_results.json',help='Output JSON path.')
    return p.parse_args()
ARGS=parse_args(); COND=ARGS.cond; BASE=ARGS.base; OUTPUT=ARGS.output
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
REFERENCE_PAIRRF={'random':0.677,'host_cold':0.448,'guest_cold':0.641,'double_cold':0.386}  # saved sklearn 1.9.0 reference
TOL=0.06  # relaxed: confirms SPLITS (deterministic) absorbing sklearn 1.7.2-vs-1.9.0 RF diff
RNG=np.random.RandomState(0)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
df=pd.read_csv(COND); base=pd.read_csv(BASE)
if not ((df['host_smiles'].values==base['host_smiles'].values).all() and (df['guest_smiles'].values==base['guest_smiles'].values).all() and np.allclose(df['y'].values,base['y'].values)):
    print("JOIN_FAIL: cond CSV row order != clean CSV"); sys.exit(1)
y=df['y'].values.astype(np.float32)
hid=pd.factorize(df['host_smiles'])[0]; gid=pd.factorize(df['guest_smiles'])[0]; fam=df['family'].values
uhost=pd.factorize(df['host_smiles'])[1]; uguest=pd.factorize(df['guest_smiles'])[1]
print(f"rows={len(df)} hosts={len(uhost)} guests={len(uguest)} JOIN_OK sklearn={sklearn.__version__} cond_sha={sha256(COND)[:12]}",flush=True)

DESC=[('MW',Descriptors.MolWt),('LogP',Descriptors.MolLogP),('TPSA',Descriptors.TPSA),('HBD',Descriptors.NumHDonors),
 ('HBA',Descriptors.NumHAcceptors),('RotB',Descriptors.NumRotatableBonds),('Ring',rdMolDescriptors.CalcNumRings),
 ('ArRing',rdMolDescriptors.CalcNumAromaticRings),('FrCSP3',Descriptors.FractionCSP3),('Heavy',lambda m:m.GetNumHeavyAtoms()),('Charge',Chem.GetFormalCharge)]
def dvec(s):
    m=Chem.MolFromSmiles(s);o=[]
    for _,fn in DESC:
        try:o.append(float(fn(m)))
        except:o.append(0.0)
    return np.array(o,np.float32)
uh={s:dvec(s) for s in df['host_smiles'].unique()}; ug={s:dvec(s) for s in df['guest_smiles'].unique()}
H=np.vstack([uh[s] for s in df['host_smiles']]); G=np.vstack([ug[s] for s in df['guest_smiles']])
def sdiv(a,b):return a/np.where(np.abs(b)<1e-6,1e-6,b)
P=np.column_stack([H[:,0]+G[:,0],sdiv(G[:,0],H[:,0]),H[:,9]-G[:,9],sdiv(G[:,9],H[:,9]),H[:,10]*G[:,10],H[:,10]+G[:,10],
   np.abs(H[:,2]-G[:,2]),H[:,3]+G[:,4],H[:,4]+G[:,3],H[:,1]+G[:,1]]).astype(np.float32)
X_pair=np.hstack([H,G,P]); X_host=H; X_guest=G
COND_OH=pd.get_dummies(df[['source','solvent_bucket','temp_bucket','ph_bucket']].astype(str)).values.astype(np.float32)
X_pair_cond=np.hstack([X_pair,COND_OH])
print("features built",flush=True)
def fp(s):return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024)
hfps=[fp(s) for s in uhost]; gfps=[fp(s) for s in uguest]
HostSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,hfps),np.float32) for f in hfps])
GuestSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,gfps),np.float32) for f in gfps])
print("sim matrices built",flush=True)

def metrics(yt,yp):
    if len(np.unique(yt))<2 or np.std(yp)<1e-9:return dict(R2=float('nan'),Spearman=0.0,MAE=float(mean_absolute_error(yt,yp)))
    sc=spearmanr(yt,yp).correlation;return dict(R2=float(r2_score(yt,yp)),Spearman=float(0 if np.isnan(sc) else sc),MAE=float(mean_absolute_error(yt,yp)))
def rf(X,tr,te):return RandomForestRegressor(300,n_jobs=-1,random_state=0,max_features='sqrt').fit(X[tr],y[tr]).predict(X[te])
def simknn(tr,te,k=5):
    htr=hid[tr];gtr=gid[tr];ytr=y[tr];pr=np.zeros(len(te));ms=np.zeros(len(te))
    for ii,i in enumerate(te):
        comb=(HostSim[hid[i],htr]+GuestSim[gid[i],gtr])/2.0
        idx=np.argpartition(-comb,min(k,len(comb)-1))[:k];w=comb[idx]+1e-6
        pr[ii]=np.sum(w*ytr[idx])/np.sum(w);ms[ii]=comb[idx].max()
    return pr,ms
def additive(tr,te):
    g=y[tr].mean();return rf(X_host,tr,te)+rf(X_guest,tr,te)-g

R={}; t0=time.time()

print("\n=== FOLD-CONSISTENCY (relaxed tol, splits confirmed) ===",flush=True)
fn={'random':list(KFold(5,shuffle=True,random_state=0).split(np.arange(len(y)))),
    'host_cold':list(GroupKFold(5).split(np.arange(len(y)),y,hid)),
    'guest_cold':list(GroupKFold(5).split(np.arange(len(y)),y,gid))}
ok=True; R['fold_consistency']={}
for sp,fo in fn.items():
    yy,pp=[],[]
    for tr,te in fo:yy.append(y[te]);pp.append(rf(X_pair,tr,te))
    r2=float(r2_score(np.concatenate(yy),np.concatenate(pp)));R['fold_consistency'][sp]=round(r2,3);ok&=abs(r2-REFERENCE_PAIRRF[sp])<TOL
    print(f"  {sp:11s} PAIR_RF {r2:.3f} (saved reference {REFERENCE_PAIRRF[sp]}, |d|={abs(r2-REFERENCE_PAIRRF[sp]):.3f})",flush=True)
def dc_folds(rows=None):
    idx=np.arange(len(y)) if rows is None else np.array(rows)
    h=hid[idx];g=gid[idx];uh_=np.unique(h);ug_=np.unique(g);out=[]
    for seed in range(12):
        rs=np.random.RandomState(seed)
        hte=set(rs.choice(uh_,max(1,int(0.25*len(uh_))),False));gte=set(rs.choice(ug_,max(1,int(0.25*len(ug_))),False))
        te=idx[[ (h[k] in hte and g[k] in gte) for k in range(len(idx))]]
        tr=idx[[ (h[k] not in hte and g[k] not in gte) for k in range(len(idx))]]
        if len(te)<25 or len(np.unique(y[te]))<2 or len(tr)<50:continue
        out.append((tr,te))
    return out
dcf=dc_folds(); dcr=[metrics(y[te],rf(X_pair,tr,te))['R2'] for tr,te in dcf]; r2dc=float(np.mean(dcr))
R['fold_consistency']['double_cold']=round(r2dc,3); ok&=abs(r2dc-REFERENCE_PAIRRF['double_cold'])<TOL
print(f"  double_cold PAIR_RF {r2dc:.3f} (saved reference {REFERENCE_PAIRRF['double_cold']}) nfolds={len(dcf)}",flush=True)
if not ok: print("FOLD_MISMATCH (>0.06) -> abort"); json.dump(R,open(OUTPUT,"w"),indent=2,default=int); sys.exit(2)
print("FOLD CONSISTENCY OK (splits confirmed)",flush=True)

print("\n=== P1 per-family condition distribution ===",flush=True)
R['P1']={'source':{k:int(v) for k,v in df['source'].value_counts().items()},'solvent':{k:int(v) for k,v in df['solvent_bucket'].value_counts().items()},
         'temp':{k:int(v) for k,v in df['temp_bucket'].value_counts().items()},'ph':{k:int(v) for k,v in df['ph_bucket'].value_counts().items()},
         'per_family_solvent':{f:{kk:int(vv) for kk,vv in sub['solvent_bucket'].value_counts().items()} for f,sub in df.groupby('family')},
         'per_family_source':{f:{kk:int(vv) for kk,vv in sub['source'].value_counts().items()} for f,sub in df.groupby('family')}}

print("=== P2 sigma decomposition ===",flush=True)
g_hg=df.groupby(['host_smiles','guest_smiles']); g_cond=df.groupby(['host_smiles','guest_smiles','solvent_bucket','temp_bucket','ph_bucket'])
same_hg=[s['y'].std() for _,s in g_hg if len(s)>1]; true_rep=[s['y'].std() for _,s in g_cond if len(s)>1]
cross=[s['y'].std() for _,s in g_hg if len(s)>1 and s[['solvent_bucket','temp_bucket','ph_bucket']].astype(str).agg('|'.join,1).nunique()>1]
R['P2']={'same_hg_sigma_mean':round(float(np.nanmean(same_hg)),4),'n_same_hg':len(same_hg),
         'true_replicate_sigma_mean':round(float(np.nanmean(true_rep)),4),'n_true_rep':len(true_rep),
         'cross_condition_spread_mean':round(float(np.nanmean(cross)),4) if cross else None,'n_cross_cond':len(cross)}
print("  P2:",R['P2'],flush=True)

print("=== P3 source-disjoint ===",flush=True)
src=df['source'].values; eup=np.where(src=='eupmc')[0]; cb7=np.where(src=='cb7_supplement')[0]
R['P3']={'note':'cb7_supplement = single host (CB7); test=cb7 is curation-shift for ONE host (CB7 also in eupmc train); reverse=single-host-train (degenerate)'}
for name,tr,te in [('train_eupmc_test_cb7',eup,cb7),('train_cb7_test_eupmc',cb7,eup)]:
    s1,_=simknn(tr,te,5); pr=rf(X_pair,tr,te)
    R['P3'][name]={'n_train':int(len(tr)),'n_test':int(len(te)),'PAIR_RF':round(metrics(y[te],pr)['R2'],3),
        'simknn_k5':round(metrics(y[te],s1)['R2'],3),'additive':round(metrics(y[te],additive(tr,te))['R2'],3),
        'PAIR_RF_MAE':round(metrics(y[te],pr)['MAE'],3),'PAIR_RF_rho':round(metrics(y[te],pr)['Spearman'],3)}
    print(f"  {name}: {R['P3'][name]}",flush=True)

print("=== P4 condition-controlled ===",flush=True)
condnull=[];paircond=[];paironly=[]
for tr,te in dcf:
    condnull.append(metrics(y[te],rf(COND_OH,tr,te))['R2']); paircond.append(metrics(y[te],rf(X_pair_cond,tr,te))['R2']); paironly.append(metrics(y[te],rf(X_pair,tr,te))['R2'])
R['P4_condition_null_double_cold']={'cond_only_RF_R2':round(float(np.nanmean(condnull)),3),'cond_only_sd':round(float(np.nanstd(condnull)),3),
  'PAIR_RF_R2':round(float(np.nanmean(paironly)),3),'PAIR_plus_cond_R2':round(float(np.nanmean(paircond)),3),
  'structure_gain_over_conditions':round(float(np.nanmean(paironly))-float(np.nanmean(condnull)),3)}
print("  P4a:",R['P4_condition_null_double_cold'],flush=True)
R['P4_within_solvent']={}
for sv in ['buffer','water','complex']:
    rows=np.where(df['solvent_bucket'].values==sv)[0]; fo=dc_folds(rows)
    if len(fo)<5:
        R['P4_within_solvent'][sv]={'n_rows':int(len(rows)),'n_folds':len(fo),'flag':'NO_CONCLUSION_small'};print(f"  within {sv}: NO_CONCLUSION (folds={len(fo)})",flush=True);continue
    pr=[];sk=[];nte=[]
    for tr,te in fo:
        pr.append(metrics(y[te],rf(X_pair,tr,te))['R2']); s5,_=simknn(tr,te,5); sk.append(metrics(y[te],s5)['R2']);nte.append(len(te))
    R['P4_within_solvent'][sv]={'n_rows':int(len(rows)),'n_folds':len(fo),'mean_test_n':int(np.mean(nte)),
        'PAIR_RF_R2':round(float(np.nanmean(pr)),3),'PAIR_RF_sd':round(float(np.nanstd(pr)),3),'simknn_k5_R2':round(float(np.nanmean(sk)),3)}
    print(f"  within {sv}: {R['P4_within_solvent'][sv]}",flush=True)

json.dump(R,open(OUTPUT,"w"),indent=2,default=int)
print(f"\nWROTE {OUTPUT}  total {time.time()-t0:.0f}s",flush=True)
