#!/usr/bin/env python
"""SUPRABENCH_PREGATE — block-level significance (K2), low-sim-bin composition, family-cold decomposition (K4).
CPU only. Reuses the saved clean-subset folds. Historical execution used scikit-learn 1.7.2."""
import argparse
import json, hashlib, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr, wilcoxon, linregress

REPO_ROOT=Path(__file__).resolve().parents[2]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cond',type=Path,default=REPO_ROOT/'data'/'clean'/'suprabench_bap_clean_cond.csv',help='Condition-annotated clean CSV.')
    p.add_argument('--base',type=Path,default=REPO_ROOT/'data'/'clean'/'suprabench_bap_clean.csv',help='Base clean CSV used for row-identity verification.')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/'diagnostics'/'suprabench_pregate_results.json',help='Output JSON path.')
    return p.parse_args()
ARGS=parse_args(); COND=ARGS.cond; BASE=ARGS.base; OUTPUT=ARGS.output
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
SMOKE={'random':0.677,'host_cold':0.448,'guest_cold':0.641,'double_cold':0.386}; TOL=0.06
BRNG=np.random.RandomState(0)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
df=pd.read_csv(COND); base=pd.read_csv(BASE)
if not ((df['host_smiles'].values==base['host_smiles'].values).all() and (df['guest_smiles'].values==base['guest_smiles'].values).all() and np.allclose(df['y'].values,base['y'].values)):
    print("JOIN_FAIL"); sys.exit(1)
y=df['y'].values.astype(np.float32)
hid=pd.factorize(df['host_smiles'])[0]; gid=pd.factorize(df['guest_smiles'])[0]; fam=df['family'].values
hname=df['host_name'].values; srcv=df['source'].values; solv=df['solvent_bucket'].values
uhost=pd.factorize(df['host_smiles'])[1]; uguest=pd.factorize(df['guest_smiles'])[1]
print(f"rows={len(df)} hosts={len(uhost)} guests={len(uguest)} JOIN_OK sklearn={sklearn.__version__} sha={sha256(COND)[:12]}",flush=True)

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
def ecfp(s):return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024),np.float32)
he={s:ecfp(s) for s in df['host_smiles'].unique()};ge={s:ecfp(s) for s in df['guest_smiles'].unique()}
X_ecfp=np.hstack([np.vstack([he[s] for s in df['host_smiles']]),np.vstack([ge[s] for s in df['guest_smiles']])])
COND_OH=pd.get_dummies(df[['source','solvent_bucket','temp_bucket','ph_bucket']].astype(str)).values.astype(np.float32)
SRC_OH=pd.get_dummies(df['source'].astype(str)).values.astype(np.float32)
hfps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024) for s in uhost]
gfps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024) for s in uguest]
HostSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,hfps),np.float32) for f in hfps])
GuestSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,gfps),np.float32) for f in gfps])
print("features+sim built",flush=True)

def R2(yt,yp): return float(r2_score(yt,yp)) if (len(np.unique(yt))>1 and np.std(yp)>1e-9) else float('nan')
def rf(X,tr,te):return RandomForestRegressor(300,n_jobs=-1,random_state=0,max_features='sqrt').fit(X[tr],y[tr]).predict(X[te])
def simknn(tr,te,k=5):
    htr=hid[tr];gtr=gid[tr];ytr=y[tr];pr=np.zeros(len(te));ms=np.zeros(len(te))
    for ii,i in enumerate(te):
        comb=(HostSim[hid[i],htr]+GuestSim[gid[i],gtr])/2.0
        idx=np.argpartition(-comb,min(k,len(comb)-1))[:k];w=comb[idx]+1e-6
        pr[ii]=np.sum(w*ytr[idx])/np.sum(w);ms[ii]=comb[idx].max()
    return pr,ms

R={}; t0=time.time()
# FOLD CONSISTENCY
fn={'random':list(KFold(5,shuffle=True,random_state=0).split(np.arange(len(y)))),
    'host_cold':list(GroupKFold(5).split(np.arange(len(y)),y,hid)),'guest_cold':list(GroupKFold(5).split(np.arange(len(y)),y,gid))}
ok=True;R['fold_consistency']={}
for sp,fo in fn.items():
    yy,pp=[],[]
    for tr,te in fo:yy.append(y[te]);pp.append(rf(X_pair,tr,te))
    r2=float(r2_score(np.concatenate(yy),np.concatenate(pp)));R['fold_consistency'][sp]=round(r2,3);ok&=abs(r2-SMOKE[sp])<TOL
def dcfolds(rows=None):
    idx=np.arange(len(y)) if rows is None else np.array(rows);h=hid[idx];g=gid[idx];uh_=np.unique(h);ug_=np.unique(g);out=[]
    for seed in range(12):
        rs=np.random.RandomState(seed);hte=set(rs.choice(uh_,max(1,int(0.25*len(uh_))),False));gte=set(rs.choice(ug_,max(1,int(0.25*len(ug_))),False))
        te=idx[[(h[k] in hte and g[k] in gte) for k in range(len(idx))]];tr=idx[[(h[k] not in hte and g[k] not in gte) for k in range(len(idx))]]
        if len(te)<25 or len(np.unique(y[te]))<2 or len(tr)<50:continue
        out.append((tr,te))
    return out
dcf=dcfolds();r2dc=float(np.mean([R2(y[te],rf(X_pair,tr,te)) for tr,te in dcf]));R['fold_consistency']['double_cold']=round(r2dc,3);ok&=abs(r2dc-SMOKE['double_cold'])<TOL
print("fold_consistency",R['fold_consistency'],"OK" if ok else "MISMATCH",flush=True)
if not ok: print("FOLD_MISMATCH");json.dump(R,open(OUTPUT,"w"),indent=2,default=int);sys.exit(2)

# ================= P1: block-level significance =================
print("\n=== P1 block-level (double-cold 12 blocks) ===",flush=True)
PROPER=['sim_knn_k5','additive','host_only_RF','guest_only_RF','cond_only_RF','source_only_RF']
blocks=[];recs=[]
for bi,(tr,te) in enumerate(dcf):
    g=y[tr].mean()
    pr={'PAIR_RF':rf(X_pair,tr,te),'ECFP_RF':rf(X_ecfp,tr,te),'host_only_RF':rf(X_host,tr,te),'guest_only_RF':rf(X_guest,tr,te),
        'cond_only_RF':rf(COND_OH,tr,te),'source_only_RF':rf(SRC_OH,tr,te)}
    pr['additive']=pr['host_only_RF']+pr['guest_only_RF']-g
    s5,ms=simknn(tr,te,5);pr['sim_knn_k5']=s5
    bs={'block':bi,'n_te':int(len(te))}
    for m,p in pr.items():bs[m]=R2(y[te],p)
    blocks.append(bs)
    for ii,i in enumerate(te):
        rec={'block':bi,'host':int(hid[i]),'family':fam[i],'source':srcv[i],'solvent':solv[i],'y':float(y[i]),'max_sim':float(ms[ii])}
        for m in pr:rec[m]=float(pr[m][ii])
        recs.append(rec)
bdf=pd.DataFrame(blocks);rdf=pd.DataFrame(recs)
bdf['best_null']=bdf[PROPER].max(axis=1);bdf['gain']=bdf['PAIR_RF']-bdf['best_null']
g=bdf['gain'].values
boot=[float(np.mean(BRNG.choice(g,len(g),True))) for _ in range(3000)]
try:wp=float(wilcoxon(bdf['PAIR_RF'],bdf['best_null']).pvalue)
except Exception:wp=None
R['P1_block_double_cold']={'n_blocks':len(g),'PAIR_RF_meanR2':round(float(bdf['PAIR_RF'].mean()),3),
  'best_proper_null_meanR2':round(float(bdf['best_null'].mean()),3),'which_null_best_per_block':[max(PROPER,key=lambda m:row[m]) for _,row in bdf.iterrows()],
  'gain_mean':round(float(g.mean()),3),'gain_block_CI':[round(np.percentile(boot,2.5),3),round(np.percentile(boot,97.5),3)],
  'n_blocks_PAIR_gt_bestnull':int((g>0).sum()),'wilcoxon_p':wp,
  'per_null_mean_gain':{m:round(float((bdf['PAIR_RF']-bdf[m]).mean()),3) for m in PROPER+['ECFP_RF']}}
print(" P1 double-cold:",{k:R['P1_block_double_cold'][k] for k in['gain_mean','gain_block_CI','n_blocks_PAIR_gt_bestnull','wilcoxon_p']},flush=True)
# low-sim<0.4 bin, host-level bootstrap
binr=rdf[rdf['max_sim']<0.4].copy()
def pooled_R2(d,col):return R2(d['y'].values,d[col].values)
hosts=binr['host'].unique()
def hostboot(d,colA,colB,reps=2000):
    gains=[]
    for _ in range(reps):
        sel=BRNG.choice(hosts,len(hosts),True);m=pd.concat([d[d['host']==h] for h in sel])
        a=pooled_R2(m,colA);b=pooled_R2(m,colB)
        if not(np.isnan(a) or np.isnan(b)):gains.append(a-b)
    return [round(float(np.percentile(gains,2.5)),3),round(float(np.percentile(gains,97.5)),3)] if gains else [None,None]
R['P1_lowsim_bin']={'n':int(len(binr)),'n_hosts':int(len(hosts)),'n_families':int(binr['family'].nunique()),
  'PAIR_R2':round(pooled_R2(binr,'PAIR_RF'),3),'sim_knn_R2':round(pooled_R2(binr,'sim_knn_k5'),3),
  'additive_R2':round(pooled_R2(binr,'additive'),3),'host_only_R2':round(pooled_R2(binr,'host_only_RF'),3),
  'gain_PAIR_minus_simknn_hostCI':hostboot(binr,'PAIR_RF','sim_knn_k5'),
  'gain_PAIR_minus_additive_hostCI':hostboot(binr,'PAIR_RF','additive')}
print(" P1 low-sim bin:",R['P1_lowsim_bin'],flush=True)

# ================= P2: low-sim bin composition =================
print("\n=== P2 low-sim bin composition ===",flush=True)
mega_kw=['cb7','cb8','cb[7','cb[8','cucurbit[7','cucurbit[8','beta-cyclodext','β-cyclodext','b-cyclodext','gamma-cyclodext']
def ismega(nm):
    n=str(nm).lower();return any(k in n for k in mega_kw)
binr['is_mega']=binr.index.map(lambda ix: ismega(hname[ix]) if ix<len(hname) else False)  # placeholder; recompute below
# recompute mega via stored host idx -> need host_name; map host idx to a representative name
hidx2name={}
for i in range(len(df)):
    hidx2name.setdefault(hid[i],hname[i])
binr['host_name']=binr['host'].map(hidx2name)
binr['is_mega']=binr['host_name'].map(ismega)
R['P2_lowsim_composition']={'n':int(len(binr)),
  'family_dist':{k:int(v) for k,v in binr['family'].value_counts().items()},
  'source_dist':{k:int(v) for k,v in binr['source'].value_counts().items()},
  'solvent_dist':{k:int(v) for k,v in binr['solvent'].value_counts().items()},
  'y_mean':round(float(binr['y'].mean()),3),'y_std':round(float(binr['y'].std()),3),'y_min':round(float(binr['y'].min()),3),'y_max':round(float(binr['y'].max()),3),
  'frac_mega_host':round(float(binr['is_mega'].mean()),3),'n_mega':int(binr['is_mega'].sum()),
  'top_hosts':[{'host_name':str(n)[:50],'n':int(c)} for n,c in binr['host_name'].value_counts().head(8).items()]}
# per-family PAIR vs sim-knn within bin (where family has >=15 rows in bin)
R['P2_lowsim_per_family']={}
for f,sub in binr.groupby('family'):
    if len(sub)<15:R['P2_lowsim_per_family'][f]={'n':int(len(sub)),'flag':'NO_CONCLUSION_small'};continue
    R['P2_lowsim_per_family'][f]={'n':int(len(sub)),'PAIR_R2':round(pooled_R2(sub,'PAIR_RF'),3),'sim_knn_R2':round(pooled_R2(sub,'sim_knn_k5'),3),'y_mean':round(float(sub['y'].mean()),3)}
print(" P2:",{k:R['P2_lowsim_composition'][k] for k in['family_dist','frac_mega_host','y_mean','y_std']},flush=True)

# ================= P3: family-cold decomposition =================
print("\n=== P3 family-cold decomposition ===",flush=True)
def calib(yt,yp):
    try:lr=linregress(yp,yt);return round(float(lr.slope),3)
    except Exception:return None
R['P3_family_cold']={}
for f in sorted(set(fam)):
    te=np.where(fam==f)[0];tr=np.where(fam!=f)[0]
    nh=len(np.unique(hid[te]));ng=len(np.unique(gid[te]));small=(nh<5 or len(te)<80)
    pr=rf(X_pair,tr,te)
    # train-test host similarity (max Tanimoto te-host to any tr-host)
    trh=np.unique(hid[tr]);teh=np.unique(hid[te]);sims=[float(HostSim[h,trh].max()) for h in teh]
    row={'n_pair':int(len(te)),'n_host':int(nh),'n_guest':int(ng),'flag':'NO_CONCLUSION_small' if small else 'ok',
      'y_mean':round(float(y[te].mean()),3),'y_std':round(float(y[te].std()),3),'y_min':round(float(y[te].min()),3),'y_max':round(float(y[te].max()),3),
      'R2':round(R2(y[te],pr),3),'MAE':round(float(mean_absolute_error(y[te],pr)),3),
      'Spearman':round(float(spearmanr(y[te],pr).correlation if len(np.unique(y[te]))>1 else 0),3),
      'Pearson':round(float(pearsonr(y[te],pr)[0]) if len(np.unique(y[te]))>1 else 0,3),
      'calib_slope_ytrue_on_ypred':calib(y[te],pr),'offset_meanYtrue_minus_meanYpred':round(float(y[te].mean()-pr.mean()),3),
      'host_sim_median':round(float(np.median(sims)),3),'host_sim_max':round(float(np.max(sims)),3),
      'cond_missing_temp':round(float((df['temp_bucket'].values[te]=='missing').mean()),3),
      'cond_missing_ph':round(float((df['ph_bucket'].values[te]=='missing').mean()),3),
      'source_frac_eupmc':round(float((srcv[te]=='eupmc').mean()),3)}
    R['P3_family_cold'][f]=row
    print(f"  {f:13s} R2={row['R2']:>7} rho={row['Spearman']:>6} Pear={row['Pearson']:>6} calib={row['calib_slope_ytrue_on_ypred']} offset={row['offset_meanYtrue_minus_meanYpred']:>6} ymean={row['y_mean']:>6} simMed={row['host_sim_median']} {row['flag']}",flush=True)

# structural-cluster family sensitivity (cluster hosts by ECFP, leave-one-cluster-out)
print("=== P3b structural-cluster sensitivity ===",flush=True)
Dh=1.0-HostSim.astype(np.float64);np.fill_diagonal(Dh,0)
ncl=8
cl=AgglomerativeClustering(n_clusters=ncl,metric='precomputed',linkage='average').fit_predict(Dh)
host_cluster=cl  # per unique-host index
row_cluster=np.array([host_cluster[hid[i]] for i in range(len(df))])
R['P3b_struct_cluster']={'n_clusters':ncl,'cluster_sizes_hosts':{int(c):int((host_cluster==c).sum()) for c in range(ncl)}}
R['P3b_struct_cluster']['leave_cluster_out']={}
for c in range(ncl):
    te=np.where(row_cluster==c)[0];tr=np.where(row_cluster!=c)[0]
    if len(te)<60 or len(np.unique(hid[te]))<3:
        R['P3b_struct_cluster']['leave_cluster_out'][int(c)]={'n_pair':int(len(te)),'n_host':int(len(np.unique(hid[te]))),'flag':'NO_CONCLUSION_small'};continue
    pr=rf(X_pair,tr,te)
    R['P3b_struct_cluster']['leave_cluster_out'][int(c)]={'n_pair':int(len(te)),'n_host':int(len(np.unique(hid[te]))),
      'R2':round(R2(y[te],pr),3),'Spearman':round(float(spearmanr(y[te],pr).correlation),3),
      'dominant_family':df['family'].values[te][0] if len(te) else None,
      'fam_mix':{k:int(v) for k,v in pd.Series(df['family'].values[te]).value_counts().head(3).items()}}
    rr=R['P3b_struct_cluster']['leave_cluster_out'][int(c)]
    print(f"  cluster {c}: n={rr['n_pair']} hosts={rr['n_host']} R2={rr['R2']} rho={rr['Spearman']} fams={rr['fam_mix']}",flush=True)

json.dump(R,open(OUTPUT,"w"),indent=2,default=int)
print(f"\nWROTE {OUTPUT} total {time.time()-t0:.0f}s",flush=True)
