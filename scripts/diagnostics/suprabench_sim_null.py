#!/usr/bin/env python
"""SUPRABENCH_SIM_NULL — chemical-similarity falsification of the double-cold interaction signal.
Reuses the saved clean-subset folds. P1 sim-pair-kNN null, P2 stratified, P3 additive null, P4 GNN seed-variance.
The scikit-learn 1.9.0 attribution comes from companion evidence; no per-run environment manifest survives.
The descriptor-RF values provide the fold-consistency check."""
import argparse
import json, hashlib, warnings, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import torch, torch.nn as nn

REPO_ROOT=Path(__file__).resolve().parents[2]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=REPO_ROOT/'data'/'clean'/'suprabench_bap_clean.csv',help='Clean SupraBench BAP CSV (default: data/clean/suprabench_bap_clean.csv).')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/'diagnostics'/'suprabench_sim_null_results.json',help='Output JSON path.')
    return p.parse_args()
ARGS=parse_args(); CSV=ARGS.csv; OUTPUT=ARGS.output
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
EXPECT_SHA="c9301b153733e847902ac7f0535802bc00aaa0bbe962673f53bc9a99eaaa1f28"
REFERENCE_PAIRRF={'random':0.677,'host_cold':0.448,'guest_cold':0.641,'double_cold':0.386}  # saved reference
DEV='cuda' if torch.cuda.is_available() else 'cpu'; RNG=np.random.RandomState(0); torch.manual_seed(0)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
got=sha256(CSV)
if got!=EXPECT_SHA: print(f"MAPPING_FAIL csv sha {got} != {EXPECT_SHA}"); sys.exit(1)
df=pd.read_csv(CSV); y=df['y'].values.astype(np.float32)
hid=pd.factorize(df['host_smiles'])[0]; gid=pd.factorize(df['guest_smiles'])[0]; fam=df['family'].values
uhost=pd.factorize(df['host_smiles'])[1]; uguest=pd.factorize(df['guest_smiles'])[1]
print(f"rows={len(df)} hosts={len(uhost)} guests={len(uguest)} csv_sha_ok device={DEV}")

# ---------- descriptors (PAIR_RF / host_only / guest_only) ----------
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

# ---------- ECFP Tanimoto similarity matrices (unique host/guest) ----------
def fp(s): return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024)
hfps=[fp(s) for s in uhost]; gfps=[fp(s) for s in uguest]
HostSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,hfps),np.float32) for f in hfps])  # [Nh,Nh]
GuestSim=np.vstack([np.array(DataStructs.BulkTanimotoSimilarity(f,gfps),np.float32) for f in gfps]) # [Ng,Ng]
print("sim matrices:",HostSim.shape,GuestSim.shape)

def metrics(yt,yp):
    if len(np.unique(yt))<2 or np.std(yp)<1e-9: return dict(R2=float('nan'),Spearman=0.0,MAE=float(mean_absolute_error(yt,yp)))
    sc=spearmanr(yt,yp).correlation; return dict(R2=float(r2_score(yt,yp)),Spearman=float(0 if np.isnan(sc) else sc),MAE=float(mean_absolute_error(yt,yp)))
def _rf(X,tr,te):return RandomForestRegressor(300,n_jobs=-1,random_state=0,max_features='sqrt').fit(X[tr],y[tr]).predict(X[te])

def simknn(tr,te,k=1,combine='mean'):
    """for each test pair: combined host+guest Tanimoto to each train pair; predict sim-weighted mean of top-k train logKa.
       returns preds, max_combined_sim per test pair."""
    htr=hid[tr]; gtr=gid[tr]; ytr=y[tr]; preds=np.zeros(len(te)); maxsim=np.zeros(len(te))
    for ii,i in enumerate(te):
        hs=HostSim[hid[i],htr]; gs=GuestSim[gid[i],gtr]
        comb=(hs+gs)/2.0 if combine=='mean' else np.minimum(hs,gs)
        if k==1:
            j=int(np.argmax(comb)); preds[ii]=ytr[j]; maxsim[ii]=comb[j]
        else:
            idx=np.argpartition(-comb,min(k,len(comb)-1))[:k]; w=comb[idx]+1e-6
            preds[ii]=np.sum(w*ytr[idx])/np.sum(w); maxsim[ii]=comb[idx].max()
    return preds, maxsim

# ---------- folds (exact saved-reference logic) ----------
def folds_named():
    return {'random':list(KFold(5,shuffle=True,random_state=0).split(np.arange(len(y)))),
            'host_cold':list(GroupKFold(5).split(np.arange(len(y)),y,hid)),
            'guest_cold':list(GroupKFold(5).split(np.arange(len(y)),y,gid))}
def dc_folds():
    uh_ids=np.unique(hid);ug_ids=np.unique(gid);out=[]
    for seed in range(12):
        rs=np.random.RandomState(seed)
        hte=set(rs.choice(uh_ids,int(0.25*len(uh_ids)),False));gte=set(rs.choice(ug_ids,int(0.25*len(ug_ids)),False))
        te=np.array([i for i in range(len(y)) if hid[i] in hte and gid[i] in gte])
        tr=np.array([i for i in range(len(y)) if hid[i] not in hte and gid[i] not in gte])
        if len(te)<25 or len(np.unique(y[te]))<2: continue
        out.append((tr,te))
    return out

R={}
t0=time.time()

# ---------- FOLD-CONSISTENCY CHECK ----------
print("\n=== FOLD-CONSISTENCY (recompute PAIR_RF against saved reference) ===")
fc_ok=True; R['fold_consistency']={}
for sp,folds in folds_named().items():
    yy,pp=[],[]
    for tr,te in folds: yy.append(y[te]);pp.append(_rf(X_pair,tr,te))
    r2=r2_score(np.concatenate(yy),np.concatenate(pp)); R['fold_consistency'][sp]=round(float(r2),3)
    ok=abs(r2-REFERENCE_PAIRRF[sp])<0.01; fc_ok&=ok
    print(f"  {sp:11s} PAIR_RF R2={r2:.3f} reference={REFERENCE_PAIRRF[sp]} {'OK' if ok else 'MISMATCH'}")
dcf=dc_folds(); dc_r2=[metrics(y[te],_rf(X_pair,tr,te))['R2'] for tr,te in dcf]
r2dc=float(np.mean(dc_r2)); R['fold_consistency']['double_cold']=round(r2dc,3)
ok=abs(r2dc-REFERENCE_PAIRRF['double_cold'])<0.02; fc_ok&=ok
print(f"  double_cold PAIR_RF R2={r2dc:.3f} reference={REFERENCE_PAIRRF['double_cold']} {'OK' if ok else 'MISMATCH'}  (nfolds={len(dcf)})")
if not fc_ok: print("FOLD_MISMATCH -> abort"); json.dump(R,open(OUTPUT,"w"),indent=2); sys.exit(2)
print("FOLD CONSISTENCY: OK")

def hci(yt,yp,ht,reps=400):
    Hh=np.unique(ht);r2=[]
    for _ in range(reps):
        sel=RNG.choice(Hh,len(Hh),True);m=np.concatenate([np.where(ht==h)[0] for h in sel])
        if len(np.unique(yt[m]))>1: r2.append(r2_score(yt[m],yp[m]))
    return [round(float(np.percentile(r2,2.5)),3),round(float(np.percentile(r2,97.5)),3)] if r2 else [None,None]

# ---------- P1 + P3: OOF splits ----------
print("\n=== P1 sim-kNN null + P3 additive null (random/host/guest OOF) ===")
for sp,folds in folds_named().items():
    acc={m:{'y':[],'p':[],'h':[]} for m in ['PAIR_RF','additive_sum','simknn_k1','simknn_k3','simknn_k5','host_only_RF','guest_only_RF']}
    for tr,te in folds:
        g=y[tr].mean()
        hp=_rf(X_host,tr,te); gp=_rf(X_guest,tr,te); pr=_rf(X_pair,tr,te)
        add=hp+gp-g
        s1,_=simknn(tr,te,1); s3,_=simknn(tr,te,3); s5,_=simknn(tr,te,5)
        for m,p in [('PAIR_RF',pr),('additive_sum',add),('simknn_k1',s1),('simknn_k3',s3),('simknn_k5',s5),('host_only_RF',hp),('guest_only_RF',gp)]:
            acc[m]['y'].append(y[te]);acc[m]['p'].append(p);acc[m]['h'].append(hid[te])
    R[sp]={}
    for m in acc:
        yy,pp,hh=np.concatenate(acc[m]['y']),np.concatenate(acc[m]['p']),np.concatenate(acc[m]['h'])
        mm=metrics(yy,pp); mm['R2_CI']=hci(yy,pp,hh); R[sp][m]={k:(round(v,3) if isinstance(v,float) else v) for k,v in mm.items()}
        print(f"  {sp:11s} {m:14s} R2={R[sp][m]['R2']} CI={mm['R2_CI']} rho={R[sp][m]['Spearman']} MAE={R[sp][m]['MAE']}")
    R[sp]['interaction_gap_PAIRRF_minus_additive']=round(R[sp]['PAIR_RF']['R2']-R[sp]['additive_sum']['R2'],3)

# ---------- P1 double-cold + P2 stratified ----------
print("\n=== P1 double-cold (12 seeds) + P2 stratification ===")
dc={m:[] for m in ['PAIR_RF','additive_sum','simknn_k1','simknn_k3','simknn_k5','host_only_RF','guest_only_RF']}
strat={b:{'PAIR_RF':{'y':[],'p':[]},'simknn_k1':{'y':[],'p':[]},'n':0} for b in ['<0.4','0.4-0.7','>0.7']}
for tr,te in dcf:
    g=y[tr].mean(); hp=_rf(X_host,tr,te); gp=_rf(X_guest,tr,te); pr=_rf(X_pair,tr,te); add=hp+gp-g
    s1,ms=simknn(tr,te,1); s3,_=simknn(tr,te,3); s5,_=simknn(tr,te,5)
    for m,p in [('PAIR_RF',pr),('additive_sum',add),('simknn_k1',s1),('simknn_k3',s3),('simknn_k5',s5),('host_only_RF',hp),('guest_only_RF',gp)]:
        dc[m].append(metrics(y[te],p)['R2'])
    for ii,i in enumerate(te):
        b='<0.4' if ms[ii]<0.4 else ('0.4-0.7' if ms[ii]<=0.7 else '>0.7')
        strat[b]['PAIR_RF']['y'].append(y[i]);strat[b]['PAIR_RF']['p'].append(pr[ii])
        strat[b]['simknn_k1']['y'].append(y[i]);strat[b]['simknn_k1']['p'].append(s1[ii]);strat[b]['n']+=1
R['double_cold']={'n_folds':len(dcf),'mean_test_n':int(np.mean([len(te) for _,te in dcf]))}
for m in dc:
    a=np.array([x for x in dc[m] if not np.isnan(x)])
    R['double_cold'][m]={'R2_mean':round(float(a.mean()),3),'R2_std':round(float(a.std()),3),'n_valid':int(len(a))} if len(a) else {'R2_mean':None,'note':'DEGEN'}
    print(f"  double_cold {m:14s} {R['double_cold'][m]}")
R['double_cold']['interaction_gap_PAIRRF_minus_additive']=round(R['double_cold']['PAIR_RF']['R2_mean']-R['double_cold']['additive_sum']['R2_mean'],3)
R['stratified_double_cold']={}
for b in strat:
    yy=np.array(strat[b]['PAIR_RF']['y']);
    if len(yy)<10: R['stratified_double_cold'][b]={'n':strat[b]['n'],'flag':'NO_CONCLUSION_small'}; print(f"  strat {b:8s} n={strat[b]['n']} NO_CONCLUSION"); continue
    pr_=np.array(strat[b]['PAIR_RF']['p']); sk_=np.array(strat[b]['simknn_k1']['p'])
    R['stratified_double_cold'][b]={'n':int(len(yy)),'PAIR_RF_R2':round(float(r2_score(yy,pr_)),3),'simknn_k1_R2':round(float(r2_score(yy,sk_)),3),
                                    'PAIR_RF_MAE':round(float(mean_absolute_error(yy,pr_)),3),'simknn_k1_MAE':round(float(mean_absolute_error(yy,sk_)),3)}
    print(f"  strat {b:8s} n={len(yy):4d} PAIR_RF R2={R['stratified_double_cold'][b]['PAIR_RF_R2']} simknn R2={R['stratified_double_cold'][b]['simknn_k1_R2']}")

# ---------- P4: GNN seed-variance on double-cold ----------
print("\n=== P4 GNN seed-variance (double-cold, 5 model seeds x folds) ===")
ELEMS=[6,7,8,16,15,9,17,35,53,11,5,14,1,3,19,20,30,64]
def onehot(x,al):
    v=[0.0]*(len(al)+1); v[al.index(x) if x in al else -1]=1.0; return v
def atomf(a):
    return (onehot(a.GetAtomicNum(),ELEMS)+onehot(min(a.GetDegree(),6),list(range(7)))+onehot(int(np.clip(a.GetFormalCharge(),-2,2)),[-2,-1,0,1,2])
            +onehot(str(a.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'])+[float(a.GetIsAromatic()),float(a.IsInRing())]+onehot(min(a.GetTotalNumHs(),4),list(range(5))))
def graph(s):
    m=Chem.MolFromSmiles(s); nf=np.array([atomf(a) for a in m.GetAtoms()],np.float32); ei=[[],[]]
    for b in m.GetBonds(): i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei[0]+=[i,j]; ei[1]+=[j,i]
    return nf,(np.array(ei,np.int64) if ei[0] else np.zeros((2,0),np.int64))
FDIM=len(atomf(Chem.MolFromSmiles('CCO').GetAtomWithIdx(0)))
def bunique(sl):
    nfs=[];eis=[];bv=[];off=0
    for gi,s in enumerate(sl):
        nf,ei=graph(s);nfs.append(nf);eis.append(ei+off);bv+=[gi]*len(nf);off+=len(nf)
    return (torch.tensor(np.vstack(nfs),dtype=torch.float32,device=DEV),torch.tensor(np.hstack(eis),dtype=torch.long,device=DEV),
            torch.tensor(bv,dtype=torch.long,device=DEV),len(sl))
HB=bunique(list(uhost));GB=bunique(list(uguest))
class GIN(nn.Module):
    def __init__(s,f,d=128,L=3,drop=0.1):
        super().__init__();s.l0=nn.Linear(f,d);s.ls=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.BatchNorm1d(d),nn.ReLU(),nn.Linear(d,d)) for _ in range(L)]);s.eps=nn.Parameter(torch.zeros(L));s.dp=nn.Dropout(drop)
    def forward(s,nf,ei,bv,ng):
        h=s.l0(nf)
        for k,l in enumerate(s.ls):
            ag=torch.zeros_like(h)
            if ei.size(1)>0: ag.index_add_(0,ei[1],h[ei[0]])
            h=s.dp(l((1+s.eps[k])*h+ag))
        po=torch.zeros(ng,h.size(1),device=h.device);po.index_add_(0,bv,h)
        cn=torch.zeros(ng,device=h.device).index_add_(0,bv,torch.ones(h.size(0),device=h.device));return po/cn.clamp(min=1).unsqueeze(1)
class PG(nn.Module):
    def __init__(s,f,d=128):super().__init__();s.enc=GIN(f,d);s.head=nn.Sequential(nn.Linear(4*d,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,1))
    def emb(s):return s.enc(*HB),s.enc(*GB)
    def forward(s,hi,gi,he,ge):a=he[hi];b=ge[gi];return s.head(torch.cat([a,b,a*b,(a-b).abs()],1)).squeeze(1)
def train_gnn(tr,te,seed):
    torch.manual_seed(seed);tr=np.array(tr);rs=np.random.RandomState(seed);pm=rs.permutation(len(tr));nv=max(30,int(0.15*len(tr)))
    val=tr[pm[:nv]];trn=tr[pm[nv:]];mu,sd=y[trn].mean(),y[trn].std()+1e-6
    M=PG(FDIM).to(DEV);opt=torch.optim.Adam(M.parameters(),lr=1e-3,weight_decay=1e-5)
    hi=torch.tensor(hid,device=DEV);gi=torch.tensor(gid,device=DEV);yt=torch.tensor((y-mu)/sd,device=DEV);best=1e9;bs=None;w=0
    for ep in range(300):
        M.train();opt.zero_grad();he,ge=M.emb();pred=M(hi[trn],gi[trn],he,ge);loss=((pred-yt[trn])**2).mean();loss.backward();opt.step()
        M.eval()
        with torch.no_grad():he,ge=M.emb();vp=M(hi[val],gi[val],he,ge)*sd+mu;vm=torch.abs(vp-torch.tensor(y[val],device=DEV)).mean().item()
        if vm<best-1e-4:best=vm;bs={k:v.detach().clone() for k,v in M.state_dict().items()};w=0
        else:w+=1
        if w>=30:break
    M.load_state_dict(bs);M.eval()
    with torch.no_grad():he,ge=M.emb();pr=(M(hi[torch.tensor(te,device=DEV)],gi[torch.tensor(te,device=DEV)],he,ge)*sd+mu).cpu().numpy()
    return pr
gnn_seed_means=[]
for s in range(5):
    perfold=[metrics(y[te],train_gnn(tr,te,s))['R2'] for tr,te in dcf]
    gnn_seed_means.append(float(np.nanmean(perfold)))
    print(f"  GNN model-seed {s}: double-cold meanR2={gnn_seed_means[-1]:.3f}")
gm=np.array(gnn_seed_means)
R['gnn_double_cold_seedvar']={'per_model_seed_mean':[round(x,3) for x in gnn_seed_means],'mean':round(float(gm.mean()),3),'sd':round(float(gm.std()),3),
                              'note':'each value = mean over 12 double-cold split-seeds at one GNN init/val seed; saved single-seed reference was 0.258'}
print("  GNN double-cold across 5 model seeds: mean=%.3f sd=%.3f"%(gm.mean(),gm.std()))

json.dump(R,open(OUTPUT,"w"),indent=2)
print(f"\nWROTE {OUTPUT}  total {time.time()-t0:.0f}s")
