"""Persist post-review diagnostics of retained family and host-cold predictions."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/family_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
GNN = ROOT / 'results/gnn_saved'
NULLS = ['sim_knn_k5', 'additive', 'host_only_RF', 'guest_only_RF', 'cond_only_RF', 'source_only_RF']
TREES = ['PAIR_RF','ECFP_RF','XGB_pair','XGB_ecfp','LGBM_pair','LGBM_ecfp','3D_pair_RF']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v


def dump(name, data):
    (OUT/name).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def metrics(y,p):
    varied = np.std(p)>1e-9
    standard=float(r2_score(y,p)) if len(np.unique(y))>1 else np.nan
    return dict(guarded_R2=standard if varied else np.nan, standard_R2=standard,
                rho=float(spearmanr(y,p).statistic) if varied else np.nan,
                slope=float(linregress(p,y).slope) if varied else np.nan,
                offset=float(np.mean(y)-np.mean(p)), MAE=float(mean_absolute_error(y,p)),
                prediction_sd=float(np.std(p)), constant_guard=not varied)


def main():
    assert not (OUT/'ANALYSIS.json').exists(), 'Refuse to overwrite completed diagnostics'
    paths=[ROOT/'data/released_predictions/predictions_family_cold.csv', ROOT/'data/folds/folds_family_cold.csv',
           GNN/'rows.csv', GNN/'folds.json', GNN/'all_predictions.csv', GNN/'job_metrics.csv', GNN/'null_scores.csv']
    read=lambda p: pd.read_csv(p,float_precision='round_trip')
    released=read(paths[0]); members=read(paths[1]); rows=read(paths[2]).set_index('row_index')
    folds=[f for f in json.loads(paths[3].read_text()) if f['regime']=='family_cold']
    gnn=read(paths[4]); prior=read(paths[5]); saved_null=read(paths[6])
    assert set(released.model)==set(TREES+NULLS)
    families=sorted(rows.family.unique()); assert len(families)==8
    def score_group(p,model,seed,origin,float32_predictions):
        assert len(p)==2383 and p.row_index.is_unique and set(p.row_index)==set(rows.index)
        aligned=rows.loc[p.row_index]
        for k in ['host_id','guest_id','y_true']:
            assert np.array_equal(p[k].to_numpy(),aligned[k].to_numpy()),(origin,model,seed,k)
        records=[]
        for family in families:
            fold=next(f for f in folds if f['fold']==family)
            z=p[p.row_index.isin(fold['test'])].set_index('row_index').loc[fold['test']]
            assert set(z.index)==set(members[members.held_family==family].row_index)
            assert (rows.loc[z.index].family==family).all()
            target=z.y_true.to_numpy().astype(np.float32)
            prediction=z.y_pred.to_numpy().astype(np.float32 if float32_predictions else np.float64)
            stat=metrics(target,prediction)
            flag='NO_CONCLUSION_small' if z.host_id.nunique()<5 or len(z)<80 else 'ok'
            assert flag==fold['flag']
            if origin=='new_GNN':
                check=prior[(prior.model==model)&(prior.seed==seed)&(prior.regime=='family_cold')&(prior.fold.astype(str)==family)].iloc[0]
                assert abs(check.R2-stat['guarded_R2'])<1e-6
            elif model in NULLS:
                check=saved_null[(saved_null.model==model)&(saved_null.regime=='family_cold')&(saved_null.fold.astype(str)==family)].iloc[0]
                assert (np.isnan(check.R2) and np.isnan(stat['guarded_R2'])) or abs(check.R2-stat['guarded_R2'])<1e-12
            records.append(dict(origin=origin,model=model,seed=seed,family=family,n_measurements=len(z),
                                n_hosts=z.host_id.nunique(),n_guests=z.guest_id.nunique(),flag=flag,**stat))
        return records
    records=[]
    for model in TREES+NULLS:
        records.extend(score_group(released[released.model==model],model,0,'released_tree_or_null',False))
    for model in ['GINE','AttentiveFP']:
        for seed in range(5):
            records.extend(score_group(gnn[(gnn.model==model)&(gnn.seed==seed)&(gnn.regime=='family_cold')],model,seed,'new_GNN',True))
    all_metrics=pd.DataFrame(records); assert len(all_metrics)==184
    all_metrics.to_csv(OUT/'family_metrics.csv',index=False)
    all_metrics[all_metrics.origin=='released_tree_or_null'].to_csv(OUT/'tree_null_family_metrics.csv',index=False)
    new=all_metrics[all_metrics.origin=='new_GNN']; new.to_csv(OUT/'gnn_family_metrics.csv',index=False)
    desc=[]
    for (model,family),group in new.groupby(['model','family']):
        assert set(group.seed)==set(range(5))
        for metric in ['guarded_R2','rho','slope','offset','MAE']:
            a=group[metric].to_numpy(); assert np.isfinite(a).all()
            desc.append(dict(model=model,family=family,metric=metric,n_seeds=5,mean=a.mean(),sample_sd=a.std(ddof=1),
                             minimum=a.min(),maximum=a.max(),flag=group.flag.iloc[0]))
    pd.DataFrame(desc).to_csv(OUT/'gnn_family_descriptive.csv',index=False)
    comparisons=[]
    for family in families:
        a=all_metrics[(all_metrics.origin=='released_tree_or_null')&(all_metrics.family==family)].set_index('model')
        n=a.loc[NULLS]; best=n.guarded_R2.idxmax(); standard_best=n.standard_R2.idxmax()
        rf=a.loc['PAIR_RF']
        comparisons.append(dict(family=family,flag=rf.flag,rf_R2=rf.guarded_R2,rf_rho=rf.rho,rf_offset=rf.offset,
            best_guarded_null=best,best_guarded_null_R2=n.loc[best,'guarded_R2'],rf_guarded_margin=rf.guarded_R2-n.loc[best,'guarded_R2'],
            best_standard_null=standard_best,best_standard_null_R2=n.loc[standard_best,'standard_R2'],
            rf_standard_margin=rf.standard_R2-n.loc[standard_best,'standard_R2'],
            guest_only_R2=n.loc['guest_only_RF','guarded_R2'],guest_only_rho=n.loc['guest_only_RF','rho'],
            source_only_guarded_R2=n.loc['source_only_RF','guarded_R2'],source_only_standard_R2=n.loc['source_only_RF','standard_R2']))
    pd.DataFrame(comparisons).to_csv(OUT/'family_null_comparisons.csv',index=False)
    # A post-review localization of the retained adverse host-cold seed, with no deletion.
    host=gnn[gnn.regime=='host_cold']; host_records=[]
    for (model,seed,fold),p in host.groupby(['model','seed','fold']):
        host_records.append(dict(model=model,seed=int(seed),fold=str(fold),n_measurements=len(p),
                            **metrics(p.y_true.to_numpy().astype(np.float32),p.y_pred.to_numpy().astype(np.float32))))
    pd.DataFrame(host_records).to_csv(OUT/'gnn_host_fold_diagnostics.csv',index=False)
    adverse=host[(host.model=='AttentiveFP')&(host.seed==4)&(host.fold.astype(str)=='0')].copy()
    assert len(adverse)==477
    adverse['squared_error']=(adverse.y_true-adverse.y_pred)**2
    contributions=adverse.groupby('host_id').agg(n_measurements=('row_index','size'),SSE=('squared_error','sum'),
                       mean_observed=('y_true','mean'),mean_predicted=('y_pred','mean')).reset_index()
    contributions['SSE_share']=contributions.SSE/contributions.SSE.sum()
    contributions.to_csv(OUT/'afp_seed4_host_fold0_by_host.csv',index=False)
    adverse.sort_values('squared_error',ascending=False).head(10).to_csv(OUT/'afp_seed4_host_fold0_top10_errors.csv',index=False)
    support=all_metrics[(all_metrics.family=='cyclodextrin')&(~all_metrics.model.isin(NULLS))]
    assert len(support)==17
    findings=dict(cyclodextrin_supervised_fits=len(support),cyclodextrin_supervised_negative_R2=int((support.guarded_R2<0).sum()),
        cyclodextrin_supervised_positive_rho=int((support.rho>0).sum()),
        cyclodextrin_afp_offsets=new[(new.model=='AttentiveFP')&(new.family=='cyclodextrin')].sort_values('seed').offset.tolist(),
        analysis_target_max=float(rows.y_true.max()),afp_adverse_host_fold=next(x for x in host_records if x['model']=='AttentiveFP' and x['seed']==4 and x['fold']=='0'))
    dump('ANALYSIS.json',dict(status='COMPLETED',source_sha={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        script_sha=sha(__file__),measurement_rows=2383,released_model_family_records=104,new_gnn_family_records=80,
        family_null_comparisons=comparisons,gnn_descriptive=desc,findings=findings))
    print(json.dumps(clean(dict(status='COMPLETED',records=184,findings=findings)),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
