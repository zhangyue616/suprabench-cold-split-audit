"""Scientific GNN training functions used by the public launcher.

Runtime paths and the released source hash are injected only after the public
verification manifest has been checked.
"""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, AttentiveFP, global_mean_pool

OUT = None
SOURCE = None
SOURCE_SHA = None
SEEDS = list(range(5))
MODELS = ['GINE', 'AttentiveFP']
REGIMES = ['double_cold', 'random', 'host_cold', 'guest_cold', 'family_cold']
NULLS = ['sim_knn_k5', 'additive', 'host_only_RF', 'guest_only_RF', 'cond_only_RF', 'source_only_RF']
MAXEP, PAT = 300, 30


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def score(yt, yp):
    r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 and np.std(yp) > 1e-9 else None
    rho = float(spearmanr(yt, yp).statistic) if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1 else None
    return plain({'R2': r2, 'MAE': float(mean_absolute_error(yt, yp)), 'Spearman': rho})


def source_namespace():
    assert digest(SOURCE) == SOURCE_SHA, 'SOURCE_SHA_MISMATCH'
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    names = {'ELEMS', 'BT', 'FDIM', 'EDIM', 'GNN', 'onehot', 'atomf', 'bondf',
             'to_data', 'GINEEnc', 'AFPEnc', 'PairNet', 'mk_gine', 'mk_afp', 'train_gnn', 'dcfolds'}
    nodes, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            selected = {node.name} & names
        elif isinstance(node, ast.Assign):
            selected = {t.id for t in node.targets if isinstance(t, ast.Name)} & names
        else:
            selected = set()
        if selected:
            nodes.append(node)
            found.update(selected)
    assert found == names, ('SOURCE_EXPORT_MISSING', sorted(names - found))
    ns = dict(np=np, torch=torch, nn=nn, Chem=Chem, Data=Data, Batch=Batch,
              GINEConv=GINEConv, AttentiveFP=AttentiveFP, global_mean_pool=global_mean_pool,
              MAXEP=MAXEP, PAT=PAT)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


def environment():
    import platform, torch_geometric, sklearn, scipy, rdkit
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, pyg=torch_geometric.__version__, numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                rdkit=rdkit.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def checkpoint_selftest():
    # Essential resume test, before any real data. Dropout and Adam continue identically on CPU.
    torch.manual_seed(13)
    m = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.1), nn.Linear(4, 1))
    o = torch.optim.Adam(m.parameters(), lr=.001, weight_decay=1e-5)
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18
    def step():
        o.zero_grad(); loss = m(x).square().mean(); loss.backward(); o.step()
        return loss.detach().clone()
    step()
    import copy
    saved = copy.deepcopy((m.state_dict(), o.state_dict(), torch.get_rng_state()))
    expected = step(); weights = copy.deepcopy(m.state_dict())
    m.load_state_dict(saved[0]); o.load_state_dict(saved[1]); torch.set_rng_state(saved[2])
    actual = step()
    assert torch.equal(actual, expected) and all(torch.equal(m.state_dict()[k], v) for k, v in weights.items())
    assert score(np.array([1., 2., 3.]), np.ones(3))['R2'] is None
    assert score(np.array([1., 2., 3.]), np.array([1., 2., 3.]))['R2'] == 1.
    return {'resume_cpu_dropout_adam_equal': True, 'actual_loss': actual.item(), 'expected_loss': expected.item(),
            'constant_prediction_R2': None, 'perfect_prediction_R2': 1.}


def save_checkpoint(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_job(ns, rows, job, fold):
    target = OUT / 'jobs' / job['job_id']; target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / 'checkpoint.pt'
    y = rows.y_true.values.astype(np.float32)
    tr = np.array(fold['outer_train']); te = np.array(fold['test'])
    seed = job['seed']; torch.manual_seed(seed)
    perm = np.random.RandomState(seed).permutation(len(tr))
    nval = max(20, int(.15 * len(tr))); val = tr[perm[:nval]]; trn = tr[perm[nval:]]
    mu, sd = y[trn].mean(), y[trn].std() + 1e-6
    membership = dict(outer_train=tr.tolist(), inner_train=trn.tolist(), inner_val=val.tolist(),
                      outer_test=te.tolist(), excluded=fold['excluded'])
    if (target / 'membership.json').exists():
        assert json.loads((target / 'membership.json').read_text()) == membership
    else:
        write_json(target / 'membership.json', membership)
    make_enc, d = ns['GNN'][job['model']]
    model = ns['PairNet'](make_enc(), d).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    hi = torch.tensor(rows.host_id.values, device='cuda'); gi = torch.tensor(rows.guest_id.values, device='cuda')
    yt = torch.tensor((y - mu) / sd, device='cuda'); yval = torch.tensor(y[val], device='cuda')
    trn_t = torch.tensor(trn, device='cuda'); val_t = torch.tensor(val, device='cuda'); te_t = torch.tensor(te, device='cuda')
    best, best_state, wait, best_epoch, start, elapsed = 1e9, None, 0, None, 0, 0.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cuda', weights_only=False)
        assert state['job'] == job and state['membership'] == membership
        assert state['mu'] == float(mu) and state['sd'] == float(sd)
        assert state['runner_sha'] == digest(__file__), 'RUNNER_DRIFT_ON_RESUME'
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        best, best_state, wait, best_epoch = state['best'], state['best_state'], state['wait'], state['best_epoch']
        start, elapsed = state['epoch'] + 1, state['elapsed_s']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    logpath = target / 'epochs.csv'
    if start and logpath.exists():
        prior = pd.read_csv(logpath); prior = prior[prior.epoch < start]
        prior.to_csv(logpath, index=False)
    def paused():
        return (OUT / 'PAUSE').exists()
    ep = start - 1
    with logpath.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['epoch', 'train_mse', 'validation_mae', 'best_validation_mae', 'wait', 'elapsed_s'])
        for ep in range(start, MAXEP):
            if wait >= PAT or paused():
                break
            model.train(); opt.zero_grad()
            pred = model(hi[trn_t], gi[trn_t]); loss = ((pred - yt[trn_t]) ** 2).mean()
            loss.backward(); opt.step(); model.eval()
            with torch.no_grad():
                vp = model(hi[val_t], gi[val_t]) * sd + mu
                vmae = torch.abs(vp - yval).mean().item()
            assert np.isfinite(vmae) and torch.isfinite(loss), 'NONFINITE_TRAINING'
            if vmae < best - 1e-4:
                best = vmae; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0; best_epoch = ep
            else:
                wait += 1
            total = elapsed + time.monotonic() - started
            state = dict(job=job, membership=membership, mu=float(mu), sd=float(sd), epoch=ep,
                         best=best, best_epoch=best_epoch, wait=wait, best_state=best_state,
                         model=model.state_dict(), optimizer=opt.state_dict(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), elapsed_s=total, runner_sha=digest(__file__))
            save_checkpoint(checkpoint, state)
            writer.writerow([ep, loss.item(), vmae, best, wait, total]); f.flush()
            if ep % 25 == 0:
                print(f"{job['job_id']} epoch={ep} val_mae={vmae:.6f} best={best:.6f} wait={wait}", flush=True)
            if wait >= PAT:
                break
    if paused():
        print('PAUSED_AT_EPOCH_BOUNDARY ' + job['job_id'], flush=True)
        return False
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        predictions = (model(hi[te_t], gi[te_t]) * sd + mu).cpu().numpy()
    assert np.isfinite(predictions).all()
    output = rows.iloc[te].copy()
    for k in ['model', 'seed', 'regime', 'fold']:
        output[k] = job[k]
    output['y_pred'] = predictions.astype(float)
    output.to_csv(target / 'predictions.csv', index=False)
    result = dict(**job, status='COMPLETED', flag=fold['flag'], n_train=len(trn), n_val=len(val), n_test=len(te),
                  mu=float(mu), sd=float(sd), best_epoch=best_epoch, stop_epoch=state['epoch'], best_validation_mae=best,
                  elapsed_s=elapsed + time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  predictions_sha=digest(target / 'predictions.csv'), runner_sha=digest(__file__), metrics=score(y[te], predictions))
    write_json(target / 'result.json', plain(result))
    print(json.dumps(result), flush=True)
    del model, opt, best_state
    torch.cuda.empty_cache()
    return True
