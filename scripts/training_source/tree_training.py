"""Scientific tree-training functions used by the public launcher.

Runtime paths and the released source hash are injected only after the public
verification manifest has been checked. Frozen features are consumed as saved.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import platform
import sys
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

OUT = None
SOURCE = None
SOURCE_SHA = None
MODELS = ['PAIR_RF', 'ECFP_RF', 'XGB_pair', 'XGB_ecfp', 'LGBM_pair', 'LGBM_ecfp', '3D_pair_RF']
NULLS = ['sim_knn_k5', 'additive', 'host_only_RF', 'guest_only_RF', 'cond_only_RF', 'source_only_RF']
CONFIG = {'PAIR_RF': ('rf', 'X_pair'), 'ECFP_RF': ('rf', 'X_ecfp'),
          'XGB_pair': ('xgbr', 'X_pair'), 'XGB_ecfp': ('xgbr', 'X_ecfp'),
          'LGBM_pair': ('lgbr', 'X_pair'), 'LGBM_ecfp': ('lgbr', 'X_ecfp'),
          '3D_pair_RF': ('rf', 'X_pair3d')}
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_json(path, obj):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def metrics(y, p):
    standard = float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    guarded = standard if np.std(p) > 1e-9 else np.nan
    rho = float(spearmanr(y, p).statistic) if np.std(p) > 1e-9 else np.nan
    return dict(R2=guarded, standard_R2=standard, MAE=float(mean_absolute_error(y, p)), Spearman=rho)


def environment():
    return dict(python=platform.python_version(), executable=sys.executable, numpy=np.__version__,
                pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                xgboost=xgb.__version__, lightgbm=lgb.__version__, rdkit=rdBase.rdkitVersion,
                logical_cpus=os.cpu_count(), platform=platform.platform(), estimator_n_jobs=-1,
                concurrent_fits=1)


def namespace(features=False, df=None):
    assert sha(SOURCE) == SOURCE_SHA
    source = SOURCE.read_text('utf-8'); tree = ast.parse(source)
    functions = {'rf', 'xgbr', 'lgbr'}
    assignments = set()
    if features:
        functions |= {'dvec', 'sdiv', 'ecfp', 'guest3d'}
        assignments = {'DESC', 'uh', 'ug', 'H', 'G', 'P', 'X_pair', 'he', 'ge', 'X_ecfp'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            if node.name in {'rf', 'xgbr', 'lgbr'}:
                node.args.args.append(ast.arg(arg='seed'))
                count = 0
                for call in ast.walk(node):
                    if isinstance(call, ast.keyword) and call.arg == 'random_state':
                        assert isinstance(call.value, ast.Constant) and call.value.value == 0
                        call.value = ast.Name(id='seed', ctx=ast.Load()); count += 1
                assert count == 1
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            selected.append(node)
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == functions
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    ns = dict(np=np, pd=pd, Chem=Chem, AllChem=AllChem, Descriptors=Descriptors,
              rdMolDescriptors=rdMolDescriptors, RandomForestRegressor=RandomForestRegressor, xgb=xgb, lgb=lgb, df=df)
    exec(compile(module, str(SOURCE) + ':selected_only', 'exec'), ns)
    return ns, ast.unparse(module)
