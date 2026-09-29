"""Persist R7 record counts and algebra over saved family summaries; no model scoring."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
OUT = ROOT / 'runs/record_audit'
MANIFEST = ROOT / 'data/population/full_bap_manifest.csv'
CONDITIONS = ROOT / 'data/clean/suprabench_bap_clean_cond.csv'
CONDITION_RESULTS = ROOT / 'results/diagnostics/suprabench_cond_audit_results.json'
PREDICTIONS = ROOT / 'data/released_predictions/predictions_family_cold.csv'
FAMILY = ROOT / 'results/family_diagnostics/tree_null_family_metrics.csv'

def main():
    OUT.mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    valid = manifest.loc[manifest['structure_valid'].eq(True)].reset_index(drop=True)
    conditions = pd.read_csv(CONDITIONS)
    assert len(valid) == len(conditions) == 2383
    for name in ['host_smiles', 'guest_smiles', 'host_name', 'source']:
        assert valid[name].equals(conditions[name]), name
    np.testing.assert_array_equal(valid['binding_affinity'].to_numpy(), conditions['y'].to_numpy())
    token_counts = valid['task_id'].astype(str).str.split('|', regex=False).str.len()
    averaged = token_counts.gt(1)
    conditions['_upstream_averaged_record'] = averaged.to_numpy()
    pair_keys = ['host_smiles', 'guest_smiles']
    condition_keys = ['solvent_bucket', 'temp_bucket', 'ph_bucket']
    pairs = [group for _, group in conditions.groupby(pair_keys) if len(group) > 1]
    same_conditions = [group for _, group in conditions.groupby(pair_keys + condition_keys) if len(group) > 1]
    cross_conditions = [group for group in pairs if group[condition_keys].astype(str).agg('|'.join, axis=1).nunique() > 1]
    counted = {}
    for name, groups in [('same_host_guest', pairs), ('same_host_guest_condition', same_conditions), ('cross_condition', cross_conditions)]:
        counted[name] = {'groups': len(groups), 'groups_containing_upstream_averaged_records': sum(bool(g['_upstream_averaged_record'].any()) for g in groups)}
    assert counted == {
        'same_host_guest': {'groups': 121, 'groups_containing_upstream_averaged_records': 28},
        'same_host_guest_condition': {'groups': 75, 'groups_containing_upstream_averaged_records': 10},
        'cross_condition': {'groups': 47, 'groups_containing_upstream_averaged_records': 18},
    }
    with CONDITION_RESULTS.open(encoding='utf-8') as f:
        p2 = json.load(f)['P2']
    saved = {key: p2[key] for key in ['same_hg_sigma_mean', 'n_same_hg', 'true_replicate_sigma_mean', 'n_true_rep', 'cross_condition_spread_mean', 'n_cross_cond']}
    assert int(averaged.sum()) == 432
    assert set(valid.loc[averaged, 'source']) == {'eupmc'}
    record_receipt = {
        'analysis_records': len(valid), 'upstream_averaged_records': int(averaged.sum()),
        'upstream_averaged_record_percentage': float(averaged.mean() * 100),
        'literal_task_id_token_range_for_averaged_records': [int(token_counts[averaged].min()), int(token_counts[averaged].max())],
        'source_labels_of_averaged_records': ['eupmc'],
        'group_counts': counted, 'saved_dispersion_values_not_recomputed': saved,
        'grouping_keys': {'same_condition': pair_keys + condition_keys, 'cross_condition_pair': pair_keys},
        'scope': 'Counts released-record memberships only. No new dispersion metrics or independent-experiment count. Source is not a grouping key; upstream semantics are supported by the public data card.'
    }
    (OUT / 'record_semantics.json').write_text(json.dumps(record_receipt, indent=2) + '\n', encoding='utf-8')

    # Use full-precision saved residuals, not the three-decimal target SD in P3.
    predictions = pd.read_csv(PREDICTIONS, float_precision='round_trip')
    predictions = predictions.loc[predictions['model'].eq('PAIR_RF')]
    with FAMILY.open(encoding='utf-8', newline='') as f:
        models = {row['family']: row for row in csv.DictReader(f) if row['model'] == 'PAIR_RF' and row['seed'] == '0'}
    algebra = []
    for family in ['calixarene', 'cucurbituril', 'cyclodextrin', 'other']:
        row = models[family]
        saved = predictions.loc[predictions['fold_or_block'].eq(family)]
        n = int(row['n_measurements'])
        assert n == len(saved) and saved.row_index.is_unique
        # Preserve the target dtype of analyze_family.py; tree predictions are float64.
        target = saved.y_true.to_numpy().astype(np.float32)
        prediction = saved.y_pred.to_numpy().astype(np.float64)
        residual = target - prediction
        r2 = float(row['standard_R2']); offset = float(row['offset'])
        assert abs(float(np.mean(target) - np.mean(prediction)) - offset) < 1e-6
        mean_residual = float(np.mean(residual))
        mse = float(np.mean(residual**2))
        target_sd = float(np.std(target.astype(np.float64)))
        ratio = mean_residual**2 / mse
        assert 0 <= ratio <= 1 + 1e-12
        algebra.append({'family': family, 'n_records': n, 'saved_standard_R2': r2,
                        'saved_target_vector_sd_ddof0': target_sd, 'saved_observed_minus_predicted_offset': offset,
                        'mean_saved_residual': mean_residual, 'MSE_of_saved_residuals': mse, 'squared_mean_residual': mean_residual**2,
                        'squared_mean_residual_share_of_MSE': ratio})
    with (OUT / 'family_error_decomposition.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(algebra[0])); writer.writeheader(); writer.writerows(algebra)
    inputs = []
    for path in [MANIFEST, CONDITIONS, CONDITION_RESULTS, PREDICTIONS, FAMILY]:
        inputs.append({'path': path.relative_to(PROJECT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs': inputs,
                  'outputs': ['runs/record_audit/record_semantics.json', 'runs/record_audit/family_error_decomposition.csv'],
                  'algebra': 'Using the saved PAIR_RF family prediction rows: residual=y_true(float32)-y_pred(float64); MSE=mean(residual**2); share=mean(residual)**2/MSE. No rounded target SD is used. Means may differ slightly from the separately saved float32 mean(y_true)-mean(y_pred) offset.',
                  'interpretation': 'Descriptive decomposition of saved errors. Not a recalibration evaluation, causal explanation or slope-direction inference.',
                  'execution': 'No training, predictions, fold generation, dispersion recomputation or new grouping policy.'}
    (OUT / 'record_and_algebra_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'records': 2383, 'averaged_records': 432, 'groups': counted, 'family_algebra': algebra}, indent=2))

if __name__ == '__main__':
    main()
