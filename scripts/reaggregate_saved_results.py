"""Reaggregate saved GNN and tree predictions without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from release_common import (
    ReleaseError,
    atomic_write_json,
    claim_output,
    compare_csv,
    default_output,
    eight_ulp_equal,
    load_json,
    manifest,
    plain,
    repo_root,
    sha256,
    verify_file_spec,
)


GNN_REL = Path("results/gnn_saved")
TREE_REL = Path("results/tree_saved")
GNN_NULLS = [
    "sim_knn_k5",
    "additive",
    "host_only_RF",
    "guest_only_RF",
    "cond_only_RF",
    "source_only_RF",
]
TREE_MODELS = ["PAIR_RF", "ECFP_RF", "XGB_pair", "XGB_ecfp", "LGBM_pair", "LGBM_ecfp", "3D_pair_RF"]


def gnn_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 and np.std(y_pred) > 1e-9 else None
    rho = (
        float(spearmanr(y_true, y_pred).statistic)
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1
        else None
    )
    return {"R2": r2, "MAE": float(mean_absolute_error(y_true, y_pred)), "Spearman": rho}


def tree_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    standard = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    guarded = standard if np.std(y_pred) > 1e-9 else np.nan
    rho = float(spearmanr(y_true, y_pred).statistic) if np.std(y_pred) > 1e-9 else np.nan
    return {
        "R2": guarded,
        "standard_R2": standard,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": rho,
    }


def _check_metric(expected: Any, actual: Any, label: str, strict_1e12: bool = False) -> None:
    if expected is None or actual is None or pd.isna(expected) or pd.isna(actual):
        if not ((expected is None or pd.isna(expected)) and (actual is None or pd.isna(actual))):
            raise ReleaseError(f"DEFINEDNESS_MISMATCH: {label}")
        return
    if strict_1e12:
        if not abs(float(expected) - float(actual)) < 1e-12:
            raise ReleaseError(f"TREE_R2_GATE_FAILED: {label}: expected={expected} actual={actual}")
    elif not eight_ulp_equal(float(expected), float(actual)):
        raise ReleaseError(f"METRIC_MISMATCH: {label}: expected={expected} actual={actual}")


def _validate_gnn_inputs(root: Path) -> tuple[Path, dict[str, Any]]:
    reference = manifest(root)["scientific_groups"]["gnn_saved_reaggregation"]
    base = root / reference["base"]
    if int(reference["expected_jobs"]) != 350:
        raise ReleaseError("GNN_JOB_COUNT_AUTHORITY_MISMATCH")
    for relative, file_spec in reference["inputs"].items():
        verify_file_spec(root, relative, file_spec)
    for name, file_spec in reference["root_files"].items():
        verify_file_spec(root, str(Path(reference["base"]) / name), file_spec)
    return base, reference


def _validate_tree_inputs(root: Path) -> tuple[Path, dict[str, Any]]:
    reference = manifest(root)["scientific_groups"]["tree_saved_reaggregation"]
    base = root / reference["base"]
    if int(reference["expected_jobs"]) != 420:
        raise ReleaseError("TREE_JOB_COUNT_AUTHORITY_MISMATCH")
    for relative, file_spec in reference["inputs"].items():
        verify_file_spec(root, relative, file_spec)
    for name, file_spec in reference["root_files"].items():
        verify_file_spec(root, str(Path(reference["base"]) / name), file_spec)
    return base, reference


def reaggregate_gnn(root: Path, output: Path, compare_historical: bool = True) -> dict[str, Any]:
    base, reference = _validate_gnn_inputs(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(base / "rows.csv", float_precision="round_trip")
    folds = json.loads((base / "folds.json").read_text(encoding="utf-8"))
    jobs = json.loads((base / "jobs.json").read_text(encoding="utf-8"))
    if len(rows) != 2383 or len(folds) != 35 or len(jobs) != 350:
        raise ReleaseError(f"GNN_FROZEN_COUNT_MISMATCH rows={len(rows)} folds={len(folds)} jobs={len(jobs)}")
    fold_map = {(str(f["regime"]), str(f["fold"])): f for f in folds}
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    runner_counts: dict[str, int] = {}
    for job in jobs:
        folder = base / "jobs" / job["job_id"]
        result_path = folder / "result.json"
        prediction_path = folder / "predictions.csv"
        if not result_path.is_file() or not prediction_path.is_file():
            raise ReleaseError(f"GNN_INCOMPLETE_JOB: {job['job_id']}")
        result = load_json(result_path)
        if result["status"] != "COMPLETED" or result["predictions_sha"] != sha256(prediction_path):
            raise ReleaseError(f"GNN_JOB_BINDING_FAILED: {job['job_id']}")
        for key in ("job_id", "seed", "model", "regime", "fold"):
            if result[key] != job[key]:
                raise ReleaseError(f"GNN_JOB_IDENTITY_MISMATCH: {job['job_id']}:{key}")
        runner = str(result["runner_sha"])
        runner_counts[runner] = runner_counts.get(runner, 0) + 1
        pred = pd.read_csv(prediction_path, float_precision="round_trip")
        fold = fold_map[(str(job["regime"]), str(job["fold"]))]
        test = fold["test"]
        if pred.row_index.tolist() != test:
            raise ReleaseError(f"GNN_TEST_ORDER_MISMATCH: {job['job_id']}")
        if not np.array_equal(pred.y_true.to_numpy(), rows.iloc[test].y_true.to_numpy()):
            raise ReleaseError(f"GNN_TARGET_MISMATCH: {job['job_id']}")
        metric = gnn_score(
            pred.y_true.to_numpy(dtype=np.float32),
            pred.y_pred.to_numpy(dtype=np.float32),
        )
        stored_metric = result["metrics"]
        for key in ("R2", "MAE", "Spearman"):
            _check_metric(stored_metric[key], metric[key], f"gnn/{job['job_id']}/{key}")
        record = dict(result)
        record.update(metric)
        record.pop("metrics")
        if job["regime"] == "family_cold":
            yp = pred.y_pred.to_numpy(dtype=np.float32)
            yt = pred.y_true.to_numpy(dtype=np.float32)
            record["calib_slope"] = float(linregress(yp, yt).slope) if pred.y_pred.nunique() > 1 else None
            record["offset"] = float(yt.mean() - yp.mean())
        results.append(record)
        predictions.append(pred)

    expected_runner_counts = {str(key): int(value) for key, value in reference["historical_runner_sha_counts"].items()}
    if runner_counts != expected_runner_counts:
        raise ReleaseError(f"GNN_RUNNER_SHA_COUNT_MISMATCH: expected={expected_runner_counts} actual={runner_counts}")

    metrics = pd.DataFrame(results)
    # Preserve the historical summarizer's default CSV float parser here.
    nulls = pd.read_csv(base / "null_scores.csv", dtype={"fold": str})
    best = (
        nulls.assign(order=nulls.model.map({name: index for index, name in enumerate(GNN_NULLS)}))
        .sort_values(["R2", "order"], ascending=[False, True])
        .drop_duplicates(["regime", "fold"])
        .rename(columns={"model": "best_null", "R2": "best_null_R2"})[
            ["regime", "fold", "best_null", "best_null_R2"]
        ]
    )
    metrics = metrics.merge(best, on=["regime", "fold"], how="left", validate="many_to_one")
    metrics["margin_R2"] = metrics.R2 - metrics.best_null_R2
    all_predictions = pd.concat(predictions, ignore_index=True)

    per_seed: list[dict[str, Any]] = []
    for (model, seed, regime), subset in metrics.groupby(["model", "seed", "regime"], sort=False):
        expected = sum(f["regime"] == regime for f in folds)
        if len(subset) != expected or regime == "family_cold":
            continue
        row: dict[str, Any] = {
            "model": model,
            "seed": int(seed),
            "regime": regime,
            "n_outer_folds": len(subset),
        }
        if regime == "double_cold":
            if len(subset) != 12:
                raise ReleaseError(f"GNN_DOUBLE_BLOCK_COUNT_MISMATCH: {model}/{seed}")
            row.update(
                n_defined_margins=int(subset.margin_R2.notna().sum()),
                mean_R2=float(subset.R2.mean()) if subset.R2.notna().all() else None,
                block_sd_R2_ddof0=float(subset.R2.std(ddof=0)) if subset.R2.notna().all() else None,
                mean_MAE=float(subset.MAE.mean()),
                mean_Spearman=float(subset.Spearman.mean()) if subset.Spearman.notna().all() else None,
                mean_margin_R2=float(subset.margin_R2.mean()) if subset.margin_R2.notna().all() else None,
                positive_blocks=int((subset.margin_R2 > 0).sum()) if subset.margin_R2.notna().all() else None,
            )
        else:
            pred = all_predictions[
                (all_predictions.model == model)
                & (all_predictions.seed == seed)
                & (all_predictions.regime == regime)
            ]
            if len(pred) != len(rows) or not pred.row_index.is_unique:
                raise ReleaseError(f"GNN_FOLD_CONCATENATION_MISMATCH: {model}/{seed}/{regime}")
            row.update(
                gnn_score(
                    pred.y_true.to_numpy(dtype=np.float32),
                    pred.y_pred.to_numpy(dtype=np.float32),
                )
            )
        per_seed.append(row)
    per_seed_frame = pd.DataFrame(per_seed)

    across_seed: list[dict[str, Any]] = []
    for (model, regime), subset in per_seed_frame.groupby(["model", "regime"]):
        if set(subset.seed) != set(range(5)):
            continue
        for key in (
            "R2",
            "MAE",
            "Spearman",
            "mean_R2",
            "block_sd_R2_ddof0",
            "mean_MAE",
            "mean_Spearman",
            "mean_margin_R2",
            "positive_blocks",
        ):
            if key not in subset or not subset[key].notna().all():
                continue
            values = subset[key].to_numpy()
            across_seed.append(
                {
                    "model": model,
                    "regime": regime,
                    "metric": key,
                    "n_training_seeds": 5,
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    across_seed_frame = pd.DataFrame(across_seed)

    family_stats: list[dict[str, Any]] = []
    for (model, family), subset in metrics[metrics.regime == "family_cold"].groupby(["model", "fold"]):
        if set(subset.seed) != set(range(5)):
            continue
        for key in ("R2", "MAE", "Spearman", "calib_slope", "offset"):
            if key not in subset or not subset[key].notna().all():
                continue
            values = subset[key].to_numpy()
            family_stats.append(
                {
                    "model": model,
                    "family": family,
                    "flag": subset.flag.iloc[0],
                    "metric": key,
                    "n_training_seeds": 5,
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    family_frame = pd.DataFrame(family_stats)
    double_frame = metrics[metrics.regime == "double_cold"].copy()

    generated = {
        "job_metrics.csv": metrics,
        "per_seed_metrics.csv": per_seed_frame,
        "across_seed_descriptive.csv": across_seed_frame,
        "family_across_seed_descriptive.csv": family_frame,
        "double_cold_120_block_records.csv": double_frame,
    }
    for name, frame in generated.items():
        frame.to_csv(output / name, index=False)

    comparisons: dict[str, Any] = {}
    if compare_historical:
        for name in generated:
            comparisons[name] = compare_csv(base / name, output / name, f"gnn/{name}")
    report = {
        "status": "PASS",
        "completed_jobs": len(results),
        "expected_jobs": len(jobs),
        "prediction_appearances": len(all_predictions),
        "per_seed_rows": len(per_seed_frame),
        "across_seed_rows": len(across_seed_frame),
        "family_rows": len(family_frame),
        "runner_sha_counts": runner_counts,
        "historical_comparisons": comparisons,
    }
    atomic_write_json(output / "reaggregation_report.json", report)
    return report


def reaggregate_trees(root: Path, output: Path, compare_historical: bool = True) -> dict[str, Any]:
    base, reference = _validate_tree_inputs(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(base / "rows.csv", float_precision="round_trip")
    folds = json.loads((base / "folds.json").read_text(encoding="utf-8"))
    jobs = json.loads((base / "jobs.json").read_text(encoding="utf-8"))
    nulls = pd.read_csv(base / "null_scores.csv", float_precision="round_trip")
    if len(rows) != 2383 or len(folds) != 12 or len(jobs) != 420:
        raise ReleaseError(f"TREE_FROZEN_COUNT_MISMATCH rows={len(rows)} folds={len(folds)} jobs={len(jobs)}")
    predictions: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for job in jobs:
        folder = base / "jobs" / job["job_id"]
        result = load_json(folder / "result.json")
        prediction_path = folder / "predictions.csv"
        membership_path = folder / "membership.json"
        if result["status"] != "COMPLETED":
            raise ReleaseError(f"TREE_INCOMPLETE_JOB: {job['job_id']}")
        for key in ("job_id", "seed", "block", "model"):
            if result[key] != job[key]:
                raise ReleaseError(f"TREE_JOB_IDENTITY_MISMATCH: {job['job_id']}:{key}")
        if (
            result["runner_sha"] != reference["historical_runner_sha256"]
            or result["features_sha"] != reference["root_files"]["features.npz"]["sha256"]
        ):
            raise ReleaseError(f"TREE_JOB_PROTOCOL_MISMATCH: {job['job_id']}")
        if result["predictions_sha"] != sha256(prediction_path) or result["membership_sha"] != sha256(membership_path):
            raise ReleaseError(f"TREE_JOB_BINDING_FAILED: {job['job_id']}")
        fold = folds[int(job["block"])]
        if load_json(membership_path) != fold:
            raise ReleaseError(f"TREE_MEMBERSHIP_MISMATCH: {job['job_id']}")
        pred = pd.read_csv(prediction_path, float_precision="round_trip")
        if pred.row_index.tolist() != fold["test"]:
            raise ReleaseError(f"TREE_TEST_ORDER_MISMATCH: {job['job_id']}")
        if job["model"].startswith("XGB_"):
            pred["y_pred"] = pred.y_pred.to_numpy().astype(np.float32).astype(float)
        expected_rows = rows.iloc[fold["test"]]
        for column in ("host_id", "guest_id", "y_true"):
            if not np.array_equal(pred[column].to_numpy(), expected_rows[column].to_numpy()):
                raise ReleaseError(f"TREE_SAVED_ROW_MISMATCH: {job['job_id']}:{column}")
        pred_dtype = np.float32 if job["model"].startswith("XGB_") else np.float64
        stat = tree_metrics(
            pred.y_true.to_numpy(dtype=np.float32),
            pred.y_pred.to_numpy().astype(pred_dtype),
        )
        _check_metric(result["metrics"]["R2"], stat["R2"], f"trees/{job['job_id']}/R2", strict_1e12=True)
        for key in ("standard_R2", "MAE", "Spearman"):
            _check_metric(result["metrics"][key], stat[key], f"trees/{job['job_id']}/{key}")
        candidates = nulls[nulls.block == int(job["block"])].set_index("model").loc[GNN_NULLS]
        finite = candidates[candidates.R2.notna()]
        winner = finite.R2.idxmax()
        standard_winner = candidates.standard_R2.idxmax()
        if result["best_null"] != winner or result["standard_best_null"] != standard_winner:
            raise ReleaseError(f"TREE_NULL_WINNER_MISMATCH: {job['job_id']}")
        _check_metric(result["margin_R2"], stat["R2"] - float(candidates.loc[winner, "R2"]), f"trees/{job['job_id']}/margin")
        _check_metric(
            result["standard_margin_R2"],
            stat["standard_R2"] - float(candidates.loc[standard_winner, "standard_R2"]),
            f"trees/{job['job_id']}/standard_margin",
        )
        predictions.append(pred)
        records.append({**{key: value for key, value in result.items() if key != "metrics"}, **result["metrics"]})

    block = pd.DataFrame(records)
    if not block.margin_R2.notna().all():
        raise ReleaseError("TREE_UNDEFINED_MARGIN")
    combined = pd.concat(predictions, ignore_index=True)
    if len(combined) != 69440 or combined.duplicated(["model", "seed", "block", "row_index"]).any():
        raise ReleaseError("TREE_COMBINED_PREDICTION_IDENTITY_MISMATCH")

    per_seed: list[dict[str, Any]] = []
    for (model, seed), group in block.groupby(["model", "seed"], sort=False):
        if set(group.block) != set(range(12)):
            raise ReleaseError(f"TREE_BLOCK_COVERAGE_MISMATCH: {model}/{seed}")
        per_seed.append(
            {
                "model": model,
                "seed": int(seed),
                "n_blocks": 12,
                "mean_R2": group.R2.mean(),
                "block_sd_R2_ddof0": group.R2.std(ddof=0),
                "mean_margin_R2": group.margin_R2.mean(),
                "positive_blocks": int((group.margin_R2 > 0).sum()),
                "mean_standard_margin_R2": group.standard_margin_R2.mean(),
                "standard_positive_blocks": int((group.standard_margin_R2 > 0).sum()),
            }
        )
    seed_frame = pd.DataFrame(per_seed)
    descriptive: list[dict[str, Any]] = []
    for model, group in seed_frame.groupby("model", sort=False):
        if group.seed.tolist() != list(range(5)):
            raise ReleaseError(f"TREE_SEED_ORDER_MISMATCH: {model}")
        for metric in ("mean_R2", "mean_margin_R2", "positive_blocks", "mean_standard_margin_R2"):
            values = group[metric].to_numpy()
            descriptive.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_seeds": 5,
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    descriptive_frame = pd.DataFrame(descriptive)

    old = pd.read_csv(root / "data/released_predictions/predictions_double_cold.csv", float_precision="round_trip")
    historical: list[dict[str, Any]] = []
    for model in TREE_MODELS:
        for block_index in range(12):
            prior = old[
                (old.model == model) & (old.fold_or_block.astype(str) == str(block_index))
            ].set_index("row_index")
            new = combined[
                (combined.model == model) & (combined.seed == 0) & (combined.block == block_index)
            ].set_index("row_index")
            if set(prior.index) != set(new.index):
                raise ReleaseError(f"TREE_HISTORICAL_ROW_SET_MISMATCH: {model}/{block_index}")
            prior = prior.loc[new.index]
            dtype = np.float32 if model.startswith("XGB_") else np.float64
            prior_metric = tree_metrics(prior.y_true.to_numpy(dtype=np.float32), prior.y_pred.to_numpy().astype(dtype))
            new_metric = tree_metrics(new.y_true.to_numpy(dtype=np.float32), new.y_pred.to_numpy().astype(dtype))
            historical.append(
                {
                    "model": model,
                    "block": block_index,
                    "n": len(new),
                    "max_abs_prediction_difference": float(np.max(np.abs(new.y_pred - prior.y_pred))),
                    "historical_R2": prior_metric["R2"],
                    "new_seed0_R2": new_metric["R2"],
                    "R2_difference": new_metric["R2"] - prior_metric["R2"],
                }
            )
    historical_frame = pd.DataFrame(historical)
    generated = {
        "block_metrics.csv": block,
        "per_seed_metrics.csv": seed_frame,
        "across_seed_descriptive.csv": descriptive_frame,
        "historical_seed0_comparison.csv": historical_frame,
    }
    for name, frame in generated.items():
        frame.to_csv(output / name, index=False)
    comparisons: dict[str, Any] = {}
    if compare_historical:
        for name in generated:
            comparisons[name] = compare_csv(base / name, output / name, f"trees/{name}")
    report = {
        "status": "PASS",
        "completed_jobs": len(jobs),
        "prediction_appearances": len(combined),
        "measurement_rows": len(rows),
        "model_seed_summaries": len(seed_frame),
        "block_records": len(block),
        "historical_comparisons": comparisons,
    }
    atomic_write_json(output / "reaggregation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Project or path-preserving candidate root")
    parser.add_argument("--output", help="New release-owned output directory")
    parser.add_argument("--protocol", choices=("gnn", "trees", "all"), default="all")
    parser.add_argument("--no-compare-historical", action="store_true")
    args = parser.parse_args()
    root = repo_root(args.root)
    output = Path(args.output) if args.output else default_output("reaggregate")
    binding = {"protocol": args.protocol, "root_layout": "project-relative-v1"}
    output = claim_output(root, output, "saved-results-reaggregation", binding)
    report: dict[str, Any] = {"status": "PASS", "protocol": args.protocol, "results": {}}
    compare = not args.no_compare_historical
    if args.protocol in ("gnn", "all"):
        report["results"]["gnn"] = reaggregate_gnn(root, output / "gnn", compare)
    if args.protocol in ("trees", "all"):
        report["results"]["trees"] = reaggregate_trees(root, output / "trees", compare)
    atomic_write_json(output / "reaggregation_report.json", report)
    print(json.dumps(plain({"status": "PASS", "output": output, "protocol": args.protocol})), flush=True)


if __name__ == "__main__":
    main()
