# SPDX-License-Identifier: MIT
"""Reaggregate frozen canonical GNN run-A predictions against the six nulls.

This script is a saved-prediction calculation only.  It does not fit a model,
change a fold, choose a new null family, or alter any source artifact.  It
reproduces the production scorer's dtype transitions:

* block GNN metrics: float32 target and float32 prediction;
* block descriptor/null metrics: float32 target and float64 prediction;
* pooled metrics: Python-float accumulation followed by float64 arrays.

The serialized target column is cast back to float32 before scoring.  The
optional ``--frozen-targets`` input checks that this reconstruction is exactly
equal to the frozen ``tabular_features.npz`` target array.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.metrics import r2_score


SCRIPT_VERSION = "canonical-gnn-r15-reaggregation-v1"

SPLITS = ("host_cold", "guest_cold", "double_cold")
EXPECTED_POOLED_N = {"host_cold": 2383, "guest_cold": 2383, "double_cold": 1716}
GNN_MODELS = ("GINE", "AttentiveFP")
TARGET_MODELS = ("PAIR_RF",) + GNN_MODELS
NULL_MODELS = (
    "sim_knn_k5",
    "additive",
    "host_only_RF",
    "guest_only_RF",
    "cond_only_RF",
    "source_only_RF",
)
CPU_MODELS = ("PAIR_RF",) + NULL_MODELS
PREDICTION_COLUMNS = (
    "split",
    "block_id",
    "fold_index",
    "seed",
    "model",
    "row_index",
    "host_id",
    "guest_id",
    "canonical_host_sha256",
    "canonical_guest_sha256",
    "y_true",
    "y_pred",
)
METRIC_COLUMNS = (
    "scope",
    "split",
    "block_id",
    "fold_index",
    "seed",
    "model",
    "n",
    "mae",
    "r2",
    "spearman",
    "r2_status",
    "spearman_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reaggregate frozen canonical GNN run-A predictions."
    )
    parser.add_argument("--gnn-predictions", type=Path, required=True)
    parser.add_argument("--gnn-metrics", type=Path, required=True)
    parser.add_argument("--cpu-predictions", type=Path, required=True)
    parser.add_argument("--cpu-metrics", type=Path, required=True)
    parser.add_argument(
        "--frozen-targets",
        type=Path,
        default=None,
        help=(
            "optional frozen tabular_features.npz witness for an exact float32 target check"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        default="canonical_gnn_aggregation_r15",
        help="basename for the JSON and TSV outputs",
    )
    parser.add_argument(
        "--blocks-filename",
        default="canonical_gnn_double_cold_blocks_r15.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this script's three existing R15 outputs",
    )
    args = parser.parse_args()
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_rows(path: Path, delimiter: str, expected: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = tuple(reader.fieldnames or ())
        if fields != tuple(expected):
            raise RuntimeError(f"schema mismatch for {path}: {fields}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty input: {path}")
    return rows


def group_predictions(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["block_id"], row["model"])].append(row)
    return dict(grouped)


def target_value(text: str) -> np.float32:
    return np.float32(float(text))


def arrays_from_rows(
    rows: list[dict[str, str]], prediction_dtype: type[np.float32] | type[np.float64], pooled: bool
) -> tuple[np.ndarray, np.ndarray]:
    if pooled:
        # Production aggregation explicitly converted each scalar through float()
        # before np.asarray, yielding float64 arrays.
        y_true = np.asarray([float(target_value(row["y_true"])) for row in rows], dtype=np.float64)
        y_pred = np.asarray(
            [float(prediction_dtype(float(row["y_pred"]))) for row in rows], dtype=np.float64
        )
    else:
        y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float32)
        y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=prediction_dtype)
    return y_true, y_pred


def production_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, str]:
    if len(np.unique(y_true)) > 1 and float(np.std(y_pred)) > 1e-9:
        return float(r2_score(y_true, y_pred)), "DEFINED"
    return None, "UNKNOWN"


def row_keys(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    keys = [(row["block_id"], int(row["row_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate (block_id,row_index) in one model/split selection")
    return keys


def metric_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["scope"], row["split"], row["block_id"], row["model"])
        if key in indexed:
            raise RuntimeError(f"duplicate metric key: {key}")
        indexed[key] = row
    return indexed


def compare_metric(actual: float, expected: float) -> dict[str, Any]:
    difference = abs(actual - expected)
    tolerance = 8 * math.ulp(expected)
    return {
        "actual": actual,
        "expected": expected,
        "absolute_difference": difference,
        "eight_ulp_tolerance": tolerance,
        "match_within_eight_ulp": difference <= tolerance,
        "exact_float_match": actual == expected,
    }


def validate_saved_metrics(
    prediction_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    models: tuple[str, ...],
    prediction_dtype: type[np.float32] | type[np.float64],
) -> dict[str, Any]:
    grouped = group_predictions(prediction_rows)
    indexed = metric_index(metric_rows)
    comparisons: list[dict[str, Any]] = []
    selected_metrics = [row for row in metric_rows if row["model"] in models]
    for stored in selected_metrics:
        scope = stored["scope"]
        split = stored["split"]
        block_id = stored["block_id"]
        model = stored["model"]
        pooled = scope == "model_split_aggregate"
        if pooled:
            selected = [
                row
                for row in prediction_rows
                if row["split"] == split and row["model"] == model
            ]
        elif scope == "model_block":
            selected = grouped[(split, block_id, model)]
        else:
            raise RuntimeError(f"unexpected metric scope: {scope}")
        y_true, y_pred = arrays_from_rows(selected, prediction_dtype, pooled)
        actual, status = production_r2(y_true, y_pred)
        if status != stored["r2_status"]:
            raise RuntimeError(
                f"r2 status mismatch for {(scope, split, block_id, model)}: "
                f"{status} != {stored['r2_status']}"
            )
        record: dict[str, Any] = {
            "scope": scope,
            "split": split,
            "block_id": block_id,
            "model": model,
            "n": len(selected),
            "status": status,
        }
        if actual is not None:
            expected = float(stored["r2"])
            record.update(compare_metric(actual, expected))
            if not record["match_within_eight_ulp"]:
                raise RuntimeError(f"stored metric mismatch: {record}")
        elif stored["r2"] != "":
            raise RuntimeError(f"UNKNOWN stored metric is not blank: {record}")
        comparisons.append(record)
    finite = [row for row in comparisons if row["status"] == "DEFINED"]
    return {
        "selected_stored_metric_rows": len(comparisons),
        "defined_metric_rows": len(finite),
        "unknown_metric_rows": len(comparisons) - len(finite),
        "within_eight_ulp": sum(row.get("match_within_eight_ulp", False) for row in finite),
        "exact_float_matches": sum(row.get("exact_float_match", False) for row in finite),
        "max_absolute_difference": max(
            (row["absolute_difference"] for row in finite), default=0.0
        ),
    }


def validate_target_reconstruction(
    prediction_sets: list[list[dict[str, str]]], frozen_targets: Path | None
) -> dict[str, Any]:
    reconstructed: dict[int, np.float32] = {}
    serialized_conflicts = 0
    total_prediction_rows = 0
    for rows in prediction_sets:
        for row in rows:
            total_prediction_rows += 1
            index = int(row["row_index"])
            value = target_value(row["y_true"])
            if index in reconstructed and reconstructed[index] != value:
                serialized_conflicts += 1
            reconstructed[index] = value
    if serialized_conflicts:
        raise RuntimeError(f"float32 target conflicts across predictions: {serialized_conflicts}")
    result: dict[str, Any] = {
        "prediction_rows_checked": total_prediction_rows,
        "unique_measurement_rows": len(reconstructed),
        "float32_conflicts_across_prediction_sources": serialized_conflicts,
        "frozen_target_witness_used": frozen_targets is not None,
    }
    if frozen_targets is not None:
        if not frozen_targets.is_file():
            raise FileNotFoundError(frozen_targets)
        with np.load(frozen_targets, allow_pickle=False) as loaded:
            y = loaded["y"]
            row_index = loaded["row_index"]
        if y.dtype != np.dtype("float32"):
            raise RuntimeError(f"unexpected frozen target dtype: {y.dtype}")
        frozen = {int(index): value for index, value in zip(row_index, y, strict=True)}
        mismatches = sum(frozen[index] != value for index, value in reconstructed.items())
        if mismatches:
            raise RuntimeError(f"serialized->float32 differs from frozen targets: {mismatches}")
        result.update(
            {
                "frozen_target_file": frozen_targets.name,
                "frozen_target_sha256": sha256(frozen_targets),
                "frozen_target_dtype": str(y.dtype),
                "frozen_target_rows": len(y),
                "float32_reconstruction_mismatches": mismatches,
            }
        )
    return result


def selected_rows(
    rows: list[dict[str, str]], split: str, model: str, block_id: str | None = None
) -> list[dict[str, str]]:
    result = [
        row
        for row in rows
        if row["split"] == split
        and row["model"] == model
        and (block_id is None or row["block_id"] == block_id)
    ]
    if not result:
        raise RuntimeError(f"missing predictions for {(split, block_id, model)}")
    return result


def score_selection(
    rows: list[dict[str, str]], dtype: type[np.float32] | type[np.float64], pooled: bool
) -> tuple[float | None, str]:
    y_true, y_pred = arrays_from_rows(rows, dtype, pooled)
    return production_r2(y_true, y_pred)


def build_pooled(
    gnn_rows: list[dict[str, str]], cpu_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in SPLITS:
        null_scores: dict[str, float] = {}
        cpu_reference = selected_rows(cpu_rows, split, "PAIR_RF")
        reference_keys = row_keys(cpu_reference)
        if len(reference_keys) != EXPECTED_POOLED_N[split]:
            raise RuntimeError(f"unexpected pooled n for {split}: {len(reference_keys)}")
        for null in NULL_MODELS:
            rows = selected_rows(cpu_rows, split, null)
            if row_keys(rows) != reference_keys:
                raise RuntimeError(f"pooled CPU membership mismatch for {split}/{null}")
            value, status = score_selection(rows, np.float64, pooled=True)
            if status != "DEFINED" or value is None:
                raise RuntimeError(f"pooled null unexpectedly undefined: {split}/{null}")
            null_scores[null] = value
        best_null = max(NULL_MODELS, key=lambda model: null_scores[model])
        for model in TARGET_MODELS:
            if model == "PAIR_RF":
                rows = cpu_reference
                dtype = np.float64
            else:
                rows = selected_rows(gnn_rows, split, model)
                dtype = np.float32
                if row_keys(rows) != reference_keys:
                    raise RuntimeError(f"pooled GNN/CPU membership mismatch for {split}/{model}")
            value, status = score_selection(rows, dtype, pooled=True)
            if status != "DEFINED" or value is None:
                raise RuntimeError(f"pooled target unexpectedly undefined: {split}/{model}")
            output.append(
                {
                    "split": split,
                    "model": model,
                    "aggregation": "pooled_test_appearances",
                    "n_appearances": len(rows),
                    "n_blocks": 12 if split == "double_cold" else 5,
                    "model_r2": value,
                    "best_null": best_null,
                    "best_null_r2": null_scores[best_null],
                    "delta_r2": value - null_scores[best_null],
                    "positive_blocks": None,
                    "total_blocks": None,
                    "null_r2": dict(null_scores),
                }
            )
    return output


def block_number(block_id: str) -> int:
    return int(block_id.rsplit("_", 1)[1])


def build_double_cold_blocks(
    gnn_rows: list[dict[str, str]], cpu_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    block_ids = sorted(
        {row["block_id"] for row in cpu_rows if row["split"] == "double_cold"},
        key=block_number,
    )
    if block_ids != [f"double_cold/seed_{index}" for index in range(12)]:
        raise RuntimeError(f"unexpected double-cold block roster: {block_ids}")
    output: list[dict[str, Any]] = []
    for block_id in block_ids:
        reference = selected_rows(cpu_rows, "double_cold", "PAIR_RF", block_id)
        reference_keys = row_keys(reference)
        null_r2: dict[str, float | None] = {}
        null_status: dict[str, str] = {}
        for null in NULL_MODELS:
            rows = selected_rows(cpu_rows, "double_cold", null, block_id)
            if row_keys(rows) != reference_keys:
                raise RuntimeError(f"CPU membership mismatch for {block_id}/{null}")
            value, status = score_selection(rows, np.float64, pooled=False)
            null_r2[null] = value
            null_status[null] = status
        finite_nulls = [model for model in NULL_MODELS if null_r2[model] is not None]
        if not finite_nulls:
            raise RuntimeError(f"no finite null for {block_id}")
        best_null = max(finite_nulls, key=lambda model: float(null_r2[model]))
        model_r2: dict[str, float] = {}
        margins: dict[str, float] = {}
        positive: dict[str, bool] = {}
        for model in TARGET_MODELS:
            if model == "PAIR_RF":
                rows = reference
                dtype = np.float64
            else:
                rows = selected_rows(gnn_rows, "double_cold", model, block_id)
                dtype = np.float32
                if row_keys(rows) != reference_keys:
                    raise RuntimeError(f"GNN/CPU membership mismatch for {block_id}/{model}")
            value, status = score_selection(rows, dtype, pooled=False)
            if status != "DEFINED" or value is None:
                raise RuntimeError(f"target metric undefined for {block_id}/{model}")
            model_r2[model] = value
            margins[model] = value - float(null_r2[best_null])
            positive[model] = margins[model] > 0.0
        output.append(
            {
                "split": "double_cold",
                "block_id": block_id,
                "seed": block_number(block_id),
                "n_appearances": len(reference),
                "finite_null_count": len(finite_nulls),
                "undefined_nulls": [
                    model for model in NULL_MODELS if null_status[model] != "DEFINED"
                ],
                "null_r2": null_r2,
                "null_status": null_status,
                "best_null": best_null,
                "best_null_r2": float(null_r2[best_null]),
                "model_r2": model_r2,
                "margin_r2": margins,
                "positive_margin": positive,
            }
        )
    return output


def summarize_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = np.asarray([row["n_appearances"] for row in blocks], dtype=np.float64)
    if int(weights.sum()) != EXPECTED_POOLED_N["double_cold"]:
        raise RuntimeError(f"double-cold appearance total mismatch: {weights.sum()}")
    output: list[dict[str, Any]] = []
    for model in TARGET_MODELS:
        model_values = np.asarray([row["model_r2"][model] for row in blocks], dtype=np.float64)
        null_values = np.asarray([row["best_null_r2"] for row in blocks], dtype=np.float64)
        margins = np.asarray([row["margin_r2"][model] for row in blocks], dtype=np.float64)
        positive = int(np.count_nonzero(margins > 0.0))
        for aggregation in ("equal_block_mean", "appearance_weighted_mean"):
            if aggregation == "equal_block_mean":
                reduce = lambda values: float(np.mean(values))
            else:
                reduce = lambda values: float(np.average(values, weights=weights))
            output.append(
                {
                    "split": "double_cold",
                    "model": model,
                    "aggregation": aggregation,
                    "n_appearances": int(weights.sum()),
                    "n_blocks": len(blocks),
                    "model_r2": reduce(model_values),
                    "best_null": "PER_BLOCK_MAX_OF_FINITE_SIX",
                    "best_null_r2": reduce(null_values),
                    "delta_r2": reduce(margins),
                    "positive_blocks": positive,
                    "total_blocks": len(blocks),
                }
            )
    return output


def tsv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "split",
        "model",
        "aggregation",
        "n_appearances",
        "n_blocks",
        "model_r2",
        "best_null",
        "best_null_r2",
        "delta_r2",
        "display_delta_4dp",
        "positive_blocks",
        "total_blocks",
        "null_policy",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{key: row.get(key, "") for key in fields},
                "model_r2": format(row["model_r2"], ".17g"),
                "best_null_r2": format(row["best_null_r2"], ".17g"),
                "delta_r2": format(row["delta_r2"], ".17g"),
                "display_delta_4dp": format(row["delta_r2"], "+.4f"),
                "positive_blocks": "" if row.get("positive_blocks") is None else row["positive_blocks"],
                "total_blocks": "" if row.get("total_blocks") is None else row["total_blocks"],
                "null_policy": (
                    "largest finite production-guarded R2 among six prespecified nulls; "
                    "block comparator selected separately per block"
                    if row["aggregation"] != "pooled_test_appearances"
                    else "largest finite pooled R2 among six prespecified nulls"
                ),
            }
        )
    return buffer.getvalue()


def block_csv_text(blocks: list[dict[str, Any]]) -> str:
    fields = [
        "split",
        "block_id",
        "seed",
        "n_appearances",
        "finite_null_count",
        "undefined_nulls",
        *[f"{model}_r2" for model in NULL_MODELS],
        *[f"{model}_status" for model in NULL_MODELS],
        "best_null",
        "best_null_r2",
        *[f"{model}_r2" for model in TARGET_MODELS],
        *[f"{model}_margin_r2" for model in TARGET_MODELS],
        *[f"{model}_positive_margin" for model in TARGET_MODELS],
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in blocks:
        item: dict[str, Any] = {
            "split": row["split"],
            "block_id": row["block_id"],
            "seed": row["seed"],
            "n_appearances": row["n_appearances"],
            "finite_null_count": row["finite_null_count"],
            "undefined_nulls": ";".join(row["undefined_nulls"]),
            "best_null": row["best_null"],
            "best_null_r2": format(row["best_null_r2"], ".17g"),
        }
        for model in NULL_MODELS:
            value = row["null_r2"][model]
            item[f"{model}_r2"] = "" if value is None else format(value, ".17g")
            item[f"{model}_status"] = row["null_status"][model]
        for model in TARGET_MODELS:
            item[f"{model}_r2"] = format(row["model_r2"][model], ".17g")
            item[f"{model}_margin_r2"] = format(row["margin_r2"][model], ".17g")
            item[f"{model}_positive_margin"] = (
                "TRUE" if row["positive_margin"][model] else "FALSE"
            )
        writer.writerow(item)
    return buffer.getvalue()


def input_record(
    path: Path, rows: list[dict[str, str]], role: str, cli_flag: str
) -> dict[str, Any]:
    return {
        "role": role,
        "file": path.name,
        "cli_flag": cli_flag,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def main() -> None:
    args = parse_args()
    outputs = {
        "json": args.output_dir / f"{args.output_stem}.json",
        "tsv": args.output_dir / f"{args.output_stem}.tsv",
        "blocks": args.output_dir / args.blocks_filename,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; use --overwrite explicitly: {existing}")

    gnn_predictions = read_rows(args.gnn_predictions, ",", PREDICTION_COLUMNS)
    gnn_metrics = read_rows(args.gnn_metrics, "\t", METRIC_COLUMNS)
    cpu_predictions = read_rows(args.cpu_predictions, ",", PREDICTION_COLUMNS)
    cpu_metrics = read_rows(args.cpu_metrics, "\t", METRIC_COLUMNS)

    if {row["model"] for row in gnn_predictions} != set(GNN_MODELS):
        raise RuntimeError("canonical GNN prediction roster mismatch")
    if not set(CPU_MODELS) <= {row["model"] for row in cpu_predictions}:
        raise RuntimeError("canonical CPU prediction roster lacks target/null models")
    if {row["split"] for row in gnn_predictions} != set(SPLITS):
        raise RuntimeError("canonical GNN split roster mismatch")

    target_check = validate_target_reconstruction(
        [gnn_predictions, cpu_predictions], args.frozen_targets
    )
    metric_checks = {
        "gnn_run_a": validate_saved_metrics(
            gnn_predictions, gnn_metrics, GNN_MODELS, np.float32
        ),
        "cpu_target_and_six_nulls": validate_saved_metrics(
            cpu_predictions, cpu_metrics, CPU_MODELS, np.float64
        ),
    }
    pooled = build_pooled(gnn_predictions, cpu_predictions)
    blocks = build_double_cold_blocks(gnn_predictions, cpu_predictions)
    block_summaries = summarize_blocks(blocks)

    pair_equal = next(
        row
        for row in block_summaries
        if row["model"] == "PAIR_RF" and row["aggregation"] == "equal_block_mean"
    )
    if format(pair_equal["delta_r2"], "+.4f") != "+0.1198":
        raise RuntimeError(f"descriptor-RF +0.1198 control failed: {pair_equal}")
    if pair_equal["positive_blocks"] != 10:
        raise RuntimeError(f"descriptor-RF 10/12 control failed: {pair_equal}")
    if any(row["null_status"]["source_only_RF"] != "UNKNOWN" for row in blocks):
        raise RuntimeError("source-only RF production guard did not trigger in every double-cold block")

    summary_rows = pooled + block_summaries
    tsv = tsv_text(summary_rows)
    block_csv = block_csv_text(blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs["tsv"].write_text(tsv, encoding="utf-8", newline="")
    outputs["blocks"].write_text(block_csv, encoding="utf-8", newline="")

    script_path = Path(__file__).resolve()
    sources = {
        "gnn_predictions": input_record(
            args.gnn_predictions,
            gnn_predictions,
            "canonical GNN primary run A saved predictions",
            "--gnn-predictions",
        ),
        "gnn_metrics": input_record(
            args.gnn_metrics,
            gnn_metrics,
            "canonical GNN primary run A saved metrics",
            "--gnn-metrics",
        ),
        "cpu_predictions": input_record(
            args.cpu_predictions,
            cpu_predictions,
            "canonical CPU primary run A descriptor-RF and six-null saved predictions",
            "--cpu-predictions",
        ),
        "cpu_metrics": input_record(
            args.cpu_metrics,
            cpu_metrics,
            "canonical CPU primary run A saved metrics",
            "--cpu-metrics",
        ),
    }
    if args.frozen_targets is not None:
        sources["frozen_targets"] = {
            "role": "float32 production-target witness",
            "file": args.frozen_targets.name,
            "cli_flag": "--frozen-targets",
            "sha256": sha256(args.frozen_targets),
            "bytes": args.frozen_targets.stat().st_size,
        }
    result = {
        "state": "CANONICAL_GNN_R15_REAGGREGATION_COMPLETED",
        "script_version": SCRIPT_VERSION,
        "scope": (
            "deterministic arithmetic on frozen canonical run-A predictions; "
            "no training, refitting, fold change, null selection change, or scientific predicate change"
        ),
        "script": {"file": script_path.name, "sha256": sha256(script_path)},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "definitions": {
            "production_r2_guard": (
                "DEFINED only when y_true has more than one unique value and "
                "population SD(y_pred)>1e-9; otherwise UNKNOWN"
            ),
            "r2": "sklearn.metrics.r2_score with production dtypes",
            "pooled_margin": (
                "model pooled-test-appearance R2 minus the largest finite pooled R2 "
                "among the six prespecified nulls"
            ),
            "double_cold_block_margin": (
                "within each of 12 blocks, model R2 minus the largest finite "
                "production-guarded R2 among the same six prespecified nulls"
            ),
            "equal_block_margin": "arithmetic mean of the 12 block margins",
            "appearance_weighted_margin": (
                "sum(n_block * block_margin) / sum(n_block), with sum(n_block)=1716"
            ),
            "positive_block": "strict block_margin > 0",
            "six_null_roster": list(NULL_MODELS),
            "dtype_semantics": {
                "gnn_block": "float32 y_true and float32 y_pred",
                "cpu_block": "float32 y_true and float64 y_pred for descriptor-RF/null roster",
                "pooled": "Python-float accumulation and float64 arrays for y_true/y_pred",
                "serialized_target_reconstruction": "np.float32(float(y_true_text))",
            },
        },
        "sources": sources,
        "checks": {
            "target_reconstruction": target_check,
            "saved_metric_reproduction": metric_checks,
            "descriptor_rf_control": {
                "equal_block_margin": pair_equal["delta_r2"],
                "display_4dp": format(pair_equal["delta_r2"], "+.4f"),
                "positive_blocks": pair_equal["positive_blocks"],
                "total_blocks": pair_equal["total_blocks"],
                "pass": True,
            },
            "double_cold_source_only_guard": {
                "unknown_blocks": sum(
                    row["null_status"]["source_only_RF"] == "UNKNOWN" for row in blocks
                ),
                "total_blocks": len(blocks),
                "pass": True,
            },
        },
        "summaries": {"pooled": pooled, "double_cold_block_aggregates": block_summaries},
        "double_cold_blocks": blocks,
        "outputs": {
            "summary_tsv": {
                "file": outputs["tsv"].name,
                "sha256": sha256(outputs["tsv"]),
                "rows": len(summary_rows),
            },
            "block_csv": {
                "file": outputs["blocks"].name,
                "sha256": sha256(outputs["blocks"]),
                "rows": len(blocks),
            },
        },
    }
    outputs["json"].write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "outputs": {key: str(value) for key, value in outputs.items()},
                "descriptor_rf_control": result["checks"]["descriptor_rf_control"],
                "pooled": pooled,
                "double_cold_block_aggregates": block_summaries,
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
