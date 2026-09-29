from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from common import (
    BUILD_ROOT,
    GateError,
    assert_file_sha,
    assert_runtime_versions,
    assert_within_build_root,
    load_config,
    project_input,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


REPORT_FIELDS = [
    "scope",
    "split",
    "block_id",
    "model",
    "metric",
    "run_a_value",
    "run_b_value",
    "comparison_value",
    "threshold",
    "metric_status",
    "acceptance_status",
    "definition",
]

FOLD_SCHEMA = [
    "split",
    "fold_index",
    "seed",
    "role",
    "row_index",
    "host_id",
    "guest_id",
    "canonical_host_sha256",
    "canonical_guest_sha256",
]
FOLD_AUTHORITIES = [
    (
        "host_cold",
        "folds/folds_host_canonical.csv",
        "A8F8D0E1B411FADCD9125A437C9B0BD3C1B454309AB4077F76E083D5A2226ED7",
    ),
    (
        "guest_cold",
        "folds/folds_guest_canonical.csv",
        "48C70058BB4D00D08C849901C288A6CC43571E6105E15D475388EBCAFDCFD23C",
    ),
    (
        "double_cold",
        "folds/folds_double_canonical.csv",
        "37D529B2F7EA7B07BB3F4EC18DB8FE8AA0C2BB1550AF6C65C41299770633DE22",
    ),
]


@dataclass(frozen=True)
class Metric:
    status: str
    value: float | None


def _strict_prediction_rows(path: Path, required_schema: list[str]) -> list[dict[str, str]]:
    resolved = assert_within_build_root(path)
    payload = resolved.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise GateError(f"prediction CSV has a UTF-8 BOM: {resolved}")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateError(f"prediction CSV is not strict UTF-8: {resolved}") from exc
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != required_schema:
            raise GateError(
                f"prediction schema mismatch for {resolved}: "
                f"expected {required_schema}, actual {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise GateError(f"prediction CSV is empty: {resolved}")
    if "\r\n" in text:
        raise GateError(f"prediction CSV is not LF-only: {resolved}")
    return rows


def _strict_fold_rows(path: Path, expected_sha: str, expected_split: str) -> list[dict[str, str]]:
    assert_file_sha(path, expected_sha)
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise GateError(f"canonical fold authority has a UTF-8 BOM: {path}")
    if b"\r\n" in payload:
        raise GateError(f"canonical fold authority is not LF-only: {path}")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateError(f"canonical fold authority is not strict UTF-8: {path}") from exc
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FOLD_SCHEMA:
            raise GateError(f"canonical fold authority schema mismatch: {path}")
        rows = list(reader)
    if not rows:
        raise GateError(f"canonical fold authority is empty: {path}")
    allowed_roles = (
        {"train", "test", "neither"}
        if expected_split == "double_cold"
        else {"train", "test"}
    )
    seen: set[tuple[int, str, int]] = set()
    for position, row in enumerate(rows):
        if row["split"] != expected_split:
            raise GateError(f"canonical fold split mismatch at {path}:{position + 2}")
        if row["role"] not in allowed_roles:
            raise GateError(f"canonical fold role mismatch at {path}:{position + 2}")
        try:
            fold_index = int(row["fold_index"])
            row_index = int(row["row_index"])
            host_id = int(row["host_id"])
            guest_id = int(row["guest_id"])
        except ValueError as exc:
            raise GateError(f"non-integer canonical membership at {path}:{position + 2}") from exc
        if any(
            [
                row["fold_index"] != str(fold_index),
                row["row_index"] != str(row_index),
                row["host_id"] != str(host_id),
                row["guest_id"] != str(guest_id),
            ]
        ):
            raise GateError(f"noncanonical integer formatting at {path}:{position + 2}")
        expected_seed = str(fold_index) if expected_split == "double_cold" else ""
        if row["seed"] != expected_seed:
            raise GateError(f"canonical fold seed mismatch at {path}:{position + 2}")
        for column in ["canonical_host_sha256", "canonical_guest_sha256"]:
            value = row[column]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise GateError(f"canonical membership hash mismatch at {path}:{position + 2}")
        key = (fold_index, row["role"], row_index)
        if key in seen:
            raise GateError(f"duplicate canonical membership key at {path}:{position + 2}")
        seen.add(key)
    return rows


def _load_row_y_authority(
    config: dict[str, Any],
) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, float]]]:
    import pandas as pd

    random_path = project_input(config, "folds_random")
    random_rows = pd.read_csv(
        random_path,
        usecols=["row_index", "host_id", "guest_id"],
        dtype=str,
        keep_default_na=False,
    )
    if len(random_rows) != config["row_authority"]["expected_rows"]:
        raise GateError("row authority count mismatch before prediction membership validation")
    authority: dict[int, tuple[str, str]] = {}
    for record in random_rows.itertuples(index=False):
        try:
            row_index = int(record.row_index)
            host_id = int(record.host_id)
            guest_id = int(record.guest_id)
        except ValueError as exc:
            raise GateError("non-integer row authority value") from exc
        if record.row_index != str(row_index):
            raise GateError("noncanonical row_index in row authority")
        if row_index in authority:
            raise GateError(f"duplicate row_index in row authority: {row_index}")
        authority[row_index] = (str(host_id), str(guest_id))

    clean_path = project_input(config, "clean_cond")
    clean_y = pd.read_csv(
        clean_path,
        usecols=["y"],
        dtype=str,
        keep_default_na=False,
    )["y"]
    float_format = config["prediction_output_contract"]["float_format"]
    y_authority: dict[int, tuple[str, float]] = {}
    for row_index in authority:
        if row_index < 0 or row_index >= len(clean_y):
            raise GateError(f"row_index outside SHA-gated clean_cond: {row_index}")
        raw = clean_y.iloc[row_index]
        numeric = _as_finite(raw, f"clean_cond y at row_index {row_index}")
        y_authority[row_index] = (format(numeric, float_format), numeric)
    return authority, y_authority


def _expected_prediction_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    authority, y_authority = _load_row_y_authority(config)
    acceptance = config["gnn_repeat_acceptance"]
    contract = config["prediction_output_contract"]
    gnn_models = [
        model for model in contract["model_order"] if model in set(acceptance["models"])
    ]
    if gnn_models != ["GINE", "AttentiveFP"]:
        raise GateError("locked GNN model ordering mismatch")

    expected: list[dict[str, str]] = []
    expected_counts = config["model_roster"]["expected_blocks_by_split"]
    for split, relative, expected_sha in FOLD_AUTHORITIES:
        fold_rows = _strict_fold_rows(BUILD_ROOT / relative, expected_sha, split)
        test_by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in fold_rows:
            if row["role"] != "test":
                continue
            row_index = int(row["row_index"])
            if row_index not in authority:
                raise GateError(
                    f"canonical test membership row_index absent from row authority: {row_index}"
                )
            expected_raw = authority[row_index]
            if (row["host_id"], row["guest_id"]) != expected_raw:
                raise GateError(
                    f"canonical fold/raw authority mismatch at {split}, row_index={row_index}"
                )
            test_by_fold[int(row["fold_index"])].append(row)
        if sorted(test_by_fold) != list(range(expected_counts[split])):
            raise GateError(f"canonical test block membership mismatch for {split}")
        for fold_index in sorted(test_by_fold):
            seed = str(fold_index) if split == "double_cold" else ""
            block_id = (
                f"double_cold/seed_{fold_index}"
                if split == "double_cold"
                else f"{split}/fold_{fold_index}"
            )
            test_rows = sorted(test_by_fold[fold_index], key=lambda row: int(row["row_index"]))
            for model in gnn_models:
                for row in test_rows:
                    row_index = int(row["row_index"])
                    expected.append(
                        {
                            "split": split,
                            "block_id": block_id,
                            "fold_index": str(fold_index),
                            "seed": seed,
                            "model": model,
                            "row_index": str(row_index),
                            "host_id": row["host_id"],
                            "guest_id": row["guest_id"],
                            "canonical_host_sha256": row["canonical_host_sha256"],
                            "canonical_guest_sha256": row["canonical_guest_sha256"],
                            "y_true": y_authority[row_index][0],
                        }
                    )
    return expected


def _validate_exact_prediction_membership(
    rows: list[dict[str, str]], expected: list[dict[str, str]], label: str
) -> None:
    if len(rows) != len(expected):
        raise GateError(
            f"prediction membership count mismatch for {label}: "
            f"expected {len(expected)}, actual {len(rows)}"
        )
    expected_columns = list(expected[0])
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for position, (actual, authority) in enumerate(zip(rows, expected, strict=True)):
        for column in expected_columns:
            if actual[column] != authority[column]:
                raise GateError(
                    f"prediction membership mismatch for {label} at position {position}, "
                    f"column {column}: expected {authority[column]!r}, actual {actual[column]!r}"
                )
        numeric_y = _as_finite(actual["y_true"], f"{label} y_true")
        if numeric_y != float(authority["y_true"]):
            raise GateError(f"numeric y_true mismatch for {label} at position {position}")
        key = (
            actual["split"],
            actual["block_id"],
            actual["fold_index"],
            actual["seed"],
            actual["model"],
            actual["row_index"],
        )
        if key in seen:
            raise GateError(f"duplicate prediction membership for {label}: {key}")
        seen.add(key)


def _as_finite(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise GateError(f"non-numeric {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise GateError(f"non-finite {label}: {value!r}")
    return number


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Metric:
    if len(y_true) == 0:
        return Metric("UNKNOWN", None)
    return Metric("DEFINED", float(np.mean(np.abs(y_true - y_pred))))


def _metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Metric:
    if len(y_true) < 2 or np.ptp(y_true) == 0.0:
        return Metric("UNKNOWN", None)
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    value = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator
    return Metric("DEFINED", value)


def _metric_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Metric:
    if len(y_true) < 2 or np.ptp(y_true) == 0.0 or np.ptp(y_pred) == 0.0:
        return Metric("UNKNOWN", None)
    value = float(spearmanr(y_true, y_pred).statistic)
    if not math.isfinite(value):
        return Metric("UNKNOWN", None)
    return Metric("DEFINED", value)


def _format(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def _threshold_row(
    *,
    scope: str,
    split: str,
    block_id: str,
    model: str,
    metric: str,
    run_a: Metric,
    run_b: Metric,
    threshold: float,
    definition: str,
) -> tuple[dict[str, Any], bool]:
    if run_a.status != run_b.status:
        status = "FAIL_ASYMMETRIC_DEFINEDNESS"
        delta: float | None = None
        failed = True
    elif run_a.status == "UNKNOWN":
        status = "UNKNOWN_SYMMETRIC"
        delta = None
        failed = False
    else:
        assert run_a.value is not None and run_b.value is not None
        delta = abs(run_a.value - run_b.value)
        failed = delta > threshold
        status = "FAIL_THRESHOLD" if failed else "PASS"
    return (
        {
            "scope": scope,
            "split": split,
            "block_id": block_id,
            "model": model,
            "metric": metric,
            "run_a_value": _format(run_a.value),
            "run_b_value": _format(run_b.value),
            "comparison_value": _format(delta),
            "threshold": _format(threshold),
            "metric_status": run_a.status if run_a.status == run_b.status else "ASYMMETRIC",
            "acceptance_status": status,
            "definition": definition,
        },
        failed,
    )


def _direct_row(
    *,
    split: str,
    block_id: str,
    model: str,
    metric: str,
    value: float,
    threshold: float,
    definition: str,
) -> tuple[dict[str, Any], bool]:
    failed = value > threshold
    return (
        {
            "scope": "model_block_prediction_repeat",
            "split": split,
            "block_id": block_id,
            "model": model,
            "metric": metric,
            "run_a_value": "",
            "run_b_value": "",
            "comparison_value": _format(value),
            "threshold": _format(threshold),
            "metric_status": "DEFINED",
            "acceptance_status": "FAIL_THRESHOLD" if failed else "PASS",
            "definition": definition,
        },
        failed,
    )


def _metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Metric]:
    return {
        "mae": _metric_mae(y_true, y_pred),
        "r2": _metric_r2(y_true, y_pred),
        "spearman": _metric_spearman(y_true, y_pred),
    }


def compare(run_a: Path, run_b: Path, output_dir: Path) -> bool:
    config = load_config()
    assert_runtime_versions(config)
    acceptance = config["gnn_repeat_acceptance"]
    schema = acceptance["required_prediction_schema"]
    contract = config["prediction_output_contract"]
    if schema != contract["columns"]:
        raise GateError("GNN schema and versioned prediction contract diverge")
    rows_a = _strict_prediction_rows(run_a, schema)
    rows_b = _strict_prediction_rows(run_b, schema)
    output_dir = assert_within_build_root(output_dir)

    if len(rows_a) != len(rows_b):
        raise GateError(f"missing or extra row: run_a={len(rows_a)}, run_b={len(rows_b)}")
    exact_columns = [column for column in schema if column != "y_pred"]
    for position, (left, right) in enumerate(zip(rows_a, rows_b, strict=True)):
        for column in exact_columns:
            if left[column] != right[column]:
                raise GateError(
                    f"row reorder or exact field mismatch at position {position}, "
                    f"column {column}: {left[column]!r} != {right[column]!r}"
                )
        y_a = _as_finite(left["y_true"], "y_true")
        y_b = _as_finite(right["y_true"], "y_true")
        if y_a != y_b:
            raise GateError(f"numeric y_true mismatch at position {position}")
        _as_finite(left["y_pred"], "run_a y_pred")
        _as_finite(right["y_pred"], "run_b y_pred")

    expected_membership = _expected_prediction_rows(config)
    _validate_exact_prediction_membership(rows_a, expected_membership, "run_a")
    _validate_exact_prediction_membership(rows_b, expected_membership, "run_b")

    allowed_models = set(acceptance["models"])
    actual_models = {row["model"] for row in rows_a}
    if actual_models != allowed_models:
        raise GateError(
            f"GNN model membership mismatch: expected {sorted(allowed_models)}, "
            f"actual {sorted(actual_models)}"
        )

    split_rank = {value: index for index, value in enumerate(contract["split_order"])}
    model_rank = {value: index for index, value in enumerate(contract["model_order"])}
    seen_keys: set[tuple[str, str, str, int]] = set()
    ordering_keys: list[tuple[int, int, int, int]] = []
    for position, row in enumerate(rows_a):
        split = row["split"]
        if split not in split_rank:
            raise GateError(f"unexpected split at position {position}: {split}")
        try:
            fold_index = int(row["fold_index"])
            row_index = int(row["row_index"])
        except ValueError as exc:
            raise GateError(f"non-integer fold_index or row_index at position {position}") from exc
        if str(fold_index) != row["fold_index"] or str(row_index) != row["row_index"]:
            raise GateError(f"noncanonical integer formatting at position {position}")
        if split == "double_cold":
            if row["seed"] != str(fold_index):
                raise GateError(f"double-cold seed/fold mismatch at position {position}")
            expected_block = f"double_cold/seed_{fold_index}"
        else:
            if row["seed"] != "":
                raise GateError(f"nonempty host/guest seed at position {position}")
            expected_block = f"{split}/fold_{fold_index}"
        if row["block_id"] != expected_block:
            raise GateError(f"block_id contract mismatch at position {position}")
        if row["model"] not in allowed_models:
            raise GateError(f"unexpected GNN model at position {position}: {row['model']}")
        for hash_column in ["canonical_host_sha256", "canonical_guest_sha256"]:
            value = row[hash_column]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise GateError(f"noncanonical lowercase SHA-256 at position {position}")
        unique_key = (split, row["block_id"], row["model"], row_index)
        if unique_key in seen_keys:
            raise GateError(f"duplicate prediction membership key: {unique_key}")
        seen_keys.add(unique_key)
        ordering_keys.append(
            (split_rank[split], fold_index, model_rank[row["model"]], row_index)
        )
    if ordering_keys != sorted(ordering_keys):
        raise GateError("prediction rows do not follow the versioned output ordering contract")

    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows_a):
        groups[(row["split"], row["block_id"], row["model"])].append(index)
    expected_counts = config["model_roster"]["expected_blocks_by_split"]
    for model in sorted(allowed_models):
        for split, expected_count in expected_counts.items():
            count = sum(1 for key in groups if key[0] == split and key[2] == model)
            if count != expected_count:
                raise GateError(
                    f"block membership mismatch for {model}/{split}: "
                    f"expected {expected_count}, actual {count}"
                )

    report_rows: list[dict[str, Any]] = []
    any_failed = False
    block_limits = acceptance["per_model_block_thresholds"]
    for split, block_id, model in sorted(groups):
        indices = groups[(split, block_id, model)]
        y_true = np.array([float(rows_a[index]["y_true"]) for index in indices])
        pred_a = np.array([float(rows_a[index]["y_pred"]) for index in indices])
        pred_b = np.array([float(rows_b[index]["y_pred"]) for index in indices])
        absolute = np.abs(pred_a - pred_b)
        direct = [
            (
                "prediction_repeat_mae",
                float(np.mean(absolute)),
                block_limits["prediction_repeat_mae_max_logKa"],
                "mean(abs(run_a y_pred - run_b y_pred))",
            ),
            (
                "prediction_repeat_rmse",
                float(np.sqrt(np.mean((pred_a - pred_b) ** 2))),
                block_limits["prediction_repeat_rmse_max_logKa"],
                "sqrt(mean((run_a y_pred - run_b y_pred)^2))",
            ),
            (
                "prediction_repeat_p95_abs",
                float(np.quantile(absolute, 0.95, method="linear")),
                block_limits["prediction_repeat_p95_abs_max_logKa"],
                "numpy.quantile(abs(run_a y_pred-run_b y_pred),0.95,method='linear')",
            ),
            (
                "prediction_repeat_max_abs",
                float(np.max(absolute)),
                block_limits["prediction_repeat_max_abs_max_logKa"],
                "max(abs(run_a y_pred - run_b y_pred))",
            ),
        ]
        for metric, value, threshold, definition in direct:
            row, failed = _direct_row(
                split=split,
                block_id=block_id,
                model=model,
                metric=metric,
                value=value,
                threshold=threshold,
                definition=definition,
            )
            report_rows.append(row)
            any_failed |= failed
        metrics_a = _metric_set(y_true, pred_a)
        metrics_b = _metric_set(y_true, pred_b)
        thresholds = {
            "mae": block_limits["absolute_delta_block_mae_max_logKa"],
            "r2": block_limits["absolute_delta_block_r2_max"],
            "spearman": block_limits["absolute_delta_block_spearman_max"],
        }
        for metric in ["mae", "r2", "spearman"]:
            row, failed = _threshold_row(
                scope="model_block_metric_delta",
                split=split,
                block_id=block_id,
                model=model,
                metric=f"absolute_delta_{metric}",
                run_a=metrics_a[metric],
                run_b=metrics_b[metric],
                threshold=thresholds[metric],
                definition=acceptance["metric_definitions"][metric],
            )
            report_rows.append(row)
            any_failed |= failed

    split_limits = acceptance["per_model_split_thresholds"]
    aggregate_thresholds = {
        "mae": split_limits["absolute_delta_aggregate_mae_max_logKa"],
        "r2": split_limits["absolute_delta_aggregate_r2_max"],
        "spearman": split_limits["absolute_delta_aggregate_spearman_max"],
    }
    for model in sorted(allowed_models):
        for split in config["model_roster"]["affected_splits"]:
            indices = [
                index
                for index, row in enumerate(rows_a)
                if row["model"] == model and row["split"] == split
            ]
            y_true = np.array([float(rows_a[index]["y_true"]) for index in indices])
            pred_a = np.array([float(rows_a[index]["y_pred"]) for index in indices])
            pred_b = np.array([float(rows_b[index]["y_pred"]) for index in indices])
            metrics_a = _metric_set(y_true, pred_a)
            metrics_b = _metric_set(y_true, pred_b)
            for metric in ["mae", "r2", "spearman"]:
                row, failed = _threshold_row(
                    scope="model_split_aggregate_delta",
                    split=split,
                    block_id="ALL_EVALUATION_APPEARANCES",
                    model=model,
                    metric=f"absolute_delta_aggregate_{metric}",
                    run_a=metrics_a[metric],
                    run_b=metrics_b[metric],
                    threshold=aggregate_thresholds[metric],
                    definition=(
                        acceptance["metric_definitions"][metric]
                        + "; pooled over all evaluation appearances in the fixed split"
                    ),
                )
                report_rows.append(row)
                any_failed |= failed

    comparison_path = output_dir / "gnn_repeat_comparison.tsv"
    write_csv_atomic(
        comparison_path,
        report_rows,
        REPORT_FIELDS,
        delimiter="\t",
    )
    write_json_atomic(
        output_dir / "gnn_repeat_comparison_manifest.json",
        {
            "schema_version": "gnn-repeat-comparison-manifest-v1",
            "execution_source_sha256": {
                "config/canonical_rebuild_config.json": sha256_file(
                    BUILD_ROOT / "config" / "canonical_rebuild_config.json"
                ),
                "scripts/common.py": sha256_file(BUILD_ROOT / "scripts" / "common.py"),
                "scripts/compare_gnn_repeats.py": sha256_file(Path(__file__).resolve()),
                **{
                    relative: sha256_file(BUILD_ROOT / relative)
                    for _, relative, _ in FOLD_AUTHORITIES
                },
            },
            "run_a": {
                "role": "PREDECLARED_PRIMARY",
                "path": run_a.resolve().relative_to(BUILD_ROOT.resolve()).as_posix(),
                "sha256": sha256_file(run_a),
            },
            "run_b": {
                "role": "VERIFICATION_ONLY",
                "path": run_b.resolve().relative_to(BUILD_ROOT.resolve()).as_posix(),
                "sha256": sha256_file(run_b),
            },
            "selection_or_averaging": "FORBIDDEN",
            "validated_prediction_schema": schema,
            "validated_membership": "EXACT_CANONICAL_AUTHORITY_AND_ORDER",
            "comparison_tsv": {
                "path": "gnn_repeat_comparison.tsv",
                "sha256": sha256_file(comparison_path),
                "bytes": comparison_path.stat().st_size,
            },
            "failed_comparisons": sum(
                row["acceptance_status"].startswith("FAIL") for row in report_rows
            ),
            "symmetric_unknown_comparisons": sum(
                row["acceptance_status"] == "UNKNOWN_SYMMETRIC" for row in report_rows
            ),
            "status": "FAIL" if any_failed else "PASS",
        },
    )
    return not any_failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed GNN repeat comparator")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        accepted = compare(args.run_a, args.run_b, args.output_dir)
    except GateError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    if not accepted:
        print("STOP: at least one prelocked GNN repeat threshold failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
