"""Verify the four public saved-result science groups without fitting models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn

from reaggregate_parent_identity_b1 import execute_after_freeze as reaggregate_parent_identity_b1
from reaggregate_saved_results import reaggregate_gnn, reaggregate_trees
from release_common import (
    ReleaseError,
    atomic_write_json,
    claim_output,
    default_output,
    eight_ulp_equal,
    load_json,
    manifest,
    plain,
    repo_root,
    verify_file_spec,
    verify_hash,
)


class Checks:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(self, check_id: str, function: Callable[[], dict[str, Any]]) -> None:
        try:
            details = function()
            self.records.append({"check_id": check_id, "status": "PASS", "details": details})
            print(f"PASS {check_id}", flush=True)
        except BaseException as error:
            self.records.append(
                {
                    "check_id": check_id,
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"FAIL {check_id}: {type(error).__name__}: {error}", flush=True)


def _row_count(path: Path, delimiter: str) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        next(reader)
        return sum(1 for _ in reader)


def _explicit_nonfinite_tokens(path: Path, delimiter: str) -> int:
    count = 0
    tokens = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            count += sum(cell.strip().lower() in tokens for cell in row)
    return count


def verify_environment(spec: dict[str, Any]) -> dict[str, str]:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "rdkit": rdkit.__version__,
    }
    expected = spec["environments"]["verify"]
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ReleaseError(f"VERIFY_ENVIRONMENT_DRIFT {key}: expected={value} actual={observed.get(key)}")
    return observed


def verify_cpu(root: Path, group: dict[str, Any]) -> dict[str, Any]:
    for relative, file_spec in group["files"].items():
        verify_file_spec(root, relative, file_spec)

    feature_manifest_path = root / group["feature_manifest_path"]
    features = load_json(feature_manifest_path)
    if int(features["row_count"]) != int(group["counts"]["feature_rows"]):
        raise ReleaseError("CPU_FEATURE_ROW_COUNT_MISMATCH")
    for artifact in features["artifacts_excluding_manifest"]:
        path = feature_manifest_path.parent / artifact["path"]
        verify_hash(path, artifact["sha256"], f"canonical_feature/{artifact['path']}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise ReleaseError(f"FEATURE_SIZE_MISMATCH: {artifact['path']}")

    run_a = root / group["run_a"]
    run_b = root / group["run_b"]
    artifacts = group["run_artifacts"]
    for filename in artifacts:
        if (run_a / filename).read_bytes() != (run_b / filename).read_bytes():
            raise ReleaseError(f"CPU_AB_NOT_BYTE_IDENTICAL: {filename}")

    actual_counts = {
        "predictions_per_run": _row_count(run_a / "predictions_cpu.csv", ","),
        "metrics_per_run": _row_count(run_a / "metrics_cpu.tsv", "\t"),
        "similarity_per_run": _row_count(run_a / "max_pair_similarity.tsv", "\t"),
    }
    for key, actual in actual_counts.items():
        if actual != int(group["counts"][key]):
            raise ReleaseError(
                f"CPU_ROW_COUNT_MISMATCH {key}: expected={group['counts'][key]} actual={actual}"
            )
    nonfinite = sum(
        _explicit_nonfinite_tokens(run_a / name, "\t" if name.endswith(".tsv") else ",")
        for name in artifacts
    )
    if nonfinite != int(group["counts"]["nan_inf"]):
        raise ReleaseError(
            f"CPU_NONFINITE_TOKEN_COUNT_MISMATCH: expected={group['counts']['nan_inf']} actual={nonfinite}"
        )

    with (run_a / "metrics_cpu.tsv").open(encoding="utf-8", newline="") as handle:
        run_rows = list(csv.reader(handle, delimiter="\t"))
    with (root / group["aggregate_audit_path"]).open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.reader(handle, delimiter="\t"))
    projection = [run_rows[0]] + [row for row in run_rows[1:] if row[0] == "model_split_aggregate"]
    expected_audit_rows = int(group["counts"]["audit_projection_rows"])
    if len(projection) - 1 != expected_audit_rows or projection != audit_rows:
        raise ReleaseError(
            f"CPU_AGGREGATE_AUDIT_PROJECTION_MISMATCH: expected={expected_audit_rows} "
            f"actual={len(projection)-1}"
        )
    return {
        **actual_counts,
        "feature_artifacts": len(features["artifacts_excluding_manifest"]),
        "audit_projection_rows": len(projection) - 1,
        "ab_byte_identical_artifacts": len(artifacts),
    }


def _compare_b1_final(expected_path: Path, actual_path: Path) -> dict[str, int]:
    expected = pd.read_csv(expected_path, sep="\t", dtype=str, keep_default_na=False)
    actual = pd.read_csv(actual_path, sep="\t", dtype=str, keep_default_na=False)
    if list(expected.columns) != list(actual.columns):
        raise ReleaseError("B1_FINAL_COLUMN_MISMATCH")
    keys = ["branch", "split", "scenario"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if len(expected) != len(actual):
        raise ReleaseError(f"B1_FINAL_ROW_COUNT_MISMATCH: expected={len(expected)} actual={len(actual)}")
    numeric = 0
    categorical = 0
    for column in expected.columns:
        for index, (left, right) in enumerate(zip(expected[column], actual[column])):
            if left == "UNDEFINED" or right == "UNDEFINED":
                if left != right:
                    raise ReleaseError(f"B1_DEFINEDNESS_MISMATCH:{column}:{index}")
                categorical += 1
                continue
            try:
                left_number = float(left)
                right_number = float(right)
            except ValueError:
                if left != right:
                    raise ReleaseError(
                        f"B1_CATEGORICAL_MISMATCH:{column}:{index}: expected={left!r} actual={right!r}"
                    )
                categorical += 1
            else:
                if not eight_ulp_equal(left_number, right_number):
                    raise ReleaseError(
                        f"B1_NUMERIC_MISMATCH:{column}:{index}: expected={left!r} actual={right!r}"
                    )
                numeric += 1
    return {"rows": len(expected), "numeric_cells_checked": numeric, "categorical_cells_checked": categorical}


def verify_b1(root: Path, output: Path, group: dict[str, Any]) -> dict[str, Any]:
    for relative, file_spec in group["static_files"].items():
        verify_file_spec(root, relative, file_spec)
    reaggregate_parent_identity_b1(output_dir=output, package_root=root)
    summary = load_json(output / "SUMMARY.json")
    if summary.get("status") != "COMPLETED" or summary.get("blocked_branches"):
        raise ReleaseError(f"B1_REPLAY_NOT_COMPLETE: {summary.get('status')}")
    primary = pd.read_csv(output / "PRIMARY_COMPARISONS.tsv", sep="\t", keep_default_na=False)
    checks = pd.read_csv(output / "BASELINE_CHECKS.tsv", sep="\t", dtype=str, keep_default_na=False)
    if len(primary) != int(group["expected_primary_rows"]):
        raise ReleaseError(f"B1_PRIMARY_ROW_COUNT_MISMATCH: {len(primary)}")
    if len(checks) != int(group["expected_baseline_checks"]):
        raise ReleaseError(f"B1_BASELINE_CHECK_COUNT_MISMATCH: {len(checks)}")
    if not checks["pass"].str.lower().eq("true").all():
        failed = int((~checks["pass"].str.lower().eq("true")).sum())
        raise ReleaseError(f"B1_BASELINE_CHECK_FAILURES: {failed}")
    comparison = _compare_b1_final(root / group["expected_final_results"], output / "PRIMARY_COMPARISONS.tsv")
    return {
        "status": summary["status"],
        "primary_rows": len(primary),
        "baseline_checks": len(checks),
        "final_results_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Public repository root")
    parser.add_argument("--output", help="New verification output directory")
    args = parser.parse_args()

    root = repo_root(args.root)
    spec = manifest(root)
    output = Path(args.output) if args.output else default_output("verify")
    output = claim_output(root, output, "saved-results-verification", {"root_layout": spec["root_layout"]})

    try:
        environment = verify_environment(spec)
    except BaseException as error:
        report = {
            "schema_version": "1.0",
            "status": "FAIL",
            "claim": "saved-results verification only; no fitting or prediction generation",
            "environment_preflight": {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            "science_groups": [],
            "summary": {"total": 4, "passed": 0, "failed": 4, "not_run_due_to_environment": 4},
        }
        atomic_write_json(output / "verification_report.json", report)
        print(json.dumps(plain({"status": "FAIL", "reason": "environment", "output": output})), flush=True)
        raise SystemExit(1)

    groups = spec["scientific_groups"]
    checks = Checks()
    checks.run("canonical_cpu_saved", lambda: verify_cpu(root, groups["canonical_cpu_saved"]))
    checks.run(
        "gnn_saved_reaggregation",
        lambda: reaggregate_gnn(root, output / "gnn", compare_historical=True),
    )
    checks.run(
        "tree_saved_reaggregation",
        lambda: reaggregate_trees(root, output / "trees", compare_historical=True),
    )
    checks.run(
        "parent_identity_b1_saved_replay",
        lambda: verify_b1(root, output / "parent_identity_b1", groups["parent_identity_b1_saved_replay"]),
    )
    failed = [record["check_id"] for record in checks.records if record["status"] != "PASS"]
    report = {
        "schema_version": "1.0",
        "status": "PASS" if not failed else "FAIL",
        "claim": "saved-results verification only; no fitting, prediction generation, or historical recovery",
        "root_layout": spec["root_layout"],
        "environment_preflight": {"status": "PASS", "observed": environment},
        "science_groups": checks.records,
        "coverage": spec["coverage"],
        "summary": {
            "total": len(checks.records),
            "passed": len(checks.records) - len(failed),
            "failed": len(failed),
        },
    }
    atomic_write_json(output / "verification_report.json", report)
    print(json.dumps(plain({"status": report["status"], "failed": failed, "output": output})), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
