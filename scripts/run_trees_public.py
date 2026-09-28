"""Launch selected tree jobs from the released frozen feature bundle."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training_common import (
    TrainingError,
    atomic_write_json,
    claim_output,
    default_output,
    file_sha,
    load_json,
    load_module,
    plain,
    public_environment,
    repository_root,
    select_jobs,
    sha256,
    training_contract,
    verify_protocol_files,
    verify_versions,
)


def _bundle_paths(protocol: dict, verified: dict[str, Path]) -> dict[str, Path]:
    bundle = protocol.get("saved_bundle")
    required = {"features", "runtime_environment", "rows", "null_scores", "folds", "jobs"}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise TrainingError("TREE_SAVED_BUNDLE_SCHEMA_MISMATCH")
    if not all(isinstance(value, str) and value in verified for value in bundle.values()):
        raise TrainingError("TREE_SAVED_BUNDLE_NOT_REQUIRED")
    return {key: verified[value] for key, value in bundle.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", help="Extracted repository root; defaults to this script's repository")
    parser.add_argument("--output", help="Fresh output directory; defaults to runs/trees_<UTC>_<pid>")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", action="append", help="Exact released jobs.json key; may be repeated")
    group.add_argument("--all", action="store_true", help="Run all 420 released jobs")
    parser.add_argument("--dry-run", action="store_true", help="Check files and versions without fitting")
    args = parser.parse_args()

    root = repository_root(args.repo_root)
    contract = training_contract(root)
    protocol, verified = verify_protocol_files(root, contract, "trees")
    bundle = _bundle_paths(protocol, verified)
    source_relative = protocol.get("model_source")
    if not isinstance(source_relative, str):
        raise TrainingError("TREE_MODEL_SOURCE_MISSING")
    source_path = verified.get(source_relative)
    if source_path is None:
        raise TrainingError("TREE_MODEL_SOURCE_NOT_REQUIRED")

    jobs_value = json.loads(bundle["jobs"].read_text(encoding="utf-8"))
    if not isinstance(jobs_value, list) or len(jobs_value) != protocol.get("job_count"):
        raise TrainingError("TREE_JOB_COUNT_MISMATCH")
    selected = select_jobs(jobs_value, args.job_id, args.all, "TREE")

    entrypoint_relative = protocol["entrypoint"]
    training_source_relative = protocol["source"]
    common_relative = "scripts/training_common.py"
    binding = {
        "protocol": "suprabench-trees-public-v1",
        "entrypoint_sha256": file_sha(contract, entrypoint_relative),
        "training_source_sha256": file_sha(contract, training_source_relative),
        "training_common_sha256": file_sha(contract, common_relative),
        "model_source_sha256": file_sha(contract, source_relative),
        "features_sha256": file_sha(contract, protocol["saved_bundle"]["features"]),
        "folds_sha256": file_sha(contract, protocol["saved_bundle"]["folds"]),
        "jobs_sha256": file_sha(contract, protocol["saved_bundle"]["jobs"]),
        "jobs": [job["job_id"] for job in selected],
    }
    output = Path(args.output) if args.output else default_output(root, "trees")
    output = claim_output(root, output, "trees-public-training", binding)

    module = load_module(verified[training_source_relative], "suprabench_public_tree_training")
    module.OUT = output
    module.SOURCE = source_path
    module.SOURCE_SHA = file_sha(contract, source_relative)
    constants = protocol.get("constants")
    if not isinstance(constants, dict):
        raise TrainingError("TREE_CONSTANTS_MISSING")
    if module.MODELS != constants.get("models") or module.CONFIG.keys() != dict.fromkeys(module.MODELS).keys():
        raise TrainingError("TREE_SCIENTIFIC_CONSTANT_DRIFT")
    environment = module.environment()
    versions = {key: environment[key] for key in protocol.get("version_gate", {})}
    verify_versions(versions, protocol.get("version_gate"), "TREE")
    saved_environment = load_json(bundle["runtime_environment"])
    verify_versions(
        {key: saved_environment.get(key) for key in protocol.get("version_gate", {})},
        protocol.get("version_gate"),
        "TREE_SAVED",
    )
    atomic_write_json(output / "runtime_environment.json", public_environment(environment))
    atomic_write_json(
        output / "run_manifest.json",
        {
            "schema_version": "1.0",
            "claim": "new protocol-compatible run over the released frozen feature bundle",
            "binding": binding,
            "environment": public_environment(environment),
            "scientific_constants": constants,
            "limitations": contract.get("limitations", []),
            "status": "DRY_RUN_PASS" if args.dry_run else "READY",
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "jobs": len(selected), "output": str(output)}), flush=True)
        return

    rows = pd.read_csv(bundle["rows"], float_precision="round_trip")
    folds = json.loads(bundle["folds"].read_text(encoding="utf-8"))
    nulls = pd.read_csv(bundle["null_scores"], float_precision="round_trip")
    matrices = np.load(bundle["features"])
    if len(folds) != constants.get("fixed_double_cold_blocks"):
        raise TrainingError("TREE_FIXED_BLOCK_COUNT_DRIFT")
    namespace, _ = module.namespace()
    target = rows.y_true.to_numpy().astype(np.float32)
    namespace["y"] = target
    lock = output / "RUNNING.json"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": binding["jobs"]}, handle)
    completed = 0
    status = "RUNNING"
    try:
        for job in selected:
            folder = output / "jobs" / job["job_id"]
            folder.mkdir(parents=True, exist_ok=False)
            fold = folds[int(job["block"])]
            train = np.array(fold["outer_train"])
            test = np.array(fold["test"])
            fit_name, feature = module.CONFIG[job["model"]]
            print("START " + job["job_id"], flush=True)
            started = time.perf_counter()
            prediction = namespace[fit_name](matrices[feature], train, test, job["seed"])
            elapsed = time.perf_counter() - started
            if len(prediction) != len(test) or not np.isfinite(prediction).all():
                raise TrainingError(f"TREE_NONFINITE_OR_LENGTH_MISMATCH: {job['job_id']}")
            table = rows.iloc[test].copy()
            table["y_pred"] = prediction
            table["model"] = job["model"]
            table["seed"] = job["seed"]
            table["block"] = job["block"]
            temporary = folder / "predictions.csv.tmp"
            table.to_csv(temporary, index=False)
            os.replace(temporary, folder / "predictions.csv")
            atomic_write_json(folder / "membership.json", fold)
            stat = module.metrics(target[test], prediction)
            candidates = nulls[nulls.block == job["block"]].set_index("model").loc[module.NULLS]
            winner = candidates[candidates.R2.notna()].R2.idxmax()
            standard_winner = candidates.standard_R2.idxmax()
            result = {
                **job,
                "status": "COMPLETED",
                "n_train": len(train),
                "n_test": len(test),
                "elapsed_s": elapsed,
                "telemetry_available": False,
                "peak_process_working_set_bytes": None,
                "feature": feature,
                "estimator_random_state": job["seed"],
                "metrics": stat,
                "best_null": winner,
                "best_null_R2": float(candidates.loc[winner, "R2"]),
                "margin_R2": stat["R2"] - float(candidates.loc[winner, "R2"]),
                "standard_best_null": standard_winner,
                "standard_best_null_R2": float(candidates.loc[standard_winner, "standard_R2"]),
                "standard_margin_R2": stat["standard_R2"] - float(candidates.loc[standard_winner, "standard_R2"]),
                "runner_sha": binding["training_source_sha256"],
                "features_sha": binding["features_sha256"],
                "predictions_sha": sha256(folder / "predictions.csv"),
                "membership_sha": sha256(folder / "membership.json"),
            }
            atomic_write_json(folder / "result.json", result)
            print(json.dumps(plain(result)), flush=True)
            completed += 1
        status = "COMPLETED"
    except BaseException:
        status = "FAILED"
        raise
    finally:
        atomic_write_json(output / "STATUS.json", {"status": status, "pid": os.getpid(), "completed_now": completed, "selected": len(selected)})
        if lock.exists():
            os.replace(lock, output / "LAST_RUN.json")
    print(json.dumps(plain({"status": status, "completed_now": completed, "output": output})), flush=True)


if __name__ == "__main__":
    main()
