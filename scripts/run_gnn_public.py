"""Launch selected GNN jobs from the released frozen protocol bundle."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Batch

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
    required = {"rows", "null_scores", "folds", "jobs"}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise TrainingError("GNN_SAVED_BUNDLE_SCHEMA_MISMATCH")
    if not all(isinstance(value, str) and value in verified for value in bundle.values()):
        raise TrainingError("GNN_SAVED_BUNDLE_NOT_REQUIRED")
    return {key: verified[value] for key, value in bundle.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", help="Extracted repository root; defaults to this script's repository")
    parser.add_argument("--output", help="Fresh output directory; defaults to runs/gnn_<UTC>_<pid>")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", action="append", help="Exact released jobs.json key; may be repeated")
    group.add_argument("--all", action="store_true", help="Run all 350 released jobs")
    parser.add_argument("--resume", action="store_true", help="Resume only the same manifest and job binding")
    parser.add_argument("--dry-run", action="store_true", help="Check files, versions, CUDA, and resume semantics without fitting")
    args = parser.parse_args()

    root = repository_root(args.repo_root)
    contract = training_contract(root)
    protocol, verified = verify_protocol_files(root, contract, "gnn")
    bundle = _bundle_paths(protocol, verified)
    source_relative = protocol.get("model_source")
    if not isinstance(source_relative, str):
        raise TrainingError("GNN_MODEL_SOURCE_MISSING")
    source_path = verified.get(source_relative)
    if source_path is None:
        raise TrainingError("GNN_MODEL_SOURCE_NOT_REQUIRED")

    jobs_value = json.loads(bundle["jobs"].read_text(encoding="utf-8"))
    if not isinstance(jobs_value, list) or len(jobs_value) != protocol.get("job_count"):
        raise TrainingError("GNN_JOB_COUNT_MISMATCH")
    selected = select_jobs(jobs_value, args.job_id, args.all, "GNN")

    entrypoint_relative = protocol["entrypoint"]
    training_source_relative = protocol["source"]
    common_relative = "scripts/training_common.py"
    binding = {
        "protocol": "suprabench-gnn-public-v1",
        "entrypoint_sha256": file_sha(contract, entrypoint_relative),
        "training_source_sha256": file_sha(contract, training_source_relative),
        "training_common_sha256": file_sha(contract, common_relative),
        "model_source_sha256": file_sha(contract, source_relative),
        "rows_sha256": file_sha(contract, protocol["saved_bundle"]["rows"]),
        "folds_sha256": file_sha(contract, protocol["saved_bundle"]["folds"]),
        "jobs_sha256": file_sha(contract, protocol["saved_bundle"]["jobs"]),
        "jobs": [job["job_id"] for job in selected],
    }
    output = Path(args.output) if args.output else default_output(root, "gnn")
    output = claim_output(root, output, "gnn-public-training", binding, args.resume)

    module = load_module(verified[training_source_relative], "suprabench_public_gnn_training")
    module.OUT = output
    module.SOURCE = source_path
    module.SOURCE_SHA = file_sha(contract, source_relative)
    constants = protocol.get("constants")
    if not isinstance(constants, dict):
        raise TrainingError("GNN_CONSTANTS_MISSING")
    if (
        module.MAXEP != constants.get("max_epochs")
        or module.PAT != constants.get("patience")
        or module.SEEDS != constants.get("seeds")
        or module.MODELS != constants.get("models")
    ):
        raise TrainingError("GNN_SCIENTIFIC_CONSTANT_DRIFT")
    if not torch.cuda.is_available():
        raise TrainingError("CUDA_REQUIRED")
    environment = module.environment()
    versions = {
        "python": platform.python_version(),
        "torch": environment["torch"],
        "pyg": environment["pyg"],
        "numpy": environment["numpy"],
        "pandas": environment["pandas"],
        "sklearn": environment["sklearn"],
        "scipy": environment["scipy"],
        "rdkit": environment["rdkit"],
        "cuda": environment["cuda"],
    }
    verify_versions(versions, protocol.get("version_gate"), "GNN")
    selftest = module.checkpoint_selftest()
    atomic_write_json(output / "runtime_environment.json", public_environment(environment))
    atomic_write_json(
        output / "run_manifest.json",
        {
            "schema_version": "1.0",
            "claim": "new protocol-compatible run over released frozen inputs",
            "binding": binding,
            "environment": public_environment(environment),
            "checkpoint_selftest": selftest,
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
    fold_map = {(str(fold["regime"]), str(fold["fold"])): fold for fold in folds}
    namespace = module.source_namespace()
    hosts = rows.drop_duplicates("host_id").sort_values("host_id")
    guests = rows.drop_duplicates("guest_id").sort_values("guest_id")
    if hosts.host_id.tolist() != list(range(len(hosts))) or guests.guest_id.tolist() != list(range(len(guests))):
        raise TrainingError("GNN_ID_BATCH_INDEX_NOT_CONTIGUOUS")
    namespace["host_batch"] = Batch.from_data_list([namespace["to_data"](value) for value in hosts.host_smiles]).to("cuda")
    namespace["guest_batch"] = Batch.from_data_list([namespace["to_data"](value) for value in guests.guest_smiles]).to("cuda")
    lock = output / "RUNNING.json"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": binding["jobs"]}, handle)
    completed = 0
    status = "RUNNING"
    try:
        for job in selected:
            result_path = output / "jobs" / job["job_id"] / "result.json"
            if result_path.exists():
                result = load_json(result_path)
                prediction_path = result_path.parent / "predictions.csv"
                if (
                    result.get("status") != "COMPLETED"
                    or result.get("runner_sha") != binding["training_source_sha256"]
                    or result.get("predictions_sha") != sha256(prediction_path)
                ):
                    raise TrainingError(f"INVALID_EXISTING_GNN_RESULT: {job['job_id']}")
                continue
            fold = fold_map[(str(job["regime"]), str(job["fold"]))]
            if not module.train_job(namespace, rows, job, fold):
                status = "PAUSED"
                break
            completed += 1
        else:
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
