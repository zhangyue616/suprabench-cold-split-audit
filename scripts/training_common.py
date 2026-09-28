"""Minimal integrity and output helpers for the public training launchers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class TrainingError(RuntimeError):
    """The public training contract failed before a scientific output was accepted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TrainingError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repository_root(value: str | Path | None) -> Path:
    root = (
        Path(value).expanduser().resolve()
        if value is not None
        else Path(__file__).resolve().parents[1]
    )
    required = [
        root / "protocols" / "verification_manifest.json",
        root / "data",
        root / "results",
        root / "scripts",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise TrainingError("REPOSITORY_LAYOUT_MISSING: " + ", ".join(str(path) for path in missing))
    return root


def training_contract(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "protocols" / "verification_manifest.json")
    training = manifest.get("training")
    if not isinstance(training, dict) or training.get("schema_version") != "1.0":
        raise TrainingError("TRAINING_MANIFEST_SCHEMA_MISMATCH")
    if not isinstance(training.get("files"), dict):
        raise TrainingError("TRAINING_MANIFEST_FILES_MISSING")
    return training


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise TrainingError(f"INVALID_REPOSITORY_RELATIVE_PATH: {value}")
    return relative


def verify_file(root: Path, contract: dict[str, Any], relative: str) -> Path:
    relative_path = _safe_relative(relative)
    spec = contract["files"].get(relative_path.as_posix())
    if not isinstance(spec, dict):
        raise TrainingError(f"TRAINING_FILE_NOT_IN_MANIFEST: {relative_path.as_posix()}")
    expected_hash = spec.get("sha256")
    expected_bytes = spec.get("bytes")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or not isinstance(expected_bytes, int):
        raise TrainingError(f"INVALID_TRAINING_FILE_SPEC: {relative_path.as_posix()}")
    path = root / relative_path
    if not path.is_file():
        raise TrainingError(f"MISSING_TRAINING_FILE: {relative_path.as_posix()}")
    if path.stat().st_size != expected_bytes:
        raise TrainingError(
            f"BYTE_COUNT_MISMATCH: {relative_path.as_posix()}: "
            f"expected={expected_bytes} actual={path.stat().st_size}"
        )
    actual_hash = sha256(path)
    if actual_hash.lower() != expected_hash.lower():
        raise TrainingError(
            f"SHA256_MISMATCH: {relative_path.as_posix()}: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return path


def verify_protocol_files(
    root: Path,
    contract: dict[str, Any],
    protocol_name: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    protocol = contract.get(protocol_name)
    if not isinstance(protocol, dict):
        raise TrainingError(f"TRAINING_PROTOCOL_MISSING: {protocol_name}")
    required_files = protocol.get("required_files")
    if not isinstance(required_files, list):
        raise TrainingError(f"TRAINING_REQUIRED_FILES_INVALID: {protocol_name}")
    required = [
        protocol.get("entrypoint"),
        protocol.get("source"),
        "scripts/training_common.py",
        *required_files,
    ]
    if not all(isinstance(value, str) and value for value in required):
        raise TrainingError(f"TRAINING_REQUIRED_FILES_INVALID: {protocol_name}")
    unique = list(dict.fromkeys(required))
    return protocol, {relative: verify_file(root, contract, relative) for relative in unique}


def file_sha(contract: dict[str, Any], relative: str) -> str:
    spec = contract["files"].get(_safe_relative(relative).as_posix())
    if not isinstance(spec, dict) or not isinstance(spec.get("sha256"), str):
        raise TrainingError(f"TRAINING_FILE_NOT_IN_MANIFEST: {relative}")
    return spec["sha256"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TrainingError(f"CANNOT_IMPORT_TRAINING_SOURCE: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_output(root: Path, kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "runs" / f"{kind}_{stamp}_{os.getpid()}"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_safe_output(root: Path, output: Path) -> None:
    protected = [
        root,
        root / "data",
        root / "results",
        root / "scripts",
        root / "protocols",
        root / "docs",
        root / "LICENSES",
        root / ".git",
    ]
    if output == root:
        raise TrainingError("OUTPUT_MUST_NOT_BE_REPOSITORY_ROOT")
    for item in protected[1:]:
        resolved = item.resolve()
        if output == resolved or _inside(output, resolved):
            raise TrainingError(f"OUTPUT_OVERLAPS_READ_ONLY_RELEASE_CONTENT: {output}")


def claim_output(
    root: Path,
    output: Path,
    kind: str,
    binding: dict[str, Any],
    allow_matching_resume: bool = False,
) -> Path:
    output = Path(output).expanduser().resolve()
    _assert_safe_output(root, output)
    owner_path = output / ".training_owner.json"
    if output.exists():
        if not allow_matching_resume or not output.is_dir() or not owner_path.is_file():
            raise TrainingError(f"OUTPUT_MUST_BE_FRESH: {output}")
        owner = load_json(owner_path)
        if owner.get("kind") != kind or owner.get("binding") != plain(binding):
            raise TrainingError(f"OUTPUT_RESUME_BINDING_MISMATCH: {output}")
        return output
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        owner_path,
        {
            "schema_version": "1.0",
            "kind": kind,
            "binding": binding,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output


def select_jobs(
    jobs: list[dict[str, Any]],
    requested: list[str] | None,
    run_all: bool,
    label: str,
) -> list[dict[str, Any]]:
    by_id = {str(job["job_id"]): job for job in jobs}
    if len(by_id) != len(jobs):
        raise TrainingError(f"DUPLICATE_{label}_JOB_ID_IN_MANIFEST")
    if run_all:
        return jobs
    requested = requested or []
    if len(requested) != len(set(requested)):
        raise TrainingError(f"DUPLICATE_{label}_JOB_ID_REQUEST")
    missing = [job_id for job_id in requested if job_id not in by_id]
    if missing:
        raise TrainingError(f"UNKNOWN_{label}_JOB_IDS: {missing}")
    return [by_id[job_id] for job_id in requested]


def verify_versions(actual: dict[str, str], expected: Any, label: str) -> None:
    if not isinstance(expected, dict) or not expected:
        raise TrainingError(f"{label}_VERSION_GATE_MISSING")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if str(actual_value) != str(expected_value):
            raise TrainingError(
                f"{label}_ENVIRONMENT_DRIFT {key}: expected={expected_value} actual={actual_value}"
            )


def public_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducibility fields while omitting the interpreter's machine-local path."""
    return {key: value for key, value in environment.items() if key != "executable"}
