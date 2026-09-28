"""Shared helpers for the public saved-result verification tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ReleaseError(RuntimeError):
    """A public release contract failed without modifying frozen inputs."""


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
        raise ReleaseError(f"JSON_OBJECT_REQUIRED: {path}")
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


def repo_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    required = [root / name for name in ("data", "results", "protocols", "scripts", "environments")]
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise ReleaseError("PUBLIC_LAYOUT_MISSING: " + ", ".join(str(path) for path in missing))
    if not (root / "protocols" / "verification_manifest.json").is_file():
        raise ReleaseError("VERIFICATION_MANIFEST_MISSING")
    return root


def manifest(root: Path) -> dict[str, Any]:
    return load_json(root / "protocols" / "verification_manifest.json")


def default_output(kind: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "runs" / f"{kind}_{stamp}_{os.getpid()}"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_output(root: Path, output: Path) -> None:
    output = output.resolve()
    runs = (root / "runs").resolve()
    if output == runs or not _inside(output, runs):
        raise ReleaseError(f"OUTPUT_MUST_BE_FRESH_RUNS_SUBDIRECTORY: {output}")


def claim_output(
    root: Path,
    output: Path,
    kind: str,
    binding: dict[str, Any],
    allow_existing_owned: bool = False,
) -> Path:
    output = Path(output).expanduser().resolve()
    assert_safe_output(root, output)
    owner_path = output / ".verification_owner.json"
    if output.exists() and any(output.iterdir()):
        if not allow_existing_owned or not owner_path.is_file():
            raise ReleaseError(f"OUTPUT_MUST_BE_NEW_OR_OWNED: {output}")
        owner = load_json(owner_path)
        if owner.get("kind") != kind or owner.get("binding") != plain(binding):
            raise ReleaseError(f"OUTPUT_OWNERSHIP_MISMATCH: {output}")
        return output
    output.mkdir(parents=True, exist_ok=True)
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


def verify_hash(path: Path, expected: str, label: str | None = None) -> None:
    if not path.is_file():
        raise ReleaseError(f"MISSING_FILE: {label or path}")
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        raise ReleaseError(f"SHA256_MISMATCH: {label or path}: expected={expected} actual={actual}")


def verify_file_spec(root: Path, relative: str, spec: dict[str, Any]) -> None:
    path = root / relative
    verify_hash(path, spec["sha256"], relative)
    if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
        raise ReleaseError(
            f"BYTE_COUNT_MISMATCH: {relative}: expected={spec['bytes']} actual={path.stat().st_size}"
        )


def eight_ulp_equal(expected: float, actual: float) -> bool:
    if math.isnan(expected) or math.isnan(actual):
        return math.isnan(expected) and math.isnan(actual)
    if math.isinf(expected) or math.isinf(actual):
        return expected == actual
    return abs(actual - expected) <= 8.0 * abs(float(np.spacing(np.float64(expected))))


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> dict[str, Any]:
    if list(expected.columns) != list(actual.columns):
        raise ReleaseError(
            f"COLUMN_MISMATCH {label}: expected={list(expected.columns)} actual={list(actual.columns)}"
        )
    if len(expected) != len(actual):
        raise ReleaseError(f"ROW_COUNT_MISMATCH {label}: expected={len(expected)} actual={len(actual)}")
    numeric_checked = 0
    categorical_checked = 0
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            for index, (left_value, right_value) in enumerate(zip(left.to_numpy(), right.to_numpy())):
                if pd.isna(left_value) or pd.isna(right_value):
                    if not (pd.isna(left_value) and pd.isna(right_value)):
                        raise ReleaseError(f"DEFINEDNESS_MISMATCH {label}:{column}:{index}")
                elif not eight_ulp_equal(float(left_value), float(right_value)):
                    raise ReleaseError(
                        f"NUMERIC_MISMATCH {label}:{column}:{index}: "
                        f"expected={left_value!r} actual={right_value!r}"
                    )
            numeric_checked += len(left)
        else:
            for index, (left_value, right_value) in enumerate(zip(left.tolist(), right.tolist())):
                if pd.isna(left_value) or pd.isna(right_value):
                    if not (pd.isna(left_value) and pd.isna(right_value)):
                        raise ReleaseError(f"DEFINEDNESS_MISMATCH {label}:{column}:{index}")
                elif str(left_value) != str(right_value):
                    raise ReleaseError(
                        f"CATEGORICAL_MISMATCH {label}:{column}:{index}: "
                        f"expected={left_value!r} actual={right_value!r}"
                    )
            categorical_checked += len(left)
    return {
        "rows": len(expected),
        "columns": len(expected.columns),
        "numeric_cells_checked": numeric_checked,
        "categorical_cells_checked": categorical_checked,
    }


def compare_csv(expected_path: Path, actual_path: Path, label: str, sep: str = ",") -> dict[str, Any]:
    expected = pd.read_csv(expected_path, sep=sep, float_precision="round_trip")
    actual = pd.read_csv(actual_path, sep=sep, float_precision="round_trip")
    return compare_frames(expected, actual, label)
