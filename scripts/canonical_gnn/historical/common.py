from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = BUILD_ROOT.parents[1]
DEFAULT_CONFIG = BUILD_ROOT / "config" / "canonical_rebuild_config.json"


class GateError(RuntimeError):
    """A fail-closed Stage-2 gate failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text_lower(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_file_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise GateError(f"required input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected.upper():
        raise GateError(
            f"SHA_MISMATCH: {path}: expected {expected.upper()}, actual {actual}"
        )


def assert_within_build_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(BUILD_ROOT.resolve()):
        raise GateError(f"write target escapes build root: {resolved}")
    return resolved


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != DEFAULT_CONFIG.resolve():
        raise GateError(f"only the versioned canonical config is accepted: {resolved}")
    payload = resolved.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise GateError("versioned canonical config has a UTF-8 BOM")
    if b"\r\n" in payload:
        raise GateError("versioned canonical config is not LF-only")
    try:
        config = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("versioned canonical config is not strict UTF-8 JSON") from exc
    if config.get("schema_version") != "canonical-rebuild-config-v1":
        raise GateError("unsupported or missing config schema_version")
    if Path(config.get("project_root", "")).resolve() != PROJECT_ROOT.resolve():
        raise GateError("config project_root does not match the script location")
    if Path(config.get("build_root", "")).resolve() != BUILD_ROOT.resolve():
        raise GateError("config build_root does not match the script location")
    if config.get("identity_policy", {}).get("policy_id") != (
        "rdkit_2026.03.3_canonical_isomeric_preserve_fragments_charge_"
        "isotopes_stereo"
    ):
        raise GateError("identity policy ID mismatch")
    return config


def project_input(config: Mapping[str, Any], key: str) -> Path:
    try:
        record = config["inputs"][key]
        path = PROJECT_ROOT / record["path"]
        expected = record["sha256"]
    except KeyError as exc:
        raise GateError(f"config input is incomplete for {key}: {exc}") from exc
    assert_file_sha(path, expected)
    return path


def assert_runtime_versions(config: Mapping[str, Any]) -> None:
    import numpy
    import pandas
    import rdkit
    import scipy
    import sklearn
    import torch
    import torch_geometric
    import xgboost
    import lightgbm

    actual = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
    }
    expected = config["runtime_versions"]
    mismatches = [
        f"{key}: expected {expected[key]}, actual {value}"
        for key, value in actual.items()
        if value != expected[key]
    ]
    if mismatches:
        raise GateError("runtime version mismatch: " + "; ".join(mismatches))


def deterministic_csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    delimiter: str = ",",
) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        delimiter=delimiter,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in fieldnames})
    return buffer.getvalue().encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    target = assert_within_build_root(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    assert_within_build_root(temporary)
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload)


def write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    delimiter: str = ",",
) -> None:
    write_bytes_atomic(path, deterministic_csv_bytes(rows, fieldnames, delimiter))


def output_manifest(root: Path) -> list[dict[str, Any]]:
    resolved = assert_within_build_root(root)
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
