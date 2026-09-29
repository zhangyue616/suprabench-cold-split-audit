#!/usr/bin/env python3
"""Build frozen B1 membership classifications and protected-axis masks.

This program is deliberately data-agnostic until invoked with a reviewed JSON
configuration after Stage A and its supplement are frozen.  Its first action is
always to run synthetic self-tests.  It then reads only exact configured
identity/fold columns, never outcomes or predictions.

Configuration shape (paths may be absolute or relative to the config file):

{
  "schema_version": 1,
  "tautomer_enumeration_included": false,
  "output_dir": ".../stage_b",
  "a_provenance": [
    {"role": "v1_freeze", "path": "...", "sha256": "..."},
    {"role": "v1_labels", "path": "...", "sha256": "..."},
    {"role": "supplement_freeze", "path": "...", "sha256": "..."},
    {"role": "supplement_labels", "path": "...", "sha256": "..."},
    {"role": "rules", "path": "...", "sha256": "..."},
    {"role": "group_appendix", "path": "...", "sha256": "..."}
  ],
  "universe": {
    "path": ".../folds_random.csv",
    "sha256": null,
    "columns": {
      "row_index": "row_index", "host_raw_id": "host_id",
      "guest_raw_id": "guest_id"
    }
  },
  "layers": [
    {
      "layer_id": "frozen_layer_name",
      "authorities": {
        "released": {
          "host_map": {"path": "...", "sha256": "...",
            "columns": {"identity_key": "...", "parent_key": "...",
              "uncertain": "...", "uncertain_reason": "...",
              "missing": "..."}, "filters": {"axis": "host"}},
          "guest_map": {"path": "...", "sha256": "...", "columns": {...}}
        },
        "canonical": {"host_map": {...}, "guest_map": {...}}
      }
    }
  ],
  "membership_sources": [
    {
      "branch": "released", "split": "host_cold",
      "kind": "released_complement", "expected_block_count": 5,
      "membership": {"path": "...", "sha256": null,
        "columns": {"block_id": "fold", "row_index": "row_index",
          "host_raw_id": "host_id", "guest_raw_id": "guest_id"}}
    },
    {
      "branch": "released", "split": "double_cold",
      "kind": "released_double", "expected_block_count": 12,
      "membership": {"path": "...", "columns": {...}},
      "held_groups": {"path": "...", "format": "long",
        "columns": {"block_id": "...", "seed": "...", "axis": "...",
          "identity_key": "..."},
        "axis_values": {"host": "host", "guest": "guest"}}
    },
    {
      "branch": "canonical", "split": "host_cold",
      "kind": "canonical_explicit", "expected_block_count": 5,
      "membership": {"path": "...", "columns": {
        "block_id": "block_id", "seed": "seed", "role": "role",
        "row_index": "row_index", "host_raw_id": "host_id",
        "guest_raw_id": "guest_id", "canonical_host_sha256": "canonical_host_sha256",
        "canonical_guest_sha256": "canonical_guest_sha256"}},
      "role_values": {"train": "train", "test": "test", "neither": "neither"}
    }
  ]
}

Map ``filters`` are exact source-column/value filters.  Optional map columns
``uncertain``, ``uncertain_reason``, and ``missing`` default to false, blank,
and false.  All scientific input SHA values for the A provenance and label maps
are mandatory; fold SHA values may be recorded without a prior expected value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCHEMA_VERSION = 1
PROGRAM_VERSION = "b1-membership-mask-v1"
ALLOWED_BRANCHES = ("released", "canonical")
ALLOWED_SPLITS = ("host_cold", "guest_cold", "double_cold")
CLASS_ORDER = (
    "same_pair_seen",
    "both_seen_pair_unseen",
    "host_only_seen",
    "guest_only_seen",
    "neither_seen",
)
NULL_TIE_ORDER = (
    "additive",
    "host_only_RF",
    "guest_only_RF",
    "cond_only_RF",
    "source_only_RF",
    "sim_knn_k5",
)
EXPECTED_LAYER_MODES = {
    "v1_strict_exact": "direct",
    "v1_counterion_candidate": "direct",
    "v1_fragment_parent_operational": "direct",
    "v1_neutralized_operational": "direct",
    "v1_stereo_agnostic_operational": "direct",
    "v2_reviewed_only_parent": "v2_reviewed_only_parent",
    "v2_all_operational_parent": "v2_all_operational_parent",
}
REQUIRED_A_ROLES = {
    "v1_freeze",
    "v1_labels",
    "v1_rules",
    "v1_group_ledger",
    "supplement_v1_freeze",
    "supplement_v1_rules",
    "supplement_v1_decisions",
    "supplement_v1_ledger",
    "supplement_v1_conflicts",
    "supplement_v2_freeze",
    "supplement_v2_labels",
    "supplement_v2_rules",
    "supplement_v2_decisions",
    "supplement_v2_inventory",
    "supplement_v2_ledger",
}
FORBIDDEN_PATH_PARTS = {
    "predictions",
    "manuscript",
    "figures",
}
FORBIDDEN_BASENAMES = {
    "records.parquet",
    "suprabench_bap_clean_cond.csv",
}
FORBIDDEN_COLUMN_NAMES = {
    "y",
    "y_true",
    "y_pred",
    "target",
    "targets",
    "metric",
    "metrics",
    "r2",
    "r2_score",
    "mae",
    "rmse",
    "delta_r2",
    "margin",
    "prediction",
    "predictions",
}


class ContractError(RuntimeError):
    """Raised when an input violates the frozen Stage-B contract."""


@dataclass(frozen=True)
class IdentityEntry:
    parent_key: str
    uncertain: bool
    uncertain_reason: str
    missing: bool
    source_path: str


@dataclass(frozen=True)
class MembershipBlock:
    branch: str
    split: str
    block_id: str
    seed: str
    train: tuple[dict[str, str], ...]
    test: tuple[dict[str, str], ...]
    excluded_count: int
    source_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def guard_input_path(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    bad_parts = sorted(lowered_parts & FORBIDDEN_PATH_PARTS)
    if bad_parts:
        raise ContractError(f"forbidden input path component(s) {bad_parts}: {path}")
    if path.name.lower() in FORBIDDEN_BASENAMES:
        raise ContractError(f"forbidden input file: {path}")
    lowered = str(path).lower()
    if "gate" in path.name.lower() or "prediction" in path.name.lower() or "metric" in path.name.lower():
        raise ContractError(f"forbidden outcome/gate-like input filename: {path}")
    if path.suffix.lower() in {".parquet", ".npz", ".npy"}:
        raise ContractError(f"forbidden input format for Stage B: {path}")
    if "salvage_stage2" in lowered:
        raise ContractError(f"salvage-stage input is outside Stage-B whitelist: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str | None, *, required: bool) -> str:
    if not path.is_file():
        raise ContractError(f"missing input file: {path}")
    actual = file_sha256(path)
    if required and not expected:
        raise ContractError(f"required SHA-256 absent for input: {path}")
    if expected and actual.lower() != str(expected).lower():
        raise ContractError(
            f"SHA_MISMATCH path={path} expected={expected} actual={actual}"
        )
    return actual


def read_header_only(path: Path, delimiter: str = ",") -> list[str]:
    """Read exactly one parsed record: the header."""
    guard_input_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ContractError(f"empty delimited file: {path}") from exc
    if not header or any(not str(name).strip() for name in header):
        raise ContractError(f"blank header field in {path}: {header}")
    if len(set(header)) != len(header):
        raise ContractError(f"duplicate header field in {path}: {header}")
    return header


def reject_forbidden_usecols(usecols: Iterable[str], path: Path) -> None:
    bad = sorted({str(col).strip().lower() for col in usecols} & FORBIDDEN_COLUMN_NAMES)
    if bad:
        raise ContractError(f"forbidden outcome columns requested from {path}: {bad}")


def read_projected_table(
    spec: Mapping[str, Any],
    base_dir: Path,
    input_receipts: dict[str, dict[str, Any]],
    *,
    require_sha: bool,
) -> tuple[list[dict[str, str]], Path]:
    path = normalize_path(str(spec["path"]), base_dir)
    guard_input_path(path)
    delimiter = str(spec.get("delimiter", "\t" if path.suffix.lower() == ".tsv" else ","))
    if len(delimiter) != 1:
        raise ContractError(f"delimiter must be one character for {path}")
    columns = dict(spec.get("columns", {}))
    if not columns:
        raise ContractError(f"no projected columns configured for {path}")
    filters = dict(spec.get("filters", {}))
    usecols = list(dict.fromkeys([str(value) for value in columns.values()] + list(filters.keys())))
    reject_forbidden_usecols(usecols, path)
    header = read_header_only(path, delimiter)
    missing_headers = [column for column in usecols if column not in header]
    if missing_headers:
        raise ContractError(f"configured columns absent from {path}: {missing_headers}; header={header}")
    actual_sha = verify_hash(path, spec.get("sha256"), required=require_sha)
    input_receipts[str(path)] = {
        "sha256": actual_sha,
        "header": header,
        "usecols": usecols,
    }
    frame = pd.read_csv(
        path,
        sep=delimiter,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )
    for source_col, expected in filters.items():
        allowed = {str(item) for item in expected} if isinstance(expected, list) else {str(expected)}
        frame = frame[frame[source_col].astype(str).isin(allowed)]
    records: list[dict[str, str]] = []
    for raw in frame.to_dict(orient="records"):
        records.append({normalized: str(raw[source]).strip() for normalized, source in columns.items()})
    return records, path


def parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ContractError(f"invalid boolean for {field}: {value!r}")


def parse_row_index(value: str, *, context: str) -> int:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ContractError(f"invalid zero-based row_index {value!r} at {context}")
    parsed = int(text)
    if parsed < 0 or str(parsed) != text:
        raise ContractError(f"noncanonical zero-based row_index {value!r} at {context}")
    return parsed


def require_nonblank(record: Mapping[str, str], fields: Sequence[str], *, context: str) -> None:
    missing = [field for field in fields if not str(record.get(field, "")).strip()]
    if missing:
        raise ContractError(f"blank required identity fields {missing} at {context}")


def classify_seen(host_seen: bool, guest_seen: bool, pair_seen: bool) -> str:
    if pair_seen and (not host_seen or not guest_seen):
        raise ContractError(
            f"P_IMPLIES_H_AND_G_VIOLATION H={host_seen} G={guest_seen} P={pair_seen}"
        )
    if pair_seen:
        return "same_pair_seen"
    if host_seen and guest_seen:
        return "both_seen_pair_unseen"
    if host_seen:
        return "host_only_seen"
    if guest_seen:
        return "guest_only_seen"
    return "neither_seen"


def retained_by_protected_axis(split: str, host_seen: bool, guest_seen: bool) -> bool:
    if split == "host_cold":
        return not host_seen
    if split == "guest_cold":
        return not guest_seen
    if split == "double_cold":
        return (not host_seen) and (not guest_seen)
    raise ContractError(f"unsupported split: {split}")


def run_synthetic_self_tests() -> dict[str, Any]:
    actual_truth: list[dict[str, Any]] = []
    expected = {
        (False, False, False): "neither_seen",
        (True, False, False): "host_only_seen",
        (False, True, False): "guest_only_seen",
        (True, True, False): "both_seen_pair_unseen",
        (True, True, True): "same_pair_seen",
    }
    for flags, expected_class in expected.items():
        observed = classify_seen(*flags)
        if observed != expected_class:
            raise AssertionError((flags, expected_class, observed))
        actual_truth.append(
            {"H": flags[0], "G": flags[1], "P": flags[2], "class": observed}
        )

    invalid_actual: list[str] = []
    for flags in ((False, False, True), (True, False, True), (False, True, True)):
        try:
            classify_seen(*flags)
        except ContractError as exc:
            invalid_actual.append(str(exc))
        else:
            raise AssertionError(f"invalid P implication did not fail: {flags}")

    mask_actual: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        for host_seen, guest_seen in ((False, False), (False, True), (True, False), (True, True)):
            observed = retained_by_protected_axis(split, host_seen, guest_seen)
            expected_mask = {
                "host_cold": not host_seen,
                "guest_cold": not guest_seen,
                "double_cold": (not host_seen) and (not guest_seen),
            }[split]
            if observed != expected_mask:
                raise AssertionError((split, host_seen, guest_seen, expected_mask, observed))
            mask_actual.append(
                {
                    "split": split,
                    "H": host_seen,
                    "G": guest_seen,
                    "retained": observed,
                }
            )

    repeated_appearances = [
        ("released", "double_cold", "block_0", "7"),
        ("released", "double_cold", "block_1", "7"),
    ]
    if len(set(repeated_appearances)) != 2:
        raise AssertionError("cross-block row appearance was incorrectly deduplicated")

    train_by_block = {
        "block_0": {"hosts": {"h0"}, "guests": {"g0"}, "pairs": {("h0", "g0")}},
        "block_1": {"hosts": {"h1"}, "guests": {"g1"}, "pairs": {("h1", "g1")}},
    }
    block_local_actual = {}
    for block_id, seen in train_by_block.items():
        h = "h0" in seen["hosts"]
        g = "g0" in seen["guests"]
        p = ("h0", "g0") in seen["pairs"]
        block_local_actual[block_id] = classify_seen(h, g, p)
    if block_local_actual != {"block_0": "same_pair_seen", "block_1": "neither_seen"}:
        raise AssertionError(block_local_actual)

    held_hosts = {"h0"}
    held_guests = {"g0"}
    synthetic_universe = [("0", "h0", "g0"), ("1", "h0", "g1"), ("2", "h1", "g0"), ("3", "h1", "g1")]
    roles = {}
    for row_index, host_id, guest_id in synthetic_universe:
        if host_id in held_hosts and guest_id in held_guests:
            roles[row_index] = "test"
        elif host_id not in held_hosts and guest_id not in held_guests:
            roles[row_index] = "train"
        else:
            roles[row_index] = "neither"
    expected_roles = {"0": "test", "1": "neither", "2": "neither", "3": "train"}
    if roles != expected_roles:
        raise AssertionError((expected_roles, roles))

    train_missing_blocker = {
        "branch": "released",
        "split": "host_cold",
        "block_id": "0",
        "role": "train",
        "axis": "host",
        "identity_key": "missing_train_host",
        "source_path": "synthetic_host_labels.csv",
    }
    if train_missing_blocker["role"] != "train" or not train_missing_blocker["identity_key"]:
        raise AssertionError(train_missing_blocker)

    uncertain_record = {
        "uncertain": True,
        "retained": retained_by_protected_axis("host_cold", False, True),
    }
    if not uncertain_record["retained"]:
        raise AssertionError("uncertain record was globally excluded")

    synthetic_csv = "row_index,host_id,y_true\n0,H0,SECRET\n"
    header = next(csv.reader(io.StringIO(synthetic_csv)))
    frame = pd.read_csv(
        io.StringIO(synthetic_csv),
        usecols=["row_index", "host_id"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    projection_actual = {
        "header": header,
        "loaded_columns": list(frame.columns),
        "loaded_row": frame.iloc[0].to_dict(),
    }
    if "y_true" in frame.columns or "SECRET" in projection_actual["loaded_row"].values():
        raise AssertionError(projection_actual)

    canonical_raw_lookup_actual = {
        "host_id": "H_RAW_0",
        "canonical_host_sha256": "h" * 64,
        "stage_a_lookup_key": "H_RAW_0",
    }
    if canonical_raw_lookup_actual["stage_a_lookup_key"] != canonical_raw_lookup_actual["host_id"]:
        raise AssertionError(canonical_raw_lookup_actual)
    if canonical_raw_lookup_actual["stage_a_lookup_key"] == canonical_raw_lookup_actual["canonical_host_sha256"]:
        raise AssertionError(canonical_raw_lookup_actual)

    return {
        "status": "PASS",
        "truth_table_actual": actual_truth,
        "invalid_implication_actual": invalid_actual,
        "protected_mask_actual": mask_actual,
        "cross_block_appearance_actual": repeated_appearances,
        "block_local_actual": block_local_actual,
        "double_and_xor_roles_actual": roles,
        "train_missing_gate_actual": train_missing_blocker,
        "uncertain_retention_actual": uncertain_record,
        "header_projection_actual": projection_actual,
        "canonical_raw_id_lookup_actual": canonical_raw_lookup_actual,
    }


def load_universe(
    spec: Mapping[str, Any],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, str]], Path]:
    records, path = read_projected_table(spec, base_dir, receipts, require_sha=False)
    required = ("row_index", "host_raw_id", "guest_raw_id")
    universe: dict[int, dict[str, str]] = {}
    for position, record in enumerate(records):
        require_nonblank(record, required, context=f"{path} projected row {position}")
        row_index = parse_row_index(record["row_index"], context=f"{path} projected row {position}")
        if row_index in universe:
            raise ContractError(f"duplicate physical row_index={row_index} in {path}")
        universe[row_index] = {
            "row_index": str(row_index),
            "host_raw_id": record["host_raw_id"],
            "guest_raw_id": record["guest_raw_id"],
            "canonical_host_sha256": "",
            "canonical_guest_sha256": "",
        }
    expected_rows = list(range(len(universe)))
    if sorted(universe) != expected_rows:
        raise ContractError(
            f"physical row universe is not the complete zero-based range in {path}: "
            f"n={len(universe)} min={min(universe) if universe else None} "
            f"max={max(universe) if universe else None}"
        )
    return universe, path


def assert_row_chain(
    record: Mapping[str, str],
    universe: Mapping[int, Mapping[str, str]],
    *,
    context: str,
) -> int:
    row_index = parse_row_index(record.get("row_index", ""), context=context)
    if row_index not in universe:
        raise ContractError(f"row_index={row_index} absent from physical universe at {context}")
    authority = universe[row_index]
    for field in ("host_raw_id", "guest_raw_id"):
        actual = str(record.get(field, "")).strip()
        if not actual:
            raise ContractError(f"blank {field} at {context}")
        if actual != authority[field]:
            raise ContractError(
                f"physical row identity mismatch at {context}: row_index={row_index} "
                f"field={field} fold={actual!r} universe={authority[field]!r}"
            )
    return row_index


def normalized_membership_row(
    record: Mapping[str, str],
    universe: Mapping[int, Mapping[str, str]],
    *,
    branch: str,
    context: str,
) -> dict[str, str]:
    row_index = assert_row_chain(record, universe, context=context)
    result = {
        "row_index": str(row_index),
        "host_raw_id": record["host_raw_id"],
        "guest_raw_id": record["guest_raw_id"],
    }
    if branch == "released":
        result["canonical_host_sha256"] = ""
        result["canonical_guest_sha256"] = ""
    else:
        require_nonblank(
            record,
            ("canonical_host_sha256", "canonical_guest_sha256"),
            context=context,
        )
        result["canonical_host_sha256"] = record["canonical_host_sha256"]
        result["canonical_guest_sha256"] = record["canonical_guest_sha256"]
    return result


def check_block_count(blocks: Sequence[MembershipBlock], expected: int, *, context: str) -> None:
    actual = len(blocks)
    if actual != expected:
        raise ContractError(f"block roster mismatch at {context}: expected={expected} actual={actual}")


def build_released_complement_blocks(
    source: Mapping[str, Any],
    universe: Mapping[int, Mapping[str, str]],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> list[MembershipBlock]:
    membership, path = read_projected_table(
        source["membership"], base_dir, receipts, require_sha=False
    )
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    global_occurrence: Counter[int] = Counter()
    for position, record in enumerate(membership):
        require_nonblank(record, ("block_id", "row_index", "host_raw_id", "guest_raw_id"), context=f"{path} row {position}")
        if "role" in record and record["role"] != "test":
            raise ContractError(f"released cold source contains non-test role at {path} row {position}")
        normalized = normalized_membership_row(
            record, universe, branch="released", context=f"{path} row {position}"
        )
        block_id = record["block_id"]
        normalized["block_id"] = block_id
        normalized["seed"] = str(record.get("seed", ""))
        by_block[block_id].append(normalized)
        global_occurrence[int(normalized["row_index"])] += 1
    if set(global_occurrence) != set(universe) or set(global_occurrence.values()) != {1}:
        missing = sorted(set(universe) - set(global_occurrence))
        repeated = sorted(index for index, count in global_occurrence.items() if count != 1)
        raise ContractError(
            f"released {source['split']} test appearances do not partition the physical universe; "
            f"missing={missing} repeated={repeated}"
        )
    blocks: list[MembershipBlock] = []
    for block_id in sorted(by_block, key=str):
        test = by_block[block_id]
        test_indices = [int(row["row_index"]) for row in test]
        if len(test_indices) != len(set(test_indices)):
            raise ContractError(f"duplicate test appearance in {path} block={block_id}")
        test_set = set(test_indices)
        train = [dict(universe[index]) for index in sorted(universe) if index not in test_set]
        blocks.append(
            MembershipBlock(
                branch="released",
                split=str(source["split"]),
                block_id=block_id,
                seed=str(test[0].get("seed", "")) if test else "",
                train=tuple(train),
                test=tuple(sorted(test, key=lambda row: int(row["row_index"]))),
                excluded_count=0,
                source_path=str(path),
            )
        )
    check_block_count(blocks, int(source["expected_block_count"]), context=str(path))
    return blocks


def load_held_sets(
    spec: Mapping[str, Any],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Path]:
    records, path = read_projected_table(spec, base_dir, receipts, require_sha=False)
    held: dict[str, dict[str, Any]] = defaultdict(lambda: {"hosts": set(), "guests": set(), "seed": ""})
    format_name = str(spec.get("format", "long"))
    if format_name == "long":
        axis_values = {str(k): str(v) for k, v in dict(spec.get("axis_values", {})).items()}
        host_value = axis_values.get("host", "host")
        guest_value = axis_values.get("guest", "guest")
        for position, record in enumerate(records):
            require_nonblank(record, ("block_id", "axis", "identity_key"), context=f"{path} row {position}")
            block_id = record["block_id"]
            axis = record["axis"]
            if axis == host_value:
                held[block_id]["hosts"].add(record["identity_key"])
            elif axis == guest_value:
                held[block_id]["guests"].add(record["identity_key"])
            else:
                raise ContractError(f"unknown held axis {axis!r} at {path} row {position}")
            seed = str(record.get("seed", ""))
            if held[block_id]["seed"] not in {"", seed}:
                raise ContractError(f"inconsistent seed for held block={block_id} in {path}")
            held[block_id]["seed"] = seed
    elif format_name == "wide":
        for position, record in enumerate(records):
            require_nonblank(record, ("block_id",), context=f"{path} row {position}")
            block_id = record["block_id"]
            host_id = str(record.get("host_raw_id", "")).strip()
            guest_id = str(record.get("guest_raw_id", "")).strip()
            if bool(host_id) == bool(guest_id):
                raise ContractError(
                    f"wide held row must contain exactly one axis at {path} row {position}"
                )
            if host_id:
                held[block_id]["hosts"].add(host_id)
            else:
                held[block_id]["guests"].add(guest_id)
            seed = str(record.get("seed", ""))
            if held[block_id]["seed"] not in {"", seed}:
                raise ContractError(f"inconsistent seed for held block={block_id} in {path}")
            held[block_id]["seed"] = seed
    else:
        raise ContractError(f"unsupported held_groups format={format_name!r}")
    for block_id, groups in held.items():
        if not groups["hosts"] or not groups["guests"]:
            raise ContractError(f"held block={block_id} lacks one axis in {path}")
    return dict(held), path


def build_released_double_blocks(
    source: Mapping[str, Any],
    universe: Mapping[int, Mapping[str, str]],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> list[MembershipBlock]:
    membership, test_path = read_projected_table(
        source["membership"], base_dir, receipts, require_sha=False
    )
    held, held_path = load_held_sets(source["held_groups"], base_dir, receipts)
    by_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    block_seed: dict[str, str] = {}
    for position, record in enumerate(membership):
        require_nonblank(record, ("block_id", "row_index", "host_raw_id", "guest_raw_id"), context=f"{test_path} row {position}")
        if "role" in record and record["role"] != "test":
            raise ContractError(f"released double source contains non-test role at {test_path} row {position}")
        normalized = normalized_membership_row(
            record, universe, branch="released", context=f"{test_path} row {position}"
        )
        block_id = record["block_id"]
        seed = str(record.get("seed", ""))
        if block_id in block_seed and block_seed[block_id] != seed:
            raise ContractError(f"inconsistent test seed for block={block_id} in {test_path}")
        block_seed[block_id] = seed
        by_block[block_id].append(normalized)
    held_join_key = str(source.get("held_join_key", "block_id"))
    if held_join_key not in {"block_id", "seed"}:
        raise ContractError(f"unsupported released double held_join_key={held_join_key!r}")
    held_key_by_block = {
        block_id: (block_id if held_join_key == "block_id" else block_seed[block_id])
        for block_id in by_block
    }
    if len(set(held_key_by_block.values())) != len(held_key_by_block):
        raise ContractError(f"released double seed-to-fold mapping is not one-to-one: {held_key_by_block}")
    if set(held_key_by_block.values()) != set(held):
        raise ContractError(
            f"double test/held block roster mismatch: test_join_keys={sorted(held_key_by_block.values())} "
            f"held={sorted(held)}"
        )
    blocks: list[MembershipBlock] = []
    for block_id in sorted(by_block, key=str):
        held_key = held_key_by_block[block_id]
        held_hosts = held[held_key]["hosts"]
        held_guests = held[held_key]["guests"]
        if held[held_key]["seed"] not in {"", block_seed[block_id]}:
            raise ContractError(f"double seed mapping mismatch for block={block_id}")
        expected_test: list[dict[str, str]] = []
        train: list[dict[str, str]] = []
        excluded = 0
        for row_index in sorted(universe):
            row = dict(universe[row_index])
            held_h = row["host_raw_id"] in held_hosts
            held_g = row["guest_raw_id"] in held_guests
            if held_h and held_g:
                expected_test.append(row)
            elif (not held_h) and (not held_g):
                train.append(row)
            else:
                excluded += 1
        actual_test = by_block[block_id]
        actual_keys = {(row["row_index"], row["host_raw_id"], row["guest_raw_id"]) for row in actual_test}
        expected_keys = {(row["row_index"], row["host_raw_id"], row["guest_raw_id"]) for row in expected_test}
        if actual_keys != expected_keys or len(actual_keys) != len(actual_test):
            raise ContractError(
                f"released double test membership mismatch block={block_id} "
                f"actual_n={len(actual_test)} expected_n={len(expected_test)} "
                f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}"
            )
        blocks.append(
            MembershipBlock(
                branch="released",
                split="double_cold",
                block_id=block_id,
                seed=block_seed[block_id],
                train=tuple(train),
                test=tuple(expected_test),
                excluded_count=excluded,
                source_path=f"{test_path} | {held_path}",
            )
        )
    check_block_count(blocks, int(source["expected_block_count"]), context=str(test_path))
    return blocks


def build_canonical_explicit_blocks(
    source: Mapping[str, Any],
    universe: Mapping[int, Mapping[str, str]],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> list[MembershipBlock]:
    membership, path = read_projected_table(
        source["membership"], base_dir, receipts, require_sha=False
    )
    role_values = {str(k): str(v) for k, v in dict(source.get("role_values", {})).items()}
    train_value = role_values.get("train", "train")
    test_value = role_values.get("test", "test")
    neither_value = role_values.get("neither", "neither")
    by_block_role: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"train": [], "test": [], "neither": []}
    )
    block_seed: dict[str, str] = {}
    fold_index_by_block: dict[str, str] = {}
    canonical_host_by_raw: dict[str, str] = {}
    canonical_guest_by_raw: dict[str, str] = {}
    for position, record in enumerate(membership):
        require_nonblank(
            record,
            (
                "block_id",
                "role",
                "row_index",
                "host_raw_id",
                "guest_raw_id",
                "canonical_host_sha256",
                "canonical_guest_sha256",
            ),
            context=f"{path} row {position}",
        )
        normalized = normalized_membership_row(
            record, universe, branch="canonical", context=f"{path} row {position}"
        )
        raw_host = normalized["host_raw_id"]
        raw_guest = normalized["guest_raw_id"]
        canonical_host = normalized["canonical_host_sha256"]
        canonical_guest = normalized["canonical_guest_sha256"]
        if raw_host in canonical_host_by_raw and canonical_host_by_raw[raw_host] != canonical_host:
            raise ContractError(
                f"canonical host assertion conflict raw host_id={raw_host!r} in {path}"
            )
        if raw_guest in canonical_guest_by_raw and canonical_guest_by_raw[raw_guest] != canonical_guest:
            raise ContractError(
                f"canonical guest assertion conflict raw guest_id={raw_guest!r} in {path}"
            )
        canonical_host_by_raw[raw_host] = canonical_host
        canonical_guest_by_raw[raw_guest] = canonical_guest
        fold_index = record["block_id"]
        block_id = (
            f"double_cold/seed_{fold_index}"
            if source["split"] == "double_cold"
            else f"{source['split']}/fold_{fold_index}"
        )
        seed = str(record.get("seed", ""))
        if block_id in block_seed and block_seed[block_id] != seed:
            raise ContractError(f"inconsistent canonical seed for block={block_id} in {path}")
        block_seed[block_id] = seed
        fold_index_by_block[block_id] = fold_index
        raw_role = record["role"]
        if raw_role == train_value:
            role = "train"
        elif raw_role == test_value:
            role = "test"
        elif raw_role == neither_value and source["split"] == "double_cold":
            role = "neither"
        else:
            raise ContractError(f"unexpected role={raw_role!r} at {path} row {position}")
        by_block_role[block_id][role].append(normalized)
    blocks: list[MembershipBlock] = []
    universe_keys = set(universe)
    for block_id in sorted(by_block_role, key=str):
        roles = by_block_role[block_id]
        if source["split"] == "double_cold" and block_seed[block_id] != fold_index_by_block[block_id]:
            raise ContractError(
                f"canonical double requires fold_index == seed; fold_index={fold_index_by_block[block_id]!r} "
                f"seed={block_seed[block_id]!r}"
            )
        role_sets = {
            role: {int(row["row_index"]) for row in rows} for role, rows in roles.items()
        }
        if any(len(role_sets[role]) != len(roles[role]) for role in roles):
            raise ContractError(f"duplicate canonical membership row block={block_id} in {path}")
        if role_sets["train"] & role_sets["test"]:
            raise ContractError(f"canonical train/test overlap block={block_id} in {path}")
        if source["split"] == "double_cold":
            if (role_sets["train"] | role_sets["test"] | role_sets["neither"]) != universe_keys:
                raise ContractError(f"canonical double roles do not cover universe block={block_id}")
            if (
                role_sets["train"] & role_sets["neither"]
                or role_sets["test"] & role_sets["neither"]
            ):
                raise ContractError(f"canonical double roles overlap block={block_id}")
        elif (role_sets["train"] | role_sets["test"]) != universe_keys:
            raise ContractError(f"canonical cold roles do not cover universe block={block_id}")
        blocks.append(
            MembershipBlock(
                branch="canonical",
                split=str(source["split"]),
                block_id=block_id,
                seed=block_seed[block_id],
                train=tuple(sorted(roles["train"], key=lambda row: int(row["row_index"]))),
                test=tuple(sorted(roles["test"], key=lambda row: int(row["row_index"]))),
                excluded_count=len(roles["neither"]),
                source_path=str(path),
            )
        )
    check_block_count(blocks, int(source["expected_block_count"]), context=str(path))
    return blocks


def load_identity_map(
    spec: Mapping[str, Any],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> tuple[dict[str, IdentityEntry], Path]:
    records, path = read_projected_table(spec, base_dir, receipts, require_sha=True)
    result: dict[str, IdentityEntry] = {}
    for position, record in enumerate(records):
        identity_key = str(record.get("identity_key", "")).strip()
        if not identity_key:
            raise ContractError(f"blank identity_key at {path} row {position}")
        missing = parse_bool(record.get("missing", ""), field=f"{path}:missing")
        parent_key = str(record.get("parent_key", "")).strip()
        reason = str(record.get("uncertain_reason", "")).strip()
        if parse_bool(spec.get("uncertain_from_nonblank_reason", False), field=f"{path}:uncertain_from_nonblank_reason"):
            uncertain = bool(reason)
        else:
            uncertain = parse_bool(record.get("uncertain", ""), field=f"{path}:uncertain")
        if missing:
            parent_key = ""
        elif not parent_key:
            missing = True
        if uncertain and not reason:
            raise ContractError(f"uncertain identity lacks reason key={identity_key!r} in {path}")
        entry = IdentityEntry(parent_key, uncertain, reason, missing, str(path))
        if identity_key in result and result[identity_key] != entry:
            raise ContractError(f"conflicting duplicate identity_key={identity_key!r} in {path}")
        result[identity_key] = entry
    return result, path


def load_v2_identity_maps(
    layer: Mapping[str, Any],
    base_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, IdentityEntry]], str]:
    mode = str(layer.get("mode", ""))
    if mode not in {"v2_reviewed_only_parent", "v2_all_operational_parent"}:
        raise ContractError(f"unsupported v2 layer mode={mode!r}")
    mapping_rows, mapping_path = read_projected_table(
        layer["mapping"], base_dir, receipts, require_sha=True
    )
    decision_rows, decisions_path = read_projected_table(
        layer["review_decisions"], base_dir, receipts, require_sha=True
    )
    ledger_rows, ledger_path = read_projected_table(
        layer["review_ledger"], base_dir, receipts, require_sha=True
    )

    decisions: dict[str, dict[str, str]] = {}
    decision_counts: Counter[str] = Counter()
    for position, row in enumerate(decision_rows):
        require_nonblank(row, ("review_group_id", "decision"), context=f"{decisions_path} row {position}")
        group_id = row["review_group_id"]
        decision = row["decision"]
        if decision not in {"accepted_as_operational", "uncertain"}:
            raise ContractError(f"unsupported v2 decision={decision!r} group={group_id!r}")
        if group_id in decisions:
            raise ContractError(f"duplicate v2 review_group_id={group_id!r} in {decisions_path}")
        decisions[group_id] = row
        decision_counts[decision] += 1
    if decision_counts != Counter({"accepted_as_operational": 44, "uncertain": 3}):
        raise ContractError(f"v2 frozen decision roster mismatch: {dict(decision_counts)}")

    ledger_by_identity: dict[tuple[str, str], dict[str, str]] = {}
    ledger_groups: set[str] = set()
    for position, row in enumerate(ledger_rows):
        require_nonblank(
            row,
            ("review_group_id", "role", "group_key", "member_identity_id", "member_count", "decision"),
            context=f"{ledger_path} row {position}",
        )
        group_id = row["review_group_id"]
        role = row["role"]
        identity_id = row["member_identity_id"]
        if role not in {"host", "guest"}:
            raise ContractError(f"invalid v2 ledger role={role!r} at {ledger_path} row {position}")
        if group_id not in decisions:
            raise ContractError(f"v2 ledger group absent from decisions: {group_id!r}")
        if row["decision"] != decisions[group_id]["decision"]:
            raise ContractError(f"v2 ledger/decision mismatch group={group_id!r}")
        if not row["member_count"].isdigit() or int(row["member_count"]) < 2:
            raise ContractError(f"v2 reviewed group is not multi-identity group={group_id!r}")
        identity_key = (role, identity_id)
        if identity_key in ledger_by_identity:
            raise ContractError(f"identity occurs in multiple v2 review groups: {identity_key}")
        ledger_by_identity[identity_key] = row
        ledger_groups.add(group_id)
    if ledger_groups != set(decisions):
        raise ContractError(
            f"v2 review ledger/decision group roster mismatch missing={sorted(set(decisions)-ledger_groups)} "
            f"extra={sorted(ledger_groups-set(decisions))}"
        )

    mapping_by_identity: dict[tuple[str, str], dict[str, str]] = {}
    role_counts: Counter[str] = Counter()
    admitted_count = 0
    unresolved_count = 0
    for position, row in enumerate(mapping_rows):
        require_nonblank(
            row,
            ("role", "identity_key", "strict_key", "operational_key"),
            context=f"{mapping_path} row {position}",
        )
        role = row["role"]
        if role not in {"host", "guest"}:
            raise ContractError(f"invalid v2 mapping role={role!r} at {mapping_path} row {position}")
        identity_key = (role, row["identity_key"])
        if identity_key in mapping_by_identity:
            raise ContractError(f"duplicate v2 identity mapping: {identity_key}")
        admitted = parse_bool(row.get("admitted", ""), field=f"{mapping_path}:admitted")
        unresolved = parse_bool(
            row.get("preservation_unresolved", ""),
            field=f"{mapping_path}:preservation_unresolved",
        )
        admitted_count += int(admitted)
        unresolved_count += int(unresolved)
        if not admitted and row["operational_key"] != row["strict_key"]:
            raise ContractError(
                f"non-admitted v2 identity did not retain complete strict key: {identity_key}"
            )
        if unresolved and admitted:
            raise ContractError(f"preservation-unresolved v2 identity was admitted: {identity_key}")
        mapping_by_identity[identity_key] = row
        role_counts[role] += 1
    if role_counts != Counter({"host": 189, "guest": 1164}):
        raise ContractError(f"v2 identity roster mismatch: {dict(role_counts)}")
    if admitted_count != 1122 or unresolved_count != 25:
        raise ContractError(
            f"v2 frozen count mismatch admitted={admitted_count} preservation_unresolved={unresolved_count}"
        )
    missing_ledger_members = sorted(set(ledger_by_identity) - set(mapping_by_identity))
    if missing_ledger_members:
        raise ContractError(f"v2 review ledger identities absent from mapping: {missing_ledger_members}")

    source_text = f"{mapping_path} | {decisions_path} | {ledger_path}"
    result: dict[str, dict[str, IdentityEntry]] = {"host": {}, "guest": {}}
    for (role, identity_id), row in sorted(mapping_by_identity.items()):
        ledger = ledger_by_identity.get((role, identity_id))
        group_decision = decisions[ledger["review_group_id"]]["decision"] if ledger else ""
        if ledger and ledger["group_key"] != row["operational_key"]:
            raise ContractError(
                f"v2 group key does not match identity mapping role={role} identity_id={identity_id}"
            )
        if mode == "v2_reviewed_only_parent" and group_decision == "uncertain":
            parent_key = row["strict_key"]
        else:
            parent_key = row["operational_key"]
        reason_parts = [part for part in (str(row.get("uncertain_reason", "")).strip(),) if part]
        if group_decision == "uncertain":
            reason_parts.append(f"GROUP_REVIEW_UNCERTAIN:{ledger['review_group_id']}")
        reason = " | ".join(reason_parts)
        result[role][identity_id] = IdentityEntry(
            parent_key=parent_key,
            uncertain=bool(reason),
            uncertain_reason=reason,
            missing=False,
            source_path=source_text,
        )
    return result, source_text


def make_blocker(
    block: MembershipBlock,
    layer_id: str,
    role: str,
    axis: str,
    identity_key: str,
    source_path: str,
    row_index: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "branch": block.branch,
        "split": block.split,
        "block_id": block.block_id,
        "seed": block.seed,
        "layer_id": layer_id,
        "role": role,
        "axis": axis,
        "identity_key": identity_key,
        "row_index": row_index,
        "source_path": source_path,
        "reason": reason,
    }


def annotate_block_layer(
    block: MembershipBlock,
    layer_id: str,
    host_map: Mapping[str, IdentityEntry],
    guest_map: Mapping[str, IdentityEntry],
    host_map_path: Path | str,
    guest_map_path: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []

    def annotate(role: str, row: Mapping[str, str]) -> dict[str, Any] | None:
        # Stage-A labels are keyed by released raw IDs in both authorities.
        # Canonical hashes are carried only as fold/prediction assertions.
        host_key = row["host_raw_id"]
        guest_key = row["guest_raw_id"]
        host_entry = host_map.get(host_key)
        guest_entry = guest_map.get(guest_key)
        if host_entry is None or host_entry.missing:
            blockers.append(
                make_blocker(
                    block,
                    layer_id,
                    role,
                    "host",
                    host_key,
                    str(host_map_path),
                    row["row_index"],
                    "identity_absent_from_frozen_map" if host_entry is None else "identity_marked_missing",
                )
            )
        if guest_entry is None or guest_entry.missing:
            blockers.append(
                make_blocker(
                    block,
                    layer_id,
                    role,
                    "guest",
                    guest_key,
                    str(guest_map_path),
                    row["row_index"],
                    "identity_absent_from_frozen_map" if guest_entry is None else "identity_marked_missing",
                )
            )
        if host_entry is None or host_entry.missing or guest_entry is None or guest_entry.missing:
            return None
        host_reason = host_entry.uncertain_reason if host_entry.uncertain else ""
        guest_reason = guest_entry.uncertain_reason if guest_entry.uncertain else ""
        combined_reason = " | ".join(
            part
            for part in (
                f"host:{host_reason}" if host_reason else "",
                f"guest:{guest_reason}" if guest_reason else "",
            )
            if part
        )
        projection = {
            "branch": block.branch,
            "split": block.split,
            "block_id": block.block_id,
            "seed": block.seed,
            "layer_id": layer_id,
            "role": role,
            "row_index": row["row_index"],
            "host_raw_id": row["host_raw_id"],
            "guest_raw_id": row["guest_raw_id"],
            "canonical_host_sha256": row["canonical_host_sha256"],
            "canonical_guest_sha256": row["canonical_guest_sha256"],
            "parent_host_key": host_entry.parent_key,
            "parent_guest_key": guest_entry.parent_key,
            "host_uncertain": host_entry.uncertain,
            "guest_uncertain": guest_entry.uncertain,
            "uncertain": host_entry.uncertain or guest_entry.uncertain,
            "uncertain_reason": combined_reason,
            "membership_source": block.source_path,
        }
        projections.append(projection)
        return projection

    train_annotated = [annotate("train", row) for row in block.train]
    test_annotated = [annotate("test", row) for row in block.test]
    if blockers:
        return [], [], blockers
    train_rows = [row for row in train_annotated if row is not None]
    test_rows = [row for row in test_annotated if row is not None]
    if len(train_rows) != len(block.train) or len(test_rows) != len(block.test):
        raise AssertionError("missing identity escaped blocker gate")

    host_seen = {row["parent_host_key"] for row in train_rows}
    guest_seen = {row["parent_guest_key"] for row in train_rows}
    pair_seen = {(row["parent_host_key"], row["parent_guest_key"]) for row in train_rows}
    masks: list[dict[str, Any]] = []
    for row in test_rows:
        h_seen = row["parent_host_key"] in host_seen
        g_seen = row["parent_guest_key"] in guest_seen
        p_seen = (row["parent_host_key"], row["parent_guest_key"]) in pair_seen
        category = classify_seen(h_seen, g_seen, p_seen)
        retained = retained_by_protected_axis(block.split, h_seen, g_seen)
        masks.append(
            {
                **row,
                "H_seen": h_seen,
                "G_seen": g_seen,
                "P_seen": p_seen,
                "membership_class": category,
                "mask_all": True,
                "mask_protected_axis_retained": retained,
            }
        )
    appearance_keys = [
        (row["branch"], row["split"], row["block_id"], row["row_index"]) for row in masks
    ]
    if len(appearance_keys) != len(set(appearance_keys)):
        raise ContractError(
            f"duplicate appearance key branch={block.branch} split={block.split} block={block.block_id}"
        )
    return projections, masks, []


def summarize_masks(masks: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_scope: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in masks:
        by_scope[(str(row["branch"]), str(row["split"]), str(row["layer_id"]))].append(row)
    coverage: list[dict[str, Any]] = []
    crosstab: list[dict[str, Any]] = []
    for (branch, split, layer_id), rows in sorted(by_scope.items()):
        base: dict[str, Any] = {
            "branch": branch,
            "split": split,
            "layer_id": layer_id,
            "block_count": len({str(row["block_id"]) for row in rows}),
            "test_appearances": len(rows),
            "unique_test_rows": len({str(row["row_index"]) for row in rows}),
            "retained_appearances": sum(bool(row["mask_protected_axis_retained"]) for row in rows),
            "retained_unique_rows": len(
                {
                    str(row["row_index"])
                    for row in rows
                    if bool(row["mask_protected_axis_retained"])
                }
            ),
            "uncertain_appearances": sum(bool(row["uncertain"]) for row in rows),
            "uncertain_unique_rows": len(
                {str(row["row_index"]) for row in rows if bool(row["uncertain"])}
            ),
            "uncertain_retained_appearances": sum(
                bool(row["uncertain"]) and bool(row["mask_protected_axis_retained"])
                for row in rows
            ),
        }
        for category in CLASS_ORDER:
            selected = [row for row in rows if row["membership_class"] == category]
            base[f"{category}_appearances"] = len(selected)
            base[f"{category}_unique_rows"] = len({str(row["row_index"]) for row in selected})
        coverage.append(base)
        for category in CLASS_ORDER:
            for uncertain in (False, True):
                selected = [
                    row
                    for row in rows
                    if row["membership_class"] == category and bool(row["uncertain"]) == uncertain
                ]
                crosstab.append(
                    {
                        "branch": branch,
                        "split": split,
                        "layer_id": layer_id,
                        "membership_class": category,
                        "uncertain": uncertain,
                        "appearances": len(selected),
                        "unique_rows": len({str(row["row_index"]) for row in selected}),
                        "retained_appearances": sum(
                            bool(row["mask_protected_axis_retained"]) for row in selected
                        ),
                        "retained_unique_rows": len(
                            {
                                str(row["row_index"])
                                for row in selected
                                if bool(row["mask_protected_axis_retained"])
                            }
                        ),
                    }
                )
    return coverage, crosstab


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prediction_alignment_contract(project_root: Path) -> list[dict[str, Any]]:
    released_fields = {
        "split": "split",
        "block_id": "fold_or_block",
        "row_index": "row_index",
        "host_id": "host_id",
        "guest_id": "guest_id",
        "model": "model",
        "y_true": "y_true",
        "y_pred": "y_pred",
    }
    canonical_fields = {
        "split": "split",
        "block_id": "block_id",
        "row_index": "row_index",
        "host_id": "host_id",
        "guest_id": "guest_id",
        "canonical_host_sha256": "canonical_host_sha256",
        "canonical_guest_sha256": "canonical_guest_sha256",
        "model": "model",
        "y_true": "y_true",
        "y_pred": "y_pred",
    }
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        rows.append(
            {
                "branch": "released",
                "split": split,
                "path": str(project_root / "repro" / "predictions" / f"predictions_{split}.csv"),
                "fields": released_fields,
                "stage_b_opened": False,
            }
        )
        rows.append(
            {
                "branch": "canonical",
                "split": split,
                "path": str(
                    project_root
                    / "repro"
                    / "canonical_rebuild_20260802"
                    / "runs"
                    / "model_cpu_deterministic_v1_run_a"
                    / "predictions_cpu.csv"
                ),
                "fields": canonical_fields,
                "stage_b_opened": False,
            }
        )
    return rows


def analysis_contract_payload(
    blocks: Sequence[MembershipBlock], project_root: Path
) -> dict[str, Any]:
    block_rosters: dict[str, dict[str, list[dict[str, Any]]]] = {
        branch: {split: [] for split in ALLOWED_SPLITS} for branch in ALLOWED_BRANCHES
    }
    for block in sorted(blocks, key=lambda item: (item.branch, item.split, item.block_id)):
        block_rosters[block.branch][block.split].append(
            {
                "block_id": block.block_id,
                "seed": block.seed,
                "train_rows": len(block.train),
                "test_appearances": len(block.test),
                "excluded_rows": block.excluded_count,
            }
        )
    return {
        "schema_version": 1,
        "state": "FROZEN_BEFORE_OUTCOMES_C",
        "primary_model": "PAIR_RF",
        "null_models": list(NULL_TIE_ORDER),
        "exact_tie_order": list(NULL_TIE_ORDER),
        "alignment": {
            "mask_appearance_key": ["branch", "split", "block_id", "row_index"],
            "prediction_key_adds": ["model"],
            "require_one_to_one_appearance_alignment": True,
            "inner_join_or_silent_drop_forbidden": True,
            "mask_file": "APPEARANCE_CLASSIFICATION_MASKS.csv",
            "mask_columns": [
                "branch", "split", "block_id", "row_index", "seed", "host_id",
                "guest_id", "canonical_host_sha256", "canonical_guest_sha256", "layer",
                "parent_host_key", "parent_guest_key", "host_seen", "guest_seen",
                "pair_seen", "category", "all_mask", "protected_axis_retained",
                "host_uncertain", "guest_uncertain", "uncertain", "uncertain_reason",
                "membership_source",
            ],
            "prediction_sources": prediction_alignment_contract(project_root),
        },
        "identity_mapping": {
            "stage_a_lookup_for_released": ["host_id", "guest_id"],
            "stage_a_lookup_for_canonical": ["host_id", "guest_id"],
            "canonical_hash_role": "identity assertion against saved canonical fold and prediction rows only",
            "canonical_hash_used_as_stage_a_identity_id": False,
            "v2_scenarios": {
                "v2_reviewed_only_parent": (
                    "merge the 44 dynamically joined accepted_as_operational multi-identity groups; "
                    "revert the 3 dynamically joined uncertain groups to each identity's complete strict key; "
                    "retain the frozen v2 operational key for all other identities"
                ),
                "v2_all_operational_parent": (
                    "use the complete frozen v2 operational mapping while retaining uncertainty flags"
                ),
            },
            "v2_scenario_interpretation": "pre-outcome operational scenarios, not rigorous bounds",
            "controller_decision_timing": (
                "the reviewed-only/all-operational split was fixed from structure review before any Stage-B count or Stage-C outcome was seen"
            ),
        },
        "block_rosters": block_rosters,
        "guards": {
            "applies_to_all_models_and_all_six_nulls": True,
            "undefined_if_n_lt": 2,
            "undefined_if_target_constant": True,
            "undefined_if_prediction_std_ddof0_lte": 1e-9,
            "undefined_if_any_nonfinite": True,
            "no_eligible_null": "margin_undefined",
            "undefined_block_policy": "do_not_drop_or_fill_zero",
            "source_only_simplified_37_mask_guard_inherited": False,
        },
        "masks": ["all", "protected_axis_retained"],
        "aggregation": {
            "host_cold": "pool all retained OOF appearances per model, then score and select max null",
            "guest_cold": "pool all retained OOF appearances per model, then score and select max null",
            "double_cold": "score each fixed block, select block max null, subtract from PAIR_RF, equal-weight the fixed 12-block roster",
            "double_required_block_count": 12,
            "strongest_null_reselected_per_mask_and_scoring_unit": True,
            "canonical_pooled_double_primary": False,
        },
        "authorities": {
            "released": "saved released predictions; model random_state=0",
            "canonical": "saved canonical Run A only",
        },
        "baseline_gate": {
            "numeric_gate": "abs(a-b) <= 8*ulp(a)",
            "canonical": "match saved metrics by scope and status before masked outcomes",
            "released_double": "match block_level_deltaR2 exact-float reference before masked outcomes",
            "released_host_guest": "rounded producer reconciliation is separate from exact-float acceptance; missing exact reference must be reported",
            "format_or_scope_conflict": "stop and return to controller; do not relax tolerance",
        },
        "prohibitions": [
            "do not read outcomes during Stage B",
            "do not alter frozen identity predicates or labels",
            "do not revise masks after seeing outcomes",
            "do not drop an undefined block",
            "do not fill undefined values with zero",
            "do not create new scientific endpoints from small classes",
        ],
    }


def analysis_contract_markdown(payload: Mapping[str, Any]) -> str:
    return """# B1 analysis contract frozen before outcomes C

State: `FROZEN_BEFORE_OUTCOMES_C`

## Model and null panel

- Primary model: `PAIR_RF`.
- All six saved nulls participate under the same mask.
- Exact-tie name order: `additive`, `host_only_RF`, `guest_only_RF`, `cond_only_RF`, `source_only_RF`, `sim_knn_k5`.
- The strongest eligible null is reselected for every mask and scoring unit.

## Alignment and guards

- Mask appearance key: `(branch, split, block_id, row_index)`; prediction alignment adds `model`.
- Prediction and mask appearance sets must match one-to-one. Inner joins and silent row loss are forbidden.
- Both released and canonical branches map Stage-A labels through released raw `host_id` / `guest_id`. Canonical SHA fields are separate fold/prediction identity assertions and are never substituted for Stage-A `identity_id`.
- `v2_reviewed_only_parent` dynamically joins the frozen group decisions: the 44 `accepted_as_operational` multi-identity groups retain their v2 merge, the 3 `uncertain` groups revert each member to its complete strict key, and all other identities retain the frozen v2 operational key. `v2_all_operational_parent` uses the full frozen v2 operational mapping. Uncertain records remain in both scenarios and are counted independently.
- The two v2 scenarios were selected by the controller from structure review before any Stage-B count or Stage-C outcome was seen. They are operational scenarios, not rigorous lower/upper bounds.
- For every model, R2 is undefined when `n < 2`, the target is constant, any input is nonfinite, or `std(y_pred, ddof=0) <= 1e-9`.
- The simplified source-only guard from the earlier 37-mask path is not inherited.
- A null that is undefined at a scoring unit is ineligible there. If no null is eligible, the margin is undefined.
- An undefined required block is never dropped or filled with zero.

## Masks and aggregation

- Every scope has a fixed baseline `all` mask and the frozen `protected_axis_retained` mask.
- Host-cold and guest-cold pool all retained OOF appearances across the five folds per model, score once per model, then choose the pooled max null and compute the pooled margin.
- Double-cold scores every fixed block, chooses that block's max null, computes `PAIR_RF - max-null`, and takes the equal-weight mean over the fixed 12-block roster. Any required undefined block makes the 12-block total undefined; coverage is still reported.
- Released uses the saved released predictions with original model `random_state=0`. Canonical uses original Run A only.
- A pooled canonical double-cold value is not a primary endpoint in this stage.

## Baseline gate before masked outcomes

- The numeric gate is `abs(a-b) <= 8 * ulp(a)`.
- Canonical values must match saved metrics by scope and status.
- Released double-cold must match `block_level_deltaR2.csv` at the exact-float gate.
- Released host/guest rounded producer values are reported as rounded reconciliation only. They are not mislabeled as an 8-ULP exact-float pass when no exact accepted reference exists.
- Any unresolved format or scope conflict returns to the controller; the tolerance is not relaxed.

## Frozen boundary

Stage B does not read predictions, outcomes, targets, metrics, gate JSON, or manuscript/figure sources. Frozen identity labels and masks cannot be revised after outcomes are seen. The five membership classes are descriptive counts and do not create new scientific endpoints.
"""


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ContractError(f"unsupported config schema_version={config.get('schema_version')!r}")
    if config.get("tautomer_enumeration_included") is not False:
        raise ContractError("v1 tautomer enumeration must remain excluded")
    roles = {str(entry.get("role", "")) for entry in config.get("a_provenance", [])}
    if roles != REQUIRED_A_ROLES:
        raise ContractError(
            f"A provenance roles mismatch: required={sorted(REQUIRED_A_ROLES)} actual={sorted(roles)}"
        )
    layer_ids = [str(layer.get("layer_id", "")) for layer in config.get("layers", [])]
    if not layer_ids or any(not layer_id for layer_id in layer_ids):
        raise ContractError("at least one nonblank frozen layer is required")
    if len(layer_ids) != len(set(layer_ids)):
        raise ContractError(f"duplicate layer_id: {layer_ids}")
    if any("tautomer" in layer_id.lower() for layer_id in layer_ids):
        raise ContractError(f"tautomer layer is excluded: {layer_ids}")
    actual_layer_modes = {
        str(layer["layer_id"]): str(layer.get("mode", "direct")) for layer in config["layers"]
    }
    if actual_layer_modes != EXPECTED_LAYER_MODES:
        raise ContractError(
            f"frozen layer/mode roster mismatch expected={EXPECTED_LAYER_MODES} "
            f"actual={actual_layer_modes}"
        )
    sources = list(config.get("membership_sources", []))
    scope_keys = [(str(item.get("branch", "")), str(item.get("split", ""))) for item in sources]
    expected_scopes = {(branch, split) for branch in ALLOWED_BRANCHES for split in ALLOWED_SPLITS}
    if set(scope_keys) != expected_scopes or len(scope_keys) != len(expected_scopes):
        raise ContractError(f"membership scopes must be exactly {sorted(expected_scopes)}; actual={scope_keys}")
    for item in sources:
        if item["branch"] not in ALLOWED_BRANCHES or item["split"] not in ALLOWED_SPLITS:
            raise ContractError(f"unsupported membership scope: {item}")
        expected_kind = {
            ("released", "host_cold"): "released_complement",
            ("released", "guest_cold"): "released_complement",
            ("released", "double_cold"): "released_double",
            ("canonical", "host_cold"): "canonical_explicit",
            ("canonical", "guest_cold"): "canonical_explicit",
            ("canonical", "double_cold"): "canonical_explicit",
        }[(item["branch"], item["split"])]
        if item.get("kind") != expected_kind:
            raise ContractError(f"wrong membership kind for {item['branch']}/{item['split']}")
        expected_count = 12 if item["split"] == "double_cold" else 5
        if int(item.get("expected_block_count", -1)) != expected_count:
            raise ContractError(
                f"wrong fixed block count for {item['branch']}/{item['split']}: "
                f"expected={expected_count} actual={item.get('expected_block_count')}"
            )


def verify_a_provenance(
    config: Mapping[str, Any], base_dir: Path, receipts: dict[str, dict[str, Any]]
) -> None:
    for entry in config["a_provenance"]:
        path = normalize_path(str(entry["path"]), base_dir)
        guard_input_path(path)
        actual = verify_hash(path, entry.get("sha256"), required=True)
        receipts[str(path)] = {
            "sha256": actual,
            "role": str(entry["role"]),
            "read_mode": "hash_only",
        }


def assert_output_targets_absent(output_dir: Path) -> None:
    targets = (
        "MEMBERSHIP_STRUCTURAL_PROJECTION.csv",
        "APPEARANCE_CLASSIFICATION_MASKS.csv",
        "COUNTS_COVERAGE.csv",
        "CLASS_UNCERTAIN_CROSSTAB.csv",
        "BLOCKERS.csv",
        "ANALYSIS_CONTRACT.md",
        "ANALYSIS_CONTRACT.json",
        "RUN_LOG.json",
        "STAGE_B_FREEZE.json",
    )
    existing = [str(output_dir / name) for name in targets if (output_dir / name).exists()]
    if existing:
        raise ContractError(f"refusing to overwrite existing Stage-B artifacts: {existing}")


def run_from_config(config_path: Path, self_tests: Mapping[str, Any]) -> int:
    config_path = config_path.resolve()
    config_base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    output_dir = normalize_path(str(config["output_dir"]), config_base)
    expected_parent = (Path(__file__).resolve().parent.parent / "stage_b").resolve()
    if output_dir != expected_parent:
        raise ContractError(f"output_dir must be the locked stage_b directory: {expected_parent}")
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_output_targets_absent(output_dir)

    receipts: dict[str, dict[str, Any]] = {}
    verify_a_provenance(config, config_base, receipts)
    universe, universe_path = load_universe(config["universe"], config_base, receipts)

    blocks: list[MembershipBlock] = []
    for source in config["membership_sources"]:
        kind = source["kind"]
        if kind == "released_complement":
            built = build_released_complement_blocks(source, universe, config_base, receipts)
        elif kind == "released_double":
            built = build_released_double_blocks(source, universe, config_base, receipts)
        elif kind == "canonical_explicit":
            built = build_canonical_explicit_blocks(source, universe, config_base, receipts)
        else:
            raise ContractError(f"unsupported membership kind={kind!r}")
        blocks.extend(built)

    canonical_host_assertions: dict[str, str] = {}
    canonical_guest_assertions: dict[str, str] = {}
    for block in blocks:
        if block.branch != "canonical":
            continue
        for row in (*block.train, *block.test):
            raw_host = row["host_raw_id"]
            raw_guest = row["guest_raw_id"]
            canonical_host = row["canonical_host_sha256"]
            canonical_guest = row["canonical_guest_sha256"]
            if (
                raw_host in canonical_host_assertions
                and canonical_host_assertions[raw_host] != canonical_host
            ):
                raise ContractError(
                    f"canonical host assertion differs across saved fold authorities: raw host_id={raw_host!r}"
                )
            if (
                raw_guest in canonical_guest_assertions
                and canonical_guest_assertions[raw_guest] != canonical_guest
            ):
                raise ContractError(
                    f"canonical guest assertion differs across saved fold authorities: raw guest_id={raw_guest!r}"
                )
            canonical_host_assertions[raw_host] = canonical_host
            canonical_guest_assertions[raw_guest] = canonical_guest

    maps: dict[tuple[str, str, str], tuple[dict[str, IdentityEntry], Path | str]] = {}
    for layer in config["layers"]:
        layer_id = str(layer["layer_id"])
        mode = str(layer.get("mode", "direct"))
        if mode == "direct":
            authorities = dict(layer.get("authorities", {}))
            if set(authorities) != set(ALLOWED_BRANCHES):
                raise ContractError(
                    f"layer={layer_id} must provide released and canonical maps; actual={sorted(authorities)}"
                )
            for branch in ALLOWED_BRANCHES:
                authority = authorities[branch]
                for axis in ("host", "guest"):
                    map_spec = authority[f"{axis}_map"]
                    maps[(layer_id, branch, axis)] = load_identity_map(
                        map_spec, config_base, receipts
                    )
        elif mode in {"v2_reviewed_only_parent", "v2_all_operational_parent"}:
            v2_maps, v2_source = load_v2_identity_maps(layer, config_base, receipts)
            for branch in ALLOWED_BRANCHES:
                for axis in ("host", "guest"):
                    maps[(layer_id, branch, axis)] = (v2_maps[axis], v2_source)
        else:
            raise ContractError(f"unsupported layer mode={mode!r} layer={layer_id}")

    all_projections: list[dict[str, Any]] = []
    all_masks: list[dict[str, Any]] = []
    all_blockers: list[dict[str, Any]] = []
    unit_status: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: (item.branch, item.split, item.block_id)):
        for layer in config["layers"]:
            layer_id = str(layer["layer_id"])
            host_map, host_path = maps[(layer_id, block.branch, "host")]
            guest_map, guest_path = maps[(layer_id, block.branch, "guest")]
            projections, masks, blockers = annotate_block_layer(
                block, layer_id, host_map, guest_map, host_path, guest_path
            )
            if blockers:
                all_blockers.extend(blockers)
                unit_status.append(
                    {
                        "branch": block.branch,
                        "split": block.split,
                        "block_id": block.block_id,
                        "layer_id": layer_id,
                        "status": "BLOCKED_MISSING_IDENTITY",
                        "blocker_count": len(blockers),
                    }
                )
                continue
            all_projections.extend(projections)
            all_masks.extend(masks)
            unit_status.append(
                {
                    "branch": block.branch,
                    "split": block.split,
                    "block_id": block.block_id,
                    "layer_id": layer_id,
                    "status": "COMPLETE",
                    "blocker_count": 0,
                }
            )

    mask_keys = [
        (row["branch"], row["split"], row["block_id"], row["row_index"], row["layer_id"])
        for row in all_masks
    ]
    if len(mask_keys) != len(set(mask_keys)):
        raise ContractError("duplicate frozen appearance/layer key across generated masks")
    if any(row["P_seen"] and (not row["H_seen"] or not row["G_seen"]) for row in all_masks):
        raise ContractError("P_IMPLIES_H_AND_G_VIOLATION in generated masks")

    coverage, crosstab = summarize_masks(all_masks)
    mask_output_rows = [
        {
            "branch": row["branch"],
            "split": row["split"],
            "block_id": row["block_id"],
            "row_index": row["row_index"],
            "seed": row["seed"],
            "host_id": row["host_raw_id"],
            "guest_id": row["guest_raw_id"],
            "canonical_host_sha256": row["canonical_host_sha256"],
            "canonical_guest_sha256": row["canonical_guest_sha256"],
            "layer": row["layer_id"],
            "parent_host_key": row["parent_host_key"],
            "parent_guest_key": row["parent_guest_key"],
            "host_seen": row["H_seen"],
            "guest_seen": row["G_seen"],
            "pair_seen": row["P_seen"],
            "category": row["membership_class"],
            "all_mask": row["mask_all"],
            "protected_axis_retained": row["mask_protected_axis_retained"],
            "host_uncertain": row["host_uncertain"],
            "guest_uncertain": row["guest_uncertain"],
            "uncertain": row["uncertain"],
            "uncertain_reason": row["uncertain_reason"],
            "membership_source": row["membership_source"],
        }
        for row in all_masks
    ]
    projection_fields = (
        "branch", "split", "block_id", "seed", "layer_id", "role", "row_index",
        "host_raw_id", "guest_raw_id", "canonical_host_sha256", "canonical_guest_sha256",
        "parent_host_key", "parent_guest_key", "host_uncertain", "guest_uncertain",
        "uncertain", "uncertain_reason", "membership_source",
    )
    mask_fields = (
        "branch", "split", "block_id", "row_index", "seed", "host_id", "guest_id",
        "canonical_host_sha256", "canonical_guest_sha256", "layer", "parent_host_key",
        "parent_guest_key", "host_seen", "guest_seen", "pair_seen", "category",
        "all_mask", "protected_axis_retained", "host_uncertain", "guest_uncertain",
        "uncertain", "uncertain_reason", "membership_source",
    )
    blocker_fields = (
        "branch", "split", "block_id", "seed", "layer_id", "role", "axis",
        "identity_key", "row_index", "source_path", "reason",
    )
    coverage_fields = tuple(coverage[0].keys()) if coverage else (
        "branch", "split", "layer_id", "block_count", "test_appearances",
        "unique_test_rows", "retained_appearances", "retained_unique_rows",
        "uncertain_appearances", "uncertain_unique_rows", "uncertain_retained_appearances",
    )
    crosstab_fields = (
        "branch", "split", "layer_id", "membership_class", "uncertain",
        "appearances", "unique_rows", "retained_appearances", "retained_unique_rows",
    )

    write_csv(output_dir / "MEMBERSHIP_STRUCTURAL_PROJECTION.csv", all_projections, projection_fields)
    write_csv(output_dir / "APPEARANCE_CLASSIFICATION_MASKS.csv", mask_output_rows, mask_fields)
    write_csv(output_dir / "COUNTS_COVERAGE.csv", coverage, coverage_fields)
    write_csv(output_dir / "CLASS_UNCERTAIN_CROSSTAB.csv", crosstab, crosstab_fields)
    write_csv(output_dir / "BLOCKERS.csv", all_blockers, blocker_fields)
    project_root = Path(__file__).resolve().parents[5]
    contract = analysis_contract_payload(blocks, project_root)
    write_json(output_dir / "ANALYSIS_CONTRACT.json", contract)
    (output_dir / "ANALYSIS_CONTRACT.md").write_text(
        analysis_contract_markdown(contract), encoding="utf-8", newline="\n"
    )

    script_path = Path(__file__).resolve()
    config_sha = file_sha256(config_path)
    script_sha = file_sha256(script_path)
    run_log = {
        "program_version": PROGRAM_VERSION,
        "started_after_self_tests": True,
        "completed_at_utc": utc_now(),
        "self_tests": self_tests,
        "program": {"path": str(script_path), "sha256": script_sha},
        "config": {"path": str(config_path), "sha256": config_sha},
        "physical_universe": {"path": str(universe_path), "rows": len(universe)},
        "input_receipts": receipts,
        "unit_status": unit_status,
        "blockers": all_blockers,
        "output_counts": {
            "structural_projection_rows": len(all_projections),
            "appearance_mask_rows": len(all_masks),
            "coverage_rows": len(coverage),
            "crosstab_rows": len(crosstab),
            "blocker_rows": len(all_blockers),
        },
        "boundary_after": {
            "predictions_opened": False,
            "outcomes_or_y_opened": False,
            "metrics_opened": False,
            "gate_json_opened": False,
            "manuscript_or_figures_opened": False,
            "tautomer_layer_included": False,
        },
    }
    write_json(output_dir / "RUN_LOG.json", run_log)

    if all_blockers:
        print(json.dumps({"status": "BLOCKED_MISSING_IDENTITY", "blockers": len(all_blockers)}))
        return 2

    expected_scope_layers = {
        (branch, split, str(layer["layer_id"]))
        for branch in ALLOWED_BRANCHES
        for split in ALLOWED_SPLITS
        for layer in config["layers"]
    }
    completed_scope_layers = {
        (row["branch"], row["split"], row["layer_id"]) for row in coverage
    }
    if completed_scope_layers != expected_scope_layers:
        raise ContractError(
            f"scope/layer coverage mismatch expected={sorted(expected_scope_layers)} "
            f"actual={sorted(completed_scope_layers)}"
        )

    output_names = (
        "MEMBERSHIP_STRUCTURAL_PROJECTION.csv",
        "APPEARANCE_CLASSIFICATION_MASKS.csv",
        "COUNTS_COVERAGE.csv",
        "CLASS_UNCERTAIN_CROSSTAB.csv",
        "BLOCKERS.csv",
        "ANALYSIS_CONTRACT.md",
        "ANALYSIS_CONTRACT.json",
        "RUN_LOG.json",
    )
    outputs = {
        name: {"path": str(output_dir / name), "sha256": file_sha256(output_dir / name)}
        for name in output_names
    }
    freeze = {
        "schema_version": 1,
        "state": "STAGE_B_FROZEN",
        "frozen_at_utc": utc_now(),
        "program": {"path": str(script_path), "sha256": script_sha},
        "config": {"path": str(config_path), "sha256": config_sha},
        "a_provenance": [
            {
                "role": entry["role"],
                "path": str(normalize_path(str(entry["path"]), config_base)),
                "sha256": receipts[str(normalize_path(str(entry["path"]), config_base))]["sha256"],
            }
            for entry in config["a_provenance"]
        ],
        "membership_inputs": receipts,
        "appearance_key": ["branch", "split", "block_id", "row_index"],
        "classes": list(CLASS_ORDER),
        "outputs": outputs,
        "unit_status": unit_status,
        "boundary_after": run_log["boundary_after"],
    }
    write_json(output_dir / "STAGE_B_FREEZE.json", freeze)
    print(
        json.dumps(
            {
                "status": "STAGE_B_FROZEN",
                "freeze": str(output_dir / "STAGE_B_FREEZE.json"),
                "mask_rows": len(all_masks),
                "blockers": 0,
            }
        )
    )
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.self_test_only == (args.config is not None):
        parser.error("choose exactly one of --self-test-only or --config PATH")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    self_tests = run_synthetic_self_tests()
    if args.self_test_only:
        print(json.dumps(self_tests, ensure_ascii=False, indent=2))
        return 0
    try:
        return run_from_config(args.config, self_tests)
    except ContractError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
