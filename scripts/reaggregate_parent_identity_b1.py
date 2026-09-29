"""B1 fixed-prediction scoring after independently frozen Stage B masks.

No fitting, fold generation, structure canonicalization or network calls.
All-model guards follow the original producer contract, not the narrower
source-only compatibility guard in the historical 37-appearance derivative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

NULL_ORDER = [
    "additive", "host_only_RF", "guest_only_RF", "cond_only_RF",
    "source_only_RF", "sim_knn_k5",
]
MODELS = ["PAIR_RF"] + NULL_ORDER
LAYERS = ["v1_strict_exact", "v1_counterion_candidate", "v1_fragment_parent_operational",
          "v1_neutralized_operational", "v1_stereo_agnostic_operational",
          "v2_reviewed_only_parent", "v2_all_operational_parent"]
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parent.parent
B1_RESULTS = REPO_ROOT / "results" / "parent_identity_b1"
B1_PROTOCOL = REPO_ROOT / "protocols" / "parent_identity_b1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        h = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def validate_package_root(value):
    if not value:
        raise ValueError("PACKAGE_ROOT_REQUIRED")
    root = Path(value).resolve()
    required = [root / name for name in ["data", "results", "protocols", "scripts", "environments"]]
    if any(not path.is_dir() for path in required):
        raise ValueError("PUBLIC_PACKAGE_ROOT_MARKERS_MISSING")
    if not (root / "protocols" / "verification_manifest.json").is_file():
        raise ValueError("PUBLIC_VERIFICATION_MANIFEST_MISSING")
    return root


def resolve_contract_source(raw_path, package_root):
    source = Path(raw_path)
    if source.is_absolute():
        raise ValueError("CONTRACT_SOURCE_MUST_BE_PUBLIC_RELATIVE:" + str(source))
    target = (package_root / source).resolve()
    try:
        target.relative_to(package_root)
    except ValueError as error:
        raise ValueError("CONTRACT_SOURCE_ESCAPES_PACKAGE:" + str(source)) from error
    if not target.is_file():
        raise ValueError("PACKAGED_CONTRACT_SOURCE_MISSING:" + str(target))
    return target


def repo_relative(path, package_root):
    """Return a portable repository-root-relative POSIX path."""
    return Path(path).resolve().relative_to(Path(package_root).resolve()).as_posix()


def guarded_r2(y_true, y_pred) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("SCORE_SHAPE_MISMATCH")
    n = len(y)
    status = None
    if n == 0:
        status = "UNDEFINED_EMPTY"
    elif n < 2:
        status = "UNDEFINED_TOO_FEW_ROWS"
    elif not np.isfinite(y).all() or not np.isfinite(p).all():
        status = "UNDEFINED_NONFINITE_INPUT"
    elif np.unique(y).size <= 1:
        status = "UNDEFINED_CONSTANT_TARGET"
    elif float(np.std(p, ddof=0)) <= 1e-9:
        status = "UNDEFINED_NEAR_CONSTANT_PREDICTION"
    if status:
        return {"n": n, "r2": None, "status": status}
    value = float(r2_score(y, p, force_finite=True))
    return {"n": n, "r2": value if math.isfinite(value) else None,
            "status": "DEFINED" if math.isfinite(value) else "UNDEFINED_NONFINITE_SCORE"}


def strongest_null(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(scores) != set(MODELS):
        raise ValueError("MODEL_PANEL_INCOMPLETE_OR_UNEXPECTED")
    eligible = [m for m in NULL_ORDER if scores[m]["status"] == "DEFINED"]
    best = max(eligible, key=lambda m: scores[m]["r2"]) if eligible else None
    pair_r2 = scores["PAIR_RF"]["r2"]
    best_r2 = scores[best]["r2"] if best else None
    delta = pair_r2 - best_r2 if pair_r2 is not None and best_r2 is not None else None
    return {"best_null": best, "best_null_r2": best_r2, "pair_r2": pair_r2,
            "delta_r2": delta, "status": "DEFINED" if delta is not None else "UNDEFINED_MARGIN",
            "eligible_nulls": eligible,
            "undefined_models": {m: scores[m]["status"] for m in MODELS if scores[m]["status"] != "DEFINED"}}


def equal_block_summary(blocks: list[dict[str, Any]], expected_roster: list[str]) -> dict[str, Any]:
    observed = [str(row["block_id"]) for row in blocks]
    if len(observed) != len(set(observed)) or set(observed) != set(expected_roster):
        raise ValueError("BLOCK_ROSTER_MISMATCH")
    ordered = {str(row["block_id"]): row for row in blocks}
    vals = [ordered[block]["delta_r2"] for block in expected_roster]
    finite = all(v is not None and math.isfinite(v) for v in vals)
    return {"block_denominator": len(expected_roster),
            "defined_blocks": sum(v is not None and math.isfinite(v) for v in vals),
            "equal_block_mean_delta_r2": float(np.mean(vals)) if finite else None,
            "positive_blocks": sum(v > 0 for v in vals) if finite else None,
            "status": "DEFINED" if finite else "UNDEFINED_DO_NOT_DROP_BLOCKS"}


def within_eight_ulp(actual: float, reference: float) -> bool:
    if not math.isfinite(actual) or not math.isfinite(reference):
        return False
    return abs(actual - reference) <= 8 * math.ulp(actual)


def align_appearance_panel(panel: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    keys = ["block_id", "row_index"]
    identity = keys + ["host_id", "guest_id"]
    if panel.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("DUPLICATE_APPEARANCE_KEY")
    lhs = panel.sort_values(keys).reset_index(drop=True)
    rhs = expected.sort_values(keys).reset_index(drop=True)
    if len(lhs) != len(rhs) or not lhs[identity].equals(rhs[identity]):
        raise ValueError("APPEARANCE_OR_IDENTITY_MISMATCH")
    return lhs


def normalized_ids(frame):
    result = frame.copy()
    result["block_id"] = result["block_id"].astype(str)
    for col in ["row_index", "host_id", "guest_id"]:
        result[col] = pd.to_numeric(result[col], errors="raise").astype("int64")
    return result


def bool_column(series):
    normalized = series.astype(str).str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("INVALID_MASK_BOOLEAN")
    return normalized.isin(["true", "1"]).to_numpy()


def score_panel(frame, branch, split, scenario, roster):
    """Keep full panel and all fixed blocks, even when the retained unit is empty."""
    details, margins = [], []
    units = [(block, frame.loc[frame.block_id.eq(block)]) for block in roster]
    if split != "double_cold":
        units.append(("POOLED_OOF", frame))
    for unit, part in units:
        scores = {}
        for model in MODELS:
            panel = part.loc[part.model.eq(model)]
            score = guarded_r2(panel.y_true.to_numpy(), panel.y_pred.to_numpy())
            scores[model] = score
            details.append({"branch": branch, "split": split, "scenario": scenario,
                            "block_id": unit, "model": model, **score})
        margin = strongest_null(scores)
        pair = part.loc[part.model.eq("PAIR_RF")]
        margins.append({"branch": branch, "split": split, "scenario": scenario,
                        "block_id": unit, "n_appearances": len(pair),
                        "n_unique_rows": int(pair.row_index.nunique()),
                        "uncertain_appearances": int(pair.uncertain.sum()) if "uncertain" in pair else None,
                        **margin})
    if split == "double_cold":
        primary = {"estimand": "equal_12_block_mean_margin",
                   **equal_block_summary(margins, roster)}
    else:
        pooled = margins[-1]
        primary = {"estimand": "pooled_oof_margin", "delta_r2": pooled["delta_r2"],
                   "pair_r2": pooled["pair_r2"], "best_null": pooled["best_null"],
                   "best_null_r2": pooled["best_null_r2"], "status": pooled["status"]}
    pair = frame.loc[frame.model.eq("PAIR_RF")]
    primary.update({"branch": branch, "split": split, "scenario": scenario,
                    "n_appearances": len(pair), "n_unique_rows": int(pair.row_index.nunique())})
    return details, margins, primary


def prepare_frozen_inputs():
    """Read only the clean public Stage B reference, contract and frozen masks."""
    b = B1_RESULTS / "stage_b"
    freeze_path = B1_PROTOCOL / "stage_b_reference.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["state"] != "STAGE_B_FROZEN":
        raise ValueError("STAGE_B_NOT_FROZEN")
    files = {
        "ANALYSIS_CONTRACT.json": B1_PROTOCOL / "analysis_contract.json",
        "APPEARANCE_CLASSIFICATION_MASKS.csv": b / "APPEARANCE_CLASSIFICATION_MASKS.csv",
    }
    for name, path in files.items():
        if sha(path) != freeze["files"][name]["sha256"]:
            raise ValueError("FROZEN_B_HASH_MISMATCH:" + name)
    contract = json.loads(files["ANALYSIS_CONTRACT.json"].read_text(encoding="utf-8"))
    if contract["primary_model"] != "PAIR_RF" or contract["null_models"] != NULL_ORDER:
        raise ValueError("CONTRACT_MODEL_ORDER_CHANGED")
    guards = contract["guards"]
    if not guards["applies_to_all_models_and_all_six_nulls"] or not guards["undefined_if_target_constant"]:
        raise ValueError("CONTRACT_GUARD_CHANGED")
    if guards["undefined_if_prediction_std_ddof0_lte"] != 1e-9:
        raise ValueError("CONTRACT_THRESHOLD_CHANGED")
    masks = pd.read_csv(files["APPEARANCE_CLASSIFICATION_MASKS.csv"], dtype=str, keep_default_na=False)
    masks = normalized_ids(masks)
    if masks.duplicated(["branch", "split", "block_id", "row_index", "layer"]).any():
        raise ValueError("DUPLICATE_FROZEN_MASK")
    for _, scope in masks.groupby(["branch", "split"]):
        if set(scope.layer) != set(LAYERS) or not bool_column(scope.all_mask).all():
            raise ValueError("FROZEN_MASK_LAYER_OR_ALL_MASK_MISMATCH")
    return freeze_path, freeze, contract, masks


def check_input_reachability(package_root):
    """Check public frozen bindings and relative package paths without reading outcomes."""
    root = validate_package_root(package_root)
    b = B1_RESULTS / "stage_b"
    freeze_path = B1_PROTOCOL / "stage_b_reference.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["state"] != "STAGE_B_FROZEN":
        raise ValueError("STAGE_B_NOT_FROZEN")
    files = {
        "ANALYSIS_CONTRACT.json": B1_PROTOCOL / "analysis_contract.json",
        "APPEARANCE_CLASSIFICATION_MASKS.csv": b / "APPEARANCE_CLASSIFICATION_MASKS.csv",
    }
    frozen = {}
    for name, path in files.items():
        actual = sha(path)
        expected = freeze["files"][name]["sha256"]
        if actual != expected:
            raise ValueError("FROZEN_B_HASH_MISMATCH:" + name)
        frozen[name] = {"sha256": actual, "bytes": path.stat().st_size}
    contract = json.loads(files["ANALYSIS_CONTRACT.json"].read_text(encoding="utf-8"))
    prediction_sources = []
    for item in contract["alignment"]["prediction_sources"]:
        path = resolve_contract_source(item["path"], root)
        prediction_sources.append({
            "branch": item["branch"], "split": item["split"],
            "relative_path": item["path"], "bytes": path.stat().st_size,
        })
    references = [
        "results/canonical_cpu/run_a/metrics_cpu.tsv",
        "data/released_predictions/block_level_deltaR2.csv",
        "data/reference/suprabench_baselines_modern_results.json",
    ]
    for relative in references:
        if not (root / relative).is_file():
            raise ValueError("PACKAGED_BASELINE_REFERENCE_MISSING:" + relative)
    return {
        "status": "INPUTS_REACHABLE", "package_root": ".",
        "frozen_stage_b": frozen, "prediction_sources": prediction_sources,
        "baseline_references": references,
    }


def load_fixed_panel(source, expected, cache, receipts, package_root):
    path = resolve_contract_source(source["path"], package_root)
    fields = source["fields"]
    if str(path) not in cache:
        opened = utc_now()
        frame = pd.read_csv(path, usecols=list(fields.values()), keep_default_na=False,
                            dtype={fields["block_id"]: str}, float_precision="round_trip")
        frame = frame.rename(columns={value: key for key, value in fields.items()})
        frame = normalized_ids(frame)
        frame = frame.loc[frame.model.isin(MODELS)].copy()
        if source["branch"] == "canonical":
            # Historical metrics use features.y = original y.astype(float32).
            # predictions.y_true instead serializes the pre-cast original y.
            # Restore that evidenced target representation, not the predictions.
            frame["y_true"] = frame.y_true.to_numpy(dtype=np.float32).astype(np.float64)
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            raise ValueError("NONFINITE_SAVED_OUTCOMES")
        cache[str(path)] = frame
        receipt_path = repo_relative(path, package_root)
        receipts[receipt_path] = {"sha256": sha(path), "opened_at_utc": opened,
                               "contract_path": source["path"], "resolved_path": receipt_path,
                               "usecols": list(fields.values()), "included_models": MODELS,
                               "target_representation": "producer_float32_values_promoted_to_float64"
                               if source["branch"] == "canonical" else "saved_csv_float64_round_trip",
                               "prediction_representation": "saved_csv_float64_round_trip"}
    frame = cache[str(path)]
    frame = frame.loc[frame.split.eq(source["split"])].copy()
    if set(frame.model) != set(MODELS):
        raise ValueError("SAVED_MODEL_PANEL_MISMATCH")
    aligned = []
    for model in MODELS:
        part = align_appearance_panel(frame.loc[frame.model.eq(model)], expected)
        if source["branch"] == "canonical":
            ordered = expected.sort_values(["block_id", "row_index"]).reset_index(drop=True)
            for col in ["canonical_host_sha256", "canonical_guest_sha256"]:
                if not part[col].equals(ordered[col]):
                    raise ValueError("CANONICAL_HASH_IDENTITY_MISMATCH")
        aligned.append(part)
    result = pd.concat(aligned, ignore_index=True)
    if not result.groupby(["block_id", "row_index"]).y_true.nunique().eq(1).all():
        raise ValueError("Y_TRUE_DIFFERS_BETWEEN_MODELS")
    return result


def self_test() -> list[dict[str, Any]]:
    actual = []

    def record(name, value, expected):
        actual.append({"name": name, "actual": value, "expected": expected,
                       "pass": value == expected})
        if value != expected:
            raise AssertionError(actual[-1])

    record("perfect", guarded_r2([0, 1, 2], [0, 1, 2])["r2"], 1.0)
    record("all_model_prediction_guard", guarded_r2([0, 1, 2], [1, 1, 1])["status"],
           "UNDEFINED_NEAR_CONSTANT_PREDICTION")
    record("target_guard", guarded_r2([1, 1, 1], [0, 1, 2])["status"], "UNDEFINED_CONSTANT_TARGET")
    record("empty", guarded_r2([], [])["status"], "UNDEFINED_EMPTY")
    record("small_n", guarded_r2([1], [2])["status"], "UNDEFINED_TOO_FEW_ROWS")
    record("nonfinite", guarded_r2([0, 1], [0, np.nan])["status"], "UNDEFINED_NONFINITE_INPUT")
    s = {m: {"r2": 0.1, "status": "DEFINED"} for m in MODELS}
    s["PAIR_RF"]["r2"] = 0.5
    s["host_only_RF"]["r2"] = 0.4
    record("initial_winner", strongest_null(s)["best_null"], "host_only_RF")
    s["host_only_RF"]["r2"] = 0.05
    record("mask_can_change_winner", strongest_null(s)["best_null"], "additive")
    b = [{"block_id": "0", "delta_r2": 0.4}, {"block_id": "1", "delta_r2": None}]
    summary = equal_block_summary(b, ["0", "1"])
    record("undefined_block_not_dropped", [summary["block_denominator"], summary["defined_blocks"],
           summary["equal_block_mean_delta_r2"]], [2, 1, None])
    within = 1.0 + 8 * math.ulp(1.0)
    outside = 1.0 + 9 * math.ulp(1.0)
    record("eight_ulp_boundary", within_eight_ulp(1.0, within), True)
    record("nine_ulp_rejected", within_eight_ulp(1.0, outside), False)
    expected = pd.DataFrame({"block_id": ["0", "1"], "row_index": [7, 7],
                             "host_id": [1, 1], "guest_id": [2, 2]})
    record("same_row_distinct_appearances", len(align_appearance_panel(expected, expected)), 2)
    for name, bad in [("missing_join_rejected", expected.iloc[:1]),
                      ("duplicate_join_rejected", pd.concat([expected, expected.iloc[:1]]))]:
        try:
            align_appearance_panel(bad, expected)
        except ValueError:
            rejected = True
        else:
            rejected = False
        record(name, rejected, True)
    # Between-fold target shifts make pooled R2 differ from the average of fold R2.
    y = np.array([0., 1., 10., 11.])
    p = np.array([0., 2., 10., 12.])
    pooled = guarded_r2(y, p)["r2"]
    block_average = (guarded_r2(y[:2], p[:2])["r2"] + guarded_r2(y[2:], p[2:])["r2"]) / 2
    record("pooled_not_fold_average", pooled != block_average, True)
    return actual


def save_table(path, rows):
    serial = []
    for row in rows:
        serial.append({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                       if isinstance(value, (dict, list)) else value for key, value in row.items()})
    pd.DataFrame(serial).to_csv(path, sep="\t", index=False, na_rep="UNDEFINED", lineterminator="\n")


def baseline_check(branch, split, details, margins, reference_cache, receipts, package_root):
    def reference_table(path, sep, columns):
        if str(path) not in reference_cache:
            opened = utc_now()
            reference_cache[str(path)] = pd.read_csv(path, sep=sep, usecols=columns,
                                                     dtype=str, keep_default_na=False)
            receipts[repo_relative(path, package_root)] = {"sha256": sha(path), "opened_at_utc": opened,
                                   "purpose": "frozen_baseline_reference", "usecols": columns}
        return reference_cache[str(path)]

    def compare(unit, model, actual, reference, kind="exact_float_8ulp"):
        both_undefined = actual is None and reference is None
        passed = both_undefined or (actual is not None and reference is not None
                                    and within_eight_ulp(actual, reference))
        return {"branch": branch, "split": split, "unit": unit, "model": model,
                "check": kind, "actual": actual, "reference": reference,
                "absolute_difference": abs(actual-reference) if actual is not None and reference is not None else None,
                "pass": passed}

    checks = []
    if branch == "canonical":
        path = package_root / "results/canonical_cpu/run_a/metrics_cpu.tsv"
        ref = reference_table(path, "\t", ["scope", "split", "block_id", "model", "n", "r2", "r2_status"])
        required = {"scope", "split", "block_id", "model", "r2", "r2_status"}
        if not required.issubset(ref.columns):
            raise ValueError("CANONICAL_METRIC_SCHEMA_REQUIRES_EXPLICIT_MAPPING:" + ",".join(ref.columns))
        for row in details:
            unit, model = row["block_id"], row["model"]
            scope = "model_split_aggregate" if unit == "POOLED_OOF" else "model_block"
            selected = ref.loc[ref.scope.eq(scope) & ref.split.eq(split) & ref.model.eq(model)]
            if unit != "POOLED_OOF":
                selected = selected.loc[selected.block_id.eq(unit)]
            if len(selected) != 1:
                raise ValueError(f"BASELINE_REFERENCE_CARDINALITY:{split}:{unit}:{model}:{len(selected)}")
            archived = selected.iloc[0]
            if archived.r2_status not in ["DEFINED", "UNKNOWN"]:
                raise ValueError("UNRECOGNIZED_REFERENCE_STATUS:" + archived.r2_status)
            if int(archived.n) != row["n"]:
                raise ValueError("BASELINE_REFERENCE_N_MISMATCH")
            reference = float(archived.r2) if archived.r2_status == "DEFINED" else None
            if reference is None and archived.r2 != "":
                raise ValueError("UNDEFINED_REFERENCE_HAS_NUMERIC_VALUE")
            checks.append(compare(unit, model, row["r2"], reference))
    elif split == "double_cold":
        path = package_root / "data/released_predictions/block_level_deltaR2.csv"
        ref = reference_table(path, ",", ["block", "delta_PAIR_RF", "best_null_R2"])
        if ref.block.duplicated().any():
            raise ValueError("DUPLICATE_RELEASED_BASELINE_BLOCK")
        if set(ref.block) != {row["block_id"] for row in margins}:
            raise ValueError("RELEASED_BASELINE_REFERENCE_ROSTER_MISMATCH")
        by_block = ref.set_index("block")
        for row in margins:
            unit = row["block_id"]
            archived = by_block.loc[unit]
            checks.append(compare(unit, "best_null_R2", row["best_null_r2"], float(archived.best_null_R2)))
            checks.append(compare(unit, "delta_PAIR_RF", row["delta_r2"], float(archived.delta_PAIR_RF)))
    else:
        # Original producer eval_group stores only round(R2(yy, p), 3), lines 299-314.
        # These rounded references are not an exact-float 8-ULP acceptance claim.
        path = package_root / "data/reference/suprabench_baselines_modern_results.json"
        if str(path) not in reference_cache:
            opened = utc_now()
            with path.open("r", encoding="utf-8-sig") as handle:
                saved = json.load(handle)
            selected = {sp: {m: saved[sp][m]["R2"] for m in MODELS}
                        for sp in ["host_cold", "guest_cold"]}
            del saved
            reference_cache[str(path)] = selected
            receipts[repo_relative(path, package_root)] = {
                "sha256": sha(path), "opened_at_utc": opened,
                "selected_key_paths": [f"{sp}.{m}.R2" for sp in selected for m in MODELS],
                "reference_precision": "producer round(value, 3); exact reference unavailable",
            }
        for row in details:
            if row["block_id"] != "POOLED_OOF":
                continue
            raw = reference_cache[str(path)][split][row["model"]]
            reference = float(raw) if raw is not None and math.isfinite(float(raw)) else None
            actual = row["r2"]
            rounded = round(actual, 3) if actual is not None else None
            checks.append({"branch": branch, "split": split, "unit": "POOLED_OOF",
                           "model": row["model"], "check": "producer_rounded_3dp_only",
                           "actual": actual, "rounded_actual": rounded, "reference": reference,
                           "pass": rounded == reference, "exact_float_reference_available": False})
    return checks


def execute_after_freeze(branch_only=None, output_dir=None, package_root=None):
    root = validate_package_root(package_root)
    if not output_dir:
        raise ValueError("FRESH_OUTPUT_DIRECTORY_REQUIRED")
    out = Path(output_dir).resolve()
    runs = (root / "runs").resolve()
    if out == runs or not out.is_relative_to(runs):
        raise ValueError("OUTPUT_MUST_BE_FRESH_RUNS_SUBDIRECTORY:" + str(out))
    if out.exists():
        raise ValueError("REFUSE_TO_WRITE_EXISTING_OUTPUT_DIRECTORY")
    script_hash = sha(SCRIPT)
    print("SCRIPT_SHA256=" + script_hash, flush=True)
    tests = self_test()
    print("SELF_TEST_ACTUAL=" + json.dumps(tests, ensure_ascii=False, allow_nan=False), flush=True)
    freeze_path, freeze, contract, masks = prepare_frozen_inputs()
    started = utc_now()
    out.mkdir()
    write_json(out / "PRE_OUTCOME_RECORD.json", {
        "prepared_at_utc": started, "script_sha256": script_hash,
        "package_root": ".",
        "stage_b_freeze_path": repo_relative(freeze_path, root), "stage_b_freeze_sha256": sha(freeze_path),
        "stage_b_frozen_at_utc": freeze["frozen_at_utc"],
        "self_test_actual": tests, "training_run": False,
        "new_model_or_seed": False, "all_model_guard": True,
    })
    all_details, all_margins, summaries, gates, failures = [], [], [], [], []
    cache, reference_cache, receipts = {}, {}, {}
    for source in contract["alignment"]["prediction_sources"]:
        branch, split = source["branch"], source["split"]
        if branch_only is not None and branch != branch_only:
            continue
        try:
            subset = masks.loc[masks.branch.eq(branch) & masks.split.eq(split)]
            layers = list(dict.fromkeys(subset.layer))
            if not layers:
                raise ValueError("NO_FROZEN_MASK_LAYER")
            expected = subset.loc[subset.layer.eq(layers[0])].copy()
            for layer in layers:
                other = subset.loc[subset.layer.eq(layer)]
                align_appearance_panel(other, expected)
            roster = [str(row["block_id"]) for row in contract["block_rosters"][branch][split]]
            if set(expected.block_id) != set(roster):
                raise ValueError("MASK_BLOCK_ROSTER_MISMATCH")
            if split == "double_cold" and len(roster) != 12:
                raise ValueError("DOUBLE_BLOCK_DENOMINATOR_NOT_12")
            frame = load_fixed_panel(source, expected, cache, receipts, root)
            details, margins, primary = score_panel(frame, branch, split, "all", roster)
            checks = baseline_check(branch, split, details, margins, reference_cache, receipts, root)
            gates.extend(checks)
            all_details.extend(details)
            all_margins.extend(margins)
            summaries.append(primary)
            if any(not row["pass"] for row in checks):
                failures.append({"branch": branch, "split": split, "status": "BASELINE_GATE_BLOCKED"})
                continue
            ordered = frame.sort_values(["model", "block_id", "row_index"]).reset_index(drop=True)
            for layer in layers:
                layer_mask = subset.loc[subset.layer.eq(layer)].sort_values(["block_id", "row_index"])
                layer_mask = layer_mask.reset_index(drop=True)
                keep_by_key = dict(zip(zip(layer_mask.block_id, layer_mask.row_index),
                                       bool_column(layer_mask.protected_axis_retained)))
                uncertain_by_key = dict(zip(zip(layer_mask.block_id, layer_mask.row_index),
                                            bool_column(layer_mask.uncertain)))
                selection = []
                uncertainty = []
                for key in zip(ordered.block_id, ordered.row_index):
                    selection.append(keep_by_key[key])
                    uncertainty.append(uncertain_by_key[key])
                annotated = ordered.copy()
                annotated["uncertain"] = uncertainty
                retained = annotated.loc[np.asarray(selection, dtype=bool)]
                if len(retained) != int(bool_column(layer_mask.protected_axis_retained).sum()) * len(MODELS):
                    raise ValueError("MASK_MODEL_CARDINALITY_CHANGED")
                detail, margin, summary = score_panel(retained, branch, split, layer, roster)
                original_estimate = primary.get("delta_r2", primary.get("equal_block_mean_delta_r2"))
                retained_estimate = summary.get("delta_r2", summary.get("equal_block_mean_delta_r2"))
                summary.update({"original_appearances": primary["n_appearances"],
                                "removed_appearances": primary["n_appearances"] - summary["n_appearances"],
                                "baseline_estimate": original_estimate,
                                "retained_minus_baseline": retained_estimate - original_estimate
                                if retained_estimate is not None and original_estimate is not None else None})
                all_details.extend(detail)
                all_margins.extend(margin)
                summaries.append(summary)
        except (ValueError, KeyError, AssertionError, RuntimeError) as error:
            failures.append({"branch": branch, "split": split, "status": "BRANCH_BLOCKED",
                             "error": type(error).__name__ + ": " + str(error)})
    save_table(out / "BASELINE_CHECKS.tsv", gates)
    save_table(out / "MODEL_NULL_SCORES.tsv", all_details)
    save_table(out / "BLOCK_MARGINS.tsv", all_margins)
    save_table(out / "PRIMARY_COMPARISONS.tsv", summaries)
    write_json(out / "SUMMARY.json", {"primary_comparisons": summaries, "blocked_branches": failures,
                                     "status": "COMPLETED" if not failures else "PARTIAL_BRANCH_BLOCKERS",
                                     "interpretation": "Fixed original training and predictions; test composition only."})
    output_names = ["BASELINE_CHECKS.tsv", "MODEL_NULL_SCORES.tsv", "BLOCK_MARGINS.tsv",
                    "PRIMARY_COMPARISONS.tsv", "SUMMARY.json", "PRE_OUTCOME_RECORD.json"]
    write_json(out / "RUN_RECEIPT.json", {
        "started_at_utc": started, "completed_at_utc": utc_now(),
        "package_root": ".",
        "script_sha256": script_hash, "input_receipts": receipts,
        "stage_b_freeze_sha256": sha(freeze_path), "self_test_actual": tests,
        "outputs": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                    for name in output_names},
        "status": "COMPLETED" if not failures else "PARTIAL_BRANCH_BLOCKERS",
        "blocked_branches": failures, "fit_or_training": False,
        "changed_frozen_rules_or_masks": False,
        "branch_scope": branch_only or "both",
    })
    print(json.dumps({"status": "COMPLETED" if not failures else "PARTIAL_BRANCH_BLOCKERS",
                      "primary_rows": len(summaries), "baseline_checks": len(gates),
                      "blocked_branches": failures}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test-only", action="store_true")
    mode.add_argument("--check-inputs-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--branch", choices=["canonical"])
    parser.add_argument("--package-root", help="Public repository root containing data/, results/ and protocols/")
    parser.add_argument("--output-dir", help="Fresh output directory below the public repository runs/ directory")
    args = parser.parse_args()
    if args.self_test_only:
        print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False, indent=2))
    elif args.check_inputs_only:
        if not args.package_root:
            parser.error("--check-inputs-only requires --package-root")
        print(json.dumps(check_input_reachability(args.package_root), ensure_ascii=False,
                         allow_nan=False, indent=2))
    else:
        if not args.package_root or not args.output_dir:
            parser.error("--execute requires --package-root and --output-dir")
        execute_after_freeze(args.branch, args.output_dir, args.package_root)
