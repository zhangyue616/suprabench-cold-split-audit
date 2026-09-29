"""Fixed-prediction, exact-appearance sensitivity; no fitting or inference tests.

2026-09-23 author-authorised scope. Run with the recorded canonical interpreter.
This script writes only beside itself. Existing scientific inputs stay read-only.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
OUT = Path(os.environ.get("SUPRABENCH_RECURRENCE_OUTPUT", ROOT / "runs/strict_recurrence_sensitivity"))
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = {
    "audit": ROOT / "data/identity/canonical_identity_recurrence_2026-08-01.tsv",
    "folds": ROOT / "data/folds/folds_double_cold_test.csv",
    "predictions": ROOT / "data/released_predictions/predictions_double_cold.csv",
    "released_delta": ROOT / "data/released_predictions/block_level_deltaR2.csv",
    "scoring_reference": ROOT / "protocols/p1_d6sc_overlap/stage_b_reaggregate.py",
}
EXPECTED_HASHES = {
    "folds": "14addf87e70baf45e6325126b04266f5162cfe91439da28a7b79022d99e20de5",
    "predictions": "4e1a0168e6ebd7f25764beab9463c650a989bb3b8eb311675fdaae1c8009d721",
    "released_delta": "7f41a1aba4f46fb757a18191a5643b519820053bc9e9e56d847ea728d3718dd4",
}
NULLS = ["additive", "host_only_RF", "guest_only_RF", "cond_only_RF", "source_only_RF", "sim_knn_k5"]
MODELS = ["PAIR_RF"] + NULLS
KEY = ["block", "row_index"]
IDENTITY = KEY + ["host_id", "guest_id"]
EXPECTED_MASK_COUNTS = {0: 7, 2: 4, 3: 1, 4: 2, 7: 1, 8: 9, 9: 2, 10: 9, 11: 2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def score(part, model):
    # Exact near-constant rule and sklearn default from the existing Stage B.
    if part.empty:
        return float("nan"), "UNDEFINED_EMPTY"
    if model == "source_only_RF" and float(part["y_pred"].std(ddof=0)) <= 1e-9:
        return float("nan"), "UNDEFINED_SOURCE_NEAR_CONSTANT"
    if len(part) < 2:
        return float("nan"), "UNDEFINED_TOO_FEW_ROWS"
    value = float(r2_score(part["y_true"].to_numpy(), part["y_pred"].to_numpy()))
    return value, "FINITE" if np.isfinite(value) else "UNDEFINED_NONFINITE"


def selected_mask(frame, frozen_keys):
    return np.fromiter(((int(b), int(r)) in frozen_keys for b, r in frame[KEY].itertuples(index=False, name=None)), dtype=bool, count=len(frame))


def self_test():
    sample = pd.DataFrame({"block": [0, 1, 1], "row_index": [7, 7, 8]})
    actual_mask = selected_mask(sample, {(0, 7)}).tolist()
    assert actual_mask == [True, False, False], actual_mask
    sample_score = pd.DataFrame({"y_true": [0.0, 1.0, 2.0], "y_pred": [0.0, 1.0, 2.0]})
    perfect = score(sample_score, "PAIR_RF")
    constant = sample_score.assign(y_pred=1.0)
    guarded = score(constant, "source_only_RF")
    ordinary = score(constant, "additive")
    assert perfect == (1.0, "FINITE"), perfect
    assert math.isnan(guarded[0]) and guarded[1] == "UNDEFINED_SOURCE_NEAR_CONSTANT", guarded
    assert ordinary == (0.0, "FINITE"), ordinary
    actual = {"appearance_mask": actual_mask, "perfect_r2": perfect[0], "source_constant_status": guarded[1], "other_constant_r2": ordinary[0]}
    print("SELF_TEST_ACTUAL=" + json.dumps(actual), flush=True)
    return actual


def identities(frame):
    return frame[IDENTITY].sort_values(KEY).reset_index(drop=True).astype("int64")


def calculate(frame, phase, original_counts):
    detail, blocks = [], []
    for block in range(12):
        full = frame.loc[frame["block"].eq(block)]
        scores, states = {}, {}
        for model in MODELS:
            part = full.loc[full["model"].eq(model)]
            value, state = score(part, model)
            scores[model], states[model] = value, state
            detail.append({"phase": phase, "block": block, "model": model, "n_appearances": len(part), "r2": value, "status": state})
        finite_nulls = [m for m in NULLS if np.isfinite(scores[m])]
        # Python max preserves the frozen null order for exact ties.
        best = max(finite_nulls, key=lambda m: scores[m]) if finite_nulls else "UNDEFINED"
        best_r2 = scores[best] if finite_nulls else float("nan")
        delta = scores["PAIR_RF"] - best_r2
        n = int(full["model"].eq("PAIR_RF").sum())
        blocks.append({"phase": phase, "block": block, "n_original": int(original_counts[block]), "n_appearances": n,
                       "n_removed": int(original_counts[block]) - n, "pair_r2": scores["PAIR_RF"], "best_null": best,
                       "best_null_r2": best_r2, "delta": delta, "status": "FINITE" if np.isfinite(delta) else "UNDEFINED",
                       "undefined_models": "|".join(m + ":" + states[m] for m in MODELS if states[m] != "FINITE")})
    return pd.DataFrame(detail), pd.DataFrame(blocks)


def aggregate(blocks):
    finite = bool(np.isfinite(blocks["delta"]).all())
    return {"block_denominator": 12, "finite_blocks": int(np.isfinite(blocks["delta"]).sum()),
            "equal_block_mean_delta": float(blocks["delta"].mean()) if finite else None,
            "positive_blocks": int(blocks["delta"].gt(0).sum()) if finite else None,
            "status": "FINITE" if finite else "UNDEFINED_DO_NOT_DROP_BLOCKS"}


def main():
    script_sha = sha(SCRIPT)
    print("SCRIPT_SHA256=" + script_sha, flush=True)
    selftest = self_test()  # Must pass before the first real input read.

    audit = pd.read_csv(INPUTS["audit"], sep="\t", usecols=["record_type", "fold", "row_index", "host_id", "guest_id"], keep_default_na=False)
    mask = audit.loc[audit["record_type"].eq("double_affected_appearance"), ["fold", "row_index", "host_id", "guest_id"]].rename(columns={"fold": "block"}).astype("int64")
    assert len(mask) == 37 and not mask.duplicated(KEY).any()
    mask = identities(mask)
    assert mask["row_index"].nunique() == 31
    assert mask.groupby("block").size().to_dict() == EXPECTED_MASK_COUNTS
    frozen_keys = set(mask[KEY].itertuples(index=False, name=None))

    fold = pd.read_csv(INPUTS["folds"], usecols=["seed", "fold_index", "row_index", "host_id", "guest_id"])
    assert fold["seed"].eq(fold["fold_index"]).all()
    fold = identities(fold.rename(columns={"seed": "block"}))
    assert len(fold) == 1984 and not fold.duplicated(KEY).any()
    pred_identity = pd.read_csv(INPUTS["predictions"], usecols=["fold_or_block", "row_index", "host_id", "guest_id", "model"]).rename(columns={"fold_or_block": "block"})
    assert set(MODELS) <= set(pred_identity["model"])
    for model in MODELS:
        panel = identities(pred_identity.loc[pred_identity["model"].eq(model)])
        pd.testing.assert_frame_equal(panel, fold)
        pd.testing.assert_frame_equal(identities(panel.loc[selected_mask(panel, frozen_keys)]), mask)
    baseline_identity = identities(pred_identity.loc[pred_identity["model"].eq("PAIR_RF")])
    retained_identity = baseline_identity.loc[~selected_mask(baseline_identity, frozen_keys)]
    assert baseline_identity["row_index"].nunique() == 1382
    assert len(retained_identity) == 1947 and retained_identity["row_index"].nunique() == 1356
    # Freeze the exact appearance mask before loading y_true or y_pred values.
    mask.to_csv(OUT / "mask_37_appearances.tsv", sep="\t", index=False, lineterminator="\n")
    mask_sha = sha(OUT / "mask_37_appearances.tsv")
    print("FROZEN_MASK_SHA256=" + mask_sha, flush=True)

    recorded_inputs = []
    for name, path in INPUTS.items():
        digest = sha(path)
        if name in EXPECTED_HASHES:
            assert digest == EXPECTED_HASHES[name], (name, "INPUT_SHA_MISMATCH")
        recorded_inputs.append({"role": name, "path": path.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": path.stat().st_size})
    write_json("INPUT_RECORD.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": script_sha,
        "mask_sha256": mask_sha, "mask_frozen_before_outcome_read": True, "self_test_actual": selftest,
        "inputs": recorded_inputs, "models_in_order": MODELS, "null_tie_order": NULLS,
        "comparison_tolerance": {"relative": 1e-12, "absolute": 0.0},
        "interpreter": sys.executable, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "identity_counts": {"baseline_appearances": 1984, "baseline_unique_rows": 1382, "excluded_appearances": 37, "excluded_unique_rows": 31, "affected_blocks": 9, "retained_appearances": 1947, "retained_unique_rows": 1356}})

    pred = pd.read_csv(INPUTS["predictions"], usecols=["fold_or_block", "row_index", "host_id", "guest_id", "model", "y_true", "y_pred"], float_precision="round_trip").rename(columns={"fold_or_block": "block"})
    pred = pred.loc[pred["model"].isin(MODELS)].copy()
    assert np.isfinite(pred[["y_true", "y_pred"]].to_numpy()).all()
    assert pred.groupby(KEY)["y_true"].nunique().eq(1).all()
    counts = pred.loc[pred["model"].eq("PAIR_RF")].groupby("block").size().to_dict()
    assert set(counts) == set(range(12))
    before_detail, before = calculate(pred, "baseline", counts)
    released = pd.read_csv(INPUTS["released_delta"], usecols=["block", "delta_PAIR_RF", "best_null_R2"], float_precision="round_trip").set_index("block")
    assert set(released.index) == set(range(12)) and released.index.is_unique
    checks = []
    for row in before.to_dict("records"):
        block = row["block"]
        archived_delta = float(released.loc[block, "delta_PAIR_RF"])
        archived_best = float(released.loc[block, "best_null_R2"])
        checks.append({"block": block, "computed_delta": row["delta"], "released_delta": archived_delta,
                       "delta_abs_difference": abs(row["delta"] - archived_delta),
                       "delta_match": math.isclose(row["delta"], archived_delta, rel_tol=1e-12, abs_tol=0.0),
                       "computed_best_null_r2": row["best_null_r2"], "released_best_null_r2": archived_best,
                       "best_null_match": math.isclose(row["best_null_r2"], archived_best, rel_tol=1e-12, abs_tol=0.0),
                       "pair_r2_match_released_sum": math.isclose(row["pair_r2"], archived_delta + archived_best, rel_tol=1e-12, abs_tol=0.0)})
    baseline_ok = all(r["delta_match"] and r["best_null_match"] and r["pair_r2_match_released_sum"] for r in checks)
    before_summary = aggregate(before)
    baseline_ok = baseline_ok and math.isclose(before_summary["equal_block_mean_delta"], 0.095039911151674, rel_tol=1e-12, abs_tol=0.0) and before_summary["positive_blocks"] == 11
    write_json("BASELINE_CHECK.json", {"status": "PASS" if baseline_ok else "BASELINE_MISMATCH_STOP", "relative_tolerance": 1e-12, "absolute_tolerance": 0.0, "checks": checks, "aggregate": before_summary})
    assert baseline_ok, "BASELINE_MISMATCH_STOP: no sensitivity calculation performed"
    print("BASELINE_CHECK=PASS actual recomputation, all 12 blocks", flush=True)

    retained = pred.loc[~selected_mask(pred, frozen_keys)]
    after_detail, after = calculate(retained, "excluded_37_appearances", counts)
    assert after["n_removed"].sum() == 37 and after["n_appearances"].sum() == 1947
    combined_detail = pd.concat([before_detail, after_detail], ignore_index=True)
    combined_blocks = pd.concat([before, after], ignore_index=True)
    combined_detail.to_csv(OUT / "model_r2_by_block.tsv", sep="\t", index=False, na_rep="UNDEFINED", lineterminator="\n")
    combined_blocks.to_csv(OUT / "block_comparison.tsv", sep="\t", index=False, na_rep="UNDEFINED", lineterminator="\n")
    after_summary = aggregate(after)
    undefined = [{"phase": r["phase"], "block": r["block"], "model": r["model"], "status": r["status"]} for r in combined_detail.to_dict("records") if r["status"] != "FINITE"]
    old_undefined = {(r["block"], r["model"], r["status"]) for r in undefined if r["phase"] == "baseline"}
    new_undefined = [r for r in undefined if r["phase"] != "baseline" and (r["block"], r["model"], r["status"]) not in old_undefined]
    changed = [{"block": int(a.block), "before": a.best_null, "after": b.best_null} for a, b in zip(before.itertuples(), after.itertuples()) if a.best_null != b.best_null]
    status = "COMPUTED" if not new_undefined and after_summary["status"] == "FINITE" else "UNDEFINED_INTERPRETATION_HOLD"
    mean_change = after_summary["equal_block_mean_delta"] - before_summary["equal_block_mean_delta"] if after_summary["status"] == "FINITE" else None
    write_json("SUMMARY.json", {"status": status, "baseline_gate": "PASS", "baseline": before_summary, "excluded_37_appearances": after_summary,
        "equal_block_mean_delta_change": mean_change, "changed_comparators": changed,
        "undefined_records": undefined, "new_undefined_records": new_undefined,
        "mask_sha256": mask_sha, "script_sha256": script_sha,
        "limits": ["Saved predictions only; no training exposure removed or models refitted.", "Exact (block,row) exclusion, not global exclusion of the 31 distinct rows.", "Twelve fixed blocks equally weighted; no dropped blocks or denominator change.", "No confidence intervals, significance tests, or new generalization claim.", "Not incorporated into accepted R2 manuscript or SI."]})
    # One output readback; validates saved shape and mask identity, not a second run.
    saved_mask = pd.read_csv(OUT / "mask_37_appearances.tsv", sep="\t")
    pd.testing.assert_frame_equal(saved_mask, mask)
    assert len(pd.read_csv(OUT / "model_r2_by_block.tsv", sep="\t")) == 168
    assert len(pd.read_csv(OUT / "block_comparison.tsv", sep="\t")) == 24
    print(json.dumps({"status": status, "baseline": before_summary, "excluded_37": after_summary, "mean_change": mean_change, "changed_comparators": changed, "new_undefined_records": new_undefined}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
