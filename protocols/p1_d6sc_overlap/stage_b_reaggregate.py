"""Stage B: frozen-label diff and outcome-side reaggregation.

This script never runs the Stage-A predicate. It reads the frozen label TSVs,
checks their hashes and level sets, reproduces the released baseline, maps
row_index through the zero-based clean-condition CSV, and then reports all
three predeclared exclusion levels without selecting among them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import r2_score


PROTOCOL_DIR = Path(__file__).resolve().parent
ROOT = PROTOCOL_DIR.parents[1]
OUTDIR = Path(os.environ.get("SUPRABENCH_P1_OUTPUT", ROOT / "runs/p1_d6sc_overlap/stage_b"))
OUTDIR.mkdir(parents=True, exist_ok=True)

HOST_LABELS = PROTOCOL_DIR / "host_labels_v2_1.tsv"
GUEST_LABELS = PROTOCOL_DIR / "guest_labels_v2_1.tsv"
DC_PRED = ROOT / "data/released_predictions/predictions_double_cold.csv"
FC_PRED = ROOT / "data/released_predictions/predictions_family_cold.csv"
BLOCK_DELTA = ROOT / "data/released_predictions/block_level_deltaR2.csv"
DC_FOLDS = ROOT / "data/folds/folds_double_cold_test.csv"
FC_FOLDS = ROOT / "data/folds/folds_family_cold.csv"
CLEAN = ROOT / "data/clean/suprabench_bap_clean_cond.csv"
HOST_MAP = ROOT / "data/folds/host_id_map.csv"
GUEST_MAP = ROOT / "data/folds/guest_id_map.csv"

RESULTS_PATH = OUTDIR / "results_stage_b.md"
DC_MANIFEST_PATH = OUTDIR / "manifest_double_cold.tsv"
FC_MANIFEST_PATH = OUTDIR / "manifest_family_cold.tsv"
LEGACY_DIFF_PATH = OUTDIR / "legacy_diff.tsv"

EXPECTED_SHA = {
    HOST_LABELS: "81f1dc80209e030c47d9e5771cc381380b3661281f0a583acadfa3d37461fff1",
    GUEST_LABELS: "f192875fb110622d632609289a73adf006a30b0516a35f4abf154f51f7ca2b93",
    DC_PRED: "4e1a0168e6ebd7f25764beab9463c650a989bb3b8eb311675fdaae1c8009d721",
    FC_PRED: "f982b2cbb9aacf3bffc82ed844d2e1e34f36a5549e8bc834f5de94bb86b66b63",
    BLOCK_DELTA: "7f41a1aba4f46fb757a18191a5643b519820053bc9e9e56d847ea728d3718dd4",
    DC_FOLDS: "14addf87e70baf45e6325126b04266f5162cfe91439da28a7b79022d99e20de5",
    FC_FOLDS: "12c99d8df84e238c603d22b68536ba88e30a00f1f05d0dda4f2de24c51040e02",
    CLEAN: "56af035c649ca5772675de608772918482ca7f74f12a91efcc50e44fc80ee900",
    HOST_MAP: "2286010652fe0ad54d2c8ee152a76c57a3f755c8535e2cc0888929fdc34df634",
    GUEST_MAP: "3448a3366fe7cc7b0bc2246d0a703c76ba7ae0241739473cfd096a749b513002",
}

STAGE_A_FILES = [
    PROTOCOL_DIR / "predicate_v2_1.py",
    PROTOCOL_DIR / "stage_a_driver.py",
    HOST_LABELS,
    GUEST_LABELS,
]

EXPECTED_LEVELS = {
    "HOST_L2": {1, 16, 21, 34, 42, 46, 68, 84, 97, 105, 111, 121, 123, 160, 168, 180, 187},
    "HOST_L3": {1, 14, 16, 21, 22, 24, 30, 34, 42, 46, 52, 55, 61, 63, 68, 73, 77,
                82, 84, 97, 99, 105, 111, 121, 123, 133, 137, 159, 160, 168, 180, 187},
    "GUEST_L2_DESIGN": {47, 254},
    "GUEST_L2_MECH": {27, 47, 254, 275, 348, 374, 1023},
    "GUEST_L3_STRESS": {27, 47, 254, 275, 348, 374, 1023},
}

LEVEL_GUEST_KEY = {
    "DESIGN": "GUEST_L2_DESIGN",
    "MECH": "GUEST_L2_MECH",
    "STRESS": "GUEST_L3_STRESS",
}
LEVEL_ORDER = ["ORIGINAL", "DESIGN", "MECH", "STRESS"]
NULLS = ["additive", "host_only_RF", "guest_only_RF", "cond_only_RF", "source_only_RF", "sim_knn_k5"]
LEGACY_STRICT = {27, 176, 315, 343, 597, 612, 752, 1173, 1464, 1950, 1961, 2129}
LEGACY_WIDE = LEGACY_STRICT | {48, 458, 495, 890}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NaN"
        return format(float(value), ".17g")
    return str(value)


def stable_tsv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buf.getvalue()


def write_new(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    if path.exists():
        assert path.read_bytes() == encoded, f"EXISTING_OUTPUT_MISMATCH: {path}"
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def exact_wilcoxon(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[values != 0.0]
    ranks = rankdata(np.abs(values), method="average")
    w_plus = float(ranks[values > 0].sum())
    w_minus = float(ranks[values < 0].sum())
    observed = min(w_plus, w_minus)
    total_rank = float(ranks.sum())
    extreme = 0
    denominator = 2 ** len(values)
    for bits in itertools.product((0, 1), repeat=len(values)):
        candidate_plus = sum(ranks[i] for i, bit in enumerate(bits) if bit)
        candidate_w = min(candidate_plus, total_rank - candidate_plus)
        if candidate_w <= observed + 1e-12:
            extreme += 1
    return {
        "W": observed,
        "p": extreme / denominator,
        "numerator": extreme,
        "denominator": denominator,
        "n_nonzero": len(values),
    }


def exact_sign_test(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    n = positive + negative
    k = min(positive, negative)
    denominator = 2 ** n
    numerator = min(denominator, 2 * sum(math.comb(n, i) for i in range(k + 1)))
    return {
        "positive": positive,
        "negative": negative,
        "n_nonzero": n,
        "p": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def read_frozen_levels() -> dict[str, set[int]]:
    host = pd.read_csv(HOST_LABELS, sep="\t", usecols=["host_id", "HOST_L2", "HOST_L3"])
    guest = pd.read_csv(
        GUEST_LABELS,
        sep="\t",
        usecols=["guest_id", "GUEST_L2_DESIGN", "GUEST_L2_MECH", "GUEST_L3_STRESS"],
    )
    actual = {
        "HOST_L2": set(host.loc[host["HOST_L2"].eq(1), "host_id"].astype(int)),
        "HOST_L3": set(host.loc[host["HOST_L3"].eq(1), "host_id"].astype(int)),
        "GUEST_L2_DESIGN": set(guest.loc[guest["GUEST_L2_DESIGN"].eq(1), "guest_id"].astype(int)),
        "GUEST_L2_MECH": set(guest.loc[guest["GUEST_L2_MECH"].eq(1), "guest_id"].astype(int)),
        "GUEST_L3_STRESS": set(guest.loc[guest["GUEST_L3_STRESS"].eq(1), "guest_id"].astype(int)),
    }
    for key, expected in EXPECTED_LEVELS.items():
        assert actual[key] == expected, f"LEVEL_SET_MISMATCH {key}: actual={sorted(actual[key])}"
    return actual


def r2_for_model(frame: pd.DataFrame, model: str, guard_source: bool) -> float:
    part = frame.loc[frame["model"].eq(model)]
    if part.empty:
        raise AssertionError(f"MODEL_MISSING: {model}")
    if guard_source and model == "source_only_RF" and float(part["y_pred"].std(ddof=0)) <= 1e-9:
        return float("nan")
    return float(r2_score(part["y_true"].to_numpy(), part["y_pred"].to_numpy()))


def block_table(
    dc: pd.DataFrame,
    excluded_rows: set[int],
    released_delta: dict[int, float],
    guard_source: bool,
) -> list[dict]:
    rows = []
    for block in sorted(int(x) for x in dc["fold_or_block"].unique()):
        full = dc.loc[dc["fold_or_block"].eq(block)]
        remaining = full.loc[~full["row_index"].isin(excluded_rows)]
        n_full = int(full.loc[full["model"].eq("PAIR_RF")].shape[0])
        n = int(remaining.loc[remaining["model"].eq("PAIR_RF")].shape[0])
        scores = {model: r2_for_model(remaining, model, guard_source) for model in ["PAIR_RF"] + NULLS}
        best_null = max(NULLS, key=lambda model: -np.inf if not np.isfinite(scores[model]) else scores[model])
        recomputed_delta = scores["PAIR_RF"] - scores[best_null]
        delta = released_delta[block] if n == n_full else recomputed_delta
        row = {
            "block": block,
            "N": n,
            "excluded_appearances": n_full - n,
            "PAIR_RF_R2": scores["PAIR_RF"],
            "best_null": best_null,
            "best_null_R2": scores[best_null],
            "delta_R2": float(delta),
            "delta_recomputed": float(recomputed_delta),
        }
        for model in NULLS:
            row[model + "_R2"] = scores[model]
        rows.append(row)
    return rows


def summarize_blocks(rows: list[dict]) -> dict:
    deltas = np.asarray([row["delta_R2"] for row in rows], dtype=float)
    wilcoxon = exact_wilcoxon(deltas)
    sign = exact_sign_test(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "positive_blocks": int((deltas > 0).sum()),
        "wilcoxon": wilcoxon,
        "sign": sign,
    }


def family_metrics(frame: pd.DataFrame, excluded_rows: set[int]) -> dict:
    remaining = frame.loc[~frame["row_index"].isin(excluded_rows)].copy()
    y = remaining["y_true"].to_numpy(dtype=float)
    pred = remaining["y_pred"].to_numpy(dtype=float)
    error = y - pred
    return {
        "excluded_n": int(frame.shape[0] - remaining.shape[0]),
        "excluded_pairs": int(
            frame.loc[frame["row_index"].isin(excluded_rows), ["host_id", "guest_id"]].drop_duplicates().shape[0]
        ),
        "remaining_n": int(remaining.shape[0]),
        "remaining_pairs": int(remaining[["host_id", "guest_id"]].drop_duplicates().shape[0]),
        "variance_ddof0": float(np.var(y, ddof=0)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "sst": float(np.sum((y - y.mean()) ** 2)),
        "sse": float(np.sum(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def set_text(values: set[int]) -> str:
    return "{" + ",".join(str(value) for value in sorted(values)) + "}"


def main() -> None:
    for path, expected in EXPECTED_SHA.items():
        actual = sha256_file(path)
        assert actual == expected, f"SHA_MISMATCH {path}: {actual} != {expected}"

    stage_a_before = {path.name: sha256_file(path) for path in STAGE_A_FILES}
    levels = read_frozen_levels()

    dc = pd.read_csv(
        DC_PRED,
        usecols=["row_index", "fold_or_block", "host_id", "guest_id", "y_true", "y_pred", "model"],
        float_precision="round_trip",
    )
    fc = pd.read_csv(
        FC_PRED,
        usecols=["row_index", "fold_or_block", "host_id", "guest_id", "y_true", "y_pred", "model"],
        float_precision="round_trip",
    )
    block_delta = pd.read_csv(
        BLOCK_DELTA,
        usecols=["block", "delta_PAIR_RF", "best_null_R2"],
        float_precision="round_trip",
    )
    released_delta = dict(zip(block_delta["block"].astype(int), block_delta["delta_PAIR_RF"].astype(float)))

    dc_pair = dc.loc[dc["model"].eq("PAIR_RF")].copy()
    fc_calix = fc.loc[fc["model"].eq("PAIR_RF") & fc["fold_or_block"].eq("calixarene")].copy()
    assert dc_pair.shape[0] == 1984
    assert dc_pair["row_index"].nunique() == 1382
    assert dc_pair[["host_id", "guest_id"]].drop_duplicates().shape[0] == 1312
    assert fc_calix.shape[0] == 536
    assert fc_calix[["host_id", "guest_id"]].drop_duplicates().shape[0] == 527

    original_guard = block_table(dc, set(), released_delta, guard_source=True)
    original_bare = block_table(dc, set(), released_delta, guard_source=False)
    original_summary = summarize_blocks(original_guard)
    original_family = family_metrics(fc_calix, set())

    expected_mean = 0.095039911151674
    expected_family_r2 = 0.246453042408168
    assert math.isclose(original_summary["mean_delta"], expected_mean, rel_tol=1e-12, abs_tol=0.0)
    assert original_summary["wilcoxon"]["W"] == 2.0
    assert original_summary["wilcoxon"]["numerator"] == 6
    assert original_summary["wilcoxon"]["denominator"] == 4096
    assert original_summary["positive_blocks"] == 11
    assert original_guard[8]["best_null"] == "sim_knn_k5"
    assert original_family["remaining_n"] == 536
    assert original_family["remaining_pairs"] == 527
    assert math.isclose(original_family["r2"], expected_family_r2, rel_tol=1e-12, abs_tol=0.0)

    source_delta_diagnostics = []
    for row in original_guard:
        source_delta_diagnostics.append(
            {
                "block": row["block"],
                "computed": row["delta_recomputed"],
                "released": released_delta[row["block"]],
                "isclose_rel1e-12": math.isclose(
                    row["delta_recomputed"], released_delta[row["block"]], rel_tol=1e-12, abs_tol=0.0
                ),
            }
        )
    assert all(row["isclose_rel1e-12"] for row in source_delta_diagnostics)

    clean = pd.read_csv(CLEAN, usecols=["host_smiles", "guest_smiles"], keep_default_na=False)
    clean.insert(0, "row_index", np.arange(clean.shape[0], dtype=int))
    host_map = pd.read_csv(HOST_MAP, usecols=["host_id", "host_smiles"], keep_default_na=False)
    guest_map = pd.read_csv(GUEST_MAP, usecols=["guest_id", "guest_smiles"], keep_default_na=False)
    assert host_map["host_smiles"].is_unique
    assert guest_map["guest_smiles"].is_unique
    clean = clean.merge(host_map, on="host_smiles", how="left", validate="many_to_one")
    clean = clean.merge(guest_map, on="guest_smiles", how="left", validate="many_to_one")
    assert clean["host_id"].notna().all() and clean["guest_id"].notna().all()
    clean["host_id"] = clean["host_id"].astype(int)
    clean["guest_id"] = clean["guest_id"].astype(int)
    clean_by_row = clean.set_index("row_index")

    for label, pred in [("double", dc_pair), ("family", fc_calix)]:
        unique = pred[["row_index", "host_id", "guest_id"]].drop_duplicates()
        mapped = clean_by_row.loc[unique["row_index"], ["host_id", "guest_id"]].reset_index(drop=True)
        assert np.array_equal(mapped["host_id"].to_numpy(), unique["host_id"].astype(int).to_numpy()), label
        assert np.array_equal(mapped["guest_id"].to_numpy(), unique["guest_id"].astype(int).to_numpy()), label

    mapping_samples = []
    for row_index in [27, 48, 315, 1950, 1961]:
        row = clean_by_row.loc[row_index]
        mapping_samples.append((row_index, int(row["host_id"]), int(row["guest_id"])))
    assert mapping_samples == [(27, 21, 27), (48, 1, 47), (315, 1, 254), (1950, 1, 1023), (1961, 21, 27)]

    dc_folds = pd.read_csv(
        DC_FOLDS,
        usecols=["fold_index", "role", "row_index", "host_id", "guest_id"],
        keep_default_na=False,
    )
    dc_folds = dc_folds.loc[dc_folds["role"].eq("test")]
    dc_keys = dc_pair[["fold_or_block", "row_index", "host_id", "guest_id"]].copy()
    dc_keys.columns = ["fold_index", "row_index", "host_id", "guest_id"]
    assert set(map(tuple, dc_keys.astype(int).to_numpy())) == set(
        map(tuple, dc_folds[["fold_index", "row_index", "host_id", "guest_id"]].astype(int).to_numpy())
    )

    fc_folds = pd.read_csv(
        FC_FOLDS,
        usecols=["held_family", "role", "row_index", "host_id", "guest_id"],
        keep_default_na=False,
    )
    fc_folds = fc_folds.loc[fc_folds["role"].eq("test") & fc_folds["held_family"].eq("calixarene")]
    assert set(map(tuple, fc_calix[["row_index", "host_id", "guest_id"]].astype(int).to_numpy())) == set(
        map(tuple, fc_folds[["row_index", "host_id", "guest_id"]].astype(int).to_numpy())
    )

    host_l3 = levels["HOST_L3"]
    global_level_rows = {}
    for level in ["DESIGN", "MECH", "STRESS"]:
        guest_ids = levels[LEVEL_GUEST_KEY[level]]
        mask = clean["host_id"].isin(host_l3) & clean["guest_id"].isin(guest_ids)
        global_level_rows[level] = set(clean.loc[mask, "row_index"].astype(int))

    dc_test_rows = set(dc_pair["row_index"].astype(int))
    fc_test_rows = set(fc_calix["row_index"].astype(int))
    dc_level_rows = {level: rows & dc_test_rows for level, rows in global_level_rows.items()}
    fc_level_rows = {level: rows & fc_test_rows for level, rows in global_level_rows.items()}

    legacy_rows = sorted(LEGACY_WIDE | global_level_rows["STRESS"])
    legacy_diff_rows = []
    for row_index in legacy_rows:
        mapped = clean_by_row.loc[row_index]
        legacy_diff_rows.append(
            {
                "row_index": row_index,
                "in_v2.2_DESIGN": int(row_index in global_level_rows["DESIGN"]),
                "in_v2.2_MECH": int(row_index in global_level_rows["MECH"]),
                "in_v2.2_STRESS": int(row_index in global_level_rows["STRESS"]),
                "in_legacy_STRICT": int(row_index in LEGACY_STRICT),
                "in_legacy_WIDE": int(row_index in LEGACY_WIDE),
                "host_id": int(mapped["host_id"]),
                "guest_id": int(mapped["guest_id"]),
            }
        )

    diff_summary = {}
    for projection_name, level_rows, legacy_projection in [
        ("global", global_level_rows, {"STRICT": LEGACY_STRICT, "WIDE": LEGACY_WIDE}),
        (
            "double_cold_test",
            dc_level_rows,
            {"STRICT": LEGACY_STRICT & dc_test_rows, "WIDE": LEGACY_WIDE & dc_test_rows},
        ),
    ]:
        diff_summary[projection_name] = {}
        for level in ["DESIGN", "MECH", "STRESS"]:
            diff_summary[projection_name][level] = {}
            for legacy_name in ["STRICT", "WIDE"]:
                new = level_rows[level]
                old = legacy_projection[legacy_name]
                diff_summary[projection_name][level][legacy_name] = {
                    "added": new - old,
                    "removed": old - new,
                    "common": new & old,
                }

    def manifest_rows(level_rows: dict[str, set[int]], test_rows: set[int]) -> list[dict]:
        rows = []
        for level in ["DESIGN", "MECH", "STRESS"]:
            for row_index in sorted(level_rows[level]):
                mapped = clean_by_row.loc[row_index]
                rows.append(
                    {
                        "row_index": row_index,
                        "host_id": int(mapped["host_id"]),
                        "guest_id": int(mapped["guest_id"]),
                        "level_hit": level,
                        "in_test": int(row_index in test_rows),
                    }
                )
        return rows

    dc_manifest_rows = manifest_rows(global_level_rows, dc_test_rows)
    fc_manifest_rows = manifest_rows(global_level_rows, fc_test_rows)

    block_rows_by_level = {"ORIGINAL": original_guard}
    bare_rows_by_level = {"ORIGINAL": original_bare}
    family_by_level = {"ORIGINAL": original_family}
    summaries = {"ORIGINAL": original_summary}
    denominator_rows = {}

    for level in ["DESIGN", "MECH", "STRESS"]:
        block_rows_by_level[level] = block_table(dc, dc_level_rows[level], released_delta, guard_source=True)
        bare_rows_by_level[level] = block_table(dc, dc_level_rows[level], released_delta, guard_source=False)
        summaries[level] = summarize_blocks(block_rows_by_level[level])
        family_by_level[level] = family_metrics(fc_calix, fc_level_rows[level])

        dc_excluded = dc_pair.loc[dc_pair["row_index"].isin(dc_level_rows[level])]
        denominator_rows[level] = {
            "appearances": int(dc_excluded.shape[0]),
            "unique_rows": int(dc_excluded["row_index"].nunique()),
            "unique_pairs": int(dc_excluded[["host_id", "guest_id"]].drop_duplicates().shape[0]),
        }

    base_best = {row["block"]: row["best_null"] for row in original_guard}
    guard_bare = {}
    affected_blocks = {}
    for level in LEVEL_ORDER:
        guard_names = {row["block"]: row["best_null"] for row in block_rows_by_level[level]}
        bare_names = {row["block"]: row["best_null"] for row in bare_rows_by_level[level]}
        guard_bare[level] = {
            "same": guard_names == bare_names,
            "different_blocks": {block for block in guard_names if guard_names[block] != bare_names[block]},
        }
        affected_blocks[level] = {
            row["block"]: row["excluded_appearances"]
            for row in block_rows_by_level[level]
            if row["excluded_appearances"] > 0
        }

    assert all(item["same"] for item in guard_bare.values())

    legacy_text = stable_tsv(
        legacy_diff_rows,
        [
            "row_index", "in_v2.2_DESIGN", "in_v2.2_MECH", "in_v2.2_STRESS",
            "in_legacy_STRICT", "in_legacy_WIDE", "host_id", "guest_id",
        ],
    )
    manifest_columns = ["row_index", "host_id", "guest_id", "level_hit", "in_test"]
    dc_manifest_text = stable_tsv(dc_manifest_rows, manifest_columns)
    fc_manifest_text = stable_tsv(fc_manifest_rows, manifest_columns)

    lines = []
    lines.append("# D6SC-overlap Stage-B reaggregation")
    lines.append("")
    lines.append("This is an outcome-side evaluation-mask reaggregation over frozen Stage-A labels. It does not rerun the predicate or retrain any model.")
    lines.append("")
    lines.append("## Frozen inputs")
    lines.append("")
    lines.append(f"- host labels SHA-256: `{sha256_file(HOST_LABELS)}`")
    lines.append(f"- guest labels SHA-256: `{sha256_file(GUEST_LABELS)}`")
    for key in ["HOST_L2", "HOST_L3", "GUEST_L2_DESIGN", "GUEST_L2_MECH", "GUEST_L3_STRESS"]:
        lines.append(f"- {key}: `{sorted(levels[key])}`")
    lines.append("")
    lines.append("All three exclusion levels use frozen HOST_L3. Guest membership is DESIGN, MECH, or STRESS respectively.")
    lines.append("")
    lines.append("## B1 baseline reproduction")
    lines.append("")
    lines.append("| Quantity | Actual | Locked reference | rel_tol=1e-12 |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| Double-cold mean delta R2 | {fmt(original_summary['mean_delta'])} | {fmt(expected_mean)} | MATCH |")
    lines.append(f"| Wilcoxon W | {fmt(original_summary['wilcoxon']['W'])} | 2 | MATCH |")
    lines.append(
        f"| Wilcoxon exact two-sided p | {fmt(original_summary['wilcoxon']['p'])} "
        f"({original_summary['wilcoxon']['numerator']}/{original_summary['wilcoxon']['denominator']}) | 6/4096 | MATCH |"
    )
    lines.append(f"| Positive blocks | {original_summary['positive_blocks']}/12 | 11/12 | MATCH |")
    lines.append(f"| Block 8 best null | {original_guard[8]['best_null']} | sim_knn_k5 | MATCH |")
    lines.append(f"| Family calixarene R2 | {fmt(original_family['r2'])} | {fmt(expected_family_r2)} | MATCH |")
    lines.append(f"| Family calixarene measurements | {original_family['remaining_n']} | 536 | MATCH |")
    lines.append(f"| Family calixarene unique pairs | {original_family['remaining_pairs']} | 527 | MATCH |")
    lines.append("")
    lines.append("The 12 recomputed original block deltas each match the released block-level values under rel_tol=1e-12.")
    lines.append("")
    lines.append("## row_index mapping audit")
    lines.append("")
    lines.append("Mapping used: predictions.row_index -> zero-based physical row in suprabench_bap_clean_cond.csv -> exact SMILES -> ID maps. parquet_row was not used.")
    lines.append("")
    lines.append("| row_index | host_id | guest_id |")
    lines.append("|---:|---:|---:|")
    for row_index, host_id, guest_id in mapping_samples:
        lines.append(f"| {row_index} | {host_id} | {guest_id} |")
    lines.append("")
    lines.append("Full prediction-to-map identifier mismatch count: 0 for both host_id and guest_id in both evaluation sets.")
    lines.append("")
    lines.append("## Legacy diff")
    lines.append("")
    lines.append("The global comparison records the supplied historical row sets. The double-cold projection intersects both sides with the released double-cold test row_index set.")
    lines.append("")
    for projection in ["global", "double_cold_test"]:
        lines.append(f"### {projection}")
        lines.append("")
        lines.append("| New level | Legacy | Added | Removed | Common |")
        lines.append("|---|---|---|---|---|")
        for level in ["DESIGN", "MECH", "STRESS"]:
            for legacy_name in ["STRICT", "WIDE"]:
                item = diff_summary[projection][level][legacy_name]
                lines.append(
                    f"| {level} | {legacy_name} | {set_text(item['added'])} | "
                    f"{set_text(item['removed'])} | {set_text(item['common'])} |"
                )
        lines.append("")
    lines.append("`legacy_diff.tsv` is the row-wise global union table. No diff was used to alter frozen labels.")
    lines.append("")
    lines.append("## Exclusion denominators")
    lines.append("")
    lines.append("Double-cold denominators: 1,984 block appearances; 1,382 unique row_index values; 1,312 unique standardized pairs.")
    lines.append("")
    lines.append("| Level | Excluded appearances | Rate | Excluded unique rows | Rate | Excluded unique pairs | Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for level in ["DESIGN", "MECH", "STRESS"]:
        item = denominator_rows[level]
        lines.append(
            f"| {level} | {item['appearances']}/1984 | {100*item['appearances']/1984:.6f}% | "
            f"{item['unique_rows']}/1382 | {100*item['unique_rows']/1382:.6f}% | "
            f"{item['unique_pairs']}/1312 | {100*item['unique_pairs']/1312:.6f}% |"
        )
    lines.append("")
    lines.append("These exclusions are small under each like-for-like denominator; DESIGN contains two frozen guest IDs and MECH/STRESS contain seven.")
    lines.append("")
    lines.append("## Double-cold summary")
    lines.append("")
    lines.append("| Level | Mean delta R2 | Positive blocks | Wilcoxon W | Exact p | Sign-test p | Affected blocks | Best-null changes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for level in LEVEL_ORDER:
        summary = summaries[level]
        changed = sum(row["best_null"] != base_best[row["block"]] for row in block_rows_by_level[level])
        affected = ", ".join(f"{block}:{count}" for block, count in sorted(affected_blocks[level].items())) or "none"
        lines.append(
            f"| {level} | {fmt(summary['mean_delta'])} | {summary['positive_blocks']}/12 | "
            f"{fmt(summary['wilcoxon']['W'])} | {fmt(summary['wilcoxon']['p'])} "
            f"({summary['wilcoxon']['numerator']}/{summary['wilcoxon']['denominator']}) | "
            f"{fmt(summary['sign']['p'])} ({summary['sign']['numerator']}/{summary['sign']['denominator']}) | "
            f"{affected} | {changed}/12 |"
        )
    lines.append("")
    lines.append("## Double-cold per-block results")
    lines.append("")
    lines.append("| Level | Block | N | Excluded appearances | PAIR_RF R2 | Best null | Best-null R2 | Delta R2 | Best-null changed |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---|")
    for level in LEVEL_ORDER:
        for row in block_rows_by_level[level]:
            lines.append(
                f"| {level} | {row['block']} | {row['N']} | {row['excluded_appearances']} | "
                f"{fmt(row['PAIR_RF_R2'])} | {row['best_null']} | {fmt(row['best_null_R2'])} | "
                f"{fmt(row['delta_R2'])} | {row['best_null'] != base_best[row['block']]} |"
            )
    lines.append("")
    lines.append("## Six-null R2 values under the released guard")
    lines.append("")
    lines.append("| Level | Block | additive | host_only_RF | guest_only_RF | cond_only_RF | source_only_RF | sim_knn_k5 | Best null |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for level in LEVEL_ORDER:
        for row in block_rows_by_level[level]:
            lines.append(
                f"| {level} | {row['block']} | {fmt(row['additive_R2'])} | {fmt(row['host_only_RF_R2'])} | "
                f"{fmt(row['guest_only_RF_R2'])} | {fmt(row['cond_only_RF_R2'])} | "
                f"{fmt(row['source_only_RF_R2'])} | {fmt(row['sim_knn_k5_R2'])} | {row['best_null']} |"
            )
    lines.append("")
    lines.append("## Guard versus bare source-only_RF")
    lines.append("")
    lines.append("The five non-source null R2 values are identical by construction; the table exposes the only changing null, source_only_RF, together with both best-null identities.")
    lines.append("")
    lines.append("| Level | Block | Guard source_only_RF R2 | Bare source_only_RF R2 | Guard best | Bare best | Same identity |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for level in LEVEL_ORDER:
        for guard_row, bare_row in zip(block_rows_by_level[level], bare_rows_by_level[level]):
            assert guard_row["block"] == bare_row["block"]
            for model in ["additive", "host_only_RF", "guest_only_RF", "cond_only_RF", "sim_knn_k5"]:
                assert guard_row[model + "_R2"] == bare_row[model + "_R2"]
            lines.append(
                f"| {level} | {guard_row['block']} | {fmt(guard_row['source_only_RF_R2'])} | "
                f"{fmt(bare_row['source_only_RF_R2'])} | {guard_row['best_null']} | "
                f"{bare_row['best_null']} | {guard_row['best_null'] == bare_row['best_null']} |"
            )
    lines.append("")
    lines.append("Across all four reported states, guard and bare source_only_RF produce identical best-null identities in 12/12 blocks; reported deltas are therefore unchanged by this guard choice.")
    lines.append("")
    lines.append("## Family-cold calixarene summary")
    lines.append("")
    lines.append("| Level | Excluded measurements | Excluded pairs | Remaining measurements | Remaining pairs | Var(y, ddof=0) | R2 | Spearman rho | SST | SSE | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for level in LEVEL_ORDER:
        item = family_by_level[level]
        lines.append(
            f"| {level} | {item['excluded_n']} | {item['excluded_pairs']} | {item['remaining_n']} | "
            f"{item['remaining_pairs']} | {fmt(item['variance_ddof0'])} | {fmt(item['r2'])} | "
            f"{fmt(item['spearman'])} | {fmt(item['sst'])} | {fmt(item['sse'])} | {fmt(item['rmse'])} |"
        )
    lines.append("")
    lines.append("## Directional evidence")
    lines.append("")
    for level in ["DESIGN", "MECH", "STRESS"]:
        mean_change = summaries[level]["mean_delta"] - summaries["ORIGINAL"]["mean_delta"]
        p_change = summaries[level]["wilcoxon"]["p"] - summaries["ORIGINAL"]["wilcoxon"]["p"]
        family_change = family_by_level[level]["r2"] - family_by_level["ORIGINAL"]["r2"]
        lines.append(
            f"- {level}: mean delta R2 change {fmt(mean_change)}; Wilcoxon-p change {fmt(p_change)}; "
            f"positive blocks {summaries['ORIGINAL']['positive_blocks']} -> {summaries[level]['positive_blocks']}; "
            f"family R2 change {fmt(family_change)}; best-null changes "
            f"{sum(row['best_null'] != base_best[row['block']] for row in block_rows_by_level[level])}/12."
        )
    lines.append("")
    lines.append("These are directional measurements for three fixed estimands, not a selection among levels and not a manuscript conclusion.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```powershell")
    lines.append("$env:PYTHONIOENCODING='utf-8'")
    lines.append("$env:PYTHONDONTWRITEBYTECODE='1'")
    lines.append("python protocols/p1_d6sc_overlap/stage_b_reaggregate.py")
    lines.append("```")
    lines.append("")
    results_text = "\n".join(lines)

    write_new(LEGACY_DIFF_PATH, legacy_text)
    write_new(DC_MANIFEST_PATH, dc_manifest_text)
    write_new(FC_MANIFEST_PATH, fc_manifest_text)
    write_new(RESULTS_PATH, results_text)

    stage_a_after = {path.name: sha256_file(path) for path in STAGE_A_FILES}
    assert stage_a_after == stage_a_before, "STAGE_A_FILE_CHANGED"

    outputs = [Path(__file__), RESULTS_PATH, DC_MANIFEST_PATH, FC_MANIFEST_PATH, LEGACY_DIFF_PATH]
    print("B1_REPRODUCED=True")
    print("B1_MEAN_DELTA=" + fmt(original_summary["mean_delta"]))
    print("B1_FAMILY_R2=" + fmt(original_family["r2"]))
    print("LEVELS_ASSERTED=True")
    print("MAPPING_MISMATCHES=0")
    print("MAPPING_SAMPLES=" + repr(mapping_samples))
    for level in LEVEL_ORDER:
        print(
            "SUMMARY %s mean_delta=%s positive=%d W=%s wilcoxon_p=%s sign_p=%s family_R2=%s family_rho=%s"
            % (
                level,
                fmt(summaries[level]["mean_delta"]),
                summaries[level]["positive_blocks"],
                fmt(summaries[level]["wilcoxon"]["W"]),
                fmt(summaries[level]["wilcoxon"]["p"]),
                fmt(summaries[level]["sign"]["p"]),
                fmt(family_by_level[level]["r2"]),
                fmt(family_by_level[level]["spearman"]),
            )
        )
    for level in ["DESIGN", "MECH", "STRESS"]:
        print("AFFECTED_BLOCKS %s %s" % (level, affected_blocks[level]))
        print("DENOMINATORS %s %s" % (level, denominator_rows[level]))
    print("GUARD_BARE_ALL_IDENTICAL=" + str(all(item["same"] for item in guard_bare.values())))
    print("STAGE_A_SHA_UNCHANGED=True")
    for path in outputs:
        print("FILE=%s SHA256=%s BYTES=%d" % (path.name, sha256_file(path), path.stat().st_size))


if __name__ == "__main__":
    main()
