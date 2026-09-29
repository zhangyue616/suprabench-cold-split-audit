
# =====================================================================
# STAGE-A DRIVER v4 for D6SC_OVERLAP_PREDICATE v2.2
# 冻结版本锁定 2026-07-18。driver v3→v4 变更:
#  - 新增 POST: EXP_GUEST_DESIGN_1={47,254}(N 端卡后严格 DESIGN)。
#  - guest 序列化列新增 NTERM_REQ / NTERM_OK(N 端规格可审计)。
#  - v2.2 vs v2.1 自证:DESIGN 应从 {47,254,374,1023} 变 {47,254};
#    host L2/L3、guest MECH/STRESS 应与 v2.1 逐一相同。
#  - 覆盖写(makedirs exist_ok=True);其余同 v3。
# POST 期望来源 = 公开化学 + 上一轮结构分类 + 冻结的 H3 位点和 N 端规格。**非 outcome、非 25-row legacy 清单。**
# =====================================================================
import os, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = os.environ.get("SUPRABENCH_P1_OUTPUT", str(ROOT / "runs/p1_d6sc_overlap/stage_a"))
HOST_CSV = str(ROOT / "data/folds/host_id_map.csv")
GUEST_CSV = str(ROOT / "data/folds/guest_id_map.csv")
MANIFEST_CSV = str(ROOT / "data/population/full_bap_manifest.csv")

HOST_COLS = ["host_id","host_smiles","n_host_names","host_names","SULFO_TOTAL",
             "calix_scaffold","is_calix_name","is_blocked_name","name_status",
             "name_calix_n","NAME_STRUCT_CONFLICT","SMILES_OK","HOST_L2","HOST_L3"]
GUEST_COLS = ["guest_id","guest_name_resolved","name_status_multi","bare","seq_len",
              "ptms","ptm_pos","notation","parse_ok","note","PEPTIDE_BB2","lys_hits",
              "KME_N","KME_counts","ANY_NME","KME","PEPTIDE","H3","D6SC_SPEC",
              "NTERM","CTERM","NTERM_REQ","NTERM_OK","SMILES_OK","full_hit","full_pos",
              "drop_last_hit","drop_last_pos","last_res","K_pos_in_H3","longest_run",
              "longest_pos","GUEST_L2_DESIGN","GUEST_L2_MECH","GUEST_L3_STRESS"]

EXP_HOST_L3_1 = {1,14,16,21,22,24,30,34,42,46,52,55,61,63,68,73,77,82,84,97,99,
                 105,111,121,123,133,137,159,160,168,180,187}
EXP_HOST_L3_0 = {57,58,110,45,141,176,177,178,89,170}
EXP_HOST_L2_1 = {1,16,21,34,42,46,68,84,97,105,111,121,123,160,168,180,187}
EXP_GUEST_MECH_1 = {27,47,254,275,348,374,1023}
EXP_GUEST_DESIGN_1 = {47,254}   # v2.2: N 端规格后的严格 DESIGN
EXP_GUEST_CTRL_0 = {86,399,724,1019,71,231,394}
V2_HOST_L3 = EXP_HOST_L3_1 | {57,58,110}
EXP_V2_SYMDIFF = {57,58,110}
V2_1_GUEST_DESIGN = {47,254,374,1023}   # v2.1 的 DESIGN(N 端卡前),供变化自证

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

def build_host_rows():
    hdf = pd.read_csv(HOST_CSV, usecols=["host_id","host_smiles","n_host_names",
                      "host_names"], keep_default_na=False)
    rows = []
    for r in hdf.itertuples(index=False):
        f, _ = feat(r.host_smiles)
        hl = host_labels(f, r.host_names)
        rows.append({"host_id": int(r.host_id), "host_smiles": r.host_smiles,
                     "n_host_names": r.n_host_names, "host_names": r.host_names,
                     "SULFO_TOTAL": hl["SULFO_TOTAL"], "calix_scaffold": hl["calix_scaffold"],
                     "is_calix_name": hl["is_calix_name"], "is_blocked_name": hl["is_blocked_name"],
                     "name_status": hl["name_status"], "name_calix_n": hl["name_calix_n"],
                     "NAME_STRUCT_CONFLICT": hl["NAME_STRUCT_CONFLICT"], "SMILES_OK": hl["SMILES_OK"],
                     "HOST_L2": hl["HOST_L2"], "HOST_L3": hl["HOST_L3"]})
    return sorted(rows, key=lambda x: x["host_id"])

def build_guest_rows():
    gdf = pd.read_csv(GUEST_CSV, usecols=["guest_id","guest_smiles"], keep_default_na=False)
    man = pd.read_csv(MANIFEST_CSV, usecols=["guest_smiles","guest_name"], keep_default_na=False)
    man = man[["guest_smiles","guest_name"]]
    smi2names = {}
    for smi, nm in man.itertuples(index=False, name=None):
        smi2names.setdefault(smi, [])
        if str(nm).strip() and nm not in smi2names[smi]:
            smi2names[smi].append(nm)
    rows = []
    for r in gdf.itertuples(index=False):
        names = smi2names.get(r.guest_smiles, [])
        rname, mstatus = resolve_guest_name(names)
        multi_amb = 0
        if len(names) >= 2:
            trips = set()
            for cand in names:
                sq = parse_seq(cand); h3c = h3_evidence(sq["bare"])
                fq, mq = feat(r.guest_smiles)
                glc = guest_labels(fq, mq, sq, h3c)
                trips.add((glc["GUEST_L2_DESIGN"], glc["GUEST_L2_MECH"], glc["GUEST_L3_STRESS"]))
            if len(trips) > 1: multi_amb = 1
        seq = parse_seq(rname); h3 = h3_evidence(seq["bare"])
        f, mol = feat(r.guest_smiles)
        gl = guest_labels(f, mol, seq, h3)
        if multi_amb:
            mstatus = "NAME_MULTI_AMBIGUOUS"
            gl["GUEST_L2_DESIGN"] = gl["GUEST_L2_MECH"] = gl["GUEST_L3_STRESS"] = 0
        rows.append({"guest_id": int(r.guest_id), "guest_name_resolved": rname,
                     "name_status_multi": mstatus, "bare": seq["bare"], "seq_len": len(seq["bare"]),
                     "ptms": seq["ptms"], "ptm_pos": seq["ptm_pos"], "notation": seq["notation"],
                     "parse_ok": seq["parse_ok"], "note": seq["note"], "PEPTIDE_BB2": f["PEPTIDE_BB2"],
                     "lys_hits": gl["lys_hits"], "KME_N": gl["KME_N"], "KME_counts": gl["KME_counts"],
                     "ANY_NME": f["ANY_NME"], "KME": gl["KME"], "PEPTIDE": gl["PEPTIDE"],
                     "H3": gl["H3"], "D6SC_SPEC": gl["D6SC_SPEC"], "NTERM": gl["NTERM"],
                     "CTERM": gl["CTERM"], "NTERM_REQ": gl["NTERM_REQ"], "NTERM_OK": gl["NTERM_OK"],
                     "SMILES_OK": gl["SMILES_OK"], "full_hit": h3["full_hit"],
                     "full_pos": h3["full_pos"], "drop_last_hit": h3["drop_last_hit"],
                     "drop_last_pos": h3["drop_last_pos"], "last_res": h3["last_res"],
                     "K_pos_in_H3": h3["K_pos_in_H3"], "longest_run": h3["longest_run"],
                     "longest_pos": h3["longest_pos"], "GUEST_L2_DESIGN": gl["GUEST_L2_DESIGN"],
                     "GUEST_L2_MECH": gl["GUEST_L2_MECH"], "GUEST_L3_STRESS": gl["GUEST_L3_STRESS"]})
    return sorted(rows, key=lambda x: x["guest_id"])

def run_stage_a(predicate_text, driver_text):
    if run_selftest() != 0:
        print("SELFTEST_FAILED_ABORT"); raise SystemExit(3)
    host_rows = build_host_rows()
    guest_rows = build_guest_rows()
    host_tsv = stable_tsv(host_rows, HOST_COLS)
    guest_tsv = stable_tsv(guest_rows, GUEST_COLS)

    os.makedirs(OUTDIR, exist_ok=True)
    write_text(OUTDIR + "/predicate_v2_1.py", predicate_text)
    write_text(OUTDIR + "/stage_a_driver.py", driver_text)
    write_text(OUTDIR + "/host_labels_v2_1.tsv", host_tsv)
    write_text(OUTDIR + "/guest_labels_v2_1.tsv", guest_tsv)

    for fn in ["predicate_v2_1.py","stage_a_driver.py","host_labels_v2_1.tsv","guest_labels_v2_1.tsv"]:
        p = OUTDIR + "/" + fn
        print("FILE=%s SHA256=%s BYTES=%d" % (fn, sha256_file(p), os.path.getsize(p)))

    host_sha = sha256_file(OUTDIR + "/host_labels_v2_1.tsv")
    guest_sha = sha256_file(OUTDIR + "/guest_labels_v2_1.tsv")
    print("HOST_LABELS_SHA256=" + host_sha)
    print("HOST_LABELS_ROWS=%d data rows" % len(host_rows))
    print("GUEST_LABELS_SHA256=" + guest_sha)
    print("GUEST_LABELS_ROWS=%d data rows" % len(guest_rows))

    HL2 = {r["host_id"] for r in host_rows if r["HOST_L2"] == 1}
    HL3 = {r["host_id"] for r in host_rows if r["HOST_L3"] == 1}
    GD  = {r["guest_id"] for r in guest_rows if r["GUEST_L2_DESIGN"] == 1}
    GM  = {r["guest_id"] for r in guest_rows if r["GUEST_L2_MECH"] == 1}
    GS  = {r["guest_id"] for r in guest_rows if r["GUEST_L3_STRESS"] == 1}
    print("HOST_L2=%s" % sorted(HL2))
    print("HOST_L3=%s" % sorted(HL3))
    print("GUEST_L2_DESIGN=%s" % sorted(GD))
    print("GUEST_L2_MECH=%s" % sorted(GM))
    print("GUEST_L3_STRESS=%s" % sorted(GS))

    print("POST_HOST_L3_MISSING=%s" % sorted(EXP_HOST_L3_1 - HL3))
    print("POST_HOST_L3_EXTRA=%s"   % sorted(HL3 - EXP_HOST_L3_1))
    print("POST_HOST_L3_0_VIOLATION=%s" % sorted(EXP_HOST_L3_0 & HL3))
    print("POST_HOST_L2_MISSING=%s" % sorted(EXP_HOST_L2_1 - HL2))
    print("POST_HOST_L2_EXTRA=%s"   % sorted(HL2 - EXP_HOST_L2_1))
    print("POST_GUEST_MECH_MISSING=%s" % sorted(EXP_GUEST_MECH_1 - GM))
    print("POST_GUEST_MECH_EXTRA=%s"   % sorted(GM - EXP_GUEST_MECH_1))
    print("POST_GUEST_DESIGN_MISSING=%s" % sorted(EXP_GUEST_DESIGN_1 - GD))
    print("POST_GUEST_DESIGN_EXTRA=%s"   % sorted(GD - EXP_GUEST_DESIGN_1))
    print("POST_GUEST_CTRL_VIOLATION=%s" % sorted(EXP_GUEST_CTRL_0 & (GD | GM | GS)))
    symdiff = HL3 ^ V2_HOST_L3
    print("POST_HOST_L3_V2_SYMMETRIC_DIFF_EXPECTED=%s" % sorted(EXP_V2_SYMDIFF))
    print("POST_HOST_L3_V2_SYMMETRIC_DIFF_ACTUAL=%s" % sorted(symdiff))
    print("V2_1_TO_V2_2_DESIGN_REMOVED=%s" % sorted(V2_1_GUEST_DESIGN - GD))
    print("V2_1_TO_V2_2_DESIGN_ADDED=%s" % sorted(GD - V2_1_GUEST_DESIGN))
    print("V2_1_TO_V2_2_MECH_CHANGED=%s" % sorted(GM ^ EXP_GUEST_MECH_1))
    fails = []
    if (EXP_HOST_L3_1 - HL3) or (HL3 - EXP_HOST_L3_1): fails.append("HOST_L3_SET")
    if (EXP_HOST_L3_0 & HL3): fails.append("HOST_L3_0_VIOLATION")
    if (EXP_HOST_L2_1 - HL2) or (HL2 - EXP_HOST_L2_1): fails.append("HOST_L2_SET")
    if (EXP_GUEST_MECH_1 - GM) or (GM - EXP_GUEST_MECH_1): fails.append("GUEST_MECH_SET")
    if (EXP_GUEST_DESIGN_1 - GD) or (GD - EXP_GUEST_DESIGN_1): fails.append("GUEST_DESIGN_SET")
    if (EXP_GUEST_CTRL_0 & (GD | GM | GS)): fails.append("GUEST_CTRL_VIOLATION")
    if symdiff != EXP_V2_SYMDIFF: fails.append("V2_SYMDIFF")
    print("POST_FAILURES=%s" % fails)

    print("HOST_L2_SUBSET_L3=%s viol=%s" % (HL2 <= HL3, sorted(HL2 - HL3)))
    print("GUEST_DESIGN_SUBSET_MECH=%s viol=%s" % (GD <= GM, sorted(GD - GM)))
    print("GUEST_MECH_SUBSET_STRESS=%s viol=%s" % (GM <= GS, sorted(GM - GS)))

    amb = sorted({r["host_id"] for r in host_rows if r["NAME_STRUCT_CONFLICT"] == 1})
    print("AMBIGUOUS_HOST_NAME_STRUCT_CONFLICT_IDS=%s" % amb)
    multi_amb_ids = sorted({r["guest_id"] for r in guest_rows
                            if r["name_status_multi"] == "NAME_MULTI_AMBIGUOUS"})
    print("NAME_MULTI_AMBIGUOUS_IDS=%s" % multi_amb_ids)
    return len(fails) == 0
