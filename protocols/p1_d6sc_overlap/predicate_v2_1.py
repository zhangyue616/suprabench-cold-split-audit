
# =====================================================================
# D6SC_OVERLAP_PREDICATE v2.2 — OUTCOME-BLIND, SELF-TESTING, LEVEL-EMITTING
# 冻结版本锁定 2026-07-18。v2.1→v2.2 唯一实质改动:
#  (G) D6SC_SPEC 加条件 N 端规格（冻结的 N 端规格）:
#      K9me3 肽规格 N 端乙酰(Ac-)、其余 PTM 规格 N 端游离(H-)。
#      依赖 h3_evidence 的 K_pos_in_H3:9 in K_pos → 需 Ac-;否则需 H-。
#      374(K27me3,Ac-)与 1023(K9me3,H-)因此掉出 DESIGN;留 MECH/STRESS 不变。
# 其余(SMARTS/序列正则/host 逻辑/guest 除 SPEC 外逻辑)逐字沿用 v2.1(bdc830…)。
#  称谓纪律:L3 = 扩展磺化 calixarene-family 结构邻域压力测试;永不称 D6SC/
#  direct-source/design-concordant overlap（冻结的称谓限制）。
# 输入:仅结构 SMILES + 名称。输出:级别标签。不接触任何 outcome。
# =====================================================================
import re, io
import csv as _csv
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# H3.1 mature (P68431, Met1 removed), 135 aa. 已核:US Patent 7056683 SEQ ID NO:1
# = H3(1-40) 与本常量前 40 位逐字相同;与 H3.3(P84243) 差异位点 31/87/89/90/96。
# 本 predicate 实际用到的全部位点在 1-40 内。
H3_SEQ = ("ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTELLIRKLPFQRL"
          "VREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA")

SMA = {
  "SULFONIC_ACID":   "[SX4](=[OX1])(=[OX1])[OX2H1]",
  "SULFONATE_ANION": "[SX4](=[OX1])(=[OX1])[OX1-]",
  "SULFONATE_SALT":  "[SX4](=[OX1])(=[OX1])[OX2][#3,#11,#19]",
  "SULFONAMIDE":     "[SX4](=[OX1])(=[OX1])[NX3]",
  "AR_CH2_AR": "[c]-[CX4H2]-[c]",
  "AR_S_AR":   "[c]-[SX2]-[c]",
  "AR_CH_AR":  "[c]-[CX4H1]-[c]",
  "PEPTIDE_BB2": "[NX3,NX4+][CX4][CX3](=[OX1])[NX3][CX4][CX3](=[OX1])",
  "LYS_EPS_N": "[CX4H1]([NX3,NX4+])([CX3]=[OX1])[CX4H2][CX4H2][CX4H2][CX4H2][NX3,NX4+]",
  "ANY_NME": "[NX3,NX4+]-[CX4H3]",
}
PAT = {k: Chem.MolFromSmarts(v) for k, v in SMA.items()}

RE_CALIX_N = re.compile(r"calix\s*\[\s*(\d+)\s*\]", re.I)
RE_NTERM   = re.compile(r"^\s*(H|Ac|Bz|Boc|Fmoc)\s*-")
RE_CTERM   = re.compile(r"-\s*(NH2|NH_2|OH)\s*$")
RE_PTM_P   = re.compile(r"\((me1|me2|me3)\)")
RE_PTM_B   = re.compile(r"(?<=[A-Z])(me1|me2|me3)(?=[A-Z]|$)")
RE_THREE   = re.compile(r"-(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|"
                        r"Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)(?=-|$)")
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

BLOCKED_FAMILY_WORDS = [
    "pillar", "cucurbit", "cyclodextrin", "bambus", "tweezer", "cyclophane",
    "naphthotube", "glycoluril", "crown", "cryptand", "cryptophane", "corral",
    "oxatub", "zorb", "exbox", "sugammadex", "viologen", "paraquat", "positand",
]

def calix_family_gate(host_names_field):
    variants = [v.strip().lower() for v in str(host_names_field).split("|") if v.strip()]
    is_calix = any(("calix" in v) or ("resorcin" in v and "arene" in v) for v in variants)
    is_blocked = any(w in v for v in variants for w in BLOCKED_FAMILY_WORDS)
    return int(is_calix), int(is_blocked)

def feat(smiles):
    out = {k: 0 for k in SMA}
    out.update({"smiles_ok": False, "sanitize_err": "", "n_atoms": 0})
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        out["sanitize_err"] = "PARSE_FAIL"; return out, None
    try:
        Chem.SanitizeMol(m); out["smiles_ok"] = True
    except Exception as e:
        out["sanitize_err"] = type(e).__name__ + ": " + str(e)[:120]
        try:
            m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
        except Exception as e2:
            out["sanitize_err"] += " | RECOVER_FAIL: " + str(e2)[:80]; return out, None
    out["n_atoms"] = m.GetNumAtoms()
    for k, p in PAT.items():
        if p is None: out[k] = -1; continue
        try: out[k] = len(m.GetSubstructMatches(p, uniquify=True))
        except Exception: out[k] = -1
    return out, m

def n_methyl_on_eps_N(m):
    if m is None: return (0, 0, [])
    ms = m.GetSubstructMatches(PAT["LYS_EPS_N"], uniquify=True)
    counts = []
    for mt in ms:
        if len(mt) != 9:
            return (-1, -1, ["MATCH_TUPLE_LEN_%d" % len(mt)])
        a = m.GetAtomWithIdx(mt[8])
        counts.append(sum(1 for nb in a.GetNeighbors()
                          if nb.GetSymbol() == "C" and nb.GetDegree() == 1
                          and nb.GetTotalNumHs() == 3))
    return (len(ms), max(counts) if counts else 0, counts)

def sulfo_total(f):
    return f["SULFONIC_ACID"] + f["SULFONATE_ANION"] + f["SULFONATE_SALT"]

def name_calix_status(host_names_field, ar_ch2):
    variants = [v.strip() for v in str(host_names_field).split("|") if v.strip()]
    ns = []
    for v in variants:
        mm = RE_CALIX_N.search(v)
        ns.append(int(mm.group(1)) if mm else None)
    given = [x for x in ns if x is not None]
    if not given: return "ABSENT", ns
    if any(x != ar_ch2 for x in given): return "CONFLICT", ns
    return "MATCH", ns

def host_labels(f, host_names_field):
    if not f.get("smiles_ok"):
        return {"SULFO_TOTAL": -1, "calix_scaffold": -1, "is_calix_name": -1,
                "is_blocked_name": -1, "name_status": "SMILES_FAIL",
                "name_calix_n": [], "NAME_STRUCT_CONFLICT": -1,
                "SMILES_OK": 0, "HOST_L2": 0, "HOST_L3": 0}
    st = sulfo_total(f)
    status, ns = name_calix_status(host_names_field, f["AR_CH2_AR"])
    is_calix, is_blocked = calix_family_gate(host_names_field)
    calix_scaffold = int((f["AR_CH2_AR"] >= 4) or (f["AR_S_AR"] >= 4) or (f["AR_CH_AR"] >= 4))
    L2 = int(st >= 1 and f["AR_CH2_AR"] == 4)
    L3 = int(st >= 1 and calix_scaffold and is_calix and not is_blocked)
    return {"SULFO_TOTAL": st, "calix_scaffold": calix_scaffold,
            "is_calix_name": is_calix, "is_blocked_name": is_blocked,
            "name_status": status, "name_calix_n": ns,
            "NAME_STRUCT_CONFLICT": int(status == "CONFLICT"),
            "SMILES_OK": 1, "HOST_L2": L2, "HOST_L3": L3}

def parse_seq(name):
    out = {"bare": "", "ptms": [], "ptm_pos": [], "nterm": "", "cterm": "",
           "notation": "not_peptide", "parse_ok": False, "note": ""}
    if not name: out["note"] = "NAME_NONE"; return out
    s = str(name).strip()
    mn, mc = RE_NTERM.search(s), RE_CTERM.search(s)
    if not (mn and mc):
        out["note"] = "NOT_PEPTIDE_NOTATION"; return out
    out["nterm"], out["cterm"] = mn.group(1), mc.group(1)
    body = RE_CTERM.sub("", RE_NTERM.sub("", s))
    if RE_THREE.search("-" + body):
        out["notation"] = "three_letter"
        out["note"] = "THREE_LETTER_NOT_CONVERTED_seq_layer_skipped"
        return out
    pos = []
    for mm in list(RE_PTM_P.finditer(body)) + list(RE_PTM_B.finditer(body)):
        pre = RE_PTM_P.sub("", RE_PTM_B.sub("", body[:mm.start()]))
        pre = pre.replace("-", "").replace(" ", "")
        pos.append((len(pre), mm.group(1)))
    out["ptms"] = [p[1] for p in pos]; out["ptm_pos"] = pos
    body = RE_PTM_B.sub("", RE_PTM_P.sub("", body)).replace("-", "").replace(" ", "")
    out["bare"] = body; out["notation"] = "one_letter"
    out["parse_ok"] = len(body) >= 2 and all(c in AA20 for c in body)
    if not out["parse_ok"]: out["note"] = "NOT_PURE_AA20"
    return out

def h3_evidence(bare):
    e = {"full_hit": 0, "full_pos": -1, "drop_last_hit": 0, "drop_last_pos": -1,
         "last_res": "", "K_pos_in_H3": [], "longest_run": 0, "longest_pos": -1}
    if not bare: return e
    i = H3_SEQ.find(bare)
    e["full_hit"], e["full_pos"] = int(i >= 0), (i + 1 if i >= 0 else -1)
    probe, ppos = bare, e["full_pos"]
    if len(bare) >= 2:
        e["last_res"] = bare[-1]
        j = H3_SEQ.find(bare[:-1])
        e["drop_last_hit"], e["drop_last_pos"] = int(j >= 0), (j + 1 if j >= 0 else -1)
        if not e["full_hit"] and j >= 0: probe, ppos = bare[:-1], j + 1
    if ppos > 0:
        e["K_pos_in_H3"] = [ppos + k for k, c in enumerate(probe) if c == "K"]
    best, bpos = 0, -1
    for a in range(len(bare)):
        for b in range(len(bare), a + 2, -1):
            if b - a <= best: break
            k = H3_SEQ.find(bare[a:b])
            if k >= 0: best, bpos = b - a, k + 1; break
    e["longest_run"], e["longest_pos"] = best, bpos
    return e

def nterm_concordant(nterm, k_pos_in_h3):
    """D6SC N 端规格(条件式):K9me3 肽用 Ac-、其余 PTM 用游离 H-。
       9 in K_pos → 需 Ac-;否则 → 需 H-。返回 (concordant:int, required:str)。"""
    if 9 in set(k_pos_in_h3):
        return int(nterm == "Ac"), "Ac"
    return int(nterm == "H"), "H"

def guest_labels(f, mol, seq, h3):
    if not f.get("smiles_ok"):
        return {"lys_hits": -1, "KME_N": -1, "KME_counts": [], "KME": 0,
                "PEPTIDE": 0, "H3": 0, "D6SC_SPEC": 0, "NTERM": seq.get("nterm",""),
                "CTERM": seq.get("cterm",""), "NTERM_REQ": "", "NTERM_OK": 0,
                "SMILES_OK": 0, "GUEST_L2_DESIGN": 0, "GUEST_L2_MECH": 0,
                "GUEST_L3_STRESS": 0}
    nh, n_me, cs = n_methyl_on_eps_N(mol)
    KME  = int(nh >= 1 and n_me >= 1)
    PEP  = int(f["PEPTIDE_BB2"] >= 1)
    H3f  = int((h3["full_hit"] or h3["drop_last_hit"]) and len(h3["K_pos_in_H3"]) > 0)
    nterm_ok, nterm_req = nterm_concordant(seq["nterm"], h3["K_pos_in_H3"])
    # D6SC_SPEC(v2.2):8-mer + Tyr8 + C端酰胺 + 条件 N 端规格。
    SPEC = int(seq["parse_ok"] and len(seq["bare"]) == 8
               and seq["bare"][-1] == "Y" and seq["cterm"] == "NH2"
               and nterm_ok)
    return {"lys_hits": nh, "KME_N": n_me, "KME_counts": cs,
            "KME": KME, "PEPTIDE": PEP, "H3": H3f, "D6SC_SPEC": SPEC,
            "NTERM": seq["nterm"], "CTERM": seq["cterm"],
            "NTERM_REQ": nterm_req, "NTERM_OK": nterm_ok, "SMILES_OK": 1,
            "GUEST_L2_DESIGN": int(KME and PEP and H3f and SPEC),
            "GUEST_L2_MECH":   int(KME and PEP and H3f),
            "GUEST_L3_STRESS": int(KME and PEP)}

def resolve_guest_name(names_list):
    names = [n for n in names_list if str(n).strip()]
    if not names: return "", "NAME_NONE"
    if len(names) == 1: return names[0], "NAME_SINGLE"
    for n in sorted(names):
        if parse_seq(n)["parse_ok"]: return n, "NAME_MULTI"
    return sorted(names)[0], "NAME_MULTI"

def stable_tsv(rows, cols):
    buf = io.StringIO()
    w = _csv.writer(buf, delimiter="\t", quoting=_csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c, "")
            if v is None: line.append("")
            elif isinstance(v, bool): line.append("True" if v else "False")
            elif isinstance(v, float): line.append(repr(v))
            elif isinstance(v, (list, tuple, dict)): line.append(repr(v))
            else: line.append(str(v))
        w.writerow(line)
    return buf.getvalue()

SELFTEST = [
  ("Ac-Lys(me3)-NHMe",   "CC(=O)NC(CCCC[N+](C)(C)C)C(=O)NC", "LYS_EPS_N", 1),
  ("Ac-Lys(me2)-NHMe",   "CC(=O)NC(CCCCN(C)C)C(=O)NC",       "LYS_EPS_N", 1),
  ("Ac-Lys(me1)-NHMe",   "CC(=O)NC(CCCCNC)C(=O)NC",          "LYS_EPS_N", 1),
  ("Ac-Lys-NHMe",        "CC(=O)NC(CCCCN)C(=O)NC",           "LYS_EPS_N", 1),
  ("free Lys monomer",   "NCCCC[C@H](N)C(=O)O",              "LYS_EPS_N", 1),
  ("choline",            "C[N+](C)(C)CCO",                   "LYS_EPS_N", 0),
  ("tetramethylammonium","C[N+](C)(C)C",                     "LYS_EPS_N", 0),
  ("Gly-Gly",            "NCC(=O)NCC(=O)O",                  "PEPTIDE_BB2", 1),
  ("Gly monomer",        "NCC(=O)O",                         "PEPTIDE_BB2", 0),
  ("benzenesulfonic",    "c1ccccc1S(=O)(=O)O",               "SULFONIC_ACID", 1),
  ("benzenesulfonate",   "c1ccccc1S(=O)(=O)[O-]",            "SULFONATE_ANION", 1),
  ("Na benzenesulfonate","c1ccccc1S(=O)(=O)O[Na]",           "SULFONATE_SALT", 1),
  ("benzenesulfonamide", "c1ccccc1S(=O)(=O)N",               "SULFONAMIDE", 1),
  ("benzenesulfonamide", "c1ccccc1S(=O)(=O)N",               "SULFONIC_ACID", 0),
  ("diphenylmethane",    "c1ccccc1Cc1ccccc1",                "AR_CH2_AR", 1),
  ("diphenylsulfide",    "c1ccccc1Sc1ccccc1",                "AR_S_AR", 1),
  ("diphenylsulfide",    "c1ccccc1Sc1ccccc1",                "AR_CH2_AR", 0),
]
SELFTEST_NME = [
  ("Ac-Lys(me3)-NHMe", "CC(=O)NC(CCCC[N+](C)(C)C)C(=O)NC", 3),
  ("Ac-Lys(me2)-NHMe", "CC(=O)NC(CCCCN(C)C)C(=O)NC",       2),
  ("Ac-Lys(me1)-NHMe", "CC(=O)NC(CCCCNC)C(=O)NC",          1),
  ("Ac-Lys-NHMe",      "CC(=O)NC(CCCCN)C(=O)NC",           0),
]
SELFTEST_SEQ = [
  ("K36me3 8mer",   "H-GGVK(me3)KPHY-NH2", "GGVKKPHY", "one_letter", "NH2"),
  ("K9me3 short",   "Ac-RKme3ST-NH2",      "RKST",     "one_letter", "NH2"),
  ("K27me3 8mer",   "Ac-AARKme3SAPY-NH2",  "AARKSAPY", "one_letter", "NH2"),
  ("Tyr8 acid Cterm","H-ARTKQTAY-OH",      "ARTKQTAY", "one_letter", "OH"),
  ("not a peptide", "D/L-Tryptophan",      "",         "not_peptide", ""),
  ("three letter",  "H-Leu-Arg-Arg-Trp-Ser-Leu-Gly-OH", "", "three_letter", "OH"),
]
SELFTEST_H3 = [
  ("RKST", "RKST", 8, [9]), ("AARKSAPY", "AARKSAPY", 24, [27]),
  ("GGVKKPHY", "GGVKKPHY", 33, [36, 37]),
  ("TARKSTGY", "TARKSTGY", 6, [9]), ("ARTKQTAY", "ARTKQTAY", 1, [4]),
]
SELFTEST_NAMEGATE = [
  ("p-sulfonatocalix[4]arene",1,0),
  ("p-Sulfonatothiacalix[4]arene",1,0),
  ("homoditopic doubly ethyl-bridged bis(p-sulfonatocalix[4]arene)",1,0),
  ("Calix[4]arene",1,0), ("Tetraundecyl calix[4]resorcinarene",1,0),
  ("Resorcin[4]arene",1,0), ("propyl sulfonated pillar[5]arene",0,1),
  ("sulfonated pillar[7]arene",0,1), ("carboxylate pillar[6]arene",0,1),
  ("Molecular Tweezer-Sodium disulfonate",0,1),
  ("Cucurbit[7]uril",0,1), ("beta-Cyclodextrin",0,1),
]
# 第 6 组:N 端规格(条件式)。(nterm, k_pos_list, expect_concordant, expect_required)
SELFTEST_NTERM = [
  ("H",  [4],  1, "H"),    # 非K9 + 游离 H- → concordant(254 型 K4me3)
  ("H",  [36], 1, "H"),    # 非K9 + 游离 H- → concordant(47 型 K36me3)
  ("Ac", [27], 0, "H"),    # 非K9 但 Ac- → 不符(374 型 K27me3,规格要游离)
  ("H",  [9],  0, "Ac"),   # K9 但 H- → 不符(1023 型 K9me3,规格要 Ac-)
  ("Ac", [9],  1, "Ac"),   # K9 + Ac- → concordant(D6SC K9me3 正例)
]

def run_selftest():
    fails = 0
    print("SMARTS_COMPILE_FAILURES=", [k for k, v in PAT.items() if v is None])
    print("--- SELFTEST 1: substructure counts ---")
    print("label\tkey\texpected\tactual\tverdict")
    for label, smi, key, exp in SELFTEST:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print("%s\t%s\t%d\tPARSE_FAIL\tFAIL" % (label, key, exp)); fails += 1; continue
        act = len(m.GetSubstructMatches(PAT[key], uniquify=True))
        ok = act == exp; fails += (not ok)
        print("%s\t%s\t%d\t%d\t%s" % (label, key, exp, act, "PASS" if ok else "FAIL"))
    print("--- SELFTEST 2: methyl count on epsilon-N ---")
    print("label\texpected\tn_hits\tmax_n_me\tall_counts\tverdict")
    for label, smi, exp in SELFTEST_NME:
        nh, mx, cs = n_methyl_on_eps_N(Chem.MolFromSmiles(smi))
        ok = (nh == 1 and mx == exp); fails += (not ok)
        print("%s\t%d\t%d\t%d\t%s\t%s" % (label, exp, nh, mx, cs, "PASS" if ok else "FAIL"))
    print("--- SELFTEST 3: sequence parser (bare/notation/cterm) ---")
    print("label\teb\tab\ten\tan\tec\tac\tptms\tverdict")
    for label, name, eb, en, ec in SELFTEST_SEQ:
        s = parse_seq(name)
        ok = (s["bare"] == eb and s["notation"] == en and s["cterm"] == ec); fails += (not ok)
        print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (label, eb, s["bare"], en,
              s["notation"], ec, s["cterm"], s["ptms"], "PASS" if ok else "FAIL"))
    print("--- SELFTEST 4: H3 mapping ---")
    print("label\texpect_pos\tfull_pos\tdrop_last_pos\texpect_K\tactual_K\tverdict")
    for label, bare, ep, ek in SELFTEST_H3:
        e = h3_evidence(bare)
        pos = e["full_pos"] if e["full_hit"] else e["drop_last_pos"]
        ok = (pos == ep and e["K_pos_in_H3"] == ek); fails += (not ok)
        print("%s\t%d\t%d\t%d\t%s\t%s\t%s" % (label, ep, e["full_pos"], e["drop_last_pos"],
              ek, e["K_pos_in_H3"], "PASS" if ok else "FAIL"))
    print("--- SELFTEST 5: calixarene-family name gate ---")
    print("name\texp_calix\tact_calix\texp_blocked\tact_blocked\tverdict")
    for name, ec, eb in SELFTEST_NAMEGATE:
        ic, ib = calix_family_gate(name)
        ok = (ic == ec and ib == eb); fails += (not ok)
        print("%s\t%d\t%d\t%d\t%d\t%s" % (name, ec, ic, eb, ib, "PASS" if ok else "FAIL"))
    print("--- SELFTEST 6: conditional N-terminal spec ---")
    print("nterm\tk_pos\texp_ok\tact_ok\texp_req\tact_req\tverdict")
    for nt, kp, eok, ereq in SELFTEST_NTERM:
        aok, areq = nterm_concordant(nt, kp)
        ok = (aok == eok and areq == ereq); fails += (not ok)
        print("%s\t%s\t%d\t%d\t%s\t%s\t%s" % (nt, kp, eok, aok, ereq, areq,
              "PASS" if ok else "FAIL"))
    print("SELFTEST_FAILURES=%d" % fails)
    return fails
