"""Outcome-blind molecular identity predicate for structure-only Stage A.

This module contains only uniform chemistry operations and synthetic self-tests.
It must be executed from the exact source text hashed by the Stage A driver.
No project data paths, row positions, identity IDs, folds, predictions, targets, or
metrics are embedded here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


RULESET_VERSION = "structure-identity-b1-stage-a-v1"

# Explicit metal set used only for a review flag. B, Si, Ge, As, Sb, Te and
# other metalloids are not silently treated as ordinary metals here.
METAL_ATOMIC_NUMBERS = frozenset(
    [
        3, 4, 11, 12, 13, 19, 20,
        *range(21, 32),
        37, 38, *range(39, 51),
        55, 56, *range(57, 85),
        87, 88, *range(89, 113),
    ]
)

# These are candidates only. Removal never becomes a strict identity claim.
# The allow-list is deliberately limited to simple monatomic ions. Polyatomic
# species and any second organic component remain unresolved.
SIMPLE_MONATOMIC_COUNTERION_KEYS = frozenset(
    {
        "[F-]",
        "[Cl-]",
        "[Br-]",
        "[I-]",
        "[Li+]",
        "[Na+]",
        "[K+]",
        "[Rb+]",
        "[Cs+]",
        "[Mg+2]",
        "[Ca+2]",
        "[Sr+2]",
        "[Ba+2]",
    }
)

_SALT_HINT_RE = re.compile(
    r"\b(?:hydrochloride|hydrobromide|hydroiodide|chloride|bromide|iodide|"
    r"sodium\s+salt|potassium\s+salt|lithium\s+salt|ammonium\s+salt|salt)\b",
    flags=re.IGNORECASE,
)
_FREE_FORM_HINT_RE = re.compile(r"\b(?:free\s+base|free\s+acid|neutral\s+form)\b", re.IGNORECASE)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_isomeric_smiles(mol: Chem.Mol) -> str:
    """Canonical full-graph SMILES retaining RDKit stereo/isotope/charge data."""

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
        allBondsExplicit=False,
        allHsExplicit=False,
    )


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _contains_carbon(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def _contains_metal(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms())


def _contains_dative_bond(mol: Chem.Mol) -> bool:
    return any(bond.GetBondType() == Chem.BondType.DATIVE for bond in mol.GetBonds())


def _stereo_flags(mol: Chem.Mol) -> Tuple[bool, bool, bool, bool]:
    atom_specified = any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
    bond_specified = any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())
    try:
        atom_centers = Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )
        atom_unassigned = any(label == "?" for _, label in atom_centers)
        perception_incomplete = False
    except RuntimeError:
        # Some very large or highly symmetric graphs exceed RDKit's CIP
        # digraph limit. The strict key still retains every stereo marker that
        # is explicitly present in the input. Potential unassigned-center
        # enumeration is then left unresolved and surfaced for review.
        atom_unassigned = False
        perception_incomplete = True
    return atom_specified, atom_unassigned, bond_specified, perception_incomplete


def _normalise_name_token(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split()).strip(" ;,")


def normalised_name_tokens(names: str) -> List[str]:
    tokens = []
    for raw in re.split(r"\s*\|\s*", names or ""):
        token = _normalise_name_token(raw)
        if token and token not in {"unknown", "unnamed", "n/a", "na", "none"}:
            tokens.append(token)
    return sorted(set(tokens))


def _counterion_candidate(mol: Chem.Mol, strict_key: str) -> Tuple[str, bool, str]:
    """Return a conservative *candidate* parent key, never a confirmed parent."""

    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if len(fragments) < 2:
        return strict_key, False, "single component"

    organic = [fragment for fragment in fragments if _contains_carbon(fragment)]
    if len(organic) != 1:
        return strict_key, False, "requires exactly one carbon-containing component"

    others = [fragment for fragment in fragments if fragment is not organic[0]]
    other_keys = [_canonical_isomeric_smiles(fragment) for fragment in others]
    if not other_keys or any(key not in SIMPLE_MONATOMIC_COUNTERION_KEYS for key in other_keys):
        return strict_key, False, "non-allow-listed or non-monatomic co-component"

    parent_charge = _formal_charge(organic[0])
    co_charge = sum(_formal_charge(fragment) for fragment in others)
    total_charge = _formal_charge(mol)
    if total_charge != 0 or parent_charge == 0 or parent_charge + co_charge != 0:
        return strict_key, False, "components do not form a neutral charge-balanced candidate salt"

    return (
        _canonical_isomeric_smiles(organic[0]),
        True,
        "one carbon-containing component plus charge-balancing allow-listed monatomic ions; candidate only",
    )


def _fragment_parent_operational(mol: Chem.Mol, strict_key: str) -> Tuple[str, bool, str]:
    try:
        parent = rdMolStandardize.FragmentParent(Chem.Mol(mol), skipStandardize=True)
        key = _canonical_isomeric_smiles(parent)
        return key, key != strict_key, "rdMolStandardize.FragmentParent(skipStandardize=True)"
    except Exception as exc:  # pragma: no cover - surfaced as row-level review state
        return strict_key, False, f"operation failed; strict key retained: {type(exc).__name__}: {exc}"


def _neutralized_operational(mol: Chem.Mol, strict_key: str) -> Tuple[str, bool, str]:
    try:
        neutral = rdMolStandardize.Uncharger().uncharge(Chem.Mol(mol))
        key = _canonical_isomeric_smiles(neutral)
        return key, key != strict_key, "rdMolStandardize.Uncharger().uncharge on the full structure"
    except Exception as exc:  # pragma: no cover - surfaced as row-level review state
        return strict_key, False, f"operation failed; strict key retained: {type(exc).__name__}: {exc}"


def _tautomer_operational(mol: Chem.Mol, strict_key: str) -> Tuple[str, bool, str]:
    try:
        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.Mol(mol))
        key = _canonical_isomeric_smiles(canonical)
        return key, key != strict_key, "rdMolStandardize.TautomerEnumerator().Canonicalize on the full structure"
    except Exception as exc:  # pragma: no cover - surfaced as row-level review state
        return strict_key, False, f"operation failed; strict key retained: {type(exc).__name__}: {exc}"


def _stereo_agnostic_operational(mol: Chem.Mol, strict_key: str) -> Tuple[str, bool, str]:
    try:
        copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(copy)
        key = _canonical_isomeric_smiles(copy)
        return key, key != strict_key, "Chem.RemoveStereochemistry on the full structure"
    except Exception as exc:  # pragma: no cover - surfaced as row-level review state
        return strict_key, False, f"operation failed; strict key retained: {type(exc).__name__}: {exc}"


def _invalid_key(smiles: str) -> str:
    return "UNPARSEABLE_SHA256:" + _sha256_text(smiles)


def classify_identity(identity_id: str, role: str, smiles: str, names: str) -> Dict[str, object]:
    """Apply every complete identity layer to one identity row."""

    role_norm = role.strip().lower()
    if role_norm not in {"host", "guest"}:
        raise ValueError(f"unsupported role {role!r}")

    identity_id = identity_id.strip()
    smiles = smiles.strip()
    names = names.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        strict_key = _invalid_key(smiles)
        reasons = ["STRUCTURE_PARSE_ERROR"]
        return {
            "role": role_norm,
            "identity_id": identity_id,
            "original_smiles": smiles,
            "original_names": names,
            "strict_exact_key": strict_key,
            "counterion_candidate_key": strict_key,
            "fragment_parent_operational_key": strict_key,
            "neutralized_operational_key": strict_key,
            "tautomer_operational_key": strict_key,
            "stereo_agnostic_operational_key": strict_key,
            "parse_ok": False,
            "fragment_count": 0,
            "carbon_fragment_count": 0,
            "multi_component_flag": False,
            "multiple_organic_components_flag": False,
            "contains_metal_flag": False,
            "coordination_bond_flag": False,
            "inorganic_guest_flag": False,
            "formal_charge_total": "",
            "has_formal_charge_flag": False,
            "specified_atom_stereo_flag": False,
            "unassigned_atom_stereo_flag": False,
            "specified_bond_stereo_flag": False,
            "stereo_perception_incomplete_flag": False,
            "isotope_flag": False,
            "name_salt_hint_flag": bool(_SALT_HINT_RE.search(names)),
            "name_free_form_hint_flag": bool(_FREE_FORM_HINT_RE.search(names)),
            "name_structure_conflict_flag": False,
            "counterion_candidate_applicable": False,
            "counterion_candidate_reason": "structure parse failed; strict fallback retained",
            "fragment_parent_changed": False,
            "fragment_parent_operation": "not applicable; structure parse failed",
            "neutralized_changed": False,
            "neutralized_operation": "not applicable; structure parse failed",
            "tautomer_changed": False,
            "tautomer_operation": "not applicable; structure parse failed",
            "stereo_removed": False,
            "stereo_operation": "not applicable; structure parse failed",
            "name_tokens_normalized": "|".join(normalised_name_tokens(names)),
            "identity_review_status": "UNCERTAIN",
            "uncertain_reasons": "|".join(reasons),
        }

    strict_key = _canonical_isomeric_smiles(mol)
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    fragment_count = len(fragments)
    carbon_fragment_count = sum(_contains_carbon(fragment) for fragment in fragments)
    total_charge = _formal_charge(mol)
    contains_metal = _contains_metal(mol)
    coordination_bond = _contains_dative_bond(mol)
    (
        specified_atom_stereo,
        unassigned_atom_stereo,
        specified_bond_stereo,
        stereo_perception_incomplete,
    ) = _stereo_flags(mol)
    isotope = any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())
    name_salt_hint = bool(_SALT_HINT_RE.search(names))
    name_free_form_hint = bool(_FREE_FORM_HINT_RE.search(names))

    counterion_key, counterion_applicable, counterion_reason = _counterion_candidate(mol, strict_key)
    fragment_parent_key, fragment_parent_changed, fragment_parent_operation = _fragment_parent_operational(
        mol, strict_key
    )
    neutralized_key, neutralized_changed, neutralized_operation = _neutralized_operational(mol, strict_key)
    tautomer_key, tautomer_changed, tautomer_operation = _tautomer_operational(mol, strict_key)
    stereo_key, stereo_removed, stereo_operation = _stereo_agnostic_operational(mol, strict_key)

    multi_component = fragment_count > 1
    multiple_organic = carbon_fragment_count > 1
    inorganic_guest = role_norm == "guest" and not _contains_carbon(mol)
    name_structure_conflict = (name_salt_hint and not multi_component and total_charge == 0) or (
        name_free_form_hint and multi_component
    )

    reasons: List[str] = []
    if multiple_organic:
        reasons.append("MULTIPLE_ORGANIC_COMPONENTS")
    elif multi_component and not counterion_applicable:
        reasons.append("MULTICOMPONENT_PARENT_UNRESOLVED")
    if counterion_applicable:
        reasons.append("COUNTERION_REMOVAL_CANDIDATE_NOT_AUTHOR_CONFIRMED")
    if contains_metal or coordination_bond:
        reasons.append("METAL_OR_COORDINATION_REVIEW")
    if inorganic_guest:
        reasons.append("INORGANIC_GUEST_REVIEW")
    if name_structure_conflict:
        reasons.append("NAME_STRUCTURE_CONFLICT_POTENTIAL")
    if unassigned_atom_stereo:
        reasons.append("UNASSIGNED_STEREOCENTER_POTENTIAL")
    if stereo_perception_incomplete:
        reasons.append("POTENTIAL_STEREO_ENUMERATION_INCOMPLETE")

    return {
        "role": role_norm,
        "identity_id": identity_id,
        "original_smiles": smiles,
        "original_names": names,
        "strict_exact_key": strict_key,
        "counterion_candidate_key": counterion_key,
        "fragment_parent_operational_key": fragment_parent_key,
        "neutralized_operational_key": neutralized_key,
        "tautomer_operational_key": tautomer_key,
        "stereo_agnostic_operational_key": stereo_key,
        "parse_ok": True,
        "fragment_count": fragment_count,
        "carbon_fragment_count": carbon_fragment_count,
        "multi_component_flag": multi_component,
        "multiple_organic_components_flag": multiple_organic,
        "contains_metal_flag": contains_metal,
        "coordination_bond_flag": coordination_bond,
        "inorganic_guest_flag": inorganic_guest,
        "formal_charge_total": total_charge,
        "has_formal_charge_flag": any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()),
        "specified_atom_stereo_flag": specified_atom_stereo,
        "unassigned_atom_stereo_flag": unassigned_atom_stereo,
        "specified_bond_stereo_flag": specified_bond_stereo,
        "stereo_perception_incomplete_flag": stereo_perception_incomplete,
        "isotope_flag": isotope,
        "name_salt_hint_flag": name_salt_hint,
        "name_free_form_hint_flag": name_free_form_hint,
        "name_structure_conflict_flag": name_structure_conflict,
        "counterion_candidate_applicable": counterion_applicable,
        "counterion_candidate_reason": counterion_reason,
        "fragment_parent_changed": fragment_parent_changed,
        "fragment_parent_operation": fragment_parent_operation,
        "neutralized_changed": neutralized_changed,
        "neutralized_operation": neutralized_operation,
        "tautomer_changed": tautomer_changed,
        "tautomer_operation": tautomer_operation,
        "stereo_removed": stereo_removed,
        "stereo_operation": stereo_operation,
        "name_tokens_normalized": "|".join(normalised_name_tokens(names)),
        "identity_review_status": "UNCERTAIN" if reasons else "STRICT_CLEAR",
        "uncertain_reasons": "|".join(reasons),
    }


def _test_case(name: str, actual: object, expected: object) -> Dict[str, object]:
    return {"name": name, "actual": actual, "expected": expected, "pass": actual == expected}


def run_synthetic_self_tests() -> List[Dict[str, object]]:
    """Return explicit actual/expected results; caller must stop on any failure."""

    tests: List[Dict[str, object]] = []

    ethanol_a = classify_identity("synthetic-a", "guest", "C(O)C", "ethanol")
    ethanol_b = classify_identity("synthetic-b", "guest", "CCO", "ethanol")
    tests.append(
        _test_case(
            "canonical_atom_order_invariance",
            ethanol_a["strict_exact_key"] == ethanol_b["strict_exact_key"],
            True,
        )
    )

    components_a = classify_identity("synthetic-a", "guest", "[Na+].[Cl-]", "sodium chloride")
    components_b = classify_identity("synthetic-b", "guest", "[Cl-].[Na+]", "sodium chloride")
    tests.append(
        _test_case(
            "canonical_component_order_invariance",
            components_a["strict_exact_key"] == components_b["strict_exact_key"],
            True,
        )
    )

    stereo_a = classify_identity("synthetic-a", "guest", "F[C@H](Cl)Br", "enantiomer a")
    stereo_b = classify_identity("synthetic-b", "guest", "F[C@@H](Cl)Br", "enantiomer b")
    tests.append(
        _test_case(
            "strict_layer_retains_specified_stereo",
            stereo_a["strict_exact_key"] != stereo_b["strict_exact_key"],
            True,
        )
    )
    tests.append(
        _test_case(
            "stereo_agnostic_layer_removes_specified_stereo",
            stereo_a["stereo_agnostic_operational_key"] == stereo_b["stereo_agnostic_operational_key"],
            True,
        )
    )

    isotope = classify_identity("synthetic", "guest", "[13CH4]", "carbon-13 methane")
    ordinary = classify_identity("synthetic", "guest", "C", "methane")
    tests.append(
        _test_case(
            "strict_layer_retains_isotope",
            isotope["isotope_flag"] and isotope["strict_exact_key"] != ordinary["strict_exact_key"],
            True,
        )
    )

    charged = classify_identity("synthetic", "guest", "CC[NH3+]", "ethylammonium")
    tests.append(_test_case("strict_layer_retains_formal_charge", charged["formal_charge_total"], 1))
    tests.append(_test_case("neutralization_is_separate_and_changes_key", charged["neutralized_changed"], True))

    salt = classify_identity("synthetic", "guest", "CC(=O)[O-].[Na+]", "sodium acetate")
    tests.append(_test_case("counterion_candidate_requires_charge_balance", salt["counterion_candidate_applicable"], True))
    tests.append(_test_case("counterion_candidate_retains_parent_charge", salt["counterion_candidate_key"], "CC(=O)[O-]"))

    mixture = classify_identity("synthetic", "guest", "CC.CCC", "two organics")
    tests.append(_test_case("multiple_organic_components_not_counterion_candidate", mixture["counterion_candidate_applicable"], False))
    tests.append(_test_case("multiple_organic_components_flagged_uncertain", "MULTIPLE_ORGANIC_COMPONENTS" in mixture["uncertain_reasons"], True))

    metal = classify_identity("synthetic", "guest", "[Fe+2]", "iron ii")
    tests.append(_test_case("metal_flag", metal["contains_metal_flag"], True))
    tests.append(_test_case("inorganic_guest_flag", metal["inorganic_guest_flag"], True))

    parse_error = classify_identity("synthetic", "guest", "not_a_smiles", "invalid")
    tests.append(_test_case("parse_failure_retains_complete_fallback_keys", all(parse_error[layer] == parse_error["strict_exact_key"] for layer in ("counterion_candidate_key", "fragment_parent_operational_key", "neutralized_operational_key", "tautomer_operational_key", "stereo_agnostic_operational_key")), True))

    named_salt_missing_component = classify_identity("synthetic", "guest", "CCN", "ethylamine hydrochloride")
    tests.append(_test_case("name_structure_conflict_potential_flag", named_salt_missing_component["name_structure_conflict_flag"], True))

    return tests
