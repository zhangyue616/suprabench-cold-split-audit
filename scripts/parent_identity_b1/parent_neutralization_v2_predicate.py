"""Outcome-blind candidate-parent neutralization predicate, supplement v2.

This version fixes one semantic defect in supplement v1: a single organic,
non-metal, non-coordination component is already a candidate parent and must
receive the same Uncharger operation as a parent extracted from an admitted
disconnected salt. No project IDs, rows, names, folds, targets, or outcomes are
used by this predicate.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


RULESET_VERSION = "structure-identity-b1-stage-a-supplement-v2"

METAL_ATOMIC_NUMBERS = frozenset(
    [
        3, 4, 11, 12, 13, 19, 20,
        *range(21, 32),
        37, 38, *range(39, 51),
        55, 56, *range(57, 85),
        87, 88, *range(89, 113),
    ]
)

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


def _canonical(mol: Chem.Mol) -> str:
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


def _contains_coordination_bond(mol: Chem.Mol) -> bool:
    return any(bond.GetBondType() == Chem.BondType.DATIVE for bond in mol.GetBonds())


def _specified_stereo_signature(mol: Chem.Mol) -> Tuple[Tuple[int, int], ...]:
    atom_signature = tuple(
        sorted(
            (atom.GetIdx(), int(atom.GetChiralTag()))
            for atom in mol.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        )
    )
    bond_signature = tuple(
        sorted(
            (bond.GetIdx() + 1000000, int(bond.GetStereo()))
            for bond in mol.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        )
    )
    return atom_signature + bond_signature


def _isotope_signature(mol: Chem.Mol) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((atom.GetIdx(), atom.GetIsotope()) for atom in mol.GetAtoms() if atom.GetIsotope()))


def _unassigned_stereo_knowledge(mol: Chem.Mol) -> Tuple[bool, bool]:
    """Return (potential_unassigned, enumeration_incomplete) as knowledge flags."""

    try:
        centers = Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )
        return any(label == "?" for _, label in centers), False
    except RuntimeError:
        return False, True


def _fallback_key(smiles: str) -> str:
    return "UNPARSEABLE_SHA256:" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def _rejected(
    strict_key: str,
    reason: str,
    parent: Chem.Mol | None,
    counterions: List[Chem.Mol],
) -> Dict[str, object]:
    potential_unassigned = False
    stereo_incomplete = False
    if parent is not None:
        potential_unassigned, stereo_incomplete = _unassigned_stereo_knowledge(parent)
    uncertain = [reason]
    if potential_unassigned:
        uncertain.append("UNASSIGNED_STEREOCENTER_POTENTIAL")
    if stereo_incomplete:
        uncertain.append("POTENTIAL_STEREO_ENUMERATION_INCOMPLETE")
    return {
        "strict_exact_key": strict_key,
        "candidate_parent_admitted": False,
        "candidate_parent_source": "REJECTED",
        "candidate_parent_key": strict_key,
        "parent_neutralized_operational_key": strict_key,
        "candidate_parent_reject_reason": reason,
        "counterion_keys": "|".join(sorted(_canonical(fragment) for fragment in counterions)),
        "counterion_contains_metal_flag": any(_contains_metal(fragment) for fragment in counterions),
        "parent_contains_metal_flag": bool(parent is not None and _contains_metal(parent)),
        "parent_coordination_bond_flag": bool(parent is not None and _contains_coordination_bond(parent)),
        "parent_initial_formal_charge": "" if parent is None else _formal_charge(parent),
        "parent_final_formal_charge": "",
        "parent_neutralized_changed": False,
        "permanent_charge_retained_flag": False,
        "specified_stereo_present_flag": bool(parent is not None and _specified_stereo_signature(parent)),
        "specified_stereo_preserved_flag": bool(parent is not None and _specified_stereo_signature(parent)),
        "isotope_present_flag": bool(parent is not None and _isotope_signature(parent)),
        "isotope_preserved_flag": bool(parent is not None and _isotope_signature(parent)),
        "normalization_preservation_unresolved": False,
        "unassigned_stereo_potential_flag": potential_unassigned,
        "stereo_perception_incomplete_flag": stereo_incomplete,
        "uncertain_reasons": "|".join(uncertain),
        "operation_note": f"not admitted ({reason}); strict full-structure key retained",
    }


def map_parent_neutralization_v2(role: str, smiles: str, names: str = "") -> Dict[str, object]:
    """Apply the uniform candidate-parent then Uncharger representation.

    A single organic non-metal and non-coordination component is treated as an
    already isolated candidate parent. A multi-component record uses the same
    conservative disconnected-counterion admission rule as supplement v1.
    Knowledge-state flags are reported independently and never control whether
    a different operational layer should drop the identity.
    """

    role_norm = role.strip().lower()
    if role_norm not in {"host", "guest"}:
        raise ValueError(f"unsupported role {role!r}")
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        strict = _fallback_key(smiles)
        result = _rejected(strict, "STRUCTURE_PARSE_ERROR", None, [])
        result["operation_note"] = "not applied; structure parse failed; strict fallback retained"
        return result

    strict = _canonical(mol)
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    carbon_fragments = [fragment for fragment in fragments if _contains_carbon(fragment)]
    parent: Chem.Mol | None = None
    counterions: List[Chem.Mol] = []
    source = ""

    if len(fragments) == 1:
        parent = fragments[0]
        if not _contains_carbon(parent):
            return _rejected(strict, "NO_CARBON_PARENT_INORGANIC_OR_UNRESOLVED", parent, [])
        if _contains_metal(parent) or _contains_coordination_bond(parent):
            return _rejected(strict, "PARENT_METAL_OR_COORDINATION", parent, [])
        source = "SINGLE_ORGANIC_EXISTING_PARENT"
    else:
        if len(carbon_fragments) == 0:
            return _rejected(strict, "NO_CARBON_PARENT_INORGANIC_OR_UNRESOLVED", None, fragments)
        if len(carbon_fragments) > 1:
            return _rejected(strict, "MULTIPLE_ORGANIC_COMPONENTS", None, fragments)
        parent = carbon_fragments[0]
        counterions = [fragment for fragment in fragments if fragment is not parent]
        if _contains_metal(parent) or _contains_coordination_bond(parent):
            return _rejected(strict, "PARENT_METAL_OR_COORDINATION", parent, counterions)
        counterion_keys = [_canonical(fragment) for fragment in counterions]
        if not counterion_keys or any(key not in SIMPLE_MONATOMIC_COUNTERION_KEYS for key in counterion_keys):
            return _rejected(strict, "NON_ALLOWLISTED_OR_NON_MONATOMIC_COCOMPONENT", parent, counterions)
        parent_charge = _formal_charge(parent)
        counterion_charge = sum(_formal_charge(fragment) for fragment in counterions)
        if _formal_charge(mol) != 0 or parent_charge == 0 or parent_charge + counterion_charge != 0:
            return _rejected(strict, "NOT_NEUTRAL_CHARGE_BALANCED_SALT", parent, counterions)
        source = "DISCONNECTED_COUNTERION_CANDIDATE"

    assert parent is not None
    parent_key = _canonical(parent)
    before_stereo = _specified_stereo_signature(parent)
    before_isotope = _isotope_signature(parent)
    potential_unassigned, stereo_incomplete = _unassigned_stereo_knowledge(parent)
    initial_charge = _formal_charge(parent)
    neutralized = rdMolStandardize.Uncharger().uncharge(Chem.Mol(parent))
    normalized_key = _canonical(neutralized)
    final_charge = _formal_charge(neutralized)
    after_stereo = _specified_stereo_signature(neutralized)
    after_isotope = _isotope_signature(neutralized)
    stereo_preserved = before_stereo == after_stereo
    isotope_preserved = before_isotope == after_isotope
    if not stereo_preserved or not isotope_preserved:
        uncertain = ["NORMALIZATION_PRESERVATION_UNRESOLVED"]
        if source == "DISCONNECTED_COUNTERION_CANDIDATE":
            uncertain.append("COUNTERION_REMOVAL_CANDIDATE_NOT_AUTHOR_CONFIRMED")
        if potential_unassigned:
            uncertain.append("UNASSIGNED_STEREOCENTER_POTENTIAL")
        if stereo_incomplete:
            uncertain.append("POTENTIAL_STEREO_ENUMERATION_INCOMPLETE")
        return {
            "strict_exact_key": strict,
            "candidate_parent_admitted": False,
            "candidate_parent_source": source,
            "candidate_parent_key": parent_key,
            "parent_neutralized_operational_key": strict,
            "candidate_parent_reject_reason": "NORMALIZATION_PRESERVATION_UNRESOLVED",
            "counterion_keys": "|".join(sorted(_canonical(fragment) for fragment in counterions)),
            "counterion_contains_metal_flag": any(_contains_metal(fragment) for fragment in counterions),
            "parent_contains_metal_flag": False,
            "parent_coordination_bond_flag": False,
            "parent_initial_formal_charge": initial_charge,
            "parent_final_formal_charge": final_charge,
            "parent_neutralized_changed": False,
            "permanent_charge_retained_flag": final_charge != 0,
            "specified_stereo_present_flag": bool(before_stereo),
            "specified_stereo_preserved_flag": stereo_preserved,
            "isotope_present_flag": bool(before_isotope),
            "isotope_preserved_flag": isotope_preserved,
            "normalization_preservation_unresolved": True,
            "unassigned_stereo_potential_flag": potential_unassigned,
            "stereo_perception_incomplete_flag": stereo_incomplete,
            "uncertain_reasons": "|".join(uncertain),
            "operation_note": "normalization preservation check unresolved; fail closed to complete original strict key; no stereo or isotope destruction inferred",
        }

    uncertain: List[str] = []
    if source == "DISCONNECTED_COUNTERION_CANDIDATE":
        uncertain.append("COUNTERION_REMOVAL_CANDIDATE_NOT_AUTHOR_CONFIRMED")
    if potential_unassigned:
        uncertain.append("UNASSIGNED_STEREOCENTER_POTENTIAL")
    if stereo_incomplete:
        uncertain.append("POTENTIAL_STEREO_ENUMERATION_INCOMPLETE")

    return {
        "strict_exact_key": strict,
        "candidate_parent_admitted": True,
        "candidate_parent_source": source,
        "candidate_parent_key": parent_key,
        "parent_neutralized_operational_key": normalized_key,
        "candidate_parent_reject_reason": "",
        "counterion_keys": "|".join(sorted(_canonical(fragment) for fragment in counterions)),
        "counterion_contains_metal_flag": any(_contains_metal(fragment) for fragment in counterions),
        "parent_contains_metal_flag": False,
        "parent_coordination_bond_flag": False,
        "parent_initial_formal_charge": initial_charge,
        "parent_final_formal_charge": final_charge,
        "parent_neutralized_changed": normalized_key != parent_key,
        "permanent_charge_retained_flag": final_charge != 0,
        "specified_stereo_present_flag": bool(before_stereo),
        "specified_stereo_preserved_flag": before_stereo == after_stereo,
        "isotope_present_flag": bool(before_isotope),
        "isotope_preserved_flag": isotope_preserved,
        "normalization_preservation_unresolved": False,
        "unassigned_stereo_potential_flag": potential_unassigned,
        "stereo_perception_incomplete_flag": stereo_incomplete,
        "uncertain_reasons": "|".join(uncertain),
        "operation_note": "use single organic component as existing parent or remove admitted disconnected monatomic counterion candidate(s), then rdMolStandardize.Uncharger().uncharge(candidate_parent)",
    }


def _test(name: str, actual: object, expected: object) -> Dict[str, object]:
    return {"name": name, "actual": actual, "expected": expected, "pass": actual == expected}


def run_synthetic_self_tests() -> List[Dict[str, object]]:
    tests: List[Dict[str, object]] = []

    amine_neutral = map_parent_neutralization_v2("guest", "CCN", "ethylamine")
    amine_protonated = map_parent_neutralization_v2("guest", "CC[NH3+]", "ethylammonium")
    amine_salt = map_parent_neutralization_v2("guest", "CC[NH3+].[Cl-]", "ethylamine hydrochloride")
    tests.append(
        _test(
            "amine_neutral_protonated_salt_exact_key_triplet",
            {
                "CCN": amine_neutral["parent_neutralized_operational_key"],
                "CC[NH3+]": amine_protonated["parent_neutralized_operational_key"],
                "CC[NH3+].[Cl-]": amine_salt["parent_neutralized_operational_key"],
                "sources": [
                    amine_neutral["candidate_parent_source"],
                    amine_protonated["candidate_parent_source"],
                    amine_salt["candidate_parent_source"],
                ],
            },
            {
                "CCN": "CCN",
                "CC[NH3+]": "CCN",
                "CC[NH3+].[Cl-]": "CCN",
                "sources": [
                    "SINGLE_ORGANIC_EXISTING_PARENT",
                    "SINGLE_ORGANIC_EXISTING_PARENT",
                    "DISCONNECTED_COUNTERION_CANDIDATE",
                ],
            },
        )
    )

    acid = map_parent_neutralization_v2("guest", "O=C(O)c1ccccc1", "benzoic acid")
    carboxylate = map_parent_neutralization_v2("guest", "O=C([O-])c1ccccc1", "benzoate")
    sodium_salt = map_parent_neutralization_v2("guest", "[Na+].O=C([O-])c1ccccc1", "sodium benzoate")
    tests.append(
        _test(
            "carboxylic_acid_carboxylate_sodium_salt_exact_key_triplet",
            {
                "acid": acid["parent_neutralized_operational_key"],
                "carboxylate": carboxylate["parent_neutralized_operational_key"],
                "sodium_salt": sodium_salt["parent_neutralized_operational_key"],
            },
            {"acid": "O=C(O)c1ccccc1", "carboxylate": "O=C(O)c1ccccc1", "sodium_salt": "O=C(O)c1ccccc1"},
        )
    )

    quat = map_parent_neutralization_v2("guest", "C[N+](C)(C)C", "tetramethylammonium")
    quat_salt = map_parent_neutralization_v2("guest", "C[N+](C)(C)C.[Cl-]", "tetramethylammonium chloride")
    tests.append(
        _test(
            "permanent_quaternary_parent_and_salt_exact_keys",
            {
                "parent_key": quat["parent_neutralized_operational_key"],
                "salt_key": quat_salt["parent_neutralized_operational_key"],
                "parent_final_charge": quat["parent_final_formal_charge"],
                "salt_final_charge": quat_salt["parent_final_formal_charge"],
                "parent_permanent_flag": quat["permanent_charge_retained_flag"],
                "salt_permanent_flag": quat_salt["permanent_charge_retained_flag"],
            },
            {
                "parent_key": "C[N+](C)(C)C",
                "salt_key": "C[N+](C)(C)C",
                "parent_final_charge": 1,
                "salt_final_charge": 1,
                "parent_permanent_flag": True,
                "salt_permanent_flag": True,
            },
        )
    )

    inorganic = map_parent_neutralization_v2("guest", "[Na+].[Cl-]", "sodium chloride")
    tests.append(
        _test(
            "inorganic_pair_rejected_with_exact_retained_key",
            {
                "admitted": inorganic["candidate_parent_admitted"],
                "reason": inorganic["candidate_parent_reject_reason"],
                "strict": inorganic["strict_exact_key"],
                "operational": inorganic["parent_neutralized_operational_key"],
            },
            {
                "admitted": False,
                "reason": "NO_CARBON_PARENT_INORGANIC_OR_UNRESOLVED",
                "strict": "[Cl-].[Na+]",
                "operational": "[Cl-].[Na+]",
            },
        )
    )

    metal_parent = map_parent_neutralization_v2("guest", "C[Cu+].[Cl-]", "organocopper chloride")
    tests.append(
        _test(
            "metal_parent_rejected_with_exact_retained_key",
            {
                "admitted": metal_parent["candidate_parent_admitted"],
                "reason": metal_parent["candidate_parent_reject_reason"],
                "strict": metal_parent["strict_exact_key"],
                "operational": metal_parent["parent_neutralized_operational_key"],
            },
            {
                "admitted": False,
                "reason": "PARENT_METAL_OR_COORDINATION",
                "strict": "[CH3][Cu+].[Cl-]",
                "operational": "[CH3][Cu+].[Cl-]",
            },
        )
    )

    multi_organic = map_parent_neutralization_v2("guest", "C[NH3+].CC(=O)[O-]", "organic ion pair")
    tests.append(
        _test(
            "multiple_organic_components_rejected_with_exact_retained_key",
            {
                "admitted": multi_organic["candidate_parent_admitted"],
                "reason": multi_organic["candidate_parent_reject_reason"],
                "strict": multi_organic["strict_exact_key"],
                "operational": multi_organic["parent_neutralized_operational_key"],
            },
            {
                "admitted": False,
                "reason": "MULTIPLE_ORGANIC_COMPONENTS",
                "strict": "CC(=O)[O-].C[NH3+]",
                "operational": "CC(=O)[O-].C[NH3+]",
            },
        )
    )

    stereo_a = map_parent_neutralization_v2("guest", "C[C@H]([NH3+])C(=O)O", "stereo amine a")
    stereo_a_salt = map_parent_neutralization_v2("guest", "[Cl-].C[C@H]([NH3+])C(=O)O", "stereo amine a hydrochloride")
    stereo_b = map_parent_neutralization_v2("guest", "C[C@@H]([NH3+])C(=O)O", "stereo amine b")
    tests.append(
        _test(
            "specified_stereo_exact_keys_preserved",
            {
                "a_parent": stereo_a["parent_neutralized_operational_key"],
                "a_salt": stereo_a_salt["parent_neutralized_operational_key"],
                "b_parent": stereo_b["parent_neutralized_operational_key"],
                "a_preserved": stereo_a["specified_stereo_preserved_flag"],
                "a_salt_preserved": stereo_a_salt["specified_stereo_preserved_flag"],
            },
            {
                "a_parent": "C[C@H](N)C(=O)O",
                "a_salt": "C[C@H](N)C(=O)O",
                "b_parent": "C[C@@H](N)C(=O)O",
                "a_preserved": True,
                "a_salt_preserved": True,
            },
        )
    )

    isotope_parent = map_parent_neutralization_v2("guest", "[13CH3][NH3+]", "carbon-13 methylammonium")
    isotope_salt = map_parent_neutralization_v2("guest", "[13CH3][NH3+].[Cl-]", "carbon-13 methylamine hydrochloride")
    tests.append(
        _test(
            "isotope_exact_keys_preserved",
            {
                "parent": isotope_parent["parent_neutralized_operational_key"],
                "salt": isotope_salt["parent_neutralized_operational_key"],
                "parent_preserved": isotope_parent["isotope_preserved_flag"],
                "salt_preserved": isotope_salt["isotope_preserved_flag"],
            },
            {"parent": "[13CH3]N", "salt": "[13CH3]N", "parent_preserved": True, "salt_preserved": True},
        )
    )

    unspecified = map_parent_neutralization_v2("guest", "CC(O)C(=O)O", "unspecified hydroxy acid")
    tests.append(
        _test(
            "uncertain_stereo_flag_is_independent_of_mapping",
            {
                "key": unspecified["parent_neutralized_operational_key"],
                "admitted": unspecified["candidate_parent_admitted"],
                "unassigned_flag": unspecified["unassigned_stereo_potential_flag"],
                "uncertain_reasons": unspecified["uncertain_reasons"],
            },
            {
                "key": "CC(O)C(=O)O",
                "admitted": True,
                "unassigned_flag": True,
                "uncertain_reasons": "UNASSIGNED_STEREOCENTER_POTENTIAL",
            },
        )
    )

    representation_change = map_parent_neutralization_v2("guest", "F/C=C/F", "synthetic E alkene")
    tests.append(
        _test(
            "stereo_representation_check_mismatch_fails_closed",
            {
                "admitted": representation_change["candidate_parent_admitted"],
                "strict": representation_change["strict_exact_key"],
                "operational": representation_change["parent_neutralized_operational_key"],
                "reason": representation_change["candidate_parent_reject_reason"],
                "normalization_preservation_unresolved": representation_change[
                    "normalization_preservation_unresolved"
                ],
                "uncertain_reasons": representation_change["uncertain_reasons"],
            },
            {
                "admitted": False,
                "strict": "F/C=C/F",
                "operational": "F/C=C/F",
                "reason": "NORMALIZATION_PRESERVATION_UNRESOLVED",
                "normalization_preservation_unresolved": True,
                "uncertain_reasons": "NORMALIZATION_PRESERVATION_UNRESOLVED",
            },
        )
    )

    return tests
