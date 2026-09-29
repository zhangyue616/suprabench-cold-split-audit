"""Structure-only candidate-parent then neutralization predicate.

The predicate is uniform and outcome-blind. It never treats its operational
mapping as proof that two records are the same experimental chemical species.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


RULESET_VERSION = "structure-identity-b1-stage-a-supplement-v1"

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


def _fallback_key(smiles: str) -> str:
    return "UNPARSEABLE_SHA256:" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def map_candidate_parent_then_uncharge(role: str, smiles: str, names: str = "") -> Dict[str, object]:
    """Map one identity into the composed operational layer.

    Admission is structural only: one carbon-containing disconnected component,
    one or more allow-listed monatomic ionic co-components, exact charge balance,
    and no metal or dative bond in the carbon-containing parent. Ineligible rows
    retain their strict key in the composed layer.
    """

    role_norm = role.strip().lower()
    if role_norm not in {"host", "guest"}:
        raise ValueError(f"unsupported role {role!r}")
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        strict = _fallback_key(smiles)
        return {
            "strict_exact_key_recomputed": strict,
            "candidate_parent_admitted": False,
            "candidate_parent_key": strict,
            "parent_neutralized_operational_key": strict,
            "candidate_parent_reject_reason": "STRUCTURE_PARSE_ERROR",
            "counterion_keys": "",
            "counterion_contains_metal_flag": False,
            "parent_contains_metal_flag": False,
            "parent_coordination_bond_flag": False,
            "parent_initial_formal_charge": "",
            "parent_final_formal_charge": "",
            "parent_neutralized_changed": False,
            "permanent_charge_retained_flag": False,
            "specified_stereo_present_flag": False,
            "specified_stereo_preserved_flag": False,
            "isotope_present_flag": False,
            "isotope_preserved_flag": False,
            "operation_note": "not applied; structure parse failed; strict fallback retained",
        }

    strict = _canonical(mol)
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    carbon_fragments = [fragment for fragment in fragments if _contains_carbon(fragment)]

    reject_reason = ""
    parent = None
    counterions: List[Chem.Mol] = []
    if len(fragments) < 2:
        reject_reason = "NO_DISCONNECTED_COUNTERION"
    elif len(carbon_fragments) == 0:
        reject_reason = "NO_CARBON_PARENT_INORGANIC_OR_UNRESOLVED"
    elif len(carbon_fragments) > 1:
        reject_reason = "MULTIPLE_ORGANIC_COMPONENTS"
    else:
        parent = carbon_fragments[0]
        counterions = [fragment for fragment in fragments if fragment is not parent]
        if _contains_metal(parent) or _contains_coordination_bond(parent):
            reject_reason = "PARENT_METAL_OR_COORDINATION"
        else:
            counterion_keys = [_canonical(fragment) for fragment in counterions]
            if not counterion_keys or any(key not in SIMPLE_MONATOMIC_COUNTERION_KEYS for key in counterion_keys):
                reject_reason = "NON_ALLOWLISTED_OR_NON_MONATOMIC_COCOMPONENT"
            else:
                parent_charge = _formal_charge(parent)
                counterion_charge = sum(_formal_charge(fragment) for fragment in counterions)
                if _formal_charge(mol) != 0 or parent_charge == 0 or parent_charge + counterion_charge != 0:
                    reject_reason = "NOT_NEUTRAL_CHARGE_BALANCED_SALT"

    if reject_reason:
        counterion_keys = [_canonical(fragment) for fragment in counterions]
        return {
            "strict_exact_key_recomputed": strict,
            "candidate_parent_admitted": False,
            "candidate_parent_key": strict,
            "parent_neutralized_operational_key": strict,
            "candidate_parent_reject_reason": reject_reason,
            "counterion_keys": "|".join(sorted(counterion_keys)),
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
            "operation_note": f"not admitted ({reject_reason}); strict full-structure key retained",
        }

    assert parent is not None
    parent_key = _canonical(parent)
    before_stereo = _specified_stereo_signature(parent)
    before_isotope = _isotope_signature(parent)
    parent_initial_charge = _formal_charge(parent)
    neutralized = rdMolStandardize.Uncharger().uncharge(Chem.Mol(parent))
    neutralized_key = _canonical(neutralized)
    parent_final_charge = _formal_charge(neutralized)
    after_stereo = _specified_stereo_signature(neutralized)
    after_isotope = _isotope_signature(neutralized)

    if before_stereo != after_stereo:
        raise RuntimeError("Uncharger changed specified stereo signature")
    if before_isotope != after_isotope:
        raise RuntimeError("Uncharger changed isotope signature")

    return {
        "strict_exact_key_recomputed": strict,
        "candidate_parent_admitted": True,
        "candidate_parent_key": parent_key,
        "parent_neutralized_operational_key": neutralized_key,
        "candidate_parent_reject_reason": "",
        "counterion_keys": "|".join(sorted(_canonical(fragment) for fragment in counterions)),
        "counterion_contains_metal_flag": any(_contains_metal(fragment) for fragment in counterions),
        "parent_contains_metal_flag": False,
        "parent_coordination_bond_flag": False,
        "parent_initial_formal_charge": parent_initial_charge,
        "parent_final_formal_charge": parent_final_charge,
        "parent_neutralized_changed": neutralized_key != parent_key,
        "permanent_charge_retained_flag": parent_final_charge != 0,
        "specified_stereo_present_flag": bool(before_stereo),
        "specified_stereo_preserved_flag": before_stereo == after_stereo,
        "isotope_present_flag": bool(before_isotope),
        "isotope_preserved_flag": before_isotope == after_isotope,
        "operation_note": "remove structurally admitted disconnected monatomic counterion candidate(s), then rdMolStandardize.Uncharger().uncharge(candidate_parent)",
    }


def _test(name: str, actual: object, expected: object) -> Dict[str, object]:
    return {"name": name, "actual": actual, "expected": expected, "pass": actual == expected}


def run_synthetic_self_tests() -> List[Dict[str, object]]:
    tests: List[Dict[str, object]] = []

    amine_salt = map_candidate_parent_then_uncharge("guest", "CC[NH3+].[Cl-]", "ethylamine hydrochloride")
    free_amine = map_candidate_parent_then_uncharge("guest", "CCN", "ethylamine")
    tests.append(_test("amine_salt_admitted", amine_salt["candidate_parent_admitted"], True))
    tests.append(
        _test(
            "amine_salt_matches_free_amine_after_parent_uncharge",
            amine_salt["parent_neutralized_operational_key"],
            free_amine["strict_exact_key_recomputed"],
        )
    )

    sulfonate = map_candidate_parent_then_uncharge("guest", "CS(=O)(=O)[O-].[Na+]", "sodium methanesulfonate")
    sulfonic_acid = map_candidate_parent_then_uncharge("guest", "CS(=O)(=O)O", "methanesulfonic acid")
    tests.append(_test("sulfonate_salt_admitted", sulfonate["candidate_parent_admitted"], True))
    tests.append(
        _test(
            "sulfonate_matches_acid_after_parent_uncharge",
            sulfonate["parent_neutralized_operational_key"],
            sulfonic_acid["strict_exact_key_recomputed"],
        )
    )
    tests.append(_test("alkali_counterion_distinguished_from_parent_metal", sulfonate["parent_contains_metal_flag"], False))
    tests.append(_test("alkali_counterion_is_recorded_as_metal_counterion", sulfonate["counterion_contains_metal_flag"], True))

    quaternary = map_candidate_parent_then_uncharge("guest", "C[N+](C)(C)C.[Cl-]", "tetramethylammonium chloride")
    tests.append(_test("quaternary_ammonium_admitted_as_candidate_salt", quaternary["candidate_parent_admitted"], True))
    tests.append(_test("permanent_quaternary_charge_retained", quaternary["parent_final_formal_charge"], 1))
    tests.append(_test("permanent_charge_flag", quaternary["permanent_charge_retained_flag"], True))

    stereo_a = map_candidate_parent_then_uncharge("guest", "[Cl-].C[C@H]([NH3+])C(=O)O", "l-alanine hydrochloride")
    stereo_b = map_candidate_parent_then_uncharge("guest", "[Cl-].C[C@@H]([NH3+])C(=O)O", "d-alanine hydrochloride")
    tests.append(_test("specified_stereo_preserved", stereo_a["specified_stereo_preserved_flag"], True))
    tests.append(
        _test(
            "specified_stereo_differences_remain_distinct",
            stereo_a["parent_neutralized_operational_key"] != stereo_b["parent_neutralized_operational_key"],
            True,
        )
    )

    isotope = map_candidate_parent_then_uncharge("guest", "[13CH3][NH3+].[Cl-]", "carbon-13 methylamine hydrochloride")
    tests.append(_test("isotope_preserved", isotope["isotope_preserved_flag"], True))
    tests.append(_test("isotope_present", isotope["isotope_present_flag"], True))

    multi_organic = map_candidate_parent_then_uncharge("guest", "C[NH3+].CC(=O)[O-]", "organic ion pair")
    tests.append(_test("multiple_organic_components_rejected", multi_organic["candidate_parent_admitted"], False))
    tests.append(_test("multiple_organic_reason", multi_organic["candidate_parent_reject_reason"], "MULTIPLE_ORGANIC_COMPONENTS"))

    metal_parent = map_candidate_parent_then_uncharge("guest", "C[Cu+].[Cl-]", "organocopper chloride")
    tests.append(_test("metal_containing_parent_rejected", metal_parent["candidate_parent_admitted"], False))
    tests.append(_test("metal_parent_reason", metal_parent["candidate_parent_reject_reason"], "PARENT_METAL_OR_COORDINATION"))

    inorganic = map_candidate_parent_then_uncharge("guest", "[Na+].[Cl-]", "sodium chloride")
    tests.append(_test("inorganic_guest_not_forced_into_parent", inorganic["candidate_parent_admitted"], False))
    tests.append(_test("inorganic_guest_reason", inorganic["candidate_parent_reject_reason"], "NO_CARBON_PARENT_INORGANIC_OR_UNRESOLVED"))

    return tests

