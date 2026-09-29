# Parent and identity construction

The public scripts retain the outcome-blind identity construction used before fixed-prediction scoring.

Stage A (`identity_predicate.py`) builds five v1 representations from structure strings: strict exact canonical isomeric full-graph SMILES; a charge-balanced monatomic-counterion candidate flag; RDKit `FragmentParent(..., skipStandardize=True)`; full-structure neutralization with `Uncharger`; and full-structure stereo removal. Parse failures use a SHA-256 fallback of the input string. The counterion layer is a candidate flag and is not treated as a chemical-equivalence judgment. Tautomer enumeration was paused at the frozen boundary and is not one of the five v1 layers.

The neutralization predicates and `build_membership_masks.py` implement Stage B membership construction. Frozen decisions, mappings, masks, coverage tables, and Stage-C fixed-prediction results are under `results/parent_identity_b1/`. The public package excludes private review prompts and execution logs. The scripts and frozen outputs support inspection and replay of the scientific rules; they do not reconstruct the historical review process.
