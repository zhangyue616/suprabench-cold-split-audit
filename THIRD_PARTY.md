# Third-party notices

Source terms checked against the cited upstream pages on 2026-09-28.

## SupraBench

- Project: SupraBench
- Upstream repository: https://github.com/Tianyi-Billy-Ma/SupraBench
- Upstream license text: https://raw.githubusercontent.com/Tianyi-Billy-Ma/SupraBench/main/LICENSE
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License summary and terms: https://creativecommons.org/licenses/by/4.0/
- Recorded paper identifier: arXiv:2606.13477

The upstream project states that its code and curated benchmark data are available under CC BY 4.0. This repository uses the Binding-Affinity-Prediction material and preserves upstream source identities where available.

Changes made for this repository include, depending on the file: removal of records without usable host or guest structures; addition of condition buckets; construction of fixed random, host-cold, guest-cold, double-cold and family-cold split artifacts; identity and operational-parent mappings; feature construction; saved model predictions; null-model comparisons; and aggregation or sensitivity analyses. Exact transformations are documented beside the corresponding files.

No endorsement by the SupraBench authors or maintainers is implied.

## SupraBank

- Project: SupraBank
- Source: https://suprabank.org/
- License reported by the SupraBench upstream source: CC BY 4.0

Some binding records in the SupraBench BAP material originate from SupraBank. Those records retain the SupraBank attribution chain in addition to the SupraBench source notice. Filtering, normalization, split assignment, prediction, and aggregation by this repository are modifications and are identified as such.

## Europe PMC source material

The upstream SupraBench documentation notes that Europe PMC text remains subject to the original article licenses. This repository does not redistribute article full text or excerpts. It uses curated numerical/structural benchmark records and source labels under the SupraBench data distribution. If article text is added later, its per-article license must be checked and recorded before distribution.

## Runtime dependencies

The public environment specifications name NumPy, Pandas, SciPy, scikit-learn, RDKit, XGBoost, LightGBM, PyTorch and PyTorch Geometric. These packages are not redistributed in the repository. Each remains subject to its own upstream license and notices.

