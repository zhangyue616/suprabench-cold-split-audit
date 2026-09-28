# Licensing

This repository contains materials with different origins and licenses. No single license applies to every file.

## Original software

Original source code that is explicitly listed as `MIT` in [FILE_LICENSES.tsv](FILE_LICENSES.tsv) is licensed under the MIT License:

Copyright (c) 2026 Contributors to the SupraBench cold-split audit

The full MIT text is in [LICENSES/MIT.txt](LICENSES/MIT.txt). A file is covered by MIT only when it is listed with `license_id=MIT` in the file-license manifest or carries `SPDX-License-Identifier: MIT`. The repository README and command documentation are MIT associated documentation only where explicitly listed.

The MIT grant does not cover upstream SupraBench code, datasets, derived scientific data, saved predictions, model artifacts, figures, manuscripts, or third-party software.

## Benchmark data and data-derived scientific artifacts

SupraBench code and curated benchmark data are distributed by the upstream project under Creative Commons Attribution 4.0 International (CC BY 4.0). The binding records used by SupraBench originate from SupraBank, which is also identified upstream as CC BY 4.0.

Files listed with `license_id=CC-BY-4.0` must retain the following. A separate machine-readable `rights_class` may identify files derived from upstream data; that classification does not create a different license identifier.

- attribution to SupraBench and, where applicable, SupraBank;
- a link to https://creativecommons.org/licenses/by/4.0/;
- the upstream source link and version or fixed file identity available to this repository;
- a description of changes, including filtering, condition bucketing, identity mapping, split construction, prediction generation, or aggregation as applicable;
- wording that does not imply endorsement by the upstream authors or projects.

This distribution does not duplicate the large upstream `records.parquet`. It records the official source, expected byte size and SHA-256 so that users can fetch and verify it.

## Documentation and figures

Repository documentation and figures are not automatically covered by the code license. Every published file is identified in `FILE_LICENSES.tsv`. The selected unchanged `Figure_1.png` and `Figure_6.png` are distributed under CC BY 4.0. If no license is listed, no additional reuse license is granted beyond applicable law.

## Third-party software

NumPy, Pandas, SciPy, scikit-learn, RDKit, XGBoost, LightGBM, PyTorch and PyTorch Geometric are runtime dependencies and are not vendored by this repository. Their names and version pins do not relicense them; each remains governed by its own upstream terms.

See [THIRD_PARTY.md](THIRD_PARTY.md) for source and attribution details.


License-text entries in the manifest identify the supplied license documents; they do not place third-party license wording under the project software license.
