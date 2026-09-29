from __future__ import annotations

import hashlib
import runpy
import sys
import warnings
from pathlib import Path


ORIGINAL_COMPARATOR = Path(__file__).resolve().parent / "historical" / "compare_gnn_repeats.py"
EXPECTED_COMPARATOR_SHA256 = (
    "bb872ab28339172a495176d588469622713e217e6a51c2cf8af4f1eb19dc7d51"
)
TARGET_WARNING_MESSAGE = (
    r"^`torch\.jit\.script` is deprecated\. Please switch to "
    r"`torch\.compile` or `torch\.export`\.$"
)
TARGET_WARNING_MODULE = r"^torch\.jit\._script$"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


actual_sha256 = sha256_file(ORIGINAL_COMPARATOR)
if actual_sha256 != EXPECTED_COMPARATOR_SHA256:
    raise RuntimeError(
        "original comparator SHA-256 mismatch: "
        f"expected {EXPECTED_COMPARATOR_SHA256}, actual {actual_sha256}"
    )

# Retain warnings-as-errors globally. Only this exact import-time deprecation
# remains visible under the normal "default" action.
warnings.simplefilter("error", append=False)
warnings.filterwarnings(
    "default",
    message=TARGET_WARNING_MESSAGE,
    category=DeprecationWarning,
    module=TARGET_WARNING_MODULE,
    append=False,
)

# Match direct script execution for argv[0] and import lookup (notably common.py).
sys.argv = [str(ORIGINAL_COMPARATOR), *sys.argv[1:]]
sys.path.insert(0, str(ORIGINAL_COMPARATOR.parent))
runpy.run_path(str(ORIGINAL_COMPARATOR), run_name="__main__")
