"""Read-only checks before publishing the three FORCE2020 notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_FORCE2020_dataset_audit_from_scratch.ipynb",
    "01_FORCE_clean_group_spatial_support_splits.ipynb",
    "01B_FORCE_fix_local_windows_and_lock_contract.ipynb",
]
OLD_PATH_MARKERS = (
    "/Users/",
    "C:\\Users\\",
    "mohaddesemoghaddam/Desktop",
    "FORCE_Causal_Project",
    "force2020_dataset_audit",
)

issues: list[str] = []
for name in NOTEBOOKS:
    file = ROOT / "notebooks" / name
    if not file.exists():
        issues.append(f"MISSING NOTEBOOK: notebooks/{name}")
        continue
    try:
        nb = json.loads(file.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        issues.append(f"INVALID IPYNB: notebooks/{name}: {exc}")
        continue

    for number, cell in enumerate(nb.get("cells", []), start=1):
        source = "".join(cell.get("source", []))
        for marker in OLD_PATH_MARKERS:
            if marker in source:
                # Only a warning: educational descriptions may mention old paths.
                issues.append(f"PATH REVIEW: {name} cell {number} contains {marker!r}")
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                issues.append(f"SAVED OUTPUTS: {name} cell {number}")
            if cell.get("execution_count") is not None:
                issues.append(f"EXECUTION COUNT: {name} cell {number}")
    size_mb = file.stat().st_size / (1024 * 1024)
    if size_mb > 10:
        issues.append(f"LARGE NOTEBOOK: {name} ({size_mb:.1f} MiB)")

for folder in (ROOT / "data" / "raw", ROOT / "outputs"):
    if folder.exists():
        print(f"Local-only directory: {folder.relative_to(ROOT)}")

if issues:
    print("\nPUBLICATION REVIEW REQUIRED:")
    for issue in issues:
        print(" -", issue)
    raise SystemExit(1)

print("Pre-publication check passed: notebook files found, no saved outputs/execution counts, no legacy path markers.")
