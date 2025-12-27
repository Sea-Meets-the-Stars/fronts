#!/usr/bin/env python
"""Generate a CSV table with UIDs extracted from thinning output plot filenames."""

import csv
import re
from pathlib import Path


def extract_uids(plots_dir: str) -> list[str]:
    """Extract UIDs from fronts_thinwk_*.png filenames."""
    plots_path = Path(plots_dir)
    pattern = re.compile(r"fronts_thinwk_(\d+)\.png")

    uids = []
    for f in plots_path.glob("fronts_thinwk_*.png"):
        match = pattern.match(f.name)
        if match:
            uids.append(match.group(1))

    return sorted(uids)


def main():
    plots_dir = Path(__file__).parent / "plots"
    output_file = Path(__file__).parent / "uid_comparison.csv"

    uids = extract_uids(plots_dir)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "CC", "BK"])
        for uid in uids:
            writer.writerow([uid, "", ""])

    print(f"Generated {output_file} with {len(uids)} entries")


if __name__ == "__main__":
    main()
