"""Deterministically verify the Walk Score colour-to-height orientation contract."""

import csv
import json
from pathlib import Path

import numpy as np
import shapefile

from prepare_inputs import ROOT, score_surface


def main():
    rows = [row for row in csv.DictReader((ROOT / "data" / "toronto_walkscore_extended.csv").open(encoding="utf-8")) if row["score"]]
    ring = shapefile.Reader(str(ROOT / "data" / "boundary" / "citygcs_regional_mun_wgs84.shp")).shape(0).points
    expected_native_height, level, city = score_surface(rows, ring)
    stored_height = np.load(ROOT / "toronto_inputs.npz")["height"]
    expected_north_up_height = np.asarray(0.03 + 0.97 * level**1.10, dtype=np.float32)

    if not np.array_equal(stored_height, expected_native_height):
        raise AssertionError("The stored heightfield no longer matches the archived Walk Score input.")
    if not np.array_equal(np.flipud(stored_height), expected_north_up_height):
        raise AssertionError("North-up score texture and the visual terrain heightfield are not aligned.")
    ordered_scores = level[city].ravel()
    ordered_heights = np.flipud(stored_height)[city].ravel()
    order = np.argsort(ordered_scores, kind="stable")
    if np.any(np.diff(ordered_heights[order]) < 0):
        raise AssertionError("A higher Walk Score maps to a lower visual height.")

    report = {
        "status": "PASS",
        "valid_scores": len(rows),
        "native_storage": "height rows are south-to-north for Forge3D terrain geometry",
        "visual_storage": "texture rows are north-to-south; flipud(height) is its matching north-up visual field",
        "native_fixture_exact": True,
        "visual_height_exact": True,
        "higher_score_to_lower_height_pairs": 0,
    }
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "height-orientation-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
