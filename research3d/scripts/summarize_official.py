"""Convert original ACE text outputs to an auditable JSON summary."""

import argparse, json
from pathlib import Path
import numpy as np


def main(a):
    run = Path(a.run)
    rows = [
        s.split()
        for s in (run / "poses_7scenes_heads_official.txt").read_text().splitlines()
    ]
    t = np.array([float(r[-2]) for r in rows])
    r = np.array([float(x[-3]) for x in rows])
    stats = np.loadtxt(run / "test_7scenes_heads_official.txt")
    result = dict(
        method="unmodified official test_ace.py + pretrained Heads head",
        n=len(rows),
        median_translation_m=float(np.median(t)),
        median_rotation_deg=float(np.median(r)),
        official_reported_mean_rotation_deg=float(stats[0]),
        official_reported_mean_translation_cm=float(stats[1]),
        mean_processing_ms=float(stats[2]) * 1000,
        recall={
            f"{m}m_{d}deg": float(np.mean((t < m) & (r < d)))
            for m, d in [(0.01, 1), (0.02, 2), (0.05, 5), (0.1, 5)]
        },
        failures_5cm_5deg=[
            dict(
                frame=x[0],
                translation_m=float(x[-2]),
                rotation_deg=float(x[-3]),
                inliers=int(x[-1]),
            )
            for x in rows
            if float(x[-2]) >= 0.05 or float(x[-3]) >= 5
        ],
        timing_scope="official loop excludes RGB DataLoader loading, includes network and DSAC*, first frame included; not directly identical scope to GT-free adapter",
    )
    # The upstream aggregate stores medians, computed with its own indexing convention.
    result["official_reported_median_rotation_deg"] = result.pop(
        "official_reported_mean_rotation_deg"
    )
    result["official_reported_median_translation_cm"] = result.pop(
        "official_reported_mean_translation_cm"
    )
    (run / "summary.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    main(p.parse_args())
