"""Paired, house-group bootstrap of fixed preregistered localization variants.
Only recall differences are averaged; poses/errors from distinct map gauges never mix.
Four test houses are a small population, so intervals do not establish real-building transfer.
"""

import csv, json
from pathlib import Path
import numpy as np
from core import atomic


def collect(snapshot, out):
    if not any(
        t["task_id"] == "F.formal.clip.ablation" and t["status"] == "COMPLETED"
        for t in snapshot["tasks"]
    ):
        return
    path = Path("/data/fyp_3d/runs/formal/relocalization_hssd/grouped_queries.csv")
    rows = list(csv.DictReader(path.open()))
    grouped = {}
    for r in rows:
        grouped.setdefault(r["method"], {})[(r["building_group"], r["query"])] = r
    baseline = "superpoint_bow_k8:geometry"
    base = grouped[baseline]
    buildings = sorted({key[0] for key in base})
    rng = np.random.default_rng(1305)
    sample = rng.integers(0, len(buildings), (10000, len(buildings)))
    results = []

    def success(row, t, r):
        if not row["translation_m"] or not row["rotation_deg"]:
            return False
        return float(row["translation_m"]) <= t and float(row["rotation_deg"]) <= r

    for method, candidate in sorted(grouped.items()):
        assert set(candidate) == set(
            base
        ), "All methods must preserve exact common queries"
        for translation, rotation in [(0.05, 5), (0.25, 10), (1.0, 30)]:
            per_house = {}
            for b in buildings:
                keys = [key for key in base if key[0] == b]
                d = [
                    int(success(candidate[k], translation, rotation))
                    - int(success(base[k], translation, rotation))
                    for k in keys
                ]
                per_house[b] = dict(queries=len(keys), delta_recall=float(np.mean(d)))
            differences = np.array([per_house[b]["delta_recall"] for b in buildings])
            boot = differences[sample].mean(1)
            results.append(
                dict(
                    method=method,
                    baseline=baseline,
                    translation_threshold_m=translation,
                    rotation_threshold_deg=rotation,
                    house_groups=len(buildings),
                    queries=len(base),
                    macro_house_delta_recall=float(differences.mean()),
                    paired_house_bootstrap_95CI=(
                        np.quantile(boot, [0.025, 0.975]).tolist()
                        if len(buildings) > 1
                        else None
                    ),
                    per_house=per_house,
                    replicates=10000,
                    seed=1305,
                    limitation="Four synthetic held-out house groups; descriptive macro-house uncertainty only, not a real-building/general population guarantee. All variants reported; no test-selected winner.",
                )
            )
    atomic(Path(out) / "relocalization_building_statistics.json", results)
    target = Path(out) / "relocalization_building_statistics.csv"
    temporary = target.with_suffix(".tmp")
    names = [
        "method",
        "baseline",
        "translation_threshold_m",
        "rotation_threshold_deg",
        "house_groups",
        "queries",
        "macro_house_delta_recall",
        "CI_low",
        "CI_high",
    ]
    with temporary.open("w") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for row in results:
            w.writerow(
                {
                    **{k: row[k] for k in names if k not in ["CI_low", "CI_high"]},
                    "CI_low": (
                        row["paired_house_bootstrap_95CI"][0]
                        if len(buildings) > 1
                        else None
                    ),
                    "CI_high": (
                        row["paired_house_bootstrap_95CI"][1]
                        if len(buildings) > 1
                        else None
                    ),
                }
            )
    temporary.replace(target)
