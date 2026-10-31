"""Paired position-cluster bootstrap; four yaws are NOT independent buildings."""

import argparse, json
from pathlib import Path
import numpy as np


def main(a):
    base = Path(a.runs)
    names = {
        "geometry": ("hloc_knownmap_relocalization", "geometry"),
        "geometry_semantic": ("hloc_knownmap_relocalization", "semantic"),
        "geometry_clip": ("clip_hloc_relocalization", "geometry"),
        "geometry_clip_semantic": ("clip_hloc_relocalization", "semantic"),
    }
    queries = json.loads(
        (base / "hloc_knownmap_relocalization/queries.json").read_text()
    )
    groups = np.array([int(Path(q["rgb"]).name.split("_")[0]) for q in queries])
    unique = np.unique(groups)
    rng = np.random.default_rng(1305)
    draw = rng.integers(0, len(unique), size=(10000, len(unique)))
    result = {}
    for title, threshold in [
        ("5cm5deg", (0.05, 5)),
        ("25cm10deg", (0.25, 10)),
        ("1m20deg", (1, 20)),
    ]:
        draws = {}
        stats = {}
        den = np.array([(groups == g).sum() for g in unique])
        denom = den[draw].sum(1)
        for name, (run, mode) in names.items():
            m = json.loads((base / run / "summary.json").read_text())[mode]
            errors = m["errors_all_queries"]
            success = np.array(
                [
                    e["translation_m"] is not None
                    and e["translation_m"] <= threshold[0]
                    and e["rotation_deg"] <= threshold[1]
                    for e in errors
                ]
            )
            num = np.array([success[groups == g].sum() for g in unique])
            values = num[draw].sum(1) / denom
            draws[name] = values
            stats[name] = dict(
                recall=float(success.mean()),
                position_cluster_bootstrap_ci95=np.percentile(
                    values, [2.5, 97.5]
                ).tolist(),
            )
        diffs = {
            name: dict(
                paired_delta_vs_geometry=stats[name]["recall"]
                - stats["geometry"]["recall"],
                cluster_bootstrap_delta_ci95=np.percentile(
                    values - draws["geometry"], [2.5, 97.5]
                ).tolist(),
            )
            for name, values in draws.items()
            if name != "geometry"
        }
        result[title] = dict(methods=stats, paired_differences=diffs)
    report = dict(
        seed=1305,
        replicates=10000,
        position_clusters=len(unique),
        queries=len(groups),
        unit="spatial position, all available yaw views resampled together",
        caveat="one held-out synthetic scene; CIs describe within-scene sampling only, not cross-building population generalization",
        results=result,
    )
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs")
    p.add_argument("output")
    main(p.parse_args())
