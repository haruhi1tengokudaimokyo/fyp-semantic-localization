"""ACE per-room paired recall differences, clustered by official test sequence.
No pooled cross-room leaderboard; incomplete training seeds remain incomplete.
"""

import csv, json, math, re
from pathlib import Path
import numpy as np
from core import atomic


def collect(tasks, out):
    out = Path(out)
    failures = []
    room_runs = {}
    comparisons = []
    for task in tasks:
        tid = task["task_id"]
        if (
            task["status"] != "COMPLETED"
            or not task["formal"]
            or task["experiment_id"] != "A"
        ):
            continue
        if not (tid.endswith(".official") or tid.endswith(".eval")):
            continue
        summary = next(
            (Path(p) for p in task["outputs"] if Path(p).name == "summary.json"), None
        )
        if summary is None or not (summary.parent / "per_frame.csv").exists():
            continue
        rows = list(csv.DictReader((summary.parent / "per_frame.csv").open()))
        values = {}
        for r in rows:
            te = float(r["translation_m"])
            rot = float(r["rotation_deg"])
            good = math.isfinite(te) and math.isfinite(rot) and te <= 0.05 and rot <= 5
            values[r["frame"]] = int(good)
            if not good:
                failures.append(
                    dict(
                        task_id=tid,
                        dataset_id=task["dataset_id"],
                        frame=r["frame"],
                        translation_m=te if math.isfinite(te) else None,
                        rotation_deg=rot if math.isfinite(rot) else None,
                        inliers=int(r["inliers"]),
                        status=r["status"],
                    )
                )
        room_runs.setdefault(task["dataset_id"], {})[tid] = values
    rng = np.random.default_rng(1305)
    for room, runs in room_runs.items():
        official = next((k for k in runs if k.endswith(".official")), None)
        if official is None:
            continue
        base = runs[official]
        for tid, values in runs.items():
            if tid == official:
                continue
            assert set(values) == set(base), "Unequal full ACE query sets"
            groups = {}
            for frame in base:
                match = re.match(r"(seq-\d+)", frame)
                group = match.group(1) if match else "unverified_single_sequence"
                groups.setdefault(group, []).append(values[frame] - base[frame])
            sums = np.array([sum(v) for v in groups.values()])
            counts = np.array([len(v) for v in groups.values()])
            sample = rng.integers(0, len(groups), (10000, len(groups)))
            boot = sums[sample].sum(1) / counts[sample].sum(1)
            comparisons.append(
                dict(
                    dataset_id=room,
                    baseline=official,
                    candidate=tid,
                    n_queries=len(base),
                    test_sequence_clusters=len(groups),
                    delta_recall_5cm_5deg=float(sums.sum() / counts.sum()),
                    sequence_cluster_bootstrap_95CI=(
                        np.quantile(boot, [0.025, 0.975]).tolist()
                        if len(groups) > 1
                        else None
                    ),
                    limitation=(
                        "No between-sequence interval estimated for a single test sequence"
                        if len(groups) == 1
                        else "Few official sequence clusters; interval describes this fixed room, not unseen-building generalization"
                    ),
                    replicates=10000,
                    seed=1305,
                )
            )
    atomic(out / "ace_paired_statistics.json", comparisons)
    atomic(out / "ace_failed_queries.json", failures)
    with (out / "ace_failed_queries.csv.tmp").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "dataset_id",
                "frame",
                "translation_m",
                "rotation_deg",
                "inliers",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(failures)
    (out / "ace_failed_queries.csv.tmp").replace(out / "ace_failed_queries.csv")
    return dict(failed_queries=len(failures), paired_comparisons=len(comparisons))
