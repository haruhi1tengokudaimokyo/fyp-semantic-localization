"""Rebuild typed metric tables from validated results, without pooling unlike tasks."""

import csv, json, math, statistics
from pathlib import Path
from core import atomic


def aggregate(snapshot, root):
    out = Path(root) / "artifacts/aggregate"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_groups = {}
    for task in snapshot["tasks"]:
        if task["status"] != "COMPLETED":
            continue
        for metric, value in task.get("key_metrics", {}).items():
            if isinstance(value, bool) or not math.isfinite(value):
                continue
            rows.append(
                dict(
                    task_id=task["task_id"],
                    experiment_id=task["experiment_id"],
                    dataset_id=task["dataset_id"],
                    formal=task["formal"],
                    seed=task["seed"],
                    metric=metric,
                    value=value,
                    context=json.dumps(
                        task.get("metric_context", {}), ensure_ascii=False
                    ),
                )
            )
            tid = task["task_id"]
            if task["formal"] and (
                (".train." in tid and tid.endswith(".eval"))
                or tid.startswith("D.train.")
                or tid == "reuse.ace.heads.local"
            ):
                family = (
                    "ACE_local"
                    if task["experiment_id"] == "A"
                    else ".".join(tid.split(".")[:3])
                )
                key = (task["dataset_id"], family, metric)
                seed_groups.setdefault(key, []).append((task["seed"], value))
    temporary = out / "metrics.csv.tmp"
    with temporary.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "experiment_id",
                "dataset_id",
                "formal",
                "seed",
                "metric",
                "value",
                "context",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    temporary.replace(out / "metrics.csv")
    statistics_rows = []
    for (dataset, family, metric), values in seed_groups.items():
        unique = dict(values)
        v = list(unique.values())
        statistics_rows.append(
            dict(
                dataset_id=dataset,
                family=family,
                metric=metric,
                seeds=sorted(unique),
                n_seeds=len(v),
                mean=statistics.mean(v),
                std_ddof1=statistics.stdev(v) if len(v) > 1 else None,
                note="Descriptive variability across fixed training seeds on identical dataset/protocol; not pooled across datasets",
            )
        )
    atomic(out / "seed_statistics.json", statistics_rows)
    atomic(
        out / "index.json",
        dict(
            generated_utc=snapshot["generated_utc"],
            metrics_csv=str(out / "metrics.csv"),
            seed_statistics=str(out / "seed_statistics.json"),
            metric_rows=len(rows),
            policy="Only validated completed outputs; formal and pilot separated; no cross-sensor/map/dataset leaderboard; absent metrics omitted, never filled",
        ),
    )
    from ace_statistics import collect

    collect(snapshot["tasks"], out)
    from relocalization_statistics import collect as collect_relocalization

    collect_relocalization(snapshot, out)
    from plot_formal import plot

    plot(snapshot, root)
    from plot_semantic_localization import plot as plot_semantic

    plot_semantic(snapshot, root)
    from common_system_statistics import collect as collect_common

    collect_common(snapshot, root)
    return str(out / "index.json")
