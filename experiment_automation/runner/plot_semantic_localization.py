"""Actual formal semantic training history and fixed TopK8 localization outcomes."""

import csv, hashlib, json
from pathlib import Path
import numpy as np
from core import atomic


def plot(snapshot, root):
    states = {t["task_id"]: t["status"] for t in snapshot["tasks"]}
    if (
        states.get("D.train.full.1305") != "COMPLETED"
        or states.get("F.formal.clip.ablation") != "COMPLETED"
    ):
        return
    model = Path("/data/fyp_3d/runs/formal/semantic/full_seed1305")
    local = Path("/data/fyp_3d/runs/formal/relocalization_hssd")
    out = Path(root) / "artifacts/aggregate"
    target = out / "formal_semantic_localization.png"
    receipt = out / "formal_semantic_localization.json"
    sources = [
        model / "history.json",
        model / "summary.json",
        local / "grouped_queries.csv",
    ]
    key = hashlib.sha256(
        Path(__file__).read_bytes()
        + b"".join(hashlib.sha256(p.read_bytes()).digest() for p in sources)
    ).hexdigest()
    if (
        target.exists()
        and receipt.exists()
        and json.loads(receipt.read_text())["input_sha256"] == key
    ):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((model / "history.json").read_text())
    summary = json.loads((model / "summary.json").read_text())
    rows = list(csv.DictReader((local / "grouped_queries.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(
        epochs,
        [r["validation"]["mean_iou"] for r in history],
        label="Validation mIoU",
        color="#1768ac",
    )
    axes[0].axvline(
        summary["best_epoch"],
        ls="--",
        color="#87919c",
        label=f"Selected epoch {summary['best_epoch']} (validation only)",
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Partial-label validation mIoU",
        ylim=(0, 1),
        title="24 synthetic house groups; fixed split 16 / 4 / 4",
    )
    twin = axes[0].twinx()
    twin.plot(
        epochs,
        [r["mean_train_loss"] for r in history],
        color="#d88b20",
        alpha=0.7,
        label="Training loss",
    )
    twin.set_ylabel("Training cross entropy")
    twin.legend(loc="lower left", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    methods = [
        ("sift_bow_k8:geometry", "SIFT"),
        ("superpoint_bow_k8:geometry", "SP + LG"),
        ("bow_k8:semantic", "SP/LG\nhard semantic"),
        ("bow_k8:soft", "SP/LG\nsoft semantic"),
        ("clip_k8:geometry", "CLIP\n+ SP/LG"),
        ("clip_k8:semantic", "CLIP + SP/LG\nhard semantic"),
        ("clip_k8:soft", "CLIP + SP/LG\nsoft semantic"),
    ]
    values = []
    counts = []
    for name, _ in methods:
        subset = [r for r in rows if r["method"] == name]
        assert len(subset) == 771
        count = sum(int(r["success_25cm_10deg"]) for r in subset)
        counts.append(count)
        values.append(count / len(subset) * 100)
    bars = axes[1].bar(
        range(len(methods)),
        values,
        color=[
            "#87919c",
            "#1768ac",
            "#5b8f76",
            "#5b8f76",
            "#b7495a",
            "#b7495a",
            "#b7495a",
        ],
    )
    axes[1].set_xticks(range(len(methods)), [label for _, label in methods], fontsize=7)
    axes[1].set(
        ylim=(0, 100),
        ylabel="Recall@25cm / 10 degrees (%)",
        title="771 identical queries, four held-out houses; fixed TopK8",
    )
    for bar, value, count in zip(bars, values, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            f"{value:.2f}%\n{count}/771",
            ha="center",
            fontsize=7,
        )
    supported = sum(
        v["labeled_pixels"] > 0 for v in summary["test"]["classes"].values()
    )
    fig.suptitle(
        f"Test segmentation mIoU={summary['test']['mean_iou']:.3f} ({supported} supported native classes; unknown pixels ignored). Different from the earlier pilot split.",
        fontsize=10,
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    temporary = target.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)
    atomic(
        receipt,
        dict(
            input_sha256=key,
            sources=[str(p) for p in sources],
            generated_utc=snapshot["generated_utc"],
            note="All fixed TopK8 methods shown; no test-selected winner. Other predeclared TopK4/16 variants and paired-house intervals remain in CSV/JSON.",
        ),
    )
