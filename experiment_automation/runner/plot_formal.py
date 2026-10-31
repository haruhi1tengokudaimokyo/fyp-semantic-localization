"""Actual accepted formal trajectories, missing poses and per-dataset error plots."""

import hashlib, json
from pathlib import Path
import numpy as np
from core import atomic


def plot(snapshot, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = {t["task_id"]: t["status"] for t in snapshot["tasks"]}
    base = Path("/data/fyp_3d/runs/formal")
    out = Path(root) / "artifacts/aggregate"
    sources = []
    for task in snapshot["tasks"]:
        if task["status"] == "COMPLETED" and (
            task["task_id"].startswith("macvo.demo.")
            or task["task_id"] == "G.ace_macvo.integration"
            or task["task_id"].startswith("J.tum.")
        ):
            for name in task["outputs"]:
                p = Path(name)
                if p.is_file() and p.name == "summary.json":
                    sources.append((str(p), hashlib.sha256(p.read_bytes()).hexdigest()))
    fingerprint = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    receipt = out / "formal_figure.json"
    target = out / "formal_trajectories.png"
    if (
        receipt.exists()
        and target.exists()
        and json.loads(receipt.read_text())["input_sha256"] == fingerprint
    ):
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#1768ac", "#d88b20", "#b7495a"]
    if states.get("macvo.demo.paper") == "COMPLETED":
        z = np.load(base / "macvo/demo_paper_1274/trajectory.npz")
        gt = z["T_world_camera_gt"]
        ax[0, 0].plot(gt[:, 0, 3], gt[:, 2, 3], color="black", lw=2, label="GT")
        for (name, label), color in zip(
            [("paper", "Paper"), ("fast", "Fast"), ("stereo", "SGBM/LK/PnP")], colors
        ):
            tid = "macvo.demo.geometric" if name == "stereo" else "macvo.demo." + name
            if states.get(tid) != "COMPLETED":
                continue
            z = np.load(base / f"macvo/demo_{name}_1274/trajectory.npz")
            T = (
                z["T_world_camera_raw"]
                if "T_world_camera_raw" in z
                else z["T_world_camera"]
            )
            anchored = (gt[0] @ np.linalg.inv(T[0])) @ T
            lost = z["lost_before_interpolation"]
            error = np.linalg.norm(anchored[:, :3, 3] - gt[:, :3, 3], axis=1)
            ax[0, 0].plot(
                anchored[:, 0, 3],
                anchored[:, 2, 3],
                color=color,
                label=f"{label}; lost={int(lost.sum())}",
            )
            ax[0, 1].plot(error, color=color, label=label)
            if lost.any():
                ax[0, 1].scatter(
                    np.flatnonzero(lost), error[lost], color=color, marker="x", s=12
                )
        ax[0, 0].set(
            title="1274 stereo frames: first-pose anchored offline",
            xlabel="World X (m)",
            ylabel="World Z (m)",
        )
        ax[0, 0].axis("equal")
        ax[0, 0].legend(fontsize=8)
        ax[0, 1].set(
            title="Raw first-pose anchored translation error",
            xlabel="Original frame index",
            ylabel="Error (m)",
        )
        ax[0, 1].legend()
    if states.get("G.ace_macvo.integration") == "COMPLETED":
        z = np.load(base / "common/continuous_joint/global_predictions.npz")
        gt = np.load("/data/fyp_3d/datasets/common_hssd105/ground_truth_eval_only.npz")[
            "T_world_camera"
        ]
        for name in z.files:
            T = z[name]
            valid = np.isfinite(T).all((1, 2))
            error = np.full(len(T), np.nan)
            error[valid] = np.linalg.norm(T[valid, :3, 3] - gt[valid, :3, 3], axis=1)
            ax[1, 0].plot(
                np.maximum(error, 1e-5), label=f"{name}: {valid.mean():.1%} pose output"
            )
        ax[1, 0].set_yscale("log")
        ax[1, 0].set(
            title="600 simulated frames: absolute map error",
            xlabel="Original frame index; unavailable poses are gaps",
            ylabel="Translation error (m, log)",
        )
        ax[1, 0].legend(fontsize=7)
    names = []
    values = []
    for name in ["sitting_static", "sitting_xyz", "walking_static", "walking_xyz"]:
        if states.get("J.tum." + name + ".rgbd_vo") == "COMPLETED":
            names.append(name)
            values.append(
                json.loads((base / "tum" / name / "summary.json").read_text())[
                    "metrics"
                ]["ate_se3_aligned_rmse_m"]
            )
    if names:
        ax[1, 1].bar(
            names,
            values,
            color=["#1768ac" if n.startswith("sitting") else "#b7495a" for n in names],
        )
        ax[1, 1].tick_params(axis="x", rotation=20)
        ax[1, 1].set(
            title="Real TUM people: each full RGB-D sequence",
            ylabel="SE(3)-aligned ATE (m)",
            xlabel="Different motions; no pooled dynamic/static ranking",
        )
    for a in ax.ravel():
        a.grid(alpha=0.2)
    fig.suptitle(
        "Validated formal outputs | cached Fast timing is not a live-inference speed comparison",
        fontsize=11,
    )
    fig.tight_layout()
    tmp = target.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=160)
    plt.close(fig)
    tmp.replace(target)
    atomic(
        receipt,
        dict(
            input_sha256=fingerprint,
            sources=sources,
            generated_utc=snapshot["generated_utc"],
            scope="Only completed validated trajectories; raw lost masks and unavailable global poses preserved; plots read GT offline only",
        ),
    )
