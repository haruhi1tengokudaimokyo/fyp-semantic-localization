"""Scientific figures from completed artifacts only."""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import chi2


def read(p):
    return json.loads(Path(p).read_text())


def main(a):
    base = Path(a.base)
    runs = base / "runs"
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fast = read(runs / "macvo/allfp32_cov_200/summary.json")
    paper = read(runs / "macvo/paper_cov_200/summary.json")
    plain = read(runs / "macvo/geometric_200/summary.json")
    replay = read(runs / "macvo/weight_replay_200/summary.json")
    seg = read(runs / "semantic/segformer_b0_40epochs/summary.json")
    hist = read(runs / "semantic/segformer_b0_40epochs/history.json")
    # Geometric baseline script stores its metrics under metrics rather than raw.
    if "metrics" in plain:
        plain = plain["metrics"]
    print("plain keys", list(plain)[:5], flush=True)
    geo = read(runs / "semantic/hloc_knownmap_relocalization/summary.json")
    clip = read(runs / "semantic/clip_hloc_relocalization/summary.json")
    sift = read(runs / "semantic/knownmap_relocalization/summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    names = ["SGBM/LK", "MACVO Fast", "MACVO paper", "Identity replay"]
    vals = [
        plain["ate_se3_aligned_rmse_m"],
        fast["raw"]["ate_se3_aligned_rmse_m"],
        paper["raw"]["ate_se3_aligned_rmse_m"],
        replay["identity"]["metrics"]["ate_se3_aligned_rmse_m"],
    ]
    ax.bar(names, vals, color=["#999", "#0072B2", "#009E73", "#E69F00"])
    ax.set_yscale("log")
    ax.set_ylabel("SE3-aligned ATE RMSE (m)")
    ax.set_title("200 stereo frames, 34.02m; no scale correction")
    ax.tick_params(axis="x", rotation=12)
    ax = axes[0, 1]
    names = ["SIFT", "SP+LG", "SP+LG+Sem", "CLIP+SP+LG", "CLIP+SP+LG+Sem"]
    vals = [
        sift["geometry"]["recall_25cm_10deg"],
        geo["geometry"]["recall_25cm_10deg"],
        geo["semantic"]["recall_25cm_10deg"],
        clip["geometry"]["recall_25cm_10deg"],
        clip["semantic"]["recall_25cm_10deg"],
    ]
    ax.bar(
        names,
        np.array(vals) * 100,
        color=["#999", "#0072B2", "#56B4E9", "#009E73", "#E69F00"],
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall @25cm/10deg (%)")
    ax.set_title("76 query views, one held-out HSSD scene")
    ax.tick_params(axis="x", rotation=15)
    ax = axes[1, 0]
    ax.bar(
        ["Initial decoder", "Selected validation", "Held-out test"],
        [
            seg["initial_random_decoder_test"]["mean_iou"] * 100,
            max(r["validation"]["mean_iou"] for r in hist) * 100,
            seg["test"]["mean_iou"] * 100,
        ],
        color=["#999", "#0072B2", "#D55E00"],
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("mIoU on known-label pixels (%)")
    ax.set_title("SegFormer-B0: large cross-scene generalization gap")
    ax = axes[1, 1]
    m2 = read(runs / "openloris_office1_1/common_window_comparison.json")["modes"]
    m3 = read(runs / "openloris_office1_1_wheel3d/common_window_comparison.json")[
        "modes"
    ]
    names = ["2D wheel+IMU", "3D wheel+IMU", "2D 3-sensor", "3D 3-sensor"]
    vals = [
        m2["imu_wheel"]["ate_se3_aligned_rmse_m"],
        m3["imu_wheel"]["ate_se3_aligned_rmse_m"],
        m2["stereo_imu_wheel"]["ate_se3_aligned_rmse_m"],
        m3["stereo_imu_wheel"]["ate_se3_aligned_rmse_m"],
    ]
    ax.bar(names, vals, color=["#999", "#0072B2", "#999", "#009E73"])
    ax.set_yscale("log")
    ax.set_ylabel("ATE RMSE (m), common time grid")
    ax.set_title("Ground-motion constraint diagnostic, office1-1")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "overview_metrics.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["epoch"] for r in hist], [r["mean_train_loss"] for r in hist])
    axes[0].set(
        xlabel="Epoch",
        ylabel="Weighted cross entropy",
        title="Real full-network fine-tuning, 3179 steps",
    )
    axes[1].plot(
        [r["epoch"] for r in hist],
        [r["validation"]["mean_iou"] for r in hist],
        label="Training-scene spatial validation",
    )
    axes[1].axhline(
        seg["test"]["mean_iou"],
        color="#D55E00",
        ls="--",
        label="Held-out scene, tested after selection",
    )
    axes[1].set(xlabel="Epoch", ylabel="Known-pixel mIoU", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "semantic_training.png", dpi=170)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    z = np.load(runs / "macvo/allfp32_cov_200/trajectory.npz")
    gt = z["T_world_camera_gt"]
    ax.plot(*gt[:, :3, 3].T, color="black", label="GT")
    for name, path, key, color in [
        (
            "MACVO paper",
            runs / "macvo/paper_cov_200/trajectory.npz",
            "T_world_camera_raw",
            "#009E73",
        ),
        (
            "MACVO Fast",
            runs / "macvo/allfp32_cov_200/trajectory.npz",
            "T_world_camera_raw",
            "#0072B2",
        ),
    ]:
        T = np.load(path)[key]
        T = gt[0] @ np.linalg.inv(T[0]) @ T
        ax.plot(*T[:, :3, 3].T, color=color, label=name)
    ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="First-pose anchored stereo trajectories",
    )
    ax.legend(fontsize=8)
    ax = fig.add_subplot(122)
    for mode, color in [("imu_wheel", "#0072B2"), ("stereo_imu_wheel", "#009E73")]:
        z = np.load(
            runs / "openloris_office1_1_wheel3d" / mode / "evaluated_trajectory.npz"
        )
        t = z["timestamp_s"] - 1560000083.9224234
        T = z["T_world_imu_first_anchored"]
        G = z["T_world_imu_gt"]
        ax.plot(
            t,
            np.linalg.norm(T[:, :3, 3] - G[:, :3, 3], axis=1),
            color=color,
            label=mode,
        )
    ax.axvspan(
        8, 12, color="#E69F00", alpha=0.15, label="Separate outage test interval"
    )
    ax.set(
        xlabel="Real time since first IMU (s)",
        ylabel="First-pose anchored translation error (m)",
        title="Startup delay is retained, not hidden",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "motion_trajectories.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, path, color in [
        ("Fast", runs / "macvo/allfp32_cov_200", "#0072B2"),
        ("Paper", runs / "macvo/paper_cov_200", "#009E73"),
    ]:
        m = read(path / "residual_audit.json")
        x = np.array([float(k) for k in m["empirical_chi2_3_coverage"]])
        axes[0].plot(
            x,
            list(m["empirical_chi2_3_coverage"].values()),
            "o-",
            label=name,
            color=color,
        )
        v = np.load(path / "residuals.npz")["observation_mahalanobis_squared"]
        v = v[np.isfinite(v) & (v > 0)]
        bins = np.logspace(-5, 4, 100)
        axes[1].hist(
            v, bins=bins, density=True, histtype="step", label=name, color=color
        )
    axes[0].plot([0, 1], [0, 1], "k--", label="Ideal independent Gaussian")
    axes[0].set(
        xlabel="Nominal chi-square(3) coverage",
        ylabel="Empirical 3D match coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Squared observation Mahalanobis distance",
        ylabel="Density",
        title="Observation consistency, NOT pose covariance",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "uncertainty_coverage.png", dpi=170)
    plt.close(fig)
    manifest = read(base / "datasets/hssd_semantic_pilot/manifest.json")
    test = [r for r in manifest["frames"] if r["split"] == "test"]
    palette = np.array(
        [
            [0, 114, 178],
            [230, 159, 0],
            [0, 158, 115],
            [204, 121, 167],
            [86, 180, 233],
            [213, 94, 0],
            [240, 228, 66],
        ],
        np.uint8,
    )
    fig, axes = plt.subplots(4, 4, figsize=(12, 9))
    for i in range(4):
        z = np.load(runs / "semantic/segformer_b0_40epochs" / f"prediction_{i:02d}.npz")
        gt = z["ground_truth_eval_only"]
        pred = z["predicted_label"]
        rgb = np.array(
            Image.open(base / "datasets/hssd_semantic_pilot" / test[i]["rgb"])
        )
        gtcolor = np.zeros((*gt.shape, 3), np.uint8)
        mask = gt != 255
        gtcolor[mask] = palette[gt[mask]]
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gtcolor)
        axes[i, 2].imshow(palette[pred])
        axes[i, 3].imshow(
            z["class_probabilities"].max(0), vmin=0, vmax=1, cmap="viridis"
        )
        for j in range(4):
            axes[i, j].axis("off")
    for j, title in enumerate(
        [
            "RGB (first four test views)",
            "Native partial GT; unknown black",
            "Trained prediction (full image)",
            "Predicted confidence",
        ]
    ):
        axes[0, j].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "semantic_examples.png", dpi=150)
    plt.close(fig)
    print("saved figures", list(out.glob("*.png")), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/data/fyp_3d")
    p.add_argument("--output", required=True)
    main(p.parse_args())
