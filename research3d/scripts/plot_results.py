"""Publication exports from real stored RGB, depth, predictions and geometry."""

import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 220,
    }
)


def save(fig, path):
    fig.savefig(str(path) + ".png", bbox_inches="tight")
    fig.savefig(str(path) + ".svg", bbox_inches="tight")
    plt.close(fig)


def equal(ax, points):
    mn = points.min(0)
    mx = points.max(0)
    c = (mn + mx) / 2
    radius = max(mx - mn) / 2
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def results(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(a.run)
    scene = Path(a.scene)
    rows = list(csv.DictReader((run / "per_frame.csv").open()))
    p = np.load(run / "predictions.npz")
    T = p["T_world_camera"]
    gt = np.stack(
        [
            np.loadtxt(
                scene / "test" / "poses" / str(n).replace(".color.png", ".pose.txt")
            )
            for n in p["frame"]
        ]
    )
    m = np.load(Path(a.geometry) / "geometry.npz")
    xyz = m["points_world"]
    c = m["colors_rgb"]
    skip = max(1, len(xyz) // 25000)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*xyz[::skip].T, c=c[::skip] / 255, s=0.4, alpha=0.32, rasterized=True)
    ax.plot(*gt[:, :3, 3].T, color="#208a85", lw=1.0, label="Ground truth (test)")
    ax.scatter(
        *T[::5, :3, 3].T,
        c="#d85d49",
        s=10,
        marker="x",
        depthshade=False,
        label="ACE independent relocalization (every 5th shown)",
    )
    corners = (
        np.array(
            [
                [-320 / 525, -240 / 525, 1],
                [320 / 525, -240 / 525, 1],
                [320 / 525, 240 / 525, 1],
                [-320 / 525, 240 / 525, 1],
            ]
        )
        * 0.12
    )
    for idx in [0, len(T) // 3, 2 * len(T) // 3]:
        pose = T[idx]
        world = corners @ pose[:3, :3].T + pose[:3, 3]
        for c in world:
            ax.plot(*np.stack([pose[:3, 3], c]).T, color="#d85d49", lw=0.8)
        ax.plot(*world[[0, 1, 2, 3, 0]].T, color="#d85d49", lw=0.8)
    equal(ax, xyz)
    ax.view_init(25, -60)
    ax.legend(loc="upper left")
    ax.set_title("7-Scenes Heads | Train-only 3D geometry + held-out query poses")
    save(fig, out / "heads_3d")
    t = np.array([float(x["translation_m"]) for x in rows])
    r = np.array([float(x["rotation_deg"]) for x in rows])
    ms = np.array([float(x["runtime_ms"]) for x in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for v, ax, label in [
        (t * 100, axs[0], "Translation error (cm)"),
        (r, axs[1], "Rotation error (deg)"),
    ]:
        ax.plot(np.sort(v), np.arange(1, len(v) + 1) / len(v), color="#208a85", lw=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axs[2].hist(ms, bins=30, color="#5068a4")
    axs[2].set_xlabel("Measured query latency (ms)")
    axs[2].set_ylabel("Frames")
    fig.suptitle(f"7-Scenes Heads | ACE + DSAC* | n={len(rows)} | test split")
    fig.tight_layout()
    save(fig, out / "heads_metrics")
    selected = np.r_[np.argsort(t)[:3], np.argsort(t)[-3:]]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, idx in zip(axs.flat, selected):
        row = rows[idx]
        ax.imshow(Image.open(scene / "test" / "rgb" / row["frame"]))
        ax.axis("off")
        ax.set_title(
            f"{row['frame']}\n{t[idx]*100:.1f} cm, {r[idx]:.2f} deg | inliers={row['inliers']}",
            fontsize=9,
        )
    fig.suptitle(
        "7-Scenes Heads | Three smallest / three largest measured translation errors"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
    save(fig, out / "heads_cases")


def simulation(a):
    run = Path(a.run)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "acceptance.json").read_text())
    # First nondegenerate frame, deterministic choice independent of estimator scores.
    idx = next(
        x["frame"]
        for x in meta["views"]
        if x["rgb_std"] > 15 and x["valid_depth_fraction"] > 0.5
    )
    rgb = np.asarray(Image.open(run / f"{idx:04d}.rgb.png"))
    depth = np.load(run / f"{idx:04d}.depth.npy")
    sem = np.load(run / f"{idx:04d}.semantic.npy")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(rgb)
    axs[0].set_title("Rendered RGB")
    im = axs[1].imshow(
        np.ma.masked_where(depth <= 0, depth), cmap="viridis", vmin=0, vmax=8
    )
    axs[1].set_title("Axial depth (m)")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    ids = np.unique(sem)
    index = np.searchsorted(ids, sem)
    axs[2].imshow(index, cmap="tab20", vmin=0, vmax=max(1, len(ids) - 1))
    axs[2].set_title(f"Semantic sensor IDs ({len(ids)} visible)")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(
        f"{meta['scene']} | Habitat {meta['habitat_version']} | actual sensor observations"
    )
    fig.tight_layout()
    save(fig, out / (run.name + "_observations"))
    cloud = np.load(run / "rendered_cloud.npz")
    xyz = cloud["points_world"]
    c = cloud["colors_rgb"]
    skip = max(1, len(xyz) // 35000)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*xyz[::skip].T, c=c[::skip] / 255, s=0.6, rasterized=True)
    ax.scatter(
        *cloud["camera_positions"].T, c="#d85d49", s=20, label="Actual camera centers"
    )
    equal(ax, xyz)
    ax.legend()
    ax.set_title(f"{meta['scene']} | RGB-D backprojection in map frame")
    save(fig, out / (run.name + "_3d"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["results", "simulation"])
    p.add_argument("run")
    p.add_argument("output")
    p.add_argument("--scene")
    p.add_argument("--geometry")
    a = p.parse_args()
    results(a) if a.kind == "results" else simulation(a)
