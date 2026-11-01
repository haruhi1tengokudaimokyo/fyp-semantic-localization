"""Independent offline metric recomputation and figures for accepted H/I/J outputs."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import atomic


def collect(snapshot, root):
    tasks = {t["task_id"]: t for t in snapshot["tasks"] if t["status"] == "COMPLETED"}
    requested = [
        "H.full.system.measurement_audit",
        "I.formal.recovery",
        "I.kidnap.full_system_recovery",
        "J.person.paired_comparison",
    ]
    sources = {}
    for tid in requested:
        if tid not in tasks:
            continue
        task = tasks[tid]
        path = Path(task["outputs"][0])
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted = task["validation"]["artifacts"]
        assert any(a["path"] == str(path) and a["sha256"] == checksum for a in accepted)
        sources[tid] = dict(
            path=str(path), sha256=checksum, value=json.loads(path.read_text())
        )
    if "H.full.system.measurement_audit" not in sources:
        return
    out = Path(root) / "artifacts/aggregate"
    home = Path(sources["H.full.system.measurement_audit"]["path"]).parent
    data = Path("/data/fyp_3d/datasets/common_hssd105")
    z = np.load(home / "global_predictions.npz")
    truth = np.load(data / "ground_truth_eval_only.npz")["T_world_camera"]
    summary = sources["H.full.system.measurement_audit"]["value"]
    diag = json.loads((home / "online_diagnostics.json").read_text())
    recalculated = {}
    for name in z.files:
        P = z[name]
        assert P.shape == truth.shape == (600, 4, 4)
        valid = np.isfinite(P).all((1, 2))
        assert np.allclose(P[valid, 3], [0, 0, 0, 1])
        assert np.allclose(np.linalg.det(P[valid, :3, :3]), 1, atol=1e-4)
        te, re = np.full(600, np.inf), np.full(600, np.inf)
        te[valid] = np.linalg.norm(P[valid, :3, 3] - truth[valid, :3, 3], axis=1)
        re[valid] = np.degrees(
            Rotation.from_matrix(
                np.swapaxes(P[valid, :3, :3], 1, 2) @ truth[valid, :3, :3]
            ).magnitude()
        )
        values = dict(
            valid_pose_fraction=float(valid.mean()),
            absolute_map_translation_rmse_valid_only_m=float(
                np.sqrt(np.mean(te[valid] ** 2))
            ),
        )
        for key, t, r in [
            ("recall_5cm_5deg", 0.05, 5),
            ("recall_25cm_10deg", 0.25, 10),
            ("recall_1m_30deg", 1, 30),
        ]:
            values[key] = float(np.mean((te <= t) & (re <= r)))
        for key, value in values.items():
            assert np.isclose(
                value, summary["results"][name][key], atol=1e-12, rtol=0
            ), (name, key)
        assert len(diag[name]) == 600
        values["accepted_updates"] = sum(r["accepted_absolute"] for r in diag[name])
        assert (
            values["accepted_updates"] == summary["results"][name]["accepted_updates"]
        )
        recalculated[name] = values
    receipt = dict(
        status="independently_recomputed",
        frames=600,
        variants=len(z.files),
        results=recalculated,
        source_summaries={
            k: {a: b for a, b in v.items() if a != "value"} for k, v in sources.items()
        },
        arrays_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                home / "global_predictions.npz",
                data / "ground_truth_eval_only.npz",
            ]
        },
        scope="One seen-building synthetic route; not cross-building significance or real-time combined inference",
        unavailable_frames_in_denominator=True,
    )
    atomic(out / "common_system_audit.json", receipt)
    plot(sources, summary, out)


def plot(sources, summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = hashlib.sha256(
        Path(__file__).read_bytes()
        + json.dumps(
            {k: v["sha256"] for k, v in sources.items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    target, cache = (
        out / "formal_common_system.png",
        out / "formal_common_system_figure.json",
    )
    if (
        target.exists()
        and cache.exists()
        and json.loads(cache.read_text())["input_sha256"] == key
    ):
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    results = summary["results"]
    labels = [k.replace("_", " ") for k in results]
    x = np.arange(len(results))
    axes[0, 0].bar(
        x - 0.18,
        [v["recall_5cm_5deg"] * 100 for v in results.values()],
        0.36,
        label="5cm / 5deg",
    )
    axes[0, 0].bar(
        x + 0.18,
        [v["recall_25cm_10deg"] * 100 for v in results.values()],
        0.36,
        label="25cm / 10deg",
    )
    axes[0, 0].set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
    axes[0, 0].set(
        title="H: all nine fixed variants, same 600 frames",
        ylabel="Recall incl. unavailable (%)",
        ylim=(0, 105),
    )
    axes[0, 0].legend(fontsize=8)
    for name in [
        "ACE_gate",
        "geometry_gate",
        "clip_semantic_soft_gate",
        "ACE_clip_soft_multicandidate_weighted",
    ]:
        errors = np.array(
            [np.nan if v is None else v for v in results[name]["translation_errors_m"]]
        )
        axes[0, 1].plot(errors, label=name, lw=1)
    axes[0, 1].axhline(0.05, color="gray", ls=":", lw=1)
    axes[0, 1].set(
        title="H: absolute map errors; gaps remain unavailable",
        xlabel="Original frame",
        ylabel="Translation error (m)",
    )
    axes[0, 1].legend(fontsize=7)
    kid = sources.get("I.kidnap.full_system_recovery", sources.get("I.formal.recovery"))
    if kid:
        value = kid["value"]
        names = list(value["results"])
        y = np.arange(len(names))
        available = [
            value["results"][n]["recovery"]["0.25m_10deg"]["post_jump_recall"] * 100
            for n in names
        ]
        axes[1, 0].barh(y, available, color="#b7495a")
        axes[1, 0].set_yticks(y, [n.replace("_", " ") for n in names], fontsize=7)
        axes[1, 0].set(
            title=f'I: real rendered camera jump {value["jump_m"]:.2f}m at frame 300',
            xlabel="Post-jump Recall@25cm/10deg (%)",
            xlim=(0, 105),
        )
        for i, n in enumerate(names):
            recovery = value["results"][n]["recovery"]["0.25m_10deg"]
            text = (
                f'{recovery["latency_frames"]} frames to 3 correct'
                if recovery["recovered"]
                else "No 3-frame recovery"
            )
            axes[1, 0].text(2, i, text, va="center", fontsize=7)
    paired = sources.get("J.person.paired_comparison")
    if paired:
        rows = paired["value"]["comparison"]
        x = np.arange(len(rows))
        axes[1, 1].bar(
            x - 0.18,
            [r["baseline"]["ate_se3_aligned_rmse_m"] for r in rows],
            0.36,
            label="RGB-D VO",
        )
        axes[1, 1].bar(
            x + 0.18,
            [r["predicted_person_exclusion"]["ate_se3_aligned_rmse_m"] for r in rows],
            0.36,
            label="Frozen predicted-person exclusion",
        )
        axes[1, 1].set_xticks(
            x, [r["sequence"].replace("_", "\n") for r in rows], fontsize=8
        )
        axes[1, 1].set(
            title="J: real TUM people, four complete sequences",
            ylabel="SE3-aligned ATE RMSE (m)",
        )
        axes[1, 1].legend(fontsize=7)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "J predicted-person comparison awaiting full validation\nNo accuracy result filled in",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    fig.suptitle(
        "Actual accepted outputs | same-building synthetic H/I; real RGB-D VO J\nHeuristic weights, no pose covariance or joint streaming FPS claim",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    temp = target.with_suffix(".tmp.png")
    fig.savefig(temp, dpi=150)
    temp.replace(target)
    plt.close(fig)
    atomic(
        cache,
        dict(input_sha256=key, sources={k: v["sha256"] for k, v in sources.items()}),
    )
