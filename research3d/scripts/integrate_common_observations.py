"""Actual H system ablations on the SAME immutable map and 600 stereo frames.
Precomputed RGB observations are predictions and reference geometry only. The
online estimator never receives GT; evaluation begins after every trajectory saves.
"""

import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion.common_observations import estimate


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(a):
    common = Path(a.common)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((common / "acceptance.json").read_text())
    n = meta["continuous_stereo_frames"]
    assert n == 600
    digest = sha(common / "readonly_prior_map.npz")
    assert digest == meta["readonly_map_sha256"]
    ace = np.load(Path(a.ace_run) / "predictions.npz")
    quality = list(csv.DictReader((Path(a.ace_run) / "per_frame.csv").open()))
    assert len(ace["frame"]) == len(quality) == n
    observations = {"ace": []}
    for T, r in zip(ace["T_world_camera"], quality):
        residual = (
            float(r["median_reprojection_px"]) if r["median_reprojection_px"] else None
        )
        observations["ace"].append(
            [
                dict(
                    T_world_camera=T.tolist(),
                    inliers=int(r["inliers"]),
                    median_reprojection_px=residual,
                )
            ]
        )
    input_hashes = {}
    for name, directory in [("bow", Path(a.bow_run)), ("clip", Path(a.clip_run))]:
        protocol = json.loads((directory / "map_protocol.json").read_text())
        queries = json.loads((directory / "queries.json").read_text())
        online = json.loads((directory / "online_results.json").read_text())
        assert len(queries) == len(online) == n
        assert all(
            "T_world_camera" not in q and "depth" not in q and "label" not in q
            for q in queries
        )
        assert all(int(row["frame_id"]) == i for i, row in enumerate(online))
        for i, q in enumerate(queries):
            assert Path(q["rgb"]).name == f"{i:06d}_left.png"
            assert (
                Path(q["rgb"]).resolve()
                == (common / "stereo/image_left" / f"{i:06d}_left.png").resolve()
            ), "Observation belongs to another continuous sequence"
            assert np.allclose(q["K"], meta["K"]), "Query calibration mismatch"
            assert str(ace["frame"][i]) == f"frame-{i:06d}.color.png"
        assert sha(directory / "readonly_map.npz") == protocol["map_sha256"]
        input_hashes[name] = dict(
            map_sha256=protocol["map_sha256"],
            observations_sha256=sha(directory / "online_results.json"),
        )
        for mode in ["geometry", "semantic", "soft"]:
            observations[name + "_" + mode] = [
                [r["hypotheses"][mode]] if r["hypotheses"][mode] else [] for r in online
            ]
            observations[name + "_" + mode + "_multi"] = [
                r["candidate_pools"][mode] for r in online
            ]
    assert (
        input_hashes["bow"]["map_sha256"] == input_hashes["clip"]["map_sha256"]
    ), "Retrieval ablation must use exact shared feature map"
    data = np.load(Path(a.macvo_run) / "trajectory.npz")
    T = data["T_world_camera_raw"]
    lost = data["lost_before_interpolation"]
    assert len(T) == n
    relative = np.repeat(np.eye(4)[None], n, 0)
    relative[1:] = np.linalg.inv(T[:-1]) @ T[1:]
    valid = np.r_[False, ~(lost[:-1] | lost[1:])]
    variants = {
        "ACE_gate": dict(modules=["ace"]),
        "geometry_gate": dict(modules=["bow_geometry"]),
        "clip_geometry_gate": dict(modules=["clip_geometry"]),
        "semantic_hard_gate": dict(modules=["bow_semantic"]),
        "semantic_soft_gate": dict(modules=["bow_soft"]),
        "clip_semantic_soft_gate": dict(modules=["clip_soft"]),
        "ACE_geometry_gate": dict(modules=["bow_geometry", "ace"]),
        "ACE_clip_soft_weighted": dict(modules=["clip_soft", "ace"], weighted=True),
        "ACE_clip_soft_multicandidate_weighted": dict(
            modules=["clip_soft_multi", "ace"], multiple_candidates=True, weighted=True
        ),
    }
    # Freeze protocol and hashes before any online estimates or GT access.
    protocol = dict(
        frames=n,
        variants=variants,
        parameters=dict(
            temporal_consistency_frames=3,
            temporal_translation_m=0.1,
            temporal_rotation_deg=10,
            min_ace_inliers=50,
            min_geometry_inliers=12,
            selection_distance_scales=[0.25, 15],
            weight="clip(1/(1+median reprojection pixels),0.25,1); three-consistent-observation recovery resets gauge",
        ),
        parent_map_sha256=digest,
        observation_inputs=input_hashes,
        GT_online=False,
        pose_covariance_available=False,
        execution_mode="Offline replay of actual cached per-image observations and completed two-frame motion estimates; continuous ordered state updates, not combined-model streaming FPS/latency on this GPU",
        correlated_RGB_modules=True,
        semantic_generalization_claim=json.loads(
            Path(a.dataset_acceptance).read_text()
        )["semantic_generalization_claim"],
        interpretation="Explicit heuristic reliability/correction weights, not calibrated uncertainty or independent-sensor fusion; fixed variants, no test-chosen winner",
    )
    atomic_json(out / "protocol.json", protocol)
    trajectories = {}
    diagnostics = {}
    runtimes = {}
    for name, options in variants.items():
        tick = time.perf_counter()
        P, rows = estimate(relative, valid, observations, **options)
        runtimes[name] = time.perf_counter() - tick
        trajectories[name] = P
        diagnostics[name] = rows
    temporary = out / "global_predictions.tmp.npz"
    np.savez_compressed(temporary, **trajectories)
    temporary.replace(out / "global_predictions.npz")
    atomic_json(out / "online_diagnostics.json", diagnostics)
    assert sha(common / "readonly_prior_map.npz") == digest
    # Offline evaluation boundary.
    truth = np.load(common / "ground_truth_eval_only.npz")["T_world_camera"]
    results = {}
    for name, P in trajectories.items():
        available = np.isfinite(P).all((1, 2))
        te = np.full(n, np.inf)
        re = te.copy()
        te[available] = np.linalg.norm(
            P[available, :3, 3] - truth[available, :3, 3], axis=1
        )
        re[available] = np.degrees(
            Rotation.from_matrix(
                P[available, :3, :3].transpose(0, 2, 1) @ truth[available, :3, :3]
            ).magnitude()
        )
        rows = diagnostics[name]
        incorrect_observation_updates = 0
        for i, row in enumerate(rows):
            if not row["accepted_absolute"]:
                continue
            observation = np.array(row["selected_absolute_pose"])
            ot = np.linalg.norm(observation[:3, 3] - truth[i, :3, 3])
            ore = np.degrees(
                Rotation.from_matrix(
                    observation[:3, :3].T @ truth[i, :3, :3]
                ).magnitude()
            )
            incorrect_observation_updates += int(ot > 0.25 or ore > 10)
        results[name] = dict(
            frames=n,
            valid_pose_fraction=float(available.mean()),
            absolute_map_translation_rmse_valid_only_m=(
                float(np.sqrt(np.mean(te[available] ** 2))) if available.any() else None
            ),
            recall_5cm_5deg=float(np.mean((te <= 0.05) & (re <= 5))),
            recall_25cm_10deg=float(np.mean((te <= 0.25) & (re <= 10))),
            recall_1m_30deg=float(np.mean((te <= 1) & (re <= 30))),
            continuous_localization_availability=float(
                np.mean([r["status"] in ["TRACKING", "DEAD_RECKONING"] for r in rows])
            ),
            lost_status_frames=sum(r["status"] == "LOST" for r in rows),
            accepted_updates=sum(r["accepted_absolute"] for r in rows),
            false_accepted_updates_over25cm=sum(
                bool(r["accepted_absolute"] and te[i] > 0.25)
                for i, r in enumerate(rows)
            ),
            false_accepted_updates_25cm_10deg=sum(
                bool(r["accepted_absolute"] and (te[i] > 0.25 or re[i] > 10))
                for i, r in enumerate(rows)
            ),
            incorrect_selected_observation_updates_25cm_10deg=incorrect_observation_updates,
            false_accepted_updates_scope=(
                "Legacy false_accepted_updates fields count incorrect corrected STATE at update frames; "
                "incorrect_selected_observation_updates counts the selected raw absolute observation before blending. "
                "These differ under weighted correction."
            ),
            runtime_fusion_only_ms_per_frame=runtimes[name] * 1000 / n,
            translation_errors_m=[float(x) if np.isfinite(x) else None for x in te],
        )
    result = dict(
        status="completed",
        audit=dict(
            frames=n,
            GT_online=False,
            parent_map_unchanged=True,
            exact_shared_geometry_map=True,
            all_variants_same_queries=True,
            simulated=True,
        ),
        results=results,
    )
    atomic_json(out / "summary.json", result)
    print(
        json.dumps(
            {
                name: {k: v for k, v in r.items() if k != "translation_errors_m"}
                for name, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("common")
    p.add_argument("ace_run")
    p.add_argument("macvo_run")
    p.add_argument("bow_run")
    p.add_argument("clip_run")
    p.add_argument("output")
    p.add_argument("--dataset-acceptance", required=True)
    main(p.parse_args())
