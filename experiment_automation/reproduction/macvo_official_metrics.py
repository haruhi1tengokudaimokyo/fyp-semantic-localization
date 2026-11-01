"""Run original MACVO evaluation functions on preserved full provided-demo arrays.
No network or trajectory recomputation, no fabricated parent benchmark identity.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np


def main(output):
    overlay = Path("/data/fyp_3d/envs/macvo_metrics_overlay")
    if "--historical-evo" in sys.argv:
        sys.path.insert(0, str(overlay))
    import evo
    from evo.core import lie_algebra

    # Explicit recommendation in official MetricsSeq.evaluateROE: serialized
    # FP32 poses need SO(3) tolerance 1e-4. Process-local, no library edit.
    def stored_so3(r):
        return bool(
            np.allclose(r.T @ r, np.eye(3), atol=1e-4, rtol=0)
            and np.isclose(np.linalg.det(r), 1.0, atol=1e-4, rtol=0)
        )

    lie_algebra.is_so3 = stored_so3

    sys.path.insert(0, "/data/fyp_3d/sources/MAC-VO")
    from Evaluation.MetricsSeq import evaluateATE, evaluateRTE, evaluateROE, evaluateRPE
    from evo.core.trajectory import PosePath3D

    root = Path("/data/fyp_3d/runs/formal/macvo")
    results = {}
    for name, directory in [
        ("Paper_FP32", "demo_paper_1274"),
        ("Fast_FP32", "demo_fast_1274"),
        ("SGBM_LK_PnP", "demo_stereo_1274"),
    ]:
        path = root / directory
        z = np.load(path / "trajectory.npz")
        estimate = (
            z["T_world_camera_raw"]
            if "T_world_camera_raw" in z.files
            else z["T_world_camera"]
        )
        rotations = estimate[:, :3, :3]
        assert all(
            stored_so3(r) for r in rotations
        ), "Trajectory exceeds upstream recommended serialized-pose tolerance"
        orthogonality = float(
            np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))
        )
        truth = z["T_world_camera_gt"]
        lost = z["lost_before_interpolation"]
        assert len(estimate) == len(truth) == 1274 and np.isfinite(estimate).all()
        metrics = {}
        for metric, function in [
            ("ATE_m", evaluateATE),
            ("t_rel_m_per_frame", evaluateRTE),
            ("r_rel_deg_per_frame", evaluateROE),
            ("RPE_full_transformation", evaluateRPE),
        ]:
            result = function(
                PosePath3D(poses_se3=truth.copy()),
                PosePath3D(poses_se3=estimate.copy()),
                correct_scale=False,
            )
            metrics[metric] = {k: float(v) for k, v in result.stats.items()}
        source = json.loads((path / "summary.json").read_text())
        results[name] = dict(
            frames=1274,
            raw_pose_max_orthogonality_error=orthogonality,
            metrics=metrics,
            lost_frames=int(lost[1:].sum()),
            runtime_context=source.get("frontend_cache"),
            source=str(path),
            sequence="full provided P001_select demo, not established as complete parent benchmark trajectory",
        )
    summary = dict(
        status="completed",
        evo_version=evo.__version__,
        stored_pose_SO3_atol=1e-4,
        precision_adaptation="Process-local upstream-recommended SO3 check tolerance, original arrays unchanged; no shared library edit",
        alignment="SE3 Umeyama, no scale correction; historical evo precedence for upstream redundant align_origin flag",
        upstream_evaluation_functions_used=True,
        GPU_recomputed=False,
        results=results,
        dataset="Author-provided selected TartanAir v2 demo P001_select",
        paper_row_identity_verified=False,
        scope="Paper metric definitions applied to existing evidence; additional original full benchmark trajectories still required",
        unit_note="Paper Eq5 t_rel is mean meters/frame and r_rel degrees/frame; not percent drift per meter. No scale correction.",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, indent=2))
    temp.replace(p)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--historical-evo", action="store_true")
    main(p.parse_args().output)
