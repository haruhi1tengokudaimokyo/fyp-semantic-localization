"""Controlled actual MACVO backend replay from fixed learned correspondences.
No GT until both complete trajectories are estimated. Same measurements, static
motion initial guess, official robust kernel and LM scheduler; only residual
covariance changes. Sequential execution prevents spawn losing an ablation patch.
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, pypose as pp
from types import SimpleNamespace


def main(a):
    sys.path.insert(0, a.source)
    from Module.Map import MatchObs, PointNode
    from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
    from Module.Optimization.TwoFramePGO.Graphs import (
        GraphInput,
        Analytic_ReprojDisp_TwoFramePGO,
    )
    from Utility.Point import pixel2point_NED
    from motion.metrics import evaluate

    root = Path(a.run)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(root / "observations.npz")
    target = z["edge/match2frame2/mapping"]
    n = len(z["frames//pose"])
    torch.set_num_threads(4)
    calls = {"identity": 0, "covariance": 0}

    class IdentityGraph(Analytic_ReprojDisp_TwoFramePGO):
        def covariance_array(self):
            c = super().covariance_array()
            v = torch.eye(3, dtype=c.dtype, device=c.device).expand_as(c).clone()
            assert torch.equal(
                v, torch.eye(3, dtype=c.dtype, device=c.device).expand_as(c)
            )
            calls["identity"] += 1
            return v

    class CovarianceGraph(Analytic_ReprojDisp_TwoFramePGO):
        def covariance_array(self):
            c = super().covariance_array()
            assert not torch.equal(
                c, torch.eye(3, dtype=c.dtype, device=c.device).expand_as(c)
            )
            calls["covariance"] += 1
            return c

    rows = {}
    poses = {}
    for mode, cls in [("covariance", CovarianceGraph), ("identity", IdentityGraph)]:
        context = TwoFrame_PGO.init_context(
            SimpleNamespace(
                autodiff=False, graph_type="disp", device="cpu", vectorize=True
            )
        )
        context["pose_graph_class"] = cls
        trajectory = [pp.identity_SE3().tensor().numpy()]
        runtime = []
        for j in range(1, n):
            indices = np.flatnonzero(target == j)
            assert np.all(z["edge/match2frame1/mapping"][indices] == j - 1)
            fields = {
                k.removeprefix("match//"): torch.from_numpy(z[k][indices].copy())
                for k in z.files
                if k.startswith("match//")
            }
            obs = MatchObs.init(fields)
            previous = pp.SE3(torch.from_numpy(trajectory[-1]).float())
            K = torch.from_numpy(z["frames//K"][j].copy())
            p0 = pixel2point_NED(fields["pixel1_uv"], fields["pixel1_d"][:, 0], K)
            world = pp.SE3_type.Act(previous, p0)[..., :3]
            R = previous.rotation().matrix().double().expand(len(indices), 3, 3)
            cov = R @ fields["obs1_covTc"] @ R.transpose(-1, -2)
            points = PointNode.init(
                dict(
                    pos_Tw=world,
                    cov_Tw=cov,
                    color=torch.zeros((len(indices), 3), dtype=torch.uint8),
                )
            )
            data = GraphInput(
                torch.tensor([j]),
                torch.tensor([j - 1]),
                previous.unsqueeze(0),
                torch.from_numpy(z["frames//baseline"][j : j + 1].copy()),
                obs,
                points,
                K,
                torch.zeros(len(indices), dtype=torch.long),
                "cpu",
            )
            start = time.perf_counter()
            context, result = TwoFrame_PGO._optimize(context, data)
            runtime.append(time.perf_counter() - start)
            trajectory.append(result.motion[0].detach().float().tensor().numpy())
            if j % 50 == 0:
                print(mode, j, "weight_calls", calls[mode], flush=True)
        poses[mode] = pp.SE3(torch.from_numpy(np.stack(trajectory))).matrix().numpy()
        rows[mode] = dict(
            backend_wall_seconds=float(sum(runtime)),
            weight_matrix_evaluations=calls[mode],
        )
        assert calls[mode] >= n - 1
    # GT is accessed after BOTH online-style backend replays finish.
    truth = np.load(root / "trajectory.npz")
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    gt = C @ truth["T_world_camera_gt"] @ C.T
    original = C @ truth["T_world_camera_raw"] @ C.T
    for mode in rows:
        rows[mode]["metrics"] = evaluate(poses[mode], gt)
    difference = float(np.max(np.abs(poses["covariance"] - original)))
    changed = float(np.max(np.abs(poses["identity"] - poses["covariance"])))
    assert changed > 1e-6, "Ablation did not alter trajectory"
    rows["audit"] = dict(
        fixed_correspondences=len(target),
        frames=n,
        original_weighted_replay_max_matrix_difference=difference,
        identity_vs_covariance_max_matrix_difference=changed,
        identity_applied_inside_actual_solver=True,
        frontend_keypoint_filter_same=True,
        GT_used_during_solve=False,
        scope="backend covariance weighting only; frontend covariance-based selection remains on",
        runtime_caveat="backend replay only, GPU training may coexist; not an end-to-end timing benchmark",
    )
    np.savez_compressed(
        out / "trajectories.npz",
        T_world_camera_covariance=C.T @ poses["covariance"] @ C,
        T_world_camera_identity=C.T @ poses["identity"] @ C,
        T_world_camera_gt=truth["T_world_camera_gt"],
    )
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("output")
    p.add_argument("--source", required=True)
    main(p.parse_args())
