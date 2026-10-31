"""Actual upstream analytic Jacobian versus independent SE3 finite differences."""

import argparse, ast, hashlib, json, sys, types, typing
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import torch, pypose as pp


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])


def main(a):
    source = Path(a.source) / "Module/Optimization/TwoFramePGO/Graphs.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Analytic_ReprojDisp_TwoFramePGO"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_jacobian"
    )
    method.decorator_list = []
    ns = {"torch": torch, "pp": pp}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    build = ns["build_jacobian"]
    z = np.load(Path(a.run) / "observations.npz")
    pose = pp.SE3(torch.from_numpy(z["frames//pose"]).double())
    T = pose.matrix().numpy()
    target = z["edge/match2frame2/mapping"]
    world = z["points//pos_Tw"][z["edge/match2point/mapping"]]
    reports = []
    for frame in [10, 50, 100, 150, 199]:
        idx = np.flatnonzero(target == frame)[:100]
        points = world[idx].astype(float)
        K = z["frames//K"][frame].astype(float)
        baseline = float(z["frames//baseline"][frame])
        P = T[frame]

        def residual(t):
            pc = (points - t[:3, 3]) @ t[:3, :3]
            return np.stack(
                [
                    K[0, 0] * pc[:, 1] / pc[:, 0] + K[0, 2],
                    K[1, 1] * pc[:, 2] / pc[:, 0] + K[1, 2],
                    baseline * K[0, 0] / pc[:, 0],
                ],
                1,
            ).ravel()

        pc = (points - P[:3, 3]) @ P[:3, :3]
        obj = types.SimpleNamespace(
            pos_Tc=torch.from_numpy(pc),
            pos_Tw=torch.from_numpy(points),
            K=torch.from_numpy(K),
            baseline=torch.tensor(baseline, dtype=torch.double),
            pose2opt=pose[frame : frame + 1],
        )
        analytic = build(obj).numpy()[:, :6]
        numeric = np.zeros_like(analytic)
        h = 1e-6
        for k in range(6):
            tangent = np.zeros((4, 4))
            if k < 3:
                tangent[k, 3] = h
            else:
                r = np.zeros(3)
                r[k - 3] = h
                tangent[:3, :3] = skew(r)
            numeric[:, k] = (
                residual(expm(tangent) @ P) - residual(expm(-tangent) @ P)
            ) / (2 * h)
        rel = float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))
        reports.append(
            dict(
                frame=frame,
                points=len(points),
                max_absolute_error=float(np.max(np.abs(analytic - numeric))),
                relative_frobenius_error=rel,
            )
        )
        assert rel < 1e-7, reports[-1]
    report = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method="actual upstream AST build_jacobian; independent scipy.linalg.expm left SE3 perturbation, central difference step1e-6",
        tangent_order="tx ty tz rx ry rz; left perturbation in world frame",
        cases=reports,
        status="passed",
    )
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("output")
    p.add_argument("--source", required=True)
    main(p.parse_args())
