"""Validate the actual upstream covariance propagation with Monte Carlo moments.
Extracts named upstream pure functions by AST; does not rewrite the formula.
"""

import argparse, ast, json
from pathlib import Path
import numpy as np
import torch


def main(a):
    source = Path(a.source) / "Module/Covariance/Project2to3.py"
    tree = ast.parse(source.read_text())
    names = {"Covariance_2to3_full", "create_3x3_matrix"}
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    rng = np.random.default_rng(1305)
    rows = []
    for u, v, d, var_u, var_v, cov_uv, var_d in [
        (340, 200, 3, 1.2, 2.0, 0.3, 0.05),
        (500, 350, 12, 3.0, 4.0, -0.8, 0.5),
        (400, 280, 35, 9.0, 12.0, 2.0, 4.0),
    ]:
        tensor = lambda x: torch.tensor([x], dtype=torch.float64)
        expected = (
            scope["Covariance_2to3_full"](
                *map(tensor, [var_u, cov_uv, var_v, var_d, u, v, d]),
                320.0,
                320.0,
                320.0,
                240.0
            )
            .double()
            .numpy()[0]
        )
        uv = rng.multivariate_normal(
            [u, v], [[var_u, cov_uv], [cov_uv, var_v]], size=500000
        )
        depth = rng.normal(d, np.sqrt(var_d), len(uv))
        points = np.stack(
            [depth, (uv[:, 0] - 320) * depth / 320, (uv[:, 1] - 240) * depth / 320], -1
        )
        measured = np.cov(points, rowvar=False)
        error = float(np.linalg.norm(measured - expected) / np.linalg.norm(expected))
        mineig = float(np.linalg.eigvalsh(expected).min())
        assert mineig > 0 and error < 0.015, (mineig, error)
        rows.append(
            dict(
                mean_uv_depth=[u, v, d],
                upstream_cov_ned=expected.tolist(),
                monte_carlo_cov_ned=measured.tolist(),
                relative_frobenius_error=error,
                min_eigenvalue=mineig,
                samples=len(uv),
            )
        )
    result = dict(
        source=str(source),
        scope="Independent Gaussian depth and correlated 2D pixel uncertainty; exact bilinear-product moments, including variance product terms",
        coordinate="NED camera [depth, right, down]",
        pose_covariance_claim=False,
        passed=True,
        tests=rows,
    )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print([(x["mean_uv_depth"], x["relative_frobenius_error"]) for x in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    main(p.parse_args())
