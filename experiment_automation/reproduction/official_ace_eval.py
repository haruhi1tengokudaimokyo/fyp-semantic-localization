"""Execute upstream test_ace.py unchanged and audit its real per-query outputs."""

import argparse, hashlib, json, subprocess, sys, time, os
from pathlib import Path
import numpy as np


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(a.source)
    dataset = Path(a.dataset)
    model = Path(a.model)
    model_link = out / model.name
    if model_link.exists():
        assert model_link.resolve() == model.resolve()
    else:
        model_link.symlink_to(model.resolve())
    command = [
        sys.executable,
        str(source / "test_ace.py"),
        str(dataset),
        str(model_link),
        "--session",
        "reproduction",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(source / "dsacstar") + os.pathsep + env.get("PYTHONPATH", "")
    )
    preflight = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,dsacstar,json; assert hasattr(dsacstar,'forward_rgb'); print(json.dumps({'compiled_module':dsacstar.__file__,'forward_rgb':True}))",
        ],
        env=env,
        cwd=out,
        capture_output=True,
        text=True,
        check=True,
    )
    (out / "dsac_module_preflight.json").write_text(preflight.stdout)
    tick = time.monotonic()
    with (out / "official_test.log").open("w") as log:
        subprocess.run(
            command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, env=env
        )
    files = list(out.glob("poses_*_reproduction.txt"))
    assert len(files) == 1
    rows = [line.split() for line in files[0].read_text().splitlines() if line.strip()]
    expected = sorted(p.name for p in (dataset / "test/rgb").iterdir())
    assert sorted(r[0] for r in rows) == expected
    # Official output: name qw qx qy qz tx ty tz rErr tErr inliers.
    assert all(len(r) == 11 for r in rows), "Upstream output format changed"
    values = np.array([[float(x) for x in r[1:]] for r in rows])
    assert np.isfinite(values).all()
    rotation = values[:, 7]
    translation = values[:, 8]
    summary = dict(
        status="completed",
        n=len(rows),
        upstream_script_unchanged=True,
        model=str(model),
        dataset=str(dataset),
        command=command,
        pose_version="original D-SLAM, official dataset RGB convention; not Pseudo-GT",
        median_rotation_deg=float(np.median(rotation)),
        median_translation_m=float(np.median(translation)),
        recall_5cm_5deg=float(np.mean((translation < 0.05) & (rotation < 5))),
        elapsed_seconds=time.monotonic() - tick,
        upstream_script_sha256=hashlib.sha256(
            (source / "test_ace.py").read_bytes()
        ).hexdigest(),
        query_pose_output_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
        inference_mean_seconds_from_upstream=float(
            (out / ("test_" + dataset.name + "_reproduction.txt"))
            .read_text()
            .split()[-1]
        ),
    )
    temp = out / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2))
    temp.replace(out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("dataset")
    p.add_argument("model")
    p.add_argument("output")
    main(p.parse_args())
