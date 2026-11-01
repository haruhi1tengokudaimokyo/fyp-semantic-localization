"""Install only into isolated paper environments, then record genuine imports."""

import argparse, importlib.util, json, subprocess, sys, time
from pathlib import Path


def main(paper, output):
    expected = Path("/data/fyp_3d/envs") / paper.lower()
    assert (
        Path(sys.prefix).resolve() == expected.resolve()
    ), "Refuse mutation of shared working environments"
    if paper == "SG2Loc":
        packages = [
            "yacs==0.1.8",
            "plyfile==1.1.4",
            "numba==0.65.1",
            "numba-cuda==0.30.3",
            "poselib==2.0.5",
            "torch_geometric==2.8.0.post1",
        ]
        extra = []
        modules = ["numba", "torch_geometric", "poselib", "yacs"]
    else:
        packages = ["kaolin==0.18.0", "hydra-core==1.3.2", "viser", "gdown"]
        extra = [
            "-f",
            "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html",
        ]
        modules = ["kaolin", "hydra", "viser", "gdown"]
        p = expected / "lib/python3.10/site-packages/fyp_clip_source.pth"
        p.write_text("/data/fyp_3d/sources/CLIP\n")
    command = (
        [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + extra + packages
    )
    tick = time.monotonic()
    subprocess.run(command, check=True, timeout=1800)
    probes = {}
    for m in modules:
        p = subprocess.run(
            [sys.executable, "-c", "import " + m],
            capture_output=True,
            text=True,
            timeout=60,
        )
        probes[m] = dict(exit_code=p.returncode, stderr=p.stderr[-3000:])
    result = dict(
        status="environment_prepared",
        paper_id=paper,
        localization_executed=False,
        prefix=sys.prefix,
        command=command,
        elapsed_seconds=time.monotonic() - tick,
        import_probes=probes,
        freeze=subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--local"], text=True
        ).splitlines(),
        caveat="Shared base runtime is read-only; per-paper dependencies isolated. Core imports do not validate CUDA kernels, RoMaV2, checkpoint or benchmark. Torch version differences remain Hardware Adapted until separately validated.",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paper")
    p.add_argument("output")
    a = p.parse_args()
    main(a.paper, a.output)
