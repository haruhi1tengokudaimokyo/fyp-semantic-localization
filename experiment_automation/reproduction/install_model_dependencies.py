"""Isolated remaining encoder/compiler dependencies with import receipts."""

import argparse, json, subprocess, sys, time
from pathlib import Path


def main(paper, output):
    assert (
        Path(sys.prefix).resolve()
        == (Path("/data/fyp_3d/envs") / paper.lower()).resolve()
    )
    packages = (
        ["timm==0.4.12", "pandas", "ftfy", "regex", "open3d"]
        if paper == "OMCL"
        else ["nvidia-cuda-nvcc-cu12==12.9.86", "nvidia-nvjitlink-cu12==12.9.86"]
    )
    tick = time.monotonic()
    command = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + packages
    subprocess.run(command, check=True, timeout=1800)
    result = dict(
        status="dependencies_installed",
        paper_id=paper,
        command=command,
        elapsed_seconds=time.monotonic() - tick,
        localization_executed=False,
        freeze=subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--local"], text=True
        ).splitlines(),
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
