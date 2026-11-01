"""Audit pinned official OMCL engineering prerequisites and online pose inputs.

Passing this audit means evidence was collected, not that OMCL localization ran.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def audit(source, output):
    source, output = Path(source).resolve(), Path(output)
    pf = source / "src/omcl/models/pf.py"
    tree = ast.parse(pf.read_text())
    uses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"estimate_odoms", "init_particles"}:
                uses.append(
                    dict(
                        function=node.func.id,
                        line=node.lineno,
                        expression=ast.unparse(node),
                    )
                )
    gt_motion = any(
        u["function"] == "estimate_odoms" and "poses44" in u["expression"] for u in uses
    )
    gt_initial = any(
        u["function"] == "init_particles" and "poses44" in u["expression"] for u in uses
    )
    assert (
        gt_motion and gt_initial
    ), "Upstream input flow changed; manual audit required"
    tracked = [
        "README.md",
        "DATA.md",
        "docker/Dockerfile",
        "src/omcl/models/pf.py",
        "src/omcl/models/odom.py",
        "src/omcl/models/image_encoders/lseg/lseg.py",
    ]
    files = {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in tracked}
    modules = {
        m: importlib.util.find_spec(m) is not None
        for m in [
            "kaolin",
            "MinkowskiEngine",
            "hydra",
            "omegaconf",
            "viser",
            "clip",
            "timm",
        ]
    }
    # Genuine import probe in the selected research environment, no model inference.
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import omcl.models.pf",
            str(source / "src"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = dict(
        status="audit_completed",
        official_localization_executed=False,
        source_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_files_sha256=files,
        online_pose_input_calls=uses,
        official_example_derives_odometry_from_dataset_poses=gt_motion,
        official_example_initializes_particles_around_dataset_pose=gt_initial,
        direct_example_eligible_for_GT_free_project=False,
        required_adapter="Replace estimate_odoms(poses44) with measured MACVO relative motion and replace GT-centred initialization with map-only global initialization. Preserve separate evaluator.",
        modules_available=modules,
        cuda_compiler=shutil.which("nvcc"),
        import_probe=dict(exit_code=probe.returncode, stderr=probe.stderr[-4000:]),
        checkpoint_found=(
            source / "src/omcl/models/image_encoders/lseg/checkpoints/demo_e200.ckpt"
        ).is_file(),
        sources=[
            "https://arxiv.org/abs/2512.15557",
            "https://github.com/AIS-Bonn/omcl",
        ],
        scope="Pinned implementation audit; not a statement that all OMCL paper experiments use ground-truth odometry, and not an OMCL accuracy result",
    )
    output.mkdir(parents=True, exist_ok=True)
    temp = output / "summary.json.tmp"
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(output / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    audit(args.source, args.output)
