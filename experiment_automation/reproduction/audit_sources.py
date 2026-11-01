"""Pinned paper/code/input audit; an audit receipt never claims localization success."""

import argparse, ast, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

BASE = Path("/data/fyp_3d")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def main(paper, output):
    names = {"MACVO": "MAC-VO", "ACE": "ace", "SG2Loc": "sg2loc", "OMCL": "omcl"}
    root = BASE / "sources" / names[paper]
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".yaml", ".yml", ".sh", ".md", ".txt")
        and ".git" not in p.parts
    ]
    files = {str(p.relative_to(root)): sha(p) for p in paths}
    evidence = []
    if paper == "SG2Loc":
        file = root / "sg2loc/particle_filter/filter.py"
        for n, line in enumerate(file.read_text().splitlines(), 1):
            if "query_image_poses" in line or "known_axis" in line:
                evidence.append(
                    dict(path=str(file.relative_to(root)), line=n, text=line.strip())
                )
        modules = ["numba", "torch_geometric", "poselib", "open3d", "yacs", "romav2"]
        probe = "import sg2loc.particle_filter.filter"
        condition = "Official code uses pose-derived relative motion and per-frame gravity axes. Strict paper-condition and GT-free online runs must be separate; no default claim of GT-free original method."
    elif paper == "OMCL":
        for rel in [
            "src/omcl/models/pf.py",
            "src/omcl/utils/mics.py",
            "data_scripts/matterport/simulation.py",
        ]:
            for n, line in enumerate((root / rel).read_text().splitlines(), 1):
                if any(
                    s in line
                    for s in [
                        "estimate_odoms(poses44)",
                        "init_particles(poses44",
                        "poses.txt",
                        "agent_state.position",
                        "agent_state.rotation",
                    ]
                ):
                    evidence.append(dict(path=rel, line=n, text=line.strip()))
        modules = ["kaolin", "hydra", "omegaconf", "viser", "MinkowskiEngine"]
        probe = "import omcl.models.pf"
        condition = "Official example uses dataset-pose odometry and GT-centred tracking initialization; assess against paper before calling this a defect. GT-free global initialization is a separate condition."
    elif paper == "ACE":
        modules = ["torch", "dsacstar", "joblib", "skimage"]
        probe = "import torch;import dsacstar"
        condition = "Calibrated original D-SLAM poses and Pseudo-GT must never mix; upstream defaults: 8M buffer,16 epochs,5120 batch,480px grayscale."
        evidence = [
            dict(
                path="datasets/setup_7scenes.py",
                detail="Default calibrated: T_world_rgb=T_world_depth@inv(d_to_rgb); raw original depth-camera poses are a distinct option.",
            ),
            dict(
                path="datasets/setup_12scenes.py",
                detail="Official split.txt interval; skip invalid INF poses; focal average from info.txt; no rendered depth with PGT.",
            ),
        ]
    else:
        modules = ["torch", "pypose", "evo", "cv2"]
        probe = "from Evaluation.MetricsSeq import evaluateATE,evaluateRTE,evaluateROE,evaluateRPE"
        condition = "Paper_Reproduce.yaml and original frontend/motion weights; FP32 is hardware adaptation; full physical sequence with stereo rectification and GT only offline."
        evidence = [
            dict(
                path="DataLoader/Dataset/EuRoC.py",
                detail="Audit YAML distortion vs hard-coded distortion, stereo rectification R1/R2, baseline P2/fx, timestamp mask and body-to-rectified-camera GT.",
            )
        ]
    available = {m: importlib.util.find_spec(m) is not None for m in modules}
    import_path = str(root / "src") if paper == "OMCL" else str(root)
    code = "import sys;sys.path[:0]=sys.argv[1:];" + probe
    extra = [str(root / "dsacstar")] if paper == "ACE" else []
    p = subprocess.run(
        [sys.executable, "-c", code, import_path] + extra,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = dict(
        status="audited",
        paper_id=paper,
        localization_executed=False,
        source_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            )
        ),
        source_files_sha256=files,
        input_condition=condition,
        input_evidence=evidence,
        environment=sys.prefix,
        modules_available=available,
        import_probe=dict(exit_code=p.returncode, stderr=p.stderr[-5000:]),
        authorized_scripts_found=[
            str(p)
            for parent in [
                BASE / "downloads",
                BASE / "datasets",
                Path("/home/csy/.codex/attachments"),
            ]
            for p in parent.rglob("*.py")
            if any(s in p.name.lower() for s in ["download-scannet", "download_3rscan"])
        ],
        missing_authorization_is_not_permission_to_download_restricted_derivatives=True,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "source_files_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paper")
    p.add_argument("output")
    a = p.parse_args()
    main(a.paper, a.output)
