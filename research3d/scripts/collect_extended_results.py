"""Package actual P2/P3/P4 evidence; never count missing runs as completed."""

import argparse, datetime, hashlib, json, shutil, subprocess
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def main(a):
    base = Path(a.base)
    repo = Path(a.repo)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = base / "runs"
    rr = repo / "research3d/results"
    report = {}

    def copy(src, dst):
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    for name in [
        "geometric_200",
        "allfp32_cov_200",
        "weight_replay_200",
        "bf16_smoke_reshape",
        "fp32_smoke_10",
        "allfp32_smoke_3",
        "paper_cov_200",
    ]:
        d = runs / "macvo" / name
        for f in [
            "summary.json",
            "protocol.json",
            "checkpoint_audit.json",
            "observation_audit.json",
            "residual_audit.json",
            "trajectory.npz",
            "trajectories.npz",
            "relative_motion.npz",
            "residuals.npz",
            "per_frame.json",
            "failure.json",
        ]:
            copy(d / f, rr / "p2" / name / f)
        for f in ["observations.npz"]:
            copy(d / f, out / "evidence/p2" / name / f)
    for f in [
        "covariance_monte_carlo.json",
        "jacobian_check.json",
        "dataset_acceptance.json",
        "all_tests.log",
        "real_data_tests_ace_env.log",
    ]:
        copy(runs / "macvo" / f, rr / "p2" / f)
    copy(
        runs / "macvo/allfp32_identity_200/INVALID_ABLATION.json",
        rr / "p2/invalid_spawn_ablation.json",
    )
    for name in [
        "openloris_office1_1",
        "openloris_office1_3",
        "openloris_office1_1_wheel3d",
        "openloris_office1_3_wheel3d",
        "openloris_office1_1_wheel3d_outage",
    ]:
        d = runs / name
        for f in [
            "audit.json",
            "comparison.json",
            "common_window_comparison.json",
            "execution.json",
            "variant.json",
        ]:
            copy(d / f, rr / "p2" / name / f)
        for mode in ["imu_wheel", "stereo_imu", "stereo_imu_wheel"]:
            for p in (d / mode).glob("*"):
                if p.suffix in [".yaml", ".json"] or p.name in [
                    "trajectory.txt",
                    "timing.txt",
                ]:
                    copy(p, rr / "p2" / name / mode / p.name)
            copy(d / mode / "run.log", out / "evidence/p2" / name / mode / "run.log")
    for name in [
        "segformer_b0_40epochs",
        "knownmap_relocalization",
        "hloc_knownmap_relocalization",
        "clip_hloc_relocalization",
        "clip_reference",
    ]:
        d = runs / "semantic" / name
        phase = "p4" if "clip" in name else "p3"
        for p in d.glob("*"):
            if p.suffix == ".json":
                copy(p, rr / phase / name / p.name)
        for p in d.glob("*.npz"):
            copy(p, out / "evidence" / phase / name / p.name)
    copy(
        runs / "semantic/paired_cluster_bootstrap.json",
        rr / "p4/paired_cluster_bootstrap.json",
    )
    copy(
        base / "datasets/hssd_semantic_pilot/manifest.json",
        rr / "p3/semantic_data_manifest.json",
    )
    copy(
        runs / "semantic/segformer_b0_40epochs/best.pt",
        out / "models/segformer_b0_hssd_partial_best.pt",
    )
    copy(base / "models/mit-b0/SegFormer_LICENSE", out / "models/SegFormer_LICENSE.txt")
    copy(base / "models/mit-b0/acquisition.json", rr / "p3/pretrained_acquisition.json")
    for name in ["office1-1", "office1-3"]:
        copy(
            base / "datasets/openloris" / (name + ".acquisition.json"),
            rr / "p2" / (name + "_acquisition.json"),
        )
    copy(
        base / "downloads/macvo_demo_range.zip.json",
        rr / "p2/macvo_demo_acquisition.json",
    )
    versions = {}
    for name in ["MAC-VO", "MINS", "hloc", "LightGlue", "omcl", "CLIP"]:
        source = base / "sources" / name
        versions[name] = dict(
            commit=subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            remote=subprocess.check_output(
                ["git", "-C", str(source), "remote", "get-url", "origin"], text=True
            ).strip(),
        )
    versions["MAC-VO"]["submodules"] = subprocess.check_output(
        ["git", "-C", str(base / "sources/MAC-VO"), "submodule", "status"], text=True
    ).splitlines()
    versions["notes"] = (
        "MAC-VO one-line reshape compatibility patch is stored separately. hloc frontend modules used on known RGB-D reference geometry; no claim of full hloc SfM or OMCL reproduction."
    )
    (repo / "research3d/docs/p2/upstream_versions_extended.json").write_text(
        json.dumps(versions, indent=2)
    )
    report["source_versions"] = versions
    report["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["completed_evidence"] = {}
    for key, p in [
        ("macvo_fast", rr / "p2/allfp32_cov_200/summary.json"),
        ("macvo_paper", rr / "p2/paper_cov_200/summary.json"),
        ("backend_weight_ablation", rr / "p2/weight_replay_200/summary.json"),
        ("semantic_training", rr / "p3/segformer_b0_40epochs/summary.json"),
        (
            "geometry_semantic_localization",
            rr / "p3/hloc_knownmap_relocalization/summary.json",
        ),
        ("clip_route_localization", rr / "p4/clip_hloc_relocalization/summary.json"),
    ]:
        report["completed_evidence"][key] = dict(
            completed=p.exists(), path=str(p.relative_to(repo))
        )
    report["limitations"] = [
        "P2 learned MACVO-to-MINS uncertainty fusion not implemented; no invented pose covariance",
        "P2 real pedestrian dynamic masks not evaluated; camera packet outage only",
        "P3 partial furniture/object labels, three scenes; physical building independence unverified",
        "P4 pooled CLIP reference-feature baseline; official OMCL octree/particle-filter reproduction not completed",
        "P5/P6 full temporal recovery and large multi-building dynamic benchmark not completed",
    ]
    (out / "RELEASE_P2_P4.json").write_text(json.dumps(report, indent=2))
    shutil.copytree(rr / "p2", out / "evidence/p2", dirs_exist_ok=True)
    shutil.copytree(rr / "p3", out / "evidence/p3", dirs_exist_ok=True)
    shutil.copytree(rr / "p4", out / "evidence/p4", dirs_exist_ok=True)
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256.json":
            rows.append(
                dict(
                    path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)
                )
            )
    (out / "SHA256.json").write_text(json.dumps(rows, indent=2))
    print("packaged", len(rows), "files", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/data/fyp_3d")
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
