"""Compare pinned paper parameters and actual source, without executing a substitute method."""

import argparse, ast, hashlib, json, subprocess
from pathlib import Path
import yaml

BASE = Path("/data/fyp_3d")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(output):
    papers = {
        k: BASE / "downloads" / f"{k}_paper.txt"
        for k in ["macvo", "ace", "sg2loc", "omcl"]
    }
    assert all(p.is_file() for p in papers.values())
    # Assert these transcribed table references against actual downloaded publisher PDFs/text.
    ace = papers["ace"].read_text()
    sg = papers["sg2loc"].read_text()
    om = papers["omcl"].read_text()
    mac = papers["macvo"].read_text()
    assert "97.1%" in ace and "80.8%" in ace and "99.9%" in ace and "99.6%" in ace
    assert "9325" in sg and "9160" in sg and "8725" in sg
    assert "2048" in om and "1024" in om and "0.02" in om
    tree = ast.parse((BASE / "sources/ace/train_ace.py").read_text())
    defaults = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            pass
    expected = {
        "--training_buffer_size": 8000000,
        "--epochs": 16,
        "--batch_size": 5120,
        "--learning_rate_min": 0.0005,
        "--learning_rate_max": 0.005,
    }
    for k, v in expected.items():
        assert defaults[k] == v
    config_path = BASE / "sources/omcl/src/omcl/configs/mp3d_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    result = dict(
        status="audited",
        localization_executed=False,
        paper_text_sha256={k: sha(v) for k, v in papers.items()},
        source_commits={
            k: subprocess.check_output(
                ["git", "-C", str(BASE / "sources" / d), "rev-parse", "HEAD"], text=True
            ).strip()
            for k, d in [
                ("MACVO", "MAC-VO"),
                ("ACE", "ace"),
                ("SG2Loc", "sg2loc"),
                ("OMCL", "omcl"),
            ]
        },
        ACE=dict(
            original_source_defaults=expected,
            source_sha256=sha(BASE / "sources/ace/train_ace.py"),
            paper_table1_recall_percent={
                "7Scenes_SfM": 97.1,
                "7Scenes_DSLAM": 80.8,
                "12Scenes_SfM": 99.9,
                "12Scenes_DSLAM": 99.6,
            },
            comparison_policy="Current D-SLAM branches compare only against D-SLAM column; averages must follow publisher scene aggregation, no pooled 97.1% claim.",
            local_adaptations=[
                "8M cache storage on CPU mmap, not smaller buffer",
                "workers2",
                "Torch optimizer-step compatibility hook",
            ],
            nominal_iterations=25000,
            actual_complete_batches=16 * (8000000 // 5120),
            original_1M_results_separate=True,
        ),
        MACVO=dict(
            paper_table1_EuRoC_average={
                "t_rel_m_per_frame": 0.0024,
                "r_rel_deg_per_frame": 0.0403,
            },
            local_P001_select_has_verified_paper_row=False,
            comparison_policy="Do not compute a reproduction gap between TartanAir selected demo and the EuRoC average. Full original sequences are still required.",
            official_evaluation_compatibility="evo1.29 overlay retains align-over-align_origin precedence; serialized-pose SO3 tolerance1e-4 is upstream recommended. Raw trajectories unchanged.",
        ),
        SG2Loc=dict(
            paper_table1_3RScan_joint_recall_025m_2deg_by_length={
                "5": 0.34,
                "10": 0.39,
                "25": 0.44,
            },
            paper_test_3RScan_sequence_counts={"5": 1865, "10": 916, "25": 349},
            paper_test_3RScan_image_counts={"5": 9325, "10": 9160, "25": 8725},
            expected_3RScan_test_scenes=30,
            input_condition="Author implementation uses pose-derived relative motion and per-frame gravity; not GT-free FYP. Offline ground-truth patch-annotation filtering must be counted separately.",
            core_method_executed=False,
            full_raw_benchmark_acquired=False,
        ),
        OMCL=dict(
            paper_table2={
                "particles": 1024,
                "rays": 2048,
                "map_resolution_m": 0.02,
                "features": 512,
                "similarity_threshold": 0.9,
                "surrounding_radius_m": 2.0,
                "matching_criterion": 500,
            },
            source_default={
                "num_particles": cfg["num_particles"],
                "num_samples": cfg["num_samples"],
                "particle_batch": cfg["particles_batch_size"],
                "loss_batch": cfg["rays_loss_batch_size"],
                "odom": cfg["odom"],
            },
            source_config_sha256=sha(config_path),
            source_pf_sha256=sha(BASE / "sources/omcl/src/omcl/models/pf.py"),
            unresolved_protocol_differences=[
                "Paper2048 rays versus released default num_samples1024; sampler applies num_samples per observed feature mask, not necessarily a total ray budget.",
                "Released feature extraction crop512/base540 versus resource-probe crop480; probe time is not full mapping-feature time.",
                "estimate_loss divides intersections in16 chunks and uses end=-1 at the boundary; possible omitted pairs require a core-method correctness test before changes.",
                "Known-start initialization and planar/up-axis assumptions must be validated against the dataset adapter; do not silently repurpose arbitrary 6DoF Hypersim flights as planar localization.",
            ],
            paper_table1_LSeg_APE_RMSE_m={"planar": 0.15, "3D": 0.20},
            paper_table4_Replica_encoder="NARADIO, not LSeg",
            paper_table4_Replica_ATE_RMSE_cm=35,
            core_method_executed=False,
        ),
        scope="Paper/source protocol audit. Does not count as any of four completed localization reproductions; unresolved source-paper differences remain visible.",
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    main(p.parse_args().output)
