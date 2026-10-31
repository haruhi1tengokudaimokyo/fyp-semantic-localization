"""Full held-out-house localization comparisons, per-scene maps, paired statistics.
Runs real SIFT / official hloc SuperPoint+LightGlue / frozen CLIP, with optional
trained hard/soft semantic observations. No cross-scene coordinate merging.
Each completed child command is hash-validated and resumed without GPU repetition.
"""

import argparse, csv, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(obj, indent=2))
    t.replace(p)


def run(cmd, output, receipt):
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    inputs = {"sources": sources}
    if "--checkpoint" in cmd:
        inputs["model_sha256"] = digest(cmd[cmd.index("--checkpoint") + 1])
    key = hashlib.sha256(json.dumps([cmd, inputs], sort_keys=True).encode()).hexdigest()
    receipt = Path(receipt)
    output = Path(output)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (
            saved["command_hash"] == key
            and output.exists()
            and saved["output_sha256"] == digest(output)
        ):
            return
    print("EXECUTE", json.dumps(cmd), flush=True)
    subprocess.run(cmd, check=True)
    assert output.is_file()
    write(receipt, dict(command_hash=key, output_sha256=digest(output)))


def main(a):
    root = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["building_plan"] and manifest["independent_layout_groups"] >= 12
    scripts = Path(__file__).resolve().parent
    data = Path("/data/fyp_3d")
    records = []
    summaries = {}
    testscenes = sorted(
        {r["scene"] for r in manifest["frames"] if r["split"] == "test"}
    )
    for scene in testscenes:
        frames = [
            r.copy()
            for r in manifest["frames"]
            if r["split"] == "test" and r["scene"] == scene
        ]
        assert len(frames) >= 80
        for r in frames:
            for k in ["rgb", "label", "depth"]:
                r[k] = str((root / r[k]).resolve())
        dataset = out / scene / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        subset = {**manifest, "frames": frames}
        write(dataset / "manifest.json", subset)
        positions = sorted({r["position_id"] for r in frames})
        threshold = positions[max(1, int(len(positions) * 0.6))]
        base = [
            sys.executable,
            str(scripts / "localize_semantic_pilot.py"),
            str(dataset),
        ]
        sift = out / scene / "sift"
        run(
            base
            + [
                str(sift),
                "--mode",
                "geometry",
                "--reference-positions",
                str(threshold),
            ],
            sift / "summary.json",
            sift / "accepted.json",
        )
        learned = out / scene / "learned"
        features = [
            "--hloc-source",
            str(data / "sources/hloc"),
            "--lightglue-source",
            str(data / "sources/LightGlue"),
        ]
        run(
            base
            + [
                str(learned),
                "--mode",
                "geometry",
                "--reference-positions",
                str(threshold),
            ]
            + features,
            learned / "summary.json",
            learned / "accepted.json",
        )
        comparisons = {"sift_bow_k8": sift, "superpoint_bow_k8": learned}
        if not a.geometry_only:
            clip = out / scene / "clip_features"
            run(
                [
                    sys.executable,
                    str(scripts / "extract_clip_reference_features.py"),
                    str(dataset),
                    str(learned),
                    str(clip),
                    "--source",
                    str(data / "sources/CLIP"),
                    "--weights",
                    str(data / "models/clip"),
                ],
                clip / "features.npz",
                clip / "accepted.json",
            )
            for retrieval in ["bow", "clip"]:
                for topk in [4, 8, 16]:
                    target = out / scene / f"{retrieval}_k{topk}_all"
                    cmd = (
                        base
                        + [
                            str(target),
                            "--mode",
                            "all",
                            "--checkpoint",
                            a.checkpoint,
                            "--reference-map",
                            str(learned),
                            "--reference-positions",
                            str(threshold),
                            "--topk",
                            str(topk),
                        ]
                        + features
                    )
                    if retrieval == "clip":
                        cmd += ["--clip-features", str(clip / "features.npz")]
                    run(cmd, target / "summary.json", target / "accepted.json")
                    comparisons[f"{retrieval}_k{topk}"] = target
        scene_results = {}
        for name, path in comparisons.items():
            result = json.loads((path / "summary.json").read_text())
            protocol = json.loads((path / "map_protocol.json").read_text())
            queries = json.loads((path / "queries.json").read_text())
            assert protocol["source_scene"] == scene
            for mode, metrics in result.items():
                if mode == "audit":
                    continue
                method = name + ":" + mode
                scene_results[method] = {
                    k: v for k, v in metrics.items() if k != "errors_all_queries"
                }
                for query, error in zip(queries, metrics["errors_all_queries"]):
                    te = error["translation_m"]
                    re = error["rotation_deg"]
                    source = next(r for r in frames if r["rgb"] == query["rgb"])
                    records.append(
                        dict(
                            scene=scene,
                            building_group=source["building_group"],
                            position_id=source["position_id"],
                            query=query["rgb"],
                            method=method,
                            translation_m=te,
                            rotation_deg=re,
                            success_25cm_10deg=int(
                                te is not None
                                and re is not None
                                and te <= 0.25
                                and re <= 10
                            ),
                        )
                    )
        summaries[scene] = scene_results
        write(out / "per_scene.json", summaries)
    with (out / "grouped_queries.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    # Pair identical query IDs within each map, bootstrap complete spatial positions.
    baseline = "superpoint_bow_k8:geometry"
    rng = np.random.default_rng(1305)
    paired = {}
    for scene in testscenes:
        rows = [r for r in records if r["scene"] == scene]
        base = {r["query"]: r for r in rows if r["method"] == baseline}
        paired[scene] = {}
        for method in sorted({r["method"] for r in rows}):
            candidate = {r["query"]: r for r in rows if r["method"] == method}
            assert set(base) == set(candidate), "Unequal query set"
            groups = sorted({r["position_id"] for r in base.values()})
            diffs = [
                [
                    candidate[q]["success_25cm_10deg"] - base[q]["success_25cm_10deg"]
                    for q in base
                    if base[q]["position_id"] == g
                ]
                for g in groups
            ]
            sums = np.array([sum(v) for v in diffs])
            counts = np.array([len(v) for v in diffs])
            sample = rng.integers(0, len(groups), (10000, len(groups)))
            boot = sums[sample].sum(1) / counts[sample].sum(1)
            paired[scene][method] = dict(
                delta_recall=float(sums.sum() / counts.sum()),
                position_cluster_bootstrap_95CI=np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
                positions=len(groups),
                queries=len(base),
                replicates=10000,
                seed=1305,
            )
    write(out / "paired_statistics.json", paired)
    write(
        out / ("geometry_summary.json" if a.geometry_only else "summary.json"),
        dict(
            status="completed",
            held_out_scenes=len(testscenes),
            independent_layout_groups=manifest["independent_layout_groups"],
            total_query_method_rows=len(records),
            unique_queries=len({(r["scene"], r["query"]) for r in records}),
            per_scene_results=summaries,
            paired_statistics_file=str(out / "paired_statistics.json"),
            semantic_checkpoint_sha256=digest(a.checkpoint) if a.checkpoint else None,
            GT_online=False,
            claim="Synthetic house-group held-out RGB-to-reference-map geometry, CLIP retrieval and hard/soft learned semantic comparisons; no OMCL reproduction; no real-building identity claim; topK variants preregistered, no test-selected winner",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--checkpoint")
    p.add_argument("--geometry-only", action="store_true")
    main(p.parse_args())
