"""Freeze a qualified subset only after every prospective scene has a terminal audit.

Quality exclusions are declared from CRC/calibration failures before training;
no localization/test performance is read, tuned or used for scene selection.
"""

import argparse, collections, hashlib, json, sqlite3, time
from pathlib import Path
import numpy as np


def main(plan_path, dataset, output):
    plan_doc = json.loads(Path(plan_path).read_text())
    plan = plan_doc["hypersim"]
    root = Path(dataset)
    out = Path(output)
    ids = ["R.data.hypersim.subset.v3." + s["scene"] for s in plan["frozen_plan"]]
    deadline = time.monotonic() + 47 * 3600
    while True:
        with sqlite3.connect(
            "file:/data/fyp_3d/automation/state/experiments.sqlite?mode=ro", uri=True
        ) as c:
            state = {
                tid: dict(status=st, attempt=n)
                for tid, st, n in c.execute("SELECT task_id,status,attempt FROM tasks")
                if tid in ids
            }
        if len(state) == len(ids) and all(
            r["status"] in ("COMPLETED", "FAILED") for r in state.values()
        ):
            break
        print(
            "WAIT_COMPLETE_DATA_AUDITS",
            dict(collections.Counter(r["status"] for r in state.values())),
            flush=True,
        )
        assert (
            time.monotonic() < deadline
        ), "All-scene audit incomplete; no training qualification created"
        time.sleep(30)
    digest = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    selected = []
    rejected = []
    frames = []
    classes = np.zeros(40, np.int64)
    hashes = {}
    counts = collections.Counter()
    for s in plan["frozen_plan"]:
        scene = s["scene"]
        tid = "R.data.hypersim.subset.v3." + scene
        if state[tid]["status"] == "FAILED":
            rejected.append(
                dict(
                    scene=scene,
                    reason="Failed predeclared source CRC/calibration/availability gate; no outcome-based selection",
                    attempt=state[tid]["attempt"],
                    failure_record=f'/data/fyp_3d/automation/artifacts/{tid}.attempt{state[tid]["attempt"]}.failure.json',
                )
            )
            continue
        accept = root / scene / "v3/acceptance.json"
        a = json.loads(accept.read_text())
        rows = json.loads((accept.parent / "supervision_manifest.json").read_text())
        assert (
            a["calibration_revision"] == "hypersim_camera_v3"
            and a["per_frame_calibration"]
            and a["frozen_plan_sha256"] == digest
        )
        assert a["geometry_round_trip"]["quantization_bound_violations"] == 0 and a[
            "geometry_round_trip"
        ]["probed_camera_frames"] == len({r["camera"] for r in rows})
        assert a["all_selected_members_crc_verified"] and a["partition"] == s["split"]
        assert {(r["camera"], r["frame"]) for r in rows} == {
            (r["camera"], r["frame"]) for r in s["planned_frames"]
        }
        for r in rows:
            key = r["rgb_decoded_sha256"]
            old = hashes.get(key)
            assert (
                old is None or old == s["split"]
            ), "Decoded RGB crosses frozen held-out split; do not drop test frames to fix it"
            hashes[key] = s["split"]
            frames.append(
                dict(
                    **r, split=s["split"], source_family_proxy=s["source_family_proxy"]
                )
            )
        selected.append(
            dict(
                scene=scene,
                frames=len(rows),
                split=s["split"],
                acceptance_sha256=hashlib.sha256(accept.read_bytes()).hexdigest(),
            )
        )
        classes += a["semantic_pixel_counts_1_to_40"]
        counts[s["split"]] += len(rows)
    # This decision manifest is immutable evidence even if scale targets are not met.
    out.mkdir(parents=True, exist_ok=True)
    decision = dict(
        original_frozen_plan_sha256=digest,
        accepted=selected,
        quarantined=rejected,
        policy="Before any new training: exact original planned frame identities retained within accepted scenes; only deterministic source/calibration/availability failures exclude whole scenes. No performance-based thresholds or replacements.",
        localization_metrics_read=False,
    )
    text = json.dumps(decision, indent=2)
    name = "cleaned_plan." + hashlib.sha256(text.encode()).hexdigest()[:16] + ".json"
    (out / name).write_text(text)
    assert (
        len(selected) >= 40 and len(frames) >= 8000
    ), "Data targets not met; honest incomplete qualification, no training release"
    assert all(counts[s] > 0 for s in ["train", "val", "test"])
    result = dict(
        status="qualified_cleaned_supervision_subset",
        calibration_revision="v3",
        acquired_scenes=len(selected),
        acquired_frames=len(frames),
        original_planned_scenes=plan["planned_scenes"],
        original_planned_frames=plan["planned_frames"],
        rejected_scenes=len(rejected),
        splits=dict(counts),
        cleaned_plan=str(out / name),
        cleaned_plan_sha256=hashlib.sha256(text.encode()).hexdigest(),
        semantic_pixel_counts_1_to_40=classes.tolist(),
        exact_decoded_RGB_cross_split_overlap=0,
        physical_building_identity_verified=False,
        localization_executed=False,
        scope="Quality-qualified custom source-family supervision subset; not original mesh or full paper localization benchmark. Class/content reuse beyond exact RGB still needs analysis before claims of building generalization.",
    )
    for name, value in [("manifest.json", frames), ("acceptance.json", result)]:
        p = out / name
        t = p.with_suffix(".tmp")
        t.write_text(json.dumps(value, indent=2))
        t.replace(p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("dataset")
    p.add_argument("output")
    a = p.parse_args()
    main(a.plan, a.dataset, a.output)
