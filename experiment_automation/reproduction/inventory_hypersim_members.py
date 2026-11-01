"""Read official ZIP directories to budget exact new scientific bytes before downloading."""

import argparse, json, shutil, zipfile
from pathlib import Path
import requests
from acquire_hypersim_scene import StrictWebFile, official


def main(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())["hypersim"]
    rows = []
    for split in ["val", "test"]:
        selections = sorted(
            [s for s in plan["frozen_plan"] if s["split"] == split],
            key=lambda s: len(s["planned_frames"]),
        )[:2]
        for s in selections:
            scene = s["scene"]
            wanted = {scene + "/_detail/metadata_scene.csv"}
            first = {}
            for r in s["planned_frames"]:
                cam, frame = r["camera"], r["frame"]
                first.setdefault(cam, frame)
                for n in [
                    "camera_keyframe_frame_indices",
                    "camera_keyframe_positions",
                    "camera_keyframe_orientations",
                ]:
                    wanted.add(f"{scene}/_detail/{cam}/{n}.hdf5")
                wanted.add(
                    f"{scene}/images/scene_{cam}_final_preview/frame.{frame:04d}.tonemap.jpg"
                )
                for n in ["depth_meters", "semantic", "semantic_instance"]:
                    wanted.add(
                        f"{scene}/images/scene_{cam}_geometry_hdf5/frame.{frame:04d}.{n}.hdf5"
                    )
            for cam, frame in first.items():
                wanted.add(
                    f"{scene}/images/scene_{cam}_geometry_hdf5/frame.{frame:04d}.position.hdf5"
                )
            url = next(u for u in official.URLS if u.endswith("/" + scene + ".zip"))
            with zipfile.ZipFile(StrictWebFile(url, requests.Session())) as z:
                sizes = [
                    dict(path=n, bytes=z.getinfo(n).file_size) for n in sorted(wanted)
                ]
            remaining = sum(
                v["bytes"]
                for v in sizes
                if not (
                    Path("/data/fyp_3d/datasets/hypersim_formal_subset") / v["path"]
                ).exists()
            )
            row = dict(
                scene=scene,
                split=split,
                planned_frames=len(s["planned_frames"]),
                scientific_bytes=sum(v["bytes"] for v in sizes),
                new_bytes=remaining,
                members=len(wanted),
                archive_directory_only=True,
            )
            rows.append(row)
            print(row, flush=True)
    chosen = [
        min((r for r in rows if r["split"] == sp), key=lambda r: r["new_bytes"])
        for sp in ["val", "test"]
    ]
    free = shutil.disk_usage("/data/fyp_3d").free
    budget = free - 30 * 1024**3 - 64 * 1024**2
    result = dict(
        status="inventoried",
        candidates=rows,
        proposed=chosen,
        reserve_bytes=30 * 1024**3,
        remaining_head_and_log_margin_bytes=64 * 1024**2,
        disk_free_bytes=free,
        combined_new_bytes=sum(r["new_bytes"] for r in chosen),
        both_fit_with_reserve_and_margin=sum(r["new_bytes"] for r in chosen) <= budget,
        reason="Balance already frozen splits, no scene replacement or performance-based selection. This is acquisition ordering and exact storage budgeting, not completed full-data qualification.",
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = out.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("output")
    a = p.parse_args()
    main(a.plan, a.output)
