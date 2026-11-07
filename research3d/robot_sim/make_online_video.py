"""CPU-only visualization of real asynchronous localization and camera capture.

All estimator rows are selected by OUTPUT availability timestamp, and candidate
rows by their received timestamp. GT is an offline overlay and supplies exactly
one explicitly labelled origin for local-odometry diagnostics. No GPU inference,
future pose substitution, trajectory fit, or best-candidate selection is used.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research3d.fusion.contracts import pose, read_jsonl, sha256_file, write_json


def available_count(rows, key, now):
    """Prefix length of already available, chronologically logged output rows."""
    times = [float(row[key]) for row in rows]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f"non-monotonic {key}; never silently reorder a run")
    return bisect_right(times, float(now))


def candidate_base_points(row, T_base_camera):
    """All actual candidate positions, transformed by a fixed mechanical rig."""
    candidate = row["candidate"]
    if candidate.get("status") != "ok":
        return []
    transforms = []
    if "spvloc" in candidate.get("source", ""):
        if candidate.get("refined_top1_T_map_camera_optical") is not None:
            transforms.append(("refined_top1", candidate["refined_top1_T_map_camera_optical"]))
        transforms.extend((f"rank{x['rank']}", x["T_map_camera_optical"])
                          for x in candidate.get("candidates", []))
    elif candidate.get("T_map_camera_optical") is not None:
        transforms.append(("raw", candidate["T_map_camera_optical"]))
    return [(name, (pose(T) @ np.linalg.inv(pose(T_base_camera)))[:2, 3])
            for name, T in transforms]


def first_origin(frames, truth, trace):
    """A single GT origin at the first frame with an already available odometry."""
    for frame in frames:
        now = frame["timestamp_ns"] / 1e9
        n = available_count(trace, "timestamp_s", now)
        valid = [r for r in trace[:n] if r.get("local_valid") and r.get("T_odom_base") is not None
                 and r.get("pose_time_s") is not None and r["pose_time_s"] <= now]
        if valid:
            row = valid[-1]
            return pose(truth[frame["query_id"]]["T_map_base"]) @ np.linalg.inv(pose(row["T_odom_base"])), dict(
                query_id=frame["query_id"], truth_timestamp_s=now,
                output_timestamp_s=row["timestamp_s"], local_pose_timestamp_s=row["pose_time_s"],
                timestamp_difference_s=now-row["pose_time_s"],
                scope="one first available origin, no interpolation or full trajectory fitting")
    return None, None


def finite_points(rows):
    points = np.asarray(rows, float).reshape(-1, 2)
    return points[np.isfinite(points).all(axis=1)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="contains capture/, localization/, evaluation/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args(argv)
    if a.stride < 1:
        p.error("stride must be positive")
    if a.output.exists():
        raise FileExistsError("fresh visualization output required")
    root = a.run.resolve(strict=True)
    dataset, run = root / "capture", root / "localization"
    source_paths = [dataset / "map_manifest.json", dataset / "query/frames.jsonl",
                    dataset / "offline_gt/query_poses.jsonl", run / "localization.jsonl",
                    run / "ace_decisions.jsonl", root / "evaluation/metrics.json"]
    manifest = json.loads(source_paths[0].read_text())
    frames = read_jsonl(source_paths[1])
    gt_rows = read_jsonl(source_paths[2])
    truth = {x["query_id"]: x for x in gt_rows}
    trace, decisions = read_jsonl(source_paths[3]), read_jsonl(source_paths[4])
    metrics = json.loads(source_paths[5].read_text())
    if not frames or len(truth) != len(frames) or set(truth) != {x["query_id"] for x in frames}:
        raise ValueError("frozen query/GT inventory mismatch")
    times = np.asarray([r["timestamp_ns"] / 1e9 for r in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError("query timestamps must be strictly increasing")
    if any(r["map_id"] != manifest["map_id"] for r in trace):
        raise ValueError("localization belongs to another map")
    for r in decisions:
        if r["candidate"]["map_id"] != manifest["map_id"]:
            raise ValueError("candidate belongs to another map")
        if r["candidate"]["stamp_sec"] > r["received_time_s"] + 1e-6:
            raise ValueError("candidate received before its actual capture")
    for r in trace:
        if r.get("pose_time_s") is not None and r["pose_time_s"] > r["timestamp_s"] + 1e-6:
            raise ValueError("pose timestamp later than output availability")
    available_count(trace, "timestamp_s", times[-1])
    available_count(decisions, "received_time_s", times[-1])
    origin, anchor = first_origin(frames, truth, trace)
    if origin is None:
        raise ValueError("no actual local pose available for origin diagnostic")
    T_base_camera = manifest["T_base_camera_optical"]
    gt_xy = np.array([pose(truth[r["query_id"]]["T_map_base"])[:2, 3] for r in frames])
    local_xy, global_xy = [], []
    for row in trace:
        usable = row["timestamp_s"] >= anchor["output_timestamp_s"]
        local_xy.append((origin @ pose(row["T_odom_base"]))[:2, 3]
                        if usable and row["local_valid"] and row["T_odom_base"] is not None else [np.nan, np.nan])
        global_xy.append(pose(row["T_map_base"])[:2, 3]
                         if row["global_valid"] and row["T_map_base"] is not None else [np.nan, np.nan])
    local_xy, global_xy = np.asarray(local_xy), np.asarray(global_xy)
    candidate_xy = [candidate_base_points(row, T_base_camera) for row in decisions]
    # Fixed axes use the full output extent solely for layout, never for pose selection.
    all_xy = np.vstack([gt_xy, finite_points(local_xy), finite_points(global_xy),
                        finite_points([xy for group in candidate_xy for _, xy in group])])
    lo, hi = all_xy.min(axis=0)-.7, all_xy.max(axis=0)+.7
    events_path, fault_path = dataset / "offline_gt/events.json", dataset / "offline_gt/camera_fault.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    fault = json.loads(fault_path.read_text()) if fault_path.exists() else {}
    source_paths += [path for path in (events_path, fault_path) if path.exists()]
    a.output.mkdir(parents=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 7.4), dpi=100)
    for obj in manifest.get("objects", []):
        if obj["category"] == "wall":
            x, y, _ = obj["center_m"]; sx, sy, _ = obj["size_m"]
            ax.add_patch(Rectangle((x-sx/2, y-sy/2), sx, sy, facecolor=".90", edgecolor=".8", linewidth=.4))
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="map x (m)", ylabel="map y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=.2)
    specs = [("GT base: offline display only", "black", "-"),
             ("Wheel/IMU: first-GT-origin diagnostic", "#2474B5", "--"),
             ("Actual map pose: EKF + gated ACE", "#149747", "-")]
    lines = [ax.plot([], [], color=c, linestyle=style, linewidth=1.6, label=name)[0] for name, c, style in specs]
    current = [ax.plot([], [], marker="o", color=c, markersize=6, linestyle="None")[0] for _, c, _ in specs]
    ace_points, = ax.plot([], [], "x", color="#DF8500", markersize=6, label="ACE candidates: accepted/rejected")
    spv_points, = ax.plot([], [], "^", color="#AA44AA", markersize=5, alpha=.65, label="SPVLoc shadow: never fused")
    ax.legend(fontsize=8, loc="upper left", framealpha=.95)
    ax.set_title("Base trajectories / ALL candidate outliers retained", fontsize=11)
    fig.tight_layout()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 15)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    chosen = list(range(0, len(frames), a.stride))
    if chosen[-1] != len(frames)-1:
        chosen.append(len(frames)-1)
    gif, durations, audit = [], [], []
    for index, i in enumerate(chosen):
        frame, now = frames[i], times[i]
        nt = available_count(trace, "timestamp_s", now)
        nd = available_count(decisions, "received_time_s", now)
        latest = trace[nt-1] if nt else None
        decision = decisions[nd-1] if nd else None
        gtx, gty = [], []
        for j, point in enumerate(gt_xy[:i+1]):
            if j and np.linalg.norm(point-gt_xy[j-1]) > 1.:
                gtx.append(np.nan); gty.append(np.nan)
            gtx.append(point[0]); gty.append(point[1])
        lines[0].set_data(gtx, gty)
        lines[1].set_data(local_xy[:nt, 0], local_xy[:nt, 1])
        lines[2].set_data(global_xy[:nt, 0], global_xy[:nt, 1])
        for marker, xy in zip(current, [gt_xy[i], local_xy[nt-1] if nt else [np.nan]*2,
                                       global_xy[nt-1] if nt else [np.nan]*2]):
            marker.set_data([xy[0]], [xy[1]])
        for artist, match_spv in [(ace_points, False), (spv_points, True)]:
            pts = finite_points([xy for row, points in zip(decisions[:nd], candidate_xy[:nd])
                                 if ("spvloc" in row["candidate"]["source"]) == match_spv for _, xy in points])
            artist.set_data(pts[:, 0], pts[:, 1])
        fig.canvas.draw()
        plot = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        image_path = (dataset / frame["image_path"]).resolve(strict=True)
        if not image_path.is_relative_to(dataset / "query") or sha256_file(image_path) != frame["image_sha256"]:
            raise ValueError("captured RGB changed or escaped query root")
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            if rgb.size != (640, 480):
                raise ValueError("unexpected capture dimensions; no implicit data resize")
            black = not bool(np.asarray(rgb).any())
        canvas = Image.new("RGB", (1440, 950), "white")
        canvas.paste(rgb, (0, 54)); canvas.paste(plot, (640, 45))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"ACTUAL GAZEBO ONLINE FAULT RUN | t = {now-times[0]:.1f}s | simulation stamp {now:.3f}s", font=bold, fill="#18324C")
        draw.text((12, 539), f"Captured left RGB: {frame['query_id']} | {'BLACK FRAME' if black else 'visible image'}", font=small, fill="black")
        local_state = latest["local_state"] if latest else "NO OUTPUT YET"
        global_state = latest["global_state"] if latest else "NO OUTPUT YET"
        draw.text((12, 569), "LOCAL: " + local_state, font=bold, fill="#2474B5")
        draw.text((12, 598), "GLOBAL: " + global_state, font=bold, fill="#149747")
        age = "n/a" if latest is None or latest["pose_time_s"] is None else f"{now-latest['pose_time_s']:.3f}s"
        draw.text((12, 630), "Latest available output / pose age: " + (f"{latest['timestamp_s']:.3f}s / {age}" if latest else "none"), font=small, fill="black")
        triggers = latest.get("query_trigger_counts", {}) if latest else {}
        draw.text((12, 658), f"Triggers: submitted={triggers.get('scheduled',0)}, image rejected={triggers.get('occluded_or_low_texture',0)}", font=small, fill="black")
        draw.text((12, 683), f"Gate cumulative: accepted={latest.get('accepted_count',0) if latest else 0}, rejected={latest.get('rejected_count',0) if latest else 0}", font=small, fill="black")
        annotation = []
        if fault:
            start, stop = [fault["origin_timestamp_s"] + v for v in fault["interval_s"]]
            if now >= start:
                annotation.append("camera blackout ACTIVE" if now <= stop else "camera blackout ended")
        for event in events:
            requested = event["requested_at_timestamp_ns"] / 1e9
            if requested <= now:
                annotation.append(f"{event['type']} requested at {requested:.1f}s")
        draw.text((12, 715), "Offline event labels: " + ("; ".join(annotation) or "none yet"), font=small, fill="#A03A00")
        recovery_text = "Recovery: no completed confirmation available at this time"
        for recovery in metrics.get("recovery_events", []):
            confirmed = recovery.get("confirmed_output_timestamp_s")
            if confirmed is not None and confirmed <= now:
                recovery_text = f"OFFLINE SCORE: recovered at {confirmed:.3f}s, {recovery['recovery_time_s']:.3f}s after actual jump"
        draw.text((12, 752), recovery_text, font=font, fill="#146B35")
        if decision:
            c, d = decision["candidate"], decision["decision"]
            info = f"Latest returned: {c['source']} | {'ACCEPT' if d['accepted'] else 'REJECT'} | {d['reason']} | captured {c['stamp_sec']:.3f}s, received {decision['received_time_s']:.3f}s"
        else:
            info = "Latest returned: no model response available yet"
        for n, line in enumerate(textwrap.wrap(info, width=145)):
            draw.text((12, 788+n*25), line, font=font, fill="#533400")
        draw.text((12, 849), "SPVLoc is SHADOW ONLY (unqualified on this scene); its candidates do not correct the state.", font=font, fill="#913291")
        draw.text((12, 876), "MAC-VO was not run on this fault capture and is not plotted. GT is never an online localization input.", font=font, fill="#34495E")
        draw.text((12, 903), "Past outputs only. Lines connect logged samples; local blue uses ONE offline GT origin. No pose interpolation.", font=small, fill="#34495E")
        gif.append(canvas.quantize(colors=160))
        next_i = chosen[index+1] if index+1 < len(chosen) else i
        duration = (times[next_i]-now)*1000 if next_i != i else 500
        durations.append(max(20, int(round(duration/10)*10)))
        audit.append(dict(query_id=frame["query_id"], display_timestamp_s=now, raw_RGB_sha256=frame["image_sha256"],
                          black_RGB=black, available_trace_count=nt, available_decision_count=nd,
                          latest_output_timestamp_s=latest["timestamp_s"] if latest else None,
                          latest_pose_timestamp_s=latest["pose_time_s"] if latest else None,
                          latest_candidate_received_s=decision["received_time_s"] if decision else None))
    gif[0].save(a.output / "online_fault_replay.gif", save_all=True, append_images=gif[1:],
                duration=durations, loop=0, optimize=False)
    gif[-1].convert("RGB").save(a.output / "final_frame.png")
    fig.savefig(a.output / "trajectory.png", dpi=180); plt.close(fig)
    write_json(a.output / "frame_availability_audit.json", audit)
    write_json(a.output / "visualization_receipt.json", dict(
        status="ACTUAL_ONLINE_RUN_OFFLINE_VISUALIZATION", frames=len(frames), rendered_frames=len(gif),
        source_duration_s=float(times[-1]-times[0]), displayed_duration_s=sum(durations)/1000,
        source=str(root), map_id=manifest["map_id"], local_origin=anchor,
        GT_scope="offline display and single local origin; no online GT input",
        output_selection="timestamp_s <= displayed capture time; candidate received_time_s <= displayed time",
        pose_interpolation=False, trajectory_fitting=False, future_candidate_selection=False,
        SPVLoc="actual shadow candidates only; no state correction", MACVO="not run on this capture, not shown",
        axis_limits_scope="full output range for layout only; future points never displayed early",
        all_outliers_visible=True, gif_bytes=(a.output / "online_fault_replay.gif").stat().st_size,
        input_sha256={str(path): sha256_file(path) for path in source_paths}))
    print(json.dumps(dict(output=str(a.output), rendered_frames=len(gif)), indent=2))


if __name__ == "__main__":
    main()
