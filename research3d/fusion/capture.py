"""Capture real, nonzero-baseline stereo events from existing ProcTHOR houses.

Run with the existing semantic_loc Python environment. Nothing is downloaded.
Synthetic acquisition time is explicit; wall-clock rendering time is separate.
Both eye images are copied from ONE final AI2-THOR Event after camera updates.
Depth and world poses are offline QA / mapping supervision, never online input.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from research3d.semantic_localization.procthor_geometry import A, convert, xyz

SCHEMA = "fyp.stereo_capture.v1"
COMMIT = "f0825767cd50d69f666c7f282e54abfe58f1e917"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def camera_pose(position_unity_m, yaw_deg: float) -> np.ndarray:
    """Unity left-handed world -> existing right-handed SPVLoc map, metres.

    Optical camera is x right, y down, z forward. A and y-flip each have
    det=-1, so their product is a proper rotation. Map z is vertical.
    """
    yaw = math.radians(yaw_deg)
    rotation = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                         [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    pose = np.eye(4)
    pose[:3, :3] = A @ rotation @ np.diag([1., -1., 1.])
    pose[:3, 3] = A @ np.asarray(position_unity_m, float)
    return pose


def stereo_positions(center, yaw_deg, baseline_m):
    if not 0 < baseline_m < 0.4:
        raise ValueError("baseline must be nonzero and less than agent diameter")
    yaw = math.radians(yaw_deg)
    right = np.array([math.cos(yaw), 0., -math.sin(yaw)])
    return np.asarray(center) - right * baseline_m / 2, np.asarray(center) + right * baseline_m / 2


@dataclass(frozen=True)
class CaptureConfig:
    width: int = 640
    height: int = 480
    vertical_fov_deg: float = 60.
    baseline_m: float = .12
    camera_height_m: float = 1.5
    fps: float = 5.
    translation_speed_m_s: float = .25
    angular_speed_deg_s: float = 20.
    grid_size_m: float = .25
    min_free_gib: float = 100.
    max_new_gib: float = 16.
    max_frames_per_split: int = 0  # nonzero = explicitly truncated PILOT
    quality: str = "High"

    def validate(self):
        if self.width < 320 or self.height < 240:
            raise ValueError("capture native images >= 320x240; do not upsample old images")
        if not 20 <= self.vertical_fov_deg <= 120:
            raise ValueError("unsupported FoV")
        if not 0 < self.baseline_m < .4:
            raise ValueError("stereo baseline must be nonzero and < 0.4 m")
        for name in ("fps", "translation_speed_m_s", "angular_speed_deg_s", "grid_size_m",
                     "camera_height_m", "max_new_gib"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_frames_per_split < 0 or self.min_free_gib < 0:
            raise ValueError("negative limit")

    def calibration(self, map_id):
        focal = self.height / (2 * math.tan(math.radians(self.vertical_fov_deg) / 2))
        extrinsic = np.eye(4)
        extrinsic[0, 3] = self.baseline_m
        return dict(schema=SCHEMA, map_id=map_id, camera_id="left", width=self.width,
                    height=self.height, native_resolution=True, distortion_model="pinhole_none",
                    K=[[focal, 0., (self.width - 1) / 2], [0., focal, (self.height - 1) / 2], [0., 0., 1.]],
                    baseline_m=self.baseline_m, T_left_right=extrinsic.tolist(),
                    transform_convention="T_A_B maps points in B into A",
                    map_axes="ProcTHOR canonical map (X,Y,Z)=(Unity x,z,y), metres",
                    camera_axes="right-handed CV: x right, y down, z forward",
                    depth_convention="optical_axis_z_metres; OFFLINE_ONLY",
                    capture_clock="synthetic fixed sampling clock; not Unity action/wall time",
                    fps=self.fps, vertical_fov_deg=self.vertical_fov_deg)


def check_storage(output_root: Path, config: CaptureConfig, expected_bytes: int = 0):
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reserve = config.min_free_gib * 1024**3
    if root == Path("/data") or Path("/data") in root.parents:
        reserve = max(reserve, 100 * 1024**3)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used + expected_bytes > config.max_new_gib * 1024**3:
        raise RuntimeError("capture storage budget exceeded; no deletion performed")
    if shutil.disk_usage(root).free - expected_bytes < reserve:
        raise RuntimeError("free-space reserve would be violated; no deletion performed")


def reachable_graph(positions, grid_size):
    points = np.array([xyz(p) for p in positions], float)
    origin = points.min(0)
    cells = np.rint((points - origin) / grid_size).astype(int)
    lookup = {tuple(c): i for i, c in enumerate(cells)}
    if len(lookup) != len(points):
        raise ValueError("reachable positions are not a unique planar grid")
    graph = {}
    for index, cell in enumerate(cells):
        neighbors = []
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            key = tuple(cell + [dx, 0, dz])
            if key in lookup:
                j = lookup[key]
                if np.linalg.norm(points[index] - points[j]) <= grid_size * 1.01:
                    neighbors.append(j)
        graph[index] = sorted(neighbors)
    return points, graph


def _farthest(graph, start, allowed):
    parents = {start: None}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        for nxt in graph[last]:
            if nxt in allowed and nxt not in parents:
                parents[nxt] = last
                queue.append(nxt)
    path = [last]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[::-1], set(parents)


def _long_path(graph, allowed):
    components = []
    unseen = set(allowed)
    while unseen:
        _, component = _farthest(graph, min(unseen), unseen)
        unseen -= component
        components.append(component)
    component = max(components, key=lambda c: (len(c), -min(c)))
    first, _ = _farthest(graph, min(component), component)
    path, _ = _farthest(graph, first[-1], component)
    return path


def plan_routes(positions, config):
    """Distinct reachable center paths, chosen without localization/GT outcomes.

    Query uses a long graph route; reference uses the longest component after
    excluding query vertices. This is a modest reference-coverage pilot, not
    a claim of exhaustive reference coverage. Every interpolated rig center is
    additionally collision-checked by TeleportFull(forceAction=False).
    """
    points, graph = reachable_graph(positions, config.grid_size_m)
    query = _long_path(graph, set(graph))
    remaining = set(graph) - set(query)
    if not remaining:
        raise ValueError("house lacks disjoint reference/query routes")
    reference = _long_path(graph, remaining)
    if min(len(reference), len(query)) < 3:
        raise ValueError("need >= 3 reachable vertices in both disjoint routes")
    return {"reference": points[reference], "query": points[query]}


def densify_route(vertices, config):
    """Piecewise smooth path: bounded translation and in-place yaw turns."""
    out = []
    previous_yaw = None
    for start, end in zip(vertices[:-1], vertices[1:]):
        delta = end - start
        yaw = math.degrees(math.atan2(delta[0], delta[2])) % 360
        if previous_yaw is not None:
            difference = (yaw - previous_yaw + 180) % 360 - 180
            turns = max(1, math.ceil(abs(difference) * config.fps / config.angular_speed_deg_s))
            for j in range(1, turns + 1):
                out.append((start.copy(), (previous_yaw + difference * j / turns) % 360))
        count = max(1, math.ceil(np.linalg.norm(delta) * config.fps / config.translation_speed_m_s))
        for j in range(count):
            out.append((start + delta * j / count, yaw))
        previous_yaw = yaw
    out.append((vertices[-1].copy(), previous_yaw))
    # Remove duplicate stationary poses at turn/move joins.
    return [p for i, p in enumerate(out) if i == 0 or not
            (np.allclose(p[0], out[i-1][0], atol=1e-10) and abs(p[1] - out[i-1][1]) < 1e-10)]


def _action(controller, **kwargs):
    event = controller.step(**kwargs)
    if not event.metadata.get("lastActionSuccess"):
        raise RuntimeError(f"AI2-THOR {kwargs['action']}: {event.metadata.get('errorMessage')}")
    return event


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def render_pair(controller, center, yaw, config, initialized):
    eye_positions = stereo_positions(center, yaw, config.baseline_m)
    # Only the final event is saved: the first intermediate event is discarded.
    for index, position in enumerate(eye_positions):
        params = dict(action="UpdateThirdPartyCamera" if initialized else "AddThirdPartyCamera",
                      position=dict(zip("xyz", map(float, position))), rotation=dict(x=0., y=yaw, z=0.),
                      fieldOfView=config.vertical_fov_deg, nearClippingPlane=.01, farClippingPlane=50.)
        if initialized:
            params["thirdPartyCameraId"] = index
        event = _action(controller, **params)
    cameras = event.metadata.get("thirdPartyCameras", [])
    images, depths = event.third_party_camera_frames, event.third_party_depth_frames
    if len(cameras) != 2 or len(images) != 2 or len(depths) != 2:
        raise ValueError("same Event must contain exactly two RGB, depth and camera records")
    poses = []
    for index, camera in enumerate(cameras):
        actual = xyz(camera["position"])
        if not np.allclose(actual, eye_positions[index], atol=1e-5):
            raise ValueError("returned third-party camera translation differs from request")
        rotation = camera["rotation"]
        if abs(camera["fieldOfView"] - config.vertical_fov_deg) > .001:
            raise ValueError("returned FoV differs from calibrated native projection")
        if abs((rotation["y"] - yaw + 180) % 360 - 180) > .001 or abs(rotation["x"]) > .001 or abs(rotation["z"]) > .001:
            raise ValueError("returned camera rotation invalid for rectified stereo")
        poses.append(camera_pose(actual, rotation["y"]))
    return event, [np.asarray(x)[..., :3].copy() for x in images], [np.asarray(x, np.float32).copy() for x in depths], poses


def capture_house(controller, house_path: Path, output: Path, source_entry: dict,
                  config: CaptureConfig, protocol_provenance=None):
    from research3d.fusion.data_validation import validate_capture
    config.validate()
    if source_entry["split"] not in {"train", "val"}:
        raise ValueError("development collector refuses test houses; seal formal test separately")
    house_hash = sha256(house_path)
    house = json.loads(house_path.read_text())
    geometry = {k: house[k] for k in ("rooms", "walls", "doors", "windows", "objects")}
    geometry_hash = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if geometry_hash != source_entry["source_geometry_sha256"]:
        raise ValueError("source geometry SHA differs from accepted split manifest")
    scene_id = int(source_entry["scene_id"])
    building_id = f"{source_entry['split']}/{source_entry['official_house_index']}"
    map_id = f"procthor/{building_id}:{geometry_hash[:16]}"
    dest = output / f"scene_{scene_id:05d}"
    stage = output / f".scene_{scene_id:05d}.partial"
    if dest.exists() or stage.exists():
        raise FileExistsError("refusing overwrite/restart of existing accepted or partial capture")
    check_storage(output, config)
    stage.mkdir(parents=True)
    try:
        floor_heights = np.array([p["y"] for room in house["rooms"] for p in room["floorPolygon"]])
        if np.ptp(floor_heights) > .001:
            raise ValueError("current planar rig planner requires a single floor elevation")
        floor_y = float(np.median(floor_heights))
        event = controller.reset(scene=house)
        if not event.metadata.get("lastActionSuccess"):
            raise RuntimeError("house reset failed: " + event.metadata.get("errorMessage", ""))
        _action(controller, action="PausePhysicsAutoSim")
        reachable = _action(controller, action="GetReachablePositions").metadata["actionReturn"]
        vertices = plan_routes(reachable, config)
        routes = {split: densify_route(route, config) for split, route in vertices.items()}
        full_lengths = {k: len(v) for k, v in routes.items()}
        if config.max_frames_per_split:
            routes = {k: v[:config.max_frames_per_split] for k, v in routes.items()}
        if any(len(v) < 3 for v in routes.values()):
            raise ValueError("need at least three frames per split")
        # Stereo RGB(6), two float32 depths(8), left instances(3), with generous
        # lossless PNG/JSON/checksum/log overhead: 24 bytes per native pixel.
        estimate = sum(map(len, routes.values())) * config.width * config.height * 24 + 8 * 1024**2
        check_storage(output, config, estimate)
        (stage / "map").mkdir()
        shutil.copyfile(house_path, stage / "map/procthor_house.json")
        atomic_json(stage / "map/annotation_3d.json", convert(house))
        atomic_json(stage / "calibration.json", config.calibration(map_id))
        calibration_sha = sha256(stage / "calibration.json")
        initialized = False
        start = time.monotonic()
        for split, route in routes.items():
            online, offline = [], []
            for index, (agent_position, yaw) in enumerate(route):
                check_storage(output, config, config.width * config.height * 24)
                _action(controller, action="TeleportFull", position=dict(zip("xyz", map(float, agent_position))),
                        rotation=dict(x=0., y=yaw, z=0.), horizon=0., standing=True, forceAction=False)
                # GetReachablePositions.y is the agent collider origin, not
                # necessarily floor height; camera height is above source floor.
                center = agent_position.copy()
                center[1] = floor_y + config.camera_height_m
                event, images, depths, poses = render_pair(controller, center, yaw, config, initialized)
                initialized = True
                frame_id = f"{split}/{index:06d}"
                identity = dict(source_id="procthor", building_id=building_id, map_id=map_id,
                                camera_id="left", query_id=f"{building_id}/{frame_id}", frame_id=frame_id,
                                timestamp_s=index / config.fps)
                row = dict(identity, capture_time_s=index / config.fps,
                           render_completed_time_s=time.monotonic() - start,
                           event_sequence_id=controller.server.sequence_id, same_event_stereo=True)
                off = dict(identity, T_map_camera=poses[0].tolist(), T_map_right=poses[1].tolist(),
                           third_party_camera_metadata=event.metadata["thirdPartyCameras"],
                           provenance=dict(source="AI2-THOR known renderer pose", role="offline supervision/evaluation only",
                                           calibration_sha256=calibration_sha, rendered_simulation=True))
                instances = event.third_party_instance_segmentation_frames
                if len(instances) != 2:
                    raise ValueError("both eye instance rasters required for independent structural QA")
                instance_path = Path("offline") / split / "left_instance" / f"{index:06d}.png"
                (stage / instance_path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(instances[0])[..., :3]).save(stage / instance_path)
                dictionary_path = Path("offline") / split / "instance_dictionary" / f"{index:06d}.json"
                atomic_json(stage / dictionary_path, {str(k): list(v) for k, v in event.object_id_to_color.items()})
                off.update(left_instance=str(instance_path), instance_dictionary=str(dictionary_path))
                for eye, rgb, depth in zip(("left", "right"), images, depths):
                    if rgb.shape != (config.height, config.width, 3) or rgb.dtype != np.uint8:
                        raise ValueError("native RGB shape/dtype mismatch")
                    rgb_path = Path("online") / split / eye / f"{index:06d}.png"
                    depth_path = Path("offline") / split / f"{eye}_depth" / f"{index:06d}.npy"
                    (stage / rgb_path).parent.mkdir(parents=True, exist_ok=True)
                    (stage / depth_path).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(stage / rgb_path)
                    np.save(stage / depth_path, depth, allow_pickle=False)
                    row[f"{eye}_rgb"] = str(rgb_path)
                    row[f"{eye}_sha256"] = sha256(stage / rgb_path)
                    off[f"{eye}_depth"] = str(depth_path)
                online.append(row)
                offline.append(off)
            _write_jsonl(stage / "online" / split / "frames.jsonl", online)
            _write_jsonl(stage / "offline" / f"{split}_poses.jsonl", offline)
        manifest = dict(schema=SCHEMA, status="CAPTURED_NOT_DATA_READY", frontend_readiness="NOT_VALIDATED", map_id=map_id,
                        source_id="procthor", building_id=building_id, source_split=source_entry["split"],
                        source_geometry_sha256=geometry_hash, source_file_sha256=house_hash, source_entry=source_entry,
                        protocol_provenance=protocol_provenance,
                        config=asdict(config), full_planned_frames=full_lengths,
                        captured_frames={k: len(v) for k, v in routes.items()},
                        pilot_truncated=any(len(routes[k]) < full_lengths[k] for k in routes),
                        native_renderer_commit=COMMIT, static_physics_paused=True,
                        reference_query_policy="disjoint collision-checked reachable-center paths; no reused captures",
                        supervision_policy="poses and depth in offline/; RGB/calibration only online",
                        structure_policy="official house polygons via existing procthor_geometry.convert; no invented raster classes",
                        seconds=time.monotonic() - start)
        atomic_json(stage / "manifest.json", manifest)
        audit = validate_capture(stage)
        atomic_json(stage / "validation.json", audit)
        if not audit["passed"]:
            raise ValueError("capture QA failed: " + "; ".join(audit["errors"]))
        manifest["status"] = "PILOT_CAPTURE_QA_PASSED" if manifest["pilot_truncated"] else "CAPTURE_QA_PASSED"
        manifest["validation_sha256"] = sha256(stage / "validation.json")
        atomic_json(stage / "manifest.json", manifest)
        hashes = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        atomic_json(stage / "checksums.json", hashes)
        stage.rename(dest)
        return manifest
    except Exception as exc:
        atomic_json(stage / "failure.json", dict(status="FAILED_NOT_DATA_READY", type=type(exc).__name__, reason=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--scenes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thor-root", type=Path, default=Path("/data/fyp_3d/tools/ai2thor"))
    parser.add_argument("--software-render", action="store_true")
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/fusion/experiment.json")
    parser.add_argument("--building-lineage", type=Path)
    for field, item in CaptureConfig.__dataclass_fields__.items():
        parser.add_argument("--" + field.replace("_", "-"), type=type(item.default))
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    source_override = args.source_root is not None
    args.source_root = args.source_root or Path(protocol["data"]["source_root"])
    lineage_path = args.building_lineage or args.protocol.with_name("building_lineage.json")
    lineage = {x["scene_id"]: x for x in json.loads(lineage_path.read_text())["development"]}
    stereo = protocol["data"]["stereo"]
    values = {k: stereo[k] for k in ("width", "height", "vertical_fov_deg", "baseline_m", "camera_height_m")}
    values.update(fps=stereo["sample_hz"],
                  translation_speed_m_s=protocol["data"]["route_targets"]["max_speed_m_s"],
                  angular_speed_deg_s=protocol["data"]["route_targets"]["max_angular_speed_deg_s"],
                  min_free_gib=protocol["resources"]["min_data_free_gib"],
                  max_new_gib=protocol["resources"]["max_new_data_peak_gib"])
    overrides = {k: getattr(args, k) for k in CaptureConfig.__dataclass_fields__ if getattr(args, k) is not None}
    values.update(overrides)
    config = CaptureConfig(**values)
    provenance = dict(protocol_sha256=sha256(args.protocol), lineage_sha256=sha256(lineage_path),
                      explicit_cli_overrides=overrides)
    config.validate()
    if args.software_render:
        os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        os.environ["VK_DRIVER_FILES"] = os.environ["VK_ICD_FILENAMES"]
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    class ExistingController(Controller):
        @property
        def base_dir(self):
            # Unity writes logs under base_dir even with a local executable.
            # Keep every new byte inside the requested capture destination.
            return str(args.output.resolve() / ".renderer")

        def unity_command(self, width, height, headless):
            command = super().unity_command(width, height, headless)
            # Unity's automatic selector may reject a CPU Vulkan device; the
            # official device-index launch argument selects the sole lavapipe
            # device explicitly, without changing shaders or geometry.
            if args.software_render:
                command += ["-force-device-index", "0"]
            return command
    manifest_path = args.split_manifest or (args.source_root / "building_split_manifest_100_20_20_accepted.json"
                                           if source_override else Path(protocol["data"]["accepted_manifest"]))
    entries = {x["scene_id"]: x for x in json.loads(manifest_path.read_text())["houses"]}
    for sid in args.scenes:
        if (sid not in entries or entries[sid]["split"] not in {"train", "val"}
                or sid not in protocol["data"]["development_scene_ids"]):
            parser.error(f"scene {sid} is not in frozen development train/val houses")
        source_path = args.source_root / f"scene_{sid:05d}" / "procthor_house.json"
        if (sid not in lineage or entries[sid]["source_geometry_sha256"] != lineage[sid]["base_building_geometry_sha256"]
                or sha256(source_path) != lineage[sid]["house_file_sha256"]):
            parser.error(f"scene {sid} no longer matches frozen building lineage")
    executable = args.thor_root / "releases" / f"thor-CloudRendering-{COMMIT}" / f"thor-CloudRendering-{COMMIT}"
    if not executable.is_file():
        raise FileNotFoundError("existing official renderer missing; no automatic install/download: " + str(executable))
    check_storage(args.output, config)
    controller = ExistingController(platform=CloudRendering, local_executable_path=str(executable),
                                    width=config.width, height=config.height, fieldOfView=config.vertical_fov_deg,
                                    renderDepthImage=True, renderInstanceSegmentation=True,
                                    renderSemanticSegmentation=False, quality=config.quality,
                                    gridSize=config.grid_size_m, snapToGrid=False,
                                    server_start_timeout=120, server_timeout=120)
    try:
        for sid in args.scenes:
            result = capture_house(controller, args.source_root / f"scene_{sid:05d}" / "procthor_house.json",
                                   args.output, entries[sid], config, provenance)
            print(json.dumps({k: result[k] for k in ("status", "building_id", "captured_frames", "seconds")}), flush=True)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
