"""GT-free local/global localization contracts for a planar mobile robot.

The local EKF owns odom->base_link. This module NEVER changes that transform.
ACE estimates optical-camera-to-map poses; delayed measurements are associated
with historical odometry, producing map->odom. Separate recovery hypotheses
are retained, never averaged. This is gated correction, not a pose graph.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from research3d.fusion.contracts import discrepancy, no_truth, pose


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def planar(value):
    """Project an already quality-gated base pose to the planar robot manifold."""
    value = pose(value)
    yaw = math.atan2(value[1, 0], value[0, 0])
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    out[:2, 3] = value[:2, 3]
    return out


class OdometryHistory:
    def __init__(self, duration_s=15., max_gap_s=.15):
        self.duration_s = duration_s
        self.max_gap_s = max_gap_s
        self.samples = deque()

    def append(self, stamp, transform, covariance_trace=0.):
        stamp = finite(stamp, "odometry stamp")
        transform = pose(transform).copy()
        finite(covariance_trace, "covariance trace")
        if self.samples and stamp <= self.samples[-1][0]:
            raise ValueError("odometry timestamps must strictly increase")
        self.samples.append((stamp, transform, covariance_trace))
        while len(self.samples) > 2 and stamp - self.samples[1][0] > self.duration_s:
            self.samples.popleft()

    def at(self, stamp):
        if not self.samples or stamp < self.samples[0][0] or stamp > self.samples[-1][0]:
            raise ValueError("observation outside odometry history; no extrapolation")
        previous = self.samples[0]
        for item in self.samples:
            if abs(item[0] - stamp) < 1e-8:
                return item[1].copy()
            if item[0] > stamp:
                gap = item[0] - previous[0]
                if gap > self.max_gap_s:
                    raise ValueError("observation crosses missing local odometry")
                fraction = (stamp - previous[0]) / gap
                out = np.eye(4)
                out[:3, 3] = previous[1][:3, 3] * (1-fraction) + item[1][:3, 3] * fraction
                out[:3, :3] = Slerp([previous[0], item[0]],
                    Rotation.from_matrix(np.stack([previous[1][:3, :3], item[1][:3, :3]])))(stamp).as_matrix()
                return out
            previous = item
        raise ValueError("no odometry bracket")


@dataclass
class LocalizationConfig:
    map_id: str
    map_sha256: str | None = None
    min_inliers: int = 100
    max_reprojection_px: float = 10.
    max_observation_age_s: float = 2.5
    local_timeout_s: float = .3
    max_local_covariance_trace: float = 100.
    max_dead_reckoning_s: float = 15.
    tracking_translation_m: float = 1.
    tracking_rotation_deg: float = 25.
    recovery_translation_m: float = .5
    recovery_rotation_deg: float = 15.
    confirmations: int = 3
    confirmation_span_s: float = 1.
    hypothesis_ttl_s: float = 5.
    conflicts_to_recover: int = 2
    max_base_height_error_m: float = .35
    max_base_tilt_deg: float = 20.
    map_base_height_m: float = 0.
    allow_spvloc_corrections: bool = False


class GlobalLocalizer:
    def __init__(self, config: LocalizationConfig, T_base_camera_optical):
        if not config.map_id or config.confirmations < 2:
            raise ValueError("map identity and at least two independent confirmations required")
        if type(config.allow_spvloc_corrections) is not bool:
            raise ValueError("SPVLoc correction qualification must be an explicit boolean")
        self.config = config
        self.T_base_camera = pose(T_base_camera_optical).copy()
        self.history = OdometryHistory()
        self.T_map_odom = None
        self.last_accepted_stamp = None
        self.last_processed_stamp = -math.inf
        self.seen_queries = set()
        self.hypotheses = []
        self.conflicts = 0
        self.global_state = "UNINITIALIZED"
        self.last_decision = {"accepted": False, "reason": "no_observation"}
        self.accepted_count = 0
        self.rejected_count = 0

    def add_odometry(self, stamp, transform, covariance_trace=0.):
        self.history.append(stamp, transform, covariance_trace)

    def local_state(self, now):
        if not self.history.samples:
            return "WAITING_FOR_ODOMETRY"
        stamp, _, covariance = self.history.samples[-1]
        if now - stamp > self.config.local_timeout_s or stamp-now > .05:
            return "STALE"
        if covariance > self.config.max_local_covariance_trace:
            return "DEGRADED"
        return "TRACKING"

    def _reject(self, reason, query_id=None):
        self.rejected_count += 1
        self.last_decision = {"accepted": False, "reason": reason, "query_id": query_id}
        return self.last_decision.copy()

    def observe(self, row, now):
        """Receive one real ACE response. No GT, pose covariance or score fabrication."""
        cfg = self.config
        try:
            no_truth(row)
            now = finite(now, "receive time")
            stamp = finite(row["stamp_sec"], "capture time")
            qid = row["query_id"]
            if not isinstance(qid, str) or not qid:
                raise ValueError("empty query identity")
        except (KeyError, TypeError, ValueError) as exc:
            return self._reject(f"invalid_contract:{exc}")
        if row.get("map_id") != cfg.map_id:
            return self._reject("map_mismatch", qid)
        if cfg.map_sha256 is not None and row.get("map_sha256") != cfg.map_sha256:
            return self._reject("map_fingerprint_mismatch", qid)
        if row.get("source") != "official_ace_dsacstar":
            return self._reject("unsupported_localizer", qid)
        if qid in self.seen_queries or stamp <= self.last_processed_stamp:
            return self._reject("duplicate_or_out_of_order", qid)
        if stamp > now + .001 or now-stamp > cfg.max_observation_age_s:
            return self._reject("future_or_stale_observation", qid)
        self.seen_queries.add(qid)
        self.last_processed_stamp = stamp
        if row.get("status") != "ok":
            return self._reject("localizer_failed", qid)
        if self.local_state(now) not in {"TRACKING","DEGRADED"}:
            return self._reject("local_odometry_unavailable", qid)
        try:
            n = row["inliers"]
            if type(n) is not int or n < cfg.min_inliers:
                return self._reject("insufficient_inliers", qid)
            reprojection = row.get("reprojection_error_px")
            # Official ACE may not expose a post-PnP mean reprojection error.
            # Missing means unavailable, not zero or a fabricated quality score.
            if reprojection is not None:
                reprojection = finite(reprojection, "reprojection error")
                # A median over *all* scene-coordinate predictions is not an
                # inlier residual: ACE can localize with a small accurate subset.
                if row.get("reprojection_statistic") == "dsac_inlier_median" and reprojection > cfg.max_reprojection_px:
                    return self._reject("reprojection_error", qid)
            T_map_camera = pose(row["T_map_camera_optical"])
            T_odom_base_capture = self.history.at(stamp)
        except (KeyError, ValueError, TypeError) as exc:
            return self._reject(f"invalid_pose_or_history:{exc}", qid)
        T_map_base = T_map_camera @ np.linalg.inv(self.T_base_camera)
        tilt_deg = math.degrees(math.acos(float(np.clip(T_map_base[2, 2], -1, 1))))
        if abs(T_map_base[2, 3]-cfg.map_base_height_m) > cfg.max_base_height_error_m or tilt_deg > cfg.max_base_tilt_deg:
            return self._reject("implausible_planar_robot_pose", qid)
        planar_base = planar(T_map_base)
        planar_base[2,3] = cfg.map_base_height_m
        candidate = planar_base @ np.linalg.inv(T_odom_base_capture)
        self._expire(now)
        if self.T_map_odom is not None:
            # Gate at the measured robot position, not at the arbitrary odom
            # origin: a small yaw correction far from the origin moves that
            # origin substantially without implying a translation innovation.
            delta_m, delta_deg = discrepancy(self.T_map_odom@T_odom_base_capture,
                                             candidate@T_odom_base_capture)
            if delta_m <= cfg.tracking_translation_m and delta_deg <= cfg.tracking_rotation_deg:
                return self._accept(candidate, stamp, qid, "tracking_correction")
            self.conflicts += 1
            if self.conflicts < cfg.conflicts_to_recover:
                return self._reject("tracking_innovation_conflict", qid)
            # A kidnapped robot must be allowed to escape its old map pose.
            self.T_map_odom = None
            self.global_state = "RECOVERING"
            self.hypotheses = []
        self.global_state = "RECOVERING"
        matches = []
        for hypothesis in self.hypotheses:
            d, r = discrepancy(hypothesis["transform"]@T_odom_base_capture,
                               candidate@T_odom_base_capture)
            if d <= cfg.recovery_translation_m and r <= cfg.recovery_rotation_deg:
                matches.append((d, hypothesis))
        if matches:
            hypothesis = min(matches, key=lambda x: x[0])[1]
            hypothesis["observations"].append((stamp, qid))
            hypothesis["transform"] = candidate  # actual newest estimate; never average modes
        else:
            hypothesis = {"transform": candidate, "observations": [(stamp, qid)]}
            self.hypotheses.append(hypothesis)
        mature = [h for h in self.hypotheses if len(h["observations"]) >= cfg.confirmations
                  and h["observations"][-1][0]-h["observations"][0][0] >= cfg.confirmation_span_s]
        if len(mature) == 1:
            best = mature[0]
            return self._accept(best["transform"], best["observations"][-1][0],
                                best["observations"][-1][1], "confirmed_global_recovery")
        return self._reject("ambiguous_recovery" if len(mature)>1 else "awaiting_temporal_confirmation", qid)

    def _accept(self, transform, stamp, qid, reason):
        self.T_map_odom = transform.copy()
        self.last_accepted_stamp = stamp
        self.global_state = "TRACKING"
        self.conflicts = 0
        self.hypotheses = []
        self.accepted_count += 1
        self.last_decision = {"accepted": True, "reason": reason, "query_id": qid,
                              "capture_time_s": stamp}
        return self.last_decision.copy()

    def _expire(self, now):
        cfg = self.config
        for h in self.hypotheses:
            h["observations"] = [x for x in h["observations"] if now-x[0] <= cfg.hypothesis_ttl_s]
        self.hypotheses = [h for h in self.hypotheses if h["observations"]]
        if self.T_map_odom is not None and now-self.last_accepted_stamp > cfg.max_dead_reckoning_s:
            self.T_map_odom = None
            self.global_state = "RECOVERING"

    def snapshot(self, now):
        self._expire(now)
        local = self.local_state(now)
        state = self.global_state
        if self.T_map_odom is not None and now-self.last_accepted_stamp > 1.5:
            state = "DEAD_RECKONING"
        current = self.history.samples[-1] if self.history.samples else None
        global_valid = local in {"TRACKING","DEGRADED"} and self.T_map_odom is not None
        return {"timestamp_s": float(now), "pose_time_s": None if current is None else current[0],
                "map_id": self.config.map_id, "local_state": local, "global_state": state,
                "local_valid": local in {"TRACKING","DEGRADED"}, "global_valid": global_valid,
                "local_quality": local,
                "local_pose_covariance_trace": None if current is None else current[2],
                "T_odom_base": None if current is None else current[1].tolist(),
                "T_map_odom": None if self.T_map_odom is None else self.T_map_odom.tolist(),
                "T_map_base": (self.T_map_odom @ current[1]).tolist() if global_valid else None,
                "last_absolute_capture_s": self.last_accepted_stamp,
                "recovery_hypotheses": len(self.hypotheses), "accepted_count": self.accepted_count,
                "rejected_count": self.rejected_count, "last_decision": self.last_decision,
                "backend": "wheel_imu_ekf_plus_gated_ace_map_odom",
                "spvloc_corrections_allowed": self.config.allow_spvloc_corrections,
                "spvloc_available": False, "macvo_available": False, "online_gt_used": False}

    def observe_candidates(self,row,now):
        """SPVLoc recovery branch: keep all modes, no ACE inliers or score probability."""
        cfg=self.config
        try:
            no_truth(row)
            stamp=finite(row["stamp_sec"],"capture time")
            qid=row["query_id"]
            if not isinstance(qid,str) or not qid:raise ValueError("empty query id")
            if row.get("source")!="official_spvloc":raise ValueError("not official SPVLoc")
            if row.get("map_id")!=cfg.map_id or (cfg.map_sha256 is not None and row.get("map_sha256")!=cfg.map_sha256):
                return self._reject("map_fingerprint_mismatch",qid)
            if qid in self.seen_queries or stamp<=self.last_processed_stamp:return self._reject("duplicate_or_out_of_order",qid)
            if stamp>now+.001 or now-stamp>cfg.max_observation_age_s:return self._reject("future_or_stale_observation",qid)
            if row.get("status")!="ok":return self._reject("localizer_failed",qid)
            if self.local_state(now) not in {"TRACKING","DEGRADED"}:return self._reject("local_odometry_unavailable",qid)
            odom=self.history.at(stamp)
            pool=row["candidates"]
            if not isinstance(pool,list) or not 1<=len(pool)<=3:raise ValueError("SPVLoc requires 1-3 candidates")
            if [x["rank"] for x in pool]!=list(range(1,len(pool)+1)):raise ValueError("candidate rank mismatch")
            transforms=[]
            for candidate in pool:
                finite(candidate["viewport_score"],"viewport score")
                value=(row.get("refined_top1_T_map_camera_optical") if candidate["rank"]==1 else None) or candidate["T_map_camera_optical"]
                base=pose(value)@np.linalg.inv(self.T_base_camera)
                tilt=math.degrees(math.acos(float(np.clip(base[2,2],-1,1))))
                if abs(base[2,3]-cfg.map_base_height_m)>cfg.max_base_height_error_m or tilt>cfg.max_base_tilt_deg:continue
                base=planar(base);base[2,3]=cfg.map_base_height_m
                transforms.append(base@np.linalg.inv(odom))
        except (KeyError,TypeError,ValueError) as exc:return self._reject(f"invalid_spvloc_contract:{exc}")
        self.seen_queries.add(qid)
        self.last_processed_stamp=stamp
        if not cfg.allow_spvloc_corrections:
            # The current model failed separate fixed-map development scoring.
            # Inference and full candidates stay in the audit, but an unqualified
            # model cannot create/update hypotheses or affect an ACE correction.
            # Explicit experimental opt-in restores the original candidate gate.
            return self._reject("spvloc_shadow_unqualified_model",qid)
        self._expire(now)
        if self.T_map_odom is not None:return self._reject("spvloc_recovery_only_currently_tracking",qid)
        if not transforms:return self._reject("no_planar_spvloc_candidate",qid)
        self.global_state="RECOVERING"
        for transform in transforms:
            matches=[]
            for h in self.hypotheses:
                d,r=discrepancy(h["transform"]@odom,transform@odom)
                if d<=cfg.recovery_translation_m and r<=cfg.recovery_rotation_deg:matches.append((d,h))
            if matches:
                h=min(matches,key=lambda x:x[0])[1]
                if qid not in {x[1] for x in h["observations"]}:
                    h["observations"].append((stamp,qid));h["transform"]=transform
            else:self.hypotheses.append({"transform":transform,"observations":[(stamp,qid)]})
        mature=[h for h in self.hypotheses if len(h["observations"])>=cfg.confirmations
                and h["observations"][-1][0]-h["observations"][0][0]>=cfg.confirmation_span_s]
        if len(mature)==1:
            h=mature[0]
            return self._accept(h["transform"],h["observations"][-1][0],h["observations"][-1][1],"confirmed_spvloc_recovery")
        return self._reject("ambiguous_spvloc_recovery" if len(mature)>1 else "awaiting_temporal_confirmation",qid)


def differential_wheel_velocity(left_rad_s, right_rad_s, radius_m, separation_m):
    values = [finite(x, "wheel input") for x in (left_rad_s, right_rad_s, radius_m, separation_m)]
    left, right, radius, separation = values
    if radius <= 0 or separation <= 0:
        raise ValueError("positive wheel calibration required")
    return radius*(left+right)/2., radius*(right-left)/separation
