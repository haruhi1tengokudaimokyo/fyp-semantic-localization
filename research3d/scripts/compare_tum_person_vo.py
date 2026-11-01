"""Paired whole-sequence robustness results, preserving negative mask effects."""

import argparse
import json
from pathlib import Path
import numpy as np


def main(root, output):
    root, output = Path(root), Path(output)
    rows = []
    for sequence in ["sitting_xyz", "sitting_static", "walking_xyz", "walking_static"]:
        baseline, masked = root / sequence, root / (sequence + "_person")
        a = json.loads((baseline / "summary.json").read_text())
        b = json.loads((masked / "summary.json").read_text())
        za, zb = np.load(baseline / "trajectory.npz"), np.load(
            masked / "trajectory.npz"
        )
        assert np.array_equal(za["timestamp_s"], zb["timestamp_s"])
        assert np.array_equal(a["K"], b["K"])
        assert a["depth_scale"] == b["depth_scale"] == 5000
        assert (
            a["metrics"]["gt_evaluable_frames"] == b["metrics"]["gt_evaluable_frames"]
        )
        assert b["predicted_person_exclusion"] and not b["GT_online"]
        names = [
            "ate_se3_aligned_rmse_m",
            "ate_first_pose_anchored_rmse_m",
            "tracking_availability_all_rgb_frames",
            "failure_duration_seconds",
            "runtime_mean_ms",
        ]
        rows.append(
            dict(
                sequence=sequence,
                frames=len(za["timestamp_s"]),
                baseline={k: a["metrics"][k] for k in names},
                predicted_person_exclusion={k: b["metrics"][k] for k in names},
                paired_delta_masked_minus_baseline={
                    k: b["metrics"][k] - a["metrics"][k] for k in names
                },
                mask_inference_ms_mean=json.loads(
                    (root / "person_masks" / sequence / "summary.json").read_text()
                )["inference_ms_mean"],
            )
        )
    result = dict(
        status="completed",
        sequences=4,
        GT_online=False,
        same_original_RGB_frames=True,
        comparison=rows,
        scope="Four whole real TUM sitting/walking sequences; frozen predicted-person exclusion, not ground-truth dynamics; same fixed VO parameters; no prior-map localization or cross-dataset significance claim",
        timing="Masked VO runtime reads cached predicted masks; CPU mask inference reported separately; sum is not measured combined streaming latency",
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("output")
    a = p.parse_args()
    main(a.root, a.output)
