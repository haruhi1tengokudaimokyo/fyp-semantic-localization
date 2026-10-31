"""Check full official-vs-adapter metric parity and actively deny GT file reads."""

import argparse, csv, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from knownmap.ace import ACE
from knownmap.contracts import Camera, SensorObservation


def main(a):
    base = Path(a.base)
    official = [
        x.split()
        for x in (base / "runs/ace_official/poses_7scenes_heads_official.txt")
        .read_text()
        .splitlines()
    ]
    adapt = list(
        csv.DictReader((base / "runs/ace_adapter_official/per_frame.csv").open())
    )
    assert len(official) == len(adapt) == 1000
    tdiff = []
    rdiff = []
    for o, b in zip(official, adapt):
        assert o[0] == b["frame"]
        tdiff.append(abs(float(o[-2]) - float(b["translation_m"])))
        rdiff.append(abs(float(o[-3]) - float(b["rotation_deg"])))
    assert max(tdiff) < 1e-5 and max(rdiff) < 1e-3, (max(tdiff), max(rdiff))
    model = ACE(base / "sources/ace", base / "runs/ace_official/heads.pt")
    cam = Camera(640, 480, [[525, 0, 320], [0, 525, 240], [0, 0, 1]])
    obs = SensorObservation(
        "gt_access_denied",
        cam,
        base / "datasets/7scenes/7scenes_heads/test/rgb" / adapt[0]["frame"],
    )
    with patch(
        "numpy.loadtxt", side_effect=RuntimeError("GT and pose text reads denied")
    ):
        state, corr = model.predict(obs)
    assert state.hypotheses and not state.accepted
    result = dict(
        frames_compared=1000,
        max_translation_error_difference_m=max(tdiff),
        max_rotation_error_difference_deg=max(rdiff),
        gt_pose_text_reads_denied=True,
        inference_succeeded=True,
        accepted_tracking_correction=state.accepted,
        covariance_is_unavailable=state.hypotheses[0].covariance_se3 is None,
        inliers=state.diagnostics.inliers,
    )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("output")
    main(p.parse_args())
