"""GT-free adapter: official ACE network + official DSAC* RANSAC.
No learned scalar is promoted to covariance. covariance_se3 remains None.
"""

import sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from .contracts import (
    PoseHypothesis,
    LocalizationDiagnostics,
    LocalizationState,
    MapCorrespondence,
)


def load_image(observation):
    """Official ACE inference image transform, with no GT/pose file access."""
    from torchvision.transforms import functional as F

    rgb = Image.open(observation.rgb_path).convert("RGB")
    if rgb.size != (observation.camera.width, observation.camera.height):
        raise ValueError("Image/calibration size mismatch")
    return F.normalize(
        F.to_tensor(F.rgb_to_grayscale(rgb)), mean=[0.4], std=[0.25]
    ).float()


class ACE:
    def __init__(self, source, head):
        import torch

        source = Path(source)
        sys.path[:0] = [str(source), str(source / "dsacstar")]
        import dsacstar
        from ace_network import Regressor

        self.torch = torch
        self.dsacstar = dsacstar
        self.network = (
            Regressor.create_from_split_state_dict(
                torch.load(
                    source / "ace_encoder_pretrained.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                torch.load(head, map_location="cpu", weights_only=True),
            )
            .cuda()
            .eval()
        )
        self.network.requires_grad_(False)

    def predict(self, observation):
        import torch

        torch.cuda.synchronize()
        start = time.perf_counter()
        image = load_image(observation)[None].cuda()
        with torch.inference_mode(), torch.autocast("cuda"):
            sc = self.network(image).float().cpu()
        K = observation.camera.K
        if not np.isclose(K[0, 0], K[1, 1]):
            raise ValueError("DSAC* requires square focal length")
        pose = torch.zeros((4, 4))
        inliers = self.dsacstar.forward_rgb(
            sc,
            pose,
            64,
            10.0,
            float(K[0, 0]),
            float(K[0, 2]),
            float(K[1, 2]),
            100.0,
            100.0,
            8,
        )
        T = pose.numpy().astype(float)
        points = sc[0].permute(1, 2, 0).numpy().reshape(-1, 3)
        h, w = sc.shape[-2:]
        y, x = np.mgrid[:h, :w]
        pixels = np.stack([x.ravel() * 8 + 4, y.ravel() * 8 + 4], -1)
        eye = (points - T[:3, 3]) @ T[:3, :3]
        uv = eye @ K.T
        projected = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-10)
        error = np.linalg.norm(projected - pixels, axis=1)
        valid = np.isfinite(error) & (eye[:, 2] > 0)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        try:
            hypotheses = (
                PoseHypothesis(T, float(inliers), "DSAC_hard_inlier_count", None),
            )
        except ValueError:
            hypotheses = ()
        diag = LocalizationDiagnostics(
            elapsed,
            int(inliers),
            len(points),
            float(np.median(error[valid])) if valid.any() else float("inf"),
            "estimated" if inliers >= 4 and hypotheses else "failed",
        )
        return LocalizationState(
            observation.frame_id, hypotheses, diag
        ), MapCorrespondence(pixels[valid], points[valid], error[valid])
