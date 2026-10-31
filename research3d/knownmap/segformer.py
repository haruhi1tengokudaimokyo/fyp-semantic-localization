"""GT-free trained semantic model adapter producing the shared observation type."""

from pathlib import Path
import hashlib, time
import numpy as np
from .contracts import SemanticObservation


class TrainedSegFormer:
    def __init__(self, checkpoint, device="cuda"):
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.classes = tuple(saved["classes"])
        self.model = SegformerForSemanticSegmentation(
            SegformerConfig.from_dict(saved["config"])
        )
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, rgb, frame_id):
        import torch
        import torch.nn.functional as F

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected HWC uint8 RGB camera observation")
        with torch.inference_mode():
            x = (
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(self.device)
                / 255
            )
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[
                :, None, None
            ]
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = self.model(pixel_values=((x - mean) / std)[None]).logits
            probs = (
                F.interpolate(
                    logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
                )
                .softmax(1)[0]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        observation = SemanticObservation(
            str(frame_id), probs, self.classes, "trained_SegFormer_B0:" + self.sha256
        )
        return observation, dict(
            runtime_ms=(time.perf_counter() - start) * 1000,
            pose_covariance=None,
            scope="semantic inference only, device transfer/preprocessing excluded",
            GT_online=False,
        )
