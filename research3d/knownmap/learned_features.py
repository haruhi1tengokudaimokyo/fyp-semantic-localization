"""Actual hloc SuperPoint and LightGlue modules for known RGB-D reference maps.
No SfM, online map updates, or query poses; hloc frontends only.
"""

import hashlib, json, sys
from pathlib import Path
import cv2, numpy as np, torch


class HlocFeatures:
    def __init__(self, hloc_source, lightglue_source, audit_path):
        torch.hub.set_dir("/data/fyp_3d/models/torch_hub")
        sys.path.insert(0, str(hloc_source))
        sys.path.insert(0, str(lightglue_source))
        from hloc.extractors.superpoint import SuperPoint
        from hloc.matchers.lightglue import LightGlue

        self.extractor = (
            SuperPoint(dict(max_keypoints=600, fix_sampling=False)).cuda().eval()
        )
        self.matcher = (
            LightGlue(
                dict(
                    features="superpoint",
                    depth_confidence=-1,
                    width_confidence=-1,
                    flash=False,
                )
            )
            .cuda()
            .eval()
        )
        weights = (
            Path(torch.hub.get_dir())
            / "checkpoints/superpoint_lightglue_v0-1_arxiv.pth"
        )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        for i in range(9):
            state = {
                k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn").replace(
                    f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                ): v
                for k, v in state.items()
            }
        report = self.matcher.net.load_state_dict(state, strict=False)
        missing = set(report.missing_keys)
        learned = set(dict(self.matcher.net.named_parameters()))
        assert not missing & learned, missing & learned
        assert set(report.unexpected_keys) <= {
            f"log_assignment.{i}.r" for i in range(9)
        }, report
        assert missing <= {"confidence_thresholds"}, report
        Path(audit_path).write_text(
            json.dumps(
                dict(
                    lightglue_missing_learned_parameters=sorted(missing & learned),
                    deterministic_missing_buffers=sorted(missing),
                    obsolete_unused_checkpoint_keys=report.unexpected_keys,
                    weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    features="official hloc SuperPoint + LightGlue frontend, max600, FP32, flash/early-stop/pruning disabled; no SfM reconstruction claim",
                ),
                indent=2,
            )
        )

    @torch.no_grad()
    def extract(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(image.copy()).float().cuda()[None, None] / 255
        features = self.extractor({"image": x})
        return (
            features["keypoints"][0].cpu().numpy(),
            features["descriptors"][0].T.cpu().numpy(),
        )

    @torch.no_grad()
    def match(self, uv0, d0, uv1, d1):
        if len(uv0) == 0 or len(uv1) == 0:
            return np.empty(0, int), np.empty(0, int)
        data = dict(
            image0=torch.empty((1, 1, 240, 320), device="cuda"),
            image1=torch.empty((1, 1, 240, 320), device="cuda"),
            keypoints0=torch.from_numpy(uv0)[None].cuda(),
            keypoints1=torch.from_numpy(uv1)[None].cuda(),
            descriptors0=torch.from_numpy(d0.T.copy())[None].cuda(),
            descriptors1=torch.from_numpy(d1.T.copy())[None].cuda(),
        )
        result = self.matcher(data)
        m = result["matches0"][0].cpu().numpy()
        q = np.flatnonzero(m >= 0)
        return q, m[q]
