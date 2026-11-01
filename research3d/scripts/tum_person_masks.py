"""Frozen official DeepLabV3-MobileNet person exclusion, RGB-only and resumable.

These are model predictions, never ground-truth dynamic labels. All predicted
people are excluded, including sitting people; other moving objects may remain.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

URL = "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def acquire(path):
    import requests

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".part")
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as f:
                for block in response.iter_content(1024**2):
                    f.write(block)
        assert sha(temp).startswith("fc3c493d"), "Official weight hash prefix mismatch"
        temp.replace(path)
    assert sha(path).startswith("fc3c493d")
    atomic(
        path.with_suffix(".json"),
        dict(
            url=URL,
            sha256=sha(path),
            bytes=path.stat().st_size,
            official_hash_prefix_verified=True,
        ),
    )


def extract(dataset, weights, output):
    import torch
    import torch.nn.functional as F
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )

    torch.set_num_threads(2)
    root, out = Path(dataset), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "acceptance.json").read_text())
    assert meta["all_png_verified"] and meta["real_people"]
    digest = sha(weights)
    assert digest.startswith("fc3c493d")
    protocol = dict(
        model="torchvision DeepLabV3 MobileNetV3 Large COCO_WITH_VOC_LABELS_V1",
        weight_sha256=digest,
        rgb_list_sha256=sha(root / "rgb.txt"),
        person_class_index=15,
        decision="bilinear logits to original image; argmax==person; no threshold tuning or dilation",
        precision="FP32 CPU",
        threads=2,
        GT_online=False,
        depth_online=False,
        training=False,
        source="https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html",
    )
    if (out / "protocol.json").exists():
        assert (
            json.loads((out / "protocol.json").read_text()) == protocol
        ), "Mask checkpoint protocol changed"
    atomic(out / "protocol.json", protocol)
    official = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    assert official.meta["categories"][15] == "person"
    model = deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=21, aux_loss=True
    ).eval()
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    transform = official.transforms()
    paths = [
        line.split()[1]
        for line in (root / "rgb.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    rows = []
    for i, name in enumerate(paths):
        rgb = root / name
        target = out / f"{i:06d}.png"
        receipt = out / f"{i:06d}.json"
        image_sha = sha(rgb)
        if receipt.exists() and target.exists():
            row = json.loads(receipt.read_text())
            assert row["rgb_sha256"] == image_sha and row["weight_sha256"] == digest
            assert row["mask_sha256"] == sha(target)
        else:
            image = Image.open(rgb).convert("RGB")
            tick = time.perf_counter()
            with torch.inference_mode():
                logits = model(transform(image)[None])["out"]
                logits = F.interpolate(
                    logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = (logits.argmax(1)[0] == 15).numpy().astype(np.uint8) * 255
            elapsed = (time.perf_counter() - tick) * 1000
            temporary = target.with_suffix(".tmp.png")
            Image.fromarray(mask).save(temporary)
            temporary.replace(target)
            row = dict(
                frame=i,
                rgb=str(rgb.resolve()),
                rgb_sha256=image_sha,
                mask=str(target.resolve()),
                mask_sha256=sha(target),
                weight_sha256=digest,
                person_fraction=float((mask > 0).mean()),
                inference_ms=elapsed,
            )
            atomic(receipt, row)
        rows.append(row)
        if i % 100 == 0:
            print(
                "PERSON_MASK",
                i,
                "/",
                len(paths),
                "fraction",
                row["person_fraction"],
                flush=True,
            )
    atomic(out / "masks.json", rows)
    atomic(
        out / "summary.json",
        dict(
            status="completed",
            frames=len(rows),
            sequence=meta["sequence"],
            GT_online=False,
            frozen_official_weights=True,
            weight_sha256=digest,
            rgb_list_sha256=protocol["rgb_list_sha256"],
            person_fraction_mean=float(np.mean([r["person_fraction"] for r in rows])),
            inference_ms_mean=float(np.mean([r["inference_ms"] for r in rows])),
            protocol=protocol,
            scope="Predicted person masks; no segmentation/dynamic-label accuracy claim; CPU inference, not online real-time claim",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["acquire", "extract"])
    p.add_argument("weights")
    p.add_argument("--dataset")
    p.add_argument("--output")
    a = p.parse_args()
    if a.action == "acquire":
        acquire(a.weights)
    else:
        assert a.dataset and a.output
        extract(a.dataset, a.weights, a.output)
