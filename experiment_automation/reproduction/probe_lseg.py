"""Actual full LSeg checkpoint, Batch=1 device/resource measurement, no benchmark claim."""

import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms


def main(weights, output):
    sys.path[:0] = ["/data/fyp_3d/sources/omcl/src", "/data/fyp_3d/sources/CLIP"]
    import clip, timm

    torch.set_num_threads(4)
    original_model = timm.create_model

    def without_redundant_imagenet(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_model(*args, **kwargs)

    timm.create_model = without_redundant_imagenet
    original_clip = clip.load

    def cached_clip(name, *args, **kwargs):
        kwargs["download_root"] = "/data/fyp_3d/models/clip"
        kwargs["device"] = "cpu"
        return original_clip(name, *args, **kwargs)

    clip.load = cached_clip
    from omcl.models.image_encoders.lseg.modules.models.lseg_net import LSegEncNet

    tick = time.monotonic()
    model = LSegEncNet(
        "", arch_option=0, block_depth=0, activation="lrelu", crop_size=480
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = {k.removeprefix("net."): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model = model.cuda().eval()
    # Canonical CLIP CUDA weight conversion; do not convert LayerNorm to half.
    clip.model.convert_weights(model.clip_pretrained)
    image = sorted(
        Path("/data/fyp_3d/datasets/7scenes/7scenes_heads/test/rgb").glob("*.png")
    )[0]
    rgb = Image.open(image).convert("RGB").crop((80, 0, 560, 480))
    x = transforms.Normalize([0.5] * 3, [0.5] * 3)(transforms.ToTensor()(rgb))[
        None
    ].cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        features, scores = model(x, ["wall", "chair", "table", "door"])
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    assert (
        features.shape[1] == 512
        and torch.isfinite(features).all()
        and torch.isfinite(scores).all()
    )
    result = dict(
        status="resource_probe_passed",
        localization_executed=False,
        original_LSeg_checkpoint_strict_loaded=True,
        model_parameters=sum(p.numel() for p in model.parameters()),
        image=str(image),
        crop=[80, 0, 560, 480],
        batch=1,
        feature_shape=list(features.shape),
        forward_seconds=elapsed,
        total_seconds=time.monotonic() - tick,
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        host_maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        dtype_scope="Upstream ViT FP32 / canonical CLIP CUDA mixed FP16; finite outputs checked, not an assumption that arbitrary FP16 works",
        adaptation="Skip redundant ImageNet download because complete original checkpoint is strictly loaded; reuse existing official CLIP weight cache",
        scope="One retained real RGB crop for resource measurement only; no Replica/Matterport localization accuracy or paper completion claim",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("output")
    a = p.parse_args()
    main(a.weights, a.output)
