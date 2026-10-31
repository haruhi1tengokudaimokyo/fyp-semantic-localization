"""Independent pretrained CLIP feature route bound to an immutable 3D reference map.
Inspired by OMCL's language-map observation principle; this pooled-reference
baseline does NOT reproduce OMCL's dense LSeg/octree map or particle filter.
"""

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image


@torch.no_grad()
def main(a):
    sys.path.insert(0, a.source)
    import clip

    torch.set_num_threads(4)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    maproot = Path(a.map)
    root = Path(a.dataset)
    protocol = json.loads((maproot / "map_protocol.json").read_text())
    queries = json.loads((maproot / "queries.json").read_text())
    assert all("T_world_camera" not in q for q in queries)
    model, preprocess = clip.load(
        "ViT-B/32", device="cpu", jit=False, download_root=a.weights
    )
    model = model.float().cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    timings = {}
    features = {}
    for kind, rows in [("reference", protocol["references"]), ("query", queries)]:
        vectors = []
        start = time.perf_counter()
        for begin in range(0, len(rows), 8):
            x = torch.stack(
                [
                    preprocess(Image.open(root / r["rgb"]).convert("RGB"))
                    for r in rows[begin : begin + 8]
                ]
            ).cuda()
            v = model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
            vectors.append(v.cpu().numpy())
        torch.cuda.synchronize()
        timings[kind] = dict(
            images=len(rows), seconds=time.perf_counter() - start, batch_size=8
        )
        features[kind + "_features"] = np.concatenate(vectors)
    # Frozen text prompts are separate from the trained 7-class network. Scores are
    # exported; no accuracy is invented for architecture classes lacking GT labels.
    prompts = [
        "a photo of a sofa",
        "a photo of a chair",
        "a photo of a bed",
        "a photo of a table",
        "a photo of a cabinet",
        "a photo of a shower",
        "a photo of a television",
        "a photo of a door",
        "a photo of a lamp",
        "a photo of a mirror",
        "a photo of a plant",
        "a photo of a painting",
        "a photo of a wall",
        "a photo of a window",
    ]
    texts = model.encode_text(clip.tokenize(prompts).cuda())
    texts = texts / texts.norm(dim=-1, keepdim=True)
    features["text_features"] = texts.cpu().numpy()
    features["query_text_cosine"] = (
        features["query_features"] @ features["text_features"].T
    )
    z = np.load(maproot / "readonly_map.npz")
    centroids = []
    for i in range(len(protocol["references"])):
        p = z[f"ref_{i}/points_world"]
        centroids.append(p.mean(0) if len(p) else np.full(3, np.nan))
    features["reference_3d_surface_centroids"] = np.array(centroids)
    np.savez_compressed(out / "features.npz", **features)
    w = Path(a.weights) / "ViT-B-32.pt"
    h = hashlib.sha256()
    with w.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    meta = dict(
        model="official OpenAI CLIP ViT-B/32",
        weight_sha256=h.hexdigest(),
        model_parameters=sum(p.numel() for p in model.parameters()),
        precision="FP32 all layers on GTX1660; no half precision",
        independent_of_trained_semantic_network=True,
        query_GT_online=False,
        readonly_geometry_map_sha256=protocol["map_sha256"],
        features_dimension=512,
        timings=timings,
        torch_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
        prompts=prompts,
        language_class_accuracy="not reported: no verified full architecture labels or region-level prediction evaluation",
        scope="posed 3D-reference image CLIP retrieval followed by learned local features and geometric PnP; NOT official OMCL reproduction",
    )
    (out / "protocol.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("map")
    p.add_argument("output")
    p.add_argument("--source", required=True)
    p.add_argument("--weights", required=True)
    main(p.parse_args())
