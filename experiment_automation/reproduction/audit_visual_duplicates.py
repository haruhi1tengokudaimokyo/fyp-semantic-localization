"""Actual cross-split perceptual duplicate diagnostics, no image/model tuning."""

import argparse, collections, hashlib, json, sqlite3, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn


def audit(dataset, output, require_full=False):
    tick = time.monotonic()
    root = Path(dataset)
    frames = []
    sources = []
    # Only accepted v3 catalogs; never read failed-scene pixels into the corpus.
    for acceptance in sorted(root.glob("*/v3/acceptance.json")):
        a = json.loads(acceptance.read_text())
        assert (
            a["calibration_revision"] == "hypersim_camera_v3"
            and a["geometry_round_trip"]["quantization_bound_violations"] == 0
        )
        p = acceptance.parent / "supervision_manifest.json"
        rows = json.loads(p.read_text())
        sources.append(
            dict(
                scene=a["scene"],
                catalog_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                frames=len(rows),
            )
        )
        for r in rows:
            frames.append(
                dict(
                    scene=r["scene"],
                    camera=r["camera"],
                    frame=r["frame"],
                    rgb=r["rgb"],
                    partition=a["partition"],
                )
            )
    assert frames
    qualified = root / "qualified_v3/acceptance.json"
    if require_full:
        q = json.loads(qualified.read_text())
        assert q["status"] == "qualified_cleaned_supervision_subset" and q[
            "acquired_frames"
        ] == len(frames)
    thumbnails = []
    codes = []
    for i, r in enumerate(frames):
        with Image.open(r["rgb"]) as im:
            rgb = np.asarray(
                im.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
            )
            gray = np.asarray(
                im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float
            )
        coeff = dctn(gray, type=2, norm="ortho")[:8, :8].reshape(-1)[1:]
        code = sum(int(v > np.median(coeff)) << bit for bit, v in enumerate(coeff))
        codes.append(code)
        thumbnails.append(rgb)
        if i % 500 == 0:
            print("REAL_RGB_DESCRIPTOR", i, "/", len(frames), flush=True)
    codes = np.asarray(codes, dtype=np.uint64)
    thumb = np.asarray(thumbnails)
    part = np.asarray([r["partition"] for r in frames])
    lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    candidates = []
    nearest = collections.Counter()
    total_pairs = 0
    for split in ["train", "val", "test"]:
        ix = np.flatnonzero(part == split)
        others = np.flatnonzero(part != split)
        if not len(others):
            continue
        for start in range(0, len(ix), 128):
            batch = ix[start : start + 128]
            xor = codes[batch, None] ^ codes[others][None, :]
            dist = lut[xor.view(np.uint8).reshape(len(batch), len(others), 8)].sum(-1)
            for bi, i in enumerate(batch):
                best = int(dist[bi].min())
                nearest[best] += 1
                js = others[np.flatnonzero(dist[bi] <= 4)]
                for j in js:
                    if j <= i:
                        continue
                    rmse = float(
                        np.sqrt(
                            np.mean(
                                (
                                    (thumb[i].astype(float) - thumb[j].astype(float))
                                    / 255.0
                                )
                                ** 2
                            )
                        )
                    )
                    candidates.append(
                        dict(
                            first=frames[i],
                            second=frames[j],
                            phash_hamming=int(int(codes[i] ^ codes[j]).bit_count()),
                            thumbnail_rgb_rmse_0_to_1=rmse,
                        )
                    )
            total_pairs += len(batch) * len(others)
    candidates.sort(key=lambda r: (r["phash_hamming"], r["thumbnail_rgb_rmse_0_to_1"]))
    result = dict(
        status="diagnostic_completed" if total_pairs else "not_evaluable_single_split",
        cross_split_comparison_performed=bool(total_pairs),
        scope=(
            "full_qualified_subset"
            if require_full
            else "partial_currently_accepted_scenes_only"
        ),
        scenes=len(sources),
        frames=len(frames),
        partition_frames=dict(collections.Counter(part.tolist())),
        unique_cross_split_pairs_compared=total_pairs // 2,
        nearest_cross_split_hamming_histogram=dict(nearest),
        diagnostic_hamming_cutoff=4,
        phash_bits=63,
        phash_definition="32x32 LANCZOS grayscale; orthonormal 2D DCT top8x8 excludingDC, median sign bits",
        candidate_pairs=len(candidates),
        near_identical_thumbnail_pairs=sum(
            r["thumbnail_rgb_rmse_0_to_1"] <= 1 / 255 for r in candidates
        ),
        candidates=candidates,
        source_catalogs=sources,
        elapsed_seconds=time.monotonic() - tick,
        selection_changed=False,
        localization_executed=False,
        interpretation="Similarity flags require review. Uniform walls/shared assets can have identical pHash without being the same building. No automatic test-frame deletion, no inferred physical building identity, no localization performance used.",
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out.parent / (out.stem + "." + digest[:16] + ".json")).write_text(text)
    temp = out.with_suffix(".tmp")
    temp.write_text(text)
    temp.replace(out)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["candidates", "source_catalogs"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--require-full", action="store_true")
    a = p.parse_args()
    audit(a.dataset, a.output, a.require_full)
