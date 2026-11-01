"""Bind new continuous RGB queries to an existing SAME-scene reference map.

This is dataset packaging, never an online estimator input. Evaluation poses are
exported separately; reference descriptors, vocabulary and geometry stay unchanged.
"""

import hashlib
import json
from pathlib import Path
import numpy as np


def bind_continuous_queries(dataset, output, protocol):
    root, out = Path(dataset), Path(output)
    meta = json.loads((root / "acceptance.json").read_text())
    references = json.loads((root / "references.json").read_text())
    expected = [str(Path(r["rgb"]).resolve()) for r in protocol["references"]]
    actual = [str(Path(r["rgb"]).resolve()) for r in references]
    assert actual == expected, "Cross-scene or reordered reference map"
    digest = hashlib.sha256((root / "readonly_prior_map.npz").read_bytes()).hexdigest()
    assert digest == meta["readonly_map_sha256"]
    assert all(np.allclose(r["K"], meta["K"]) for r in references)
    n = meta["continuous_stereo_frames"]
    images = [root / "stereo/image_left" / f"{i:06d}_left.png" for i in range(n)]
    assert all(p.is_file() for p in images), "Missing continuous RGB measurement"
    truth = np.load(root / "ground_truth_eval_only.npz")["T_world_camera"]
    assert truth.shape == (n, 4, 4) and np.isfinite(truth).all()
    queries = [
        dict(rgb=str(p.resolve()), K=meta["K"], frame_id=str(i))
        for i, p in enumerate(images)
    ]
    evaluation = [
        dict(rgb=q["rgb"], T_world_camera=T.tolist()) for q, T in zip(queries, truth)
    ]
    for name, value in [
        ("queries.json", queries),
        ("ground_truth_eval_only.json", evaluation),
    ]:
        temp = out / (name + ".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(out / name)
    return {
        **protocol,
        "query_images": n,
        "query_positions": list(range(n)),
        "query_binding": "new continuous RGB; exact ordered original reference paths",
        "parent_prior_map_sha256": digest,
        "query_dataset": str(root.resolve()),
        "no_query_GT_online": True,
        "claim": "Same-building continuous integration; not unseen-building generalization",
    }
