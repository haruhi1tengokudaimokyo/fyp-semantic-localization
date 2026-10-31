"""Known 3D reference-map relocalization: SIFT geometry vs trained semantic gating.
Reference images/poses/depth form the read-only map. Query records have no pose
field; GT is written to a separate evaluation file and read only after inference.
This is a classical SIFT baseline with our semantic filter, not hloc reproduction.
"""

import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def rootsift(d):
    return np.sqrt(d / (d.sum(1, keepdims=True) + 1e-8)).astype(np.float32)


def features(sift, p):
    if hasattr(sift, "extract"):
        return sift.extract(p)
    image = cv2.imread(str(p))
    assert image is not None
    kp, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], np.float32), rootsift(desc)


def make_map(a, out):
    root = Path(a.dataset)
    m = json.loads((root / "manifest.json").read_text())
    frames = [r for r in m["frames"] if r["split"] == "test"]
    refs = [r for r in frames if r["position_id"] < a.reference_positions]
    queries = [r for r in frames if r["position_id"] >= a.reference_positions]
    sift = (
        a.feature_engine
        if hasattr(a, "feature_engine")
        else cv2.SIFT_create(nfeatures=600)
    )
    refdata = []
    all_d = []
    for r in refs:
        uv, d = features(sift, root / r["rgb"])
        depth = np.load(root / r["depth"])
        label = cv2.imread(str(root / r["label"]), cv2.IMREAD_UNCHANGED)
        xy = np.rint(uv).astype(int)
        z = depth[xy[:, 1], xy[:, 0]]
        valid = np.isfinite(z) & (z > 0.1) & (z < 20)
        uv = uv[valid]
        d = d[valid]
        xy = xy[valid]
        z = z[valid]
        K = np.array(r["K"])
        T = np.array(r["T_world_camera"])
        pc = np.stack(
            [(uv[:, 0] - K[0, 2]) * z / K[0, 0], (uv[:, 1] - K[1, 2]) * z / K[1, 1], z],
            1,
        )
        world = pc @ T[:3, :3].T + T[:3, 3]
        refdata.append(
            dict(
                rgb=r["rgb"],
                uv=uv,
                descriptors=d,
                points_world=world.astype(np.float32),
                semantic=label[xy[:, 1], xy[:, 0]],
                position_id=r["position_id"],
            )
        )
        all_d.append(d)
    cv2.setRNGSeed(1305)
    ds = np.concatenate(all_d)
    rng = np.random.default_rng(1305)
    sample = ds[rng.choice(len(ds), min(12000, len(ds)), replace=False)]
    _, _, vocab = cv2.kmeans(
        sample,
        64,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def bow(d):
        words = np.array([x.trainIdx for x in bf.match(d, vocab)], int)
        return np.bincount(words, minlength=64).astype(float)

    hist = np.stack([bow(d) for d in all_d])
    idf = np.log((1 + len(refs)) / (1 + (hist > 0).sum(0))) + 1
    hist = hist * idf
    hist /= np.maximum(1e-12, np.linalg.norm(hist, axis=1, keepdims=True))
    arrays = dict(vocabulary=vocab, idf=idf, reference_tfidf=hist)
    for i, r in enumerate(refdata):
        for k in ["uv", "descriptors", "points_world", "semantic"]:
            arrays[f"ref_{i}/{k}"] = r[k]
    np.savez_compressed(out / "readonly_map.npz", **arrays)
    # Queries are explicitly stripped of T_world_camera and semantic labels.
    (out / "queries.json").write_text(
        json.dumps(
            [
                dict(rgb=r["rgb"], K=r["K"], frame_id=str(i))
                for i, r in enumerate(queries)
            ],
            indent=2,
        )
    )
    (out / "ground_truth_eval_only.json").write_text(
        json.dumps(
            [dict(T_world_camera=r["T_world_camera"], rgb=r["rgb"]) for r in queries]
        )
    )
    protocol = dict(
        map_sha256=digest(out / "readonly_map.npz"),
        map_reference_images=len(refs),
        query_images=len(queries),
        reference_positions=sorted({r["position_id"] for r in refs}),
        query_positions=sorted({r["position_id"] for r in queries}),
        source_scene=frames[0]["scene"],
        references=[dict(rgb=r["rgb"], position_id=r["position_id"]) for r in refs],
        classes=m["classes"],
        semantic_map_source="verified native class IDs in offline prior map; unknown remains255",
        map_source="rendered reference RGB/depth/poses only; query images and poses excluded",
        retrieval="reference-only RootSIFT BoW 64 words, TF-IDF, top8",
        geometric_verification="Lowe0.75, EPNP-RANSAC2000iterations3px, >=12inliers, LM refinement, median3px and cheirality0.95 gates",
        semantic_gate="reject known-reference-class mismatch only if predicted query confidence>=0.7; unknown map labels retained",
        no_query_GT_online=True,
        no_initial_pose=True,
        localization_map_readonly=True,
        claim="single unseen synthetic scene pilot; not independent-building benchmark or hloc reproduction",
    )
    (out / "map_protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


@torch.no_grad()
def semantic_prediction(model, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255
    x = (
        x - torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
    probs = F.interpolate(
        model(pixel_values=x[None]).logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).softmax(1)[0]
    conf, pred = probs.max(0)
    return conf.cpu().numpy(), pred.cpu().numpy(), probs.cpu().numpy()


def localize(
    query,
    root,
    mapdata,
    model,
    sift,
    clip_features=None,
    modes=("geometry", "semantic"),
    topk=8,
):
    assert "T_world_camera" not in query
    start = time.perf_counter()
    uv, desc = features(sift, root / query["rgb"])
    image = cv2.imread(str(root / query["rgb"]))
    conf, pred, probabilities = (
        semantic_prediction(model, image)
        if any(m in modes for m in ["semantic", "soft"])
        else (None, None, None)
    )
    vocab = mapdata["vocabulary"]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    words = np.array([x.trainIdx for x in bf.match(desc, vocab)], int)
    hist = np.bincount(words, minlength=64) * mapdata["idf"]
    hist /= max(1e-12, np.linalg.norm(hist))
    retrieval_scores = (
        mapdata["reference_tfidf"] @ hist
        if clip_features is None
        else clip_features["reference_features"]
        @ clip_features["query_features"][int(query["frame_id"])]
    )
    candidates = np.argsort(retrieval_scores)[-topk:][::-1]
    K = np.array(query["K"], float)
    best = {}
    diag = {"matched": 0, "semantic_rejected": 0}
    match_cache = {}
    for mode in modes:
        winner = None
        for rank, r in enumerate(candidates):
            rd = mapdata[f"ref_{r}/descriptors"]
            if int(r) not in match_cache:
                if hasattr(sift, "match"):
                    qidx, ridx = sift.match(uv, desc, mapdata[f"ref_{r}/uv"], rd)
                else:
                    pairs = bf.knnMatch(desc, rd, k=2)
                    good = [
                        pair[0]
                        for pair in pairs
                        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
                    ]
                    qidx = np.array([p.queryIdx for p in good], int)
                    ridx = np.array([p.trainIdx for p in good], int)
                match_cache[int(r)] = (qidx.copy(), ridx.copy())
            qidx, ridx = (x.copy() for x in match_cache[int(r)])
            if mode == "semantic" and len(qidx):
                xy = np.rint(uv[qidx]).astype(int)
                label = mapdata[f"ref_{r}/semantic"][ridx]
                reject = (
                    (label != 255)
                    & (conf[xy[:, 1], xy[:, 0]] >= 0.7)
                    & (pred[xy[:, 1], xy[:, 0]] != label)
                )
                diag["matched"] += len(qidx)
                diag["semantic_rejected"] += int(reject.sum())
                qidx = qidx[~reject]
                ridx = ridx[~reject]
            if len(qidx) < 12:
                continue
            points = mapdata[f"ref_{r}/points_world"][ridx].astype(np.float64)
            pixels = uv[qidx].astype(np.float64)
            cv2.setRNGSeed(1305 + rank)
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                points,
                pixels,
                K,
                None,
                iterationsCount=2000,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 12:
                continue
            ix = inliers[:, 0]
            rv, tv = cv2.solvePnPRefineLM(points[ix], pixels[ix], K, None, rv, tv)
            R = cv2.Rodrigues(rv)[0]
            pc = points[ix] @ R.T + tv[:, 0]
            proj = cv2.projectPoints(points[ix], rv, tv, K, None)[0][:, 0]
            error = np.linalg.norm(proj - pixels[ix], axis=1)
            if np.median(error) > 3 or np.mean(pc[:, 2] > 0) < 0.95:
                continue
            score = float(len(ix))
            if mode == "soft":
                from scipy.optimize import least_squares

                xy = np.rint(pixels[ix]).astype(int)
                labels = mapdata[f"ref_{r}/semantic"][ridx[ix]]
                weight = np.ones(len(ix))
                known = labels != 255
                weight[known] = (
                    0.25
                    + 0.75 * probabilities[labels[known], xy[known, 1], xy[known, 0]]
                )

                def weighted_residual(v):
                    projected = cv2.projectPoints(points[ix], v[:3], v[3:], K, None)[0][
                        :, 0
                    ]
                    return ((projected - pixels[ix]) * np.sqrt(weight[:, None])).ravel()

                fit = least_squares(
                    weighted_residual,
                    np.r_[rv.ravel(), tv.ravel()],
                    loss="huber",
                    f_scale=3,
                    max_nfev=30,
                )
                if not fit.success or not np.isfinite(fit.x).all():
                    continue
                rv = fit.x[:3, None]
                tv = fit.x[3:, None]
                R = cv2.Rodrigues(rv)[0]
                proj = cv2.projectPoints(points[ix], rv, tv, K, None)[0][:, 0]
                error = np.linalg.norm(proj - pixels[ix], axis=1)
                pc = points[ix] @ R.T + tv[:, 0]
                if np.median(error) > 3 or np.mean(pc[:, 2] > 0) < 0.95:
                    continue
                score = float(weight.sum())
            if winner is None or score > winner["selection_score"]:
                T = np.eye(4)
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ tv[:, 0]
                winner = dict(
                    T_world_camera=T.tolist(),
                    inliers=len(ix),
                    selection_score=score,
                    matched=len(qidx),
                    median_reprojection_px=float(np.median(error)),
                    reference_index=int(r),
                )
        best[mode] = winner
    return dict(
        frame_id=query["frame_id"],
        rgb=query["rgb"],
        hypotheses=best,
        diagnostics=diag,
        runtime_ms=(time.perf_counter() - start) * 1000,
    )


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if a.hloc_source:
        from knownmap.learned_features import HlocFeatures

        a.feature_engine = HlocFeatures(
            a.hloc_source,
            a.lightglue_source,
            out / "learned_feature_loading_audit.json",
        )
    if a.reference_map:
        import shutil

        original = Path(a.reference_map)
        for name in ["readonly_map.npz", "queries.json", "ground_truth_eval_only.json"]:
            shutil.copyfile(original / name, out / name)
        protocol = json.loads((original / "map_protocol.json").read_text())
    else:
        protocol = make_map(a, out)
    if a.hloc_source:
        protocol["local_features"] = (
            "actual hloc SuperPoint600 + LightGlue, default learned score threshold; no Lowe ratio"
        )
        protocol["retrieval"] = "reference-only SuperPoint descriptor BoW64 TF-IDF top8"
        (out / "map_protocol.json").write_text(json.dumps(protocol, indent=2))
    if a.clip_features:
        protocol["retrieval"] = (
            "frozen official CLIP ViT-B/32 image cosine top8, same read-only 3D reference map"
        )
    protocol["topk"] = a.topk
    protocol["reference_positions_threshold"] = a.reference_positions
    protocol["semantic_soft_weight"] = (
        "0.25+0.75*predicted probability of known reference class, weighted Huber PnP refinement on unchanged RANSAC inliers; unknown class weight1; no pose covariance claim"
    )
    (out / "map_protocol.json").write_text(json.dumps(protocol, indent=2))
    clip_features = (
        np.load(a.clip_features, allow_pickle=False) if a.clip_features else None
    )
    root = Path(a.dataset)
    modes = (
        ["geometry", "semantic", "soft"]
        if a.mode == "all"
        else (["geometry", "semantic"] if a.mode == "both" else [a.mode])
    )
    model = None
    if any(m in modes for m in ["semantic", "soft"]):
        if not a.checkpoint:
            raise ValueError("Semantic route requires a trained --checkpoint")
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = SegformerForSemanticSegmentation(
            SegformerConfig.from_dict(checkpoint["config"])
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.cuda().eval()
    z = np.load(out / "readonly_map.npz", allow_pickle=False)
    queries = json.loads((out / "queries.json").read_text())
    sift = (
        a.feature_engine
        if hasattr(a, "feature_engine")
        else cv2.SIFT_create(nfeatures=600)
    )
    rows = []
    for q in queries:
        rows.append(localize(q, root, z, model, sift, clip_features, modes, a.topk))
        if len(rows) % 10 == 0:
            print("QUERY", len(rows), "/", len(queries), flush=True)
    (out / "online_results.json").write_text(json.dumps(rows, indent=2))
    assert digest(out / "readonly_map.npz") == protocol["map_sha256"], "Map mutated"
    # The evaluation-only file is loaded only after ALL estimates are finished.
    truth = json.loads((out / "ground_truth_eval_only.json").read_text())
    summary = {}
    for mode in modes:
        errors = []
        for r, g in zip(rows, truth):
            p = r["hypotheses"][mode]
            if p is None:
                errors.append((float("inf"), float("inf")))
                continue
            T = np.array(p["T_world_camera"])
            G = np.array(g["T_world_camera"])
            errors.append(
                (
                    float(np.linalg.norm(T[:3, 3] - G[:3, 3])),
                    float(
                        np.degrees(
                            Rotation.from_matrix(T[:3, :3].T @ G[:3, :3]).magnitude()
                        )
                    ),
                )
            )
        e = np.array(errors)
        valid = np.isfinite(e[:, 0])
        summary[mode] = dict(
            queries=len(rows),
            geometrically_verified=int(valid.sum()),
            failure_rate=float(1 - valid.mean()),
            recall_5cm_5deg=float(np.mean((e[:, 0] <= 0.05) & (e[:, 1] <= 5))),
            recall_25cm_10deg=float(np.mean((e[:, 0] <= 0.25) & (e[:, 1] <= 10))),
            recall_1m_30deg=float(np.mean((e[:, 0] <= 1) & (e[:, 1] <= 30))),
            recall_1m_20deg=float(np.mean((e[:, 0] <= 1) & (e[:, 1] <= 20))),
            median_translation_m_success_only=(
                float(np.median(e[valid, 0])) if valid.any() else None
            ),
            median_rotation_deg_success_only=(
                float(np.median(e[valid, 1])) if valid.any() else None
            ),
            errors_all_queries=[
                dict(
                    translation_m=float(t) if np.isfinite(t) else None,
                    rotation_deg=float(r) if np.isfinite(r) else None,
                )
                for t, r in errors
            ],
        )
    summary["audit"] = dict(
        readonly_map_verified=True,
        no_query_GT_online=True,
        semantic_rejected_matches=sum(
            r["diagnostics"]["semantic_rejected"] for r in rows
        ),
        tested_matches=sum(r["diagnostics"]["matched"] for r in rows),
        combined_modes_plus_semantic_runtime_mean_ms=float(
            np.mean([r["runtime_ms"] for r in rows])
        ),
        semantic_checkpoint_sha256=digest(Path(a.checkpoint)) if a.checkpoint else None,
        active_modes=modes,
        torch_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k == "audit" or isinstance(v, dict) and "queries" not in v
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--checkpoint")
    p.add_argument("--reference-positions", type=int, default=30)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=["geometry", "semantic", "soft", "both", "all"],
        default="both",
    )
    p.add_argument("--hloc-source")
    p.add_argument("--lightglue-source")
    p.add_argument("--reference-map")
    p.add_argument("--clip-features")
    main(p.parse_args())
