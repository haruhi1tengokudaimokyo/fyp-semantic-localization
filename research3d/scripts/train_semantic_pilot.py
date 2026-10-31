"""Real SegFormer-B0 fine-tuning on verified partial HSSD labels.
Official NVIDIA MiT-B0 ImageNet weights; Transformers implementation. This is
our dataset/experiment adaptation, not a SemRayLoc or CVPR2026 reproduction.
"""

import argparse, copy, hashlib, json, random, resource, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation


class Frames(Dataset):
    def __init__(self, root, rows, augment=False):
        self.root = root
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = np.array(Image.open(self.root / r["rgb"]).convert("RGB"))
        y = np.array(Image.open(self.root / r["label"]), dtype=np.int64)
        if self.augment:
            if random.random() < 0.5:
                x = x[:, ::-1].copy()
                y = y[:, ::-1].copy()
            x = np.clip(x.astype(float) * random.uniform(0.8, 1.2), 0, 255)
        x = torch.from_numpy(x.copy()).permute(2, 0, 1).float() / 255
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        return x, torch.from_numpy(y.copy())


def metrics(cm, names):
    cm = cm.astype(float)
    tp = cm.diagonal()
    support = cm.sum(1)
    pred = cm.sum(0)
    union = support + pred - tp
    iou = np.divide(tp, union, out=np.full(len(names), np.nan), where=union > 0)
    return dict(
        mean_iou=float(np.nanmean(iou)),
        known_pixel_accuracy=float(tp.sum() / max(1, support.sum())),
        classes={
            n: dict(
                iou=float(iou[i]) if np.isfinite(iou[i]) else None,
                precision=float(tp[i] / pred[i]) if pred[i] else None,
                recall=float(tp[i] / support[i]) if support[i] else None,
                labeled_pixels=int(support[i]),
                predicted_on_labeled_pixels=int(pred[i]),
            )
            for i, n in enumerate(names)
        },
        confusion_matrix=cm.astype(int).tolist(),
        evaluation_scope="only verified labeled pixels; unknown is IGNORE, not background; no full-scene accuracy claim",
    )


@torch.no_grad()
def evaluate(model, loader, names, device, save_samples=None):
    model.eval()
    cm = np.zeros((len(names), len(names)), np.int64)
    times = []
    ece_n = np.zeros(10)
    ece_conf = np.zeros(10)
    ece_acc = np.zeros(10)
    saved = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        torch.cuda.synchronize()
        t = time.perf_counter()
        logits = F.interpolate(
            model(pixel_values=x).logits,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        probs = logits.softmax(1)
        conf, p = probs.max(1)
        torch.cuda.synchronize()
        times.extend([(time.perf_counter() - t) * 1000 / len(x)] * len(x))
        mask = y != 255
        a = y[mask].cpu().numpy()
        b = p[mask].cpu().numpy()
        c = conf[mask].cpu().numpy()
        cm += np.bincount(a * len(names) + b, minlength=len(names) ** 2).reshape(
            cm.shape
        )
        bins = np.minimum(9, (c * 10).astype(int))
        ece_n += np.bincount(bins, minlength=10)
        ece_conf += np.bincount(bins, weights=c, minlength=10)
        ece_acc += np.bincount(bins, weights=(a == b).astype(float), minlength=10)
        if save_samples is not None and saved < 8:
            # Probabilities are actual network predictions, not oracle masks.
            for j in range(min(len(x), 8 - saved)):
                np.savez_compressed(
                    save_samples / f"prediction_{saved:02d}.npz",
                    class_probabilities=probs[j].cpu().numpy().astype(np.float16),
                    predicted_label=p[j].cpu().numpy().astype(np.uint8),
                    ground_truth_eval_only=y[j].cpu().numpy().astype(np.uint8),
                )
                saved += 1
    m = metrics(cm, names)
    v = ece_n > 0
    m["known_pixel_ece_10_bins"] = float(
        np.sum(np.abs(ece_conf[v] / ece_n[v] - ece_acc[v] / ece_n[v]) * ece_n[v])
        / max(1, ece_n.sum())
    )
    m["inference_mean_ms_per_image_batch4"] = float(np.mean(times))
    m["inference_p95_ms_per_image_batch4"] = float(np.percentile(times, 95))
    return m


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.dataset)
    manifest = json.loads((root / "manifest.json").read_text())
    names = manifest["classes"]
    seed = 1305
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = "cuda"
    split = {
        s: [r for r in manifest["frames"] if r["split"] == s]
        for s in ["train", "val", "test"]
    }
    assert not (
        set(r["scene"] for r in split["train"]) & set(r["scene"] for r in split["test"])
    )
    assert not (
        set((r["scene"], r["position_id"]) for r in split["train"])
        & set((r["scene"], r["position_id"]) for r in split["val"])
    )
    loaders = {
        s: DataLoader(
            Frames(root, rs, s == "train"),
            batch_size=4,
            shuffle=s == "train",
            num_workers=0,
        )
        for s, rs in split.items()
    }
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        a.weights,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
        output_loading_info=True,
        local_files_only=True,
    )
    assert all(k.startswith("decode_head.") for k in info["missing_keys"]), info
    assert all(k.startswith("classifier.") for k in info["unexpected_keys"]), info
    assert not info.get("mismatched_keys"), info
    (out / "pretrained_loading_audit.json").write_text(json.dumps(info, indent=2))
    initial = copy.deepcopy(model.state_dict())
    model = model.to(device)
    counts = (
        np.array(
            [
                v[:7]
                for k, v in manifest["pixel_histogram"].items()
                if k.startswith("train/")
            ]
        )
        .sum(0)
        .astype(float)
    )
    w = np.sqrt(counts.sum() / np.maximum(1, counts))
    w = np.minimum(w / w.mean(), 3)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    protocol = dict(
        seed=seed,
        epochs=a.epochs,
        batch_size=4,
        learning_rate=6e-5,
        weight_decay=0.01,
        optimizer="AdamW",
        scheduler="polynomial power0.9",
        image_size=[240, 320],
        pretrained="official NVIDIA mit-b0 ImageNet-1K backbone, random SegFormer segmentation decoder",
        pretrained_revision="80983a413c30d36a39c20203974ae7807835e2b4",
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        class_weights=w.tolist(),
        augmentation="random horizontal flip and brightness[0.8,1.2] on training only",
        selection="best validation mIoU, test accessed only after selection",
        splits={s: len(rs) for s, rs in split.items()},
        limitations=manifest["limitations"],
        GT_pose_used_in_training=False,
        semantic_unknown_index=255,
        precision="FP32 all modules; no AMP on GTX1660",
        manifest_sha256=hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    total = a.epochs * len(loaders["train"])
    step = 0
    history = []
    best = -1
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    skipped = 0
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for x, y in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            if not (y != 255).any():
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = F.interpolate(
                model(pixel_values=x).logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = F.cross_entropy(logits, y, weight=w, ignore_index=255)
            assert torch.isfinite(loss), "nonfinite training loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = 6e-5 * (1 - step / total) ** 0.9
            losses.append(float(loss.detach()))
        val = evaluate(model, loaders["val"], names, device)
        row = dict(
            epoch=epoch + 1,
            mean_train_loss=float(np.mean(losses)),
            validation=val,
            elapsed_s=time.perf_counter() - start,
        )
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if val["mean_iou"] > best:
            best = val["mean_iou"]
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    epoch=epoch + 1,
                    classes=names,
                    config=model.config.to_dict(),
                ),
                out / "best.pt",
            )
        print(
            "EPOCH",
            epoch + 1,
            "LOSS",
            row["mean_train_loss"],
            "VAL_MIOU",
            best,
            "SEC",
            row["elapsed_s"],
            flush=True,
        )
    train_seconds = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test = evaluate(model, loaders["test"], names, device, save_samples=out)
    model.load_state_dict(initial, strict=True)
    untrained = evaluate(model, loaders["test"], names, device)
    majority = np.zeros((7, 7), np.int64)
    testcounts = np.array(
        [v[:7] for k, v in manifest["pixel_histogram"].items() if k.startswith("test/")]
    ).sum(0)
    majority[:, int(counts.argmax())] = testcounts
    result = dict(
        status="completed",
        best_epoch=checkpoint["epoch"],
        test=test,
        initial_random_decoder_test=untrained,
        train_majority_class_baseline_test=metrics(majority, names),
        training_seconds=train_seconds,
        optimizer_steps=step,
        skipped_all_ignore_batches=skipped,
        torch_peak_allocated_MiB=peak_alloc,
        torch_peak_reserved_MiB=peak_reserved,
        host_maxrss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        claim="partial-label cross-scene pilot only; no independent-building or localization claim",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--weights", required=True)
    p.add_argument("--epochs", type=int, default=40)
    main(p.parse_args())
