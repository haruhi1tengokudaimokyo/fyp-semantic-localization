"""Equal-budget SPVLoc source experiments; importing/preflighting never trains.

Network, renderer and losses are the official SPVLoc implementation. The changes
are a source/building/view sampler, update-based budget/scheduler, deterministic
validation, and an explicit BN policy. ZInD raw downloads fail closed until an
independently qualified official conversion manifest is supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys


class NotReady(ValueError):
    """Configuration/data contract does not permit this experiment."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_config(c):
    for key in ("optimizer_updates", "micro_batch_size", "accumulate_grad_batches",
                "validation_interval_updates", "validation_views_per_building"):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise NotReady(f"{key} must be a positive integer")
    if c["precision"] != "32-true":
        raise NotReady("This audited GTX 1660 profile requires FP32; BF16 is unsupported")
    if c["bn_policy"] != "frozen_running_statistics_trainable_affine":
        raise NotReady("Changing BN policy requires a separately registered common profile")
    if c["num_workers"] != 0:
        raise NotReady("Only the deterministic single-process loader is validated")
    batch = c["micro_batch_size"] * c["accumulate_grad_batches"]
    if c["optimizer_updates"] % c["validation_interval_updates"]:
        raise NotReady("Budget must end on a validation boundary")
    if sorted(set(c["lr_milestone_updates"])) != c["lr_milestone_updates"]:
        raise NotReady("LR milestones must be strictly increasing")
    if any(not 0 < step < c["optimizer_updates"] for step in c["lr_milestone_updates"]):
        raise NotReady("LR milestones must be inside the optimizer budget")
    for arm, sources in c["arms"].items():
        if not sources or len(sources) != len(set(sources)) or batch % len(sources):
            raise NotReady(f"{arm}: each effective batch must have equal source quotas")
        if not set(sources) <= c["sources"].keys():
            raise NotReady(f"{arm}: unknown source")
    if not c["validation_sources"] or not set(c["validation_sources"]) <= c["sources"].keys():
        raise NotReady("Unknown or missing common validation sources")
    return batch


def source_manifest(source_id, source):
    """Read qualification, check lineage/splits and the official loader inputs."""
    manifest_path = Path(source["manifest"])
    if not manifest_path.is_file():
        raise NotReady(f"{source_id}: missing qualified conversion manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "DATA_READY":
        raise NotReady(f"{source_id}: qualification is not DATA_READY")
    rows = manifest.get("houses", []) if source_id == "procthor" else manifest.get("scenes", [])
    if not rows:
        raise NotReady(f"{source_id}: empty accepted scene list")
    partition = None
    if source_id == "zind":
        if manifest.get("converter") != "spvloc.tools.zind_to_s3d":
            raise NotReady("ZInD must use the official converter, not invented S3D labels")
        if manifest.get("offline_only_gt") is not True:
            raise NotReady("ZInD qualification must explicitly separate GT from online inputs")
        partition = json.loads(Path(source["official_partition"]).read_text())
        partition = {str(home).zfill(4): split for split, homes in partition.items() for home in homes}
    normalized, groups, scenes = [], {}, set()
    for row in rows:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise NotReady(f"Invalid split {split}")
        sid = int(row["scene_id"])
        if sid in scenes:
            raise NotReady(f"{source_id}: duplicate scene {sid}")
        scenes.add(sid)
        if source_id == "procthor":
            # ProcTHOR official indices are local to each official split. Include
            # source-geometry hash to catch re-labelled duplicates across splits.
            building = row.get("building_group_id", f"{split}:{row['official_house_index']}")
            lineage = row["source_geometry_sha256"]
            folder = Path(source["root"]) / f"scene_{sid:05d}"
            for filename, status in (("qualification.json", "DATA_READY_PILOT"),
                                     ("panorama_qualification.json", "PANORAMA_READY")):
                if json.loads((folder / filename).read_text()).get("status") != status:
                    raise NotReady(f"{source_id} scene {sid}: {filename} not qualified")
        else:
            building = str(row["original_home_id"]).zfill(4)
            lineage = building
            if sid // 10 != int(building) or partition.get(building) != split:
                raise NotReady(f"ZInD {sid}: official home/floor split mismatch")
            checks = row.get("checks", {})
            required = ("rgb_decode", "camera_pose", "structural_semantic_ids",
                        "coordinate_alignment", "projection", "official_loader")
            if any(checks.get(name) != "PASSED" for name in required):
                raise NotReady(f"ZInD {sid}: conversion QA is incomplete")
            evidence = row.get("evidence_files", [])
            if not evidence:
                raise NotReady(f"ZInD {sid}: missing qualification evidence hashes")
            for item in evidence:
                path = Path(source["root"]) / item["path"]
                if path.resolve().is_relative_to(Path(source["root"]).resolve()) is False:
                    raise NotReady("Qualification evidence escapes dataset root")
                if not path.is_file() or sha256(path) != item["sha256"]:
                    raise NotReady(f"ZInD {sid}: evidence hash mismatch")
        for group in (f"building:{building}", f"lineage:{lineage}"):
            if group in groups and groups[group] != split:
                raise NotReady(f"{source_id}: building/derived-data leakage across splits")
            groups[group] = split
        folder = Path(source["root"]) / f"scene_{sid:05d}"
        annotation = folder / "annotation_3d.json"
        if not annotation.is_file() or not (folder / "2D_rendering").is_dir():
            raise NotReady(f"{source_id} {sid}: not an official S3D-format scene")
        # The loader writes missing pickle caches. Refuse to create caches in
        # protected datasets during a training or read-only interface check.
        if not (Path(source["root"]) / source["prepared_cache"] / f"temp_{sid:05d}").is_file():
            raise NotReady(f"{source_id} {sid}: qualified official loader cache missing")
        normalized.append(dict(source_id=source_id, scene_id=sid, split=split,
                               building_id=str(building), lineage_id=str(lineage)))
    if {r["split"] for r in normalized} != {"train", "val", "test"}:
        raise NotReady(f"{source_id}: qualification must retain all three independent splits")
    return normalized


def preflight(c):
    validate_config(c)
    sources = {}
    for key, source in c["sources"].items():
        try:
            rows = source_manifest(key, source)
            sources[key] = dict(status="QUALIFIED_LOADER_INPUTS", counts=dict(Counter(r["split"] for r in rows)),
                                manifest_sha256=sha256(source["manifest"]))
        except (OSError, KeyError, ValueError) as exc:
            sources[key] = dict(status="BLOCKED", reason=str(exc))
        if key == "zind" and Path(source["raw_status"]).is_file():
            raw = json.loads(Path(source["raw_status"]).read_text())
            sources[key]["raw_download"] = {k: raw.get(k) for k in ("stage", "complete_houses", "total_houses", "raw_dataset_only")}
    source = Path(c["official_source"])
    errors = []
    try:
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != c["official_commit"]:
            errors.append("Official SPVLoc commit differs from the registered revision")
    except (OSError, subprocess.CalledProcessError):
        commit = None
        errors.append("Official SPVLoc checkout unavailable")
    if not Path(c["initialization"]).is_file() or sha256(c["initialization"]) != c["initialization_sha256"]:
        errors.append("Common initialization checkpoint missing or SHA256 mismatch")
    arms = {}
    for arm, names in c["arms"].items():
        blocked = [name for name in sorted(set(names + c["validation_sources"])) if sources[name]["status"] == "BLOCKED"]
        arms[arm] = dict(status="BLOCKED" if blocked or errors else "READY_FOR_EXPLICIT_RUNTIME_CHECK",
                         blocked_sources=blocked, runtime_errors=errors)
    z = c["sources"]["zind"]
    return dict(status="PREFLIGHT_ONLY_NO_TRAINING", official_commit=commit,
                sources=sources, arms=arms, conversion_commands=[
                    [c["python"], "-m", "spvloc.tools.zind_to_s3d", "-i", z["raw_root"], "-o", z["root"], "-idx", "0"],
                    [c["python"], "-m", "spvloc.tools.dataset_from_csv", "-d", "data/datasets/zind/dataset_info.csv",
                     "-i", z["root"], "-o", z["root"] + "_official_test"]],
                conversion_cwd=c["official_source"], conversion_is_not_qualification=True)


@dataclass(frozen=True)
class View:
    source_id: str
    building_id: str
    scene_id: int
    room_id: int
    dataset_index: int


class HierarchicalSchedule:
    """Equal source quotas per optimizer update, uniform building then view.

    Repeated crops derived from one panorama do not increase its sampling weight.
    Source quota applies before the unchanged official semantic validity filter.
    """
    def __init__(self, views, sources, effective_batch, updates, seed):
        if not sources or effective_batch % len(sources) or len(sources) != len(set(sources)):
            raise NotReady("Unequal source quota")
        self.sources = sorted(sources)
        self.effective_batch, self.updates, self.seed = effective_batch, updates, seed
        self.groups = {s: defaultdict(list) for s in self.sources}
        seen = set()
        for view in views:
            if view.source_id not in self.groups:
                continue
            key = (view.source_id, view.scene_id, view.room_id)
            if key not in seen:
                self.groups[view.source_id][view.building_id].append(view)
                seen.add(key)
        if any(not buildings for buildings in self.groups.values()):
            raise NotReady("Source has no qualified training views")

    def generate(self, epoch=0):
        rng = random.Random(self.seed + epoch * 1000003)
        names = {s: sorted(groups) for s, groups in self.groups.items()}
        for _ in range(self.updates):
            order = self.sources * (self.effective_batch // len(self.sources))
            rng.shuffle(order)
            for source in order:
                building = rng.choice(names[source])
                yield rng.choice(self.groups[source][building])


def validation_macro(losses, expected_sources):
    """Prevent large buildings or a larger source from dominating selection."""
    by_source = defaultdict(list)
    for (source_id, _building), values in losses.items():
        if not values or any(not math.isfinite(float(value)) for value in values):
            raise NotReady("Empty/nonfinite validation building loss")
        by_source[source_id].append(sum(values) / len(values))
    if set(by_source) != set(expected_sources):
        raise NotReady("Validation source missing; cannot select a checkpoint")
    means = {name: sum(values) / len(values) for name, values in by_source.items()}
    return sum(means.values()) / len(means), means


def plan(c, seed, config_path="configs/fusion/training_budget.json"):
    effective = validate_config(c)
    return dict(protocol="equal_update_source_comparison_v1", seed=seed,
                initialization_sha256=c["initialization_sha256"],
                optimizer_updates=c["optimizer_updates"], attempted_queries=c["optimizer_updates"] * effective,
                effective_batch=effective, micro_batch=c["micro_batch_size"],
                bn_policy=c["bn_policy"], validation_sources=c["validation_sources"],
                selection="minimum source-macro/building-macro validation original total loss; no test access",
                arms={arm: dict(training_sources=sources,
                                expected_attempted_queries_per_source=c["optimizer_updates"] * effective // len(sources),
                                command=["env", *[f"{key}={value}" for key, value in c["runtime_environment"].items()],
                                         c["python"], "-m", "research3d.fusion.training", "train",
                                         "--config", str(config_path), "--arm", arm,
                                         "--seed", str(seed), "--output", str(Path(c["run_root"]) / f"{arm.replace('+','_')}_{seed}")])
                      for arm, sources in c["arms"].items()},
                limitations=["This is an adapted equal-budget experiment, not an original-data paper benchmark",
                             "Frozen BN running statistics differ from paper native batch-40 training",
                             "Post-filter valid counts must be reported per source; equal attempts are not equal usable samples",
                             "MAC-VO remains frozen; no VLM training; official S3D initialization already has S3D exposure"])


def load_runtime(c):
    # Lazy imports keep preflight/plan usable without CUDA or SPVLoc installed.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_HOME", "/data/fyp_3d/models/torchvision")
    sys.path.insert(0, c["official_source"])
    os.chdir(c["official_source"])
    from spvloc.config.defaults import get_cfg_defaults
    import spvloc.data.dataset as official_data
    from spvloc.model.spvloc import PerspectiveImageFromLayout
    from research3d.semantic_localization.float_targets import install
    dataset_class = install()
    return get_cfg_defaults, official_data, PerspectiveImageFromLayout, dataset_class


def make_config(c, source_id, get_defaults):
    cfg = get_defaults()
    cfg.merge_from_file(str(Path(c["official_source"]) / "configs/config_s3d.yaml"))
    spec = c["sources"][source_id]
    cfg.DATASET.PATH = spec["root"]
    cfg.DATASET.NAME = spec["official_dataset_name"]
    cfg.DATASET.PREPARED_DATA_PATH = spec["prepared_cache"]
    cfg.DATASET.TRAIN_FURNITURE = ["full"]
    cfg.DATASET.TEST_SET_FURNITURE = ["full"]
    cfg.DATASET.PERSP_FROM_PANO_CORRECT_ROLL = c["correct_roll"]
    cfg.SYSTEM.NUM_WORKERS = 0
    cfg.TRAIN.BATCH_SIZE = c["micro_batch_size"]
    cfg.TRAIN.INITIAL_LR = c["initial_lr"]
    cfg.TEST.IGNORE_PRETRAINED = True
    cfg.TEST.PLOT_OUTPUT = cfg.TEST.SAVE_PLOT_OUTPUT = False
    cfg.TEST.VAL_AS_TEST = False
    return cfg


def build_datasets(c, source_ids, split, runtime, limit_scenes=None):
    get_defaults, official_data, _, dataset_class = runtime
    datasets, views = {}, []
    for source_id in source_ids:
        rows = [r for r in source_manifest(source_id, c["sources"][source_id]) if r["split"] == split]
        if limit_scenes is not None:
            rows = rows[:limit_scenes]
        by_scene = {r["scene_id"]: r for r in rows}
        original_split = official_data.scenes_split
        try:
            official_data.scenes_split = lambda *_args, **_kwargs: list(by_scene)
            dataset = dataset_class(make_config(c, source_id, get_defaults), split, visualise=True)
        finally:
            official_data.scenes_split = original_split
        datasets[source_id] = dataset
        # In panorama mode persp_idx is ignored by upstream __getitem__. Collapse
        # those repeated index rows before source/building/panorama sampling.
        seen = set()
        for i, (sid, room, _perspective) in enumerate(dataset.scenes):
            if (sid, room) not in seen:
                views.append(View(source_id, by_scene[sid]["building_id"], sid, room, i))
                seen.add((sid, room))
        if not seen:
            raise NotReady(f"Official loader has no {split} panoramas for {source_id}")
    return datasets, views


def runtime_check(c, source_id):
    """Load one real scene index and checkpoint on CPU; no rendering/optimizer."""
    source_manifest(source_id, c["sources"][source_id])
    if sha256(c["initialization"]) != c["initialization_sha256"]:
        raise NotReady("Initialization checksum mismatch")
    runtime = load_runtime(c)
    _, views = build_datasets(c, [source_id], "train", runtime, limit_scenes=1)
    import torch
    model = runtime[2](make_config(c, source_id, runtime[0]))
    saved = torch.load(c["initialization"], map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    return dict(status="CPU_OFFICIAL_LOADER_AND_CHECKPOINT_PASSED", source=source_id,
                real_scene=views[0].scene_id, panorama_views=len(views),
                parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=0,
                rendered_batches=0, full_training_qualification=False)


@contextmanager
def fixed_random(seed):
    import numpy as np
    import torch
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def train(c, arm, seed, output):
    """Explicit opt-in training; never called by preflight/plan/runtime-check."""
    report = preflight(c)
    if report["arms"][arm]["status"] == "BLOCKED":
        raise NotReady(json.dumps(report["arms"][arm]))
    output = Path(output).resolve()
    if output.exists():
        raise NotReady("Run output already exists; refusing silent restart/overwrite")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < (c["minimum_free_gib"] + c["reserved_run_gib"]) * 2**30:
        raise NotReady("Insufficient space after preserving the registered reserve")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if occupied:
        raise NotReady("GPU occupied; protected workloads will not be stopped")
    runtime = load_runtime(c)
    import torch
    import numpy as np
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning.loggers import CSVLogger
    pl.seed_everything(seed, workers=True)
    output.mkdir(parents=True)
    effective = validate_config(c)
    training_data, training_views = build_datasets(c, c["arms"][arm], "train", runtime)
    validation_data, validation_views = build_datasets(c, c["validation_sources"], "val", runtime)
    schedule = HierarchicalSchedule(training_views, c["arms"][arm], effective, c["validation_interval_updates"], seed)
    val_groups = defaultdict(list)
    for view in validation_views:
        val_groups[(view.source_id, view.building_id)].append(view)
    val_selected = []
    for key in sorted(val_groups):
        candidates = val_groups[key]
        # Fixed sorted/evenly spaced panoramas; no localization score selection.
        indices = np.linspace(0, len(candidates) - 1, min(c["validation_views_per_building"], len(candidates)), dtype=int)
        val_selected.extend(candidates[int(i)] for i in indices)

    class ScheduledData(Dataset):
        def __init__(self, data, views, validation=False):
            self.data, self.views, self.validation = data, views, validation

        def __len__(self):
            return len(self.views)

        def __getitem__(self, index):
            view = self.views[index]
            if self.validation:
                with fixed_random(99173 + index):
                    batch = self.data[view.source_id][view.dataset_index]
            else:
                batch = self.data[view.source_id][view.dataset_index]
            return dict(batch, _source_id=view.source_id, _building_id=view.building_id)

    train_data = ScheduledData(training_data, list(schedule.generate()))
    val_data = ScheduledData(validation_data, val_selected, validation=True)

    class Model(runtime[2]):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.valid_losses = defaultdict(list)
            self.sample_counts = defaultdict(Counter)

        def train(self, mode=True):
            super().train(mode)
            if mode:
                for module in self.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()  # affine parameters remain trainable
            return self

        def configure_optimizers(self):
            optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=c["initial_lr"])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, c["lr_milestone_updates"], c["lr_gamma"])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        def train_dataloader(self):
            return DataLoader(train_data, batch_size=c["micro_batch_size"], shuffle=False, num_workers=0)

        def val_dataloader(self):
            return DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        def on_train_epoch_start(self):
            train_data.views = list(schedule.generate(self.current_epoch))

        def training_step(self, batch, batch_idx):
            sources = batch.pop("_source_id")
            batch.pop("_building_id")
            h = batch["reference_histogram"]
            valid = (h[:, 0] < .05) & ((h[:, 1:] > .002).sum(-1) > 2) & (h[:, 4] < .7) & (h[:, 5] < .7)
            valid &= batch["room_idx"].reshape(-1) != -1
            for source_id, flag in zip(sources, valid.tolist()):
                self.sample_counts[source_id]["attempted"] += 1
                self.sample_counts[source_id]["passes_official_filter"] += int(flag)
                self.sample_counts[source_id]["actually_forwarded"] += int(flag or not valid.any())
            result = super().training_step(batch, batch_idx)
            if not torch.isfinite(result["loss"]).all():
                raise NotReady("Nonfinite official training loss; refusing further optimizer updates")
            return result

        def on_validation_epoch_start(self):
            self.valid_losses.clear()

        def validation_step(self, batch, batch_idx):
            source_id = batch.pop("_source_id")[0]
            building_id = batch.pop("_building_id")[0]
            loss, _ = self.common_step(batch, "validation")
            if not torch.isfinite(loss).all():
                raise NotReady("Nonfinite validation loss; no checkpoint selection allowed")
            self.valid_losses[(source_id, building_id)].append(float(loss.detach().cpu()))
            return loss

        def on_validation_epoch_end(self):
            macro, source_means = validation_macro(self.valid_losses, c["validation_sources"])
            self.log("validation/source_building_macro_loss", macro, on_epoch=True)
            write_json(output / "validation_latest.json", dict(optimizer_step=self.global_step,
                       source_macro_loss=source_means, buildings=len(self.valid_losses), test_used=False))

    class Evidence(Callback):
        def on_train_epoch_end(self, trainer, module):
            write_json(output / "progress.json", dict(optimizer_steps=trainer.global_step,
                       source_counts={s: dict(v) for s, v in module.sample_counts.items()},
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       note="Official all-invalid microbatch behavior is retained; actually_forwarded records its fallback"))

    model = Model(make_config(c, c["arms"][arm][0], runtime[0]))
    checkpoint = torch.load(c["initialization"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    before = {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name, p in model.named_parameters()}
    del checkpoint
    write_json(output / "protocol.json", dict(config=c, plan=plan(c, seed), selected_arm=arm, preflight=report,
               official_network_and_losses=True, validation_queries=len(val_selected),
               excluded_test_from_training_and_selection=True, macvo_frozen=True))
    best = ModelCheckpoint(dirpath=output / "checkpoints", monitor="validation/source_building_macro_loss",
                           mode="min", save_top_k=1, save_last=True, filename="step{step}")
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=c["precision"],
                         max_steps=c["optimizer_updates"], max_epochs=-1,
                         accumulate_grad_batches=c["accumulate_grad_batches"],
                         num_sanity_val_steps=0, callbacks=[best, Evidence()],
                         logger=CSVLogger(str(output), name="loss_logs"), enable_progress_bar=False,
                         log_every_n_steps=1, check_val_every_n_epoch=1)
    trainer.fit(model)
    changed = [name for name, p in model.named_parameters()
               if hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() != before[name]]
    if trainer.global_step != c["optimizer_updates"] or not changed or not best.best_model_path:
        raise NotReady("Update budget/parameter update/validation checkpoint acceptance failed")
    write_json(output / "training_evidence.json", dict(status="TRAINED_HELD_OUT_POSE_EVALUATION_PENDING",
               optimizer_steps=trainer.global_step, updated_tensors=changed,
               best_checkpoint_by_validation=best.best_model_path, test_used=False,
               updated_parameter_count=sum(p.numel() for name, p in model.named_parameters() if name in changed)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "plan", "runtime-check", "train"])
    parser.add_argument("--config", type=Path, default=Path("configs/fusion/training_budget.json"))
    parser.add_argument("--arm", default="P", choices=["P", "Z", "P+Z"])
    parser.add_argument("--source", default="procthor", choices=["procthor", "zind"])
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # The official renderer requires its source directory as cwd. Resolve user
    # paths before entering it, so --output never accidentally writes to /data.
    args.config = args.config.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    c = json.loads(args.config.read_text())
    validate_config(c)
    try:
        if args.action == "preflight":
            result = preflight(c)
        elif args.action == "plan":
            result = plan(c, args.seed, args.config)
        elif args.action == "runtime-check":
            result = runtime_check(c, args.source)
        else:
            if args.output is None:
                parser.error("train requires a new --output directory")
            train(c, args.arm, args.seed, args.output)
            return 0
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except NotReady as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
