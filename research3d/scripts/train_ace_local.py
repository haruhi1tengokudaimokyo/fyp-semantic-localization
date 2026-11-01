"""Run official ACE trainer on this 6GB GPU with a predeclared buffer budget.
Only runtime workers change in memory; network/loss/optimization are upstream.
Default1M compatibility runs and original8M runs are recorded separately.
Both retain16 epochs and batch5120; storage placement is explicit.
"""

import argparse, json, resource, runpy, sys, time
from pathlib import Path


def restore_step_counter(optimizer):
    """Count actual optimizer steps; GradScaler overflow skips must not advance LR."""
    if not hasattr(optimizer, "_step_count"):
        optimizer._step_count = 0

        def count_actual_step(opt, args, kwargs):
            opt._step_count += 1

        optimizer.register_step_post_hook(count_actual_step)


def main(a):
    import torch

    torch.set_num_threads(4)
    sys.path[:0] = [str(Path(a.ace)), str(Path(a.ace) / "dsacstar")]
    import types

    cache = (
        Path(a.buffer_cache_dir)
        / (Path(a.scene).name + "_seed" + str(a.seed) + ".feature_buffer.tmp")
        if a.buffer_cache_dir
        else Path(a.output) / "heads_local.pt.feature_buffer.tmp"
    )

    trainer_path = Path(a.ace) / "ace_trainer.py"
    trainer_module = types.ModuleType("ace_trainer")
    trainer_module.__file__ = str(trainer_path)
    text = trainer_path.read_text()
    if a.buffer_storage in ("cpu", "mmap_cpu"):
        # Change only storage placement. Keep 8M samples, GPU RNG order, loss,
        # batch, optimizer and schedule identical to the upstream recipe.
        first = text.index("        self.training_buffer = {")
        last = text.index("        }", first) + len("        }")
        allocation = text[first:last]
        assert allocation.count("device=self.device") == 5
        text = (
            text[:first]
            + allocation.replace("device=self.device", "device='cpu'")
            + text[last:]
        )
        if a.buffer_storage == "mmap_cpu":
            original_features = "torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),\n                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device='cpu')"
            assert text.count(original_features) == 1
            replacement = f"torch.from_file({str(cache)!r}, shared=True, size=self.options.training_buffer_size*self.regressor.feature_dim, dtype=(torch.float32, torch.float16)[self.options.use_half]).reshape(self.options.training_buffer_size, self.regressor.feature_dim)"
            text = text.replace(original_features, replacement)
        for key in [
            "features",
            "target_px",
            "gt_poses_inv",
            "intrinsics",
            "intrinsics_inv",
        ]:
            before = (
                "self.training_buffer['" + key + "'][random_batch_indices].contiguous()"
            )
            assert text.count(before) == 1
            text = text.replace(
                before,
                "self.training_buffer['"
                + key
                + "'][random_batch_indices.cpu()].contiguous().to(self.device)",
            )
    assert text.count("self.base_seed = 2089") == 1
    exec(
        compile(
            text.replace("self.base_seed = 2089", "self.base_seed = " + str(a.seed)),
            str(trainer_path),
            "exec",
        ),
        trainer_module.__dict__,
    )
    sys.modules["ace_trainer"] = trainer_module
    from ace_trainer import TrainerACE

    original = TrainerACE.__init__

    def init(self, options):
        original(self, options)
        self.num_data_loader_workers = 2
        restore_step_counter(self.optimizer)

    TrainerACE.__init__ = init
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.buffer_storage == "mmap_cpu":
        import shutil

        required = a.buffer_features * 512 * 2
        cache.parent.mkdir(parents=True, exist_ok=True)
        required = max(0, required - (cache.stat().st_size if cache.exists() else 0))
        assert (
            shutil.disk_usage(cache.parent).free
            > required + (8 if a.buffer_cache_dir else 30) * 1024**3
        ), "Feature cache would violate disk reserve"
        import os

        with cache.open("ab") as handle:
            os.posix_fallocate(handle.fileno(), 0, a.buffer_features * 512 * 2)
    sys.argv = [
        str(Path(a.ace) / "train_ace.py"),
        a.scene,
        str(out / "heads_local.pt"),
        "--training_buffer_size",
        str(a.buffer_features),
        "--epochs",
        "16",
        "--batch_size",
        "5120",
    ]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runpy.run_path(str(Path(a.ace) / "train_ace.py"), run_name="__main__")
    result = dict(
        wall_seconds=time.perf_counter() - start,
        torch_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
        torch_peak_reserved_MiB=torch.cuda.max_memory_reserved() / 1024**2,
        process_maxrss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        buffer_features=a.buffer_features,
        buffer_storage=a.buffer_storage,
        temporary_feature_cache=str(cache) if a.buffer_storage == "mmap_cpu" else None,
        epochs=16,
        batch_size=5120,
        base_seed=a.seed,
        workers=2,
        scene=a.scene,
        scope="Official ACE trainer/encoder, original split. Buffer size recorded explicitly. Storage offload and optimizer-counter compatibility are hardware/runtime adaptations, not a changed model/loss/batch/schedule.",
    )
    (out / "training_resources.json").write_text(json.dumps(result, indent=2))
    print(result)
    if a.buffer_storage == "mmap_cpu":
        cache.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("output")
    p.add_argument("--ace", required=True)
    p.add_argument("--seed", type=int, default=2089)
    p.add_argument("--buffer-features", type=int, default=1000000)
    p.add_argument(
        "--buffer-cache-dir",
        help="Ephemeral workspace cache on a separate filesystem, 8GiB free-space reserve; all persistent model/results stay in output",
    )
    p.add_argument(
        "--buffer-storage", choices=["cuda", "cpu", "mmap_cpu"], default="cuda"
    )
    main(p.parse_args())
