"""Run official ACE trainer on this 6GB GPU with a predeclared buffer budget.
Only runtime workers change in memory; network/loss/optimization are upstream.
1000000 features x 16 epochs, batch 5120; explicitly NOT paper 8M-buffer recipe.
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
    from ace_trainer import TrainerACE

    original = TrainerACE.__init__

    def init(self, options):
        original(self, options)
        self.num_data_loader_workers = 2
        restore_step_counter(self.optimizer)

    TrainerACE.__init__ = init
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(Path(a.ace) / "train_ace.py"),
        a.scene,
        str(out / "heads_local.pt"),
        "--training_buffer_size",
        "1000000",
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
        buffer_features=1000000,
        epochs=16,
        batch_size=5120,
        base_seed=2089,
        workers=2,
        scene=a.scene,
        scope="Official ACE trainer and pretrained encoder; new scene head fit only to official train split; reduced buffer protocol, not semantic training",
    )
    (out / "training_resources.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("output")
    p.add_argument("--ace", required=True)
    main(p.parse_args())
