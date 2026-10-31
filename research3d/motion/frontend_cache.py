"""Atomic dense learned-observation cache, never pose/GT cache.
A restart reconstructs the optimizer from cached prefix measurements instead of
rerunning the GPU frontend or trying to pickle unsafe asynchronous solver state.
"""

import dataclasses, hashlib, json, os, time
from pathlib import Path
import torch


def install_cache(frontend, directory, namespace):
    from Module.Frontend.StereoDepth import IStereoDepth
    from Module.Frontend.Matching import IMatcher

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    counts = {"hits": 0, "misses": 0, "corrupt": 0}
    ns = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
    root = root / ns
    root.mkdir(exist_ok=True)

    def key(kind, frames):
        h = hashlib.sha256(kind.encode())
        for f in frames:
            for x in [f.imageL, f.imageR]:
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
        default_K = torch.tensor([[[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]]])
        for f in frames:
            if not torch.equal(f.K.cpu(), default_K) or not torch.all(
                f.baseline.cpu() == 0.25
            ):
                h.update(f.K.cpu().contiguous().numpy().tobytes())
                h.update(f.baseline.cpu().contiguous().numpy().tobytes())
        return root / (h.hexdigest() + ".pt")

    def pack(x):
        return {
            f.name: (
                getattr(x, f.name).detach().cpu()
                if isinstance(getattr(x, f.name), torch.Tensor)
                else getattr(x, f.name)
            )
            for f in dataclasses.fields(x)
        }

    def wrap(original, kind):
        def call(*frames):
            path = key(kind, frames)
            if path.exists():
                try:
                    saved = torch.load(
                        path, map_location=frontend.config.device, weights_only=True
                    )
                    result = [IStereoDepth.Output(**saved[0])]
                    if kind == "pair":
                        result.append(IMatcher.Output(**saved[1]))
                    counts["hits"] += 1
                    return tuple(result) if kind == "pair" else result[0]
                except Exception:
                    counts["corrupt"] += 1
                    path.rename(path.with_suffix(".corrupt." + str(time.time_ns())))
            result = original(*frames)
            items = list(result) if kind == "pair" else [result]
            tmp = path.with_suffix(".tmp." + str(os.getpid()))
            torch.save([pack(x) for x in items], tmp)
            os.replace(tmp, path)
            counts["misses"] += 1
            return result

        return call

    frontend.estimate_depth = wrap(frontend.estimate_depth, "depth")
    frontend.estimate_pair = wrap(frontend.estimate_pair, "pair")
    return counts
