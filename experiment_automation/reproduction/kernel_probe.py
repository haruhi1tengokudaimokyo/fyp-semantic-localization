"""Real device/kernel checks, never a paper localization result."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch


def main(paper, output):
    tick = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if paper == "OMCL":
        import kaolin

        p = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.int16,
            device="cuda",
        )
        code = kaolin.ops.spc.points_to_morton(p)
        assert code.is_cuda and len(torch.unique(code)) == 4
        details = dict(kaolin=kaolin.__version__, morton_codes=code.cpu().tolist())
    else:
        from numba import cuda

        @cuda.jit
        def transform(x, y):
            i = cuda.grid(1)
            if i < x.size:
                y[i] = 2 * x[i] + 1

        x = np.arange(128, dtype=np.float32)
        dx = cuda.to_device(x)
        dy = cuda.device_array_like(x)
        transform[1, 128](dx, dy)
        np.testing.assert_array_equal(dy.copy_to_host(), 2 * x + 1)
        sys.path.insert(0, "/data/fyp_3d/sources/sg2loc")
        from sg2loc.particle_filter.raycasting import RayCasting

        details = dict(
            numba_cuda_kernel_executed=True,
            official_raycast_class_imported=RayCasting.__name__,
        )
    torch.cuda.synchronize()
    result = dict(
        status="kernel_verified",
        paper_id=paper,
        localization_executed=False,
        elapsed_seconds=time.monotonic() - tick,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        details=details,
        scope="Dependency/device check only; no scene, no accuracy and no completed paper claim",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paper")
    p.add_argument("output")
    a = p.parse_args()
    main(a.paper, a.output)
