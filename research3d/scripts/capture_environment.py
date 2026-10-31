import json, platform, subprocess, sys
from pathlib import Path
import torch, cv2, numpy

commands = {
    "disk": ["df", "-h", "/data"],
    "ram": ["free", "-h"],
    "gpu": [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv",
    ],
    "cpu": ["lscpu"],
    "upstream_ace": ["git", "-C", "/data/fyp_3d/sources/ace", "rev-parse", "HEAD"],
}
out = {
    k: subprocess.run(cmd, capture_output=True, text=True).stdout
    for k, cmd in commands.items()
}
out.update(
    python=sys.version,
    python_executable=sys.executable,
    os=platform.platform(),
    torch=torch.__version__,
    torch_cuda_runtime=torch.version.cuda,
    cuda_available=torch.cuda.is_available(),
    opencv=cv2.__version__,
    numpy=numpy.__version__,
    environment_note="New venv uses a .pth to read pre-existing semrayloc torch packages. Added packages install in the new venv; old ROS/environment untouched. See lock and setup script.",
)
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
