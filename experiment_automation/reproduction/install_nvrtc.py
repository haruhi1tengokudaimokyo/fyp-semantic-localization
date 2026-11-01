"""Repair the isolated SG2Loc compiler runtime, preserving earlier receipts."""

import ctypes, hashlib, json, subprocess, sys
from pathlib import Path

assert Path(sys.prefix).resolve() == Path("/data/fyp_3d/envs/sg2loc").resolve()
command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "nvidia-cuda-nvrtc-cu12==12.9.86",
]
subprocess.run(command, check=True, timeout=900)
root = Path(sys.prefix) / "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
library = root / "libnvrtc.so.12"
assert library.is_file()
lib = ctypes.CDLL(str(library))
major, minor = ctypes.c_int(), ctypes.c_int()
assert lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0
result = dict(
    status="compiler_runtime_loaded",
    version=[major.value, minor.value],
    library=str(library),
    sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    command=command,
    localization_executed=False,
)
p = Path(sys.argv[1])
p.parent.mkdir(parents=True, exist_ok=True)
t = p.with_suffix(".tmp")
t.write_text(json.dumps(result, indent=2))
t.replace(p)
print(json.dumps(result, indent=2))
