"""Isolate evo's historical alignment semantics; never downgrade the VO runtime."""

import hashlib, json, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1])
target = Path("/data/fyp_3d/envs/macvo_metrics_overlay")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
        "--target",
        str(target),
        "evo==1.29.0",
    ],
    check=True,
    timeout=600,
)
sys.path.insert(0, str(target))
import evo

assert evo.__version__ == "v1.29.0"
result = dict(
    status="historical_metrics_runtime_installed",
    version=evo.__version__,
    path=str(target),
    main_ape_sha256=hashlib.sha256(
        (target / "evo/main_ape.py").read_bytes()
    ).hexdigest(),
    reason="MAC-VO passes align=True and align_origin=True; evo1.29 uses Umeyama align precedence, evo1.38 rejects both. Preserve unmodified upstream MAC-VO metric calls.",
    localization_executed=False,
)
out.parent.mkdir(parents=True, exist_ok=True)
temp = out.with_suffix(".tmp")
temp.write_text(json.dumps(result, indent=2))
temp.replace(out)
print(json.dumps(result, indent=2))
