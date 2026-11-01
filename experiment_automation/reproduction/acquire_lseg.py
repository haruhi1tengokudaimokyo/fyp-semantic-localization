"""Acquire the exact public LSeg checkpoint referenced by OMCL, never a proxy encoder."""

import argparse, hashlib, json, shutil, time
from pathlib import Path
import gdown
import torch


def main(output):
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    if not p.exists():
        assert shutil.disk_usage(p.parent).free > 32 * 1024**3
        temp = p.with_suffix(".part")
        value = gdown.download(
            id="1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb",
            output=str(temp),
            resume=True,
            quiet=False,
        )
        assert value and temp.is_file() and temp.stat().st_size > 100 * 1024**2
        checkpoint = torch.load(temp, map_location="cpu", weights_only=False)
        assert isinstance(checkpoint, dict) and isinstance(
            checkpoint["state_dict"], dict
        )
        assert len(checkpoint["state_dict"]) > 100
        del checkpoint
        temp.replace(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    record = dict(
        status="acquired",
        source="https://github.com/AIS-Bonn/omcl/blob/main/src/omcl/models/image_encoders/lseg/lseg.py",
        checkpoint="official LSeg demo_e200.ckpt",
        bytes=p.stat().st_size,
        sha256=h.hexdigest(),
        publisher_checksum_available=False,
        elapsed_seconds=time.monotonic() - tick,
        localization_executed=False,
        query_GT_used=False,
    )
    target = p.with_suffix(".json")
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(target)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    main(p.parse_args().output)
