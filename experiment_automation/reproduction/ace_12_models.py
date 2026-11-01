"""Original D-SLAM 12Scenes weights; individually CRC/structure verified.

Cached-prefix inventory does not claim a complete gzip. The acquisition task
streams the official HTTPS archive once and stops after all12 requested heads.
"""

import argparse, hashlib, io, json, shutil, tarfile, zipfile
from pathlib import Path
import requests, torch

URL = "https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
SCENES = {
    "apt1": ["kitchen", "living"],
    "apt2": ["bed", "kitchen", "living", "luke"],
    "office1": ["gates362", "gates381", "lounge", "manolis"],
    "office2": ["5a", "5b"],
}
EXPECTED = {
    f"ace_models/12Scenes/12scenes_{g}_{s}.pt" for g, ss in SCENES.items() for s in ss
}


def verify(data, size):
    assert len(data) == size
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    assert (
        isinstance(state, dict)
        and state
        and all(
            isinstance(v, torch.Tensor) and torch.isfinite(v).all()
            for v in state.values()
        )
    )
    return dict(
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        member_zip_crc_verified=True,
        state_tensors=len(state),
    )


def inventory(path, output):
    records = {}
    truncated = False
    try:
        with tarfile.open(path, mode="r|gz") as tar:
            for m in tar:
                if m.name not in EXPECTED:
                    continue
                assert m.isfile()
                data = tar.extractfile(m).read()
                records[m.name] = verify(data, m.size)
    except (EOFError, tarfile.ReadError):
        truncated = True
    result = dict(
        status="inventoried",
        original_DSLAM_12_heads_complete=set(records) == EXPECTED,
        models=records,
        local_archive=str(path),
        full_gzip_integrity_verified=not truncated,
        local_archive_truncated=truncated,
        source_claim=URL,
        source_origin_independently_verified=False,
        localization_executed=False,
        scope="Local member structure/CRC audit only; fresh official stream required for later published-weight acquisition. Existing valid7Scenes results unchanged.",
    )
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(value, indent=2))
    t.replace(p)


def acquire(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    assert (
        shutil.disk_usage(out).free > 30 * 1024**3 + 64 * 1024**2 + 12 * 5 * 1024**2
    ), "Preserve data reserve and remaining experiment outputs"
    with requests.get(URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for m in tar:
                if m.name not in EXPECTED:
                    continue
                assert m.isfile()
                data = tar.extractfile(m).read()
                record = verify(data, m.size)
                p = out / Path(m.name).name
                if p.exists():
                    assert (
                        hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"]
                    )
                else:
                    t = p.with_suffix(".tmp")
                    t.write_bytes(data)
                    t.replace(p)
                records[m.name] = dict(**record, path=str(p))
                print("ORIGINAL_12SCENES_MODEL", m.name, flush=True)
                if set(records) == EXPECTED:
                    break
    assert set(records) == EXPECTED
    write(
        out / "acceptance.json",
        dict(
            status="acquired",
            source=URL,
            models=records,
            complete_original_12scenes_heads=True,
            pose_version="original D-SLAM, not Pseudo-GT",
            member_zip_crc_verified=True,
            archive_scope="HTTPS stream stopped after required12 models; complete gzip trailer not claimed",
            localization_executed=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["inventory", "acquire"])
    p.add_argument("output")
    p.add_argument("--archive", default="/data/fyp_3d/downloads/ace_models.tar.gz")
    a = p.parse_args()
    if a.action == "inventory":
        inventory(a.archive, a.output)
    else:
        acquire(a.output)
