"""Probe all publisher split-archive sizes before committing a full-data download."""

import argparse, json, shutil, string, time
from pathlib import Path
import requests


def main(output):
    assets = []
    for letter in string.ascii_lowercase[:17]:
        name = "replica_v1_0.tar.gz.parta" + letter
        url = (
            "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/"
            + name
        )
        with requests.get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 90)
        ) as r:
            r.raise_for_status()
            assert r.status_code == 206 and r.headers["Content-Range"].startswith(
                "bytes 0-0/"
            )
            n = int(r.headers["Content-Range"].split("/")[-1])
        assets.append(dict(name=name, url=url, bytes=n))
        print(name, n, flush=True)
    total = sum(a["bytes"] for a in assets)
    free = shutil.disk_usage("/data/fyp_3d").free
    result = dict(
        status="inventoried",
        dataset="Replica original v1.0",
        archives=assets,
        compressed_bytes=total,
        disk_free_bytes=free,
        disk_reserve_bytes=30 * 1024**3,
        compressed_archive_fits_reserve=total <= free - 30 * 1024**3,
        uncompressed_bytes_measured=False,
        full_dataset_acquired=False,
        authorization="Public original Replica release; license in pinned Replica-Dataset checkout",
        distinction="ReplicaCAD is not Replica; partial probes are not a downloaded dataset",
        storage_note="Compressed bytes alone are a lower bound; extraction requires additional space. Never delete existing datasets or silently lower reserve.",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(p)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    main(p.parse_args().output)
