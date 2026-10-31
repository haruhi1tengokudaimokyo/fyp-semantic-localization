"""Official TUM Freiburg3 full-sequence download, safe extraction and PNG audit."""

import argparse, hashlib, json, tarfile
from pathlib import Path
import requests
from PIL import Image


def main(a):
    archive = Path(a.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_"
        + a.sequence
        + ".tgz"
    )
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as f:
                for chunk in response.iter_content(4 * 1024**2):
                    f.write(chunk)
        # Entire gzip stream is consumed by getmembers, checking decompression/CRC before acceptance.
        with tarfile.open(temporary, "r:gz") as t:
            assert len(t.getmembers()) > 100
        temporary.replace(archive)
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    files = 0
    with tarfile.open(archive, "r:gz") as t:
        for member in t:
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not member.issym() and not member.islnk()
            if not member.isfile():
                continue
            target = dest / Path(*path.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.size:
                files += 1
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            source = t.extractfile(member)
            with temporary.open("wb") as f:
                while True:
                    block = source.read(4 * 1024**2)
                    if not block:
                        break
                    f.write(block)
            temporary.replace(target)
            files += 1
    counts = {}
    for kind in ["rgb", "depth"]:
        paths = list((dest / kind).glob("*.png"))
        assert len(paths) > 100
        counts[kind] = len(paths)
        for p in paths:
            with Image.open(p) as image:
                image.verify()
    for name in ["rgb.txt", "depth.txt", "groundtruth.txt"]:
        assert (dest / name).is_file()
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    result = dict(
        sequence=a.sequence,
        url=url,
        archive_sha256=h.hexdigest(),
        files=files,
        images=counts,
        all_png_verified=True,
        full_sequence=True,
        real_people=True,
        dynamics=(
            "low: sitting gesturing people"
            if a.sequence.startswith("sitting")
            else "high: walking people"
        ),
        no_prior_3D_map_claim=True,
        official_formats="https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
    )
    (dest / "acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sequence")
    p.add_argument("archive")
    p.add_argument("output")
    main(p.parse_args())
