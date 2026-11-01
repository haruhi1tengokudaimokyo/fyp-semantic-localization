"""Resumable publisher download with strict Range, disk reserve and ZIP CRC audit."""

import argparse, hashlib, json, shutil, time, zipfile
from pathlib import Path
import requests


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, output):
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    receipt = p.with_suffix(p.suffix + ".json")
    if p.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        assert old["url"] == url and old["sha256"] == sha(p)
        assert old["bytes"] == p.stat().st_size
        if url.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None, "Cached archive CRC failed"
        old.update(revalidated_existing=True, revalidated_epoch=time.time())
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(old, indent=2))
        temp.replace(receipt)
        return old
    part = p.with_suffix(p.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    start = time.monotonic()
    source_path = part.with_suffix(part.suffix + ".source.json")
    headers = {"Range": f"bytes={offset}-"}
    previous = None
    if offset:
        assert (
            source_path.exists()
        ), "Partial lacks source identity; preserve and quarantine, do not append blindly"
        previous = json.loads(source_path.read_text())
        assert previous["url"] == url
        if previous.get("etag"):
            headers["If-Range"] = previous["etag"]
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset:
            assert response.status_code == 206 and response.headers[
                "Content-Range"
            ].startswith(f"bytes {offset}-")
        total = (
            int(response.headers["Content-Range"].split("/")[-1])
            if response.status_code == 206
            else int(response.headers["Content-Length"])
        )
        if previous:
            assert previous["total"] == total
            if previous.get("etag"):
                assert previous["etag"] == response.headers.get(
                    "ETag"
                ), "Source identity changed; partial retained"
        identity = dict(url=url, total=total, etag=response.headers.get("ETag"))
        temporary = source_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, indent=2))
        temporary.replace(source_path)
        assert (
            shutil.disk_usage(p.parent).free - (total - offset) > 30 * 1024**3
        ), "Insufficient disk: preserve 30GiB reserve"
        with part.open("ab" if offset else "wb") as f:
            for block in response.iter_content(1024**2):
                assert (
                    shutil.disk_usage(p.parent).free > 30 * 1024**3
                ), "Disk reserve reached; resumable partial retained"
                f.write(block)
    assert part.stat().st_size == total
    if url.endswith(".zip"):
        with zipfile.ZipFile(part) as z:
            assert z.testzip() is None, "Archive CRC failed"
    record = dict(
        url=url,
        bytes=total,
        sha256=sha(part),
        downloaded_seconds=time.monotonic() - start,
        resumed_bytes=offset,
        zip_crc_verified=url.endswith(".zip"),
        source_identity=identity,
    )
    part.replace(p)
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(receipt)
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("output")
    a = p.parse_args()
    fetch(a.url, a.output)
