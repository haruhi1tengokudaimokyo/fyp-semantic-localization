"""Download one exact tar member from official HTTP source, not the whole archive."""

import argparse, concurrent.futures, hashlib, json, tarfile, time
from pathlib import Path
from inspect_remote_rosbag import fetch


def main(a):
    inv = json.loads(Path(a.inventory).read_text())
    m = next(x for x in inv["members"] if x["name"] == a.member)
    url = inv["url"]
    base = m["offset_data"]
    size = m["bytes"]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = out.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    header, _ = fetch(url, base - 512, base - 1)
    actual = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
    assert actual.name == a.member and actual.size == size
    chunk = 8 * 1024**2
    start = time.perf_counter()

    def get(i):
        lo = i * chunk
        hi = min(size, lo + chunk) - 1
        p = parts / str(i)
        if p.exists() and p.stat().st_size == hi - lo + 1:
            return p
        for attempt in range(5):
            try:
                data, _ = fetch(url, base + lo, base + hi)
                p.write_bytes(data)
                print("part", i, "complete", flush=True)
                return p
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        paths = list(ex.map(get, range((size + chunk - 1) // chunk)))
    h = hashlib.sha256()
    with out.open("wb") as f:
        for p in paths:
            with p.open("rb") as g:
                while data := g.read(1024**2):
                    f.write(data)
                    h.update(data)
    assert out.stat().st_size == size
    report = dict(
        url=url,
        member=a.member,
        offset=base,
        bytes=size,
        sha256=h.hexdigest(),
        seconds=time.perf_counter() - start,
        validation="tar header name/size/checksum and all exact HTTP ranges; local member SHA256 recorded, full parent tar SHA not claimed",
    )
    out.with_suffix(".acquisition.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inventory")
    p.add_argument("member")
    p.add_argument("output")
    main(p.parse_args())
