"""List remote tar members by reading only verified 512-byte header ranges."""

import argparse, json, tarfile
from pathlib import Path
from inspect_remote_rosbag import fetch


def main(a):
    offset = 0
    rows = []
    for _ in range(10000):
        header, total = fetch(a.url, offset, offset + 511)
        if not header.strip(b"\0"):
            break
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        if info.type not in [tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE]:
            raise ValueError(
                "Extended headers not supported; refuse to invent member offsets"
            )
        rows.append(dict(name=info.name, bytes=info.size, offset_data=offset + 512))
        offset += 512 + ((info.size + 511) // 512) * 512
    else:
        raise RuntimeError("Too many tar members")
    Path(a.output).write_text(json.dumps(dict(url=a.url, members=rows), indent=2))
    print(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("output")
    main(p.parse_args())
