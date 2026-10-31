"""Inspect ROS bag v2 connection topics using two verified HTTP ranges.
Does not download sensor streams or feed LiDAR/GT to any estimator.
"""

import argparse, json, struct
from pathlib import Path
import requests


def fields(data):
    out = {}
    i = 0
    while i < len(data):
        n = struct.unpack_from("<I", data, i)[0]
        i += 4
        k, v = data[i : i + n].split(b"=", 1)
        out[k.decode()] = v
        i += n
    return out


def record(data, i):
    h = struct.unpack_from("<I", data, i)[0]
    i += 4
    header = fields(data[i : i + h])
    i += h
    n = struct.unpack_from("<I", data, i)[0]
    i += 4
    if i + n > len(data):
        raise ValueError("Incomplete record range")
    return header, data[i : i + n], i + n


def fetch(url, lo, hi):
    with requests.get(
        url, headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=90
    ) as r:
        r.raise_for_status()
        if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(
            f"bytes {lo}-{hi}/"
        ):
            raise RuntimeError("Server did not honor requested exact byte range")
        body = r.content
        assert len(body) == hi - lo + 1
        return body, int(r.headers["Content-Range"].split("/")[-1])


def main(a):
    first, size = fetch(a.url, 0, 4095)
    assert first.startswith(b"#ROSBAG V2.0\n")
    h, _, _ = record(first + b"\0" * 64, 13)
    index = struct.unpack("<Q", h["index_pos"])[0]
    n = struct.unpack("<I", h["conn_count"])[0]
    hi = min(size - 1, index + 1024 * 1024 - 1)
    tail, _ = fetch(a.url, index, hi)
    connections = []
    i = 0
    while len(connections) < n:
        header, data, i = record(tail, i)
        if header["op"] == b"\x07":
            d = fields(data)
            connections.append(
                dict(
                    topic=header["topic"].decode(),
                    type=d.get("type", b"").decode(),
                    connection_id=struct.unpack("<I", header["conn"])[0],
                )
            )
    assert len(connections) == n, (len(connections), n)
    result = dict(
        url=a.url,
        total_remote_bytes=size,
        index_pos=index,
        declared_connection_count=n,
        connections=connections,
        bytes_fetched=len(first) + len(tail),
        scope="Header/index topic inventory only; no full-file CRC/SHA validation and no sensor-stream decoding",
        online_policy="No topics are enabled by this inspector. Future estimator must explicitly allowlist camera/IMU/wheel; all LiDAR and ground-truth topics excluded.",
    )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("output")
    main(p.parse_args())
