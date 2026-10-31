"""Stream official ACE archive only until Heads member, not all 700MB models."""

import argparse, hashlib, json, tarfile, zipfile
from pathlib import Path
import requests

URL = "https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
MEMBER = "ace_models/7Scenes/7scenes_heads.pt"
SHA = "eddd08cd9678709dc55c158a2b51e36735586b64dc2ebeefa9a31639851265f2"


def main(output):
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as archive:
            for member in archive:
                if member.name == MEMBER:
                    data = archive.extractfile(member).read()
                    assert len(data) == member.size
                    assert hashlib.sha256(data).hexdigest() == SHA
                    p.write_bytes(data)
                    break
            else:
                raise RuntimeError("Missing official model member")
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
    p.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(
                url=URL,
                member=MEMBER,
                sha256=SHA,
                bytes=p.stat().st_size,
                member_zip_crc_verified=True,
                archive_scope="stream stopped after required member; full gzip CRC not claimed",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    a = p.parse_args()
    main(a.output)
