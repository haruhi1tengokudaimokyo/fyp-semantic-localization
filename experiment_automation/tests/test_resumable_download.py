"""Real local HTTP Range/ETag tests using infrastructure-only bytes, no benchmark data."""

import importlib.util, json, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from collections import namedtuple

p = Path(__file__).resolve().parents[1] / "reproduction/acquire_public.py"
s = importlib.util.spec_from_file_location("download", p)
d = importlib.util.module_from_spec(s)
s.loader.exec_module(d)


@pytest.fixture
def server(monkeypatch):
    body = b"original source data" * 100

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(
                self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0]
            )
            changed = self.headers.get("If-Range") not in (None, '"version1"')
            if changed:
                start = 0
            self.send_response(200 if changed else 206)
            self.send_header("Content-Length", str(len(body) - start))
            self.send_header("ETag", '"version1"')
            if not changed:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}"
                )
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        d.shutil, "disk_usage", lambda _: Usage(100 * 1024**3, 0, 100 * 1024**3)
    )
    yield f"http://127.0.0.1:{http.server_port}/source.bin", body
    http.shutdown()
    thread.join()
    http.server_close()


def test_resume_preserves_exact_bytes_and_cached_receipt_is_revalidated(
    server, tmp_path
):
    url, body = server
    out = tmp_path / "source.bin"
    part = tmp_path / "source.bin.part"
    part.write_bytes(body[:77])
    part.with_suffix(".part.source.json").write_text(
        json.dumps(dict(url=url, total=len(body), etag='"version1"'))
    )
    result = d.fetch(url, out)
    assert out.read_bytes() == body and result["resumed_bytes"] == 77
    assert d.fetch(url, out)["revalidated_existing"] is True


def test_changed_source_never_appends_to_partial(server, tmp_path):
    url, body = server
    out = tmp_path / "source.bin"
    part = tmp_path / "source.bin.part"
    part.write_bytes(body[:77])
    part.with_suffix(".part.source.json").write_text(
        json.dumps(dict(url=url, total=len(body), etag='"old-version"'))
    )
    with pytest.raises(AssertionError):
        d.fetch(url, out)
    assert part.read_bytes() == body[:77] and not out.exists()


def test_partial_without_origin_identity_is_quarantined(server, tmp_path):
    url, body = server
    out = tmp_path / "source.bin"
    part = tmp_path / "source.bin.part"
    part.write_bytes(body[:77])
    with pytest.raises(AssertionError, match="source identity"):
        d.fetch(url, out)
    assert part.read_bytes() == body[:77] and not out.exists()
