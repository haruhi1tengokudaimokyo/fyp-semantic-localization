from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from reporting import snapshot, html_page
import json


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ["/", "/index.html", "/api/status"]:
            self.send_error(404)
            return
        s = snapshot()
        body = (
            json.dumps(s, ensure_ascii=False)
            if self.path == "/api/status"
            else html_page(s)
        ).encode()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            (
                "application/json; charset=utf-8"
                if self.path == "/api/status"
                else "text/html; charset=utf-8"
            ),
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
