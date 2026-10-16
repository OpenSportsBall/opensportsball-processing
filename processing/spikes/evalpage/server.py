#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Local page for drawing the must-see box of each moment in a playeval.py set.

    uv run processing/spikes/evalpage/server.py footage/rec3/playeval/<set> [port]

Serves the page, the panoramas and the boxes, and saves every change to
<set>/boxes.json: {moment id: {"yaw0", "yaw1", "pitch0", "pitch1" (panorama
degrees, yaw right +, pitch down +), "note", "draftBy", "edited",
"reviewed", "at"}}. Listens on localhost only: the footage shows people
and stays on this machine.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
LOCK = threading.Lock()


def make_handler(root: Path):
    boxes_path = root / "boxes.json"

    def load_boxes() -> dict:
        return json.loads(boxes_path.read_text()) if boxes_path.exists() else {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/items":
                return self.send(200, (root / "items.json").read_bytes(), "application/json")
            if path == "/pano.json":
                return self.send(200, (root / "pano.json").read_bytes(), "application/json")
            if path == "/boxes":
                with LOCK:
                    return self.send(200, json.dumps(load_boxes()).encode(), "application/json")
            if path.startswith("/pano/"):
                f = (root / path.lstrip("/")).resolve()
                if f.is_file() and f.is_relative_to(root.resolve()):
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    return self.send(200, f.read_bytes(), ctype)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/boxes/"):
                return self.send(404, b"not found", "text/plain")
            iid = self.path[len("/boxes/"):]
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                boxes = load_boxes()
                if body.get("delete"):
                    boxes.pop(iid, None)
                else:
                    old = boxes.get(iid, {})
                    y0, y1 = sorted(float(body[k]) for k in ("yaw0", "yaw1"))
                    p0, p1 = sorted(float(body[k]) for k in ("pitch0", "pitch1"))
                    boxes[iid] = {"yaw0": round(y0, 2), "yaw1": round(y1, 2),
                                  "pitch0": round(p0, 2), "pitch1": round(p1, 2),
                                  "note": str(body.get("note", old.get("note", ""))),
                                  "draftBy": old.get("draftBy", "reviewer"),
                                  "edited": bool(body.get("edited", old.get("edited", False))),
                                  "reviewed": bool(body.get("reviewed", False)),
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                tmp = boxes_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(boxes, indent=1))
                os.replace(tmp, boxes_path)
            self.send(200, b"{}", "application/json")

    return Handler


def main(root: Path, port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    print(f"must-see boxes for {root} at http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8766)
