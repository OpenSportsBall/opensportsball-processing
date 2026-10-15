#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Local ball-labelling page for a set made by labelset.py.

    uv run processing/spikes/label/server.py footage/rec3/labels [port]

Serves the page, the frames and the suggestions, and saves every label as
it is made to <set>/labels.json: {item id: {"status", "balls", "at"}}.
"status" is about the match ball: "ball" (visible), "none" (not visible) or
"unsure". "balls" lists every ball marked in the frame, in full-resolution
pixels: [{"x", "y", "inPlay"}]; at most one is in play, spare balls are not.
Older labels have "x", "y" for the match ball instead of "balls". Listens on
localhost only: the footage shows people and stays on this machine.
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
    labels_path = root / "labels.json"

    def load_labels() -> dict:
        return json.loads(labels_path.read_text()) if labels_path.exists() else {}

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
            if path == "/labels":
                with LOCK:
                    return self.send(200, json.dumps(load_labels()).encode(), "application/json")
            if path == "/pano_maps.json" and (root / "pano_maps.json").exists():
                return self.send(200, (root / "pano_maps.json").read_bytes(), "application/json")
            if path.startswith("/images/") or path.startswith("/pano/"):
                f = (root / path.lstrip("/")).resolve()
                if f.is_file() and f.is_relative_to(root.resolve()):
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    return self.send(200, f.read_bytes(), ctype)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/labels/"):
                return self.send(404, b"not found", "text/plain")
            iid = self.path[len("/labels/"):]
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                labels = load_labels()
                if body.get("status") in ("ball", "none", "unsure"):
                    balls = [{"x": round(float(b["x"]), 1), "y": round(float(b["y"]), 1),
                              "inPlay": bool(b.get("inPlay"))} for b in body.get("balls", [])]
                    labels[iid] = {"status": body["status"], "balls": balls,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                else:
                    labels.pop(iid, None)
                tmp = labels_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(labels, indent=1))
                os.replace(tmp, labels_path)
            self.send(200, b"{}", "application/json")

    return Handler


def main(root: Path, port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(root))
    print(f"labelling {root} at http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8765)
