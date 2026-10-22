#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Local page for tapping pitch keypoints on a recording's background panorama.

    uv run processing/spikes/pitchtap/server.py footage/rec4 [port]

Serves <recording>/pitch/background.png and saves every tap to
<recording>/pitch/taps.json: {point name: {"yaw", "pitch" (panorama degrees,
yaw right +, pitch down +), "at"}} or {"skipped": true}. The points are the
ones pitch.py's taps mode refits with (pitch.TAP_POINTS). Listens on
localhost only: the footage shows people and stays on this machine.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
LOCK = threading.Lock()

# Kept in step with pitch.py (importing it would pull in torch for a page server).
YAW_RANGE, PITCH_RANGE, DEG_PER_PX = (-100.0, 100.0), (-2.0, 38.0), 0.03
POINTS = [
    ("{e} corner flag", "where the near touchline meets the goal line"),
    ("{e} 18-yard goal-line corner", "where the 18-yard box's near side meets the goal line"),
    ("{e} 18-yard front corner", "the 18-yard box's near front corner"),
    ("{e} 6-yard goal-line corner", "where the 6-yard box's near side meets the goal line"),
    ("{e} 6-yard front corner", "the 6-yard box's near front corner"),
    ("{e} near post base", "the foot of the goal post nearer the near touchline"),
]


CENTRE_POINTS = [
    ("halfway far end", "where the halfway line meets the far touchline"),
    ("circle near crossing", "where the centre circle crosses the halfway line, on the near side"),
    ("circle far crossing", "where the centre circle crosses the halfway line, on the far side"),
]


def make_handler(rec: Path):
    taps_path = rec / "pitch" / "taps.json"
    points = [{"name": n.format(e=e), "hint": h, "end": e} for e in ("left", "right") for n, h in POINTS]
    points += [{"name": n, "hint": h, "end": "centre"} for n, h in CENTRE_POINTS]

    def load() -> dict:
        return json.loads(taps_path.read_text()) if taps_path.exists() else {}

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
            if path == "/meta":
                body = {"points": points, "yawDeg": YAW_RANGE, "pitchDeg": PITCH_RANGE, "panoDeg": DEG_PER_PX,
                        "recording": rec.name}
                return self.send(200, json.dumps(body).encode(), "application/json")
            if path == "/taps":
                with LOCK:
                    return self.send(200, json.dumps(load()).encode(), "application/json")
            if path == "/background.png":
                return self.send(200, (rec / "pitch" / "background.png").read_bytes(), "image/png")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/taps/"):
                return self.send(404, b"not found", "text/plain")
            name = self.path[len("/taps/"):].replace("%20", " ")
            if name not in {p["name"] for p in points}:
                return self.send(400, b"unknown point", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                taps = load()
                if body.get("delete"):
                    taps.pop(name, None)
                elif body.get("skipped"):
                    taps[name] = {"skipped": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                else:
                    taps[name] = {"yaw": round(float(body["yaw"]), 3), "pitch": round(float(body["pitch"]), 3),
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                tmp = taps_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(taps, indent=1))
                os.replace(tmp, taps_path)
            self.send(200, b"{}", "application/json")

    return Handler


def main(rec: Path, port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(rec))
    print(f"pitch keypoints for {rec} at http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8774)
