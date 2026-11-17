# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Footage that is still somewhere else: only the parts a step reads are
fetched, by byte range, into a local file of the segment's own size.

prepare reads every segment's sound (the sync) and single frames from a
few segments (the stitch, the pitch, the kit sample): about 5 GB of a
40-50 GB game. With OSB_FOOTAGE_REMOTE set, a segment file that isn't on
the disk is made as a placeholder: an empty (sparse) file of the
segment's size holding only the MP4's index (everything but `mdat`), with
`<file>.partial` beside it saying what else has been fetched. The two
places the engine reads media then ask for what they are about to read:

    remote.audio(path)           before phase0.load_audio: the sound track's chunks
    remote.video(path, t0, t1)   before phase0.frames_at: the video samples for those
                                 file times, from the keyframe before t0 - BACK_S

and ffmpeg reads the placeholder exactly as it reads the whole file (the
bytes it touches are the real ones, at their real offsets). With the
variable unset, or for a file that is whole, both do nothing.

OSB_FOOTAGE_REMOTE names a JSON file:

    {"recording": "/scratch/<job>/rec",          the recording folder: a segment's key is its path under it
     "s3": {"bucket": "...", "prefix": "..."},   where the segments are (boto3), or
     "urls": {"<objectKey>": "https://..."},     a link per segment that honours Range (plain HTTP)
     "stats": "/scratch/<job>/remote-stats.jsonl"}   optional: each process adds a line with what it fetched

A placeholder is never the whole file: `partial(path)` says so, engine.py
won't hash one or start `run` on one, and cloud.py's full download
replaces it. Only video read through phase0.frames_at / frame_at is
covered, which is all prepare does; follow.frame_stream (the game itself)
needs the whole files.
"""

from __future__ import annotations

import atexit
import bisect
import concurrent.futures
import functools
import json
import os
import struct
import threading
import time
import urllib.request
from pathlib import Path

ENV = "OSB_FOOTAGE_REMOTE"
SIDE = ".partial"
# Either side of the times asked for. Before: the files' timestamps start a
# little after 0 (a Pixel's video at 0.04 s) and frames are stored out of
# order, so the sample the decoder seeks to can be a few before the one
# found here. After: the decoder reads ahead of the last frame it gives
# (a frame per thread), and reading holes there fills the log with errors.
BACK_S, AHEAD_S = 1.0, 1.0
ATTEMPTS = 3
PIECE = 32 << 20          # a longer stretch of video is fetched in pieces this size, a few at once
AUDIO_STREAMS = 16        # the sound is 300-600 small chunks a file

_spec: dict | None | bool = False
_lock = threading.Lock()
_stats = {"gets": 0, "bytes": 0, "seconds": 0.0}
_client = None


class RemoteError(Exception):
    pass


def spec() -> dict | None:
    """Where the footage is, or None when it is all here."""
    global _spec
    if _spec is False:
        with _lock:  # threads asking at once: read once, and one stats line a process
            if _spec is False:
                p = os.environ.get(ENV)
                found = json.loads(Path(p).read_text()) if p else None
                if found and found.get("stats"):
                    atexit.register(_write_stats)
                _spec = found
    return _spec


def reset() -> None:
    """Forget the spec and the indexes read so far (tests)."""
    global _spec, _client
    _spec, _client = False, None
    _index.cache_clear()
    _stats.update(gets=0, bytes=0, seconds=0.0)


def _write_stats() -> None:
    s = spec()
    if s and s.get("stats") and _stats["gets"]:
        try:
            with open(s["stats"], "a") as f:
                f.write(json.dumps({**_stats, "seconds": round(_stats["seconds"], 2), "pid": os.getpid()}) + "\n")
        except OSError:
            pass


def _key(path: Path) -> str:
    return Path(path).resolve().relative_to(Path(spec()["recording"]).resolve()).as_posix()


def _s3():
    global _client
    with _lock:
        if _client is None:
            import boto3
            from botocore.config import Config

            _client = boto3.client("s3", config=Config(max_pool_connections=AUDIO_STREAMS * 2))
    return _client


def _get(key: str, a: int, n: int) -> bytes:
    """Bytes a..a+n of a segment, tried again when the stream drops."""
    s = spec()
    t0 = time.time()
    for attempt in range(ATTEMPTS):
        try:
            if "s3" in s:
                body = _s3().get_object(Bucket=s["s3"]["bucket"], Key=s["s3"]["prefix"] + key,
                                        Range=f"bytes={a}-{a + n - 1}")["Body"].read()
            else:
                req = urllib.request.Request(s["urls"][key], headers={"Range": f"bytes={a}-{a + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
            if len(body) != n:  # a server that ignores Range sends the whole file
                raise RemoteError(f"asked for {n} bytes of {key} at {a}, got {len(body)}")
            break
        except Exception:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(1 + attempt)
    with _lock:
        _stats["gets"] += 1
        _stats["bytes"] += n
        _stats["seconds"] += time.time() - t0
    return body


# ------------------------------------------------------------ what is here

def _side(path: Path) -> Path:
    return Path(str(path) + SIDE)


def partial(path: Path) -> bool:
    """Whether `path` is a placeholder: some of the segment, not all of it."""
    return _side(path).exists()


_file_locks: dict[str, threading.Lock] = {}


def _file_lock(path: Path) -> threading.Lock:
    """One at a time per file: callers grab frames from several threads
    (phase0.grabs), and two fetches into one file would each write its note
    of what is here over the other's."""
    with _lock:
        return _file_locks.setdefault(str(path), threading.Lock())


def _load(path: Path) -> dict:
    return json.loads(_side(path).read_text())


def _save(path: Path, st: dict) -> None:
    tmp = Path(str(path) + SIDE + ".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(_side(path))


def completed(path: Path) -> None:
    """The whole file has been put at `path`: it is no longer a placeholder."""
    _side(path).unlink(missing_ok=True)


def placeholder(path: Path, size: int) -> None:
    """Makes `path` as a placeholder when it isn't on the disk: the right
    size, holding the MP4's index and nothing else."""
    if spec() is None:
        return
    path = Path(path)
    if path.exists() and not partial(path):
        return  # the whole file
    if partial(path) and path.exists() and path.stat().st_size == size and _load(path).get("moov"):
        return
    key = _key(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = {"bytes": size, "moov": None, "audio": False, "have": []}
    _save(path, st)  # first: a file here without it would pass for the whole segment
    with path.open("wb") as f:
        f.truncate(size)
        pos = 0
        while pos < size:
            head = _get(key, pos, min(16, size - pos))
            if len(head) < 8:
                break
            n, kind = struct.unpack(">I4s", head[:8])
            if n == 1 and len(head) == 16:
                n = struct.unpack(">Q", head[8:16])[0]
            elif n == 0:
                n = size - pos
            if n < 8 or pos + n > size:
                raise RemoteError(f"{key}: not an MP4 this can read (box {kind!r} of {n} bytes at {pos})")
            f.seek(pos)
            if kind == b"mdat":
                f.write(head)
            else:  # ftyp, moov, free, ...: small, kept whole
                f.write(_get(key, pos, n))
                if kind == b"moov":
                    st["moov"] = [pos, n]
            pos += n
    if not st["moov"]:
        raise RemoteError(f"{key}: no index (moov) in the file")
    _save(path, st)


# ------------------------------------------------------------- the index

def _boxes(buf: bytes, a: int, b: int):
    while a + 8 <= b:
        n, kind = struct.unpack(">I4s", buf[a:a + 8])
        head = 8
        if n == 1:
            n, head = struct.unpack(">Q", buf[a + 8:a + 16])[0], 16
        elif n == 0:
            n = b - a
        if n < head:
            return
        yield kind, a + head, a + n
        a += n


def _find(buf: bytes, span: tuple[int, int], kind: bytes) -> tuple[int, int] | None:
    return next(((x, y) for k, x, y in _boxes(buf, *span) if k == kind), None)


def _track(moov: bytes, handler: bytes) -> dict | None:
    """The first track of that kind: each sample's offset and size, its
    decode time in seconds, and which samples the file marks as keyframes
    (None when it marks none or all of them: a Pixel marks every one)."""
    for k, a, b in _boxes(moov, 8, len(moov)):
        if k != b"trak":
            continue
        mdia = _find(moov, (a, b), b"mdia")
        hdlr = mdia and _find(moov, mdia, b"hdlr")
        if not hdlr or moov[hdlr[0] + 8:hdlr[0] + 12] != handler:
            continue
        x, _ = _find(moov, mdia, b"mdhd")
        scale = struct.unpack(">I", moov[x + 20:x + 24] if moov[x] == 1 else moov[x + 12:x + 16])[0]
        stbl = _find(moov, _find(moov, mdia, b"minf"), b"stbl")
        x, _ = _find(moov, stbl, b"stsz")
        fixed, n = struct.unpack(">II", moov[x + 4:x + 12])
        sizes = [fixed] * n if fixed else list(struct.unpack(f">{n}I", moov[x + 12:x + 12 + 4 * n]))
        co = _find(moov, stbl, b"stco")
        if co:
            n = struct.unpack(">I", moov[co[0] + 4:co[0] + 8])[0]
            chunk_at = struct.unpack(f">{n}I", moov[co[0] + 8:co[0] + 8 + 4 * n])
        else:
            x, _ = _find(moov, stbl, b"co64")
            n = struct.unpack(">I", moov[x + 4:x + 8])[0]
            chunk_at = struct.unpack(f">{n}Q", moov[x + 8:x + 8 + 8 * n])
        x, _ = _find(moov, stbl, b"stsc")
        n = struct.unpack(">I", moov[x + 4:x + 8])[0]
        runs = [struct.unpack(">III", moov[x + 8 + 12 * i:x + 20 + 12 * i]) for i in range(n)]
        offsets, chunks, s = [], [], 0
        for i, (first, per, _) in enumerate(runs):
            last = runs[i + 1][0] - 1 if i + 1 < len(runs) else len(chunk_at)
            for c in range(first, last + 1):
                pos = chunk_at[c - 1]
                for size in sizes[s:s + per]:
                    offsets.append(pos)
                    pos += size
                chunks.append((chunk_at[c - 1], pos - chunk_at[c - 1]))
                s += per
        if s != len(sizes):
            raise RemoteError(f"the index lists {len(sizes)} samples and places {s}")
        x, _ = _find(moov, stbl, b"stts")
        n = struct.unpack(">I", moov[x + 4:x + 8])[0]
        times, tick = [], 0
        for i in range(n):
            cnt, delta = struct.unpack(">II", moov[x + 8 + 8 * i:x + 16 + 8 * i])
            for _ in range(cnt):
                times.append(tick / scale)
                tick += delta
        times = times[:len(sizes)] + [tick / scale] * (len(sizes) - len(times))
        keys = None
        ss = _find(moov, stbl, b"stss")
        if ss:
            n = struct.unpack(">I", moov[ss[0] + 4:ss[0] + 8])[0]
            if 0 < n < len(sizes):
                keys = [k - 1 for k in struct.unpack(f">{n}I", moov[ss[0] + 8:ss[0] + 8 + 4 * n])]
        return {"offsets": offsets, "sizes": sizes, "times": times, "keys": keys, "chunks": chunks}
    return None


@functools.lru_cache(maxsize=None)
def _index(path: str, kind: bytes) -> dict:
    pos, n = _load(Path(path))["moov"]
    with open(path, "rb") as f:
        f.seek(pos)
        moov = f.read(n)
    t = _track(moov, kind)
    if t is None:
        raise RemoteError(f"{Path(path).name}: no {'sound' if kind == b'soun' else 'video'} track")
    return t


# ------------------------------------------------------ fetching what is read

def _merged(spans: list[list[int]]) -> list[list[int]]:
    """[a, b) spans, sorted, with the ones that touch or overlap joined."""
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _missing(have: list[list[int]], a: int, b: int) -> list[tuple[int, int]]:
    """The parts of [a, b) that `have` doesn't cover."""
    out = []
    for x, y in have:
        if y <= a:
            continue
        if x >= b:
            break
        if x > a:
            out.append((a, x))
        a = max(a, y)
    if a < b:
        out.append((a, b))
    return out


def _fetch_into(path: Path, key: str, spans: list[tuple[int, int]], streams: int) -> None:
    if not spans:
        return
    with path.open("r+b") as f, concurrent.futures.ThreadPoolExecutor(streams) as pool:
        for (a, b), data in zip(spans, pool.map(lambda s: _get(key, s[0], s[1] - s[0]), spans)):
            f.seek(a)
            f.write(data)


def audio(path: Path) -> None:
    """Before the file's sound is read: the sound track's chunks, once."""
    if spec() is None or not partial(Path(path)):
        return
    path = Path(path)
    with _file_lock(path):
        st = _load(path)
        if st["audio"]:
            return
        spans = _merged([[a, a + n] for a, n in _index(str(path), b"soun")["chunks"]])
        todo = [s for a, b in spans for s in _missing(st["have"], a, b)]
        _fetch_into(path, _key(path), todo, AUDIO_STREAMS)
        st["have"] = _merged(st["have"] + spans)
        st["audio"] = True
        _save(path, st)


def video(path: Path, t0: float, t1: float) -> None:
    """Before frames between file times t0 and t1 (s) are decoded: the
    video samples from the keyframe at or before t0 - BACK_S to t1 +
    AHEAD_S, less what is already here."""
    if spec() is None or not partial(Path(path)):
        return
    path = Path(path)
    v = _index(str(path), b"vide")
    if not v["sizes"]:
        return
    i0 = max(0, bisect.bisect_right(v["times"], t0 - BACK_S) - 1)
    if v["keys"]:
        i0 = v["keys"][max(0, bisect.bisect_right(v["keys"], i0) - 1)]
    i1 = min(len(v["sizes"]) - 1, bisect.bisect_right(v["times"], t1 + AHEAD_S))
    if i1 < i0:
        return
    # Samples are stored out of order by a frame or two, never by much: the
    # stretch between the lowest start and the highest end is what is read
    # (with the sound that lies between, a few kB).
    a = min(v["offsets"][i0:i1 + 1])
    b = max(o + n for o, n in zip(v["offsets"][i0:i1 + 1], v["sizes"][i0:i1 + 1]))
    with _file_lock(path):
        st = _load(path)
        todo = [(x, min(y, x + PIECE)) for p, q in _missing(st["have"], a, b)
                for x, y in ((s, q) for s in range(p, q, PIECE))]
        if not todo:
            return
        _fetch_into(path, _key(path), todo, 4)
        st["have"] = _merged(st["have"] + [[a, b]])
        _save(path, st)
