# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""What a cloud job used of its machine, for cloud.py: when each part of
the job began, how busy the GPU, its memory, the CPUs and the RAM were
while the engine worked, and how long each engine step and chunk took.

One profile per command, written to the job's output folder:

    profile/<command>.json           the summary below
    profile/<command>-samples.jsonl  every sample (one each SAMPLE_S), as taken

    {
      "command": "run", "engineVersion": "...", "exit": 0,
      "machine": {"gpu": "NVIDIA A10G", "gpuMemoryMiB": 23028, "driver": "580.x", "cpus": 4,
                  "memoryMiB": 15731, "instanceType": "g5.xlarge", "bootToStartS": 95.2},
      "timeline": [{"mark": "start", "t": 0.0}, {"mark": "footage_fetched", "t": 412.7}, ...],
      "transfers": {"footage": {"bytes": ..., "seconds": ..., "mbPerS": ...}, ...},
      "steps": [{"step": "sync", "seconds": 61.0, "gpuUtil": {...}, ...}],
      "chunks": [...],   # run: engine.py status's chunkDetails (each chunk's stages and times)
      "usage": {"gpuUtil": {"mean", "p50", "p95", "max"}, "gpuMemoryMiB": {...}, "gpuDecoder": {...},
                "gpuEncoder": {...}, "cpuUtil": {...}, "memoryUsedMiB": {...}, "containerMemoryMiB": {...}},
      "samples": 1234, "sampleEveryS": 5
    }

`t` is seconds since cloud.py started. `bootToStartS` is how long the
machine had been up when it did (/proc/uptime): instance boot, the ECS
agent and the image pull. Batch's own times (submitted, started) are in
`aws batch describe-jobs`.

Nothing here can fail a job: a machine without nvidia-smi or /proc gives
a profile without those numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

SAMPLE_S = 5.0
# Asked in this order; nvidia-smi answers [N/A] where a GPU has no such counter.
GPU_FIELDS = ["utilization.gpu", "memory.used", "memory.total", "utilization.decoder", "utilization.encoder"]
GPU_BASIC = GPU_FIELDS[:3]


def _smi() -> str:
    return os.environ.get("OSB_NVIDIA_SMI", "nvidia-smi")


def _num(s: str) -> float | None:
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _meminfo() -> dict[str, float]:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        n = _num(v)
        if n is not None:
            out[k] = n / 1024  # kB -> MiB
    return out


def _cpu_times() -> tuple[float, float] | None:
    line = (_read("/proc/stat") or "").split("\n", 1)[0].split()
    if len(line) < 5 or line[0] != "cpu":
        return None
    vals = [float(x) for x in line[1:]]
    return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)  # total, idle + iowait


def _container_memory_mib() -> float | None:
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        n = _num(_read(p) or "")
        if n is not None:
            return n / (1 << 20)
    return None


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    at = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return {"mean": round(sum(v) / len(v), 1), "p50": round(at(0.5), 1), "p95": round(at(0.95), 1), "max": round(v[-1], 1)}


METRICS = ["gpuUtil", "gpuMemoryMiB", "gpuDecoder", "gpuEncoder", "cpuUtil", "memoryUsedMiB", "containerMemoryMiB"]


def summarize(samples: list[dict]) -> dict:
    return {m: s for m in METRICS if (s := _stats([x[m] for x in samples if x.get(m) is not None]))}


class Profile:
    def __init__(self, command: str, engine_version: str):
        self.command, self.engine_version = command, engine_version
        self.t0 = time.time()
        self.timeline: list[dict] = []
        self.transfers: dict[str, dict] = {}
        self.steps: list[dict] = []
        self.chunks: list | None = None
        self.samples: list[dict] = []
        self.step: str | None = None
        self._fields = GPU_FIELDS
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu = _cpu_times()
        self.machine = self._machine()
        self.mark("start")

    def _machine(self) -> dict:
        m: dict = {"cpus": os.cpu_count()}
        up = _num(_read("/proc/uptime") or "")
        if up is not None:
            m["bootToStartS"] = round(up, 1)
        total = _meminfo().get("MemTotal")
        if total:
            m["memoryMiB"] = round(total)
        try:
            out = subprocess.run([_smi(), "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
            if out:
                name, mem, driver = [x.strip() for x in out[0].split(",")]
                m.update(gpu=name, gpuMemoryMiB=_num(mem), driver=driver, gpus=len(out))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        # ECS puts the instance's type in the container's metadata file, when it writes one.
        meta = _read(os.environ.get("ECS_CONTAINER_METADATA_FILE", "")) if os.environ.get("ECS_CONTAINER_METADATA_FILE") else None
        if meta:
            try:
                m["availabilityZone"] = json.loads(meta).get("AvailabilityZone")
            except json.JSONDecodeError:
                pass
        if os.environ.get("OSB_INSTANCE_TYPE"):
            m["instanceType"] = os.environ["OSB_INSTANCE_TYPE"]
        return m

    def t(self) -> float:
        return round(time.time() - self.t0, 1)

    def mark(self, name: str) -> None:
        self.timeline.append({"mark": name, "t": self.t()})

    def transfer(self, name: str, nbytes: int, seconds: float) -> None:
        self.transfers[name] = {"bytes": nbytes, "seconds": round(seconds, 1),
                                "mbPerS": round(nbytes / 1e6 / seconds, 1) if seconds > 0 else None}

    def event(self, e: dict) -> None:
        """An engine event: which step the samples belong to, and how long each took."""
        if e.get("event") != "step":
            return
        if e.get("status") == "started":
            self.step = e.get("step")
            self.mark(f"step:{self.step}")
        elif e.get("status") == "done":
            mine = [s for s in self.samples if s.get("step") == e.get("step")]
            self.steps.append({"step": e.get("step"), "seconds": e.get("seconds"), **summarize(mine)})
            self.step = None

    # -- sampling

    def _gpu(self) -> dict:
        try:
            out = subprocess.run([_smi(), f"--query-gpu={','.join(self._fields)}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return {}
        if out.returncode != 0 and self._fields is not GPU_BASIC:
            self._fields = GPU_BASIC  # an older driver without the decoder and encoder counters
            return self._gpu()
        line = out.stdout.strip().splitlines()[:1]
        if out.returncode != 0 or not line:
            return {}
        vals = [_num(x) for x in line[0].split(",")]
        names = ["gpuUtil", "gpuMemoryMiB", "gpuMemoryTotalMiB", "gpuDecoder", "gpuEncoder"]
        return {n: v for n, v in zip(names, vals) if v is not None and n != "gpuMemoryTotalMiB"}

    def sample(self) -> dict:
        s: dict = {"t": self.t(), "step": self.step, **self._gpu()}
        now = _cpu_times()
        if now and self._cpu and now[0] > self._cpu[0]:
            s["cpuUtil"] = round(100 * (1 - (now[1] - self._cpu[1]) / (now[0] - self._cpu[0])), 1)
        self._cpu = now
        mem = _meminfo()
        if "MemTotal" in mem and "MemAvailable" in mem:
            s["memoryUsedMiB"] = round(mem["MemTotal"] - mem["MemAvailable"])
        c = _container_memory_mib()
        if c is not None:
            s["containerMemoryMiB"] = round(c)
        self.samples.append(s)
        return s

    def start(self) -> None:
        def loop():
            while not self._stop.wait(float(os.environ.get("OSB_PROFILE_EVERY_S", SAMPLE_S))):
                try:
                    self.sample()
                except Exception:  # a profile must never stop the job
                    pass

        self._thread = threading.Thread(target=loop, name="profile", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)

    # -- output

    def summary(self, exit_code: int | None) -> dict:
        return {"command": self.command, "engineVersion": self.engine_version, "exit": exit_code,
                "machine": self.machine, "seconds": self.t(), "timeline": self.timeline, "transfers": self.transfers,
                "steps": self.steps, **({"chunks": self.chunks} if self.chunks is not None else {}),
                "usage": summarize(self.samples), "samples": len(self.samples),
                "sampleEveryS": float(os.environ.get("OSB_PROFILE_EVERY_S", SAMPLE_S))}

    def samples_jsonl(self) -> str:
        return "".join(json.dumps(s) + "\n" for s in self.samples)
