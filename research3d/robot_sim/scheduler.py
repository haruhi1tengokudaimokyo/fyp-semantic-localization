"""Single persistent ACE process with one latest pending image, never a GPU queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np


class ExecutionAlternator:
    """Choose from actual execution history, never from overwritten submissions."""
    def __init__(self, ace_available, spvloc_available):
        self.ace_available, self.spvloc_available = ace_available, spvloc_available
        self.last_started = None
        self.started = {"ace": 0, "spvloc": 0}

    def choose(self, recovering):
        if self.ace_available:
            branch = "spvloc" if recovering and self.spvloc_available and self.last_started == "ace" else "ace"
        elif self.spvloc_available:
            branch = "spvloc"
        else:
            raise ValueError("no executable absolute localizer")
        self.last_started = branch
        self.started[branch] += 1
        return branch


class LatestOnlyWorker:
    def __init__(self, work, result):
        self.work, self.result = work, result
        self.condition = threading.Condition()
        self.pending = None
        self.active = False
        self.closed = False
        self.overwritten = 0
        self.completed = 0
        self.thread = threading.Thread(target=self._loop, name="ace-latest-query", daemon=True)
        self.thread.start()

    def submit(self, item):
        with self.condition:
            if self.closed:
                raise RuntimeError("worker closed")
            if self.pending is not None:
                self.overwritten += 1
            self.pending = item
            self.condition.notify()

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                item, self.pending = self.pending, None
                self.active = True
            try:
                value = self.work(item)
            except Exception as exc:
                value = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                         "request_id": item.get("request_id"), "query_id": item.get("query_id"),
                         "map_id": item.get("map_id"), "stamp_sec": item.get("timestamp_ns",0)/1e9,
                         "map_sha256":item.get("map_sha256"),
                         "source": "official_spvloc" if item.get("_localizer")=="spvloc" else "official_ace_dsacstar"}
            self.result(value)
            with self.condition:
                self.active = False
                self.completed += 1

    def close(self, timeout=3.):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout)


class AceRpcProcess:
    """JSON-lines subprocess. No shell, GT arguments, network or model substitutes."""
    def __init__(self, command, log_path, timeout_s=90.):
        if not isinstance(command, list) or not command or not all(isinstance(x,str) for x in command):
            raise ValueError("ACE command must be an argv JSON list")
        self.log = Path(log_path).open("a")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        self.timeout_s = timeout_s
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, name="ace-rpc-reader", daemon=True)
        self.reader.start()
        self.ready = None

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def wait_ready(self):
        """Bounded real initialization receipt, before route/capture starts."""
        if self.ready is not None:
            return self.ready
        deadline = time.monotonic()+self.timeout_s
        while True:
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("localizer initialization timed out")
            try:
                line = self.lines.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty as exc:
                self.close()
                raise TimeoutError("localizer initialization timed out") from exc
            if line is None:
                self.close()
                raise RuntimeError("localizer stdout closed before ready")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line); self.log.flush()
                continue
            if reply.get("event") == "fatal":
                self.close()
                raise RuntimeError(f"localizer initialization failed: {reply.get('error_type','')}: {reply.get('error','unknown failure')}")
            if reply.get("event") == "ready" or reply.get("status") == "ready":
                self.ready = reply
                return reply
            self.close()
            raise RuntimeError("localizer emitted a response before ready")

    def __call__(self, request):
        if self.process.poll() is not None:
            raise RuntimeError(f"ACE worker exited {self.process.returncode}")
        self.process.stdin.write(json.dumps(request, allow_nan=False)+"\n")
        self.process.stdin.flush()
        deadline = time.monotonic()+self.timeout_s
        while True:
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("ACE RPC timeout; worker terminated")
            try:
                line = self.lines.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty as exc:
                # A timed-out request must not leak a later reply into another image.
                self.close()
                raise TimeoutError("ACE RPC timeout; worker terminated") from exc
            if line is None:
                raise RuntimeError("ACE stdout closed")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line)
                self.log.flush()
                continue
            if reply.get("event") == "ready" or reply.get("status") == "ready":
                self.ready = reply
                continue
            if reply.get("event") == "fatal":
                self.close()
                raise RuntimeError(f"ACE initialization failed: {reply.get('error_type','')}: {reply.get('error','unknown failure')}")
            if reply.get("request_id") != request["request_id"]:
                self.close()
                raise RuntimeError("ACE request identity mismatch")
            return reply

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.)
        if not self.log.closed:
            self.log.close()


class QueryTrigger:
    """At most 1 Hz; query for unlocalized/degraded/aged/moved states.

    Image quality rejects all-black/flat occlusion before expensive inference.
    Thresholds are deployment engineering settings, not tuned on test GT.
    """
    def __init__(self, max_hz=1., refresh_s=3., min_std=4., min_mean=2., movement_m=.25):
        if max_hz <= 0 or max_hz > 1.:
            raise ValueError("ACE rate must be in (0,1] Hz")
        self.interval = 1./max_hz
        self.refresh_s, self.min_std, self.min_mean, self.movement_m = refresh_s,min_std,min_mean,movement_m
        self.last_query_s = -float("inf")
        self.last_position = None
        self.counts = {}

    def decide(self, image, now, snapshot):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("query must be uint8 RGB")
        reason = "scheduled"
        if now-self.last_query_s < self.interval:
            reason = "rate_limited"
        elif float(image.mean()) < self.min_mean or float(image.std()) < self.min_std:
            reason = "occluded_or_low_texture"
        elif not snapshot["local_valid"]:
            reason = "local_odometry_unavailable"
        elif snapshot.get("local_state") == "DEGRADED":
            reason = "scheduled"
        elif snapshot["global_valid"]:
            p = np.array(snapshot["T_odom_base"])[:3,3]
            moved = self.last_position is None or np.linalg.norm(p-self.last_position) >= self.movement_m
            age = now-snapshot["last_absolute_capture_s"]
            if not moved and age < self.refresh_s:
                reason = "healthy_recent_global_pose"
        self.counts[reason] = self.counts.get(reason,0)+1
        if reason != "scheduled":
            return False,reason
        self.last_query_s = now
        self.last_position = np.array(snapshot["T_odom_base"])[:3,3].copy()
        return True,reason
