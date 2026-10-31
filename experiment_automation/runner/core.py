"""Durable experiment state and strict output acceptance. Standard library + numpy for NPZ QA."""

import contextlib, datetime, hashlib, json, math, os, shutil, sqlite3, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STATES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "BLOCKED", "RETRYING")


def utc(t=None):
    return datetime.datetime.fromtimestamp(
        time.time() if t is None else t, datetime.timezone.utc
    ).isoformat()


def atomic(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp." + str(os.getpid()))
    tmp.write_text(
        data
        if isinstance(data, str)
        else json.dumps(data, ensure_ascii=False, indent=2)
    )
    os.replace(tmp, p)


def db(root=ROOT):
    c = sqlite3.connect(Path(root) / "state/experiments.sqlite", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


SCHEMA = """CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,experiment_id TEXT,dataset_id TEXT,spec TEXT,input_hash TEXT,code_commit TEXT,status TEXT,attempt INTEGER DEFAULT 0,unit TEXT,pid INTEGER,start REAL,end REAL,heartbeat REAL,output_paths TEXT,validation TEXT,failure TEXT,retry_at REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,time REAL,task_id TEXT,event TEXT,details TEXT);
CREATE TABLE IF NOT EXISTS resources(time REAL,payload TEXT);
CREATE TABLE IF NOT EXISTS reports(time REAL PRIMARY KEY,path TEXT,payload TEXT);"""


def init(root=ROOT):
    for d in ["state", "logs", "checkpoints", "reports", "dashboards", "artifacts"]:
        (Path(root) / d).mkdir(parents=True, exist_ok=True)
    with db(root) as c:
        c.executescript(SCHEMA)
        c.execute(
            "INSERT OR IGNORE INTO meta VALUES (?,?)",
            ("service_started", str(time.time())),
        )
        c.execute(
            "INSERT OR IGNORE INTO meta VALUES (?,?)", ("report_interval_s", "14400")
        )


def event(c, tid, kind, details):
    c.execute(
        "INSERT INTO events(time,task_id,event,details) VALUES (?,?,?,?)",
        (time.time(), tid, kind, json.dumps(details, ensure_ascii=False)),
    )


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def code_version():
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def fingerprint(s):
    files = []
    for entry in s.get("inputs", []):
        p = Path(entry)
        files.append(
            {
                "path": str(p),
                "sha256": sha(p) if p.is_file() else None,
                "exists": p.exists(),
            }
        )
    payload = {
        "dataset_id": s["dataset_id"],
        "model_dataset_id": s.get("model_dataset_id"),
        "seed": s.get("seed"),
        "command": s.get("command"),
        "config": s.get("config"),
        "inputs": files,
        "code_commit": code_version(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def register(specs, root=ROOT):
    init(root)
    with db(root) as c:
        for s in specs:
            if s.get("model_dataset_id") and s["model_dataset_id"] != s["dataset_id"]:
                raise ValueError(
                    "Scene-specific model/dataset mismatch: " + s["task_id"]
                )
            tid = s["task_id"]
            old = c.execute("SELECT * FROM tasks WHERE task_id=?", (tid,)).fetchone()
            if old and old["status"] == "RUNNING":
                continue
            if old:
                c.execute(
                    "UPDATE tasks SET spec=?,experiment_id=?,dataset_id=? WHERE task_id=?",
                    (json.dumps(s), s["experiment_id"], s["dataset_id"], tid),
                )
                continue
            status = "BLOCKED" if s.get("blocked_reason") else "PENDING"
            c.execute(
                "INSERT INTO tasks(task_id,experiment_id,dataset_id,spec,code_commit,status,failure) VALUES (?,?,?,?,?,?,?)",
                (
                    tid,
                    s["experiment_id"],
                    s["dataset_id"],
                    json.dumps(s),
                    code_version(),
                    status,
                    s.get("blocked_reason"),
                ),
            )
            event(c, tid, "REGISTERED", {"status": status})


def lookup(data, path):
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def validate(s, started=None):
    checks = s.get("validation", [])
    evidence = []
    if not checks:
        raise ValueError("No acceptance criteria; cannot mark experiment complete")
    for v in checks:
        p = Path(v["path"])
        if not p.is_file() or p.stat().st_size < v.get("min_bytes", 1):
            raise ValueError("Missing/empty output: " + str(p))
        if (
            started is not None
            and v.get("fresh", True)
            and p.stat().st_mtime < started - 2
        ):
            raise ValueError("Stale output from earlier attempt: " + str(p))
        if v.get("kind") == "json":
            data = json.loads(p.read_text())
            for k, expected in v.get("equals", {}).items():
                actual = lookup(data, k)
                if actual != expected:
                    raise ValueError(f"{p}: {k}={actual!r}, expected {expected!r}")
            for k in v.get("finite", []):
                x = lookup(data, k)
                if (
                    not isinstance(x, (int, float))
                    or isinstance(x, bool)
                    or not math.isfinite(x)
                ):
                    raise ValueError("Invalid metric " + k)
            for k, minimum in v.get("minimum", {}).items():
                if lookup(data, k) < minimum:
                    raise ValueError("Insufficient completed work: " + k)
        if v.get("kind") == "npz":
            import numpy as np

            with np.load(p, allow_pickle=False) as z:
                for k, n in v.get("lengths", {}).items():
                    if len(z[k]) != n:
                        raise ValueError(f"{k}: wrong frame count")
                for k in v.get("finite", []):
                    if not np.isfinite(z[k]).all():
                        raise ValueError("Nonfinite array " + k)
        evidence.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    return {"accepted": True, "validated_utc": utc(), "artifacts": evidence}


def imported(root=ROOT):
    with db(root) as c:
        rows = c.execute("SELECT * FROM tasks WHERE status='PENDING'").fetchall()
        for row in rows:
            s = json.loads(row["spec"])
            if not s.get("reuse_existing"):
                continue
            try:
                result = validate(s)
            except Exception as exc:
                event(c, row["task_id"], "REUSE_REJECTED", str(exc))
                continue
            h = fingerprint(s)
            c.execute(
                "UPDATE tasks SET status='COMPLETED',input_hash=?,validation=?,output_paths=?,end=? WHERE task_id=?",
                (
                    h,
                    json.dumps(result),
                    json.dumps([v["path"] for v in s["validation"]]),
                    time.time(),
                    row["task_id"],
                ),
            )
            event(c, row["task_id"], "REUSED_VALID_RESULT", result)


def resources():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.strip().split()[0]) * 1024
    disk = shutil.disk_usage("/data")
    r = {
        "time_utc": utc(),
        "load1": os.getloadavg()[0],
        "cpu_count": os.cpu_count(),
        "memory_available_bytes": info["MemAvailable"],
        "memory_total_bytes": info["MemTotal"],
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
    }
    try:
        v = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()[0]
            .split(",")
        )
        r.update(
            gpu_total_mib=int(v[0]), gpu_used_mib=int(v[1]), gpu_util_percent=int(v[2])
        )
        r["gpu_free_mib"] = r["gpu_total_mib"] - r["gpu_used_mib"]
    except Exception as exc:
        r["gpu_error"] = str(exc)
    return r


def due(start, last, now, interval=14400):
    return now >= start + interval * (max(0, int((last - start) // interval)) + 1)
