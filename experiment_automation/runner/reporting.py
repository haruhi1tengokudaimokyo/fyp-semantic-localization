"""Actual database/log snapshots, UTC-aligned four-hour reporting, no sample data."""

import argparse, csv, html, json, time, subprocess
from pathlib import Path
from core import ROOT, REPO, db, init, atomic, utc, resources, code_version, due


def tail(path):
    with Path(path).open("rb") as f:
        f.seek(max(0, Path(path).stat().st_size - 8192))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-6:])


def selected_metrics(path):
    import re

    result = {}
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return result

    def walk(v, prefix="", depth=0):
        if depth > 5 or not isinstance(v, dict):
            return
        for k, x in v.items():
            key = prefix + k
            if isinstance(x, (int, float)) and re.search(
                r"ate_|recall|mean_iou|availability|epochs|runtime_|failure_rate|finished_frames|optimizer_steps",
                key,
            ):
                result[key] = x
            elif isinstance(x, dict):
                walk(x, key + ".", depth + 1)

    walk(data)
    return result


def snapshot(root=ROOT):
    now = time.time()
    with db(root) as c:
        start = float(
            c.execute("SELECT value FROM meta WHERE key='service_started'").fetchone()[
                0
            ]
        )
        tasks = []
        counts = {
            k: 0
            for k in [
                "PENDING",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "BLOCKED",
                "RETRYING",
            ]
        }
        coverage = {}
        unique_sequences = {}
        unique_images = {}
        for row in c.execute("SELECT * FROM tasks ORDER BY experiment_id,task_id"):
            d = dict(row)
            s = json.loads(d.pop("spec"))
            d["resources"] = s.get("resources", {})
            d["seed"] = s.get("seed")
            d["formal"] = s.get("formal", False)
            d["outputs"] = json.loads(d["output_paths"] or "[]")
            d["validation"] = json.loads(d["validation"]) if d["validation"] else None
            counts[d["status"]] += 1
            if d["status"] == "RUNNING":
                log = Path(root) / "logs" / f"{d['task_id']}.attempt{d['attempt']}.log"
                d["log_path"] = str(log)
                d["last_valid_log_utc"] = (
                    utc(log.stat().st_mtime) if log.exists() else None
                )
                if log.exists():
                    d["log_tail"] = tail(log)
            d["key_metrics"] = {}
            if d["status"] == "COMPLETED":
                for check in s.get("validation", []):
                    if check.get("kind") == "json":
                        d["key_metrics"].update(selected_metrics(check["path"]))
            tasks.append(d)
            if d["status"] == "COMPLETED":
                cov = coverage.setdefault(
                    d["experiment_id"],
                    {
                        "tasks": 0,
                        "formal_tasks": 0,
                        "full_sequence_runs": 0,
                        "formal_completed_image_runs": 0,
                        "pilot_image_runs": 0,
                        "images": 0,
                        "epochs": 0,
                        "scene_ids": [],
                        "building_ids": [],
                    },
                )
                cov["tasks"] += 1
                cov["formal_tasks"] += int(d["formal"])
                v = s.get("coverage", {})
                cov["full_sequence_runs"] += int(
                    v.get("full_sequence", False) and d["formal"]
                )
                image_count = v.get("images", 0)
                if not image_count and d["formal"]:
                    for check in s.get("validation", []):
                        if check.get("kind") == "json":
                            try:
                                measured = json.loads(Path(check["path"]).read_text())
                                image_count = max(
                                    image_count,
                                    measured.get("n", 0),
                                    measured.get("finished_frames", 0),
                                )
                            except Exception:
                                pass
                cov["images"] += image_count
                cov["formal_completed_image_runs"] += image_count if d["formal"] else 0
                cov["pilot_image_runs"] += image_count if not d["formal"] else 0
                if d["formal"] and v.get("full_sequence"):
                    unique_sequences.setdefault(d["experiment_id"], set()).add(
                        d["dataset_id"]
                    )
                    unique_images.setdefault(d["experiment_id"], {})[
                        d["dataset_id"]
                    ] = max(
                        unique_images.get(d["experiment_id"], {}).get(
                            d["dataset_id"], 0
                        ),
                        image_count,
                    )
                cov["epochs"] += v.get("epochs", 0)
                for k in ["scene_ids", "building_ids"]:
                    cov[k] = sorted(set(cov[k] + v.get(k, [])))
        for group, cov in coverage.items():
            cov["unique_full_dataset_sequences"] = len(
                unique_sequences.get(group, set())
            )
            cov["unique_full_sequence_images"] = sum(
                unique_images.get(group, {}).values()
            )
            cov["image_count_note"] = (
                "Image-runs include repeated model evaluations; pilots are separated. Unique sequence counts deduplicate dataset IDs. Scene IDs do not prove physical building identity."
            )
        recent = [
            dict(r)
            for r in c.execute(
                "SELECT * FROM events WHERE time>=? ORDER BY id DESC LIMIT 200",
                (now - 14400,),
            )
        ]
        for e in recent:
            if len(e["details"]) > 3000:
                e["details"] = (
                    e["details"][:3000]
                    + " ... [full evidence retained in SQLite/artifacts]"
                )
        old = c.execute(
            "SELECT payload FROM reports WHERE time<=? ORDER BY time DESC LIMIT 1",
            (now - 14400,),
        ).fetchone()
        previous = json.loads(old[0])["counts"] if old else {}
        meta = {r["key"]: r["value"] for r in c.execute("SELECT * FROM meta")}
    return dict(
        generated_utc=utc(now),
        elapsed_seconds=now - start,
        service_started_utc=utc(start),
        next_four_hour_due_utc=utc(start + 14400 * (int((now - start) // 14400) + 1)),
        counts=counts,
        delta_vs_4h={k: counts[k] - previous.get(k, 0) for k in counts},
        coverage=coverage,
        resources=resources(),
        tasks=tasks,
        recent_events=recent,
        git_commit=code_version(),
        reporting_note="Reports use actual wall-clock execution time. No computation is claimed during shutdown/suspend. Scene IDs are not automatically counted as independent buildings.",
        meta=meta,
    )


def html_page(s):
    data = json.dumps(s, ensure_ascii=False).replace("</", "<\\/")
    return (
        """<!doctype html><meta charset="utf-8"><title>FYP Live Experiments</title><style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172034}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:8px} .RUNNING{color:#0867ae}.COMPLETED{color:#087b44}.FAILED,.BLOCKED{color:#ad3434}</style><h1>FYP 实验状态 · 真实 SQLite 数据</h1><button onclick="location.reload()">刷新</button><p id="summary"></p><pre id="resources"></pre><h2>已验收实验指标（含明确标记的初步结果）</h2><pre id="metrics"></pre><div id="queue"></div><h2>最近四小时事件</h2><pre id="events"></pre><script>let state="""
        + data
        + """;function render(s){document.querySelector('#summary').textContent=s.generated_utc+' | '+JSON.stringify(s.counts)+' | commit '+s.git_commit;document.querySelector('#resources').textContent=JSON.stringify({elapsed_seconds:s.elapsed_seconds,resources:s.resources,coverage:s.coverage},null,2);document.querySelector('#metrics').textContent=JSON.stringify(s.tasks.filter(t=>t.status==='COMPLETED'&&Object.keys(t.key_metrics||{}).length).map(t=>({task:t.task_id,formal:t.formal,dataset:t.dataset_id,metrics:t.key_metrics})),null,2);let table=document.createElement('table');let header=table.insertRow();for(let h of ['任务','实验/数据','状态','尝试','最近日志','失败诊断'])header.appendChild(Object.assign(document.createElement('th'),{textContent:h}));for(let t of s.tasks){let row=table.insertRow();row.className=t.status;for(let v of [t.task_id,t.experiment_id+' / '+t.dataset_id,t.status,t.attempt,t.last_valid_log_utc||'',t.failure||''])row.insertCell().textContent=v;}document.querySelector('#queue').replaceChildren(table);document.querySelector('#events').textContent=JSON.stringify(s.recent_events,null,2);}render(state);if(location.protocol.startsWith('http'))setInterval(()=>fetch('/api/status').then(r=>r.json()).then(render),10000);</script>"""
    )


def report(root=ROOT, force=False):
    init(root)
    s = snapshot(root)
    from aggregate_results import aggregate

    s["aggregate_index"] = aggregate(s, root)
    now = time.time()
    with db(root) as c:
        start = float(s["meta"]["service_started"])
        row = c.execute("SELECT MAX(time) FROM reports").fetchone()
        last = row[0]
        if not force and last is not None and not due(start, last, now):
            return None
        filename = "progress_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
        md = [
            "# FYP 持久实验进度",
            f"生成：{s['generated_utc']}。启动后真实经过 {s['elapsed_seconds']/3600:.2f} 小时。",
            "",
            "状态：" + json.dumps(s["counts"], ensure_ascii=False),
            "",
            "## 覆盖与结果来源",
            "```json",
            json.dumps(s["coverage"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 资源",
            "```json",
            json.dumps(s["resources"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 任务",
            "|task|experiment/dataset|status|attempt|failure|",
            "|---|---|---|---|---|",
        ]
        for t in s["tasks"]:
            md.append(
                "|"
                + "|".join(
                    str(v).replace("|", "/").replace("\n", " ")
                    for v in [
                        t["task_id"],
                        t["experiment_id"] + "/" + t["dataset_id"],
                        t["status"],
                        t["attempt"],
                        t["failure"] or "",
                    ]
                )
                + "|"
            )
        md += [
            "",
            "## 四小时变化",
            json.dumps(s["delta_vs_4h"], ensure_ascii=False),
            "",
            "## 最近事件",
            "```json",
            json.dumps(s["recent_events"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"Git: {s['git_commit']}",
            "大型数据、权重和完整日志保存在 /data/fyp_3d；没有将不同任务类型混成平均排行榜。",
            "停机/休眠期间不能计算；重启后恢复任务和最新真实快照。",
        ]
        md.extend(["", "## 实际关键结果与对照"])
        for task in s["tasks"]:
            if task.get("key_metrics"):
                md.extend(
                    [
                        task["task_id"],
                        "```json",
                        json.dumps(task["key_metrics"], ensure_ascii=False, indent=2),
                        "```",
                    ]
                )
        text = "\n".join(md)
        path = Path(root) / "reports" / (filename + ".md")
        atomic(path, text)
        atomic(Path(root) / "reports/LATEST.md", text)
        atomic(Path(root) / "reports/LATEST.json", s)
        atomic(Path(root) / "dashboards/index.html", html_page(s))
        atomic(Path(root) / "reports" / (filename + ".json"), s)
        c.execute(
            "INSERT OR REPLACE INTO reports VALUES (?,?,?)",
            (now, str(path), json.dumps(s)),
        )
        print(path)
    # Online SQLite backup gives a consistent snapshot even while workers commit.
    import sqlite3, os

    backup = Path(root) / "checkpoints" / ("state_" + filename + ".sqlite")
    temporary = backup.with_suffix(".sqlite.tmp")
    with db(root) as source, sqlite3.connect(temporary) as target:
        source.backup(target)
    os.replace(temporary, backup)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args()
    report(Path(a.root), a.force)
