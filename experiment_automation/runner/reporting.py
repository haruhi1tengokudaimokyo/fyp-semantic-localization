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
                r"ate_|recall|mean_iou|availability|epochs|runtime_|failure_rate|finished_frames|optimizer_steps|translation|rotation|rpe|drift|duration|coverage|inliers",
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
            d["metric_context"] = {}
            if d["status"] == "COMPLETED":
                for check in s.get("validation", []):
                    if check.get("kind") == "json":
                        d["key_metrics"].update(selected_metrics(check["path"]))
                        try:
                            measured = json.loads(Path(check["path"]).read_text())
                            cache = measured.get("frontend_cache")
                            if cache is not None:
                                d["metric_context"]["frontend_cache"] = cache
                                d["metric_context"][
                                    "runtime_scope"
                                ] = "Cached frontend hits are excluded from neural inference cost. Warm-cache Fast timing is not comparable to mostly-live Paper end-to-end timing."
                        except Exception:
                            pass
                if d["start"] is not None and d["end"] is not None:
                    d["key_metrics"]["worker_wall_seconds"] = d["end"] - d["start"]
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
                        "formal_completed_epoch_runs": 0,
                        "pilot_epoch_runs": 0,
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
                                image_count = max(
                                    image_count,
                                    measured.get("metrics", {}).get(
                                        "raw_rgb_frames", 0
                                    ),
                                )
                                if isinstance(measured.get("frames"), list):
                                    image_count = max(
                                        image_count, len(measured["frames"])
                                    )
                                if measured.get("building_plan"):
                                    cov["audited_synthetic_layout_groups"] = measured[
                                        "independent_layout_groups"
                                    ]
                                    cov["rendered_images_by_split"] = {
                                        split: sum(
                                            row["split"] == split
                                            for row in measured["frames"]
                                        )
                                        for split in ["train", "val", "test"]
                                    }
                                if "raw_duration_s" in measured.get(
                                    "stereo_imu", {}
                                ).get("metrics", {}):
                                    cov.setdefault("sequence_durations_s", {})[
                                        d["dataset_id"]
                                    ] = measured["stereo_imu"]["metrics"][
                                        "raw_duration_s"
                                    ]
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
                epoch_count = v.get("epochs", d["key_metrics"].get("epochs", 0))
                cov["epochs"] += epoch_count
                cov["formal_completed_epoch_runs"] += epoch_count if d["formal"] else 0
                cov["pilot_epoch_runs"] += epoch_count if not d["formal"] else 0
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
        """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FYP Live Experiments</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172034}h1{margin-bottom:4px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}button,input,select{padding:8px} .RUNNING{color:#0867ae}.COMPLETED{color:#087b44}.FAILED,.BLOCKED{color:#ad3434}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{padding:14px;background:white;border-radius:8px;min-width:110px}.card b{display:block;font-size:28px}.scroll{max-height:540px;overflow:auto}details{background:white;padding:12px;margin:12px 0}#running{background:#eaf3ff;padding:12px}small{color:#516177}</style>
<h1>FYP 持久实验 Dashboard</h1><small>真实 SQLite / 原始日志；初步结果与正式实验分别标记</small><p id="summary"></p><div id="counts" class="cards"></div><p id="resources"></p>
<h2>正在执行</h2><pre id="running"></pre><h2>实际完成覆盖</h2><div id="coverage"></div><small>序列按 dataset ID 去重；图像运行量包含重复模型评价，不能解释为新增独立样本。建筑仅计入经过划分审查的合成布局组。</small>
<h2>全量任务队列</h2><input id="search" placeholder="按任务 / 数据集搜索"><select id="filter"><option value="">全部状态</option><option>RUNNING</option><option>PENDING</option><option>COMPLETED</option><option>BLOCKED</option><option>FAILED</option><option>RETRYING</option></select><button onclick="location.reload()">刷新</button><div class="scroll" id="queue"></div>
<details><summary>已验收的真实指标与对照（含 formal 标记）</summary><pre id="metrics"></pre></details><details><summary>最近四小时事件、失败诊断与恢复证据</summary><pre id="events"></pre></details><details><summary>原始资源与覆盖记录</summary><pre id="raw"></pre></details>
<script>let state="""
        + data
        + """;
function table(headers,rows){let t=document.createElement('table');let h=t.insertRow();for(let x of headers)h.appendChild(Object.assign(document.createElement('th'),{textContent:x}));for(let values of rows){let r=t.insertRow();for(let v of values)r.insertCell().textContent=v;}return t;}
function render(s){state=s;document.querySelector('#summary').textContent=s.generated_utc+' | 已运行 '+(s.elapsed_seconds/3600).toFixed(2)+' 小时 | 下次报告 '+s.next_four_hour_due_utc+' | Git '+s.git_commit.slice(0,12);
let cards=[];for(let [status,n] of Object.entries(s.counts)){let e=document.createElement('div');e.className='card '+status;let b=document.createElement('b');b.textContent=n;e.append(b,document.createTextNode(status));cards.push(e);}document.querySelector('#counts').replaceChildren(...cards);
let r=s.resources;document.querySelector('#resources').textContent='GPU '+r.gpu_used_mib+'/'+r.gpu_total_mib+' MiB · '+r.gpu_util_percent+'% | CPU load '+r.load1.toFixed(2)+' / '+r.cpu_count+' cores | 可用内存 '+(r.memory_available_bytes/1073741824).toFixed(1)+' GiB | 可用磁盘 '+(r.disk_free_bytes/1073741824).toFixed(1)+' GiB';
document.querySelector('#running').textContent=s.tasks.filter(t=>t.status==='RUNNING').map(t=>t.task_id+' | PID '+t.pid+' | attempt '+t.attempt+' | 最近日志 '+t.last_valid_log_utc+'\\n'+(t.log_tail||'')).join('\\n\\n')||'当前没有运行任务；请检查依赖与资源状态。';
document.querySelector('#coverage').replaceChildren(table(['模块','正式完成任务','完整序列（去重）','正式图像运行量','初步图像运行量','正式完成 epoch','合成布局组'],Object.entries(s.coverage).map(([k,c])=>[k,c.formal_tasks,c.unique_full_dataset_sequences,c.formal_completed_image_runs,c.pilot_image_runs,c.formal_completed_epoch_runs,c.audited_synthetic_layout_groups??"未审查"])));
let filter=document.querySelector('#filter').value,search=document.querySelector('#search').value.toLowerCase();let tasks=s.tasks.filter(t=>(!filter||t.status===filter)&&(!search||(t.task_id+' '+t.dataset_id).toLowerCase().includes(search)));document.querySelector('#queue').replaceChildren(table(['任务','模块 / 数据集','正式','状态','尝试','最近日志','失败原因'],tasks.map(t=>[t.task_id,t.experiment_id+' / '+t.dataset_id,t.formal,t.status,t.attempt,t.last_valid_log_utc||'',t.failure||''])));
document.querySelector('#metrics').textContent=JSON.stringify(s.tasks.filter(t=>t.status==='COMPLETED'&&Object.keys(t.key_metrics||{}).length).map(t=>({task:t.task_id,formal:t.formal,dataset:t.dataset_id,context:t.metric_context,metrics:t.key_metrics})),null,2);document.querySelector('#events').textContent=JSON.stringify(s.recent_events,null,2);document.querySelector('#raw').textContent=JSON.stringify({resources:s.resources,coverage:s.coverage},null,2);}
document.querySelector('#search').addEventListener('input',()=>render(state));document.querySelector('#filter').addEventListener('change',()=>render(state));render(state);if(location.protocol.startsWith('http'))setInterval(()=>fetch('/api/status').then(r=>r.json()).then(render),10000);</script></html>"""
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
