"""Measured reproduction ETA and independent, honest time-checkpoint reporting.

No localization job is launched here. A missing four-paper time baseline stays
missing until all four core pipelines have measured representative runtimes.
"""

import argparse, collections, datetime, hashlib, html, json, math, re, sqlite3, statistics, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiment_automation"
BASE = Path("/data/fyp_3d")
CST = datetime.timezone(datetime.timedelta(hours=8))
OUTPUTS = REPO.parents[1] / "outputs/FYP_Automation"


def stamp(t=None):
    return datetime.datetime.fromtimestamp(
        time.time() if t is None else t, CST
    ).isoformat(timespec="seconds")


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    temp.replace(path)


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def measured_ace_eta(tasks, now):
    """Project only the unchanged executable 7Scenes 8M train+eval queue."""
    train = [
        t
        for t in tasks
        if t["task_id"].startswith("R.ACE.7scenes.")
        and t["task_id"].endswith(".train8M")
    ]
    rates = []
    buffers = []
    current = {}
    total_steps = 16 * (8000000 // 5120)
    for t in train:
        p = ROOT / "logs" / f"{t['task_id']}.attempt{t['attempt']}.log"
        if not p.exists():
            continue
        samples = [
            (int(i), float(s))
            for i, s in re.findall(
                r"Iteration:\s+(\d+).*?Time:\s+([\d.]+)s", p.read_text(errors="replace")
            )
        ]
        dt = [
            (b[1] - a[1]) / (b[0] - a[0])
            for a, b in zip(samples, samples[1:])
            if b[0] > a[0] and b[1] > a[1]
        ]
        if len(dt) >= 3:
            step = statistics.median(dt[-100:])
            rates.append(step)
            buffers.append(max(0, samples[0][1] - samples[0][0] * step))
            if t["status"] == "RUNNING":
                current[t["task_id"]] = samples[-1][0]
    if not rates:
        return dict(
            status="not_measured",
            scope="ACE 7Scenes 8M queue only",
            four_paper_completion_date=None,
        )
    step = statistics.median(rates)
    buffer = statistics.median(buffers)
    eval_tasks = [
        t
        for t in tasks
        if t["experiment_id"] == "A"
        and t["status"] == "COMPLETED"
        and t["start"]
        and t["end"]
        and (t["task_id"].endswith(".official") or t["task_id"].endswith(".eval"))
    ]
    sec_per_query = []
    for t in eval_tasks:
        for v in t["spec"].get("validation", []):
            if v.get("kind") == "json":
                s = load(v["path"])
                if isinstance(s, dict) and s.get("n"):
                    sec_per_query.append((t["end"] - t["start"]) / s["n"])
    per_query = statistics.median(sec_per_query) if sec_per_query else None
    remain = 0.0
    details = []
    for t in train:
        if t["status"] == "COMPLETED":
            secs = 0.0
        elif t["task_id"] in current:
            secs = max(0, total_steps - current[t["task_id"]]) * step
        else:
            secs = buffer + total_steps * step
        remain += secs
        details.append(
            dict(
                task_id=t["task_id"],
                status=t["status"],
                estimated_remaining_seconds=secs,
            )
        )
    audit = load(BASE / "runs/reproduction/ACE/data_audit_7scenes.json")
    query_total = (
        sum(s["splits"]["test"]["frames"] for s in audit["scenes"]) if audit else 17000
    )
    eval_remaining = [
        t
        for t in tasks
        if t["task_id"].startswith("R.ACE.7scenes.")
        and t["task_id"].endswith(".official_eval")
        and t["status"] != "COMPLETED"
    ]
    # New task names are checked below; absence of an estimate is explicit.
    all_new_eval = [
        t
        for t in tasks
        if t["task_id"].startswith("R.ACE.7scenes.") and ".eval" in t["task_id"]
    ]
    remaining_queries = query_total
    for t in all_new_eval:
        if t["status"] == "COMPLETED":
            for v in t["spec"].get("validation", []):
                data = load(v["path"])
                if isinstance(data, dict):
                    remaining_queries -= data.get("n", 0)
    evaluation = remaining_queries * per_query if per_query else None
    known = remain + (evaluation or 0)
    return dict(
        status="measured_partial_queue_projection",
        scope="ACE 7Scenes full 8M train+upstream eval, excluding blocked 12Scenes and other three papers",
        measured_seconds_per_optimizer_step=step,
        measured_feature_buffer_seconds=buffer,
        steps_per_head=total_steps,
        remaining_train_seconds=remain,
        remaining_eval_seconds=evaluation,
        evaluation_estimate_source="Historical complete same-dataset ACE evaluation worker wall/query; upstream-script wrapper difference remains uncertain",
        remaining_eval_queries=remaining_queries,
        remaining_seconds=known,
        assumes_no_other_GPU_work=True,
        uncertainty="One unfinished full-8M head used for extrapolation; future scene I/O and CPU DSAC vary. Not a completed-head timing.",
        scenarios={
            name: dict(
                remaining_seconds=known * factor,
                completion_cst=stamp(now + known * factor),
            )
            for name, factor in [
                ("optimistic", 0.8),
                ("normal", 1.0),
                ("conservative", 1.5),
            ]
        },
        four_paper_completion_date=None,
        heads=details,
    )


def due_milestones(state, baseline, now):
    """Time progress only. No inferred authorization or fabricated total duration."""
    if (
        not baseline
        or baseline.get("scope") != "four_paper_total"
        or not baseline.get("representative_core_measurements_complete")
    ):
        return []
    duration = float(baseline["wall_seconds"])
    start = float(baseline["start_epoch"])
    assert duration > 0 and math.isfinite(duration) and math.isfinite(start)
    done = set(state.get("emitted", []))
    return [
        p
        for p in (25, 50, 75, 100)
        if p not in done and now >= start + duration * p / 100
    ]


def read_tasks():
    with sqlite3.connect(
        f"file:{ROOT/'state/experiments.sqlite'}?mode=ro", uri=True
    ) as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute("SELECT * FROM tasks ORDER BY task_id")]
    for r in rows:
        r["spec"] = json.loads(r["spec"])
    return rows


def role(t):
    s = t["spec"]
    tid = t["task_id"]
    if s.get("task_role"):
        return s["task_role"]
    if any(
        k in tid
        for k in [
            "audit",
            "probe",
            "environment",
            "dependencies",
            "runtime",
            "inventory",
            "download",
            "models.all",
            "data.",
            "weights",
            "qualification",
        ]
    ):
        return "supporting_data_or_runtime"
    if ".train" in tid:
        return "training"
    return "evaluation_or_method"


def build(now=None):
    now = time.time() if now is None else now
    tasks = read_tasks()
    eta = measured_ace_eta(tasks, now)
    datasets = []
    for t in tasks:
        if t["status"] == "COMPLETED" and t["task_id"].startswith(
            "R.data.hypersim.subset.v3."
        ):
            a = load(
                BASE
                / "datasets/hypersim_formal_subset"
                / t["task_id"].split(".")[-1]
                / "v3/acceptance.json"
            )
            if a:
                datasets.append(a)
    data = dict(
        generated_cst=stamp(now),
        four_papers_completed=False,
        total_counts=dict(collections.Counter(t["status"] for t in tasks)),
        four_paper_total_eta=dict(
            status="blocked_unmeasured",
            completion_cst=None,
            reasons=[
                "MAC-VO original multi-sequence source access incomplete",
                "SG2Loc legal raw benchmark or real Replica mesh not acquired; core PF/refinement not executed",
                "OMCL map+localizer on benchmark not executed; encoder/kernel probes are not runtime measurements",
            ],
        ),
        ace_queue_eta=eta,
        hypersim_v3=dict(
            accepted_scenes=len(datasets),
            accepted_frames=sum(a["acquired_frames"] for a in datasets),
            source_families_are_physical_buildings=False,
            subset_qualification_passed=any(
                t["task_id"] == "R.data.hypersim.cleaned_qualification.v3"
                and t["status"] == "COMPLETED"
                for t in tasks
            ),
            scene_records=[
                {
                    k: a[k]
                    for k in [
                        "scene",
                        "acquired_frames",
                        "partition",
                        "geometry_round_trip",
                        "downloaded_scientific_bytes",
                    ]
                }
                for a in datasets
            ],
        ),
        tasks=[
            {
                k: t[k]
                for k in [
                    "task_id",
                    "experiment_id",
                    "dataset_id",
                    "status",
                    "attempt",
                    "start",
                    "end",
                    "paper_id",
                    "reproduction_level",
                    "online_input_condition",
                ]
            }
            | {"role": role(t), "config": t["spec"]}
            for t in tasks
        ],
    )
    lines = [
        "# FYP 原论文复现报告：当前审计版本",
        f"采样：{stamp(now)}。四篇完整复现尚未完成；此报告随持久实验推进更新。",
        "## 总体结论",
        "各论文的原始训练、权重评价、硬件适配、替代数据和独立基线分别报告。数据审计、模型导入、CUDA 内核与 Batch=1 探针不能算核心定位实验。任务完成数量不是论文完成率。",
        "|论文|已执行证据|正式完整复现尚缺|",
        "|---|---|---|",
        "|MAC-VO|作者提供的 P001_select 1,274 帧完整演示；Paper/Fast/独立 SGBM 对照；同匹配后端消融|多条原始完整基准轨迹及对应论文行|",
        "|ACE|7Scenes 全场景原始划分核查；官方权重及本地 1M 结果保留；原始 8M/16epoch 训练在运行|7 个 8M 场景头及上游评价；12Scenes 实际源数据|",
        "|SG2Loc|固定源码、授权审查、隔离依赖与 CUDA 恢复证据|真实场景图、PF、Refinement、长度5/10/25完整实验|",
        "|OMCL|固定源码、GT运动协议审计；Kaolin CUDA通过；原始LSeg权重校验|语言地图、原始MCL与全局初始化的完整实验|",
        "## 数据处理和隔离",
        f"Hypersim v3 已通过 {len(datasets)} 个场景、{data['hypersim_v3']['accepted_frames']} 张实际图像。冻结目标51场景/10,493帧不算已下载结果。所有51个候选场景都必须得到终态数据审计；失败场景按预先声明的CRC/标定规则隔离。实际保留至少40场景/8000帧、原计划帧身份不变、跨划分解码RGB哈希无重叠后，才生成显式清洗决策文件和训练资格；不得依据定位测试结果更换场景。旧v1/v2验收继续保留。",
        "Hypersim 使用作者发布有效帧清单，排除作者移除帧；ai_### 仅为来源族代理，尚无证据证明物理建筑身份。按来源族隔离的自定义划分不冒称官方 v1 划分。无免费公开完整原始 Mesh，不将其冒充已知地图定位基准。",
        "标定 v3 将作者每帧 world_from_camera 基矩阵、M_cam_from_uv 和半像素中心变换整体做RQ解析分解，获得逐帧内参与刚体相机位姿；源相机轴的各向异性不再被误当作非法旋转，也不用SVD把它强行投影成旋转。该过程不使用图像或世界坐标观测拟合。OpenGL 轴到 OpenCV 轴转换同时作用于位姿。光心距离先转浮点64位，再转 Z-depth；世界坐标在单位换算前转浮点64位，避免二次 float16 舍入。",
        "作者 generate_hdf5_from_vrimg.py 对 position 和 depth_meters 均存 float16。v3 对每台相机首个选中帧的每个有效像素，要求重建误差不超过位置向量半ULP与光心距离半ULP之和，加1微米数值运算余量；零超界才通过。它依据数据存储格式解析推导，不依据定位测试成绩调阈值。旧固定5毫米验收及其失败记录保留。",
        "每一选中 ZIP 成员核验 CRC、长度和 SHA256；分块 HTTP 必须206且 Content-Range/ETag一致。先保存独立 raw receipt，再进行标定资格验收。RGB/语义/实例/深度按场景、相机、帧ID连接；GT位姿单独文件。当前几何反投影审计每台相机首个选中帧，其余帧做形状/标签/有效深度/RGB解码检查，不能声称所有像素均已做世界坐标核查。",
        "ADE20K 原始标签0为ignore，1..150转0..149；Hypersim有符号-1为ignore，1..40转0..39。未记录标签直接拒绝。ADE原始数据尚未获取，SegFormer原始160k训练未执行。",
        "Replica 全部17分卷经官方 Range 探测总压缩33,859,047,808字节，解压还需空间；完整18场景尚未获取；2026-11-01用户已授权清理旧派生数据，实际腾出59.94GiB，当前空间见交接报告，不再沿用清理前的空间阻塞结论。没有用 ReplicaCAD 或渲染资产代替原始 Replica。",
        "3RScan/ScanNet 需要授权；未找到本机授权脚本。SG2Loc约19GB预处理包不能代替原始授权数据。HPointLoc源于Matterport3D，代码MIT不自动授权源数据；其GPS/quat是评估真值，不能用于在线输入。",
        "## MAC-VO",
        "研究问题：双目匹配的不确定性经三维协方差进入加权局部优化。原始网络、Paper FP32与Fast缓存条件分开。官方代码 https://github.com/MAC-VO/MAC-VO ，固定Commit 8683b532d6f6e8981f6b1a646fb5f52f02c1ea61。",
        "数据：作者 P001_select 演示为1,274帧完整提供文件，不证明是完整父基准轨迹。没有将其映射到未经核实的论文表格行。EuRoC旧源TLS/HTTP失败，当前官方数据入口曾返回429；不绕过授权或请求限制。",
        "现存三个完整演示轨迹的 SE3 对齐 ATE（不校尺度）分别为 Paper 0.226573m、Fast 1.8457m、SGBM 16.47m。这里最后两项为历史摘要近似值；完整精确数值与官方指标由后附JSON提供。Fast全1274帧前端命中缓存，不能与Paper大多实时推理比较端到端FPS。",
        "官方 MetricsSeq 同时传align与align_origin；evo1.38拒绝。独立 evo1.29 保留作者时代Umeyama优先语义，不降级共享VO环境。官方旋转评价注释建议序列化SO3容差1e-4；仅评估进程采用该容差，不修改轨迹或共享库。t_rel是m/frame、r_rel是deg/frame，不冒称每米百分比。",
        "复现等级：Hardware Adapted，独立SGBM为Independent Baseline。未完成多序列论文对照，因此现有结果仅验证工程链条，不足以支持全部论文主张。",
        "## ACE",
        "研究问题：固定共享编码器、场景头回归三维场景坐标，DSAC*获得位姿。官方 https://github.com/nianticlabs/ace 。共享编码器使用官方权重，未声称重新训练未公开部分。",
        "7Scenes 逐图检查官方训练/测试序列、RGB/pose/校准连接。原D-SLAM深度相机位姿按官方外参转换到RGB；Pseudo-GT单独分支，未混用。各场景训练/测试张数如下。",
        "|场景|训练|测试|完全相同RGB跨划分重叠|",
        "|---|---:|---:|---:|",
    ]
    audit = load(BASE / "runs/reproduction/ACE/data_audit_7scenes.json")
    if audit:
        for a in audit["scenes"]:
            lines.append(
                f"|{a['scene']}|{a['splits']['train']['frames']}|{a['splits']['test']['frames']}|{a['train_test_exact_rgb_overlap']}|"
            )
    lines += [
        "原始8,000,000特征缓存、batch5120、16epochs、原优化器/损失/LR/增强不缩减。6GiB显卡用CPU映射缓存逐batch搬运，属于Hardware Adapted，不声称作者GPU上的五分钟速度。本轮正在运行的任务未被重启。旧1M训练、官方权重评价及2089/2090/2091种子均保留，不能当作8M配置结果。",
        "12Scenes 按官方ordinal边界先分割，再排除无效INF位姿；不让清洗改变训练/测试边界。3项实际格式防错测试通过，但四个官方分组ZIP在有限3次尝试后TLS失败；没有实际数据时不解除准备/训练阻塞。",
        "已验收ACE评价结果（不同配置明确分列；adapter与未经修改的上游脚本也分开）：",
        "|任务|等级/配置|查询|Recall 5cm/5deg|中位平移m|中位旋转deg|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for t in tasks:
        if t["status"] != "COMPLETED" or t["paper_id"] != "ACE":
            continue
        if not (
            t["task_id"].endswith((".official", ".eval"))
            or t["task_id"].startswith("reuse.ace")
        ):
            continue
        for check in t["spec"].get("validation", []):
            s = load(check["path"]) if check.get("kind") == "json" else None
            if not isinstance(s, dict) or not s.get("n"):
                continue
            recall = s.get("recall_5cm_5deg", s.get("recall", {}).get("0.05m_5deg"))
            kind = (
                "官方权重/adapter"
                if ".official" in t["task_id"]
                else (
                    "8M/上游脚本" if t["task_id"].startswith("R.") else "本地1M/adapter"
                )
            )
            lines.append(
                f"|{t['task_id']}|{kind}|{s['n']}|{recall}|{s.get('median_translation_m')}|{s.get('median_rotation_deg')}|"
            )
    first_head = load(
        BASE
        / "runs/reproduction/ACE/7scenes/chess/buffer8M_mmap/training_resources.json"
    )
    if first_head:
        lines.append(
            f"首个真实8M场景头 chess 已完成16epochs：wall {first_head['wall_seconds']:.3f}秒，峰值CUDA分配{first_head['torch_peak_allocated_MiB']:.2f}MiB，主机RSS{first_head['process_maxrss_MiB']:.2f}MiB（含映射页）。完整上游测试尚须按DAG验收，不由训练成功推断准确率。"
        )
    protocol = load(BASE / "runs/reproduction/protocol_comparison.json")
    if protocol:
        lines += [
            "原论文对照锚点（仅作者数值；当前未完成对应条件时不计算复现差值）：",
            "ACE Table1：7Scenes SfM97.1%、D-SLAM80.8%；12Scenes SfM99.9%、D-SLAM99.6%，阈值5cm/5deg。当前D-SLAM成绩不能与97.1%的SfM列混用。",
            "MAC-VO TableI：EuRoC全轨迹平均t_rel0.0024m/frame、r_rel0.0403deg/frame；当前TartanAir演示与此列数据集不同，不计算性能差值。",
            "SG2Loc Table1：3RScan长度5/10/25的联合Recall@0.25m/2deg为0.34/0.39/0.44，对应9325/9160/8725测试图像；当前无同条件定位结果。",
            "OMCL TableI：Matterport3D LSeg平面/3D APE RMSE为0.15/0.20m；Replica TableIV使用NARADIO，ATE RMSE35cm。TableII写2048射线，发布配置num_samples1024且采样器按观测mask调用，差异须保留，不能默认为同一总射线预算。",
        ]
    mac_metrics = load(
        BASE / "runs/reproduction/MACVO/official_metrics_existing_demo_v2.json"
    )
    if mac_metrics:
        lines += [
            "MAC-VO 官方函数在现有完整演示上的精确结果（不是多序列论文表复现）：",
            "|配置|ATE RMSE m|t_rel mean m/frame|r_rel mean deg/frame|失踪帧|",
            "|---|---:|---:|---:|---:|",
        ]
        for name, r in mac_metrics["results"].items():
            m = r["metrics"]
            lines.append(
                f"|{name}|{m['ATE_m']['rmse']:.9f}|{m['t_rel_m_per_frame']['mean']:.9f}|{m['r_rel_deg_per_frame']['mean']:.9f}|{r['lost_frames']}|"
            )
    lines += [
        "辅助与核心任务的实时状态（辅助完成不计论文复现完成）：",
        "|任务|状态|角色|复现等级|",
        "|---|---|---|---|",
    ]
    for t in tasks:
        if t["task_id"].startswith(
            ("R.MACVO.", "R.ACE.7scenes.", "R.SG2Loc.", "R.OMCL.")
        ):
            lines.append(
                f"|{t['task_id']}|{t['status']}|{role(t)}|{t['reproduction_level']}|"
            )
    lines += [
        "## SG2Loc",
        "研究问题：在离线三维语义物体图中，通过视觉语义观测、粒子滤波和位姿Refinement连续定位。论文 https://proceedings.mlr.press/v306/damblon26a.html ，代码 https://github.com/DmblnNicole/sg2loc ，Commit e34e498c61b1892b345aeec48573950e371476eb。",
        "原始场景图、SceneGraphLoc、RoMaV2和PF/Refinement不能替换成图像检索。官方3RScan/ScanNet数据尚未合法取得，Replica原始三维实例Mesh也未取得；正式核心方法实验仍BLOCKED。",
        "官方实现使用GT推导的相对运动和逐帧重力轴，深度与候选筛选也需分别审计；这种原论文输入条件不等价于FYP GT-free在线定位。当前隔离环境Torch2.4/cu124与作者环境版本不同。NVRTC缺库已通过新任务安装；实际CUDA内核与官方RayCasting类导入在第二次尝试通过，首个失败保留；依赖通过也不代表SG2Loc复现完成。",
        "5/10/25序列、Refinement开关、Oracle与Predicted Semantic对照均未产生正式定位成绩，不填造对应论文表格。尚无数据证明跨建筑语义定位提升。",
        "## OMCL",
        "研究问题：离线三维语言特征八叉树地图，视觉语言观测与Monte Carlo粒子定位。代码 https://github.com/AIS-Bonn/omcl ，Commit 97fa34482013d393275113e3ab313eb8d20ba9fc。",
        "论文原始Matterport条件含GT派生噪声运动、已知初始位置，允许作为原论文条件独立复现；不能混入GT-free VO或声称全局未知初始化。Replica论文表使用NARADIO编码器；LSeg-on-Replica必须单列为配置/替代数据，不冒称该论文表原样复现。",
        "Kaolin0.18 CUDA Morton内核已在1660SUPER SM75实际通过。LSeg原始demo_e200.ckpt实际3.095GB，SHA256 f21aeb5532f53a5109d453d7d0d10320701ea11ece9a002291f84f95bda73af4；严格checkpoint与Batch1测试已通过：480方形真实RGB裁剪，完整模型前向0.61743秒，PyTorch峰值分配2930.33MiB、保留3420MiB，主机RSS5491.36MiB。该裁剪资源测量不代表作者512裁剪/540特征图建图的完整耗时；没有换小网络。",
        "缺少可用的合法原始三维语义地图；语言地图构建、PF/MCL、prompt/global初始化尚未端到端完成。没有将代码导入、权重下载和八叉树内核当成定位结果。",
        "## 实测时间与资源",
        "四篇总体墙钟ETA尚不可辨识：SG2Loc/OMCL核心定位未执行且数据权限/空间未解决；最乐观、正常、保守的四篇完成日期均为未知。内核小测试不能充当PF/MCL计时。不会为得到有限日期缩小实验范围。",
        f"ACE可执行队列实测投影：`{json.dumps(eta,ensure_ascii=False)}`",
        "GPU任务由原SQLite调度器串行；下载并发2、CPU任务并行。磁盘保留30GiB，不删除旧实验来凑空间。CPU映射特征缓存临时位于工作分区，训练成功后仅清理该任务自己的临时缓存。",
        "25/50/75/100%仅表示时间进度。用户现已要求停止巡检，因此Codex heartbeat以及本地5分钟/4小时报表timer均已停用；实验调度器和独立worker继续运行。历史timer测试保留。此前四篇总时长尚未形成有效实测基准，因此当前不会生成伪造25%或100%科研完成报告。时间引擎用隔离fixture测试边界和只触发一次；实际timer触发证据另存。原每四小时timer现按用户最新指令停用。",
        "当前实际通过v3标定的场景仍集中在训练划分；先完成的4021张RGB做过DCT感知哈希诊断，但该次快照没有验证/测试图像，因此实际跨划分比较对数为0，状态not_evaluable_single_split，不能将0候选解释为去重通过。全体清洗资格完成后另有全量感知相似性审计，不按定位成绩静默删除测试帧。",
        "存储限制已按用户授权解除：清理旧Diffusion/Freeloc未被当前FYP引用的派生数据64364883968字节（59.94GiB），当前Hypersim下载已自动继续。原始数据、旧模型、结果与生成代码保留。清洗规模仍以已验收数量为准，不因释放磁盘就声称达到40-80场景/8000-15000帧目标。",
        "## 原始证据、配置和失败索引",
        "本报告JSON包含数据库采样、每项实际配置、复现等级、输入条件、状态、attempt与路径；CSV将训练/评价和辅助审计分开。原始worker日志、source archive、provenance、receipt、checkpoint保存在现有持久目录。",
        "复核命令：`./experiment_automation/fypctl status`；`./experiment_automation/fypctl logs TASK_ID`；`./experiment_automation/fypctl report`。新增任务通过`fypctl sync`登记，不重启健康服务。",
        "所有未通过验收的任务索引：",
        "|任务|状态|尝试次数|角色|",
        "|---|---|---:|---|",
    ]
    for t in tasks:
        if t["status"] in ("FAILED", "BLOCKED", "RETRYING"):
            lines.append(f"|{t['task_id']}|{t['status']}|{t['attempt']}|{role(t)}|")
    md = "\n\n".join(lines) + "\n"
    md = re.sub(r"(\|[^\n]*\|)\n\n(?=\|)", r"\1\n", md)
    return data, md


def write_pdf(md, path):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, LongTable, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors

    pdfmetrics.registerFont(
        TTFont(
            "FYPChinese", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        )
    )
    pdfmetrics.registerFont(
        TTFont("FYPLatin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    normal = ParagraphStyle(
        "Normal",
        fontName="FYPLatin",
        fontSize=9,
        leading=15,
        wordWrap="CJK",
        spaceAfter=7,
    )
    heading = ParagraphStyle(
        "Heading", parent=normal, fontSize=14, leading=21, spaceBefore=10
    )
    cell = ParagraphStyle("Cell", parent=normal, fontSize=7, leading=11, spaceAfter=1)

    def markup(line):
        text = html.escape(line)
        return re.sub(
            r"([\u2e80-\u9fff\uff00-\uffef]+)",
            r'<font name="FYPChinese">\1</font>',
            text,
        )

    story = []
    rows = []

    def flush():
        if not rows:
            return
        parsed = [
            [c.strip() for c in row.strip("|").split("|")]
            for row in rows
            if not set(row) <= set("|-: ")
        ]
        parsed = [r for r in parsed if not all(set(c) <= set("-: ") for c in r)]
        n = len(parsed[0])
        assert all(len(r) == n for r in parsed)
        widths = {
            4: [265, 80, 65, 109],
            5: [155, 95, 95, 95, 79],
            6: [166, 85, 40, 80, 74, 74],
        }.get(n, [519 / n] * n)
        table = LongTable(
            [[Paragraph(markup(c), cell) for c in row] for row in parsed],
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef2")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c2c8cd")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)
        rows.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            rows.append(line)
            continue
        flush()
        if not line:
            continue
        ishead = line.startswith("#")
        line = line.lstrip("# ").replace("`", "")
        if line.startswith("ACE可执行队列实测投影："):
            line = "ACE可执行队列的三个具体日期、逐头耗时和步长测量见附属 reproduction_eta.md 与 JSON。四篇总ETA仍未知。"
        story.append(Paragraph(markup(line), heading if ishead else normal))
    flush()

    def footer(c, doc):
        c.setFont("FYPLatin", 8)
        c.drawString(
            38, 22, "FYP / Interim audit - four full reproductions remain incomplete"
        )
        c.drawRightString(555, 22, str(doc.page))

    tmp = path.with_suffix(".pdf.tmp")
    SimpleDocTemplate(
        str(tmp),
        pagesize=(595, 842),
        leftMargin=38,
        rightMargin=38,
        topMargin=32,
        bottomMargin=38,
        title="FYP Original Paper Reproduction - Interim Audit",
        author="FYP research workspace",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    tmp.replace(path)


def report_html(md):
    parts = []
    rows = []

    def flush():
        if rows:
            parts.append("<table>")
            for i, row in enumerate(rows):
                if set(row) <= set("|-: "):
                    continue
                tag = "th" if i == 0 else "td"
                cells = row.strip("|").split("|")
                parts.append(
                    "<tr>"
                    + "".join(
                        "<" + tag + ">" + html.escape(c.strip()) + "</" + tag + ">"
                        for c in cells
                    )
                    + "</tr>"
                )
            parts.append("</table>")
            rows.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            rows.append(line)
            continue
        flush()
        if not line.strip():
            continue
        if line.startswith("#"):
            n = min(6, len(line) - len(line.lstrip("#")))
            parts.append(f"<h{n}>" + html.escape(line.lstrip("# ")) + f"</h{n}>")
        else:
            parts.append("<p>" + html.escape(line) + "</p>")
    flush()
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FYP 原论文复现审计</title><style>body{max-width:1120px;margin:2rem auto;padding:0 1rem;font:15px/1.7 sans-serif;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ccc;padding:5px;text-align:left}th{background:#eee}h2{margin-top:2em}</style>'
        + "\n".join(parts)
        + "</html>"
    )


def main(pdf=False):
    import fcntl

    lock_path = ROOT / "state/reproduction_progress.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    data, md = build()
    from export_evidence import main as export_evidence

    export_evidence()
    from data_cleaning_index import build as cleaning_index

    cleaning, cleaning_csv = cleaning_index(read_tasks())
    atomic(OUTPUTS / "dataset_cleaning_diagnostics.json", cleaning)
    atomic(OUTPUTS / "dataset_cleaning_scenes.csv", cleaning_csv)

    ROOT.joinpath("reports").mkdir(exist_ok=True)
    eta = [
        "# 实测复现 ETA",
        f"采样：{data['generated_cst']}",
        "四篇总体：未知，无法给出有证据的最乐观/正常/保守日期。原因：MAC原始多序列获取未完成，SG2Loc/OMCL核心方法未在合法完整场景上实测。",
        "## 可执行 ACE 7Scenes 原始8M队列（局部范围）",
        "下列不是四篇总完成日期；假设GPU不插入其他工作，其他场景I/O与CPU评价会变化。",
        "```json",
        json.dumps(data["ace_queue_eta"], ensure_ascii=False, indent=2),
        "```",
        "数据处理：Hypersim受实际网络/磁盘限制，51场景全体尚未验收，无有证据的全量完成日期。12Scenes三次TLS失败；清理后/data约90GiB可用；Replica压缩33.86GB并需额外解压空间，完整获取前须重新核算峰值空间。",
        "旧1274帧MAC-VO Paper实时前端、Fast全缓存不能互推实时性能；新序列需实际重新测量。SG2Loc PF/Refinement、OMCL语言地图/MCL计算时间未知，环境探针不替代核心计时。",
        "## 25/50/75/100时间基准",
        "总墙钟基准BLOCKED_UNMEASURED。只有四条核心链有代表性计时后，才接受scope=four_paper_total且representative_core_measurements_complete=true的基准。按用户最新要求巡检及本地报表timer已停用，实验worker继续运行；禁止用局部ACE估计冒充全体。",
    ]
    atomic(ROOT / "reports/reproduction_eta.md", "\n\n".join(eta) + "\n")
    stem = "FYP_Original_Paper_Reproduction_Report"
    for out in (ROOT / "reports", OUTPUTS):
        atomic(out / (stem + ".md"), md)
        atomic(out / (stem + ".json"), data)
        document = report_html(md)
        atomic(out / (stem + ".html"), document)
        atomic(out / "reproduction_eta.md", "\n\n".join(eta) + "\n")
    # Machine-readable config index is exactly the actual DB specs, without network environment file.
    import csv, io

    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "task_id",
            "paper",
            "level",
            "role",
            "status",
            "attempt",
            "online_input_condition",
        ]
    )
    for t in data["tasks"]:
        writer.writerow(
            [
                t["task_id"],
                t["paper_id"],
                t["reproduction_level"],
                t["role"],
                t["status"],
                t["attempt"],
                t["online_input_condition"],
            ]
        )
    atomic(OUTPUTS / "paper_task_index.csv", stream.getvalue())
    import subprocess

    subprocess.run(
        [
            str(BASE / "envs/macvo/bin/python"),
            str(ROOT / "reproduction/plot_training.py"),
        ],
        check=True,
        timeout=60,
    )
    state_path = ROOT / "state/reproduction_progress.json"
    state = load(state_path) or {"emitted": []}
    baseline = load(ROOT / "state/reproduction_total_time_baseline.json")
    for p in due_milestones(state, baseline, time.time()):
        text = f"# {p}% 总预计时间检查点\n\n时间进度不代表科研完成率。\n\n" + md
        atomic(ROOT / "reports" / f"reproduction_progress_{p}.md", text)
        atomic(OUTPUTS / f"reproduction_progress_{p}.md", text)
        state["emitted"].append(p)
    state.update(
        last_actual_run_cst=stamp(),
        pid=__import__("os").getpid(),
        baseline_status=(
            "ready"
            if baseline and baseline.get("representative_core_measurements_complete")
            else "blocked_unmeasured"
        ),
        report_sha256=hashlib.sha256(md.encode()).hexdigest(),
    )
    atomic(state_path, state)
    if pdf:
        write_pdf(md, OUTPUTS / (stem + ".pdf"))
        pdf_path = OUTPUTS / (stem + ".pdf")
        import shutil

        shutil.copyfile(pdf_path, ROOT / "reports" / (stem + ".pdf"))
        atomic(
            OUTPUTS / "pdf_snapshot_receipt.json",
            dict(
                generated_cst=stamp(),
                pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                source_markdown_sha256=hashlib.sha256(md.encode()).hexdigest(),
                scope="PDF is a time-stamped snapshot; live MD/HTML/JSON refresh independently",
            ),
        )
    print(
        json.dumps(
            {k: data[k] for k in ["generated_cst", "total_counts", "hypersim_v3"]}
            | {"timer_state": state},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    main(parser.parse_args().pdf)
