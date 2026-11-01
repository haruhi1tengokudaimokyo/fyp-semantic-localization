# FYP 持久科研实验系统

## 2026-11-01 本轮阻塞修复（覆盖下方较早的状态描述）

`reproduction/repair_manifest.py` 将真实数据获取、原方法执行、独立指标验收加入原 SQLite DAG；不是另起调度器。SG2Loc/OMCL 原来空命令的核心 ID 已替换为定位执行入口，只有实际完整轨迹和独立指标通过才接受。已完成任务的 command/config/inputs/seed/env/validation/场景绑定保持不变；失败尝试保留。

- OMCL 已在 ReplicaCAD `v3_sc0_staging_00` 真正完成 600 帧全局初始化定位，独立指标接受；原 LSeg、Kaolin 语言八叉树、原观测似然、预测/重采样/位姿估计均实际运行。使用明确标记的噪声 GT 相对运动条件，不是无 GT 在线实验；两种布局来自共享 FRL 公寓，不能称独立建筑或原论文 Replica/NARADIO 表格复现。
- SG2Loc 使用官方 ScanNet SceneGraphLoc checkpoint、真实 DINOv2-g 多视图对象/查询特征、实例三角网格及 BVH；语义关系未标注的边按官方 `none` 处理，未制造特征。查询 GT 与在线入口隔离，实际 RGB-D 里程计及声明的水平安装提供运动/重力，是替代数据条件。粒子滤波与 RoMaV2 refinement 的最终完成状态以 SQLite 和独立 metrics.json 为准，特征提取成功不等于定位成功。
- MAC-VO 新增完整 TartanAir hospital/P000（563 帧）和 abandonedfactory/P000（2176 帧）。原 Paper 配置保留 ICP 优化器；完整源序列执行及原官方指标/独立 SE3 复算另有验收。`original_icp_weight_ablation` 是固定学习对应点、固定静态初值的原 ICP 后端受控消融，不能称完整端到端 Paper 消融；较早 DISP replay 结果保留且不混称。
- Hypersim 冻结 51 候选已全部终态，39 场景/7901 帧接受、12 场景隔离；不足冻结 40/8000 训练门槛，qualification 保留 FAILED，不放宽阈值，也不额外启动语义训练。
- MINS 原 office bag 本身只有 12–53 秒，并非读取截断；新 café 源 bag 逐消息覆盖验收后运行原 stereo_imu，保留严重漂移等真实失败。不得称完整长序列或纯轮速成功。
- EuRoC 新 ETH DOI 元数据可访问，但实际公开内容连接仍有 TLS/429；12Scenes 公开源仍受 TLS/502 阻塞。下载失败不阻止其他任务。

独立原 Replica 场景链有真实获取/转换入口，OMCL 首条完整定位验收后即可获取数据，不依赖 SG2Loc 成功；尚未验证的数据适配不称已就绪。源文件按场景选择性获取，保留 30GiB 磁盘余量，不删除历史接受产物。所有新实验均保留替代数据/运动条件/场景绑定，绝不将不同场景地图或轨迹混用。先尝试首条 SG2Loc 核心定位，再执行备好的 MAC-VO/OMCL/ACE，最后其他 SG2Loc 消融；SG2Loc 按完整窗口保存与来源/配置/代码绑定的原子检查点及精确 NumPy RNG 状态，恢复不重复已完成窗口。

无人值守仍由 enabled 的 `fyp-experiments.service` 和独立 `fyp-exp-*` worker 运行。Codex heartbeat 和本轮前已关闭的两个实验进度/报告 timer 保持关闭；不恢复巡检。最终交接以本轮阻塞修复报告的时间戳和真实 worker 快照为准，旧 ETA 不再代表新增 GPU 队列。

## 2026-11-01 20:30 用户交接指令（覆盖此前巡检要求）

用户要求：清理旧Diffusion项目无用数据，启动能执行的实验后停止Codex及巡检，电脑自己完成队列。已通过应用工具将fyp heartbeat设为PAUSED；`fyp-reproduction-progress.timer`和`fyp-experiment-report.timer`均disable/stop。不重新启用，除非用户随后要求。`fyp-experiments.service`与独立worker不停止，仍有限重试、验收与资源调度，关闭Codex界面不影响它们。

已清理旧v5/v6的native_capture和procthor派生文件，64364883968字节（59.94GiB）。检查当前DAG无引用、当前FYP无指向这些目录的符号链接、无进程打开其文件。保留旧源场景/几何、manifests、生成代码和全部旧模型/实验结果/日志；`/data/fyp_3d`科学数据未删除。删除记录见artifacts/storage_cleanup_20261101.json；磁盘释放后Hypersim原队列已自动恢复。

20:17系统重启后后台自行恢复。chess/fire的8M训练完成；heads正在第二次尝试。旧评测失败是`test_ace.py`导入了dsacstar命名空间而非编译扩展，非模型/数据故障。`.eval.v2`固定子进程PYTHONPATH到真实扩展目录，保持上游脚本和权重不变；真实CPU图像端到端回归与完整GPU评价分别验收。不重训chess/fire。

最终预计时间、当前实际状态和剩余外部阻塞，以outputs/FYP_Automation/无人值守交接_20261101.md为准。不要把ACE可执行队列ETA说成四篇论文全部复现的完成日期。


## 当前优先级：2026-11-01 用户四篇原论文续接指令

最新人工任务以 MAC-VO、ACE、SG2Loc、OMCL 的原论文复现和严谨数据处理为主。本节覆盖旧 heartbeat 中新增自研融合、Dashboard、动态工程的优先级；历史成果全部保留，暂停继续扩展无关算法。健康控制器和 worker 不重启，新增任务用 `./experiment_automation/fypctl sync`（登记并验收可复用产物）接入现有 SQLite。不得通过修改已完成任务的科学配置，将旧结果重新标成新实验；`core.register` 对 command/config/inputs/seed/env/validation/scene 变化要求新任务版本。

四条核心方法同时在 DAG 中。`paper_id/reproduction_level/online_input_condition` 进入真实 SQLite；完成支持性审计或环境任务不代表完成论文。当前：

- MAC-VO：保留1274帧作者提供的 P001_select 完整演示，但未证明是完整父基准或论文对应行。官方指标已用独立 evo1.29 CPU环境重算，原GPU轨迹不重算。旧evo1.38兼容失败及SO3数值精度失败保留。
- ACE：所有7Scenes原始训练/测试划分、外参、逐RGB/pose核查通过；旧官方模型和1M训练结果保留。原8M缓存/16epochs/batch5120仅改变缓存存储位置。chess已真实完成2688.504秒训练（CUDA峰值分配2344.51MiB），其余场景继续；准确率以完整上游测试通过为准。12Scenes四源分组ZIP各3次TLS失败，HTTP502和额外TLS诊断保留，不能解除数据准备阻塞。
- SG2Loc：固定官方源码，隔离环境及实际Numba CUDA内核/RayCasting导入已通过；3RScan/ScanNet原数据授权与原始Replica空间仍缺，PF/Refinement/5-10-25序列成绩没有产生。原代码GT相对运动/重力条件不等于FYP GT-free。
- OMCL：固定源码，Kaolin CUDA通过；原始3.095GB LSeg checkpoint严格加载、真实480裁剪Batch1前向通过，0.61743秒、CUDA峰值2930.33MiB。没有实际语言地图/MCL成绩。论文Replica使用NARADIO，不能冒用LSeg称原表复现。论文2048射线与发布配置num_samples1024/每mask采样差异仍待核查。

### 数据清洗版本与现有证据

Hypersim冻结候选51来源族/10493帧，仅从发布有效帧元数据选取。来源族ID不是已证实物理建筑，未声称免费获得原始Mesh。`acquire_hypersim_scene*.py` 的三个版本与独立验收目录均保留，不覆盖已接受版本：

1. v1固定5mm门槛、假设标准相机，真实多场景验收暴露错误，保留失败。
2. v2使用一般相机RQ、float64换算、解析float16量化界限，避免把远处储存误差误判；但仍假定源相机轴至多均匀缩放。旧成功和失败保留，禁止作为新训练的资格。
3. v3对每帧完整 `source_orientation @ M_cam_from_uv @ half_pixel_UV` 做RQ，生成逐帧K与SO3相机位姿；不依赖图像、GT世界点或测试成绩拟合。非均匀源基不直接当旋转。每台相机首个计划帧的全部有效像素，必须落在位置/光心距离的IEEE半ULP合成界限内（另加1微米运算余量），零超界才通过；其余帧核查元数据/单位/有效深度/语义/形状/RGB解码。

完整分解已在真实 ai_043_009 两台相机1572864像素通过；原 cam_01 基矩阵非刚体，但其解析针孔表示合法。ai_027_003实际有24254个像素射线不一致、光心距离一致，集中门窗柜体，原因未定，保留隔离，不放松阈值。每次诊断、原始CRC/SHA receipt和版本产物独立保留。

v3使用与候选冻结计划相同的帧身份。51个候选全部得到终态审计后，`qualify_hypersim_cleaned.py` 记录CRC/标定/来源失败导致的整场景隔离，并冻结清洗决策；只在真实保留至少40场景/8000帧、原划分族不交叉、解码RGB跨划分无重叠后接受训练资格。没有读取定位成绩或静默换入测试场景。确定性科学失败exit65不盲目重试；网络失败按有限策略恢复。若空间不足则保留原数据等待资源，不清理历史实验。

Replica17个官方分卷总压缩33859047808字节，还需解压空间；当前 `/data` 仅约31GiB可用且保留30GiB，不满足完整18场景获取。授权脚本和额外存储路径的问题已向用户提出，未获得回复前不伪造授权或数据。

### ETA与独立进度报告

`reports/reproduction_eta.md` 对实际8M队列给出步长、buffer与三个具体日期；四篇总ETA因SG2Loc/OMCL核心尚未实测且数据受阻，明确未知。不得用ACE局部时间冒充四篇总墙钟基准。

新增 `fyp-reproduction-progress.timer` 每5分钟独立运行 `reproduction/progress.py`，实际自动首触发16:23:29 CST成功；原4小时timer不变。引擎已测试25/50/75/100边界、离线补报和不重复触发；全体时间基准尚未知，因此不会伪造百分比报告。未来只有四条核心都有代表性实测且明确scope=four_paper_total时才接受基准。证据：`artifacts/reproduction_timer_test.json`。进度引擎使用互斥文件锁，保护手动报告与timer并发。

实时MD/HTML/JSON与任务CSV持续更新；PDF是明确带时间戳的审计快照，不声称四篇全部完成。真实8M训练曲线与CSV从日志生成。用户交付文件统一在主线程 `outputs/FYP_Automation/`；完整配置和全部失败任务可审计，不用任务总完成率代替论文验收。


本系统已经实际启动。主控制器读取 `manifests/tasks.json`，以 SQLite WAL 保存任务 DAG；每个实验运行在独立 systemd user worker 中。所有结果经文件存在性、SHA256、长度及指定指标验收后才能 COMPLETED。复用的初步结果带 `formal=false`，不会成为扩大实验的完成证据。

## 查看与运行

在本仓库运行：

```bash
./experiment_automation/fypctl status
./experiment_automation/fypctl report
./experiment_automation/fypctl logs macvo.demo.paper
systemctl --user status fyp-experiments.service
systemctl --user list-timers fyp-experiment-report.timer
```

Dashboard：http://127.0.0.1:8766 ，读取真实数据库，每十秒刷新。静态副本 `dashboards/index.html` 在每次报告时更新。报告目录 `reports/`，始终维护 `LATEST.md` 与 `LATEST.json`。

## 持久运行与四小时报告

- `fyp-experiments.service`：控制器，异常自动重启；退出控制器不会停止独立 worker。
- `fyp-exp-<task ID 哈希>-a<attempt>.service`：真实实验进程；每次尝试各自保存日志。
- `fyp-experiment-dashboard.service`：本地只读 Dashboard。
- `fyp-experiment-report.timer`：独立于 Codex 的 systemd 日历 timer，Persistent=true。
- 启动基准：2026-10-31 23:04:26.668（Asia/Shanghai）。首次周期报告已于 2026-11-01 03:04:26.804 实际生成；随后每日 03:04、07:04、11:04、15:04、19:04、23:04。
- 已执行 `loginctl enable-linger csy` 并验证 Linger=yes；登出后 user manager 保持运行。未使用 sudo 或修改系统级服务。
- 关机、断电、休眠期间不能计算；开机后 enabled user service 与 Persistent timer 恢复，未正常结束的 worker 由数据库状态和 unit 状态识别，有限重试。
- Codex 四小时 heartbeat 是补充的代码开发和诊断机制，计算任务和本地报告均不依赖 Codex 界面。

真实控制器退出/重启测试、SIGKILL worker 检查点恢复、失败后重试、四小时边界测试和实际报告证据保存在 `artifacts/controller_restart_test.json`、`artifacts/infrastructure_selftest.json`；测试使用隔离状态库，未制造科学数据。

## 数据、协议和恢复

`state/ logs/ checkpoints/ reports/ dashboards/ artifacts/` 是指向 `/data/fyp_3d/automation/` 的 Git 忽略链接。大型数据、模型、原始日志都在 `/data/fyp_3d`，原始数据不删除、不提交到 Git。SQLite 为 `state/experiments.sqlite`。每个任务保存数据集、模型场景、seed、command、config、input_hash、code_commit、输出、失败证据、attempt、时间和资源采样。新 worker 保存内容寻址的源代码归档，包括尚未提交但实际执行的代码；`artifacts/*.provenance.json` 记录归档哈希。该归档与本机持久数据在同一磁盘，尚不能视为异地备份。每次报告还通过 SQLite backup API 原子保存一致状态副本到 checkpoints/state_progress_*.sqlite；按任务/数据集/指标拆分的 CSV 和多种子描述统计见 artifacts/aggregate/，不混合不同任务排行榜。

每个 GPU 同时只调度一个大型任务。下载最多并发两个，CPU 任务与 GPU 按槽位及可用内存调度，保留 2 GiB 内存及 30 GiB 磁盘安全余量。不静默降低精度、改变输入规模或重调测试参数。失败通常最多两次，网络下载三次；诊断后增补一次修复尝试必须有事件记录。上游问题导致的失败结果保留，其他任务继续。

MACVO 的密集前端以图像、配置、权重及标定为键原子缓存；恢复重放优化器前缀，而不是伪造中间状态。SegFormer 保存 epoch 边界模型、优化器及 Python/NumPy/Torch/CUDA RNG 状态，严格检查协议后恢复；渲染保存每个已完成场景的 manifest，恢复不重采已完成场景。下载采用临时文件和哈希验收。

systemd 不继承交互式 shell 的代理。必要的网络变量保存到 `state/network_environment.json`（0600），只供 worker 注入，不展示或提交该文件；更换代理后仅更新该受保护文件。

## 研究分支和当前限制

A：完整 7-Scenes 官方划分，官方场景权重及场景专用本机训练，三个固定 seed；不得跨房间直接混用 head。1M feature buffer 为公开记录的本机配置，不称为论文 8M buffer 的完全等同复现。

B：下载的 1,274 帧连续双目全序列，Paper/Fast/普通 Stereo VO 与固定匹配的单位协方差消融；原始 lost mask 单独保存。SE(3) 对齐 ATE 与地图绝对误差分开。其他完整轨迹尚待合法输入和标定获取。

C：七条完整 OpenLORIS office 序列的 Stereo+IMU 可独立运行；原论文 III.A 已明确 office 的 Segway 底盘 odometry 使用编码器+底盘 IMU 的厂商融合，不能称为纯 Wheel。原始编码器不可用，纯 Wheel 组合保持阻塞；另排队七条序列的 vendor-base-odometry+T265-IMU（含/不含 stereo）、2D/3D 有条件约束及固定 8–12s 相机断流实验，名称和输出均明确不是纯 Wheel。已有平地非完整约束结果属于条件实验，不能声称通用 AI 增益。初始化失败作为实际实验结局记录，不填造零误差。

D：扩展到 24 个 HSSD 合成房屋组，同房屋前缀或字节相同建筑网格归到同一组，再冻结 train/val/test。这些不是经验证的真实物理建筑身份。每场景目标 150 个空间位置、4 朝向；低纹理/无效深度/重复视图拒收，正式 train 少于 3,000 帧则失败。只训练原生可验证的七类物体语义，未知为 IGNORE255，不虚构墙/窗/柱。三种策略 full/frozen/scratch、三个 seed、40 epochs，选择仅用 val。decomposed 子目录遗漏曾导致渲染失败，已保留日志并增加显式闭包审查。

E/F：独立测试房屋各自构图，参考/查询位置分离；SIFT 与实际官方 hloc SuperPoint/LightGlue、CLIP Top-K=4/8/16、语义硬过滤及概率软加权，固定共同查询，逐空间位置配对 bootstrap。CLIP 检索+PnP 不称为 OMCL 完整复现。

G：真实 Habitat 渲染的 600 帧连续双目、384 个离线参考视图、标定、只读三维先验地图、独立评估 GT；实际场景 ACE 训练和 MACVO 已完成联合端到端执行。状态传播、绝对校正与 Reliability Gate 真实运行。低初始化覆盖和错误绝对观测必须照实报告，不能仅报成功帧误差。

I 的共同场景控制器故障重放已增加，明确不是物理绑架；错误接受计数的首版已失效并保留，由 v2 替代。J 已加入 TUM Freiburg3 sitting_xyz/sitting_static/walking_xyz/walking_static 四条完整序列，真实坐姿小幅运动与行走人员分别标记，只评估 RGB-D VO，未声称具备可靠三维先验地图。相机内参、深度注册和 1/5000 深度比例来自官方格式说明：https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats 。

H、I 的物理恢复、J 的其余跨数据集、OMCL、多轨迹及 Wheel 仍有真实依赖/实现/数据合法性阻塞，数据库逐项保存原因；不能把队列启动或初步扰动实验当成整项研究完成。健康后台任务保持运行。后续 heartbeat 应优先解除真实阻塞、完成测试，并对稳定实现提交 Git。

## 原始参考

- ACE：https://github.com/nianticlabs/ace
- MAC-VO：https://github.com/MAC-VO/MAC-VO ；https://mac-vo.github.io/
- MINS：https://github.com/rpng/MINS
- HSSD：https://huggingface.co/datasets/hssd/hssd-hab
- SegFormer：https://github.com/NVlabs/SegFormer
- hloc：https://github.com/cvg/Hierarchical-Localization
- LightGlue：https://github.com/cvg/LightGlue
- CLIP：https://github.com/openai/CLIP
- OMCL：https://github.com/AIS-Bonn/omcl

各实际源码 checkout、权重哈希、兼容修改和已有结果见 `/data/fyp_3d/sources`、`/data/fyp_3d/models` 与验收产物。本文件仅解释实际协议，不替代状态数据库和原始日志。

运行时间必须区分缓存：完整 Paper 前端 1271 misses / 3 hits，Fast 为 1274 hits。Fast 暖缓存的逐帧时间不能与 Paper 大多实时推理的时间直接当作速度提升；Dashboard/CSV 已附缓存上下文及独立 worker 总耗时。

2026-11-01 初次四小时后的实际验收：HSSD 24 合成房屋组已完成 11,446 张有效渲染图像（train 7,494 / val 1,995 / test 1,957，房屋组 16/4/4），正式首个 full seed1305 训练在 GPU 继续。七条 office 的来源审查结果包含实际 /odom 帧名、消息计数、连接元数据及原论文 SHA256；没有 raw encoder topic。已验收的全量正式结果、配对统计与失败查询会在每次报告自动重建到 artifacts/aggregate。首版恢复计数的错误已标无效并保留，v2 通过，不能把旧 FAILED 清零隐藏。

正式轨迹图由每次报告从已验收的真实数组自动更新到 artifacts/aggregate/formal_trajectories.png，保留原始丢帧标记和全局位姿缺口。新 worker 另有 48 小时单任务时长与 2 小时日志静默 watchdog；触发后保留证据和检查点，按原有限尝试策略处理。

## 2026-11-01 07:00 巡检新增（当前正式结果与待验收分支分开）

第二份周期报告实际在 07:04:26 采样、07:04:28 写出，systemd report service 成功。full_seed1305 完成40epochs，按验证集选 epoch1；1957张独立测试图像的已标注像素 mIoU=0.6321619，door无已验证测试像素，不能声称完整七类建筑分割。其余种子和策略继续运行。

E/F 已完成四个独立测试合成房屋的771个共同查询、20个预设方法配置。固定TopK8：SuperPoint/LightGlue Recall@25cm/10°=530/771，语义硬过滤532/771，软加权533/771，CLIP检索+几何529/771。不要因先前76查询试验而断言CLIP普遍提升。新增 `relocalization_building_statistics.json/csv` 对所有预设方法和三个阈值作四房屋配对bootstrap（10000次、seed1305），少量合成房屋的区间不代表真实建筑总体。

H 分支已加入真实命令链，等待健康的 full_seed1306 训练释放GPU：在既有384个离线参考位姿处重新渲染真实深度/原生语义，核对RGB对齐和原始地图SHA；在原600帧查询上计算真实SuperPoint/LightGlue几何、语义硬/软及CLIP观测；最后运行九个连续融合变体，包含可关闭模块、多候选选择和重投影残差权重。每个观测先接受时间一致性审查，不把同一RGB产生的相关观测当独立传感器，不把重投影权重称校准位姿协方差。共同房屋105515211属于语义训练房屋，所以这部分明确是同建筑系统集成，不声称跨建筑泛化。H全部产物验收前不能标正式完成。

I 分支新增真实渲染相机绑架：新seed1315、600帧、在第300帧实际将Habitat相机移动至少2m，重新渲染左右相机，复用相同只读地图和原场景ACE权重；不修改图像或伪造IMU/Wheel。排队计算实际MACVO运动、ACE校正、Gate和三个预设阈值下连续三帧恢复时间。单次路线不存密集前端缓存，以保留30GiB磁盘余量。它与旧位姿/缓存扰动重放分开；当前端到端尚未执行，不声称恢复成功。

新增在线融合已通过4个测试，包括既有600帧ACE/MACVO缓存回放的一致性（原float32与新float64传播容差1e-6）、世界坐标变换不变性和禁止凭空生成初始地图位姿。候选池改动另用3个真实SIFT成功查询回归，原选择位姿未改变。完整19个核心/真实预处理测试通过（16 pytest + 3 ACE环境unittest），这些测试不能代替尚未运行的H/I渲染端到端验收。

Heads原始本机训练的真实seed是2089（training_resources.json），导入任务曾误用通用默认1305，现已记录元数据更正事件，保留历史receipt/input hash。增排2090/2091训练和完整评价，复用已完成的2089，不重复训练。旧G protocol的physical_timestamps标志曾依据时间文件存在性推断；其标定/acceptance始终明确是仿真。新MACVO运行已把simulation time与physical timestamps分开；旧轨迹和帧索引指标不修改或重跑。

H/I 的融合计时只测状态更新，神经观测与运动来自实际计算过的缓存。上游 TwoFrame_PGO 只回写当前优化帧，通常在下一次主循环接收该结果；本轮回放使用完成的双帧估计，不声称在1660 SUPER上以仿真10Hz实时同时运行全部模型。绑架后的秒数明确为simulation-clock latency，不冒称真实联合处理wall-clock latency。连续状态有效率另排除LOST状态，不能把无运动输入时保持的旧位姿算可靠跟踪。
