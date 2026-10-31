# 三维先验地图室内定位：P0/P1 实测技术报告

日期：2026-10-31。FYP：Exploring Localization Techniques for Indoor Robotics。

## 本轮结论

本轮完成了真实三维数据获取、完整性检查、实际 Habitat 渲染、独立 Mesh 深度检验、训练序列专用三维地图，以及 **ACE 官方真实重定位和本机完整场景头训练**。P1 的最小三维定位 Baseline 已通过实测验收。P0 的数据审计完成，但 HSSD 原生墙/窗/柱监督未满足后续训练门槛，缺失已测量而非伪造。P2–P6 尚未完成，本轮没有用接口定义冒充这些算法。

原仓库的二维代码、历史实验与 Git 历史完整保留。升级位于 `research3d/`，Git 分支 `research/3d-p0-p1`，旧版本检查点为 `266f04136b4a13d3edd5067b811ec3699241d305`。最终提交哈希见交付版本文件；不覆盖旧研究结论。

## 1. 真实数据和授权验收

| 数据 | 实际取得内容 | 下载字节数 | 完整性与用途 |
|---|---|---:|---|
| ReplicaCAD interactive | 官方共享资产602文件；使用 apt_0 / empty_stage | 157,450,746 | 201个LFS SHA256和401个Git blob摘要匹配，大小匹配；CC-BY-4.0 |
| HSSD-Habitat | 3个场景、117个对象模板及门窗依赖、语义/官方划分，共455文件 | 79,111,079 | 322个LFS SHA256和133个Git blob摘要匹配，无坏文件；CC-BY-NC-4.0 |
| Microsoft 7-Scenes Heads | 完整场景，训练seq-02与测试seq-01，各1000帧 | 956,332,240 | 外层ZIP CRC通过，6000个RGB/Depth/Pose文件CRC通过；非商业研究许可 |
| OpenLORIS | 仅远端头/索引和tar成员清单 | 约1MB+元数据 | market1-1确认21个连接；未下载完整bag，未跑融合 |

7-Scenes 原始嵌套 ZIP 的两个 `Thumbs.db` CRC 错误被明确排除；科研文件没有跳过 CRC。记录见 `acceptance/heads_dataset.json`。压缩包SHA256为 `402f8760ba150100bda5360241b605383dc5f949c055863301bde1b45f6a7a5d`。没有声称原包所有附属文件均完好。

`dataset_registry.json` 保存官方来源、实际下载URL、授权说明、字节数、文件校验、传感器和任务适配。每个文件的本地SHA256与来源路径可追溯；HSSD/ReplicaCAD 的固定版本见各 acquisition 清单。VIODE、TUM RGB-D、InCrowd-VI 未下载，不宣称本机可用。没有绕过 gated 访问或虚构 IMU/Wheel。

数据存放 `/data/fyp_3d/datasets`；原数据目录保持原状。开始时 `/data` 约209GiB可用，没有批量下载全部大型数据集。

## 2. Habitat 实际渲染、像素与坐标

ReplicaCAD apt_0 24视角、empty_stage 16视角；HSSD三个完整场景各24视角，另一个 stage-only 检验12视角。生成真实 RGB、米制轴向 Depth、原生 Semantic ID、实际传感器矩阵和点云。ReplicaCAD 必须使用 headless+Bullet 才能加载铰接家具；未修复的 nobullet 首次运行日志保留，最终 apt_0 没有使用缺物体版本。启用 Bullet 后不进行任何物理步进，场景保持固定。

统一约定：`T_world_camera` 将 OpenCV 相机点变到其地图世界坐标，平移单位米。Habitat 相机从实际 sensor node 的 OpenGL 变换右乘 `diag(1,-1,-1,1)`；不是从机器人机体位置猜测相机高度。320×240、90°水平FOV对应 f=160；整数像素中心约定下 cx=159.5、cy=119.5。

独立检查直接对**原始GLB三角网格**求射线交点，和渲染深度比较，未用同一深度生成网格自证：

| 数据/范围 | 原始网格射线数 | 深度差中位(mm) | P95(mm) | 最大(mm) | <1cm |
|---|---:|---:|---:|---:|---:|
| ReplicaCAD empty_stage / 16视角 | 1456 | 0.0131 | 0.1124 | 0.7218 | 100% |
| HSSD stage-only / 12视角 | 1180 | 0.0101 | 0.0612 | 0.2079 | 100% |

它验证了相机、stage网格、深度在同一坐标系；不冒称逐一核查所有家具表面、材质或标签。ReplicaCAD 官方配置还列出一些仓库未包含的可选 variation navmesh，日志保留；本轮使用的 apt_0/empty_stage 有效，不把未测场景算作验收通过。

## 3. HSSD 语义训练适配的实际发现

| 场景ID | 全图unknown像素 | 有效深度中unknown | 门像素 | 墙像素 | 窗像素 |
|---|---:|---:|---:|---:|---:|
| 105515211_173104185 | 82.69% | 82.20% | 0 | 0 | 0 |
| 107734110_175999914 | 81.52% | 81.52% | 43161 | 0 | 0 |
| 107734119_175999932 | 75.07% | 75.07% | 0 | 0 | 0 |

柱类同样未出现。unknown=原生语义ID 0，不能直接当作墙。刚体对象配置有类别 semantic_id，但若干 `objects/openings` 门窗配置没有该字段，stage也没有完整建筑类别覆盖。原生输出中的相同类别ID**不区分重复实例**；不能把“有 semantic sensor”写成“有完整实例真值”。报告同时保留有效深度条件下比例，避免把空背景全部解释为标注缺失。

三个场景均正常渲染。官方 `scene_splits.yaml` 将105515211_173104185放入val，另外两个放入train。已读取不同场景ID及官方划分，但尚未独立证明物理建筑身份互斥，不能宣称三栋独立建筑泛化。当前不启动以墙/门/窗/柱为完整类别集合的 SegFormer 正式训练；后续可先采用有可靠像素的物体类别，或补充可验证的建筑结构标注，并另设建筑级测试集。

## 4. 7-Scenes 三维地图与信息隔离

遵循 [ACE 官方适配器](https://github.com/nianticlabs/ace/blob/main/datasets/setup_7scenes.py) 的RGB标定约定：RGB f=525、主点(320,240)；原始Depth f=585、毫米、65535无效。原始 Kinect Depth 与 RGB **没有直接像素对齐**；RGB位姿按 `T_world_rgb = T_world_depth @ inv(T_rgb_depth)` 转换。微软说明缺少逐设备精确标定；外参与默认内参是ACE公开约定，不是新测量的精确标定。

显式三维地图仅用训练seq-02：每10帧取一帧、像素步长4、1cm体素，得到 **161,390个三维点**。先在原始depth坐标反投影并使用训练depth位姿变换，再投影到RGB采色；未把未注册Depth直接按RGB内参反投影。保存 NPZ、PLY、100参考帧的RGB路径/标定/位姿目录及带SHA256的只读地图清单。实际世界/相机往返最大误差 9.49e-16 m，像素代数重投影最大 5.24e-13 px。此项只是代数检验，不替代现实传感器标定证明。

ACE 的场景头是隐式场景坐标地图。显式点云用于同一世界坐标的审计和展示，本轮没有假称把点云额外融合进神经网络。训练只读训练图像/位姿；在线适配器的输入仅含 RGB、相机内参和观测标识，无query真值字段。离线评估在 `predict()` 返回后才读GT。主动禁止 `numpy.loadtxt` 时推理仍成功；1000帧适配器与官方入口的平移误差最大差 3.88e-08 m、旋转误差最大差 2.28e-06°。

没有在线SLAM、LiDAR或GPS输入，没有使用测试位姿建立地图，也没有轨迹对齐来降低误差。这里未知的是**已知Heads场景内的初始相机位姿**；没有实现未知建筑检索。

## 5. 真实定位结果

相同官方测试seq-01全部1000帧、stride=1、相同默认64假设/10px DSAC*参数。主表均用统一适配器，计时包含图像读取、灰度归一化、CUDA前向、CPU几何求解和重投影诊断，排除模型载入，1次仅网络warmup；GPU训练没有与评测并行。

| 模型/协议 | 帧数 | 平移中位(cm) | 旋转中位(°) | Recall@5cm,5° | Recall@2cm,2° | 平均延迟(ms) | P95(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 官方预训练场景头 | 1000 | 0.929 | 0.621 | 99.8% | 92.0% | 136.10 | 144.81 |
| 本机训练场景头：1M/16 epochs | 1000 | 1.062 | 0.721 | 98.4% | 88.5% | 139.11 | 151.90 |

原始官方 `test_ace.py` 独立运行也得到99.8% @5cm/5°，平均126.32ms；它不含DataLoader读取等时间，不能把与适配器的计时差解释为算法回归。官方聚合脚本对偶数帧中位数的索引约定与 NumPy 略异，完整逐帧结果已保留。

官方权重的两个5cm失败帧是632和633：分别5.93cm/2.25°、5.37cm/1.99°，不能从汇总删掉。图册展示三张最小和三张最大平移误差图；本机模型也独立展示尾部案例。5cm失败率是阈值失败，不等于跟踪中断；另列几何求解失败/低内点率。

这些是单场景绝对重定位结果，不是跨建筑泛化、VO/VIO的ATE/RPE、真实人群鲁棒性或机器人端到端实时性能。尚无时间一致性门控，研究估计不自动写入跟踪状态（accepted=False）。

## 6. 本机训练与资源

使用官方预训练编码器及官方 ACE trainer/loss，随机种子2089，100万特征缓存、batch5120、16epochs、2个DataLoader worker。它是充分运行的场景头训练，不是单图loss smoke test；也**不是自训练语义网络**。缓存小于论文的8M，相应训练更新预算也更小，不能冒称原论文完整训练配方。

本机实测：326.65秒；峰值 allocated 3443.36MiB、reserved 3462.00MiB；进程maxRSS 1625.66MiB。官方权重推理峰值 allocated 87.64MiB，进程maxRSS 900.82MiB。显存数字为本进程PyTorch统计，不含桌面占用；RAM未汇总子worker峰值。系统仍有桌面和其他后台进程，未声称独占硬件基准。

GTX1660 SUPER 6GB、torch2.5.1+cu121。驱动显示CUDA13.0是驱动能力，不是本环境CUDA runtime。第一次训练因上游访问已移除的 `_step_count` 失败；采用真实 optimizer post-step hook 兼容，AMP overflow跳步不会错误推进学习率。失败日志、成功完整日志、最终checkpoint、资源记录与每100个batch的真实训练曲线均保留。曲线是训练日志，不是验证集曲线；只评估最终checkpoint，未按测试成绩选择checkpoint。

## 7. 接口、测试与可复现性

`knownmap/contracts.py` 提供实际使用的 SensorObservation、PoseHypothesis、MapCorrespondence、LocalizationState、LocalizationDiagnostics、只读地图，以及后续 MotionEstimate/SemanticObservation/PoseConstraint 数据契约。SE(3)、协方差PSD、维度、单调时间戳与地图摘要检查可执行；未为后续网络或融合器添加空的算法实现。

位姿协方差当前不可用，显式为None；DSAC内点数仅是几何质量统计，不是成功概率。7-Scenes 无本工程可验证的IMU/Wheel硬件时间戳，stamp_ns=None，不由帧号捏造。每张地图都有独立map_id，不混合不同数据集的世界坐标。

测试涵盖真实数据划分、官方预处理逐值一致、独立Mesh深度校验、地图不可变与SHA拒绝、GT-free输入、SE(3)/协方差拒绝，以及AMP正常/overflow两种优化器更新计数。测试与主动GT读保护的日志随仓库交付。精确命令见 `docs/RUNBOOK_zh.md`，上游版本、补丁、依赖快照和论文边界见 `docs/REPRODUCTION_AUDIT_zh.md`。

## 8. 后续优先任务与当前边界

1. 优先选择 OpenLORIS office1-3：远端tar清单实测单bag为517,491,443字节，优于下载整个9.27GB office归档；下载后再核查时间戳、相机/IMU标定、wheel/odom语义与GT。官方说明office真值来自OptiTrack，其他场景来自离线LiDAR SLAM；GT仅用于评估。当前只检查了market话题，不把其标定套到office。
2. 为MAC-VO建立独立CUDA≥12.4兼容环境，先验证官方Fast Mode与整流双目。OpenLORIS的T265是原始鱼眼，不能直接当针孔双目输入；本轮没有MAC-VO成绩，也没有拼接MINS后端。
3. 解决HSSD建筑类别标注缺失与建筑互斥身份，再进行真正的轻量语义网络训练及定位消融。VLM另行评价；不使用oracle掩码冒充预测。
4. 加入第二个完整真实场景和独立建筑协议，再研究重定位门控、kidnapped recovery与动态鲁棒性。当前结果只证明Heads基线可运行。

论文及工程依据为 [ACE](https://arxiv.org/abs/2305.14059)、[Habitat-Sim](https://github.com/facebookresearch/habitat-sim)、[ReplicaCAD](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset)、[HSSD](https://huggingface.co/datasets/hssd/hssd-hab)、[Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)。所有本轮精度数字来自本机输出，不引用论文成绩填表。
