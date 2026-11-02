# SPVLoc 数据与运行交接（2026-11-02）

最新用户指令：今晚只完成 100/20/20 栋语义数据准备，保留现有训练自然结束，明天再启动全量训练；VLM 暂停。本目录沿用官方方法与真实数据适配，不新增实验管理系统。

## 已运行与待完成

- `fyp-spvloc-16-pipeline.service`：正在运行原版 SPVLoc 的小规模 25 epoch 训练。实际冻结 9/3/3 栋；第10栋训练候选后来通过数据检查，也不会加入当前模型或覆写其冻结清单。
- `fyp-semantic-data-140.service`：已启用、独立于 Codex 的一次性数据服务。先等待上述完整训练/评价自然退出及 CUDA 空闲，再采集、验收、补齐目标；没有全量训练启动代码。
- 现有16栋均160视角、20全景，结构几何验收通过。3个官方划分各抽1栋，原版全景投影与真实透视RGB的20视角交叉检查均通过；真实原版随机全景训练样本也已成功生成，无训练/优化器调用。
- 140栋最终清单尚未生成。只有严格凑齐100/20/20，才写 `DATA_READY`；不足时报告阻塞，不降低标准。

## 入口

`prepare_large_split.py` 从官方已下载10K房屋文件冻结150/30/30候选：前100/20/20为目标，另50%为有限备用。按源文件顺序、至少2房间/1门/1窗选择，检查跨划分几何SHA唯一性。保留全部旧16栋的房屋文件与活动训练清单，不下载旧已清理数据。

`prepare_semantic_overnight.py` 逐栋执行真实 `capture_procthor.py`、独立 `qualify_procthor.py` 和原版投影的 `qualify_panorama.py`。失败建筑单独记录、使用同一源划分下一候选；没有按定位成绩筛选。采集完整性收据与输出清单原子提交，超时终止该子进程组，保留其余场景。重启复用完整收据，不重复采集已完成建筑。

`/data/fyp_3d/datasets/procthor_spvloc/building_candidates_100_20_20.json` 保存冻结候选；最终 `building_split_manifest_100_20_20_accepted.json` 保存实际合格身份。训练/验证/测试严格取官方ProcTHOR划分，Structured3D风格编号只是接口编号。

最终审计位于 `/data/fyp_3d/runs/semantic_localization_20261102/data_100_20_20/`。每栋 `accepted_files.sha256` 覆盖房屋、结构、标定、RGB、深度、实例/类别编码、姿态和验收文件。`SEMANTIC_CLASS_DISTRIBUTION.csv` 保留真实原始类别计数，不将未知实例编成墙/门/窗。

## 固定验收协议

真实AI2-THOR RGB、float32米制深度、原始实例/类别图与精确字典、CV相机位姿和内参齐全；256×256图像全部解码、深度/标定全部检查。Unity `(x,y,z)` 转规范世界 `(x,z,y)`，地图与姿态以毫米匹配官方原接口。不存在的empty隔墙不入图，按源开敞连通关系合并房间，实际门窗洞保留。

每5个查询一次独立CPU原版ray renderer检查，对原生可见墙/地板/天花板掩码腐蚀3像素。固定深度误差p95≤0.15m、语义一致率中位数≥0.95；当前实际p95约1–3cm。家具遮挡导致不足100个结构像素的探针标为不适用，但每栋仍要求至少10个独立有效视角和10,000结构采样像素。缺有效覆盖的整栋拒收。

全景与同光心真实透视RGB用原版 `pano2persp_from_rotation`、实测相机旋转交叉检查，不拟合变换。固定图像内圈RGB绝对误差中位均值≤25/255，每栋至少10个探针，原检查值约2–3/255。该检查在新数据阶段也必须通过。

严格保留100GiB磁盘底线、新数据增长不超过60GiB，逐栋检查；不清理ACE/MAC-VO任何产物。不持久保留额外归档副本。VLM无活动worker、不再调用API，密钥不入仓库。

## 原论文协议与差异

官方源码：SPVLoc commit `754861115e38bf6f9b6b6b642db095f8e1ff6958`；CPU redner fork commit `84f00efda06a16f7872f3ca202805cd31f3f8f81`。原论文：[SPVLoc ECCV 2024](https://fraunhoferhhi.github.io/spvloc/)，代码：[fraunhoferhhi/spvloc](https://github.com/fraunhoferhhi/spvloc)。官方模型SHA256 `074ec5d9d2e45b74358b5cd0d0fc139eef1d2499c20b98daa0f0a80f07682e67`，全部状态严格加载。

当前训练原网络56,462,316个可训练参数、全部原损失/Adam、25 epochs、LR0.00025、12/18里程碑、4近1远参考、查询256²、参考128×256。GT只作训练监督及离线评价。Batch8显存不足后唯一硬件回退为FP32 Batch4×累积10；这与原生Batch40的BN/动态过滤并不等价。

原版用Structured3D全景随机透视裁剪；当前已启动小规模模型使用真实固定透视视角模式。新数据额外提供已验证的RGB全景，明天全量配置应显式使用官方随机45–135°FoV/yaw/pitch/roll机制。官方投影函数注明 `CORRECT_ROLL=False` 才对应原论文精确roll行为，而发布配置为True；启动正式配置前须明确记录选择，不静默改已运行模型。

ProcTHOR替代数据不是原Structured3D benchmark，只提供full家具渲染；不冒充原文empty/full双条件。原生语义目标为结构六类（unknown/floor/ceiling/wall/door/window），家具另有真实类别/实例记录，不把家具误标结构。ReplicaMultiAgent因实例映射与Mesh坐标未获可信验证，仍不能混入监督数据。

依赖环境、源码与大型数据留在 `/data/fyp_3d`；本仓库提供代码、配置和轻量真实证据。关闭Codex不影响systemd用户服务，已确认 `Linger=yes`；不恢复自动巡检。
