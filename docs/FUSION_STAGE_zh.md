# 融合阶段：同一已知地图下的可靠定位与失跟恢复

本分支 `research/fusion-stage` 接续 ACE、MAC-VO、SPVLoc 的已完成实验。当前交付是可执行的共同数据采集、协议检查、因果回放与评价、等预算训练入口，**不是三方法融合效果已经得到验证**。原实验代码、模型、数据与服务保持原状。没有启动新训练、租卡、恢复 VLM 或自动巡检。

## 对方案的判断与本轮范围

建议采用这条路线。它将课题落到错误纠偏、全程成功覆盖和失跟恢复，而不是只比较有效帧上的平均误差。三个主要判断必须由开发集验证：

1. ACE 失败时 SPVLoc 是否确实提供正确候选？若 `P(非ACE正确 且 SPV正确)` 接近零，不能靠增加融合复杂度保证收益。
2. SPVLoc Top-3 是否比 Top-1 提供更多正确位置？如果是，时序候选选择值得做；如果不是，优先改善数据与域适配。
3. 接受策略是否在相同错误率下提升覆盖，或相同覆盖下减少坏纠偏？需要同时保留无输出和未恢复事件。

论文中的网络保留各自职责，MAC-VO 冻结、ACE 为目标地图训练 head、SPVLoc 做多源适配。新增融合策略属于本项目方法，不能称为三篇论文原本就提供的统一算法。当前回放后端是候选门控与 `map←odom` 更新，**没有冒称实现 GTSAM 优化器**。固定时窗图优化、在线事件调度及对应公平预算对照明确登记为后续项；先拿到共同前端输出和校准证据再推进。

## 已固定的协议

主配置：[experiment.json](../configs/fusion/experiment.json)。建筑谱系：[building_lineage.json](../configs/fusion/building_lineage.json)。

- 开发建筑：原 ProcTHOR 验证集 `val/0`、`val/1`、`val/2`，适配 ID `3000/3001/3002`。它们用于系统开发，不作为未参与系统调参的最终测试。
- 新系统测试：按官方 test 文件顺序选 `test/39,41,45,50,51`；排除全部旧候选、已验收建筑及重复几何。当前只封存身份与几何 SHA，未采集、未跑分。
- 同一基础建筑、楼层/布局派生、全景裁剪保留共同 lineage。原 100/20/20 SPVLoc 实验不重写。
- 目标地图 reference 允许训练该地图 ACE head；query 不能混进 reference 或 SPVLoc 训练。最终测试 query 不能用于门限、调用频率或 checkpoint 选择。
- 原生采集 640×480、垂直 FoV 60°、12 cm 水平基线、相机高 1.5 m、仿真采样 5 Hz、最大平移 0.25 m/s、最大旋转 20°/s。
- 左右图来自同一个最终 AI2-THOR Event、固定平行光轴；ACE/SPVLoc 使用同一左图。5 Hz 是声明的仿真采集时钟，不能当成渲染或在线算法达到 5 Hz。
- 保留 `/data` 至少 100 GiB；新阶段数据峰值预算 16 GiB。实际采集前和每帧写入前检查；不会删除旧项目腾空间。
- 默认没有已知 GT 全局初始位姿。MAC-VO 单独只提供局部坐标结果，不能以恒等变换冒充地图初始化。

配置中的参考/查询帧数是采集规模目标；实际连续路径长度由可达导航区域决定。路径过短、跨 split 视点重合或数据 QA 失败必须修复/隔离，不能靠重复帧凑够目标。截断的 `--max-frames-per-split` 只能获得 `PILOT_CAPTURE_QA_PASSED`，不标正式完整序列。完整采集通过仅为 `CAPTURE_QA_PASSED`；两者均需另做前端接口验收，不能直接解锁正式训练。

## 代码和执行入口

以下命令从仓库根目录执行，文件路径可按新机器修改；配置中保留本机真实路径便于复现。

### 1. 配置、数据、建筑隔离

```bash
python3 -m research3d.fusion.protocol validate
python3 -m research3d.fusion.protocol preflight
```

`configs/fusion/building_lineage.json` 已封存。`freeze-lineage --output 新路径` 可核验生成规则，但不得在看到测试结果后重新挑建筑。

### 2. 真双目采集与独立磁盘验收

```bash
/data/fyp_3d/envs/semantic_loc/bin/python -m research3d.fusion.capture \
  --protocol configs/fusion/experiment.json --scenes 3000 3001 3002 \
  --output /data/fyp_3d/datasets/fusion_procthor_development \
  --software-render
```

采集器读取本机已下载的房屋，不下载新资源包、不读取测试分数。选择可达网格的连续路径，同步输出图像、标定、建筑结构与独立 offline 位姿/深度。真双目 QA 包括 SO(3)、外参、图像解码、时序、路径连续性、reference/query 分离和左右深度重投影；大规模采集仍需完整序列与跨模型输入验收。命令的完整 CLI 及独立验收参数可用 `--help` 查看。

已实际完成 val/0 的 3 reference＋3 query 帧软件渲染探针：每段0.10m/0.4s，双目深度重投影最坏 P95 为2.36mm，结构墙/地/顶重投影 P95 不超过0.494mm。采集约177秒，不含渲染器启动；`Low`画质是明确的pilot覆盖设置，完整采集默认High。它验证实际像素、尺度和结构接口，**不是完整定位轨迹、不是三方法性能结果**。另一次5＋5帧试采超时，partial保留未验收，详见运行收据。

`annotation_3d.json` 为官方 SPVLoc/S3D 兼容的毫米结构；相机 T_map_camera 平移与深度为米。模型适配时必须显式转换1000倍，不能仅因map_id相同就跳过单位转换。

在线数据与离线 GT 分目录；为训练 ACE，reference 位姿是合法离线监督。`render_completed_time_s` 是采集程序的实际墙钟信息，不能直接当作模型的 `completed_time_s`。模型适配器必须真实测量推理完成时刻，并在统一时钟上登记。

### 3. 三分支统一观测

在线绝对观测 JSONL：

```json
{
  "schema_version": 1,
  "source": "spvloc",
  "source_id": "procthor",
  "building_id": "val/0",
  "map_id": "固定地图版本及哈希标识",
  "camera_id": "left",
  "query_id": "每条路线全局唯一图像ID",
  "capture_time_s": 0.0,
  "completed_time_s": 0.4,
  "status": "VALID",
  "candidates": [
    {"rank": 1, "T_map_camera": "实际4x4相机到地图矩阵，平移米", "quality": {"viewport_score": 0.5}}
  ],
  "provenance": {"image_sha256": "实际64位哈希", "model_sha256": "实际64位哈希", "calibration_sha256": "实际64位哈希"}
}
```

这是字段说明，非可当成真实输入的示例预测。ACE 的质量字段为整数 `pnp_inliers`，不能把 SPVLoc 分数改名为内点数或正确概率。SPVLoc 候选严格按原始排名保存 1–3 个；没有第三候选时不得复制凑数。GT 单独保存 `query_id,timestamp_s,T_map_camera` 和同一 source/building/map/camera 标识。

MAC-VO 行包含同一图像身份和 provenance、`source=macvo`、capture/completed time、`valid`、`segment_id`、`T_odom_camera`。丢帧时 pose 必须为 null；断链后的局部坐标段需显式标记。不得插值掩盖丢帧。现有 7-Scenes head 和外部轨迹不能直接改成 ProcTHOR map_id。

**本轮尚未运行新共同场景三分支前端。** 它们需要完成新轨迹到官方 loader 的接口验收、对应 ACE head 训练和模型绑定。统一 schema 的单测不能代替这些步骤，主预检明确保留 NOT_YET_ACCEPTED。

### 4. 互补性诊断与因果回放

```bash
python3 -m research3d.fusion.diagnostics \
  --observations observations.jsonl --ground-truth ground_truth_eval_only.jsonl \
  --output complementarity.json

python3 -m research3d.fusion.replay \
  --motion macvo_motion.jsonl --observations observations.jsonl \
  --config configs/fusion/experiment.json --variant E_three_recovery \
  --output results/new_fusion_replay

python3 -m research3d.fusion.diagnostics \
  --observations observations.jsonl --ground-truth ground_truth_eval_only.jsonl \
  --outputs results/new_fusion_replay/outputs.jsonl --events recovery_events.jsonl \
  --output results/new_fusion_replay/offline_metrics.json
```

`replay` 没有 GT 参数；输入包含 GT 或正确标签会拒绝。每张图以 MAC-VO 实际完成时间作为回放输出时间，只使用此前完成、且采集时刻不晚于该输出对应图像的观测。晚到结果通过其原始图像的连续 odom 链传播，不能穿过中间失效段，已输出的过去结果不会重写。

正常状态使用分别定义的 ACE/SPVLoc 门限；恢复状态维护多个地图候选，不对不同房间取平均。TTL 内不同 query 的支持计数不等于严格连续帧；同一图像两分支不会多加一次支持。多个成熟模式仍有歧义时拒绝输出。一段连贯但错误的候选仍可能通过，这正是需要实际坏纠偏统计的原因，不能把确认帧数当成正确保证。

门限是 **UNVALIDATED 开发初值**，不是论文最优参数或已校准概率。需要开发集校准，冻结配置 SHA 后才可最终测试。当前回放仅验证候选策略，不证明端到端实时速度；从缓存任意免费调用分支也不能证明事件触发更省算力。

## 主对照与消融

| 配置 ID | 分支与策略 | 当前可执行层级 |
|---|---|---|
| A_macvo_local | MAC-VO 局部里程计 | 保留局部输出；全局无初始化，不参与地图绝对覆盖排名 |
| B_macvo_ace | MAC-VO＋ACE，Top-1 基本门控 | 有合格共同观测后可回放 |
| C_macvo_spvloc | MAC-VO＋SPVLoc，Top-1 基本门控 | 同上 |
| D_three_fixed | 三分支、固定基本门控 | 同上 |
| E_three_recovery | 三分支、Top-3、多帧、独立恢复状态 | 同上；研究策略，非已证明改进 |
| E_top1 | E 仅保留 Top-1 | 单因素消融 |
| E_single_frame | E 仅需一个不同 query 支持 | 单因素消融 |
| E_no_independent_recovery | E 取消冲突触发的独立恢复 | 单因素消融 |

固定频率/事件调用需真正因果调度与相同调用数或 GPU 时间预算，本轮未冒充实现。共享已知初始位姿的跟踪条件需单独协议，不能偷偷给 A 注入测试 GT。参考采集量 100%/50%/25% 也需先生成固定空间覆盖子集并分别训练 ACE head。

“去掉 SPVLoc 分支”只证明模块贡献；几何/语义因果对照仍需匹配训练预算的几何分支，不能把语义图清零称为纯几何公平实验。

## 评价与决定下一步

评价分母是预注册全部 GT 查询时刻。错误或缺失预测保留失败；全局误差不事后对齐；局部 VO ATE/RPE另报。记录错误候选接受、正确→错误纠偏、恢复成功和未恢复事件、最长失效、误差 P50/P95、输出年龄。一次路线一个文件，不把多个从0开始的时钟拼在一起；汇总先路线后建筑。

`recovery_events.jsonl` 记录预注册事件的 identity、`event_id,kind,start_time_s,end_time_s`。恢复判定使用预先指定的连续成功时刻，确认延迟包含模型完成时间；观察期结束仍未恢复的事件保留右删失。不要用本系统误差挑选“更好恢复”的事件。

如果互补性不足，应优先完成 ZInD 域适配和数据/候选质量修复。如果存在明显 ACE失败/SPV成功，则推进多候选、可靠权重与成熟图优化。只有在共同新测试轨迹上超过基线，并给出真实时序运行结果后，才可写“融合有效”“可在线运行”。

## 多源训练和资源

详细入口：[FUSION_TRAINING_zh.md](FUSION_TRAINING_zh.md)。P/Z/P+Z 同初始化、8575 次 optimizer update、FP32 B4×10、共同冻结 BN running statistics、逐更新 LR；相同尝试查询预算，实际有效样本另报。BN、采样、LR调度属于明确的适配，与旧25轮结果不能无说明地当作同条件重跑。

ZInD 目前只有原始下载，未完成官方转换及投影、结构、loader 验收。三组共同验证集含 P 和 Z，因此正式多源训练入口会拒绝启动，不以假 DATA_READY 解锁。官方转换命令已核对，可先对一栋做合法小规模验证；整套转换需磁盘预算，不能另复制完整 RGB 数据。

本地可以先完成3栋开发集和冻结前端诊断；是否租用24GB GPU需先测真实batch、100–300次更新和完整验证周期。配置不包含租卡、付费服务或云端上传授权。现有训练的49.57小时只能给计算量参考，当前未实际开始的新训练没有可靠完成时间。

## 来源与实现边界

- ACE 官方：<https://github.com/nianticlabs/ace>。目标地图专属 head，reference 位姿属于离线准备。
- SPVLoc 官方：<https://github.com/fraunhoferhhi/spvloc>。复用网络、损失、ZInD转换和原语义地图表示。
- MAC-VO 官方：<https://github.com/MAC-VO/MAC-VO>。冻结核心方法；三维点协方差不是现成位姿协方差。
- AI2-THOR 官方：<https://ai2thor.allenai.org/>；本机固定 Unity commit 的 AddThirdPartyCamera/UpdateThirdPartyCamera、同Event多相机结果。
- GTSAM：<https://borglab.github.io/gtsam/isam2/>。后续图优化候选，本分支没有把规则回放声称为 iSAM2。
- ROS 坐标约定：<https://www.ros.org/reps/rep-0105.html>。map 可纠正全局误差，odom 保持局部连续。

测试命令：

```bash
python3 -m pytest tests/test_fusion_replay.py tests/test_fusion_diagnostics.py \
  tests/test_fusion_training.py research3d/tests/test_fusion_capture.py -q
```

测试中的解析轨迹仅证明软件契约，不能用其指标作定位实验结果。真实采集/官方接口检查证据见 `results/fusion_stage_20261107/`。
