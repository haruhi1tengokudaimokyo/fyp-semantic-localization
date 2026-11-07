# 三算法共同地图与 Gazebo 机器人定位实验

本阶段沿用 `research/fusion-stage`。原 ACE 七场景、MAC-VO Benchmark 和 SPVLoc 100/20/20 权重、数据及结果均保留。新增工作使用 ProcTHOR `scene_03000` 的建筑结构，研究局部运动、绝对纠偏与全局恢复如何配合。机场只是应用示例。

## 算法职责与真实输入

| 模块 | 实际实现 | 在线输入与输出 |
|---|---|---|
| 局部底座 | 安装的 ROS `robot_localization/ekf_node` | 左右轮关节速度、IMU gyro z，以及差速车有限不确定度的 `vy=0` 约束；输出连续 `odom→base_link` |
| ACE | 官方预训练 encoder、本 Gazebo 地图新训练 head、官方 DSAC* | 左目 RGB、实际内参；输出相机到固定地图的位姿与内点数 |
| SPVLoc | 已验收自训 checkpoint，官方 viewport matching、Top-3、Top-1 refinement | 左目中心裁剪、正确内参、实际建筑结构渲染；输出多个独立候选与原始匹配分数 |
| MAC-VO | 官方 Paper 配置、释放的权重、TartanMotionNet、协方差 ICP | 完整同步双目轨迹和标定；输出连续视觉相对运动，失效保留为失效 |
| 调度与门控 | 独立局部/全局状态、历史时间关联、模式确认 | 正常时 ACE；初始化/恢复时 ACE 与 SPVLoc 轮流请求。单执行 worker，最多一个可替换待处理图像，合计最高 1Hz |

在线主链不读模拟器世界位姿、GPS、LiDAR 或查询标签。世界位姿仅供独立参考集监督和离线评价。`map→odom` 的计算关联到图像拍摄时刻的历史里程计，并使用实际相机外参；不修改局部 EKF 的连续轨迹，不平均多个房间候选，不把匹配分数伪装成正确概率。

当前实际定位验收发现 SPVLoc 在这个新渲染域的561帧中，Top-1、refined Top-1和Top-3 oracle均为0/561（1m/30°）。因此demo默认真实调用并记录SPVLoc，但作为未合格候选审计，不允许它改动map→odom。显式 `--allow-spvloc-corrections` 只用于保留原多候选门控的实验消融；不能把它当作已经验证可靠的部署配置。离线A–E仍完整保留SPVLoc结果，观察其真实影响。

MAC-VO 单独完整运行同一条双目轨迹，并进入 A–E 融合回放。它是否能满足实时要求由实测耗时决定；当前 ROS 高频链不把慢速 MAC-VO 当作必需底座。MINS 尚未接入，这里没有将旧短序列测试升级成部署验收。

## 场景和数据边界

使用官方 ProcTHOR 房屋几何的 7 个房间、22 面去重且非空墙、6 个门洞和4个窗洞，保留原坐标与开口位置。未还原原家具资产，墙地外观为程序纹理，Gazebo 当前是无顶场景。SPVLoc 参考必须反映实际无顶几何，而不能继续使用自动补出的天花板；第一次含天花板的运行保留为地图不一致诊断，不用于正式共同地图对照。

这是旧验证建筑上的系统开发与渲染域适配，**不是未见建筑泛化结果，也不是原论文 Benchmark**。更换相机高度、外观和渲染器都有域差异，不能把原 AI2-THOR 上的精度直接搬来。

双目为 640×480、10cm 基线、约0.60m相机高度。左右图像要求纳秒时间戳相等；参考采集须在实测位姿稳定150ms后取图，图像和GT差不超过15ms。首条封存轨迹为59.9秒、561对图像、7.13m；捕获过程中存在丢采样，最大间隔0.3秒，未插值或假称600帧无缺口。

新 ACE 训练使用192个独立参考视点、1M特征缓存、官方16轮训练、batch5120；encoder冻结、head实际更新。这是资源受限的新地图工程训练，不称原8M论文配置复现。查询与参考文件和图像哈希无交集，查询位姿不用于训练。

## 一条命令运行真正的仿真系统

使用本轮已经验收并保持不变的世界文件、ACE head和SPVLoc参考。不要重新生成或覆盖正在绑定的world；改变文件时应重新检查地图绑定。

```bash
cd /home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash

/usr/bin/python3 -m research3d.robot_sim.demo \
  --world-manifest tmp/robot_sim_procthor_stereo/map_manifest.json \
  --ace-binding ../../outputs/RobotSim_20261107/ace_train/model_binding.json \
  --spvloc-references ../../outputs/RobotSim_20261107/spvloc_openroof \
  --mapping-dataset ../../outputs/RobotSim_20261107/stereo_dataset \
  --output ../../outputs/RobotSim_20261107/my_new_demo \
  --domain-id 72 --seconds 60
```

输出目录必须不存在。此命令启动本次专用Gazebo、桥接、EKF及真实ACE/SPVLoc模型，执行冻结的轮速路线，记录同步双目、原始轮编码器/陀螺仪、独立GT、调用日志和拒绝原因，最后独立计算指标并绘图。结束时只清理自己创建的进程组，不停止其它实验，不创建巡检或常驻timer。

增加 `--events configs/robot_sim/kidnap_events.json` 可在预先指定的时刻把机器人搬到另一处，用于实际失跟恢复实验。这个事件只进入模拟器与离线事件记录，不作为定位模型输入。恢复成功需连续至少3个评价帧满足1m/30°，并跨越至少0.2秒；未恢复按删失失败保留。

增加 `--camera-blackout 15 35` 可同时屏蔽两路相机20秒，图像仍保留原时间戳，采集器与定位器看到相同的黑屏。这是明确声明的图像接口故障，不称物理遮挡或自然行人实验。

更底层的传感器启动、坐标、话题和采集验收见 [SENSORS_zh.md](../research3d/robot_sim/SENSORS_zh.md)。实际执行日志、结果和限制见 [完整本轮报告](../results/robot_sim_20261107/REPORT_zh.md)。

## 可重复评价

`run_macvo.py` 不裁短轨迹、不用后处理插值掩盖失败；`run_ace_queries.py` 在封存的同一批PNG上推理；`run_spvloc.py` 使用相同查询ID与地图绑定。模型推理完成之后，`replay_models.py` 才独立加载GT评价。

A–E 分别为 MAC-VO局部、MAC-VO＋ACE、MAC-VO＋SPVLoc、三方法固定调用、三方法多候选恢复。A没有绝对地图原点，不能偷偷用GT初始化后与全局定位比较。局部误差另报告明确的初始原点或刚体对齐口径；全局误差不做GT对齐。结果缺失计入失败分母。

回放同时输出理想离线准确率和按真实耗时模拟的单资源FIFO积压。后者仍不包含完整模型切换/I/O费用，**不是在线实时性证明**。真正在线的延迟、覆盖和拒绝统计来自 `demo.py` 的实际ROS进程日志。

`replay_sensors.py` 可把新demo保存的原始IMU、轮编码器、双目和时钟按1倍速发布到独立ROS domain，用于同一轨迹的EKF与算法消融。它不发布GT、真实轨迹或控制命令。此前首条561帧序列没有保存完整原始轮/IMU消息，不能将它伪称完整ROS传感器回放包；新demo修正了这个记录缺口。

## 验证后的下一步

只有对照结果显示实质改善，才能声称融合有效。少量建筑内的运行成功还不能证明跨建筑、光照、动态遮挡或真实机器人可靠性。后续需要冻结多场景路线与干扰协议、验证SPVLoc对ACE失败帧的真实互补性、核查错误纠偏与恢复失败，再做相机/IMU/轮速标定和真实数据适配。ROS接口可复用，但真实地图采集、时钟、计算延迟和现场安全验收仍须重新完成。

实际同传感器消融可用 `python3 -m research3d.robot_sim.replay_ablation --dataset <capture> --output <fresh_output> --domain-id 74`，默认EKF-only；`--mode ace --ace-binding <binding>`启用真实ACE；`--mode ace_spv_experimental`另需SPV参考和模型，显式检验原门控。原始流、模型ready、实际RPC和输出时序均被保留。

本轮已记录正常和黑屏＋搬动两条完整60秒路线、五组同传感器对照。故障ACE-only回放5.934秒确认恢复，物理在线运行（SPV审计开启）8.734秒；每组只有一次运行，不代表恢复成功率已充分验证。报告同时保留SPV零候选召回、正常路线有害纠偏及DDS采样缺口。
