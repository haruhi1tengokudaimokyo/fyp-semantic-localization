# Gazebo 已知地图定位：真实三模型与恢复实验

2026-11-07，分支 `research/fusion-stage`。本轮已完成实际模型运行、Gazebo 在线定位、故障恢复及同原始传感器消融。**已经跑通有实证支持的 EKF＋ACE 纠偏/恢复链；尚未证明 ACE＋MAC-VO＋SPVLoc 三方法融合更可靠。** SPVLoc 的真实失败及额外调用成本均保留，不通过换算法或筛选成功帧掩盖。

## 1. 实际做了什么

- 复用已有 ProcTHOR `scene_03000` 建筑：7房间、22面去重墙、6门洞、4窗洞；实际轮驱运动、同步双目、原始编码器和IMU。未复制原家具，使用程序纹理与无顶几何。
- 用192个严格同步、姿态稳定的参考视点，为该Gazebo地图真实训练新ACE head：官方encoder/trainer/DSAC*，1M特征缓存、16轮、batch5120、seed2089，345.36秒，峰值显存3443MiB。旧7-Scenes head没有用于新场景。
- 同一59.9秒、561双目帧的完整记录，分别运行官方ACE、MAC-VO Paper FP32及已自训SPVLoc。原始预测未插值；GT只在输出保存后评价。
- 原版SPVLoc语义/法向量参考按实际无顶世界重建：82个全局候选参考，0.60m相机高度、128×256全景；官方viewport matching、Top3和Top1 refinement。第一次含不存在天花板的诊断结果已隔离。
- 在线运行真实 `robot_localization` EKF、ACE和SPVLoc RPC、单worker事件调度、历史odom关联、时序确认及map→odom纠偏。SPVLoc未通过本域资格验收，默认真实推理但仅审计；显式实验模式保留原多候选门控。
- 记录两次60秒物理运行：正常570对图像；黑屏＋搬动533对图像。新记录均包含完整落盘的原始轮/IMU流，可在隔离ROS域按1倍速重放。

原有ACE/MAC-VO/SPVLoc数据、环境、权重与结果均保留；未启动VLM、云GPU、巡检或新自动化框架。MINS尚未接入；MAC-VO目前作为实测研究对照，没有被伪称ROS高频实时底座。

## 2. 同561帧，三种真实算法

| 方法 | 真实结果 | 耗时与口径 |
|---|---|---|
| ACE新场景head | 5cm/5°：235/561=41.89%；25cm/10°：446/561=79.50%；1m/30°：497/561=88.59%；平移中位5.93cm，RMSE1.920m | 算法均值171.85ms；固定地图原坐标，不拟合GT；64帧在1m/30°外 |
| MAC-VO Paper FP32 | 0算法lost；刚体对齐ATE0.12381m；仅首帧离线定原点ATE0.31728m；逐帧RPE RMSE2.855mm/0.05542° | 均值1.3308秒/帧、全程756.6秒、峰值2960MiB；无尺度修正；显然不能逐帧跟上10Hz |
| SPVLoc自训checkpoint | Top1、refined Top1、Top3 oracle在1m/30°均0/561；Top1平移中位7.974m | 中位338.24ms；真实模型、真实地图、真实错误；不能称语义定位成功 |

MAC-VO的8.477m是**相机中心**轨迹长度，底盘轨迹为7.131m；转弯时0.20m相机力臂造成差异。输入有采集缺口（最大0.3秒），与“MAC-VO算法lost=0”是不同概念。

SPVLoc在这批查询上没有任何可救回ACE失败的候选：ACE独对497，共同错64，SPV独对0。已排除显式天花板地图不一致并核查相机转换，但尚不能把失败唯一归因于域差异；低相机高度、程序外观、缺家具/顶面和原训练域差异均需进一步受控诊断。

## 3. A–E融合回放：保留负结果

| 理想可用时间口径 | 全局成功帧/561（1m/30°） | 结论 |
|---|---:|---|
| A：MAC-VO局部 | 不具备绝对地图原点 | 使用单独VO指标，不用GT给它在线定原点 |
| B：MAC-VO＋ACE | 559 / 561，99.64% | 中位平移5.54cm；465次接受无超过1m/30°的错误接受 |
| C：MAC-VO＋SPVLoc | 0 / 561 | 553有效但错误输出；229次错误接受 |
| D：三方法固定调用 | 0 / 561 | 冲突模式使全局初始化持续拒绝，不能算成功 |
| E：三方法多候选恢复 | 0 / 561 | 同样保持歧义/RECOVERING，不能宣称三模型增益 |

B的收益只证明本条轨迹上**离线运动约束＋ACE门控**有用。按实测推理耗时模拟单资源FIFO后，全部配置无法满足0.5秒时效限制；B的P95输出积压约747秒。这个FIFO是未包含模型切换和额外I/O的下界，仍不是实际在线系统测速。不能将99.64%当成本机实时性能。

独立脚本直接从5,610条输出和原始GT复算10组结果，没有复用原评分函数。详见 [融合独立核验](fusion_replay_report.json)。

## 4. 真正在线的Gazebo运行

主链：原始轮速＋IMU → EKF → 历史odom；ACE → map绝对观测；SPVLoc → 当前不合格候选审计；门控/状态机 → map→odom。在线节点不订阅世界GT。每个新模型真正ready后才启动路线，记录真实RPC耗时及capture/receive时间；推理最多单worker＋一个可替换pending图像。

| 物理运行 | 正常路线 | 黑屏＋搬动路线 |
|---|---:|---:|
| 查询帧 | 570 | 533 |
| 全局有效且满足1m/30° | 522/570，91.58% | 390/533，73.17% |
| 有效输出平移RMSE / 中位 | 0.11845m / 0.03401m | 0.03998m / 0.02518m |
| 首次全局输出 | 路线开始后5.534s | 路线开始后3.934s |
| 接受纠偏 | 28可评价且均在1m/30°内 | 18次中15可评价且正确；3次严格时间关联不足 |
| 误差增加>25cm或>10°的纠偏 | 1 | 0（可评价部分） |
| 真实ACE / SPV调用 | 38 / 3 | 24 / 5 |
| 完成60s采集的墙钟时间 | 61.23s | 61.43s |

正常运行那次有害纠偏将平移误差0.0617m增加至0.3251m，仍在1m成功阈值内。因此“没有超过1m的错误接受”不能写成“所有纠偏都改善”。短正常路线中，首帧GT定原点的局部EKF诊断RMSE比绝对纠偏链更小；绝对地图定位及失跟恢复才是本轮证明的能力，不声称每个时刻精度都改善。

故障协议预先固定：双目图像接口黑屏15–35秒，在路线31秒通过模拟器服务搬动机器人；事件不传给在线算法。实际GT跳跃约2.317m，首次采到跳跃后位姿为41.1s；正确输出49.634、49.734、49.834s连续确认，恢复时间**8.734s**。跳跃采样存在约0.1s量化不确定性。不是物理行人遮挡，也不是实际机器人搬运实验。

逐张解码与hash审计显示：声明故障时间窗内有183个查询，实际全黑双目为182对（原浮点边界使24.4s帧尚可见）。实际黑屏帧局部覆盖179/182=98.35%。原始记录保持冻结；后续代码已改整型纳秒判断，并对该真实边界添加回归测试。另修复故障节点收到SIGINT时重复shutdown的退出异常；数据写入已先完成，独立ROS清理回归退出0，不重跑或改写实验结果。

[在线独立审计](online_independent_audit.json)逐条验证全局位姿、恢复、接受结果，以及全部1,066张故障双目PNG。

## 5. 同原始传感器的真实1×回放消融

不是重复运行物理路线：以下每个normal配置使用完全相同570帧、25,263轮速和6,011 IMU记录；fault配置使用相同533帧、26,815轮速和5,988 IMU记录。地图、ACE head、门控参数保持不变，SPV资格开关明确记录。实际DDS收发差异和DSAC随机性保留；每个配置仅运行一次，尚无显著性结论。

| 数据 / 配置 | 全局成功覆盖1m/30° | 有效全局RMSE | 搬动后恢复 |
|---|---:|---:|---|
| normal / EKF-only | 无绝对地图原点 | 不适用；局部首原点诊断0.06582m | 无故障事件 |
| normal / EKF＋ACE | 545/570，95.61% | 0.17471m | 无故障事件 |
| normal / EKF＋ACE＋SPV原门控 | 528/570，92.63% | 0.17287m | SPV真实调用2次，接受0次 |
| fault / EKF-only | 无全局输出 | 局部首原点诊断1.4483m | 未恢复 |
| fault / EKF＋ACE | 435/533，81.61% | 0.05910m | **5.934s**；20次接受均正确，0有害纠偏 |

正常ACE-only和SPV实验组分别有2次>25cm或>10°误差增加，完整保留。SPV没有提供被接受的纠偏；本轮看到的是调用占用和初始化延迟，不能声称语义增益。故障中纯EKF无法感知无轮运动的全局位置跳跃，ACE能够重新锚定地图，这是直接对照证据。

缺失预测计入全部帧分母；RMSE只对有效输出计算，不能脱离覆盖率单看。原始故障流最大间隔为wheel0.218s、IMU0.190s、stereo0.500s；没有补造中间输入。详见 [回放汇总](online_ablation/summary.json)、[同输入与参数核验](online_ablation/comparison_protocol_audit.json)。

## 6. 边界与下一步

这是一栋旧验证建筑上的系统开发试验，scene03000曾参与SPV checkpoint选择，不是新建筑泛化、论文Benchmark或真实机器人部署验收。模型训练只使用建图参考监督，查询GT用于离线评价；部署资格是本开发域诊断决策，不能当成独立测试集性能优化。

下一步应冻结新建筑和路线，先解决SPVLoc候选召回：用独立训练建筑准备与机器人相机高度、视场和渲染域一致的训练视图，保持原网络/损失/推理核心；用独立验证建筑选择checkpoint，再看它是否真正救回ACE失败。没有互补性时保留负结果，不能强行三模型排名。

MAC-VO保留完整算法对照；实时升级需在目标硬件上验证异步运动约束的时间关联与计算预算，不能通过跳帧或换算法后仍称同一配置。MINS、多建筑多次故障、动态物体、光照变化、ZInD真实外观/真实机器人序列、Nav2消费接口和现场标定验收均尚未完成。本轮也没有训练新的融合神经网络；已实现的是可检验的调度、门控和状态管理。

## 7. 启动与复算

启动命令和接口见 [运行说明](../../docs/ROBOT_SIMULATION_zh.md)。省略SPV引用参数可运行仅EKF＋ACE；增加SPV参数则真实调用且默认shadow，`--allow-spvloc-corrections`仅显式实验消融。

```bash
cd /home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m research3d.robot_sim.demo   --world-manifest tmp/robot_sim_procthor_stereo/map_manifest.json   --ace-binding ../../outputs/RobotSim_20261107/ace_train/model_binding.json   --spvloc-references ../../outputs/RobotSim_20261107/spvloc_openroof   --mapping-dataset ../../outputs/RobotSim_20261107/stereo_dataset   --output ../../outputs/RobotSim_20261107/new_recovery_run   --seconds 60 --domain-id 72   --events configs/robot_sim/kidnap_events.json --camera-blackout 15 35
```

只清理该次命令自己创建的子进程，已有结果目录不可覆盖。Gazebo路线由轮驱指令实际执行，未实现基于估计位姿的自主导航规划。所有模型权重与大数据保留本地；GitHub包含可执行适配代码、冻结配置、实际指标、压缩日志和可视化。

## 8. 可视化与证据

- [实际故障恢复动画](media/online_fault_online_fault_replay.gif)：真实RGB/黑屏、连续局部轨迹、map定位、调用/拒绝和恢复状态；严格按返回时间显示，不借用未来响应。该capture没有运行MAC-VO，画面明确标注。
- [三真实模型同轨迹动画](media/three_model_localization_replay.gif)：包含失败的SPV候选以及E持续拒绝的负结果；MAC-VO只有离线首原点诊断对齐。
- [正常在线轨迹](online_normal/trajectory.png)、[故障在线轨迹](online_fault/trajectory.png)。
- [ACE](ace/)、[MAC-VO](macvo/)、[SPVLoc](spvloc/)、[传感器验收](sensors/)、[源码及数据版本](provenance.json)。

官方实现：[ACE](https://github.com/nianticlabs/ace)、[MAC-VO](https://github.com/MAC-VO/MAC-VO)、[SPVLoc](https://github.com/fraunhoferhhi/spvloc)。原方法没有被CLIP检索或模拟位姿替换。

最终验证：146项CPU/契约测试通过，1项需显式主机ROS环境的测试在该次批量执行中跳过；它已另外实际运行30秒并通过（约30.01Hz，证据见estimator_verification.json）。本轮所有自建仿真、模型和回放进程均已结束，无常驻巡检。磁盘报告见cleanup_storage.json。

GitHub中的逐帧大表与原始在线trace使用 `.json.gz` / `.jsonl.gz` 无损压缩；原始未压缩文件仍在本地运行目录，不改变任何评价数值。
