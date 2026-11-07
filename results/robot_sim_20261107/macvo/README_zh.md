# Gazebo 完整双目序列 MAC-VO 验证

使用原官方 MAC-VO 的 Paper_Reproduce 配置、发布权重、FlowFormerCov、CovAwareSelector、TartanMotionNet 和三维协方差加权 ICP 后端。适配器只转换 Gazebo 双目数据与输出坐标，不修改 MAC-VO 核心。GT depth、flow、pose 始终为空；GT 仅由推理结束后的独立 CPU 评价命令读取。

GTX 1660 SUPER 不支持 BF16，采用既有 FP32 前端适配；关闭与本次 VO 评价无关的稠密建图。已有上游兼容补丁及检查点键适配记录在 `protocol.json`、`checkpoint_audit.json`。本方法没有重新训练，也不是原论文数据集 Benchmark。

实际完整处理 561 对 640×480 RGB，双目基线 0.10 m，左右时间戳一致；轨迹覆盖 59.9 秒仿真时间。保存所有原始采样间隔：中位数 0.1 秒，最大 0.3 秒，37 个间隔超过 0.15 秒，实际 9.35 Hz。采集间隔变大与算法失跟分别统计，没有插帧掩盖数据缺口。

保存的位姿在官方 `MotionInterpolate` 执行前从图状态复制，并转换为首帧相机光学坐标规范下的 `T_odom_camera`。输出是相对里程计，不是已知地图中的全局定位；完整出处在 `raw_pose_provenance.json`。

| 指标 | 结果 |
| --- | ---: |
| 完整输入 / 原始位姿数 | 561 / 561 |
| 算法失跟数 | 0 |
| SE(3) 刚体对齐 ATE RMSE，无尺度修正 | 0.12381 m |
| 仅首帧锚定 ATE RMSE | 0.31728 m |
| 首帧锚定末端误差 | 0.37554 m |
| 相邻帧 RPE 平移 / 旋转 RMSE | 0.002855 m / 0.05542° |
| 推理总墙钟时间 | 756.57 秒 |
| 单帧平均耗时 | 1330.80 ms |
| 峰值显存分配 | 2960.35 MiB |

评价中的刚体对齐和首帧锚定都属于离线 VO 指标，不能用作在线全局初始化。相机 GT 路程是 8.4767 m，而 `base_link` 路程约 7.131 m；相机位于机器人前方 0.20 m，转弯时力臂使两条轨迹路程不同，不能误认为尺度或坐标错误。若转为底盘里程计，必须同时转换坐标规范和固定相机外参。

60 秒输入耗时约 12.6 分钟，本机此配置不满足实时连续 MAC-VO。此结果证明官方算法在共同模拟序列上完成了全程计算，不能证明三模型并发实时、长时间可靠性、未见建筑泛化或真实机器人部署表现。本轮未按查询 GT 调整模型、阈值或轨迹。

此目录只保存小型协议、指标和来源记录；完整预测、逐帧耗时、原始 `trajectory.npz`、控制台日志位于 `/home/csy/Documents/Codex/2026-10-31/new-chat/outputs/RobotSim_20261107/macvo_queries`。可复现入口为 `research3d/robot_sim/run_macvo.py` 的 `validate`、`infer`、`evaluate` 子命令。
