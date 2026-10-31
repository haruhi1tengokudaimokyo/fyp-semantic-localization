# 论文与工程复现边界（2026-10-31）

## 本轮实际执行的方法

[ACE 论文（CVPR 2023）](https://arxiv.org/abs/2305.14059)与[官方工程](https://github.com/nianticlabs/ace)，固定版本见 `upstream_versions.json`。已实际下载论文 PDF 并提取方法及附录阅读。其场景无关编码器提取图像特征，场景专属回归头通过训练图像位姿的重投影监督学习世界坐标；查询图像的像素与预测三维点交给 DSAC* 的几何 RANSAC 求 6DoF 位姿。测试深度与测试真值不属于定位器输入。

原论文/官方配置采用 8M 特征缓存、batch 5120、16 epochs。6 GB 本机的本地训练预先采用 1M 缓存、相同 batch/epochs，保持主干、损失、采样及优化器来自官方；这减少了训练样本和更新总量，不能称为原论文完整训练配方。官方预训练 Heads 权重与本地训练分别评估，不选择测试集成绩最好的一次训练。

兼容性：`dsacstar/setup.py` 仅修改系统 OpenCV include/lib 路径及非 conda 环境检测。官方训练器依赖已移除的 PyTorch `Optimizer._step_count`；本地入口通过公共 post-step hook 记录真实优化步数，保留 GradScaler 跳步时不推进学习率的逻辑。首次失败日志保留。DataLoader worker 从12降为2。没有把重新设计的网络冒充 ACE。

## 已核实入口，尚未运行的方法

- [hloc](https://github.com/cvg/Hierarchical-Localization)：完整检索、局部匹配与几何求解框架。本轮满足“hloc 或 ACE”的 ACE 路径，未报告 hloc 成绩。
- [MAC-VO](https://github.com/MAC-VO/MAC-VO)：项目公开 Fast Mode、前端协方差和 PoseNet 权重；其 README 列明 CUDA runtime≥12.4、Python≥3.10，Fast Mode 的显存数字是作者声明，**本机未测量**。当前 ACE 的 torch 2.5.1+cu121 环境不能直接当作其兼容环境。本轮未运行，也没有把 confidence 当成位姿协方差。
- [MINS](https://github.com/rpng/MINS)、[OpenVINS](https://github.com/rpng/open_vins)：MINS 包含轮速等多传感器更新，但不能未经数据时间戳、外参、状态定义与后端验证就接 MAC-VO。当前数据没有 IMU/Wheel；不虚构。
- [SemRayLoc](https://github.com/TAU-VAILab/SemRayLoc)：旧工程完整保留。二维 floorplan 射线结果未混入本轮三维结果。[CVPR 2026 指定页面](https://openaccess.thecvf.com/content/CVPR2026/html/Ye_Fusion_of_Depth_and_Semantics_for_Probabilistic_Floorplan_Localization_CVPR_2026_paper.html)本次请求返回403，不能据此证明已读取准确论文版本或完成复现。
- [SegFormer](https://github.com/NVlabs/SegFormer)：网络代码入口已核实，当前 HSSD 原生结构标签不满足完整墙门窗训练，未启动形式化小样本训练来冒充 P3。
- [OMCL](https://github.com/AIS-Bonn/omcl)、[VLG-Loc](https://github.com/CyberAgentAILab/VLG-Loc)：官方入口已核实；尚未推理或融合。语义网络与 VLM 接口分离。
- [FastForward 官方项目](https://nianticspatial.github.io/fastforward/)：准确题目为 *A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features*，ICLR 2026。官方页面描述特征锚定场景表示与查询场景坐标预测。本次页面未发现可核验代码下载入口，仅能确认论文和项目；没有自拼模型并称为复现。
- [DynaVINS](https://github.com/HKUST-Aerial-Robotics/DynaVINS)：只核实入口；未声称动态掩码或 VIO 实验。

GitHub REST 公开 API 对本次主机返回403；使用 `git ls-remote` 成功核实各代码 HEAD，二者状态均保存在版本清单。具体 commit 只代表本次查询，不代表已安装和实测。
