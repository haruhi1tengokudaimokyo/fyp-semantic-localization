# MAC-VO 消融与运行速度核查（2026-11-02）

ACE 比较口径固定为 7Scenes 原 D-SLAM 位姿、七场景宏平均、5cm/5°：本机 80.5221%，论文表中 80.8%，相差约 -0.28 个百分点。97.1% 属 SfM/Pseudo-GT 条件，不能混比；这七个场景均重新训练了专用 8M 模型并完整评价。未为本次核查重训 ACE。论文基准见[ACE 原论文表1](https://openaccess.thecvf.com/content/CVPR2023/papers/Brachmann_Accelerated_Coordinate_Encoding_Learning_to_Relocalize_in_Minutes_Using_RGB_CVPR_2023_paper.pdf)。

MAC-VO 旧 `original_icp_weight_ablation` 确实不等于端到端轨迹。代码从优化后的历史图中重建世界点，两个分支都用上一帧位姿作为静态初值；真实 Paper 链通过 `TartanMotionNet.predict` 提供初值。ICP 的非线性鲁棒优化只执行有限迭代，初值/世界点不同会改变结果。旧 Hospital/Factory 加权重放与原轨迹最大矩阵元素差分别为 0.1022/5.8028。因此这些结果保留为受控后端消融，不用来声称完整论文消融。

本次增加的链保持原前端、协方差选点、TartanMotionNet 和原并行 ICP：

|入口|用途|
|---|---|
|`macvo_instrumented_sequence.py --backend-covariance original`|完整 FP32 控制轨迹与模块耗时，捕获实际 GraphInput（包括真实初值/世界点/协方差）和子进程结果|
|`--backend-covariance identity`|真正的端到端 w/o CovOpt，只将原 ICP 求解中使用的协方差替换为单位矩阵；各帧仍走实时完整原算法|
|`--encoder fp16 --decoder fp16`|独立硬件适配速度候选，保持 Paper 的12次 decoder 更新；不是 BF16/4次更新的 Fast 模式|
|`macvo_exact_solver_replay.py`|用逐项 SHA 核对的原始 GraphInput 重算全部实际优化任务，验证与真正子进程输出的一致性；不是新的 VO 消融|

spawn 子进程兼容性已用真实 Hospital 数据的198个对应点、原 ICP 优化器检查：原协方差透传与未挂钩求解参数最大差为0；单位权重改变求解结果（最大参数差0.0057799），子进程中确实执行协方差调用。该合同检查不能代替完整轨迹的最终验收；完整轨迹由 SQLite 对 summary/profile、独立官方指标、精确重放和最终 comparison 分别验收。

速度比较须区分配置和硬件。论文表V使用 Ryzen 9 5950X + RTX 3090 Ti、640×640：Raw 2.15 FPS，TRT+MP 3.96 FPS，Fast 10.57 FPS。Fast 使用 BF16 并将 decoder 更新从12次减为4次。本机 GTX1660 SUPER 的 FP32 Paper 约0.74 FPS（且图像640×480），不能把与 Fast 的差距全归因于复现错误。来源：[MAC-VO 表V、IV-C与附录C](https://arxiv.org/html/2409.09479v2)。

新计时移除逐帧全局同步，使用不阻塞的 CUDA event 记录；完整 pipeline 时间包含序列读取、首次 CUDA Graph 建立和最后一次优化收尾。各阶段 CPU/GPU 时间可能重叠，不相加；记录并行优化子进程的实际 CPU 求解时间。旧版已经使用 CUDA Graph 与并行 ICP，不能把这些既有功能当作本轮新加速。只有完成后的 FP16 profile 才能报告速度增益。新GPU计时任务独占现有DAG的4个CPU调度槽，避免精确后端重放等CPU任务与速度试验并行；不抢占已运行任务。

FP16 判据在运行前写入代码与测试：ATE ≤ max(1.05×FP32 ATE, FP32 ATE+0.01m)，丢帧数 ≤ FP32+1；完整序列、同场景/同权重/同seed/同12次迭代。若失败，comparison 明确标失败，不能只展示更快的速度。该检查是公开测试序列上的透明硬件适配验证，不是重新选择论文超参数或宣称独立验证集。正式 FP32 原结果保持不变。

新增真实 DAG 任务使用 `R.MACVO.{hospital,abandonedfactory}.P000.audit.{original_fp32,original_fp16,identity_fp32}.v1`，其后连接官方指标、精确求解重放和全链比较任务。所有旧已接受任务的科学字段保持不变，健康任务不停止，不恢复巡检。

同论文 Benchmark 尚未补齐：2026-11-02 再测原 ETH 公开源读取超时、新 ETH 内容入口429、UZH镜像TLS错误；结果保存在 automation/artifacts/macvo_euroc_source_recheck_20261102.json。官方配置的 TartanAir v2 自采测试序列使用作者本地路径，已取得的 v1 序列与官方 Demo 不能冒充表格同轨迹。见[官方源码](https://github.com/MAC-VO/MAC-VO)。另外已检查TartanVO官方CMU Box样例ZIP：1678个左目PNG、0个右目，不能作为完整双目替代。公开源未取得合法完整双目文件前保持阻塞。
