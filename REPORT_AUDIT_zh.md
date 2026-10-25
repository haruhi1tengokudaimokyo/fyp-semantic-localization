# FYP 工程审计与研究系统升级：真实运行报告

日期：2026-10-25。项目：Exploring Localization Techniques for Indoor Robotics。

## 结论与验收状态

已在原工程上完成系统性 P0 诊断、官方 coarse-to-fine 流程接入、严格区分的两套配置、32 图 A/B/C/D × 三种融合消融，以及 16 项实际通过的单元/集成测试。**低定位准确率尚未解决；P0 仅部分关闭，P1 的流程跑通，但独立建筑 Baseline 验收未通过。P2/P3 保持关闭。** 本轮没有训练新模型。

所有定位表格都是本机实测，既不是论文成绩，也不是跨建筑测试。只有 ZInD 官方公开示例的一栋住宅、8 个全景位置 × 4 个方向；预训练权重是否见过该住宅未知。不能将 32 个相关视图视为 32 个独立研究样本。

旧代码和结果先保留为 Git 检查点 `268ff7f`；流程实现检查点 `b8ba450`。本轮运行时工作树含新增诊断脚本，不能声称仅检出该提交即可逐字还原全部结果；最终源码和文件 SHA256 随交付归档。v1 的 clean/evaluation/training 等结果未覆盖。

## 1. 复用来源、版本、数据与授权

网络、coarse 匹配、候选提取和 fine 代价均来自 [SemRayLoc 官方仓库](https://github.com/TAU-VAILab/SemRayLoc)，固定提交 `072b7a3d6056c44ad95d08b269bdd296fc3eadab`；依据 [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/html/Grader_Supercharging_Floorplan_Localization_with_Semantic_Rays_ICCV_2025_paper.html)。没有重写一个新的大网络。

当前能合法取得并实际使用的是 [ZInD 官方 sample_tour](https://github.com/zillow/zind/tree/main/sample_tour)，源仓库提交 `06cfdbf295fa21500a857efe81f80f2243b8fb40`。正式结构射线训练首选 [Structured3D](https://structured3d-dataset.org/)，须取得官方授权；完整 ZInD 下载也需要官方流程。未使用镜像、未取得完整训练/验证/测试集。当前磁盘不足以盲目下载整个大型数据集，先申请并按建筑取小子集。

数据路径 `/data/fyp-semrayloc/data/zind_public_sample`。每图包含 RGB、固定语义楼层图路径、free mask、地图尺度、相机 GT 位姿、40 列轴向结构深度、40 列语义、building_id。RGB 为 640×360、80°水平 FOV；地图 0.01m/px，候选网格 0.1m、36 朝向。类别 0墙/1窗/2门/3未知；RGB 黑墙、蓝窗、红门。结构射线穿过门直到墙/窗，保留满足官方规则的门类别；不是家具表面 RGB-D 深度。

GT 只能用于预处理/离线评价。新在线入口仅接收 RGB、固定地图、free mask 和配置。没有 query GT pose、GT ray 或 oracle mask 参数。模型/地图加载不读取 query 的 sample.json。

## 2. P0 审计：证据与未知项

| 项目 | 已证实的结果 | 仍不能推出的结论 |
|---|---|---|
| 全景方向和位姿 | 由原始 ZInD spherical theta=atan2(-X,Y) 独立推导 bearing=90°−pano rotation−global rotation−crop yaw；32 图误差 < 2e−15rad；官方重裁剪逐像素差为0 | 不证明实际场景 Roll/Pitch 完全为0 |
| 颜色/归一化 | PIL RGB 输入，严格加载官方网络；实际 RGB-only 输出与保存的原权重输出一致；地图实际颜色与类别一致 | 不能因代码里 BGR 注释就断言通道错置 |
| 高度/FOV | 原注释还原相机高度约1.435m；本批裁剪80°与模型相符 | 机器人低视点/60°相机不能直接缩放当作同分布 |
| axial/radial | 显式 d_radial=d_axial/cos(angle)，逆变换有测试；精确 GT pose 重投1280条射线，深度最大差2.34e−6m，语义0差异 | 标签由地图生成，重投一致本身不能证明现实场景结构/地图完全准确 |
| coarse 语义分支 | 原代码未正确派发 semantic 类型会把类别数值当深度处理，单独行为补丁并测试 | 不是纯运行兼容修复；影响结果必须披露 |
| fine 深度接口 | 我们保存的轴向 GT 直接交给上游 radial fine 比较时，精确 GT pose 平均代价0.06346；统一 radial 后约7.13e−8 | checkpoint 无深度约定元数据，不能反推作者权重究竟按哪种标签训练 |
| checkpoint | 原始 depth/semantic strict load 成功，输出40列、4射线类别和14房间类别；SHA固定；未使用单图微调权重 | 无训练划分/超参字段，无法证明公开住宅与预训练集互斥 |
| 候选/fine | 上游 XY 候选与集成 SE(2) NMS 分开；全部候选和±5°细化代价留档 | fine 不是保证改进的优化器；这里只使用官方固定位置、方向偏移搜索 |

原 aggregate patch 保留。新增 `patches/01_runtime_compat.patch`、`02_semantic_dispatch.patch`、`03_dataset_storage.patch`：第二项改变匹配行为；第三项含 float depth 等标签存储变更，可能改变监督值，不能假称无行为变化。三补丁已在固定上游干净归档应用成功。快速射线后端只是性能替代，已对200条真实地图射线及上游 fine 输出逐值验证。

## 3. 同批 A/B/C/D 控制实验

A=GT Depth+GT Semantic；B=Pred Depth+GT Semantic；C=GT Depth+Pred Semantic；D=双预测。为隔离射线感知，以下 A–D 均关闭房间先验；另单独运行 official D_room_aware 恢复完整配置先验。

两种设置预先固定而非按结果挑选：official_reproduction 保留上游语义 multinomial 采样、vote 重采样、XY候选与原 fine 深度约定；our_integration 使用 argmax、针孔 nearest 重采样、0.5m/20° SE(2) NMS 和明确的轴向→径向 fine 转换。二者有多处差异，不能据此宣称某单项改动提升或降低效果。两者都仍依赖公开小样本，不是已验收论文复现。

| 配置 | 条件 | coarse XY中位(m) | coarse R@1m,30° | fine XY中位(m) | fine R@1m,30° |
|---|---|---:|---:|---:|---:|
| official_reproduction | A_gt_gt | 0.172 | 68.750% | 0.196 | 62.500% |
| official_reproduction | B_pred_gt | 4.085 | 9.375% | 5.108 | 6.250% |
| official_reproduction | C_gt_pred | 2.976 | 43.750% | 2.605 | 40.625% |
| official_reproduction | D_pred_pred | 5.087 | 12.500% | 4.971 | 12.500% |
| our_integration | A_gt_gt | 0.166 | 68.750% | 0.211 | 62.500% |
| our_integration | B_pred_gt | 4.682 | 9.375% | 5.210 | 12.500% |
| our_integration | C_gt_pred | 2.723 | 40.625% | 4.151 | 37.500% |
| our_integration | D_pred_pred | 6.017 | 6.250% | 5.051 | 6.250% |

| 配置 | 条件 | Depth-only coarse→fine | Semantic-only coarse→fine | Combined coarse→fine |
|---|---|---:|---:|---:|
| official_reproduction | A_gt_gt | 40.625% → 43.750% | 3.125% → 3.125% | 68.750% → 62.500% |
| official_reproduction | B_pred_gt | 3.125% → 3.125% | 3.125% → 3.125% | 9.375% → 6.250% |
| official_reproduction | C_gt_pred | 40.625% → 43.750% | 0.000% → 0.000% | 43.750% → 40.625% |
| official_reproduction | D_pred_pred | 3.125% → 3.125% | 0.000% → 0.000% | 12.500% → 12.500% |
| our_integration | A_gt_gt | 40.625% → 46.875% | 6.250% → 9.375% | 68.750% → 62.500% |
| our_integration | B_pred_gt | 3.125% → 0.000% | 6.250% → 9.375% | 9.375% → 12.500% |
| our_integration | C_gt_pred | 40.625% → 46.875% | 0.000% → 3.125% | 40.625% → 37.500% |
| our_integration | D_pred_pred | 3.125% → 0.000% | 0.000% → 3.125% | 6.250% → 6.250% |

完整预测模式的 fine 指标：

| 配置/条件 | R@0.1m | R@0.5m | R@1m | R@1m,30° | XY RMSE(m) | Yaw MAE(°) | XY>3m |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_reproduction/D_pred_pred_room_aware/combined | 0.000% | 9.375% | 12.500% | 12.500% | 6.801 | 76.74 | 62.500% |
| our_integration/D_pred_pred/combined | 0.000% | 6.250% | 6.250% | 6.250% | 6.905 | 79.44 | 81.250% |

A 的 coarse 联合 Recall 仅68.75%，fine 还降到62.5%，说明感知之外还存在地图重复/离散化/候选和 fine 选择限制，不能将所有失败归因于网络。B 极差而 C 明显好于 D，支持本批深度感知误差是主要瓶颈之一，但没有用一个替换实验唯一确定因果。A 中 top5 已含正确候选的覆盖率84.375%，仍有7图 fine 未选对；D 中 official top5覆盖仅18.75%、集成9.375%，候选阶段已丢失大量正确区域。

### 语义和深度

1280条射线，argmax 语义混淆矩阵（行GT，列预测，顺序墙/窗/门/未知）：

```
970   4  49  0
 27  22  17  0
111  17  63  0
  0   0   0  0
```

| 类别 | GT数量 | 预测数量 | Precision | Recall |
|---|---:|---:|---:|---:|
| 墙 | 1023 | 1108 | 87.55% | 94.82% |
| 窗 | 66 | 43 | 51.16% | 33.33% |
| 门 | 191 | 129 | 48.84% | 32.98% |
| 未知 | 0 | 0 | 未定义 | 未定义 |

17/32 图 argmax 全墙。总体准确率82.42%，但恒预测墙已能得到79.92%，所以总体准确率掩盖门窗漏检。本批类别不平衡和门窗漏检已证实；训练集不平衡、域偏移、可见门窗与结构标签不一致仍是待查假设，不能当作既定原因。

预测深度相对本批 axial GT 的平均 MAE=0.827m，按 radial 假设 MAE=0.870m；此小差异不能确定作者训练约定。敏感性实验分别固定5个种子及 axial/radial 假设：axial multinomial coarse 联合 Recall 范围3.125%–12.5%；argmax为6.25%。不能挑最高种子报作改进，亦不能把相对v1数字变化归为算法贡献。

### 高匹配分数不是高成功概率

coarse raw score 来自启发式归一化误差的指数，9条稀疏射线和大量墙类可让错误位置也得到接近1的分数。原先0.9775不能解释为97.75%定位可信度。新结果明确输出 `score_kind`、`success_probability: null`、完整候选和离散网格归一化质量/熵；网格质量也不是经过校准的成功概率。未经独立建筑校准，`accepted_pose` 保持 null；`estimated_pose` 只供研究分析。日志中 counterfactual ECE 是“错误地把原始分数当成功概率”的诊断，不是已完成校准。

## 4. 运行、资源与测试

GTX1660 SUPER 6GB，RAM约15GiB；/data剩18GiB、根分区剩29GiB。驱动580.178.04显示的CUDA13.0是驱动能力；本虚拟环境实际 PyTorch2.5.1+cu121，二者不要混同。没有覆盖ROS Python、Gazebo工程或CUDA依赖。

RGB-only完整单查询基准实际32图、2次warmup、每图清fine cache：网络均值54.43ms；含RGB读取的处理均值313.38ms、P95 420.23ms、约3.19FPS；PyTorch峰值 allocated263.80MiB、reserved294MiB。排除模型载入和地图射线库预计算，不是ROS端到端实时性能，也不包含其他进程显存。

16项测试全部实际通过，7.131s，完整日志在 `results/audit_v2/logs/tests_acceptance.log`：坐标逆变换与尺度拒绝、轴向/径向、原注释方向及裁剪、颜色与门穿透、真实射线后端、GT定位、fine等价、候选方向多样性、全遮挡拒绝、部分遮挡支持检查、GT loader抛异常时在线仍可运行、原checkpoint strict load及历史训练主干不变/头部实际更新、正式训练门控等。全遮挡拒绝推理；基线对部分射线权重显式报错，避免假装已支持鲁棒融合。

旧20步单图 smoke checkpoint 保留，仅检查已有权重确有更新；本轮没有再次训练。`src/train.py` 正式训练分支已门控，独立建筑基线/完整定位验证/最佳权重选择协议未完成前拒绝启动。不能用单图loss下降跨过验收门槛。

## 5. 遮挡诊断与研究边界

仅新增受控灰矩形感知敏感性实验：无遮挡与10/25/40/60%遮挡，同32图、同权重，没有向网络提供oracle mask。

| 遮挡面积 | 深度MAE(m) | 语义准确率 | 语义平均熵 |
|---|---:|---:|---:|
| 0% | 0.827 | 82.42% | 0.389 |
| 10% | 0.881 | 80.78% | 0.448 |
| 25% | 0.938 | 78.98% | 0.461 |
| 40% | 1.052 | 80.94% | 0.452 |
| 60% | 1.120 | 80.47% | 0.422 |

深度误差增加，但熵不单调，因此不能只靠置信度识别人物。这里没有真实人物检测准确率，也没有宣称动态鲁棒性改善。v1 oracle过滤/JSD实验保留为历史工程检查，未纳入新 official 流程。

[CVPR2026几何语义融合论文](https://openaccess.thecvf.com/content/CVPR2026/html/Ye_Fusion_of_Depth_and_Semantics_for_Probabilistic_Floorplan_Localization_CVPR_2026_paper.html)的DINOv2、跨注意力、Depth Confidence与JSD提供后续研究依据；此前核查未找到可核验官方实现，本项目JSD不冒充其实现。当前无独立数据，未训练或测量DINOv2交互层显存，不能断言6GB可以完成该方法。

P1通过后先比较无处理/真实人物分割过滤/独立校准不确定性加权/合并四组；同建筑互斥测试、人物外观与轨迹互斥。保留静态地图；oracle mask仅另列上界。冻结主干、只训练小交互层可以作为后续实验设计，但应先做batch1显存测量，再决定AMP/累积，不能先承诺效果。最终再进入[Habitat3](https://aihabitat.org/habitat3/)真实运动遮挡。

## 6. 现有 ROS2/Gazebo：实际轨迹预检，不是连续定位成绩

只读扫描发现第一份bag空、第二份 `/data/FYP_Indoor_Localization/results/indoor_bag_20261014_171719` 实有约30秒36848条消息：300帧RGB/深度/CameraInfo，882条odom、882条GT，以及scan/IMU/TF。已解码首帧和真实轨迹，未修改已有工程。

相机320×240，fx≈277.19，水平FOV≈59.99°，低机器人视点；与当前80°/约1.435m数据存在明显差异。未将简单resize后的图强送模型制造成绩。GT frame=world，odom frame=odom，当前图分开画，未直接混合两个坐标系计算RMSE。

本轮无模型预测轨迹、无连续XY RMSE/Yaw MAE、无全局收敛/重定位时间。可视化中的实际GT路径不能误称定位预测。ObservationModel 已独立；Temporal接口设计见 `docs/AUDIT_V2.md`，尚未实现或运行filter。[F3Loc](https://github.com/felix-ch/f3loc)与[UnLoc](https://github.com/matthias-wueest/UnLoc)作为后续参考；AMCL不是系统依赖。

## 7. 交付位置、运行与剩余阻碍

- 原工程新增 `src/localization/{datasets,models,floorplan,matching,evaluation}`；在线 `src/localize.py` 与离线 `src/audit.py` 分开。
- 配置 `configs/official_reproduction` / `integration` / `experimental` 清楚区分；补丁与测试可审阅。
- 本轮完整原始逐图JSON和热图npz位于 `/data/fyp-semrayloc/runs/audit_v2/diagnostic_controlled/per_image`；交付目录 `results/audit_v2/raw_per_image` 是本机链接。汇总CSV/JSON、版本/权重SHA、真实命令、日志、32幅逐图图和4幅总图已交付；图册 `results/audit_v2/gallery.html`。
- 本机图册服务 http://127.0.0.1:18766/gallery.html，旧训练记录18765保留；跨机器须重新启动本地服务或打开图册文件。
- 新的推理/诊断/可视化复跑命令见 `docs/AUDIT_V2.md`。完整审计可直接运行网络，不必依赖本轮感知缓存。

当前真实阻碍：完整授权数据与独立建筑缺失；原checkpoint训练约定/重叠未知；候选与fine在GT输入上仍会失败；深度误差和门窗漏检明显；成功概率尚未校准；机器人内参与视点尚未适配。下一步先取得少量授权且建筑互斥的数据，核实checkpoint的训练深度约定并复跑完整官方Baseline；进一步用精确GT/离散网格/候选覆盖/候选选择实验定位剩余matcher失败。确认有效Baseline后才能正式微调或做鲁棒性研究。
