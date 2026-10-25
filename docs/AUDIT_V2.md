# Audit v2: reproduction and interface contract

本文件是实际已运行程序的复跑说明。实测和验收边界见 [中文报告](../REPORT_AUDIT_zh.md)。不启动新训练，不将单楼示例当独立测试。

## Environment

在本仓库根目录运行，保留原ROS2环境：

```bash
export FYP_ROOT=/data/fyp-semrayloc
export FYP_PY="$FYP_ROOT/envs/semrayloc/bin/python"
export MPLCONFIGDIR="$FYP_ROOT/cache/matplotlib"
unset FYP_FINETUNED
```

新输出目录必须不存在，避免覆盖。以下 `rerun_*` 是复跑用的新目录，不冒称本轮原始运行目录；本轮原命令在 `results/audit_v2/run_manifest.json`。

## Online RGB-only query

```bash
"$FYP_PY" src/localize.py \
 --rgb "$FYP_ROOT/data/zind_public_sample/samples/000_000/rgb.png" \
 --map "$FYP_ROOT/data/zind_public_sample/floorplan_semantic.png" \
 --free-mask "$FYP_ROOT/data/zind_public_sample/free_mask.npy" \
 --profile configs/integration/zind.json \
 --out "$FYP_ROOT/runs/rerun_query"
```

result.json 输出位姿、5个候选和每候选0/±5°试验、raw score、fine cost、网格熵、显存/计时及地图hash。likelihood.npz 保存完整原始score volume和XY归一化边缘质量。estimated_pose仅供研究分析；accepted_pose=null，success_probability=null。在线接口没有query GT或oracle mask；无房间多边形输入时显式标明room prior未应用。

当前地图坐标为x图像向右、y图像向下，位姿theta=0向右、theta正方向的射线为[cos(theta),-sin(theta)]，单位米/弧度。ROS world通常y向上，需显式已知map origin、轴向和camera→base外参；不能直接发未经变换的TF。

## Offline controlled diagnostic

```bash
"$FYP_PY" src/audit.py --out "$FYP_ROOT/runs/rerun_audit"
"$FYP_PY" src/audit_exact_gt.py --out "$FYP_ROOT/runs/rerun_exact_gt.json"
"$FYP_PY" src/audit_hypotheses.py --out "$FYP_ROOT/runs/rerun_hypotheses"
"$FYP_PY" src/perception_sensitivity.py --out "$FYP_ROOT/runs/rerun_occlusion"
"$FYP_PY" src/benchmark_online.py --out "$FYP_ROOT/runs/rerun_benchmark"
"$FYP_PY" -m unittest discover -s tests -v
```

主诊断对同32图运行2profile×A–D×3fusion×coarse/fine；额外official D_room_aware。A–D统一关闭房间prior，以隔离射线因素。直接运行网络，结果中perception.json可复用；`--cached-perception`仅限同顺序/同源manifest已核实缓存。本轮用 `/data/fyp-semrayloc/runs/audit_v2/perception_v1.json`，其SHA随run manifest。热图有max score和marginal mass两种，不能混为成功概率。

exact GT audit为离线诊断，不属于在线输入。候选coverage和最终fine recall不同：有正确候选也可能选错。公开小样本没有可靠的独立校准集，不能用同32图拟合并报告置信度有效。

## Visualization

```bash
"$FYP_PY" src/visualize_audit.py \
 --run "$FYP_ROOT/runs/rerun_audit" --out results/rerun_gallery
python3 -m http.server 18766 --bind 127.0.0.1 --directory results/audit_v2
```

本轮图册 [gallery](../results/audit_v2/gallery.html) 逐图展示RGB、GT/预测射线、候选、GT/预测位姿、8个A–D热图。包含失败图，不筛选成功案例。原始RGB/逐图图保留本地；不将原始数据和作者权重重新发布到Git。只上传汇总图和指标。

## Existing ROS bag preflight (read-only)

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 src/rosbag_preflight.py \
 --bag /data/FYP_Indoor_Localization/results/indoor_bag_20261014_171719 \
 --out /data/fyp-semrayloc/runs/rerun_bag_preflight
```

用ROS自有Python，不向模型venv安装ROS依赖。SQLite以mode=ro打开。轨迹只是实际GT/odom；frame未对齐、没有模型预测。可视化world轮廓对应当前既有indoor.sdf，其他world须重新读取验证，不能照搬。

## Planned temporal API — not implemented

ObservationModel.predict(rgb, optional_validity_mask)返回40列深度、语义/深度/房间概率、射线support、计时。基线matcher对部分权重显式拒绝，实验性mask尚未接入新管线。

后续TemporalEstimator应有initialize(prior_SE2_grid)、predict(odom_delta,covariance,dt)、update(observation_likelihood,validity,timestamp)、estimate()四个独立接口。状态需含概率分布/多峰假设、地图与frame id、lost/recovering状态、校准来源；运动缺失/观测全失时仅传播运动与增加不确定度，不能凭高raw score强制全局跳变。未来以同轨迹比较histogram/particle，处理角度周期、运动协方差、重定位注入和失效检测。此处是接口设计，无虚构实现或轨迹指标。

## Training gate

正式训练被src/train.py拒绝。保留的--smoke仅供历史工程检查，单图20步不是研究训练。授权建筑互斥数据、官方baseline、全量定位验证、最佳checkpoint选择及独立测试协议落实后，才打开正式训练。当前没有DINOv2微调显存测量或可训练跨注意力模块。
