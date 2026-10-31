# 本机复跑命令

从仓库根目录执行。原二维入口仍在 `src/`；本轮独立代码在 `research3d/`。本机代码、数据与环境分别存放；不改 ROS Python。以下是已执行步骤及其可复跑形式。首次全新安装脚本是新环境配方，没有额外声称在第二台机器完成过复现。

```bash
export FYP3D_ROOT=/data/fyp_3d
export PYTHONPATH="$PWD/research3d"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# 本机环境已安装，不必重复运行 setup_ace.sh。
# 新环境需 Python3.10、C++编译器、pkg-config、系统OpenCV开发头文件。
# FYP3D_PYTHON=python3.10 bash research3d/scripts/setup_ace.sh
```

本机 ACE 环境通过 `.pth` 只读复用旧 semrayloc 环境的 torch/cu121 基础包，新增包在新 venv 中安装。依赖快照为 `requirements-observed.txt`；若旧环境不再存在，使用新环境安装配方，不能只复制 `.pth`。

## 下载与验收

```bash
python3 research3d/scripts/download_range.py \
  https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip \
  /data/fyp_3d/downloads/heads_range.zip --workers 16
/data/fyp_3d/envs/ace/bin/python research3d/scripts/prepare_heads.py \
  /data/fyp_3d/downloads/heads_range.zip /data/fyp_3d/datasets/7scenes \
  --ace /data/fyp_3d/sources/ace
python3 research3d/scripts/fetch_assets.py /data/fyp_3d/datasets/replicacad --dataset replica
python3 research3d/scripts/fetch_assets.py /data/fyp_3d/datasets/hssd --scenes 3
python3 research3d/scripts/audit_assets.py /data/fyp_3d/datasets/hssd
/data/fyp_3d/envs/ace/bin/python research3d/scripts/fetch_ace_head.py /data/fyp_3d/runs/ace_official/heads.pt
```

模型下载器流式读取官方归档，在取得所需 Heads 权重后停止；单独检查权重内部 ZIP CRC 与固定 SHA256，不假称验证整个 700 MB gzip。下载与使用仅限相应许可范围，许可证见官方页面和本目录 ACE-LICENSE。

## 三维渲染与独立坐标验收

本机 Habitat 0.3.3 / Python3.9 / headless+Bullet 环境已经准备好。精确包 URL 列在 `habitat-explicit-linux64.txt`。安装 headless **nobullet** 会漏掉 ReplicaCAD 铰接家具，因此不能用于 apt_0 完整验收。Bullet 仅用于加载资产，没有仿真步进或在线地图更新。

```bash
/data/fyp_3d/envs/habitat/bin/python research3d/scripts/render_habitat.py \
  /data/fyp_3d/datasets/replicacad replicaCAD.scene_dataset_config.json apt_0 \
  /data/fyp_3d/runs/replicacad_apt0 --views 24
/data/fyp_3d/envs/habitat/bin/python research3d/scripts/render_habitat.py \
  /data/fyp_3d/datasets/replicacad replicaCAD.scene_dataset_config.json empty_stage \
  /data/fyp_3d/runs/replicacad_empty --views 16
/data/fyp_3d/envs/ace/bin/python research3d/scripts/check_mesh_depth.py \
  /data/fyp_3d/runs/replicacad_empty \
  /data/fyp_3d/datasets/replicacad/stages/frl_apartment_stage.glb
/data/fyp_3d/envs/habitat/bin/python research3d/scripts/render_habitat.py \
  /data/fyp_3d/datasets/hssd hssd-hab.scene_dataset_config.json 105515211_173104185 \
  /data/fyp_3d/runs/hssd_105515211 --views 24
# 另外两场景：107734110_175999914、107734119_175999932，各24视角。
python3 research3d/scripts/audit_semantics.py \
  /data/fyp_3d/datasets/hssd /data/fyp_3d/runs /data/fyp_3d/runs/hssd_semantic_audit.json
/data/fyp_3d/envs/ace/bin/python research3d/scripts/build_geometry.py \
  /data/fyp_3d/datasets/7scenes /data/fyp_3d/runs/heads_geometry --ace /data/fyp_3d/sources/ace
```

独立 mesh 检查使用原始三角网格，不使用被检查的深度反投影生成网格。HSSD 相同的检查先将 scene 参数设为 `stages/105515211_173104185.glb` 的绝对路径，生成 `/data/fyp_3d/runs/hssd_stage_only`，再用 check_mesh_depth 比较该 GLB。它验证 stage 坐标和深度，不冒称覆盖全部家具三角面。

## 官方推理、本地训练与统一接口评估

```bash
# 原官方入口，输出1000条真实位姿和官方指标
(cd /data/fyp_3d/sources/ace && \
 PYTHONPATH=/data/fyp_3d/sources/ace/dsacstar OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
 /data/fyp_3d/envs/ace/bin/python test_ace.py \
 /data/fyp_3d/datasets/7scenes/7scenes_heads /data/fyp_3d/runs/ace_official/heads.pt --session official)

# 官方训练器：1M特征缓存、16epochs、5120batch，只有train序列
/data/fyp_3d/envs/ace/bin/python research3d/scripts/train_ace_local.py \
 /data/fyp_3d/datasets/7scenes/7scenes_heads /data/fyp_3d/runs/ace_local --ace /data/fyp_3d/sources/ace

# 统一接口：定位器不读GT，预测后离线评价
/data/fyp_3d/envs/ace/bin/python research3d/scripts/evaluate_ace.py \
 /data/fyp_3d/datasets/7scenes/7scenes_heads /data/fyp_3d/runs/ace_official/heads.pt \
 /data/fyp_3d/runs/ace_adapter_official --ace /data/fyp_3d/sources/ace
/data/fyp_3d/envs/ace/bin/python research3d/scripts/evaluate_ace.py \
 /data/fyp_3d/datasets/7scenes/7scenes_heads /data/fyp_3d/runs/ace_local/heads_local.pt \
 /data/fyp_3d/runs/ace_adapter_local --ace /data/fyp_3d/sources/ace
FYP3D_BASE=/data/fyp_3d /data/fyp_3d/envs/ace/bin/python -m unittest discover -s research3d/tests -v
/data/fyp_3d/envs/ace/bin/python research3d/scripts/plot_results.py results \
 /data/fyp_3d/runs/ace_adapter_local research3d/results/figures/local \
 --scene /data/fyp_3d/datasets/7scenes/7scenes_heads --geometry /data/fyp_3d/runs/heads_geometry
```

推理固定模型/地图；每帧输出绝对位姿，不累积在线地图。`stamp_ns=None` 表示 7-Scenes 未提供本工程可验证的硬件时间戳，不把帧序号虚构成 IMU/轮速时间。默认不把独立全局估计写入任何连续状态；`accepted=False`，因为本轮尚未评价时间一致性恢复门控。

## OpenLORIS 仅远端元数据预检

```bash
python3 research3d/scripts/inspect_remote_rosbag.py \
 https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/rosbag/market1-1.bag \
 /data/fyp_3d/runs/openloris_remote_topics.json
python3 research3d/scripts/inspect_remote_tar.py \
 https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/rosbag/office1-1_7-rosbag.tar \
 /data/fyp_3d/runs/openloris_office_tar_members.json
```

这里只验收远端大小、bag连接类型和tar成员头，不声称完整包SHA验证或运动估计。office1-3是office归档中最小的bag，517,491,443 bytes；下一轮可以只抽取此成员。官方项目说明office真值来自OptiTrack，其他场景来自离线LiDAR SLAM；均仅离线评价使用。market bag未含LiDAR topic不能反推其他场景也没有，定位入口始终需相机/IMU/Wheel白名单。
