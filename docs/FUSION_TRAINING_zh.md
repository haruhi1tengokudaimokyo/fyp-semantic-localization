# 融合阶段的 SPVLoc 多源等预算训练

本模块为 P（ProcTHOR）、Z（ZInD）、P+Z 提供真实的官方 SPVLoc Trainer 接口。它不启动 MAC-VO 训练、不训练 VLM、不租云 GPU。`preflight`、`plan`、`runtime-check` 都不会启动训练。

## 当前就绪状态

2026-11-07 读取本机数据：ProcTHOR 已有 100/20/20 栋通过既有几何与全景验收的数据；ZInD 原始下载状态为 1574/1575，缺少 `zind_spvloc_converted/qualification_manifest.json`。原始下载成功也不能代替转换和定位数据验收。

三组正式对照共同使用 **P 与 Z 的验证建筑**选择 Checkpoint，因此当前三组正式入口均会因 Z 未验收而拒绝训练。P 的既有数据、官方 loader 与权重可以独立执行 CPU 接口检查。单独 P 先导实验如要先跑，必须另存配置，将 `validation_sources` 改成仅 P，并在实验名称中标为先导；其结果不混入采用共同验证集的正式三组对照。

本轮只完成配置、预检、采样和入口测试，没有宣称新的训练或 ZInD 适配已经完成。未经实际训练的入口不等于已通过完整 GPU 训练验收。

实际执行验证：`preflight` 返回上述真实资格状态；官方 ProcTHOR scene 0 数据索引的 137 条原始记录在全景模式下去重为 20 个全景位置；`ckpt_s3d.ckpt` 严格加载全部 56,462,316 参数成功，0 次优化更新。官方 `zind_to_s3d --help` 在现有环境成功，说明命令及依赖可导入，但没有执行转换。`train --arm P+Z` 在资格检查处以退出码 2 拒绝，未创建新 run 目录，也未进入 GPU 查询或训练。

## 固定的公平对照条件

| 条件 | 配置 |
|---|---|
| 初始化 | 官方 `ckpt_s3d.ckpt`；运行前校验 SHA256 |
| 主要预算 | 每组 8575 次 optimizer update，共 343000 次尝试查询 |
| 批量 | FP32、micro-batch 4、梯度累积 10；单 GPU |
| BN | 冻结所有 running mean/variance，保留 affine 参数更新；三组一致 |
| 来源混合 | P+Z 每次 update 各 20 次尝试；P/Z 各 40 |
| 采样 | 来源配额 → 均匀建筑 → 均匀全景位置 → 官方随机视角裁剪 |
| 优化器 | 官方 Adam，初始 LR 0.00025 |
| 学习率 | 第 4116、6174 次 update 乘 0.5 |
| 验证 | 每 343 updates；各建筑最多 8 个固定全景位置，以固定随机种子生成视角 |
| 选模型 | 官方原总损失：先平均建筑内视角、再平均来源内建筑、最后平均来源 |
| 种子 | 20261107 / 20261108 / 20261109；不会自动启动三种子 |
| 网络 | 原始 PerspectiveImageFromLayout，原损失、4 近参考＋1 远参考，语义参考图 |

8575 updates 是本机现有训练工作量附近的首轮等预算对照。配置中额外记录 42000 updates 的延伸参考，**不会自动覆盖主预算**。启用延伸时三组必须统一更改 update、学习率里程碑和验证间隔（如 1000）；另存配置、重新登记实验。

重要差异：原论文按 epoch 调学习率，这里为公平比较不同数据规模改成按 optimizer update 调度；均匀建筑采样、共同验证宏平均、冻结 BN 也属于明确的方法适配。B4×累积10 **不等于**原生 B40 的 BatchNorm。这里保留核心网络和损失，不宣称是原论文硬件、采样和全部训练细节的严格同配置复现。

官方 S3D 初始化已接触 Structured3D，P/Z 表示**本阶段适配数据**，不表示模型从未见过其他来源。P+Z 与单来源组在相同总更新下比较；不是给混合组双倍预算。

## 为什么不能直接拼接图片

官方 ProcTHOR/S3D loader 的 `scenes` 对同一全景可能列出多个透视索引，而 `PERSP_FROM_PANO=True` 实际忽略该透视索引并随机裁剪全景。本适配器先按 `(source, scene, panorama/room)` 去重，防止某全景因派生图片更多获得额外权重。

ZInD 同一住宅可能转换为多个楼层 `scene_HHHHF`。采样与划分以原始住宅 `HHHH` 为建筑，全部楼层归同一个官方 split。ProcTHOR official_house_index 是各官方 split 内部编号，使用 `split:index` 标识，并额外检查 geometry hash，防止把相同几何重新标成其他 split。

官方训练会进一步按语义分布和所在房间筛掉样本。严格保持其核心过滤机制；统计每来源 `attempted`、`passes_official_filter`、`actually_forwarded`。因此只声称等更新、等尝试查询预算，不能把混合前各 50% 声称为过滤后仍各 50%。原版全无效 micro-batch 会回退原 batch，本实现保留并显式统计；正式 pilot 需查看这类比例，异常时不能直接扩大训练。

## 命令

在仓库根目录执行：

```bash
python3 -m research3d.fusion.training --help
python3 -m research3d.fusion.training preflight
python3 -m research3d.fusion.training plan --seed 20261107
```

用现有语义环境做真实 CPU loader/权重接口检查：

```bash
LD_LIBRARY_PATH=/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/fyp_3d/envs/semantic_loc/bin/python -m research3d.fusion.training runtime-check --source procthor
```

它构造一个真实训练场景的官方数据索引、严格加载原模型全部权重，不渲染训练 batch、不调用 optimizer。输出明确保留 `optimizer_steps=0`、`full_training_qualification=false`。

在数据和运行条件满足、明确决定启动时，实际训练命令为：

```bash
LD_LIBRARY_PATH=/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/fyp_3d/envs/semantic_loc/bin/python -m research3d.fusion.training train \
  --config configs/fusion/training_budget.json --arm P+Z --seed 20261107 \
  --output /data/fyp_3d/runs/fusion_training/P_Z_20261107
```

入口先检查三组共同数据验收、源码版本、初始化 SHA、GPU 占用和空间，再调用原版网络的 Lightning Trainer；任一条件缺失即失败，不清理其他实验。输出目录已存在时拒绝覆盖，当前不支持无审计的自动续跑。正式长作业启动前仍需在明确授权的短 pilot 中验证真实 batch、有限损失和实际参数更新；本轮不启动。

空间条件按目标文件系统检查：最低保留 100 GiB，再预留本轮 4 GiB；避免把报告中的 `/data` 旧余量当成当前实际余量。日志、最优/最后 checkpoint、参数变化证据都留在新 run 目录，不更改旧 100/20/20 实验。

## ZInD 官方转换与解锁条件

官方入口已核对本机固定版本源码与 README：

```bash
cd /data/fyp_3d/sources/spvloc
/data/fyp_3d/envs/semantic_loc/bin/python -m spvloc.tools.zind_to_s3d \
  -i /data/fyp_3d/datasets/zind_original \
  -o /data/fyp_3d/datasets/zind_spvloc_converted -idx 0
```

`-idx 0` 只转换一栋；它不是完整训练集。后续分批选择官方 train/val/test 的独立住宅，检查空间后执行。不能通过手工写一个 `DATA_READY` 文件解锁。转换后要实际完成：RGB 解码、相机位置、结构类别、坐标/投影一致性、官方 loader 真实输入检查，并为每项保存证据哈希。验收清单示意：

```json
{
  "status": "DATA_READY",
  "converter": "spvloc.tools.zind_to_s3d",
  "offline_only_gt": true,
  "scenes": [{
    "scene_id": 0, "original_home_id": "0000", "split": "与官方划分一致",
    "checks": {
      "rgb_decode": "PASSED", "camera_pose": "PASSED",
      "structural_semantic_ids": "PASSED", "coordinate_alignment": "PASSED",
      "projection": "PASSED", "official_loader": "PASSED"
    },
    "evidence_files": [{"path": "qa/真实证据.json", "sha256": "实际SHA256"}]
  }]
}
```

上面只是 schema 示意，**不是合格清单**。文件哈希、官方住宅划分、楼层归属和已有 loader 缓存都会重新检查。训练前转换缓存准备应通过专门数据验收流程进行，训练入口不会静默创建或修改原数据 cache。

原论文测试视角应调用 `spvloc.tools.dataset_from_csv` 及官方 `data/datasets/zind/dataset_info.csv`。是否加 `-c` 必须事先固定：官方说明原设置存在 roll/pitch 耦合，`-c` 是修正版。当前本阶段主配置使用 `correct_roll=false` 记录论文原设置；若改修正版，三组同步另立实验。

## 验收边界

- preflight 通过：仅表示证据、路径、split、初始化等静态条件满足。
- runtime-check 通过：仅表示真实 CPU 数据索引和原模型权重兼容。
- 实际 train 完成：必须准确达到更新预算，参数真实改变，有共同验证选出的最优 Checkpoint；状态仍为 `TRAINED_HELD_OUT_POSE_EVALUATION_PENDING`。
- 只有固定未见建筑上的完整定位输出、独立指标复算和失败记录完成后，才可判断多源训练效果。
- 选择验证集使用 P 与 Z，故第三方来源才可作为完全未参与训练和调参的跨来源泛化证据。

官方资料：<https://github.com/fraunhoferhhi/spvloc>，固定源码 `754861115e38bf6f9b6b6b642db095f8e1ff6958`。
