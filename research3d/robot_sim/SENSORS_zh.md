# Gazebo 传感器、共同地图与回放接口

本模块将已有 `scene_03000/procthor_house.json` 转为 Gazebo Harmonic 物理场景。机场只保留为早期接口名称 `airport_hall`；正式运行使用 ProcTHOR 建筑。保留 7 个房间、22 面去重且非空墙、6 个门洞和 4 个窗洞。墙洞按原始 `holePolygon` 切分，`empty` 墙保持开放；房间地板按原始正交多边形分解。原始家具 Mesh 未导入，墙地纹理为程序生成，因此不能称 AI2-THOR 的外观复现，也不能用此场景代替原论文 Benchmark。

源文件 SHA256：`9e6b74a4c6b640c5c70013ed3701e2109de62790a0c788bed1a069e5710cfcc9`。原始场景属于以前的验证建筑，当前实验仅为共同地图系统开发，不构成未见建筑泛化验证。

## 启动

在仓库根目录执行，输出必须使用新的独立目录。不要覆盖正在使用的世界文件。

```bash
python3 -m research3d.robot_sim.world \
  --house /data/fyp_3d/datasets/procthor_spvloc/scene_03000/procthor_house.json \
  --output tmp/my_new_robot_world

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=71 GZ_PARTITION=fyp_robot_sim_20261107
gz sim -s -r --headless-rendering "$PWD/tmp/my_new_robot_world/airport_hall.sdf"
```

另一个已加载同样 ROS 环境和变量的终端启动桥接：

```bash
/usr/bin/python3 -m research3d.robot_sim.sim_bridge --offline-gt
```

在线部署不需要 `--offline-gt`。这个开关只允许独立采集器/评价器获取 `/robot/ground_truth`，定位节点不得订阅。桥接器默认白名单不包含任何 DiffDrive odometry 或 TF。按 Ctrl-C 只结束该桥接及其自己的进程组；不使用全局 `pkill`。

若用户 shell 的启动脚本影响 ROS，可在 `PATH=/usr/bin:/bin:/opt/ros/jazzy/bin bash --noprofile --norc` 子 shell 中加载 ROS 后运行。实际 Gazebo/ROS 网络发现需要主机本地通信权限；受限沙箱的 `getifaddrs` 错误不是算法故障。

## 数据接口

| Topic | ROS 类型 | 用途 |
|---|---|---|
| `/robot/joint_states` | `sensor_msgs/JointState` | 实际左右轮关节位置和角速度；不是世界真值里程计 |
| `/robot/imu` | `sensor_msgs/Imu` | 100 Hz；在线融合只使用陀螺仪角速度，不使用模拟器绝对朝向 |
| `/robot/camera/image` | `sensor_msgs/Image` | 左目 RGB，640×480，10 Hz |
| `/robot/camera_right/image` | `sensor_msgs/Image` | 右目 RGB，同参数，基线 0.10 m |
| `/robot/camera/camera_info`、右目对应 Topic | `sensor_msgs/CameraInfo` | 实际运行时标定 |
| `/robot/cmd_vel` | `geometry_msgs/Twist` | 物理差速轮速度指令 |
| `/robot/ground_truth` | `tf2_msgs/TFMessage` | 仅建图监督与离线评价；选择 child=`robot`、parent=`airport_hall` |

轮半径 0.10 m，轮距 0.42 m。`base_link` 是轮轴高度处的机器人本体，初始实际高度约 0.10 m。左相机在 base `(0.20,0,0.50)`，右相机在 `(0.20,-0.10,0.50)`。optical 轴是右、下、前，完整刚体外参保存在 manifest。Unity 源坐标位置 `(x,y,z)` 转为 Gazebo `(x,z,y)`，相机姿态由 Gazebo 原生世界姿态和 optical 外参计算，禁止直接交换 Unity 四元数分量。

实际 CameraInfo 验证：`fx=320.0, fy=320.00000953674316, cx=320, cy=240`，零畸变。左右相机使用相同仿真时钟；采集器只接受纳秒时间戳完全相同的图像对。

## 已完成的传感器验收

2026-11-07 已在本机实际运行 Gazebo 8.15.0、ROS Jazzy 和 ros_gz_bridge。机器人前后无摩擦支撑轮修复了早期原型的前倾问题，静止基座俯仰接近零。

严格同步的三组双目参考图像均满足：左右时间戳一致，左右图像哈希不同。SIFT 只用于独立传感器几何检查：极线内点分别 76、137、122；竖直误差中位数分别 0.0236、0.0146、0.0198 像素；水平视差中位数为正，分别 4.31、4.72、7.77 像素，符合右目正基线。这不是定位实验结果。

证据位于本次运行目录 `tmp/robot_sim_procthor_stereo/`：`map_manifest.json`、`stereo_sensor_validation.json`、`smoke_capture/reference/frames.jsonl`、`gazebo.log` 和 `bridge.log`。

## 建图与查询隔离

`reference_poses.json` 给出 192 个预先指定的独立建图视点。建图可调用 Gazebo `set_pose`，但标签必须来自实测世界位姿；每次位姿改变后要求实测 GT 连续稳定至少 150 ms，GT 采样间隔不超过 30 ms，图像与 GT 时间差不超过 15 ms。不能仅按服务请求里的目标坐标直接标注。

查询通过物理轮驱连续运行。在线 `query/frames.jsonl` 只有图像、标定、时间和已知地图绑定；真实查询位姿单独写入 `offline_gt/query_poses.jsonl`。右目字段固定为 `right_image_path`、`right_image_sha256`、`right_timestamp_ns`。相机外参、地图哈希和双目基线在 map manifest 中冻结。

## 三前端回放

`replay_models.py` 消费真实 MAC-VO、ACE、SPVLoc 的 `predictions.jsonl`。MAC-VO 必须覆盖全部连续查询并保留 lost，ACE/SPVLoc 使用同一预先声明的采样子集。适配器拒绝不同地图、时间、图像校验值和标定的混用；不会补造预测、插值 lost 或用 GT 选择候选。

```bash
python3 -m research3d.robot_sim.replay_models \
  --dataset /absolute/path/to/capture \
  --macvo /absolute/path/to/macvo_run \
  --ace /absolute/path/to/ace_run \
  --spvloc /absolute/path/to/spvloc_run \
  --output /absolute/path/to/new_replay \
  --ground-truth /absolute/path/to/capture/offline_gt/query_poses.jsonl
```

输出 A–E 五组，每组两种明确隔离的时间口径：`offline_accuracy` 仅评价冻结预测的准确率，假定理想可用时间；`measured_service_fifo` 用真实测得的每次推理时间做单资源 FIFO 回放，保留计算积压，未计模型切换和未测 I/O，所以仍只是延迟下界，不能称实时部署。先完成不读 GT 的回放并保存，再开启独立离线指标计算。A 没有全局地图原点，绝对定位应为空，局部 ATE/RPE 由独立 MAC-VO 评价报告，不能偷偷用 GT 给 A 初始化。

互补性统计只用 ACE/SPVLoc 相同采样查询；系统成功覆盖率使用全部连续帧作为分母。没有单独记录的失跟事件时，不报告恢复成功率。当前工具实现和测试通过，不等于三个真实模型的共同场景结果已完成。
