# 论文、数据与实现核查

核查日期：2026-10-25。来源均为原论文、作者仓库或数据集官方站点。没有使用第三方代码充当 CVPR 2026 原实现。

## SemRayLoc：复用边界

官方源码版本和权重 SHA256 见 `provenance.json`。网络与核心匹配算法直接导入官方仓库，自主代码只承担数据适配、可复现实验、资源控制、记录与扩展。

RGB 先经 ImageNet 归一化和 ResNet50（layer4 dilation）。深度网络将图像特征沿竖直方向平均得到列 query，使用位置编码和 attention，预测每列128个非均匀深度 bin 的分布；范围0.1–15m，指数 d_hyp=-0.2；深度为分布期望。640px输入宽度产生40列输出。

语义网络另用一个 ResNet50，卷积降维并加入二维位置编码，使用40个 learnable ray query 和1个 room query读取全图特征。ray head输出40×4，room head输出16类S3D或14类ZInD。注意：该 cross-attention 是 query 读取图像特征，和CVPR 2026几何分支与语义分支之间的双向交互不同。

监督不是普通图像语义分割：先由建筑标注生成固定语义楼层图，100px/m。相机位置来自原始camera_pose（S3D原单位mm）或ZInD的floor_plan_transformation与scale_meters_per_coordinate。由地图从GT位姿投射40条射线得到墙/窗/门/未知和距离。S3D将径向距离乘cos列角，生成轴向深度；地图候选atlas保存的是径向深度。RGB-D中家具、人体表面深度不能直接替代此监督。

官方cast_ray还包含门的穿透规则：先记录红色，继续找到墙/窗，若先前红色位置>60px则返回后方结构的距离但标为门。这是原实现行为，不能随意改成“首个红点就截断”。本适配器保留该规则，并使用真实地图随机射线与官方函数逐项核对。

匹配阶段：地图离散0.1m、方向10°（36层）。深度40列经官方插值和cos转换为9根等角射线；语义40列经官方邻域投票变为9根。深度分数exp(-L1/40)，语义按候选地图类别加权不匹配（墙1/窗5/门3/未知0），exp(-penalty/40)，0.6深度+0.4语义融合。候选必须在建筑内部。得分没有经过后验校准。

论文还使用top-k稀疏候选、40根密射线、±5°细化以及可选房间先验。本次自主推理入口先运行官方coarse匹配，没有宣称等价于完整refinement评价。确定性入口语义用argmax，官方eval脚本用multinomial采样；须在正式复现协议中明确这项差异。

## 官方仓库审计到的问题

`official_fixes.patch` 保存已应用修复，未重写网络。

1. Train_models/Train.py GPU devices=0，应为1；S3D数据加载路径缺processed层。
2. Python3.10下旧attrdict不兼容collections.Mapping。用小型递归配置字典替代，仅修正配置访问。
3. eval_localization.py传num_classes给只接受num_ray_classes的语义模型；修正参数名。
4. 语义概率体调用localize时遗漏localize_type='semantic'，默认走深度L1分支；修正为语义分支。
5. ZInD预处理传cast_type并解包4项，但当前官方ray_cast不接受该参数且只返回3项；移除不匹配参数，normal留空。
6. ZInD loader将深度转long造成米级截断；改float32。mask曾用CHW的前两维，得到3×360，修正为360×640。取ref_pose必须copy，避免random yaw修改共享GT。
7. README声明AMP/early stopping/gradient clipping，但训练入口未全部实现。独立train.py明确配置冻结主干、FP16、梯度裁剪，20步smoke不假装运行early stopping。

尚待标准复现前解决：官方ZInD loader在线生成的深度是径向，而S3D监督和get_ray_from_depth的转换约定是轴向；需要以作者确认/标准数据检查统一，不能静默把论文协议改掉。官方refinement也把预测列深度与候选径向距离直接比较；本阶段未启用，以免掩盖坐标问题。官方ZInD验证默认随机yaw、is_train=True，需要明确固定采样；其专用loader还依赖mapped room文件。完整S3D数据尚未取得，因此不能宣称全部官方数据处理/训练/标准评价已通过。

## CVPR 2026：Fusion of Depth and Semantics

已取得CVF原论文，核对第3.2–3.4节。DINOv2输出dense tokens和CLS；地图encoder提供全局floorplan token，加入每列视觉token序列。几何和语义query各自读取，再用双向cross-attention迭代交换信息。它不是只把两个ResNet网络的最终概率相加。

Depth Confidence读取ray token、深度离散分布以及熵，MLP+sigmoid输出c_i。深度仍以建筑GT深度监督；置信度损失为mean(|d_hat-d_gt|*c + λ(1-c)^2)。不能把高熵直接等同于“有人”，也不能把任意entropy heuristic称为论文置信度网络复现。

JSD比较语义概率向量与地图分布：M=(P+Q)/2，JSD=.5KL(P||M)+.5KL(Q||M)。与硬类别匹配相比保留语义模糊性，并允许概率插值。当前weighted_jsd只实现一个可消融的匹配端扩展，沿用SemRayLoc输出；没有实现论文DINOv2联合架构、学习式深度置信度，温度亦未独立调参。

核查CVF原文、作者/标题网络检索与仓库链接后，未找到可核验官方实现。结论是“本次未找到”，不是断言官方代码永远不存在。未下载或运行第三方声称的复现代码。

## F3Loc 与 UnLoc

F3Loc的官方eval_observation.py评估观测似然，eval_filtering.py实际使用prior×likelihood，并调用transit传播概率体。可复用运动模型和histogram filtering接口；后续应先验证odom与图像时间戳、SE(2)坐标和地图分辨率，再比较particle filter。不能先验认定histogram或particle最好。

UnLoc官方实现基于F3Loc，使用冻结Depth Anything V2与显式深度不确定性。utils/localization_utils.py按每根射线Laplace scale归一化深度残差，区分ray flip与no-flip数据约定，并含transit/get_filters。utils/postprocessing.py提供基于Laplace似然的序列位姿修正。它需要自己的backbone/checkpoint/数据和submodule，本轮只审计，没有独立训练或声称运行。

## 正式数据选择

首选Structured3D完整家具版本用于同论文协议训练和静态定位；empty/full可对照静态杂物，但不能把家具对照称为动态人体测试。ZInD用于真实建筑域验证；完整数据需要Bridge批准和token。当前实际数据只选官方仓库sample_tour/000。

Structured3D官网明确须填写agreement form，收到授权后提供下载；没有运行仓库中自动下载9个zip的脚本。该脚本存在并不代表我们已获得授权。ZInD完整下载也没有尝试绕过token；仅使用官方主动发布的单住宅示例。没有用ADE20K替代定位数据。

取得授权后先按官方建筑编号划分，选少量训练/验证/测试建筑；每栋保留对应RGB、原始3D建筑标注、camera_pose和完整地图，不随机按图像切分。不下载全量数据。当前/data空间不足以同时保存全量原始与处理后S3D/ZInD，需要追加或整理用户指定存储，不能擅自删除既有实验数据。

## 动态鲁棒性实施顺序

先比较clean与25%灰矩形遮挡；再加入已知mask过滤，与soft JSD分开消融。矩形不能模拟真实人体外观、接触关系或运动，oracle mask结果也不能宣称检测器性能。

可选person_mask.py复用torchvision官方轻量LR-ASPP，不训练新网络；按人像区域比例减少列射线权重，保留静态建筑结构。全遮挡时应拒绝输出高置信度位姿，固定地图内容不变。检测器的边界误差、Transformer全局特征污染、未检出的移动家具都需要实测；不保证简单加权必然改善定位。

其后再接入Habitat 3.0具有人体的轨迹，记录RGB、相机GT、静态建筑图、动态mask和odom。评价位置/角度/Recall/延迟/失锁率/重定位时间，并按建筑和轨迹分组bootstrap，避免相邻帧夸大置信度。此次没有安装Habitat，避免6GB显存和当前磁盘不足时盲目加载完整场景。

## 原始来源

- SemRayLoc：https://github.com/TAU-VAILab/SemRayLoc；https://arxiv.org/abs/2507.09291
- CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Fusion_of_Depth_and_Semantics_for_Probabilistic_Floorplan_Localization_CVPR_2026_paper.pdf
- F3Loc：https://github.com/felix-ch/f3loc
- UnLoc：https://github.com/matthias-wueest/UnLoc
- Structured3D：https://structured3d-dataset.org/
- ZInD：https://github.com/zillow/zind
- Habitat 3.0：https://aihabitat.org/habitat3/
