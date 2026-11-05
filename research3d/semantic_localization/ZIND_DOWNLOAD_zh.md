# 已授权 ZInD 下载（2026-11-05）

用户提供的Bridge凭证已通过真实API验证，并成功取得1575栋住宅的完整清单。凭证只在隐藏终端输入和请求内存中使用，没有放入命令行参数、环境文件、源码、下载日志或Git。取得清单的进程已退出，后台图像下载只使用授权响应中的下载地址，不再需要持久保存Server Token。

- 数据目录：`/data/fyp_3d/datasets/zind_original/`，权限700；授权响应和图片均不上传GitHub。
- 原版源码：`/data/fyp_3d/sources/zind/`，commit `06cfdbf295fa21500a857efe81f80f2243b8fb40`。
- 独立环境：`/data/fyp_3d/envs/zind_download/`，没有修改ACE、MAC-VO或SPVLoc环境。
- 后台服务：`fyp-zind-download.service`，独立于Codex，不使用GPU，不启动训练。

首批官方Train/Val/Test各1栋（0919/0340/0990）已完成，115张全景、6张平面图全部通过官方MD5、实际解码和SHA256记录。后续全量下载正在继续；这不是完整数据集已经下载完成的声明。

## 下载与校验

使用官方同一Bridge API，以及官方 `download_data.py` 的 `process_house` 标注/路径转换和源MD5。大清单用流式解析，两个低优先级下载线程，CPU限额1核、内存限额3GiB。下载支持Range断点恢复；每张图必须通过源MD5和Pillow实际解码，原子提交后才计入验收。网络失败最多3次，失败住宅单独记录，其余住宅继续。

各栋完整收据：`<home_id>/download_receipt.json`；文件SHA256/MD5清单：`FILES_VERIFIED.jsonl`；实时进度：`download_status.json`。全部完成后生成 `DATASET_DOWNLOAD_MANIFEST.json`。保留官方 `zind_partition.json` 划分，未将住宅混入现有ProcTHOR训练。

保证至少100GiB磁盘底线，额外为正在运行的训练保留3GiB检查点余量，下载目录预算45GiB；不足时下载暂停，不删除已验收实验或数据。图片直接落地，不另保留完整压缩包。

```bash
systemctl --user status fyp-zind-download.service
cat /data/fyp_3d/datasets/zind_original/download_status.json
```

服务异常中断后，使用同一入口重新执行会校验并复用已下载文件、续传剩余文件。服务最长72小时，不无限重试，也不恢复Codex巡检。停止状态应结合systemd、`worker_failure.json`与进度判断，不能仅看旧进度文件。

这次仅下载和验证原始授权数据。`DOWNLOAD_VERIFIED`不等于SPVLoc数据转换和几何验收完成；后续还需官方ZInD→Structured3D适配、坐标及投影验收，才可用于训练。数据许可来自用户获批的Bridge/Zillow数据协议，不能把代码Apache许可当成原始数据再分发许可。

官方说明：https://github.com/zillow/zind#download
