# FYP — SPVLoc semantic indoor localization

Current repository: https://github.com/haruhi1tengokudaimokyo/fyp-semantic-localization

## 2026-11-07: fusion-stage experiment setup

Branch `research/fusion-stage` adds a [known-map localization and recovery protocol](docs/FUSION_STAGE_zh.md), genuine synchronous stereo collection, causal candidate replay, offline complementarity/recovery metrics, and [equal-budget SPVLoc P/Z/P+Z training entry points](docs/FUSION_TRAINING_zh.md). Three development buildings and five new, sealed system-test building identities are fixed separately from the completed experiments.

This is implementation and configuration validation, **not a completed fusion benchmark**. Shared-map ACE heads and three-frontend trajectories remain pending; raw ZInD has not passed SPVLoc conversion acceptance, so the formal multi-source training fails closed. No new training, cloud rental, or background monitoring was started. Existing ACE/MAC-VO/SPVLoc results below remain unchanged.

## 2026-11-06: corrected full training and paired evaluation completed

The corrected run finished normally on Nov 5 at 23:31 (Asia/Shanghai): **25 epochs / 8,575 updates**, all 56,462,316 trainable parameters updated, approximately 49.57 hours including validation. The checkpoint was selected at epoch 19 using validation loss only. The one-shot training service has exited successfully; no retraining was started.

Both models were evaluated on the same **2,752 eligible queries across 20 unseen ProcTHOR buildings**. With one original refinement iteration, the fine-tuned model achieves **54.22% at <1m / <30°**, **0.464m** median 3D translation error and **4.29°** median full rotation error. Official pretrained weights achieve **22.06%**, **3.376m**, **82.64°**. Every saved pose error was independently recomputed. Tail failures remain substantial: fine-tuned translation P90 is 8.26m; this is not a reliable deployment claim or the original paper benchmark.

Detailed Chinese report, 25-epoch curves, per-query/per-building CSV, raw training metrics and checkpoint provenance: **[completed training results](results/semantic_training_20261106/TRAINING_RESULTS_zh.md)**. Model and original pose evidence: [results-20261106-spvloc release](https://github.com/haruhi1tengokudaimokyo/fyp-semantic-localization/releases/tag/results-20261106-spvloc). Pure-geometry/semantic contribution ablations and the full paper evaluation coverage remain outstanding. VLM remains paused, and accepted ACE/MAC-VO results are preserved.

Sections dated Nov 2–3 below preserve historical handoff context and do not describe current running jobs.

## 2026-11-03 evening: supervision fix, corrected full training

The first full training attempt was **invalid**: the current torchvision default float-to-PIL conversion wrapped semantic classes 2–5 into values near 1 and quantized metre depths / signed normals. Shape-only and finite-value startup checks missed this. The original validity metric was zero throughout the attempt. This was a training input conversion problem; the accepted raw dataset remains valid.

The explicit scalar **PIL F-mode adapter** now preserves all rendered semantic, depth and normal targets exactly. Eight real queries across four train/val/test houses passed pre/post-transform comparisons with zero numerical error; the actual GPU batch also passes target-value assertions. RGB transforms, original network, criterion and sampling remain unchanged.

The user authorized deletion of invalid runs: the first full attempt and both earlier small SPVLoc training attempts/checkpoints have been deleted (4.76 GB). Their semantic-training conclusions are withdrawn. A compact failure/cleanup audit remains. ACE/MAC-VO results and release assets are unaffected.

The user has now authorized full training. **100/20/20 buildings, 22,400 raw views and 2,800 panoramas are DATA_READY**; all 185,500 accepted files were independently SHA256 rechecked, with 140 unique house geometries and zero rejected candidates. The following Nov 2 section is historical.

`fyp-spvloc-140-train-v2.service` runs the corrected original SPVLoc network/losses for 25 epochs on the frozen full split, freshly initialized from official weights. It uses original random panorama query crops (45–135° FoV and yaw/pitch/roll; upstream-documented paper roll convention), FP32 Batch4 with accumulation10. The 20 validation buildings select the checkpoint; the 20 test buildings are evaluated afterwards, including original pose refinement. The official pretrained baseline then uses identical test queries. A separate CPU process recomputes saved pose errors against offline-only GT before accepting each evaluation.

Run directory: `/data/fyp_3d/runs/semantic_localization_20261103/spvloc_100_20_20_float_targets_v2/`. Progress includes the original validity-filter fraction and actual GPU target ranges. Best/last epoch checkpoints and a rolling 100-update recovery checkpoint are retained. Evidence: [results/semantic_training_20261103](results/semantic_training_20261103).

The earlier 9/3/3 result is withdrawn as evidence of correct semantic training because it used the same defective conversion. VLM remains paused.

## Historical instruction: 2026-11-02 evening

Focus on **SPVLoc**, using our own independently split ProcTHOR houses. **VLM experiments are paused**. Tonight's new worker prepares and validates data only; **it never launches full-scale training**. The existing 9-train/3-val/3-test model continues its original 25-epoch train/evaluate pipeline without restart. Tomorrow's full-scale run is a separate decision.

The target dataset contains **at least 100 training, 20 validation and 20 test buildings**, around 160 genuinely rendered perspective views and 20 RGB panoramas per building: **22,400 raw views / 2,800 panoramas** before the original SPVLoc image filters. These are synthetic independent house geometries, not 140 real captured buildings.

At this handoff, **16 houses / 2,560 views / 320 panoramas are captured and geometrically qualified**. The complete 140-house dataset is pending. The worker has frozen 210 publisher-ordered candidates, including bounded same-split reserves; it must reach all three acceptance quotas before marking `DATA_READY`. No test localization score is used to accept or replace data.

## Data preparation

Implementation and detailed protocol: [research3d/semantic_localization/README_zh.md](research3d/semantic_localization/README_zh.md).

```bash
# Read-only preflight; no GPU generation or training.
/data/fyp_3d/envs/semantic_loc/bin/python research3d/semantic_localization/prepare_semantic_overnight.py --preflight-only

# Existing enabled, one-shot user service; runs independently of Codex.
systemctl --user status fyp-semantic-data-140.service
```

The worker first waits for `fyp-spvloc-16-pipeline.service` and other CUDA compute users to finish. It then captures, independently checks and hashes one building at a time. Fixed geometry/semantic rejection thresholds remain unchanged. At least 100 GiB free space is protected; new dataset growth is limited below 60 GiB. ACE/MAC-VO files are never removed.

Final reports will be under `/data/fyp_3d/runs/semantic_localization_20261102/data_100_20_20/`, including `DATASET_SCENE_MANIFEST.json`, `DATASET_DOWNLOAD_AUDIT.md`, `SEMANTIC_CLASS_DISTRIBUTION.csv`, `DATASET_STORAGE_REPORT.md`. Pending preparation is not a completed training result.

## Method and reproducibility limits

We use the [official SPVLoc implementation](https://github.com/fraunhoferhhi/spvloc), including its network, complete original losses, Adam/scheduler, map rendering, global pose estimates and refinement. The dataset adapter converts actual source floor/wall/door/window topology into its original Structured3D representation, merges open-plan empty separators and retains exact camera calibration. It does not replace localization with retrieval or segmentation.

This is a **ProcTHOR adaptation**, not reproduction of the original Structured3D benchmark. Native perspective training is used by the currently running small dataset. Panoramas are now independently crossprojected against genuine native queries; the upstream random panorama-crop training loader has also produced a real, finite original training item. Full-scale training has not started. GTX1660SUPER uses FP32 and Batch4 with accumulation10, rather than the paper's native Batch40; BatchNorm and filtering make these numerically different.

The earlier SPVLoc training attempts were invalidated and deleted as described above. The corrected experiment retains a frozen split and validation-only checkpoint selection.

## Preserved accepted ACE / MAC-VO results

Compact summaries: [results/reproduction_20261102](results/reproduction_20261102). Complete verified result archives, each below 500 MB, are attached to the repository's `results-20261102` release.

| Experiment | Accepted evidence |
|---|---|
| ACE | Seven scene-specific original 8M-buffer models, 17,000 official test poses; macro 5cm/5° recall **80.5221%**, original D-SLAM. Comparable paper reference **80.8%**, not 97.1% Pseudo-GT. |
| MAC-VO demo | All 1,274 supplied frames; Paper FP32 ATE **0.2266 m**. Not established as a complete parent benchmark sequence. |
| Hospital v1 Easy P000 | All 563 frames; Paper FP32 ATE **0.1547 m**, identity-covariance full-pipeline ablation **0.4337 m**; 0 lost. |
| AbandonedFactory v1 Easy P000 | All 2,176 frames; Paper FP32 ATE **2.7301 m**, identity-covariance full-pipeline ablation **4.8791 m**; 1 lost. |

MAC-VO's repeated original FP32 controls match their existing trajectories; 562 Hospital optimizer jobs have exact captured-input/output replay acceptance. Static-initialization backend replays are separate from full-pipeline ablations. About 1.3 s/frame remains a hardware limitation; no unaccepted FP16 speedup is claimed. Full paper benchmark coverage remains incomplete. GT is used offline for evaluation, with no scale correction or dropped-frame interpolation.

Earlier projects, retirement decisions and historical results are preserved in [README_legacy.md](README_legacy.md) and existing source directories. They are not reactivated by this data-only worker. No Codex heartbeat or periodic report timer is restored.
