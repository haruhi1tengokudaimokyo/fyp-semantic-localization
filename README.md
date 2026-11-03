# FYP — SPVLoc semantic indoor localization

Current repository: https://github.com/haruhi1tengokudaimokyo/fyp-semantic-localization

## 2026-11-03: accepted data and full training

The user has now authorized full training. **100/20/20 buildings, 22,400 raw views and 2,800 panoramas are DATA_READY**; all 185,500 accepted files were independently SHA256 rechecked, with 140 unique house geometries and zero rejected candidates. The following Nov 2 section is historical.

`fyp-spvloc-140-train.service` runs the original SPVLoc network/losses for 25 epochs on the frozen full split, initialized from official weights. It uses the original random panorama query crops (45–135° FoV and yaw/pitch/roll; upstream-documented paper roll convention), FP32 Batch4 with accumulation10. The 20 validation buildings select the checkpoint; the 20 test buildings are evaluated afterwards, including original pose refinement. The official pretrained baseline then uses the identical test queries. A separate CPU process recomputes saved pose errors against offline-only GT before accepting each evaluation.

Run directory: `/data/fyp_3d/runs/semantic_localization_20261103/spvloc_100_20_20_paper_pano/`. Progress and timing are recorded by training callbacks, without Codex monitoring. Best/last epoch checkpoints and a rolling 100-update recovery checkpoint are retained. No previous ACE, MAC-VO or small SPVLoc result is overwritten. Evidence: [results/semantic_training_20261103](results/semantic_training_20261103).

The previous 9/3/3 perspective-mode run completed all 25 epochs and 401 test queries: refined median translation **0.896 m**, median rotation **5.55°**, success at 1m/30° **49.13%**. It is a separate small-data adaptation, not the new full-scale result. VLM remains paused.

## Current instruction: 2026-11-02 evening

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

The initial 3-building / 100-update pilot worsened held-out localization; it is retained as a failed small-data result, not presented as successful convergence. The larger running experiment retains a frozen split and validation-only checkpoint selection.

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
