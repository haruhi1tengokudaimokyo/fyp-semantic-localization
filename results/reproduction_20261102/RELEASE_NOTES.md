Verified ACE and MAC-VO results. Original accepted experiments were not retrained.

- ACE: all seven scene-specific original 8M-buffer heads and 17,000 official test poses. Macro recall 80.5221% at 5cm/5°, original D-SLAM condition (paper reference 80.8%; 97.1% is a different Pseudo-GT condition).
- MAC-VO: author 1,274-frame demo; full Hospital v1 Easy P000 (563 frames), AbandonedFactory v1 Easy P000 (2,176 frames), complete original FP32 and identity-covariance controls, independent metrics, exact solver replay acceptance and stage timings. The repeated original controls are bit-identical to the existing full trajectories.
- Hospital ATE: original 0.154737 m; full-pipeline identity covariance 0.433731 m. Factory ATE: original 2.730054 m; identity covariance 4.879089 m.
- MAC-VO is FP32 adapted to GTX1660SUPER, approximately 1.3 s/frame. These v1 trajectories are not the full original paper benchmark; static-initialization backend replay is separately marked. No unaccepted speed improvement is claimed.

Both archives contain internal per-file SHA256 checksums and were fully CRC/SHA256 verified after writing. Raw datasets, environments and dense correspondence caches are excluded; original source data and accepted outputs remain untouched on the host. ACE includes the seven trained scene heads.

| Archive | Bytes | SHA256 |
|---|---:|---|
| ACE_7Scenes_results.zip | 29516263 | ef8381c18d84f2ff977f42afbb3e1c977524c43f8b2fdd96602c91ebc7384bd0 |
| MACVO_full_sequence_results.zip | 53847021 | 129712217c57565068e6b8570a14231394c6685f53b21e653607872c7455ac5e |

SPVLoc code and the initial real data/training evidence are in the main branch. The new data-only worker targets 100/20/20 ProcTHOR houses; it waits for the existing small training/evaluation to finish, prepares and accepts the dataset, then exits. It does not start full-scale training. VLM experiments are paused.
