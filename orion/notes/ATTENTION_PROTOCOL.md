# Attention-gesture classifier protocol — 2026-11-07

Declared before any training run is scored.

## Data (examples/attention_datasets.json)

Public PYSKL/OpenMMLab HRNet 2D COCO-17 skeletons, SHA-256 pinned:
- NTU RGB+D 120 (non-commercial research agreement). Official cross-subject split; validation = 9 training subjects (4,17,27,38,49,56,74,84,95), never in test.
- HMDB51 (CC BY 4.0). Official split 1; validation = 15% of training clips grouped by source video.

Classes: `attention` = NTU A23 hand waving + HMDB `wave`. `confuser` = arm/leg motions that are not attention-seeking (clapping, cheer up, kicking, pointing, salute, cross hands, fanning, flick hair, shake fist, hands up, arm circles, arm swings, side kick, stretch; HMDB clap, kick, punch, throw). `other` = every remaining class. No public dataset here contains beckoning or leg waving; these are proxies.

One person per clip (most confident/moving), resampled to 15 fps. Model sees causal 32-frame (~2.1 s) windows.

## Metrics

Clip score = max attention probability over causal sliding windows (stride 4). Primary: attention-vs-rest average precision (AP). Operating point: threshold set on validation so ≤2% of validation non-attention clips fire; report test recall, precision, and false-positive rate per `other` / `confuser`, overall and per source. Three seeds (0,1,2); report each.

## Rules

- Checkpoint, threshold, and every improvement decision use validation only.
- Test is scored once per final configuration; test numbers never drive a change. Iterations are logged with their validation AP even when discarded.
- No claim about YOLO11n-pose inputs, beckoning, leg waving, real rover camera, or Jetson latency follows from these results.
