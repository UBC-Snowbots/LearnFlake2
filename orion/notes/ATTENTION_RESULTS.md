# Attention-gesture classifier — first results, 2026-11-07

**Verdict:** a 1.1M-parameter causal ST-GCN finds hand waving among 120 lab actions (NTU test AP ≈ 0.70) but is weak on in-the-wild video (HMDB test AP ≈ 0.15). Not deployable yet: it was trained on HRNet skeletons, not YOLO11n-pose output, and has never seen beckoning, leg waving, or rover footage. Protocol: `notes/ATTENTION_PROTOCOL.md` (declared before scoring).

## Data

`python -m orion_bench.temporal_data --spec examples/attention_datasets.json --raw-dir data/raw/pyskl --output-dir data/attention_v1` downloads and SHA-256-verifies PYSKL HRNet 2D skeletons for NTU RGB+D 120 and HMDB51, then builds single-person 15 fps sequences. Spec SHA-256 `fdcaa3e9…a7de`.

| Split | Clips | attention | confuser | other |
|---|---:|---:|---:|---:|
| train | 55,385 | 609 | 6,349 | 48,427 |
| val | 11,208 | 131 | 1,285 | 9,792 |
| test | 52,447 | 304 | 6,070 | 46,073 |

NTU: official cross-subject test; validation = 9 held-out training subjects. HMDB: official split 1; validation = 15% of training videos. NTU license is non-commercial research only.

## Model

`orion_bench/attention_model.py`: 6-block spatio-temporal graph convolution over the COCO-17 skeleton, causal (past-only) temporal convolutions, features = body-centred position, confidence, velocity, bone vectors. 3 classes (other / attention / confuser). Trained with class-balanced sampling, rotation/scale/flip/keypoint-dropout augmentation (`scripts/train_attention.py`).

Clip score = max attention probability over sliding windows. Threshold frozen on validation at ≤2% non-attention firing.

## Results (test scored once per run, after selection on validation)

| Config | Seed | Val AP | Test AP | Test recall | Test precision | FPR other | FPR confuser | NTU AP | HMDB AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 0.422 | 0.581 | 0.812 | 0.214 | 0.015 | 0.036 | 0.634 | 0.086 |
| Baseline | 1 | 0.399 | 0.619 | 0.812 | 0.236 | 0.013 | 0.035 | 0.716 | 0.050 |
| Baseline | 2 | 0.417 | 0.561 | 0.796 | 0.213 | 0.014 | 0.040 | 0.608 | 0.151 |
| Improved | 0 | 0.553 | 0.647 | 0.793 | 0.207 | 0.015 | 0.041 | 0.700 | 0.151 |
| Improved | 1 | 0.584 | 0.658 | 0.842 | 0.225 | 0.012 | 0.055 | 0.707 | 0.166 |
| Improved | 2 | 0.576 | 0.641 | 0.829 | 0.232 | 0.011 | 0.055 | 0.690 | 0.152 |

Baseline: 32-frame window (2.1 s), 12k steps. Improved: 48-frame window (3.2 s), 20k steps. Both changes were made together; the gain cannot be attributed to either one alone. Selected on validation AP (baseline 0.40–0.42 → improved 0.55–0.58).

Iteration log (validation only, seed 0, early-stage): 2× confuser sampling gave lower val AP (0.27 at 5k steps) and was stopped; width 64 gave 0.34 at 2.5k steps and was stopped for time. A 20k-step 32-frame run and the first w48 seed-2 attempt died from GPU out-of-memory while four jobs shared the GPU; evaluation batch was lowered to 512 and seed 2 rerun. Failed logs are kept under `runs/attention/`.

Error analysis (baseline, validation): most false alarms are hand-near-body gestures that 17 COCO keypoints cannot separate from a wave — pointing, taking a selfie, shaking a fist, brushing teeth, fanning, OK/thumb signs — plus HMDB movie clips.

## Compute

TorchScript export `models/attention_gcn_w48_s0/` (manifest-bound, export error 0). Orion CPU benchmark, input [1,48,17,3], 1 thread, desktop Ryzen 9950X while training jobs loaded the machine: **7.7 ms mean, 8.3 ms P95** (`runs/attention-cpu-reference/20261107T103331Z-48c5bb56`). Not a Jetson measurement.

## What it does not show / next steps

1. **Pose source mismatch:** train on YOLO11n-pose keypoints extracted from the same videos (or confirm transfer) before trusting any rover result.
2. **Missing gestures:** no public set here has beckoning or leg waving. Record team clips (positives + everyday negatives, different people/lighting/distances) and keep them as a separate held-out test.
3. **In-the-wild weakness:** HMDB AP ≈ 0.15. Multi-person, moving camera, partial bodies are the realistic rover case.
4. Precision at the 2%-FPR operating point is ~0.2 because negatives outnumber positives ~150:1; a rover trigger needs temporal smoothing (e.g. N consecutive windows) and per-track debouncing, evaluated on continuous video.
