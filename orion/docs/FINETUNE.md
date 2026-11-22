# Fine-tuning the pretrained pose-action model

`orion-actions171-base` is the causal ST-GCN (`orion_bench/attention_model.py`, width 48, 48-frame / 3.2 s windows at 15 Hz, COCO-17 keypoints) pretrained to recognise all 171 actions in NTU RGB+D 120 and HMDB51. It is a starting point, not a gesture detector: fine-tune it on team recordings of the chosen Follow! gesture.

Contents: `best.pt` (PyTorch `state_dict`), `summary.json` (training settings, class list, validation/test accuracy, checkpoint SHA-256), `SHA256SUMS`. Training data are HRNet skeletons; the rover uses YOLO11n-pose, so fine-tuning on YOLO-extracted team sessions also adapts to that pose source. NTU RGB+D 120 is non-commercial research data; review terms before non-research use.

## Fine-tune

```sh
sha256sum -c SHA256SUMS
python -m orion_bench.team_dataset --sessions data/gesture_sessions --output-dir data/gesture_v1 \
  --val-people carol --test-people dave,erin --public data/attention_v1
python scripts/train_attention.py --data data/gesture_v1 --output-dir runs/gesture/ft \
  --init path/to/orion-actions171-base/best.pt --freeze-steps 500 \
  --window 48 --width 48 --steps 8000 --lr 0.02 --eval-every 1000 --seed 0
python scripts/eval_attention.py --run runs/gesture/ft/seed-0 --data data/gesture_v1   # test once
python scripts/export_attention.py --run runs/gesture/ft/seed-0 --output-dir models/gesture_ft
```

`--init` loads every layer whose shape matches; the 171-way class head is replaced by a fresh 3-way head (other / gesture / confuser). `--freeze-steps` trains only that head first so random head gradients do not disturb the pretrained backbone; then everything trains at `--lr`. `--window` and `--width` must stay 48. With few team recordings, prefer fewer steps, a lower learning rate, and several seeds; pick checkpoints and thresholds on validation people only.

To continue pretraining instead (e.g. more public data), use `--objective classify` with a dataset built by `orion_bench.temporal_data`.
