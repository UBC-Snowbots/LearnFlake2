# Training the Follow! gesture on team recordings

URC 2027 Autonomy Q&A 11: the astronaut chooses any gesture using **arm, leg and torso motion only**; no finger or facial gestures. Q10: the astronaut can stand anywhere, so frame the full body. Q12: any suit colours/patterns are allowed. Q6: a gesture earns full Follow! points; voice is a 2/3-point fallback.

## 1. Pick the gesture (team decision)

Big, two-sided, repeated and unlike everyday motion. Default name in the tools: **"two-arm overhead beckon"** — both arms straight up, then swept down to the chest, three times. Avoid single-arm waving: it is the most common false alarm and a public-data "confuser". Record the exact definition in each session's `--gesture-name`.

## 2. Record sessions

Each session = one performer, one setting. Splits are by performer, so record **at least 5 different people** (more is better); test people never appear in training.

```sh
python -m orion_bench.record_gestures record --pose models/yolo11n_pose_640 --source 0 \
  --person alice --suit "white coverall, red sleeves" --distance 6 \
  --output-dir data/gesture_sessions/alice-field-6m
```

The terminal beeps and prints prompts on a timed protocol (default `gesture:4,other:6,confuser:4,other:4` × 5 after a 5 s lead-in). An operator reads them aloud; the astronaut performs them. Segments are labelled automatically: `gesture` = our gesture; `other` = walking, standing, turning, carrying the hammer; `confuser` = lookalikes (waving, stretching, pointing, arm swings, hands up).

Already have phone video? Write `labels.csv` (`start_s,end_s,label`) and run `record_gestures extract --video clip.mp4 --labels labels.csv --person bob --output-dir ...`.

Vary across sessions: distance (3–20 m), facing angle, lighting/time of day, suit, background people, camera height (match the rover's mast camera). Partial-body and walking-while-gesturing clips are worth recording too.

## 3. Build, train, evaluate

```sh
python -m orion_bench.team_dataset --sessions data/gesture_sessions --output-dir data/gesture_v1 \
  --val-people carol --test-people dave,erin --public data/attention_v1
python scripts/train_attention.py --data data/gesture_v1 --output-dir runs/gesture/v1 --window 48 --steps 20000 --seed 0
python scripts/eval_attention.py --run runs/gesture/v1/seed-0 --data data/gesture_v1      # test once
python scripts/export_attention.py --run runs/gesture/v1/seed-0 --output-dir models/gesture_gcn_v1
python -m orion_bench.live_demo --pose models/yolo11n_pose_640 --classifier models/gesture_gcn_v1 \
  --source 0 --device cuda --output-dir runs/live/gesture-v1 --trigger-k 6 --trigger-m 8
```

`--public` adds public clips as negatives only (public hand waving becomes a confuser). Choose threshold/trigger on validation people; score test people once.

## 4. Acceptance before using it for Follow!

Decide numbers before testing, e.g.: on held-out people, ≥ 90% of gesture performances trigger within 3 s, and **zero** triggers during a continuous 10-minute recording of normal activity with other people present. A false Follow! makes the rover drive, so false triggers matter more than misses. Keep voice Follow! as fallback.
