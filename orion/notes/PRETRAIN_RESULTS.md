# Pretrained base model — 2026-11-07

**Verdict:** `orion-actions171-base` (1.15M params, causal ST-GCN, width 48, 48-frame windows) recognises 171 actions: **val top-1 76.3%, test top-1 66.8% (top-5 89.1%)**, one seed. Fine-tuning it on the hand-wave proxy reached **val AP 0.71** in 8k steps vs **0.55–0.58** training from scratch for 20k steps (3 seeds). Fine-tune result is validation only; its test split was not scored.

- Data: `examples/pretrain_actions.json` → `data/pretrain_v1`; same pinned NTU RGB+D 120 + HMDB51 HRNet skeletons, splits and validation subjects as `attention_datasets.json`, all 120 + 51 classes. Validation subjects/videos were excluded from pretraining training, so fine-tune validation is not leaked.
- Pretrain: `--objective classify --window 48 --steps 30000`, clip prediction = mean softmax over sliding windows, checkpoint selected on val top-1 (final step). Test scored once. RTX 5080, ~27 min. Run: `runs/pretrain/actions171/seed-0`, checkpoint SHA-256 `b0d298101f00…47cb0`.
- Fine-tune check: `--init` loaded 122 tensors, reinitialised the class head; `--freeze-steps 500`, `--lr 0.02`, 8000 steps. Run: `runs/pretrain/ft-attention/seed-0`. Single seed and different step budget from the scratch runs: indicates transfer, not a controlled comparison.
- GPU: Thoth paused/held jobs 170–172 at Pranav's request and was told to resume after job 173.
- Limits: HRNet skeletons, not YOLO11n-pose; no beckoning/team gesture; NTU non-commercial research terms.
