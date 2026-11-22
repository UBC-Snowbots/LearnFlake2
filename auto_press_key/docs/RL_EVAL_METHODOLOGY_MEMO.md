# RL Evaluation Methodology — What the Success Numbers Actually Measure

**Ptolemy / arm autonomy — internal memo**
**Author:** Pranav (Ptolemy lead) · **Date:** 2026-09-26

## Purpose

Two arm-autonomy RL results are circulating with very different headline numbers:

- **HierarchicalSAC (RoboSuite)** — ~**99.7%** "success" (custom implementation, earlier LearnFlake work)
- **Keypad-Reach (Isaac Lab)** — **93%** of arms within 5 cm of the target key

Before either number is cited as a benchmark, this memo documents **what each one measures**, with file:line evidence from the repo. The short version: **the two numbers are not comparable**, and the higher one is measured under conditions that do not reflect the real task.

## TL;DR

| | HierarchicalSAC (RoboSuite) | Keypad-Reach (Isaac Lab) |
|---|---|---|
| Task actually run | RoboSuite **cube-lift** | **Keypad reach** (rover2026 arm → target key) |
| Target location given to policy? | **Yes — exact ground-truth coords every step** | No — must reach a key by pose, not fed as reward-free oracle |
| Perception in the loop? | **No** (`use_camera_obs=False`; camera is a cosmetic overlay) | N/A yet (reach policy); real YOLO path exists separately |
| Success metric | RoboSuite's built-in **4 cm lift** heuristic | EE within **5 cm** of the key's 3D press-point |
| Success rate emitted by the code? | **No** — pipeline prints reward/height only | Yes — eval computes within-2cm / within-5cm fractions |

**Conclusion:** the ~99.7% is **not** a perception-to-action success rate. It reflects a policy driving to **oracle object coordinates** it is handed each timestep, on the **wrong task** (cube-lift, not keypad), scored — if at all — by a lenient 4 cm-lift heuristic. It should not be compared against a perception-honest reach metric.

## Evidence

### 1. The policy is fed ground-truth object coordinates (the "reader")

The observation vector is built from exact simulator state, not perception:

- `env_wrapper.py:91` (old LearnFlake repo) — `use_camera_obs=False` (no pixels enter the pipeline)
- `env_wrapper.py:184` — obs keys include `'cube_pos'` and `'gripper_to_cube_pos'`
- `env_wrapper.py:265-282` — those raw values are concatenated straight into the policy input, consumed by `agent.get_action()` (`trainer.py:315`)
- Source of `cube_pos` — `external_pkgs/RoboSuite/.../lift.py:372`:
  ```python
  def cube_pos(obs_cache):
      return np.array(self.sim.data.body_xpos[self.cube_body_id])  # MuJoCo ground truth
  ```

The agent is handed the precise 3D location of the target every timestep. It never has to *find* the object — only move to a coordinate it already knows. This is the mechanism that inflates performance.

### 2. No success rate is actually computed

- `trainer.py:357-361` — eval prints only reward (`R=`), max height (`MaxH=`), recoveries. No success threshold, no per-episode pass/fail tally.
- `env_wrapper.py:638-661` — `info` never contains an `is_success` flag.

So the 99.7% figure does not originate from the code; there is no artifact in the repo that produces it.

### 3. The only implicit "success" is a 4 cm lift

- `external_pkgs/RoboSuite/.../lift.py:433-444` — `_check_success()` = cube lifted 4 cm above the table. The implementation's own reward milestones fire at the same 4 cm / 8 cm thresholds (`env_wrapper.py:438-446`).

### 4. The camera is cosmetic

That eval opens an eye-in-hand view (`trainer.py:262-346`), but the pixels are drawn as a `cv2` overlay and **never encoded or passed to the policy** — which still runs on the ground-truth `cube_pos` vector. It looks like perception; it is not.

## A fair word on the HierarchicalSAC work

This is not a claim that the work is worthless. It is a real, substantial HierarchicalSAC implementation (skill selector + skill-conditioned actor + double critic) and a working RoboSuite training loop. The problem is **narrowly the evaluation**: (a) it runs cube-lift, not the keypad task; (b) it consumes privileged ground-truth object pose instead of perception; (c) it reports reward, not a success rate, so "99.7%" has no traceable definition. Those are fixable — but until they are, the number cannot stand next to a perception-honest metric.

## Recommendation — a common, honest eval

To compare arm-autonomy approaches fairly, agree on one evaluation protocol:

1. **Same task:** keypad code-entry (the actual URC objective), not cube-lift.
2. **Same success definition:** EE reaches the key press-point within a fixed tolerance (propose ≤ 2 cm for a "press", ≤ 5 cm for "reach"), tallied per episode as pass/fail, reported as a rate over N randomized trials.
3. **Declared observation tier.** State explicitly whether the target pose is (a) ground-truth/privileged, (b) from perception (YOLO→pose), or (c) noised. A privileged-state result is a useful *upper bound*, but must be labeled as such — never reported as end-to-end success.
4. **Perception in the loop for the headline number.** The number that goes in a report should route through the real YOLO→pose path (`rl_agent_pranav/keyboard_stack_v2/sim_perception_eval.py`), not oracle coordinates.

Under a shared protocol, the current standings are: the Isaac keypad-reach policy (93% within 5 cm, on the real arm, target not handed to it as an oracle in the reward) measures something far closer to the real problem than the ~99.7% cube-lift figure — even though its headline number is lower.

---
*Evidence gathered from the old UBC-Snowbots/LearnFlake repo (MuJoCo/RoboSuite work, kept there as is) at its commit state of 2026-09-26. File names above refer to that repo and are reproducible via `git blame` / grep there.*
