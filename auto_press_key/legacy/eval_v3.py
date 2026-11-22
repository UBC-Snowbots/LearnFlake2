"""Evaluate Rover-Keypad-Press-v3: one episode per env, first outcome counted (no NaN-prone averaging).

  OMNI_KIT_ACCEPT_EULA=YES python eval_v3.py --headless --baseline            # IK only
  OMNI_KIT_ACCEPT_EULA=YES python eval_v3.py --headless --checkpoint <model_*.pt>

Prints JSON: success / wrong-press / timeout rates, time-to-press, xy error at press, per-key success.
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--baseline", action="store_true", help="IK only (zero residual)")
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keypad_press_v3  # noqa: E402,F401
from keypad_press_v3.agents import KeypadPressPPORunnerCfg  # noqa: E402
from keypad_press_v3.env_cfg import KeypadPressEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

cfg = KeypadPressEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
env = RslRlVecEnvWrapper(gym.make("Rover-Keypad-Press-v3", cfg=cfg))
u = env.unwrapped
N, dev = u.num_envs, u.device

if args.baseline:
    policy = None
else:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    agent = handle_deprecated_rsl_rl_cfg(KeypadPressPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

obs = env.get_observations()
outcome = torch.zeros(N, dtype=torch.long, device=dev)  # 0 pending, 1 success, 2 wrong spot, 3 timeout, 4 slam, 5 overpress
t_done = torch.full((N,), float("nan"), device=dev)
xy_at = torch.full((N,), float("nan"), device=dev)
speed_at = torch.full((N,), float("nan"), device=dev)
key = u.target_key.clone()
step_dt = u.step_dt
with torch.inference_mode():
    for step in range(u.max_episode_length + 2):
        act = torch.zeros(N, 6, device=dev) if policy is None else policy(obs)
        obs, _, dones, _ = env.step(act)
        new = (outcome == 0) & dones.bool()
        if new.any():
            outcome[new & u.success] = 1
            fk = u.last_fail_kind
            outcome[new & u.failed & (fk == 1)] = 2
            outcome[new & u.failed & (fk == 2)] = 4
            outcome[new & u.failed & (fk == 3)] = 5
            outcome[new & ~u.success & ~u.failed] = 3
            t_done[new] = (step + 1) * step_dt
            xy_at[new] = u.last_xy[new]
            speed_at[new] = u.last_speed[new]
        if (outcome > 0).all():
            break

ok = outcome == 1
pct = lambda m: round(100.0 * m.float().mean().item(), 1)  # noqa: E731
R = {
    "mode": "baseline_ik" if policy is None else os.path.basename(args.checkpoint),
    "episodes": N,
    "success_%": pct(ok), "wrong_spot_%": pct(outcome == 2), "slam_%": pct(outcome == 4),
    "overpress_%": pct(outcome == 5), "timeout_%": pct(outcome == 3),
    "time_to_press_s": {"median": round(t_done[ok].median().item(), 2) if ok.any() else None,
                        "p90": round(t_done[ok].quantile(0.9).item(), 2) if ok.any() else None},
    "xy_err_at_press_mm": {"median": round(1000 * xy_at[ok].median().item(), 2) if ok.any() else None,
                           "p90": round(1000 * xy_at[ok].quantile(0.9).item(), 2) if ok.any() else None},
    "tip_speed_at_press_m_s": {"median": round(speed_at[ok].median().item(), 3) if ok.any() else None,
                               "max": round(speed_at[ok].max().item(), 3) if ok.any() else None},
    "per_key_success_%": {str(k): pct(ok[key == k]) for k in range(10)},
}
print("EVAL_JSON " + json.dumps(R))
env.close()
app.close()
