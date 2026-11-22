"""Evaluate Rover-Keypad-Type-v4: one full episode (a 6-digit code) per env.

  OMNI_KIT_ACCEPT_EULA=YES python eval_v4.py --headless --baseline            # IK only
  OMNI_KIT_ACCEPT_EULA=YES python eval_v4.py --headless [--checkpoint <model_*.pt>]   # default: models/model_2499.pt

Prints JSON: full-code success, digits typed, how episodes ended, per-digit press quality, and motion
smoothness (measured joint speed / acceleration vs the configured limits).
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--checkpoint", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_2499.pt"))
parser.add_argument("--baseline", action="store_true", help="IK only (zero residual)")
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keypad_type_v4  # noqa: E402,F401
from keypad_type_v4.agents import KeypadTypePPORunnerCfg  # noqa: E402
from keypad_type_v4.env_cfg import KeypadTypeEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

cfg = KeypadTypeEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
env = RslRlVecEnvWrapper(gym.make("Rover-Keypad-Type-v4", cfg=cfg))
u = env.unwrapped
N, dev = u.num_envs, u.device

policy = None
if not args.baseline:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    agent = handle_deprecated_rsl_rl_cfg(KeypadTypePPORunnerCfg(), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

obs = env.get_observations()
active = torch.ones(N, dtype=torch.bool, device=dev)
end = torch.zeros(N, dtype=torch.long, device=dev)  # 1 code done, 2 wrong spot, 3 slam, 4 overpress, 5 timeout
digits = torch.zeros(N, dtype=torch.long, device=dev)
t_end = torch.zeros(N, device=dev)
press_xy, press_speed = [], []
peak_qd = torch.zeros(N, 6, device=dev)
peak_qdd = torch.zeros(N, 6, device=dev)
jid = u.joint_ids
prev_qd = u._robot.data.joint_vel.torch[:, jid].clone()
dt = u.step_dt
with torch.inference_mode():
    for step in range(u.max_episode_length + 2):
        act = torch.zeros(N, 6, device=dev) if policy is None else policy(obs)
        obs, _, dones, _ = env.step(act)
        qd = u._robot.data.joint_vel.torch[:, jid]
        m = active & ~dones.bool() & (step * dt > 0.5)  # skip reset settling (arm sags onto its targets)
        peak_qd[m] = torch.maximum(peak_qd[m], qd[m].abs())
        peak_qdd[m] = torch.maximum(peak_qdd[m], ((qd - prev_qd) / dt)[m].abs())
        prev_qd = qd.clone()
        p = active & u.pressed_now
        press_xy.append(u.last_xy[p]); press_speed.append(u.last_speed[p])
        digits += p.long()
        new = active & dones.bool()
        if new.any():
            fk = u.fail_kind
            e = torch.full_like(end, 5)
            e = torch.where(u.code_done, 1, torch.where(u.failed & (fk == 1), 2, torch.where(
                u.failed & (fk == 2), 3, torch.where(u.failed & (fk == 3), 4, e))))
            end[new] = e[new]
            t_end[new] = (step + 1) * dt
            active &= ~new
        if not active.any():
            break

press_xy = torch.cat(press_xy)
press_speed = torch.cat(press_speed)
ok = end == 1
pct = lambda m: round(100.0 * m.float().mean().item(), 1)  # noqa: E731
q = lambda x, p: round(x.quantile(p).item(), 3) if len(x) else None  # noqa: E731
vmax = torch.tensor(cfg.max_joint_vel, device=dev)
amax = torch.tensor(cfg.max_joint_acc, device=dev)
R = {
    "mode": "baseline_ik" if policy is None else os.path.basename(args.checkpoint),
    "episodes": N, "code_length": cfg.code_length,
    "full_code_%": pct(ok),
    "digits_typed": {"mean": round(digits.float().mean().item(), 2), "median": int(digits.median().item())},
    "ended_by_%": {"code_done": pct(ok), "wrong_spot": pct(end == 2), "slam": pct(end == 3),
                   "overpress": pct(end == 4), "timeout": pct(end == 5)},
    "time_per_digit_s_median": round((t_end[ok] / cfg.code_length).median().item(), 2) if ok.any() else None,
    "press_xy_err_mm": {"median": q(1000 * press_xy, 0.5), "p90": q(1000 * press_xy, 0.9)},
    "press_tip_speed_m_s": {"median": q(press_speed, 0.5), "max": round(press_speed.max().item(), 3) if len(press_speed) else None},
    "peak_joint_speed_rad_s_p99": [round(v, 2) for v in peak_qd.quantile(0.99, dim=0).tolist()],
    "joint_speed_limit_rad_s": list(cfg.max_joint_vel),
    "peak_joint_accel_rad_s2_p99": [round(v, 1) for v in peak_qdd.quantile(0.99, dim=0).tolist()],
    "joint_accel_limit_rad_s2": list(cfg.max_joint_acc),
}
print("EVAL_JSON " + json.dumps(R))
env.close()
app.close()
