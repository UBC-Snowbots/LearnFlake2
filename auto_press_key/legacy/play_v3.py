"""Watch Rover-Keypad-Press-v3 in the GUI (loops forever, real-time).

  DISPLAY=:0 OMNI_KIT_ACCEPT_EULA=YES python play_v3.py --checkpoint <model_*.pt> [--baseline]
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--slowmo", type=float, default=4.0, help="playback slow-down factor")
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
cfg.scene.env_spacing = 1.2
cfg.spawn_keypad = True
cfg.keypad_jitter = 0.0  # keep the visual keypad where the targets are
if args.num_envs == 1:  # close-up on the one arm, same framing as record_v3.py
    cfg.viewer.origin_type, cfg.viewer.env_index = "env", 0
    cfg.viewer.eye, cfg.viewer.lookat = (0.62, -1.00, 0.58), (0.06, -0.50, 0.34)
else:
    cfg.viewer.eye, cfg.viewer.lookat = (2.2, -3.2, 1.6), (1.2, -0.8, 0.3)
env = RslRlVecEnvWrapper(gym.make("Rover-Keypad-Press-v3", cfg=cfg))
dev = env.unwrapped.device
policy = None
if not args.baseline:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    agent = handle_deprecated_rsl_rl_cfg(KeypadPressPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

# highlight each env's target key (visual only)
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdShade  # noqa: E402

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
stage = omni.usd.get_context().get_stage()


def mat(path, rgb, emissive=0.0):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emissive for c in rgb]))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


M_KEY, M_TGT = mat("/World/Looks/key", (0.86, 0.87, 0.90)), mat("/World/Looks/target", (1.0, 0.78, 0.05), 2.0)
keys = {}
for p in stage.Traverse():
    parts = str(p.GetPath()).split("/")
    if len(parts) > 4 and parts[4] == "Keypad" and p.GetName().startswith("key_"):
        keys[(int(parts[3].split("_")[1]), p.GetName()[4:])] = p
shown = {}


def paint():
    for e, t in enumerate(env.unwrapped.target_key.tolist()):
        if shown.get(e) != t:
            for w in WORDS:
                if (e, w) in keys:
                    UsdShade.MaterialBindingAPI.Apply(keys[(e, w)]).Bind(
                        M_TGT if w == WORDS[t] else M_KEY, UsdShade.Tokens.strongerThanDescendants)
            shown[e] = t


obs = env.get_observations()
dt = env.unwrapped.step_dt * args.slowmo
with torch.inference_mode():
    while app.is_running():
        t0 = time.time()
        act = torch.zeros(env.num_envs, 6, device=dev) if policy is None else policy(obs)
        obs, _, _, _ = env.step(act)
        paint()
        time.sleep(max(0.0, dt - (time.time() - t0)))
env.close()
app.close()
