"""Record a video of ONE rover arm pressing keys with the v3 policy (or --baseline IK only).

  OMNI_KIT_ACCEPT_EULA=YES python record_v3.py --headless --enable_cameras \
      --checkpoint <model_*.pt> --presses 10 --out keypad_v3.mp4
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--presses", type=int, default=10)
parser.add_argument("--slowmo", type=int, default=4, help="video plays this many times slower than real time")
parser.add_argument("--hold_s", type=float, default=0.8, help="freeze on each press outcome (video seconds)")
parser.add_argument("--out", type=str, default="keypad_v3.mp4")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdShade  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keypad_press_v3  # noqa: E402,F401
from keypad_press_v3.agents import KeypadPressPPORunnerCfg  # noqa: E402
from keypad_press_v3.env_cfg import KeypadPressEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

cfg = KeypadPressEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.spawn_keypad = True
cfg.keypad_jitter = 0.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (0.62, -1.00, 0.58)
cfg.viewer.lookat = (0.06, -0.50, 0.34)
cfg.viewer.resolution = (1280, 720)
env = RslRlVecEnvWrapper(gym.make("Rover-Keypad-Press-v3", cfg=cfg, render_mode="rgb_array"))
u = env.unwrapped
dev = u.device

policy = None
if not args.baseline:
    from importlib import metadata

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    agent = handle_deprecated_rsl_rl_cfg(KeypadPressPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=dev)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

# --- key highlighting (visual only)
stage = omni.usd.get_context().get_stage()


def mat(path, rgb, emissive=0.0):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * emissive for c in rgb]))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


M = {"key": mat("/World/Looks/key", (0.86, 0.87, 0.90)), "target": mat("/World/Looks/target", (1.0, 0.78, 0.05), 2.0),
     "ok": mat("/World/Looks/ok", (0.1, 0.95, 0.3), 4.0), "bad": mat("/World/Looks/bad", (0.95, 0.15, 0.1), 4.0)}
keys = {}
for p in stage.Traverse():
    if str(p.GetPath()).startswith("/World/envs/env_0/Keypad") and p.GetName().startswith("key_"):
        keys[p.GetName()[4:]] = p


def paint(target, state="target"):
    for w, p in keys.items():
        UsdShade.MaterialBindingAPI.Apply(p).Bind(M[state] if w == WORDS[target] else M["key"],
                                                  UsdShade.Tokens.strongerThanDescendants)


try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except OSError:
    font = small = ImageFont.load_default()
title = "IK baseline only" if policy is None else "IK + RL residual (v3)"


def grab():
    return np.asarray(u.render())[..., :3].copy()


def frame(raw, text, sub, color=(255, 210, 60)):
    im = Image.fromarray(raw)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 92), fill=(0, 0, 0))
    d.text((20, 10), f"Rover arm · {title}", font=font, fill=(255, 255, 255))
    d.text((20, 52), text, font=small, fill=color)
    d.text((im.width - 330, 52), sub, font=small, fill=(180, 180, 180))
    return np.asarray(im)


fps = max(1, round(1 / (u.step_dt * args.slowmo)))
writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
obs = env.get_observations()
done_n, t, ok_n = 0, 0, 0
with torch.inference_mode():
    target = int(u.target_key[0])
    paint(target)
    for _ in range(3):
        app.update()
    last = grab()
    while done_n < args.presses and app.is_running():
        act = torch.zeros(1, 6, device=dev) if policy is None else policy(obs)
        obs, _, dones, _ = env.step(act)
        t += 1
        if bool(dones[0]):
            if bool(u.success[0]):
                res, st = "PRESSED", "ok"
                ok_n += 1
            else:
                res, st = {1: "WRONG SPOT", 2: "SLAM", 3: "OVERPRESS"}.get(int(u.last_fail_kind[0]), "TIMEOUT"), "bad"
            done_n += 1
            msg = (f"key {target}: {res} in {t * u.step_dt:.2f}s · xy err {1000 * float(u.last_xy[0]):.1f} mm · "
                   f"tip {float(u.last_speed[0]):.2f} m/s")
            # the env already reset the arm on this step: freeze on the last frame before contact
            f = frame(last, msg, f"press {done_n}/{args.presses} · {ok_n} ok",
                      (80, 240, 110) if st == "ok" else (255, 90, 70))
            for _ in range(int(args.hold_s * fps)):
                writer.append_data(f)
            target, t = int(u.target_key[0]), 0
            paint(target)
            continue
        last = grab()
        writer.append_data(frame(last, f"target key: {target}", f"{args.slowmo}x slow-mo · press {done_n + 1}/{args.presses}"))
writer.close()
print(f"WROTE {args.out}  presses={done_n} ok={ok_n}")
env.close()
app.close()
