"""Train Rover-Keypad-Press-v3 with Isaac Lab's stock rsl_rl train.py.

  OMNI_KIT_ACCEPT_EULA=YES python train_v3.py --headless [--num_envs 2048] [--max_iterations 1500]

Extra args pass straight through to IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py.
Logs: ./logs/rsl_rl/keypad_press_v3/<timestamp>/
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ISAACLAB = os.environ.get("ISAACLAB", os.path.expanduser("~/IsaacLab"))
sys.path.insert(0, HERE)
import keypad_press_v3  # noqa: E402,F401  (registers the gym id)

sys.path.insert(0, f"{ISAACLAB}/scripts/reinforcement_learning/rsl_rl")  # train.py imports cli_args next to it
sys.argv = [f"{ISAACLAB}/scripts/reinforcement_learning/rsl_rl/train.py", "--task", "Rover-Keypad-Press-v3", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
