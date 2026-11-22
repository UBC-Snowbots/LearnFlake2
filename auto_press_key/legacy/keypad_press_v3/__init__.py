# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""Keypad PRESS v3 (Isaac Lab 3.0 / Isaac Sim 6.0.1): IK baseline + RL residual, presser EE, press reward.

Only registers the gym id here (string entry points) so it can be imported before the Kit app starts.
"""
import gymnasium as gym

gym.register(
    id="Rover-Keypad-Press-v3",
    entry_point=f"{__name__}.env:KeypadPressEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KeypadPressEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:KeypadPressPPORunnerCfg",
    },
)
