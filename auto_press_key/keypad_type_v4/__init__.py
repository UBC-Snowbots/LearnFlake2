# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""Keypad TYPE v4 (Isaac Lab 3.0 / Isaac Sim 6.0.1): type a full code, IK + RL residual, joint vel/acc limits.

Only registers the gym id here (string entry points) so it can be imported before the Kit app starts.
"""
import gymnasium as gym

gym.register(
    id="Rover-Keypad-Type-v4",
    entry_point=f"{__name__}.env:KeypadTypeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KeypadTypeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:KeypadTypePPORunnerCfg",
    },
)
