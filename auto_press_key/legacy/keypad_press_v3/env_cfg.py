# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""Config for KeypadPressEnv v3 (kept free of runtime imports: train.py loads it before Kit starts).

KeypadPressEnv v3 — press the commanded key with the temporary presser end effector.

What changed vs v1/v2 (see RL_EVAL_METHODOLOGY_MEMO.md for why):
  * Realistic keypad placement: keypad on a table in front of the arm (v1/v2 used RoboSuite layout coords,
    which put the keypad 83 cm up, almost straight above the base). Per-env keypad jitter +-3 cm in xy.
  * Wrist a6 is a real continuous joint (the exporter had locked it at 0/0) — a 6-DoF arm can now align the tool.
  * Controlled body is `ee_tip`, the ball end of a 60 mm presser rod on the a6_EE_holder mounting face
    (rover2026_ee.urdf, built by make_ee_urdf.py). Previously the target was the bare holder origin.
  * Action = IK baseline + RL residual. Every control step a damped-least-squares IK step drives the tip
    to a hover point above the key and then down into it, keeping the presser vertical; the policy adds a
    bounded residual to the joint targets. Targets are integrated (not q + dq), which cancels gravity sag.
  * Press reward: success = tip within 6 mm (xy) of the key centre, 2-5 mm below the key top (key travel is
    4 mm) AND moving < 0.10 m/s. Arriving > 0.25 m/s (slam) or deeper than 5 mm (overpress) fails. Going below the key top anywhere else (wrong key / plate / table) is a failure. All shaping terms are
    <= 0 so finishing early is always better than loitering.
"""
from __future__ import annotations

import os


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

ASSETS = os.environ.get("ROVER_ASSETS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets"))
ROBOT_USD = f"{ASSETS}/ee_usd/rover2026_ee/rover2026_ee.usda"
KEYPAD_USD = f"{ASSETS}/keypad.usd"

# keyboard_stack_v2/layout.py key-top centres (RoboSuite frame) + offset that puts the keypad on a table
LAYOUT = [
    (0.11, -0.04, 0.828), (0.05, -0.08, 0.828), (0.07, -0.08, 0.828), (0.09, -0.08, 0.828),
    (0.05, -0.06, 0.828), (0.07, -0.06, 0.828), (0.09, -0.06, 0.828), (0.05, -0.04, 0.828),
    (0.07, -0.04, 0.828), (0.09, -0.04, 0.828),
]
KP_OFFSET = (0.0, -0.49, -0.528)
KEY_TOPS = [(x + KP_OFFSET[0], y + KP_OFFSET[1], z + KP_OFFSET[2]) for x, y, z in LAYOUT]  # key tops at z=0.300
NUM_KEYS = len(KEY_TOPS)
TABLE_TOP = 0.816 + KP_OFFSET[2]  # keypad plate bottom rests on the table (0.288)

ARM_JOINTS = ["shoulder_joint", "link_1_joint", "link1_link2", "a4_rotation", "a5_rotation", "a6_rotation"]
TIP_BODY = "ee_tip"
TOOL_AXIS_LOCAL = (1.0, 0.0, 0.0)  # presser points along +x of a6_EE_holder / ee_tip
# IK "ready" pose: tip ~11 cm above the keypad centre, presser vertical (from press_demo.py IK)
READY = (2.9571, -1.9357, 1.2767, -0.0017, 1.9017, -1.3818)


@configclass
class KeypadPressEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0
    decimation = 2
    action_space = 6
    observation_space = 6 + 6 + 3 + 3 + 3 + 6 + 6  # q, qd, tip->goal, tip->key, tool axis, dq_ik, a_prev
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=2
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(ARM_JOINTS, READY)), pos=(0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINTS[:3], effort_limit_sim=60.0, velocity_limit_sim=4.0,
                stiffness=400.0, damping=40.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINTS[3:], effort_limit_sim=15.0, velocity_limit_sim=6.0,
                stiffness=100.0, damping=10.0,
            ),
        },
    )
    table: sim_utils.CuboidCfg = sim_utils.CuboidCfg(
        size=(0.40, 0.36, TABLE_TOP),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.22, 0.25)),
    )
    spawn_keypad = False  # visual only (collisions off); keep False for training — jitter isn't mirrored visually
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)

    # control
    ik_gain = 0.25           # fraction of the DLS step applied per control step (integrating controller)
    ik_damping = 0.05
    ik_max_step = 0.04       # rad per control step
    residual_scale = 0.03    # rad per control step, RL residual on the joint targets
    hover_height = 0.03      # m above key top while aligning
    press_goal = 0.003       # m below key top the IK aims for once aligned
    align_tol = 0.004        # xy error below which the IK starts descending
    keypad_jitter = 0.03     # +- m, per-env keypad xy offset
    init_noise = 0.15        # rad, joint noise around READY at reset

    # success / failure
    press_xy_tol = 0.006
    press_depth = 0.002
    press_max_depth = 0.005   # key travel is 4 mm; deeper = driving into the plate
    press_max_speed = 0.10    # m/s — a press only counts if the tip arrives gently
    slam_speed = 0.25         # m/s below the key top => hit, not a press (fail)
    bad_xy = 0.008

    # rewards
    w_dist = 1.0
    w_xy = 0.5
    w_tilt = 0.5
    w_time = 0.02
    w_action = 0.01
    w_qd = 1e-4
    w_speed = 2.0
    success_bonus = 20.0
    fail_penalty = 10.0
