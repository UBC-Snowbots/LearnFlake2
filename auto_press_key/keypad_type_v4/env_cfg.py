# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""Config for KeypadTypeEnv v4 (kept free of runtime imports: train.py loads it before Kit starts).

KeypadTypeEnv v4 — type a whole code with the presser EE, moving like a real arm could.

Builds on v3 (IK baseline + RL residual, realistic keypad on a table, free wrist, `ee_tip` presser, gentle-press
rule). What v3 got wrong, visible in playback: each episode was ONE press, then the env teleported the arm back to
the start pose (looked like a "ninja" jump between digits), and the policy learned 0.18 s hover->press lunges.
v4 fixes:
  * One episode = a code of `code_length` digits typed in sequence. No reset between digits: after a press the
    arm must lift the tip `lift_clear` above the key before the next digit counts (no dragging across keys).
    Reset only after the full code or a failure.
  * Joint velocity + acceleration limits on the COMMANDED motion (targets are integrated from a rate- and
    accel-limited joint velocity), plus an anti-windup band around the measured joint positions.
    Limits below are CONSERVATIVE PLACEHOLDERS — replace with motor/gearbox datasheet values
    (docs/rover2026_arm_specs.md: the URDF has no effort/velocity data).
  * Wrong spot only counts over the keypad plate footprint (below key-top height beside the keypad is not a
    keypad press; the table is a real collider).
  * Slam = moving DOWN faster than `slam_speed` below the key top (lifting off quickly is fine).
Success of one digit: tip within 6 mm (xy) of the key centre, 2-5 mm below the key top, speed < 0.10 m/s.
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

ASSETS = os.environ.get("ROVER_ASSETS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))
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
# keypad plate footprint (keypad.usd plate bounds + KP_OFFSET): centre + half extents in xy
PLATE_CENTER_XY = (0.080 + KP_OFFSET[0], -0.060 + KP_OFFSET[1])
PLATE_HALF_XY = (0.044, 0.034)
TABLE_TOP = 0.816 + KP_OFFSET[2]  # keypad plate bottom rests on the table (0.288)

ARM_JOINTS = ["shoulder_joint", "link_1_joint", "link1_link2", "a4_rotation", "a5_rotation", "a6_rotation"]
TIP_BODY = "ee_tip"
TOOL_AXIS_LOCAL = (1.0, 0.0, 0.0)  # presser points along +x of a6_EE_holder / ee_tip
# IK "ready" pose: tip ~11 cm above the keypad centre, presser vertical (from press_demo.py IK)
READY = (2.9571, -1.9357, 1.2767, -0.0017, 1.9017, -1.3818)

# PLACEHOLDER joint limits (big joints / wrist) — replace with datasheet values
MAX_JOINT_VEL = (0.8, 0.8, 0.8, 1.5, 1.5, 1.5)   # rad/s
MAX_JOINT_ACC = (2.0, 2.0, 2.0, 4.0, 4.0, 4.0)   # rad/s^2


@configclass
class KeypadTypeEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 2
    action_space = 6
    # q, qd, tip->goal, tip->key, tool axis, dq_ik, a_prev, cmd_vel, need_lift, digits_left
    observation_space = 6 + 6 + 3 + 3 + 3 + 6 + 6 + 6 + 1 + 1
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
                joint_names_expr=ARM_JOINTS[:3], effort_limit_sim=60.0, velocity_limit_sim=1.2,
                stiffness=400.0, damping=40.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINTS[3:], effort_limit_sim=15.0, velocity_limit_sim=2.0,
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

    # task
    code_length = 6
    lift_clear = 0.012       # m above the pressed key's top before the next digit counts

    # control
    max_joint_vel = MAX_JOINT_VEL
    max_joint_acc = MAX_JOINT_ACC
    windup = 0.25            # rad, max distance of the joint target from the measured joint position
    ik_gain = 0.15           # fraction of the DLS step applied per control step (integrating controller)
    cart_speed_far = 0.25    # m/s tip speed cap for the IK in free space
    cart_speed_near = 0.04   # m/s tip speed cap within near_height above the key top (brake before contact)
    near_height = 0.03
    ik_damping = 0.05
    ik_max_step = 0.04       # rad per control step (before the vel/acc limits)
    residual_scale = 0.005   # rad per control step in free space; scaled by the IK speed cap near the key
                             # (an unscaled residual swamped the braking IK and no key was ever pressed)
    hover_height = 0.03      # m above key top while aligning / after lifting
    press_goal = 0.003       # m below key top the IK aims for once aligned
    align_tol = 0.004        # xy error below which the IK starts descending
    keypad_jitter = 0.03     # +- m, per-env keypad xy offset
    init_noise = 0.08        # rad, joint noise around READY at reset

    # success / failure (per digit)
    press_xy_tol = 0.006
    press_depth = 0.002
    press_max_depth = 0.005   # key travel is 4 mm; deeper = driving into the plate
    press_max_speed = 0.10    # m/s — a press only counts if the tip arrives gently
    slam_speed = 0.25         # m/s DOWNWARD below the key top => hit, not a press (fail)
    bad_xy = 0.008
    plate_margin = 0.008      # tip this close to the plate edge still counts as 'on the keypad'

    # rewards
    # potential-based progress shaping weights (phi = -(w_dist*tanh(d/10cm) + w_xy*tanh(xy/1cm) + w_tilt*tilt))
    w_dist = 5.0
    w_xy = 2.0
    w_tilt = 1.0
    gamma_shaping = 0.99   # match PPO gamma
    w_time = 0.01
    w_action = 0.01
    w_acc = 0.002            # commanded joint acceleration (smoothness)
    w_speed = 0.5            # tip speed near the key top
    digit_bonus = 30.0       # must beat P(fail per attempt) * fail_penalty, or pressing is -EV
    code_bonus = 50.0
    fail_penalty = 15.0
