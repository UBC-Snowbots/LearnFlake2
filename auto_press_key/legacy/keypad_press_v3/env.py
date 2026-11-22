# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""KeypadPressEnv v3 runtime. Design notes + all tunables live in env_cfg.py."""
from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, sample_uniform

from .env_cfg import (ARM_JOINTS, KEY_TOPS, KEYPAD_USD, KP_OFFSET, NUM_KEYS, TABLE_TOP, TIP_BODY, TOOL_AXIS_LOCAL,
                      KeypadPressEnvCfg)


class KeypadPressEnv(DirectRLEnv):
    cfg: KeypadPressEnvCfg

    def __init__(self, cfg: KeypadPressEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids = self._robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self.lower = self._robot.data.soft_joint_pos_limits.torch[0, self.joint_ids, 0].to(self.device)
        self.upper = self._robot.data.soft_joint_pos_limits.torch[0, self.joint_ids, 1].to(self.device)
        self.tip_idx = self._robot.find_bodies(TIP_BODY)[0][0]
        self.jac_idx = self.tip_idx - 1 if self._robot.is_fixed_base else self.tip_idx
        self.key_tops = torch.tensor(KEY_TOPS, device=self.device)
        self.tool_local = torch.tensor(TOOL_AXIS_LOCAL, device=self.device).expand(self.num_envs, 3)
        self.down = torch.tensor((0.0, 0.0, -1.0), device=self.device)
        self.kp_off = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_key = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.dq_ik = torch.zeros(self.num_envs, 6, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.table.func("/World/envs/env_0/Table", self.cfg.table,
                            translation=(0.08, -0.55, TABLE_TOP / 2))
        if self.cfg.spawn_keypad:
            kp = sim_utils.UsdFileCfg(usd_path=KEYPAD_USD, collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False))
            kp.func("/World/envs/env_0/Keypad", kp, translation=KP_OFFSET)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ helpers
    def _q(self):
        return self._robot.data.joint_pos.torch[:, self.joint_ids]

    def _tip(self):
        pos = self._robot.data.body_pos_w.torch[:, self.tip_idx] - self.scene.env_origins
        quat = self._robot.data.body_quat_w.torch[:, self.tip_idx]
        return pos, quat_apply(quat, self.tool_local)

    def _key(self):
        return self.key_tops[self.target_key] + self.kp_off

    def _goal(self, tip, key):
        """Hover above the key until aligned in xy, then aim press_goal below the key top."""
        xy = torch.norm(tip[:, :2] - key[:, :2], dim=-1)
        z = torch.where(xy < self.cfg.align_tol, key[:, 2] - self.cfg.press_goal, key[:, 2] + self.cfg.hover_height)
        return torch.cat((key[:, :2], z.unsqueeze(-1)), dim=-1)

    def _ik_step(self, tip, axis, goal):
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx][:, :, self.joint_ids]  # (N,6,6)
        J5 = torch.cat((J[:, :3], J[:, 3:5]), dim=1)  # yaw about vertical is free
        ang = torch.cross(axis, self.down.expand_as(axis), dim=-1)[:, :2]
        err = torch.cat((goal - tip, ang), dim=-1).unsqueeze(-1)  # (N,5,1)
        lam2 = self.cfg.ik_damping ** 2 * torch.eye(5, device=self.device)
        dq = (J5.transpose(1, 2) @ torch.linalg.solve(J5 @ J5.transpose(1, 2) + lam2, err)).squeeze(-1)
        return torch.clamp(self.cfg.ik_gain * dq, -self.cfg.ik_max_step, self.cfg.ik_max_step)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
        tip, axis = self._tip()
        self.dq_ik = torch.nan_to_num(self._ik_step(tip, axis, self._goal(tip, self._key())), nan=0.0)
        t = self.targets + self.dq_ik + self.cfg.residual_scale * self.actions
        self.targets = torch.clamp(t, self.lower, self.upper)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self.targets, joint_ids=self.joint_ids)

    def _press_state(self):
        tip, axis = self._tip()
        key = self._key()
        xy = torch.norm(tip[:, :2] - key[:, :2], dim=-1)
        depth = key[:, 2] - tip[:, 2]
        speed = torch.norm(self._robot.data.body_lin_vel_w.torch[:, self.tip_idx], dim=-1)
        in_key = (depth >= self.cfg.press_depth) & (depth <= self.cfg.press_max_depth)
        success = (xy < self.cfg.press_xy_tol) & in_key & (speed < self.cfg.press_max_speed)
        wrong_spot = (depth >= 0.0) & (xy > self.cfg.bad_xy)
        slam = (depth >= 0.0) & (speed > self.cfg.slam_speed)
        overpress = depth > self.cfg.press_max_depth
        self.fail_kind = torch.where(wrong_spot, 1, torch.where(slam, 2, torch.where(overpress, 3, 0)))
        failed = ~success & (wrong_spot | slam | overpress)
        self.tip_speed = speed
        return tip, axis, key, xy, depth, success, failed

    def _get_dones(self):
        _, _, _, xy, depth, success, failed = self._press_state()
        self.success, self.failed, self.last_xy, self.last_depth = success, failed, xy, depth
        self.last_speed, self.last_fail_kind = self.tip_speed, self.fail_kind
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return success | failed, truncated

    def _get_rewards(self):
        tip, axis, key, xy, depth, success, failed = self._press_state()
        d3 = torch.norm(tip - (key - torch.tensor((0, 0, self.cfg.press_goal), device=self.device)), dim=-1)
        tilt = 1.0 - (axis * self.down).sum(-1)
        qd = self._robot.data.joint_vel.torch[:, self.joint_ids]
        r = (-self.cfg.w_dist * torch.tanh(d3 / 0.1)
             - self.cfg.w_xy * torch.tanh(xy / 0.01)
             - self.cfg.w_tilt * tilt
             - self.cfg.w_time
             - self.cfg.w_action * (self.actions ** 2).sum(-1)
             - self.cfg.w_qd * (qd ** 2).sum(-1)
             # approach gently: tip speed costs more the closer it gets to the key top
             - self.cfg.w_speed * self.tip_speed * torch.exp(-(torch.clamp(-depth, min=0.0) / 0.03) ** 2))
        r = r + self.cfg.success_bonus * success.float() - self.cfg.fail_penalty * failed.float()
        self.extras["log"] = {
            "success_rate": success.float().mean(), "fail_rate": failed.float().mean(),
            "xy_err_cm": 100 * xy.mean(), "tilt": tilt.mean(),
            "slam_rate": (failed & (self.fail_kind == 2)).float().mean(),
        }
        return torch.nan_to_num(r, nan=-1.0)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        q = self._robot.data.default_joint_pos.torch[env_ids][:, self.joint_ids]
        q = torch.clamp(q + sample_uniform(-self.cfg.init_noise, self.cfg.init_noise, (n, 6), self.device),
                        self.lower, self.upper)
        self.targets[env_ids] = q
        self._robot.set_joint_position_target_index(target=q, joint_ids=self.joint_ids, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=self.joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=self.joint_ids,
                                                      env_ids=env_ids)
        self.target_key[env_ids] = torch.randint(0, NUM_KEYS, (n,), device=self.device)
        j = self.cfg.keypad_jitter
        self.kp_off[env_ids, :2] = sample_uniform(-j, j, (n, 2), self.device)
        self.actions[env_ids] = 0.0
        self.dq_ik[env_ids] = 0.0

    def _get_observations(self):
        q = self._q()
        q = torch.atan2(torch.sin(q), torch.cos(q))  # a6 is continuous
        tip, axis = self._tip()
        key = self._key()
        obs = torch.cat((
            q,
            0.1 * self._robot.data.joint_vel.torch[:, self.joint_ids],
            self._goal(tip, key) - tip,
            key - tip,
            axis,
            self.dq_ik / self.cfg.ik_max_step,
            self.actions,
        ), dim=-1)
        return {"policy": torch.clamp(torch.nan_to_num(obs, nan=0.0), -10.0, 10.0)}

