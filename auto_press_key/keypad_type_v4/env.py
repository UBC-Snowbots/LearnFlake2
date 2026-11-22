# Copyright (c) 2026, UBC Rover Ptolemy. SPDX-License-Identifier: BSD-3-Clause
"""KeypadTypeEnv v4 runtime. Design notes + all tunables live in env_cfg.py."""
from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, sample_uniform

from .env_cfg import (ARM_JOINTS, KEY_TOPS, KEYPAD_USD, KP_OFFSET, NUM_KEYS, PLATE_CENTER_XY, PLATE_HALF_XY, TABLE_TOP,
                      TIP_BODY, TOOL_AXIS_LOCAL, KeypadTypeEnvCfg)


class KeypadTypeEnv(DirectRLEnv):
    cfg: KeypadTypeEnvCfg

    def __init__(self, cfg: KeypadTypeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.joint_ids = self._robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self.lower = self._robot.data.soft_joint_pos_limits.torch[0, self.joint_ids, 0].to(dev)
        self.upper = self._robot.data.soft_joint_pos_limits.torch[0, self.joint_ids, 1].to(dev)
        self.tip_idx = self._robot.find_bodies(TIP_BODY)[0][0]
        self.jac_idx = self.tip_idx - 1 if self._robot.is_fixed_base else self.tip_idx
        self.key_tops = torch.tensor(KEY_TOPS, device=dev)
        self.plate_c = torch.tensor(PLATE_CENTER_XY, device=dev)
        self.plate_h = torch.tensor(PLATE_HALF_XY, device=dev) + self.cfg.plate_margin
        self.tool_local = torch.tensor(TOOL_AXIS_LOCAL, device=dev).expand(n, 3)
        self.down = torch.tensor((0.0, 0.0, -1.0), device=dev)
        self.vmax = torch.tensor(self.cfg.max_joint_vel, device=dev)
        self.amax = torch.tensor(self.cfg.max_joint_acc, device=dev)
        self.kp_off = torch.zeros(n, 3, device=dev)
        self.target_key = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_key_pos = torch.zeros(n, 3, device=dev)
        self.need_lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self.digits_done = torch.zeros(n, dtype=torch.long, device=dev)
        self.targets = torch.zeros(n, 6, device=dev)
        self.cmd_vel = torch.zeros(n, 6, device=dev)
        self.cmd_acc = torch.zeros(n, 6, device=dev)
        self.actions = torch.zeros(n, 6, device=dev)
        self.dq_ik = torch.zeros(n, 6, device=dev)
        self.prev_phi = torch.zeros(n, device=dev)
        self.phi_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        # per-step event flags / stats (read by eval + video scripts)
        self.pressed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pressed_key = torch.zeros(n, dtype=torch.long, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.code_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fail_kind = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_xy = torch.zeros(n, device=dev)
        self.last_speed = torch.zeros(n, device=dev)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.table.func("/World/envs/env_0/Table", self.cfg.table, translation=(0.08, -0.55, TABLE_TOP / 2))
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
        vel = self._robot.data.body_lin_vel_w.torch[:, self.tip_idx]
        return pos, quat_apply(quat, self.tool_local), vel

    def _key(self):
        return self.key_tops[self.target_key] + self.kp_off

    def _goal(self, tip, key):
        """After a press: straight up above the pressed key. Otherwise hover over the key until aligned, then
        aim press_goal below its top."""
        xy = torch.norm(tip[:, :2] - key[:, :2], dim=-1)
        z = torch.where(xy < self.cfg.align_tol, key[:, 2] - self.cfg.press_goal, key[:, 2] + self.cfg.hover_height)
        goal = torch.cat((key[:, :2], z.unsqueeze(-1)), dim=-1)
        lift = self.prev_key_pos + torch.tensor((0.0, 0.0, self.cfg.hover_height), device=self.device)
        return torch.where(self.need_lift.unsqueeze(-1), lift, goal)

    def _ik_step(self, tip, axis, goal, key):
        # Cartesian speed cap: fast in free space, slow in the last few cm above the key (the joint accel limit
        # can't stop a full-speed arm in time, so the planner has to brake early)
        near = ((tip[:, 2] - key[:, 2]) < self.cfg.near_height) & ~self.need_lift
        vcap = torch.where(near, self.cfg.cart_speed_near, self.cfg.cart_speed_far)
        self.res_gain = (vcap / self.cfg.cart_speed_far).unsqueeze(-1)  # residual brakes with the IK
        step = vcap * self.step_dt / self.cfg.ik_gain
        pe = goal - tip
        pe = pe * torch.clamp(step / (torch.norm(pe, dim=-1) + 1e-9), max=1.0).unsqueeze(-1)
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx][:, :, self.joint_ids]  # (N,6,6)
        J5 = torch.cat((J[:, :3], J[:, 3:5]), dim=1)  # yaw about vertical is free
        ang = torch.cross(axis, self.down.expand_as(axis), dim=-1)[:, :2]
        err = torch.cat((pe, ang), dim=-1).unsqueeze(-1)  # (N,5,1)
        lam2 = self.cfg.ik_damping ** 2 * torch.eye(5, device=self.device)
        dq = (J5.transpose(1, 2) @ torch.linalg.solve(J5 @ J5.transpose(1, 2) + lam2, err)).squeeze(-1)
        return torch.clamp(self.cfg.ik_gain * dq, -self.cfg.ik_max_step, self.cfg.ik_max_step)

    def _new_key(self, env_ids, avoid=None):
        k = torch.randint(0, NUM_KEYS, (len(env_ids),), device=self.device)
        if avoid is not None:  # next digit differs from the one just pressed
            k = torch.where(k == avoid, (k + torch.randint(1, NUM_KEYS, k.shape, device=self.device)) % NUM_KEYS, k)
        self.target_key[env_ids] = k

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
        tip, axis, _ = self._tip()
        key = self._key()
        self.dq_ik = torch.nan_to_num(self._ik_step(tip, axis, self._goal(tip, key), key), nan=0.0)
        dt = self.step_dt
        v_des = (self.dq_ik + self.cfg.residual_scale * self.res_gain * self.actions) / dt
        v = torch.clamp(v_des, self.cmd_vel - self.amax * dt, self.cmd_vel + self.amax * dt)  # accel limit
        v = torch.clamp(v, -self.vmax, self.vmax)                                            # velocity limit
        self.cmd_acc = (v - self.cmd_vel) / dt
        self.cmd_vel = v
        q = self._q()
        t = torch.clamp(self.targets + v * dt, q - self.cfg.windup, q + self.cfg.windup)
        self.targets = torch.clamp(t, self.lower, self.upper)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self.targets, joint_ids=self.joint_ids)

    def _get_dones(self):
        tip, axis, vel = self._tip()
        key = self._key()
        xy = torch.norm(tip[:, :2] - key[:, :2], dim=-1)
        depth = key[:, 2] - tip[:, 2]
        speed = torch.norm(vel, dim=-1)
        prev_xy = torch.norm(tip[:, :2] - self.prev_key_pos[:, :2], dim=-1)
        # lifting off the key just pressed: clear once the tip is lift_clear above it
        self.need_lift &= tip[:, 2] < self.prev_key_pos[:, 2] + self.cfg.lift_clear
        in_key = (depth >= self.cfg.press_depth) & (depth <= self.cfg.press_max_depth)
        pressed = ~self.need_lift & (xy < self.cfg.press_xy_tol) & in_key & (speed < self.cfg.press_max_speed)
        near_prev = self.need_lift & (prev_xy < self.cfg.bad_xy)  # still sitting in the key we just pressed
        on_plate = ((tip[:, :2] - self.plate_c - self.kp_off[:, :2]).abs() <= self.plate_h).all(-1)
        wrong_spot = on_plate & (depth >= 0.0) & (xy > self.cfg.bad_xy) & ~near_prev
        slam = (depth >= 0.0) & (-vel[:, 2] > self.cfg.slam_speed)
        overpress = depth > self.cfg.press_max_depth
        failed = ~pressed & (wrong_spot | slam | overpress)
        self.fail_kind = torch.where(wrong_spot, 1, torch.where(slam, 2, torch.where(overpress, 3, 0)))

        self.pressed_now, self.failed, self.last_xy, self.last_speed = pressed, failed, xy, speed
        self.pressed_key = self.target_key.clone()
        self.digits_done += pressed.long()
        self.code_done = self.digits_done >= self.cfg.code_length
        ids = torch.nonzero(pressed & ~self.code_done).squeeze(-1)
        if len(ids):
            self.prev_key_pos[ids] = key[ids]
            self.need_lift[ids] = True
            self._new_key(ids, avoid=self.target_key[ids])
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return failed | self.code_done, truncated

    def _get_rewards(self):
        tip, axis, vel = self._tip()
        goal = self._goal(tip, self._key())
        d3 = torch.norm(tip - goal, dim=-1)
        xy = torch.norm(tip[:, :2] - self._key()[:, :2], dim=-1)
        depth = self._key()[:, 2] - tip[:, 2]
        tilt = 1.0 - (axis * self.down).sum(-1)
        speed = torch.norm(vel, dim=-1)
        approach = ~self.need_lift
        # Potential-based progress shaping (Ng et al. 1999): r = gamma*phi(s') - phi(s). Hovering earns nothing,
        # so the digit bonus / failure penalty decide behaviour. (All-negative shaping made "press once then
        # overpress to end the episode" optimal; all-positive shaping made hovering forever optimal.)
        phi = -(self.cfg.w_dist * torch.tanh(d3 / 0.1)
                + self.cfg.w_xy * torch.tanh(xy / 0.01) * approach.float()
                + self.cfg.w_tilt * tilt)
        valid = self.phi_valid & ~self.pressed_now  # key just switched: phi is w.r.t. the new goal
        shaping = torch.where(valid, self.cfg.gamma_shaping * phi - self.prev_phi, torch.zeros_like(phi))
        # the goal jumps to the next key on a press: restart the potential instead of charging for the jump
        self.prev_phi, self.phi_valid = phi.detach(), torch.ones_like(self.phi_valid)
        r = (shaping
             - self.cfg.w_time
             - self.cfg.w_action * (self.actions ** 2).sum(-1)
             - self.cfg.w_acc * ((self.cmd_acc / self.amax) ** 2).sum(-1)
             - self.cfg.w_speed * speed * torch.exp(-(torch.clamp(-depth, min=0.0) / 0.03) ** 2) * approach.float())
        r = (r + self.cfg.digit_bonus * self.pressed_now.float() + self.cfg.code_bonus * self.code_done.float()
             - self.cfg.fail_penalty * self.failed.float())
        self.extras["log"] = {
            "digit_rate": self.pressed_now.float().mean(), "fail_rate": self.failed.float().mean(),
            "code_done_rate": self.code_done.float().mean(), "digits_done": self.digits_done.float().mean(),
            "slam_rate": (self.failed & (self.fail_kind == 2)).float().mean(),
            "peak_cmd_vel_frac": (self.cmd_vel.abs() / self.vmax).max(-1).values.mean(),
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
        self._new_key(env_ids)
        j = self.cfg.keypad_jitter
        self.kp_off[env_ids, :2] = sample_uniform(-j, j, (n, 2), self.device)
        self.need_lift[env_ids] = False
        self.digits_done[env_ids] = 0
        self.cmd_vel[env_ids] = 0.0
        self.cmd_acc[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.dq_ik[env_ids] = 0.0
        self.phi_valid[env_ids] = False

    def _get_observations(self):
        q = self._q()
        q = torch.atan2(torch.sin(q), torch.cos(q))  # a6 is continuous
        tip, axis, _ = self._tip()
        key = self._key()
        obs = torch.cat((
            q,
            0.1 * self._robot.data.joint_vel.torch[:, self.joint_ids],
            self._goal(tip, key) - tip,
            key - tip,
            axis,
            self.dq_ik / self.cfg.ik_max_step,
            self.actions,
            self.cmd_vel / self.vmax,
            self.need_lift.float().unsqueeze(-1),
            ((self.cfg.code_length - self.digits_done).float() / self.cfg.code_length).unsqueeze(-1),
        ), dim=-1)
        return {"policy": torch.clamp(torch.nan_to_num(obs, nan=0.0), -10.0, 10.0)}
