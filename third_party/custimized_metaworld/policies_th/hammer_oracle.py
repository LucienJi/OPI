from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerHammerOraclePolicy(TorchBaseOraclePolicy):
    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "gripper": obs[:, 3],
            "hammer_pos": obs[:, 4:7],
            "unused_info": obs[:, 7:],
        }

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        o_d = self._parse_obs(obs_t)
        desired_pos, stage_id = self._desired_pos(o_d)
        grab_effort = self._grab_effort(o_d)
        desired_pos = self._maybe_add_noise(desired_pos, o_d["hand_pos"], stage_id)
        delta = self._clip_delta_pos(self._move(o_d["hand_pos"], desired_pos, p=10.0))
        return self._action(delta, grab_effort)

    def _desired_pos(self, o_d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["hammer_pos"] + torch.tensor([-0.04, 0.0, -0.01], device=device, dtype=dtype)
        pos_goal = torch.tensor([0.24, 0.71, 0.11], device=device, dtype=dtype) + torch.tensor(
            [-0.19, 0.0, 0.05], device=device, dtype=dtype
        )

        desired = pos_goal.expand(pos_curr.shape[0], -1)
        stage = torch.ones((pos_curr.shape[0],), device=device, dtype=torch.long)

        cond0 = torch.linalg.norm(pos_curr[:, :2] - pos_puck[:, :2], dim=-1) > 0.04
        mask = cond0
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_puck + torch.tensor([0.0, 0.0, 0.1], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)

        remaining = ~cond0
        cond1 = (torch.abs(pos_curr[:, 2] - pos_puck[:, 2]) > 0.05) & (pos_puck[:, 2] < 0.03)
        mask = remaining & cond1
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_puck + torch.tensor([0.0, 0.0, 0.03], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond1

        cond2 = torch.linalg.norm(
            torch.stack([pos_curr[:, 0], pos_curr[:, 2]], dim=-1)
            - torch.stack([pos_goal[0].expand(pos_curr.shape[0]), pos_goal[2].expand(pos_curr.shape[0])], dim=-1),
            dim=-1,
        ) > 0.02
        mask = remaining & cond2
        desired = torch.where(
            mask.unsqueeze(-1),
            torch.stack([pos_goal[0].expand(pos_curr.shape[0]), pos_curr[:, 1], pos_goal[2].expand(pos_curr.shape[0])], dim=-1),
            desired,
        )
        stage = torch.where(mask, torch.ones_like(stage), stage)

        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["hammer_pos"] + torch.tensor([-0.04, 0.0, -0.01], device=device, dtype=dtype)
        cond = (
            torch.linalg.norm(pos_curr[:, :2] - pos_puck[:, :2], dim=-1) > 0.04
        ) | (torch.abs(pos_curr[:, 2] - pos_puck[:, 2]) > 0.06)
        return torch.where(cond, torch.zeros_like(pos_curr[:, 0]), torch.full_like(pos_curr[:, 0], 0.8))

