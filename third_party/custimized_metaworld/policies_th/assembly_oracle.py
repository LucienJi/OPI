from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerAssemblyOraclePolicy(TorchBaseOraclePolicy):
    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "gripper": obs[:, 3],
            "wrench_pos": obs[:, 4:7],
            "peg_pos": obs[:, -3:],
            "unused_info": obs[:, 7:-3],
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
        pos_wrench = o_d["wrench_pos"] + torch.tensor([-0.02, 0.0, 0.0], device=device, dtype=dtype)
        pos_peg = o_d["peg_pos"] + torch.tensor([0.12, 0.0, 0.14], device=device, dtype=dtype)

        desired = pos_peg
        stage = torch.ones((pos_curr.shape[0],), device=device, dtype=torch.long)

        cond0 = torch.linalg.norm(pos_curr[:, :2] - pos_wrench[:, :2], dim=-1) > 0.02
        mask = cond0
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_wrench + torch.tensor([0.0, 0.0, 0.1], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)

        remaining = ~cond0
        cond1 = torch.linalg.norm(pos_curr[:, :2] - pos_peg[:, :2], dim=-1) <= 0.02
        mask = remaining & cond1
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_peg + torch.tensor([0.0, 0.0, -0.2], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond1

        cond2 = torch.abs(pos_curr[:, 2] - pos_wrench[:, 2]) > 0.05
        mask = remaining & cond2
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_wrench + torch.tensor([0.0, 0.0, 0.03], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond2

        cond3 = torch.abs(pos_curr[:, 2] - pos_peg[:, 2]) > 0.04
        mask = remaining & cond3
        desired = torch.where(
            mask.unsqueeze(-1),
            torch.stack([pos_curr[:, 0], pos_curr[:, 1], pos_peg[:, 2]], dim=-1),
            desired,
        )
        stage = torch.where(mask, torch.ones_like(stage), stage)

        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + torch.tensor([-0.02, 0.0, 0.0], device=device, dtype=dtype)
        cond = (
            torch.linalg.norm(pos_curr[:, :2] - pos_wrench[:, :2], dim=-1) > 0.02
        ) | (torch.abs(pos_curr[:, 2] - pos_wrench[:, 2]) > 0.05)
        return torch.where(cond, torch.zeros_like(pos_curr[:, 0]), torch.full_like(pos_curr[:, 0], 0.6))

