from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerBasketballOraclePolicy(TorchBaseOraclePolicy):
    _movement_speed: float = 25.0

    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "gripper": obs[:, 3],
            "ball_pos": obs[:, 4:7],
            "hoop_x": obs[:, -3],
            "hoop_yz": obs[:, -2:],
            "unused_info": obs[:, 7:-3],
        }

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        o_d = self._parse_obs(obs_t)
        desired_pos, stage_id = self._desired_pos(o_d)
        grab_effort = self._grab_effort(o_d)
        desired_pos = self._maybe_add_noise(desired_pos, o_d["hand_pos"], stage_id)
        delta = self._clip_delta_pos(self._move(o_d["hand_pos"], desired_pos, p=25.0))
        return self._action(delta, grab_effort)

    def _desired_pos(self, o_d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"] + torch.tensor([0.0, 0.0, 0.01], device=device, dtype=dtype)
        pos_hoop = torch.stack(
            [o_d["hoop_x"], torch.full_like(o_d["hoop_x"], 0.875), torch.full_like(o_d["hoop_x"], 0.35)],
            dim=-1,
        )

        desired = pos_hoop
        stage = torch.ones((pos_curr.shape[0],), device=device, dtype=torch.long)

        cond0 = torch.linalg.norm(pos_curr[:, :2] - pos_ball[:, :2], dim=-1) > 0.04
        mask = cond0
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_ball + torch.tensor([0.0, 0.0, 0.3], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)

        remaining = ~cond0
        cond1 = torch.abs(pos_curr[:, 2] - pos_ball[:, 2]) > 0.025
        mask = remaining & cond1
        desired = torch.where(mask.unsqueeze(-1), pos_ball, desired)
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond1

        cond2 = torch.abs(pos_ball[:, 2] - pos_hoop[:, 2]) > 0.025
        mask = remaining & cond2
        desired = torch.where(
            mask.unsqueeze(-1),
            torch.stack([pos_curr[:, 0], pos_curr[:, 1], pos_hoop[:, 2]], dim=-1),
            desired,
        )
        stage = torch.where(mask, torch.ones_like(stage), stage)

        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"]
        cond = (
            torch.linalg.norm(pos_curr[:, :2] - pos_ball[:, :2], dim=-1) > 0.04
        ) | (torch.abs(pos_curr[:, 2] - pos_ball[:, 2]) > 0.06)
        return torch.where(cond, torch.full_like(pos_curr[:, 0], -1.0), torch.full_like(pos_curr[:, 0], 0.6))

