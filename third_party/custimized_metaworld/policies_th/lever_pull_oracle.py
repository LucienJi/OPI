from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerLeverPullOraclePolicy(TorchBaseOraclePolicy):
    _movement_speed: float = 25.0

    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "gripper": obs[:, 3],
            "lever_pos": obs[:, 4:7],
            "unused_info": obs[:, 7:],
        }

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        o_d = self._parse_obs(obs_t)
        desired_pos, stage_id = self._desired_pos(o_d)
        grab_effort = torch.full_like(o_d["hand_pos"][:, 0], 1.0)
        desired_pos = self._maybe_add_noise(desired_pos, o_d["hand_pos"], stage_id)
        delta = self._clip_delta_pos(self._move(o_d["hand_pos"], desired_pos, p=25.0))
        return self._action(delta, grab_effort)

    def _desired_pos(self, o_d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_lever = o_d["lever_pos"] + torch.tensor([0.0, -0.055, 0.0], device=device, dtype=dtype)

        desired = pos_lever + torch.tensor([0.0, 0.08, 0.02], device=device, dtype=dtype)
        stage = torch.full((pos_curr.shape[0],), 2, device=device, dtype=torch.long)

        cond0 = torch.linalg.norm(pos_curr[:, :2] - pos_lever[:, :2], dim=-1) > 0.02
        mask = cond0
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_lever + torch.tensor([0.0, 0.0, -0.1], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)

        remaining = ~cond0
        cond1 = torch.abs(pos_curr[:, 2] - pos_lever[:, 2]) > 0.02
        mask = remaining & cond1
        desired = torch.where(mask.unsqueeze(-1), pos_lever, desired)
        stage = torch.where(mask, torch.ones_like(stage), stage)

        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.full_like(o_d["hand_pos"][:, 0], 1.0)

