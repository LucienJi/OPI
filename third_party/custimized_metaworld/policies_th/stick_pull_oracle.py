from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerStickPullOraclePolicy(TorchBaseOraclePolicy):
    _movement_speed: float = 25.0

    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "unused_1": obs[:, 3],
            "stick_pos": obs[:, 4:7],
            "unused_2": obs[:, 7:11],
            "obj_pos": obs[:, 11:14],
            "unused_3": obs[:, 14:-3],
            "goal_pos": obs[:, -3:],
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
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + torch.tensor([-0.015, 0.0, 0.03], device=device, dtype=dtype)
        thermos_pos = o_d["obj_pos"] + torch.tensor([-0.015, 0.0, 0.03], device=device, dtype=dtype)
        goal_pos = o_d["goal_pos"] + torch.tensor([-0.05, 0.0, 0.0], device=device, dtype=dtype)

        desired = goal_pos
        stage = torch.ones((hand_pos.shape[0],), device=device, dtype=torch.long)

        cond0 = torch.abs(stick_pos[:, 0] - thermos_pos[:, 0]) > 0.04
        desired = torch.where(cond0.unsqueeze(-1), thermos_pos, desired)
        stage = torch.where(cond0, torch.ones_like(stage), stage)

        remaining = cond0
        cond1 = torch.linalg.norm(hand_pos[:, :2] - stick_pos[:, :2], dim=-1) > 0.02
        mask = remaining & cond1
        desired = torch.where(
            mask.unsqueeze(-1),
            stick_pos + torch.tensor([0.0, 0.0, 0.1], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond1

        cond2 = torch.abs(hand_pos[:, 2] - stick_pos[:, 2]) > 0.02
        mask = remaining & cond2
        desired = torch.where(mask.unsqueeze(-1), stick_pos, desired)
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond2

        cond3 = torch.abs(stick_pos[:, 1] - thermos_pos[:, 1]) > 0.02
        mask = remaining & cond3
        desired = torch.where(
            mask.unsqueeze(-1),
            torch.stack([stick_pos[:, 0], thermos_pos[:, 1], stick_pos[:, 2]], dim=-1),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        remaining = remaining & ~cond3

        cond4 = torch.abs(stick_pos[:, 2] - thermos_pos[:, 2]) > 0.02
        mask = remaining & cond4
        desired = torch.where(
            mask.unsqueeze(-1),
            torch.stack([stick_pos[:, 0], thermos_pos[:, 1], thermos_pos[:, 2]], dim=-1),
            desired,
        )
        stage = torch.where(mask, torch.ones_like(stage), stage)

        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + torch.tensor([-0.015, 0.0, 0.03], device=device, dtype=dtype)
        cond = (
            torch.linalg.norm(hand_pos[:, :2] - stick_pos[:, :2], dim=-1) > 0.02
        ) | (torch.abs(hand_pos[:, 2] - stick_pos[:, 2]) > 0.06)
        return torch.where(cond, torch.full_like(hand_pos[:, 0], -1.0), torch.full_like(hand_pos[:, 0], 0.7))

