from __future__ import annotations

import torch

from .oracle_base import TorchBaseOraclePolicy


class SawyerDrawerOpenOraclePolicy(TorchBaseOraclePolicy):
    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_pos": obs[:, :3],
            "gripper": obs[:, 3],
            "drwr_pos": obs[:, 4:7],
            "unused_info": obs[:, 7:],
        }

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        o_d = self._parse_obs(obs_t)
        desired_pos, stage_id, p = self._desired_pos_with_p(o_d)
        grab_effort = self._grab_effort(o_d)
        desired_pos = self._maybe_add_noise(desired_pos, o_d["hand_pos"], stage_id)
        # p is (batch_size,); make it (batch_size, 1) so (batch_size, 3) * p broadcasts
        delta = self._clip_delta_pos(self._move(o_d["hand_pos"], desired_pos, p=p.unsqueeze(-1)))
        return self._action(delta, grab_effort)

    def _desired_pos_with_p(self, o_d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (desired_pos, stage_id, movement_speed_p) for drawer_open oracle."""
        device = o_d["hand_pos"].device
        dtype = o_d["hand_pos"].dtype
        pos_curr = o_d["hand_pos"]
        pos_drwr = o_d["drwr_pos"] + torch.tensor([0.0, 0.0, -0.02], device=device, dtype=dtype)

        desired = pos_drwr + torch.tensor([0.0, -0.06, 0.0], device=device, dtype=dtype)
        stage = torch.full((pos_curr.shape[0],), 2, device=device, dtype=torch.long)
        p = torch.full((pos_curr.shape[0],), 50.0, device=device, dtype=dtype)

        cond0 = torch.linalg.norm(pos_curr[:, :2] - pos_drwr[:, :2], dim=-1) > 0.06
        mask = cond0
        desired = torch.where(
            mask.unsqueeze(-1),
            pos_drwr + torch.tensor([0.0, 0.0, 0.3], device=device, dtype=dtype),
            desired,
        )
        stage = torch.where(mask, torch.zeros_like(stage), stage)
        p = torch.where(mask, torch.full_like(p, 4.0), p)

        remaining = ~cond0
        cond1 = torch.abs(pos_curr[:, 2] - pos_drwr[:, 2]) > 0.04
        mask = remaining & cond1
        desired = torch.where(mask.unsqueeze(-1), pos_drwr, desired)
        stage = torch.where(mask, torch.ones_like(stage), stage)
        p = torch.where(mask, torch.full_like(p, 4.0), p)

        return desired, stage, p

    def _desired_pos(self, o_d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (desired_pos, stage_id) for base class compatibility."""
        desired, stage, _ = self._desired_pos_with_p(o_d)
        return desired, stage

    @staticmethod
    def _grab_effort(o_d: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.full_like(o_d["hand_pos"][:, 0], -1.0)

    def sample_actions(self, obs: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Override to handle variable movement speed p per observation.

        This oracle has stage-dependent movement speeds, so we need custom vectorized logic.
        """
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        batch_size = obs_t.shape[0]
        o_d = self._parse_obs(obs_t)

        # Compute deterministic parts once (includes variable p)
        desired_pos, stage_id, p = self._desired_pos_with_p(o_d)  # (batch_size, 3), (batch_size,), (batch_size,)
        grab_effort = self._grab_effort(o_d)  # (batch_size,)
        hand_pos = o_d["hand_pos"]  # (batch_size, 3)

        # Add noise for all samples at once: (n_samples, batch_size, 3)
        noisy_desired_pos = self._maybe_add_noise_vectorized(
            desired_pos, hand_pos, stage_id, n_samples
        )

        # Compute delta for all samples with variable p
        # hand_pos: (batch_size, 3) -> (1, batch_size, 3) for broadcasting
        # p: (batch_size,) -> (1, batch_size, 1) for broadcasting
        delta = self._move(hand_pos.unsqueeze(0), noisy_desired_pos, p=p.unsqueeze(0).unsqueeze(-1))
        delta = self._clip_delta_pos(delta)  # (n_samples, batch_size, 3)

        # grab_effort: (batch_size,) -> (n_samples, batch_size, 1)
        grab_effort_expanded = grab_effort.unsqueeze(0).unsqueeze(-1).expand(n_samples, batch_size, 1)

        return torch.cat([delta, grab_effort_expanded], dim=-1)  # (n_samples, batch_size, 4)

