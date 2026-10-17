from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch


class TorchBaseOraclePolicy:
    # Movement speed for delta computation. Subclasses should override this.
    _movement_speed: float = 10.0

    def __init__(
        self,
        mode: Literal["spatial", "condition"] = "spatial",
        variant: int = 0,
        noise_scales: Tuple[float, float] = (0.0, 0.05),
        grid_size: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.mode = mode
        self.variant = int(variant)
        self.low_scale, self.high_scale = float(noise_scales[0]), float(noise_scales[1])
        self.grid_size = float(grid_size)
        self.seed = int(seed)

    def _get_noise(self, hand_pos: torch.Tensor, current_stage_id: torch.Tensor) -> torch.Tensor:
        if self.mode == "spatial":
            x_idx = torch.floor(hand_pos[:, 0] / self.grid_size).to(torch.long)
            y_idx = torch.floor(hand_pos[:, 1] / self.grid_size).to(torch.long)
            is_even_cell = (x_idx + y_idx) % 2 == 0
            is_low_noise = is_even_cell if self.variant == 0 else ~is_even_cell
        elif self.mode == "condition":
            is_even_stage = (current_stage_id % 2) == 0
            is_low_noise = is_even_stage if self.variant == 0 else ~is_even_stage
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        device = hand_pos.device
        dtype = hand_pos.dtype
        low = torch.full((hand_pos.shape[0],), self.low_scale, device=device, dtype=dtype)
        high = torch.full((hand_pos.shape[0],), self.high_scale, device=device, dtype=dtype)
        scale = torch.where(is_low_noise, low, high)
        noise = torch.randn((hand_pos.shape[0], 3), device=device, dtype=dtype)
        return noise * scale.unsqueeze(-1)

    @staticmethod
    def _clip_delta_pos(delta_pos: torch.Tensor) -> torch.Tensor:
        return torch.clamp(delta_pos, -1.0, 1.0)

    def _maybe_add_noise(
        self,
        desired_pos: torch.Tensor,
        hand_pos: torch.Tensor,
        current_stage_id: torch.Tensor,
    ) -> torch.Tensor:
        return desired_pos + self._get_noise(hand_pos, current_stage_id)

    def _get_noise_scale(
        self,
        hand_pos: torch.Tensor,
        current_stage_id: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-element noise scale based on mode and variant.

        Args:
            hand_pos: (batch_size, 3) hand positions
            current_stage_id: (batch_size,) stage ids

        Returns:
            (batch_size,) noise scale for each element
        """
        if self.mode == "spatial":
            x_idx = torch.floor(hand_pos[:, 0] / self.grid_size).to(torch.long)
            y_idx = torch.floor(hand_pos[:, 1] / self.grid_size).to(torch.long)
            is_even_cell = (x_idx + y_idx) % 2 == 0
            is_low_noise = is_even_cell if self.variant == 0 else ~is_even_cell
        elif self.mode == "condition":
            is_even_stage = (current_stage_id % 2) == 0
            is_low_noise = is_even_stage if self.variant == 0 else ~is_even_stage
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        device = hand_pos.device
        dtype = hand_pos.dtype
        low = torch.full((hand_pos.shape[0],), self.low_scale, device=device, dtype=dtype)
        high = torch.full((hand_pos.shape[0],), self.high_scale, device=device, dtype=dtype)
        return torch.where(is_low_noise, low, high)

    def _get_noise_vectorized(
        self,
        hand_pos: torch.Tensor,
        current_stage_id: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Generate vectorized noise for n_samples at once.

        Args:
            hand_pos: (batch_size, 3) hand positions
            current_stage_id: (batch_size,) stage ids
            n_samples: number of samples to generate

        Returns:
            (n_samples, batch_size, 3) noise tensor
        """
        device = hand_pos.device
        dtype = hand_pos.dtype
        batch_size = hand_pos.shape[0]

        scale = self._get_noise_scale(hand_pos, current_stage_id)  # (batch_size,)
        noise = torch.randn((n_samples, batch_size, 3), device=device, dtype=dtype)
        # scale: (batch_size,) -> (1, batch_size, 1) for broadcasting
        return noise * scale.unsqueeze(0).unsqueeze(-1)

    def _maybe_add_noise_vectorized(
        self,
        desired_pos: torch.Tensor,
        hand_pos: torch.Tensor,
        current_stage_id: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Add vectorized noise to desired_pos for n_samples.

        Args:
            desired_pos: (batch_size, 3) desired positions
            hand_pos: (batch_size, 3) hand positions
            current_stage_id: (batch_size,) stage ids
            n_samples: number of samples to generate

        Returns:
            (n_samples, batch_size, 3) noisy desired positions
        """
        noise = self._get_noise_vectorized(hand_pos, current_stage_id, n_samples)
        # desired_pos: (batch_size, 3) -> (1, batch_size, 3)
        return desired_pos.unsqueeze(0) + noise

    @staticmethod
    def _move(from_xyz: torch.Tensor, to_xyz: torch.Tensor, p: torch.Tensor | float) -> torch.Tensor:
        return (to_xyz - from_xyz) * p

    @staticmethod
    def _action(delta_pos: torch.Tensor, grab_effort: torch.Tensor) -> torch.Tensor:
        if grab_effort.dim() == 1:
            grab_effort = grab_effort.unsqueeze(-1)
        return torch.cat([delta_pos, grab_effort], dim=-1)

    def _parse_obs(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse observation tensor into named components. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _parse_obs")

    def _desired_pos(self, o_d: Dict[str, torch.Tensor]) -> tuple:
        """Compute desired position and stage id. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _desired_pos")

    def _grab_effort(self, o_d: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute grab effort. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _grab_effort")

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get a single action for each observation. Subclasses may override."""
        raise NotImplementedError("Subclasses must implement get_action")

    def sample_actions(self, obs: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Generate n_samples actions for each observation in a vectorized manner.

        This method computes the deterministic parts (desired_pos, stage_id, grab_effort)
        once, then generates all noise samples at once and computes all actions in parallel.
        This is more efficient than calling get_action n_samples times and is compile-friendly.

        Args:
            obs: (batch_size, obs_dim) or (obs_dim,) observation tensor
            n_samples: number of action samples to generate

        Returns:
            (n_samples, batch_size, action_dim) tensor of actions
        """
        obs_t = obs if obs.dim() == 2 else obs.unsqueeze(0)
        batch_size = obs_t.shape[0]
        o_d = self._parse_obs(obs_t)

        # Compute deterministic parts once
        desired_pos, stage_id = self._desired_pos(o_d)  # (batch_size, 3), (batch_size,)
        grab_effort = self._grab_effort(o_d)  # (batch_size,)
        hand_pos = o_d["hand_pos"]  # (batch_size, 3)

        # Add noise for all samples at once: (n_samples, batch_size, 3)
        noisy_desired_pos = self._maybe_add_noise_vectorized(
            desired_pos, hand_pos, stage_id, n_samples
        )

        # Compute delta for all samples
        # hand_pos: (batch_size, 3) -> (1, batch_size, 3) for broadcasting
        delta = self._move(hand_pos.unsqueeze(0), noisy_desired_pos, p=self._movement_speed)
        delta = self._clip_delta_pos(delta)  # (n_samples, batch_size, 3)

        # grab_effort: (batch_size,) -> (n_samples, batch_size, 1)
        grab_effort_expanded = grab_effort.unsqueeze(0).unsqueeze(-1).expand(n_samples, batch_size, 1)

        return torch.cat([delta, grab_effort_expanded], dim=-1)  # (n_samples, batch_size, 4)

