from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt

from metaworld.policies.policy import Policy


class BaseOraclePolicy(Policy):
    def __init__(
        self,
        mode: Literal["spatial", "condition"] = "spatial",
        variant: int = 0,
        noise_scales: Tuple[float, float] = (0.0, 0.05),
        grid_size: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.mode = mode
        self.variant = variant
        self.low_scale, self.high_scale = noise_scales
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

    def _get_noise(
        self, hand_pos: npt.NDArray[np.float64], current_stage_id: int
    ) -> npt.NDArray[np.float64]:
        if self.mode == "spatial":
            x_idx = int(np.floor(hand_pos[0] / self.grid_size))
            y_idx = int(np.floor(hand_pos[1] / self.grid_size))
            is_even_cell = (x_idx + y_idx) % 2 == 0
            is_low_noise = is_even_cell if self.variant == 0 else not is_even_cell
        elif self.mode == "condition":
            is_even_stage = current_stage_id % 2 == 0
            is_low_noise = is_even_stage if self.variant == 0 else not is_even_stage
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        scale = self.low_scale if is_low_noise else self.high_scale
        return self.rng.normal(0.0, scale, size=3)

    def _clip_delta_pos(
        self, delta_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.clip(delta_pos, -1.0, 1.0)

    def _maybe_add_noise(
        self,
        desired_pos: npt.NDArray[np.float64],
        hand_pos: npt.NDArray[np.float64],
        current_stage_id: int,
    ) -> npt.NDArray[np.float64]:
        return desired_pos + self._get_noise(hand_pos, current_stage_id)

