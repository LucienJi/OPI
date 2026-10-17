from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from algo.algo_utils import ObservationNormalizer
from model.simba import DeterministicSimbaPolicy


class DeterministicSimbaOracleWrapper(nn.Module):
    def __init__(
        self,
        *,
        obs_shape,
        action_dim: int,
        model_args: Any,
        obs_normalizer_state: Optional[dict] = None,
        normalize_observations: bool = True,
    ):
        super().__init__()
        self.normalize_observations = bool(normalize_observations)

        self.policy = DeterministicSimbaPolicy(
            obs_shape=obs_shape,
            action_dim=action_dim,
            model_args=model_args,
        )

        self.obs_normalizer: Optional[ObservationNormalizer] = ObservationNormalizer(obs_dim=int(self.policy.obs_dim))
        if obs_normalizer_state is not None:
            self.obs_normalizer.load_state_dict(obs_normalizer_state)

    @property
    def action_high(self) -> torch.Tensor:
        return self.policy.action_high

    @property
    def action_low(self) -> torch.Tensor:
        return self.policy.action_low

    @property
    def action_scale(self) -> torch.Tensor:
        return self.policy.action_scale

    @property
    def action_bias(self) -> torch.Tensor:
        return self.policy.action_bias

    @staticmethod
    def _extract_obs_tensor(observations: Any) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return observations
        if isinstance(observations, dict) and "robot_state" in observations:
            v = observations["robot_state"]
            if not isinstance(v, torch.Tensor):
                raise TypeError("observations['robot_state'] must be a torch.Tensor")
            return v
        raise TypeError("observations must be a torch.Tensor or a dict containing key 'robot_state'")

    def _maybe_normalize_obs_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        if not self.normalize_observations:
            return obs_t
        if self.obs_normalizer is None:
            return obs_t
        return self.obs_normalizer.normalize_tensor(obs_t)

    @torch.no_grad()
    def get_action(self, observations: Any, std: float):
        obs_t = self._extract_obs_tensor(observations)
        obs_t = self._maybe_normalize_obs_tensor(obs_t)
        return self.policy.get_action(obs_t, float(std))

    @torch.no_grad()
    def sample_action(self, observations: Any, std: float, n_actions: int):
        obs_t = self._extract_obs_tensor(observations)
        obs_t = self._maybe_normalize_obs_tensor(obs_t)
        return self.policy.sample_action(obs_t, float(std), int(n_actions))

    @torch.no_grad()
    def get_eval_action(self, observations: Any):
        obs_t = self._extract_obs_tensor(observations)
        obs_t = self._maybe_normalize_obs_tensor(obs_t)
        return self.policy.get_eval_action(obs_t)

    def load_policy_state_dict(self, state_dict: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        self.policy.load_state_dict(state_dict, strict=strict)

    def load_obs_normalizer_state(self, state: dict) -> None:
        if self.obs_normalizer is None:
            self.obs_normalizer = ObservationNormalizer(obs_dim=int(self.policy.obs_dim))
        self.obs_normalizer.load_state_dict(state)
