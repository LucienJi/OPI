from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from third_party.custimized_metaworld.oracle_factory import _load_policy_class
from third_party.custimized_metaworld.oracle_factory_th import _load_policy_class as _load_policy_class_th

try:
    import torch._dynamo as dynamo

    _dynamo_disable = dynamo.disable
except Exception:
    def _dynamo_disable(fn):
        return fn


def _normalize_env_name(env_name: str) -> str:
    name = str(env_name).strip()
    if "-v" in name:
        return name.rsplit("-v", 1)[0]
    return name


def _as_action_bound(value: Optional[Iterable[float]], action_dim: int, *, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((action_dim,), float(default), dtype=torch.float32)
    if isinstance(value, (float, int)):
        return torch.full((action_dim,), float(value), dtype=torch.float32)
    return torch.as_tensor(list(value), dtype=torch.float32)


class MetaworldOraclePolicyWrapper(nn.Module):
    def __init__(
        self,
        *,
        env_name: str,
        mode: str,
        variant: int,
        noise_scales: Sequence[float],
        grid_size: float,
        seed: int,
        action_dim: int,
        action_high: Optional[Iterable[float]] = None,
        action_low: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.env_name = _normalize_env_name(env_name)
        self.mode = mode
        self.variant = int(variant)
        self.noise_scales = tuple(float(x) for x in noise_scales)
        self.grid_size = float(grid_size)
        self.seed = int(seed)
        self.action_dim = int(action_dim)

        policy_cls = _load_policy_class(self.env_name)
        self.policy = policy_cls(
            mode=self.mode,
            variant=self.variant,
            noise_scales=self.noise_scales,
            grid_size=self.grid_size,
            seed=self.seed,
        )

        action_high_t = _as_action_bound(action_high, self.action_dim, default=1.0)
        action_low_t = _as_action_bound(action_low, self.action_dim, default=-1.0)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    @staticmethod
    def _to_numpy(observations: Any) -> np.ndarray:
        if isinstance(observations, torch.Tensor):
            return observations.detach().cpu().numpy()
        if isinstance(observations, np.ndarray):
            return observations
        raise TypeError("observations must be a torch.Tensor or numpy.ndarray")

    def _get_action_batch(self, obs_np: np.ndarray) -> np.ndarray:
        if obs_np.ndim == 1:
            action = self.policy.get_action(obs_np)
            return np.asarray(action, dtype=np.float32)[None, :]
        actions = [self.policy.get_action(obs) for obs in obs_np]
        return np.asarray(actions, dtype=np.float32)

    @torch.no_grad()
    def get_action(self, observations: Any, std_multiplier: float = 1.0) -> torch.Tensor:
        obs_np = self._to_numpy(observations)
        actions = self._get_action_batch(obs_np)
        device = observations.device if isinstance(observations, torch.Tensor) else self.action_scale.device
        dtype = observations.dtype if isinstance(observations, torch.Tensor) else torch.float32
        return torch.as_tensor(actions, device=device, dtype=dtype)

    @torch.no_grad()
    def sample_action(self, observations: Any, std_multiplier: float, n_actions: int) -> torch.Tensor:
        obs_np = self._to_numpy(observations)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        batch_size = obs_np.shape[0]
        actions = np.empty((int(n_actions), batch_size, self.action_dim), dtype=np.float32)
        for i in range(int(n_actions)):
            actions[i] = self._get_action_batch(obs_np)
        device = observations.device if isinstance(observations, torch.Tensor) else self.action_scale.device
        dtype = observations.dtype if isinstance(observations, torch.Tensor) else torch.float32
        return torch.as_tensor(actions, device=device, dtype=dtype)

    @torch.no_grad()
    def get_eval_action(self, observations: Any) -> torch.Tensor:
        return self.get_action(observations, std_multiplier=1.0)


class MetaworldTorchOraclePolicyWrapper(nn.Module):
    def __init__(
        self,
        *,
        env_name: str,
        mode: str,
        variant: int,
        noise_scales: Sequence[float],
        grid_size: float,
        seed: int,
        action_dim: int,
        action_high: Optional[Iterable[float]] = None,
        action_low: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.env_name = _normalize_env_name(env_name)
        self.mode = mode
        self.variant = int(variant)
        self.noise_scales = tuple(float(x) for x in noise_scales)
        self.grid_size = float(grid_size)
        self.seed = int(seed)
        self.action_dim = int(action_dim)

        policy_cls = _load_policy_class_th(self.env_name)
        self.policy = policy_cls(
            mode=self.mode,
            variant=self.variant,
            noise_scales=self.noise_scales,
            grid_size=self.grid_size,
            seed=self.seed,
        )

        action_high_t = _as_action_bound(action_high, self.action_dim, default=1.0)
        action_low_t = _as_action_bound(action_low, self.action_dim, default=-1.0)
        self.register_buffer("action_high", action_high_t)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def _to_tensor(self, observations: Any) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return observations
        if isinstance(observations, np.ndarray):
            return torch.as_tensor(observations, device=self.action_scale.device, dtype=torch.float32)
        raise TypeError("observations must be a torch.Tensor or numpy.ndarray")

    def _get_action_batch(self, obs_t: torch.Tensor) -> torch.Tensor:
        obs_t = obs_t if obs_t.dim() == 2 else obs_t.unsqueeze(0)
        return self._policy_get_action(obs_t)

    def _policy_get_action(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.policy.get_action(obs_t)

    @torch.no_grad()
    def get_action(self, observations: Any, std_multiplier: float = 1.0) -> torch.Tensor:
        obs_t = self._to_tensor(observations)
        actions = self._get_action_batch(obs_t)
        return actions

    @torch.no_grad()
    def sample_action(self, observations: Any, std_multiplier: float, n_actions: int) -> torch.Tensor:
        obs_t = self._to_tensor(observations)
        obs_t = obs_t if obs_t.dim() == 2 else obs_t.unsqueeze(0)
        return self.policy.sample_actions(obs_t, int(n_actions))

    @torch.no_grad()
    def get_eval_action(self, observations: Any) -> torch.Tensor:
        return self.get_action(observations, std_multiplier=1.0)
