from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.utils import _standard_normal
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def layer_init(layer, std=None, bias_const=0.0):
    if std is None:
        torch.nn.init.orthogonal_(layer.weight)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(layer_init(nn.Linear(hidden_size, hidden_size)), nn.LayerNorm(hidden_size), nn.ReLU(),
                                   layer_init(nn.Linear(hidden_size, hidden_size)), nn.LayerNorm(hidden_size))

    def forward(self, x):
        out = self.block(x)
        return x + out


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, bronet_layers):
        super().__init__()
        self.projection = nn.Sequential(layer_init(nn.Linear(input_size, hidden_size)), nn.LayerNorm(hidden_size), nn.ReLU())
        self.embedding = nn.ModuleList()
        for i in range(bronet_layers):
            self.embedding.append(Block(hidden_size))
        if output_size is not None:
            self.final_layer = layer_init(nn.Linear(hidden_size, output_size), std=1e-3, bias_const=0.0)
        self.bronet_layers = bronet_layers
        self.output_size = output_size

    def forward(self, x):
        x = self.projection(x)
        for i in range(self.bronet_layers):
            x = self.embedding[i](x)
        if self.output_size is not None:
            x = self.final_layer(x)
        return x

class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-5):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped_x.detach()

    def rsample(self, sample_shape=torch.Size(), clip=None):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = torch.tanh(self.loc) + eps
        return self._clamp(x)

    def sample(self, sample_shape=torch.Size(), clip=None):
        with torch.no_grad():
            return self.rsample(sample_shape, clip=clip)

    @property
    def mean_truncated(self):
        mean = torch.tanh(self.loc)
        return mean

class TanhNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, eps=1e-5):
        super().__init__(loc, scale, validate_args=False)
        self.eps = eps
        self.pretanh_value = None

    def rsample(self, sample_shape=torch.Size()):
        z = super().rsample(sample_shape)
        self.pretanh_value = z
        return torch.tanh(z)

    def resample_with_logprob(self, sample_shape=torch.Size()):
        u = super().rsample(sample_shape)
        a = torch.tanh(u)

        log_prob_u = super().log_prob(u)
        squash_corr = 2.0 * (math.log(2.0)- u - F.softplus(-2.0 * u))
        log_pi = (log_prob_u - squash_corr).sum(dim=-1, keepdim=True)

        a = torch.clamp(a, -1.0 + self.eps, 1.0 - self.eps)
        log_pi = torch.clamp(log_pi, min=-100.0, max=100.0)
        return a, log_pi

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value):
        value = value.float()
        clamped = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(clamped)
        base_log_prob = super().log_prob(pre_tanh)
        correction = torch.log(1 - clamped.pow(2) + self.eps)
        return (base_log_prob - correction).sum(dim=-1, keepdim=True)

    @property
    def mean_tanh(self):
        return torch.tanh(self.loc)

class PolicyBase(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, bronet_layers: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.bronet_layers = int(bronet_layers)
        self.net = Net(self.input_dim, None, self.hidden_size, self.bronet_layers)

    @staticmethod
    def _extract_obs(observations: Any) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return observations
        if isinstance(observations, dict) and "robot_state" in observations:
            return observations["robot_state"]
        raise TypeError("observations must be a torch.Tensor or a dict containing key 'robot_state'")


class OraclePolicyBase(nn.Module):
    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        model_args: Dict[str, Any],
    ):
        super().__init__()
        self.obs_dim = np.prod(obs_shape)
        self.action_dim = int(action_dim)
        self.hidden_size = int(model_args.hidden_size)
        self.n_layers = int(model_args.n_layers)
        self._register_action_bounds(action_high=model_args.action_high, action_low=model_args.action_low)

    def _register_action_bounds(
        self,
        *,
        action_high: Optional[torch.Tensor],
        action_low: Optional[torch.Tensor],
    ):
        if action_high is None or action_low is None:
            ah = torch.ones(self.action_dim, dtype=torch.float32)
            al = -torch.ones(self.action_dim, dtype=torch.float32)
        else:
            ah = torch.as_tensor(action_high, dtype=torch.float32)
            al = torch.as_tensor(action_low, dtype=torch.float32)

        self.register_buffer("action_high", ah)
        self.register_buffer("action_low", al)
        self.register_buffer("action_scale", (ah - al) / 2.0)
        self.register_buffer("action_bias", (ah + al) / 2.0)

    def set_optimizer(self, actor_lr: float, weight_decay: float = 1e-2):
        actor_params = list(self.policy.parameters())
        extra_args = {"foreach": True}
        return torch.optim.AdamW(actor_params, lr=actor_lr, weight_decay=weight_decay, **extra_args)

    def update_target(self, tau: float = 0.005):
        return None


class _GaussianPolicyNet(PolicyBase):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, bronet_layers: int):
        super().__init__(input_dim=input_dim, hidden_size=hidden_size, bronet_layers=bronet_layers)
        self.action_dim = int(action_dim)

        self.means = layer_init(nn.Linear(self.hidden_size, self.action_dim), std=1e-3, bias_const=0.0)
        self.log_stds = layer_init(nn.Linear(self.hidden_size, self.action_dim), std=1e-3, bias_const=0.62)

    def forward(self, observations: Any, std_multiplier: float = 1.0):
        x = self.net(self._extract_obs(observations))
        means = self.means(x)
        log_stds = self.log_stds(x)
        log_stds = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (1 + torch.tanh(log_stds))
        stds = log_stds.exp() * (float(std_multiplier) + 1e-8)
        return means, stds


class _DeterministicPolicyNet(PolicyBase):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int, bronet_layers: int):
        super().__init__(input_dim=input_dim, hidden_size=hidden_size, bronet_layers=bronet_layers)
        self.action_dim = int(action_dim)
        self.means = layer_init(nn.Linear(self.hidden_size, self.action_dim), std=1e-3, bias_const=0.0)

    def forward(self, observations: Any):
        x = self.net(self._extract_obs(observations))
        return self.means(x)


class StochasticPolicy(OraclePolicyBase):
    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        model_args: Dict[str, Any],
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, model_args=model_args)
        self.policy = _GaussianPolicyNet(self.obs_dim, self.action_dim, self.hidden_size, self.n_layers)

    def forward(self, observations: Any, std_multiplier: float = 1.0):
        return self.policy(observations, std_multiplier=std_multiplier)

    def get_action(self, observations: Any, std_multiplier: float = 1.0):
        means, stds = self.policy(observations, std_multiplier=std_multiplier)
        return TanhNormal(means, stds).rsample()


    def sample_action(self, observations: Any, std_multiplier: float, n_actions: int):
        means, stds = self.policy(observations, std_multiplier=std_multiplier)
        return TanhNormal(means, stds).sample(sample_shape=(n_actions,))

    def get_eval_action(self, observations: Any):
        means, stds = self.policy(observations, std_multiplier=1.0)
        return TanhNormal(means, stds).mean_tanh


class DeterministicPolicy(OraclePolicyBase):
    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        model_args: Dict[str, Any],
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, model_args=model_args)
        self.noise_clip = float(model_args.noise_clip)

        self.policy = _DeterministicPolicyNet(self.obs_dim, self.action_dim, self.hidden_size, self.n_layers)

    def forward(self, observations: Any):
        return self.policy(observations)


    def get_action(self, observations: Any, std: float):
        means = self.policy(observations)
        truncated_normal = TruncatedNormal(means, std)
        return truncated_normal.rsample(clip=self.noise_clip)

    def sample_action(self, observations: Any, std: float, n_actions: int):
        means = self.policy(observations)
        truncated_normal = TruncatedNormal(means, std)
        return truncated_normal.sample(sample_shape=(n_actions,), clip=self.noise_clip)

    def get_eval_action(self, observations: Any):
        means = self.policy(observations)
        return TruncatedNormal(means, 1.0).mean_truncated
