import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import TanhTransform


EPS = 1e-8

def l2normalize(x, dim=-1, eps=EPS):
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)

import re

def l2normalize_network(
    network: nn.Module,
    eps: float = 1e-8,
    regex: str = None
) -> nn.Module:
    def _normalize_weight(weight):
        l2norm = torch.linalg.norm(weight, ord=2, dim=1, keepdim=True)
        weight.div_(torch.clamp(l2norm, min=eps))

    with torch.no_grad():
        if regex is None:
            weights = getattr(network, "_l2norm_weight_cache", None)
            if weights is None:
                weights = [m.weight for _, m in network.named_modules() if isinstance(m, HyperDense)]
                network._l2norm_weight_cache = weights
            for weight in weights:
                _normalize_weight(weight)
        else:
            for name, module in network.named_modules():
                if isinstance(module, HyperDense) and re.search(regex, name):
                    _normalize_weight(module.weight)

    return network


class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.init_val = init
        self.scale_val = scale

        self.scaler = nn.Parameter(torch.full((dim,), 1.0 * scale))

        self.forward_scaler = init / scale

    def forward(self, x):
        return x * self.scaler * self.forward_scaler

class HyperDense(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.weight, gain=1.0)

class HyperMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps

        self.w1 = HyperDense(in_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, out_dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, dim=-1)
        return x

class HyperEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float
    ):
        super().__init__()
        self.c_shift = c_shift

        self.w = HyperDense(input_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x):
        shape = x.shape[:-1] + (1,)
        new_axis = torch.ones(shape, device=x.device, dtype=x.dtype) * self.c_shift

        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, dim=-1)
        return x

class HyperLERPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4
    ):
        super().__init__()

        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
        )

        self.alpha_scaler = Scaler(
            hidden_dim,
            init=alpha_init,
            scale=alpha_scale
        )

    def forward(self, x):
        residual = x

        out = self.mlp(x)

        out = residual + self.alpha_scaler(out - residual)

        out = l2normalize(out, dim=-1)
        return out


class HyperNormalTanhPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean_w1 = HyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.mean_w2 = HyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        self.std_w1 = HyperDense(hidden_dim, hidden_dim)
        self.std_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.std_w2 = HyperDense(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x, temperature: float = 1.0):
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        log_std = self.std_w1(x)
        log_std = self.std_scaler(log_std)
        log_std = self.std_w2(log_std) + self.std_bias

        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + torch.tanh(log_std)
        )

        std = torch.exp(log_std) * temperature

        base_dist = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        info = {
            "mean": mean,
            "std": std,
            "log_std": log_std,
        }

        return dist, info

class HyperDeterministicPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
    ):
        super().__init__()

        self.mean_w1 = HyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.mean_w2 = HyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))


    def forward(self, x):
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        return mean


class HyperCategoricalValue(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float,
        scaler_scale: float
    ):
        super().__init__()
        self.num_bins = num_bins

        self.w1 = HyperDense(hidden_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_bins))

        bin_values = torch.linspace(min_v, max_v, num_bins).reshape(1, -1)
        self.register_buffer('bin_values', bin_values)

    def forward(self, x):
        value_feat = self.w1(x)
        value_feat = self.scaler(value_feat)
        logits = self.w2(value_feat) + self.bias

        log_prob = F.log_softmax(logits, dim=-1)

        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1, keepdim=True)

        info = {"log_prob": log_prob}
        return value, info


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float
    ):
        super().__init__()

        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.predictor = HyperDeterministicPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations):
        y = self.embedder(observations)
        z = self.encoder(y)
        mean = self.predictor(z)
        return mean


class CategoricalCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()

        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=-1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        return q, info

class CategoricalValue(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()

        self.embedder = HyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations,):
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        value, info = self.predictor(z)
        return value, info
