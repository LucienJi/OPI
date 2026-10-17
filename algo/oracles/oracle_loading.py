from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf

from env.env_utils import _is_missing_checkpoint, _banner
from model.mlp import DeterministicPolicy, OraclePolicyBase
from model.metaworld_oracle_policy import (
    MetaworldOraclePolicyWrapper,
    MetaworldTorchOraclePolicyWrapper,
)
from model.simba import SimbaCritics, SimbaValues
from model.oracle_wrappers import DeterministicSimbaOracleWrapper


def _load_oracle_model_args(ckpt_path: str):
    file_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(file_dir, "cfg.yaml")
    if not os.path.exists(cfg_path):
        return None
    oracle_cfg = OmegaConf.load(cfg_path)
    for key in ("actor_args", "oracle_actor_args", "source_actor_args", "rl_agent_args", "oracle_simba_actor_args"):
        if key in oracle_cfg:
            return oracle_cfg[key]
    return None


def _build_oracle_modules(*, obs_shape, act_dim, device, actor_args, critic_args):
    actor = DeterministicPolicy(obs_shape=obs_shape, action_dim=act_dim, model_args=actor_args).to(device)
    critic = SimbaCritics(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    value = SimbaValues(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    return actor, critic, value


def _build_oracle_critic_value(*, obs_shape, act_dim, device, critic_args):
    critic = SimbaCritics(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    value = SimbaValues(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    return critic, value


def _is_metaworld_oracle_spec(spec) -> bool:
    if isinstance(spec, str):
        return spec.strip().lower() in {"metaworld", "metaworld_oracle", "mw"}
    if isinstance(spec, dict):
        return str(spec.get("type", "")).strip().lower() in {"metaworld", "metaworld_oracle", "mw"}
    return False


def _resolve_oracle_cfg(cfg, spec):
    base_cfg = cfg.get("oracle_policy", None)
    if base_cfg is None:
        raise ValueError("oracle_policy must be provided in config for scripted oracle loading.")
    if isinstance(spec, dict):
        merged = OmegaConf.merge(base_cfg, spec)
    else:
        merged = base_cfg
    resolved = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(resolved, dict) or len(resolved) == 0:
        raise ValueError("oracle_policy must be a non-empty mapping.")
    return resolved


def _require_keys(cfg_dict: Dict[str, Any], keys, *, context: str):
    missing = [k for k in keys if k not in cfg_dict]
    if missing:
        raise KeyError(f"Missing required {context} keys: {missing}")


def _build_metaworld_oracle_actor(*, cfg, oracle_index: int, oracle_name: str, spec, act_dim: int):
    oracle_cfg = _resolve_oracle_cfg(cfg, spec)
    _require_keys(
        oracle_cfg,
        ["env_name", "mode", "min_noise_scale", "grid_size", "seed", "action_high", "action_low"],
        context="oracle_policy",
    )
    if "max_noise_scales" not in oracle_cfg and "max_noise_scale" not in oracle_cfg:
        raise KeyError("oracle_policy must set either 'max_noise_scales' or 'max_noise_scale'.")


    env_name = oracle_cfg["env_name"]
    mode = oracle_cfg["mode"]
    min_noise = float(oracle_cfg["min_noise_scale"])
    max_noise_scales = oracle_cfg.get("max_noise_scales", None)
    if isinstance(max_noise_scales, (list, tuple)):
        if oracle_index < len(max_noise_scales):
            max_noise = float(max_noise_scales[oracle_index])
        else:
            max_noise = float(max_noise_scales[-1])
    elif max_noise_scales is None:
        if "max_noise_scale" not in oracle_cfg:
            raise KeyError("oracle_policy must provide max_noise_scales or max_noise_scale.")
        max_noise = float(oracle_cfg["max_noise_scale"])
    else:
        max_noise = float(max_noise_scales)
    grid_size = float(oracle_cfg["grid_size"])
    seed = int(oracle_cfg["seed"])

    variants = oracle_cfg["variants"]
    variant = int(variants[oracle_index])


    action_high = oracle_cfg["action_high"]
    action_low = oracle_cfg["action_low"]
    backend = str(oracle_cfg.get("policy_backend", "numpy")).strip().lower()

    wrapper_cls = (
        MetaworldTorchOraclePolicyWrapper
        if backend in {"torch", "pytorch"}
        else MetaworldOraclePolicyWrapper
    )
    return wrapper_cls(
        env_name=env_name,
        mode=mode,
        variant=variant,
        noise_scales=(min_noise, max_noise),
        grid_size=grid_size,
        seed=seed,
        action_dim=act_dim,
        action_high=action_high,
        action_low=action_low,
    )


def _load_oracle_checkpoint(*, name: str, ckpt_path: str, device, actor):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
    print(f"Loaded oracle actor {name} from {ckpt_path}")
    for p in actor.parameters():
        p.requires_grad = False


def _load_simba_oracle_actor_args(ckpt_path: str):
    file_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(file_dir, "cfg.yaml")
    if not os.path.exists(cfg_path):
        return None
    oracle_cfg = OmegaConf.load(cfg_path)
    for key in ("oracle_simba_actor_args", "rl_agent_args", "oracle_actor_args", "actor_args"):
        if key in oracle_cfg:
            return oracle_cfg[key]
    return None


def _build_simba_oracle_modules(*, obs_shape, act_dim, device, actor_args, critic_args):
    actor = DeterministicSimbaOracleWrapper(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=actor_args,
        obs_normalizer_state=None,
        normalize_observations=True,
    ).to(device)
    critic = SimbaCritics(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    value = SimbaValues(obs_shape=obs_shape, action_dim=act_dim, model_args=critic_args).to(device)
    return actor, critic, value


def _load_simba_oracle_checkpoint(*, name: str, ckpt_path: str, device, actor: DeterministicSimbaOracleWrapper):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_policy_state_dict(checkpoint["rl_agent_state_dict"], strict=False)
    actor.load_obs_normalizer_state(checkpoint["obs_rms"])
    print(f"Loaded Simba oracle actor+obs_rms {name} from {ckpt_path}")
    for p in actor.parameters():
        p.requires_grad = False


def build_oracle_modules_from_cfg(
    *,
    cfg,
    obs_shape,
    act_dim: int,
    device: torch.device,
    oracles_dict: Dict[str, Any],
    default_oracle_actor_args,
    default_oracle_critic_args,
) -> Tuple[Dict[str, OraclePolicyBase], Dict[str, SimbaCritics], Dict[str, SimbaValues]]:
    oracles_actors: Dict[str, OraclePolicyBase] = {}
    oracles_critics: Dict[str, SimbaCritics] = {}
    oracles_values: Dict[str, SimbaValues] = {}

    for oracle_index, (name, ckpt_path) in enumerate(oracles_dict.items()):
        if _is_missing_checkpoint(ckpt_path):
            actor, critic, value = _build_oracle_modules(
                obs_shape=obs_shape,
                act_dim=act_dim,
                device=device,
                actor_args=default_oracle_actor_args,
                critic_args=default_oracle_critic_args,
            )
            oracles_actors[name] = actor
            oracles_critics[name] = critic
            oracles_values[name] = value
            _banner(f"Use Random Oracle: {name}")
            continue

        if _is_metaworld_oracle_spec(ckpt_path):
            actor = _build_metaworld_oracle_actor(
                cfg=cfg,
                oracle_index=oracle_index,
                oracle_name=name,
                spec=ckpt_path,
                act_dim=act_dim,
            ).to(device)
            critic, value = _build_oracle_critic_value(
                obs_shape=obs_shape,
                act_dim=act_dim,
                device=device,
                critic_args=default_oracle_critic_args,
            )
            oracles_actors[name] = actor
            oracles_critics[name] = critic
            oracles_values[name] = value
            _banner(f"Use Metaworld Oracle: {name}")
            continue

        oracle_actor_cfg = _load_oracle_model_args(ckpt_path)
        actor_args = oracle_actor_cfg if oracle_actor_cfg is not None else default_oracle_actor_args

        actor, critic, value = _build_oracle_modules(
            obs_shape=obs_shape,
            act_dim=act_dim,
            device=device,
            actor_args=actor_args,
            critic_args=default_oracle_critic_args,
        )
        oracles_actors[name] = actor
        oracles_critics[name] = critic
        oracles_values[name] = value

        print(f"Oracle {name} model_args: actor={actor_args}, critic/value={default_oracle_critic_args}")
        _load_oracle_checkpoint(
            name=name,
            ckpt_path=ckpt_path,
            device=device,
            actor=actor,
        )

    return oracles_actors, oracles_critics, oracles_values


def build_simba_oracle_modules_from_cfg(
    *,
    cfg,
    obs_shape,
    act_dim: int,
    device: torch.device,
    oracles_dict: Dict[str, Any],
    default_oracle_actor_args,
    default_oracle_critic_args,
) -> Tuple[Dict[str, DeterministicSimbaOracleWrapper], Dict[str, SimbaCritics], Dict[str, SimbaValues]]:
    oracles_actors: Dict[str, DeterministicSimbaOracleWrapper] = {}
    oracles_critics: Dict[str, SimbaCritics] = {}
    oracles_values: Dict[str, SimbaValues] = {}

    for name, ckpt_path in oracles_dict.items():
        if _is_missing_checkpoint(ckpt_path):
            actor, critic, value = _build_simba_oracle_modules(
                obs_shape=obs_shape,
                act_dim=act_dim,
                device=device,
                actor_args=default_oracle_actor_args,
                critic_args=default_oracle_critic_args,
            )
            oracles_actors[name] = actor
            oracles_critics[name] = critic
            oracles_values[name] = value
            _banner(f"Use Random Simba Oracle: {name}")
            continue

        actor_args = _load_simba_oracle_actor_args(ckpt_path) or default_oracle_actor_args
        actor, critic, value = _build_simba_oracle_modules(
            obs_shape=obs_shape,
            act_dim=act_dim,
            device=device,
            actor_args=actor_args,
            critic_args=default_oracle_critic_args,
        )
        oracles_actors[name] = actor
        oracles_critics[name] = critic
        oracles_values[name] = value

        print(f"Simba Oracle {name} model_args: actor={actor_args}, critic/value={default_oracle_critic_args}")
        _load_simba_oracle_checkpoint(
            name=name,
            ckpt_path=ckpt_path,
            device=device,
            actor=actor,
        )

    return oracles_actors, oracles_critics, oracles_values
