#!/usr/bin/env python

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algo.artifacts import (
    build_artifact_paths,
    ensure_artifact_dirs,
    update_cfg_with_artifacts,
    write_run_manifest,
)
from algo.oracles.oracle_loading import build_oracle_modules_from_cfg
from data_buffer.replay_buffer import ReplayBuffer
from env.env_utils import (
    build_obs_shape_from_obs_config,
    infer_env_category,
    make_env_and_eval_env_from_cfg,
    set_random_seed,
)
from model.simba import (
    DeterministicSimbaPolicy,
    SimbaCritics,
    SimbaValues,
)


def _count_trainable_params_mb(*modules) -> float:
    total = 0
    for module in modules:
        total += sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total / 1024 / 1024


def _require_positive_int(value, field_name: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{field_name} must be positive, got {value_int}.")
    return value_int


def _validate_opi_cfg(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    env_category = infer_env_category(cfg)
    if env_category == "dmc":
        if getattr(env_cfg, "domain_name", None) is None or getattr(env_cfg, "task_name", None) is None:
            raise ValueError("OPI on DMC requires cfg.env.domain_name and cfg.env.task_name.")
    elif env_category == "myosuite":
        if getattr(env_cfg, "env_name", None) is None and getattr(env_cfg, "myosuite_env_name", None) is None:
            raise ValueError("OPI on myosuite requires cfg.env.env_name or cfg.env.myosuite_env_name.")
    else:
        if getattr(env_cfg, "env_name", None) is None:
            raise ValueError(f"OPI on {env_category} requires cfg.env.env_name.")

    obs_config = cfg.get("obs_config", {})
    if not obs_config:
        raise ValueError("OPI training requires a non-empty obs_config.")

    training_cfg = cfg.training
    for field_name in ("total_timesteps", "capacity", "eval_every", "save_freq", "n_oracles"):
        _require_positive_int(training_cfg[field_name], f"training.{field_name}")

    discount = float(training_cfg.discount)
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"training.discount must be in (0, 1], got {discount}.")

    oracles_dict = cfg.get("oracles_dict", None)
    if not oracles_dict:
        raise ValueError("OPI training requires a non-empty oracles_dict mapping.")


def _write_cfg_snapshot(cfg: DictConfig) -> None:
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w", encoding="utf-8") as handle:
        OmegaConf.save(cfg, handle, resolve=True)


@hydra.main(config_path="../config", config_name="metaworld/OPI_assembly", version_base=None)
def train_OPI(cfg: DictConfig) -> None:
    from algo.oracles.OPI import OPI

    _validate_opi_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.training.seed)

    OmegaConf.set_struct(cfg, False)
    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    env = None
    eval_env = None
    try:
        env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)
        action_shape = getattr(env.action_space, "shape", None)
        if not action_shape or len(action_shape) != 1:
            raise ValueError(f"OPI expects a 1D continuous action space, got shape={action_shape!r}.")

        act_dim = action_shape[0]
        obs_shape = build_obs_shape_from_obs_config(obs_config)
        if len(obs_shape) != 1 or obs_shape[0] <= 0:
            raise ValueError(f"Failed to build a flat observation shape from obs_config: {obs_shape!r}")

        learner_actor = DeterministicSimbaPolicy(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_agent_args,
        ).to(device)
        learner_critic = SimbaCritics(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_critic_args,
        ).to(device)
        learner_value = SimbaValues(
            obs_shape=obs_shape,
            action_dim=act_dim,
            model_args=cfg.rl_critic_args,
        ).to(device)

        n_oracles = int(cfg.training.n_oracles)
        oracles_dict = cfg.oracles_dict
        if len(oracles_dict) != n_oracles:
            raise ValueError(
                f"Number of entries in oracles_dict ({len(oracles_dict)}) must match training.n_oracles ({n_oracles})."
            )
        default_oracle_actor_args = (
            cfg.get("oracle_actor_args", None)
            or cfg.get("mlp_based_oracle_args", None)
        )
        if default_oracle_actor_args is None:
            raise ValueError("Missing oracle actor args in config (oracle_actor_args/mlp_based_oracle_args).")
        default_oracle_critic_args = cfg.oracle_critic_args

        oracles_actors, oracles_critics, oracles_values = build_oracle_modules_from_cfg(
            cfg=cfg,
            obs_shape=obs_shape,
            act_dim=act_dim,
            device=device,
            oracles_dict=oracles_dict,
            default_oracle_actor_args=default_oracle_actor_args,
            default_oracle_critic_args=default_oracle_critic_args,
        )

        print(f"Trainable parameters: {_count_trainable_params_mb(learner_actor, learner_critic):.2f} MB")

        args = OmegaConf.to_container(cfg.training, resolve=True)
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(args, dict) or not isinstance(cfg_resolved, dict):
            raise TypeError("Expected resolved Hydra configs to convert to dictionaries.")
        args["full_config"] = cfg_resolved

        agent = OPI(
            n_oracles=n_oracles,
            oracles_actors=oracles_actors,
            oracles_critics=oracles_critics,
            oracles_values=oracles_values,
            learner_actor=learner_actor,
            learner_critic=learner_critic,
            learner_value=learner_value,
            args=args,
            device=device,
        )

        replay_buffer = ReplayBuffer(
            replay_dir=cfg.training.replay_dir,
            observation_shape=obs_shape,
            action_dim=act_dim,
            capacity=cfg.training.capacity,
            discount=cfg.training.discount,
            device=device.type,
            save_episodes_to_disk=cfg.training.save_episodes_to_disk,
        )

        _write_cfg_snapshot(cfg)
        write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

        print("Starting training (OPI)...")
        agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
        print("Training completed.")
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    train_OPI()
