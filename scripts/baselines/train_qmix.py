#!/usr/bin/env python
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from algo.oracles.oracle_loading import build_oracle_modules_from_cfg
from model.simba import DeterministicSimbaPolicy, SimbaCritics
from env.env_utils import (
    make_env_and_eval_env_from_cfg,
    set_random_seed,
    build_obs_shape_from_obs_config,
)
from data_buffer.replay_buffer import ReplayBuffer


@hydra.main(config_path="../../config", config_name="baselines_configs/qmix/hopper/qmix_hopper", version_base=None)
def train_qmix(cfg: DictConfig):
    from algo.baselines.qmix import QMPAlgo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(cfg.training.seed)

    OmegaConf.set_struct(cfg, False)
    artifacts = build_artifact_paths(cfg)
    ensure_artifact_dirs(artifacts)
    update_cfg_with_artifacts(cfg, artifacts)

    env, eval_env, obs_config = make_env_and_eval_env_from_cfg(cfg)
    act_dim = env.action_space.shape[0]
    obs_shape = build_obs_shape_from_obs_config(obs_config)

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

    oracles_dict = cfg.get("oracles_dict", None) or cfg.get("source_policies", None) or {}
    if not isinstance(oracles_dict, dict):
        oracles_dict = dict(oracles_dict)

    default_oracle_actor_args = (
        cfg.get("oracle_actor_args", None)
        or cfg.get("mlp_based_oracle_args", None)
    )
    default_oracle_critic_args = (
        cfg.get("oracle_critic_args", None)
        or cfg.get("rl_critic_args", None)
    )

    oracles_actors = {}
    if len(oracles_dict) > 0:
        if default_oracle_actor_args is None:
            raise ValueError("Missing oracle actor args in config (oracle_actor_args/source_actor_args/actor_args).")
        if default_oracle_critic_args is None:
            raise ValueError("Missing oracle critic args in config (oracle_critic_args/rl_critic_args).")
        oracles_actors, _, _ = build_oracle_modules_from_cfg(
            cfg=cfg,
            obs_shape=obs_shape,
            act_dim=act_dim,
            device=device,
            oracles_dict=oracles_dict,
            default_oracle_actor_args=default_oracle_actor_args,
            default_oracle_critic_args=default_oracle_critic_args,
        )

    trainable_params = sum(p.numel() for p in learner_actor.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in learner_critic.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")

    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved
    if "num_bins" not in args and "rl_critic_args" in cfg:
        args["num_bins"] = int(cfg.rl_critic_args.num_bins)
        args["min_v"] = float(cfg.rl_critic_args.min_v)
        args["max_v"] = float(cfg.rl_critic_args.max_v)

    agent = QMPAlgo(
        learner_actor=learner_actor,
        learner_critic=learner_critic,
        oracles_actors=oracles_actors,
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

    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

    print("Starting training (QMP)...")
    agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
    print("Training completed.")


if __name__ == "__main__":
    train_qmix()
