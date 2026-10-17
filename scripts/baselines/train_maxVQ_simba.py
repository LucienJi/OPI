#!/usr/bin/env python
import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.artifacts import build_artifact_paths, ensure_artifact_dirs, update_cfg_with_artifacts, write_run_manifest
from algo.oracles.oracle_loading import build_simba_oracle_modules_from_cfg
from model.simba import DeterministicSimbaPolicy
from env.env_utils import (
    make_env_and_eval_env_from_cfg,
    set_random_seed,
    build_obs_shape_from_obs_config,
)
from data_buffer.replay_buffer import ReplayBuffer


@hydra.main(config_path="../../config", config_name="baselines_configs/maxVQ/bipedal/learner", version_base=None)
def train_VQ_simba(cfg: DictConfig):
    from algo.baselines.maxVQ import OracleSelectionAlgo

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

    oracles_dict = cfg.get("oracles_dict", None) or cfg.get("source_policies", None) or {}
    if not isinstance(oracles_dict, dict):
        oracles_dict = dict(oracles_dict)
    if len(oracles_dict) == 0:
        raise ValueError("VQ requires oracles_dict/source_policies to be specified in the config.")

    selection_mode = cfg.get("selection_mode", None)
    if selection_mode is None and "training" in cfg:
        selection_mode = cfg.training.get("selection_mode", None)
    if selection_mode is None:
        selection_mode = "max_v"
    selection_mode = str(selection_mode).lower().strip()

    default_oracle_actor_args = cfg.get("simba_based_oracle_args", None)
    if default_oracle_actor_args is None:
        raise ValueError("Missing simba_based_oracle_args in config. Please include config/oracles/simba_based_oracle_args.yaml.")

    default_estimator_args = cfg.get("oracle_critic_args", None) or cfg.get("rl_critic_args", None)
    if default_estimator_args is None:
        raise ValueError("Missing oracle critic/value args in config (oracle_critic_args/rl_critic_args).")

    oracles_actors, oracles_critics_all, oracles_values_all = build_simba_oracle_modules_from_cfg(
        cfg=cfg,
        obs_shape=obs_shape,
        act_dim=act_dim,
        device=device,
        oracles_dict=oracles_dict,
        default_oracle_actor_args=default_oracle_actor_args,
        default_oracle_critic_args=default_estimator_args,
    )
    oracles_values = oracles_values_all if selection_mode == "max_v" else {}
    oracles_critics = oracles_critics_all if selection_mode == "max_q" else {}

    trainable_params = sum(p.numel() for p in learner_actor.parameters() if p.requires_grad)
    if selection_mode == "max_v":
        for value in oracles_values.values():
            trainable_params += sum(p.numel() for p in value.parameters() if p.requires_grad)
    else:
        for critic in oracles_critics.values():
            trainable_params += sum(p.numel() for p in critic.parameters() if p.requires_grad)
    trainable_params_mb = trainable_params / 1024 / 1024
    print(f"Trainable parameters: {trainable_params_mb:.2f} MB")

    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    args["full_config"] = cfg_resolved

    agent = OracleSelectionAlgo(
        selection_mode=selection_mode,
        learner_actor=learner_actor,
        oracles_actors=oracles_actors,
        oracles_values=oracles_values if selection_mode == "max_v" else None,
        oracles_critics=oracles_critics if selection_mode == "max_q" else None,
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

    print(f"Starting training (VQ Simba mode={selection_mode})...")
    agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)
    print("Training completed.")


if __name__ == "__main__":
    train_VQ_simba()
