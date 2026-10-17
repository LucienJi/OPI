#!/usr/bin/env python
import os
import sys
import torch
import hydra
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
from algo.oracles.oracle_loading import build_simba_oracle_modules_from_cfg
from model.simba import DeterministicSimbaPolicy, SimbaCritics, SimbaValues

from env.env_utils import (
    make_env_and_eval_env_from_cfg,
    set_random_seed,
    build_obs_shape_from_obs_config,
)
from data_buffer.replay_buffer import ReplayBuffer


@hydra.main(config_path="../config", config_name="box2d/OPI_bipedal_learner", version_base=None)
def train_OPI_simba(cfg: DictConfig):
    from algo.oracles.OPI import OPI

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
    learner_value = SimbaValues(
        obs_shape=obs_shape,
        action_dim=act_dim,
        model_args=cfg.rl_critic_args,
    ).to(device)

    n_oracles = int(cfg.training.n_oracles)
    oracles_dict = cfg.oracles_dict
    assert len(oracles_dict) == n_oracles, "Number of oracles in oracles_dict must match n_oracles"

    default_oracle_actor_args = cfg.get("simba_based_oracle_args", None)
    if default_oracle_actor_args is None:
        raise ValueError("Missing simba_based_oracle_args in config. Please include config/oracles/simba_based_oracle_args.yaml.")
    default_oracle_critic_args = cfg.get("oracle_critic_args", None)
    if default_oracle_critic_args is None:
        raise ValueError("Missing oracle_critic_args in config.")

    oracles_actors, oracles_critics, oracles_values = build_simba_oracle_modules_from_cfg(
        cfg=cfg,
        obs_shape=obs_shape,
        act_dim=act_dim,
        device=device,
        oracles_dict=oracles_dict,
        default_oracle_actor_args=default_oracle_actor_args,
        default_oracle_critic_args=default_oracle_critic_args,
    )

    args = OmegaConf.to_container(cfg.training, resolve=True)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
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

    with open(os.path.join(cfg.training.checkpoint_dir, "cfg.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
    write_run_manifest(artifacts.run_dir, cfg_resolved, artifacts)

    agent.run(env, replay_buffer=replay_buffer, eval_env=eval_env)


if __name__ == "__main__":
    train_OPI_simba()
