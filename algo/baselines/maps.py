from __future__ import annotations

import random
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from algo.base_algo import BaseAlgo
from algo.algo_utils import ObservationNormalizer, RewardNormalizer, reshape_into_blocks, save_eval_results_to_csv, categorical_td_loss
from data_buffer.replay_buffer import ReplayBuffer
from model.simba import DeterministicSimbaPolicy, SimbaValues
from model.simba_base import EPS, l2normalize_network
from model.mlp import OraclePolicyBase


class MAPSAlgo(BaseAlgo):
    def __init__(
        self,
        *,
        learner_actor: DeterministicSimbaPolicy,
        oracles_actors: Dict[str, OraclePolicyBase],
        oracles_values: Dict[str, SimbaValues],
        args: Dict[str, Any],
        device: torch.device,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)

        self.device = device
        self.learner_actor = learner_actor.to(device)
        self.oracles_actors = {k: v.to(device) for k, v in dict(oracles_actors).items()}
        self.oracles_values = {k: v.to(device) for k, v in dict(oracles_values).items()}
        self.oracles_names = list(self.oracles_actors.keys())
        assert set(self.oracles_names) == set(self.oracles_values.keys()), "oracles_actors/values mismatch"
        self.n_oracles = int(len(self.oracles_names))

        for actor in self.oracles_actors.values():
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)

        self.gamma = float(self.args["discount"])
        self.num_bins = int(self.args["num_bins"])
        self.min_v = float(self.args["min_v"])
        self.max_v = float(self.args["max_v"])
        self.maps_beta = float(self.args["maps_beta"])

        self.learning_starts = int(self.args["learner_learning_starts"])
        self.policy_delay = int(self.args["policy_delay"])
        self.replay_ratio = float(self.args["replay_ratio"])

        self.roll_in_max_k = int(self.args.get("roll_in_max_k", 0))
        self.per_oracle_explore_steps = int(self.args.get("per_oracle_explore_steps", self.learning_starts // (self.n_oracles + 1) if self.n_oracles > 0 else self.learning_starts))
        self.total_explore_steps = int(self.per_oracle_explore_steps * (self.n_oracles + 1))

        self.min_exploration_noise = float(self.args["min_exploration_noise"])
        self.max_exploration_noise = float(self.args["max_exploration_noise"])
        self.target_policy_noise = float(self.args["target_policy_noise"])
        self.task_specific_noise = float(self.args["task_specific_noise"])
        self.clip_noise = float(self.args["clip_noise"])

        _osm = self.args["oracle_std_multiplier"]
        if isinstance(_osm, (list, tuple, np.ndarray)):
            self.oracle_std_multipliers = [float(x) for x in list(_osm)]
            assert len(self.oracle_std_multipliers) == self.n_oracles, (
                f"oracle_std_multiplier must have length n_oracles={self.n_oracles}, "
                f"got {len(self.oracle_std_multipliers)}"
            )
        else:
            self.oracle_std_multipliers = [float(_osm) for _ in range(self.n_oracles)]
        self.oracle_std_multiplier_by_name = {
            name: self.oracle_std_multipliers[i] for i, name in enumerate(self.oracles_names)
        }
        self.normalize_observations = bool(self.args["normalize_observations"])
        self.normalize_rewards = bool(self.args["normalize_rewards"])
        self.normalized_g_max = float(self.args["normalized_g_max"])

        self.obs_normalizer = None
        self.obs_rms = None
        if self.normalize_observations:
            obs_dim = int(self.learner_actor.obs_dim)
            self.obs_normalizer = ObservationNormalizer(obs_dim=obs_dim, dtype=np.float32, eps=float(EPS))
            self.obs_rms = self.obs_normalizer.rms

        self.reward_normalizer = None
        if self.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.gamma), normalized_g_max=float(self.normalized_g_max), dtype=np.float32, eps=float(EPS)
            )
            self.G = self.reward_normalizer.G
            self.G_rms = self.reward_normalizer.G_rms
            self.G_r_max = self.reward_normalizer.G_r_max

        actor_wd = float(self.args["actor_weight_decay"])
        value_wd = float(self.args["value_weight_decay"])
        self.actor_optimizer = self.learner_actor.set_optimizer(self.args["actor_lr"], weight_decay=actor_wd)
        self.oracle_value_optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name in self.oracles_names:
            self.oracle_value_optimizers[name] = self.oracles_values[name].set_optimizer(
                self.args["value_lr"], weight_decay=value_wd
            )

        self.log_every = int(self.args["log_every"])
        self.action_selection_buffer = deque(maxlen=int(self.args["action_selection_buffer_size"]))

        self.use_compile = bool(self.args["use_compile"])
        self._compiled_compute_oracle_value_loss = self._compute_oracle_value_loss
        self._compiled_compute_actor_loss = self._compute_actor_loss
        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_oracle_value_loss = self._try_compile(
                self._compiled_compute_oracle_value_loss, "oracle_value_loss", fullgraph=True
            )
            self._compiled_compute_actor_loss = self._try_compile(
                self._compiled_compute_actor_loss, "learner_actor_loss", fullgraph=True
            )

    def _try_compile(self, fn, name: str, *, fullgraph: bool = True):
        try:
            try:
                import torch._dynamo as dynamo

                limit = int(self.args["dynamo_cache_size_limit"])
                if limit > 0:
                    dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), limit)
            except Exception:
                pass
            return torch.compile(fn, fullgraph=fullgraph)
        except Exception as e:
            print(f"[MAPSAlgo] torch.compile failed for {name}: {e}. Falling back to eager.")
            return fn

    def _normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize_tensor(obs)

    def _scale_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_normalizer is None:
            return rewards
        return self.reward_normalizer.scale_rewards_tensor(rewards)

    def _update_obs_normalizer(self, obs: np.ndarray):
        if self.obs_normalizer is None:
            return
        self.obs_normalizer.update(obs)

    def _update_reward_normalizer(self, reward: float, done: bool):
        if self.reward_normalizer is None:
            return
        self.reward_normalizer.update(float(reward), bool(done))
        self.G = self.reward_normalizer.G
        self.G_r_max = self.reward_normalizer.G_r_max

    def _prepare_sampled_batch(self, batch: dict) -> dict:
        if self.normalize_observations:
            if "observations_raw" not in batch:
                batch["observations_raw"] = batch["observations"]
            if "next_observations_raw" not in batch:
                batch["next_observations_raw"] = batch["next_observations"]
            batch["observations"] = self._normalize_obs_tensor(batch["observations"])
            batch["next_observations"] = self._normalize_obs_tensor(batch["next_observations"])
        if self.normalize_rewards:
            batch["rewards"] = self._scale_reward_tensor(batch["rewards"])
        return batch

    def _obs_to_tensor(self, raw_obs: np.ndarray) -> torch.Tensor:
        obs_np = np.asarray(raw_obs, dtype=np.float32)
        return torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

    def get_action_noise(self, global_step: int, max_steps: int, min_noise: float, max_noise: float) -> float:
        if global_step > max_steps:
            return float(min_noise)
        epsilon = float(global_step) / float(max_steps)
        return float(max_noise + (min_noise - max_noise) * epsilon)

    def add_noise_to_action(self, action: torch.Tensor, noise: float, clip_noise: float) -> torch.Tensor:
        eps = torch.randn_like(action)
        eps.mul_(noise)
        eps.clamp_(-clip_noise, clip_noise)
        action = action.add(eps)
        return action.clamp_(-1.0, 1.0)

    @torch.no_grad()
    def _oracle_value_stats(self, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        uncs = []
        for name in self.oracles_names:
            vs = self.oracles_values[name].get_value(obs_t)
            mean_v = vs.mean(dim=0).squeeze(-1).squeeze(0)
            if vs.shape[0] == 2:
                unc = (vs[0] - vs[1]).abs().squeeze(-1).squeeze(0)
            else:
                unc = vs.std(dim=0).squeeze(-1).squeeze(0)
            means.append(mean_v)
            uncs.append(unc)
        return torch.stack(means, dim=0), torch.stack(uncs, dim=0)

    @torch.no_grad()
    def _get_oracle_action(self, obs_t_raw: torch.Tensor, oracle_index: int) -> Tuple[np.ndarray, np.ndarray]:
        name = self.oracles_names[int(oracle_index)]
        actor = self.oracles_actors[name]
        std = self.oracle_std_multiplier_by_name[name]
        normalized_action = actor.get_action(obs_t_raw, std)
        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        selected_action = self.add_noise_to_action(normalized_action, exploration_noise, clip_noise=self.clip_noise)
        final_selected_action = selected_action * actor.action_scale + actor.action_bias
        return (
            final_selected_action.squeeze(0).detach().cpu().numpy(),
            selected_action.squeeze(0).detach().cpu().numpy(),
        )

    @torch.no_grad()
    def _get_learner_action(self, obs_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        normalized_action = self.learner_actor.get_action(obs_t, self.target_policy_noise)
        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        selected_action = self.add_noise_to_action(normalized_action, exploration_noise, clip_noise=self.clip_noise)
        final_selected_action = selected_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return (
            final_selected_action.squeeze(0).detach().cpu().numpy(),
            selected_action.squeeze(0).detach().cpu().numpy(),
        )

    @torch.no_grad()
    def get_training_action_with_index(self, raw_obs: np.ndarray, actor_index: int):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw

        if actor_index == 0:
            final_action, normalized_action = self._get_learner_action(obs_t)
            selected_idx = 0
            infos = {
                "maps/selected_oracle": -1.0,
                "maps/selected_uncertainty": 0.0,
                "maps/selected_value": 0.0,
                "maps/uncertainty_avg": 0.0,
                "maps/use_oracle": 0.0,
            }
        else:
            oracle_idx = actor_index - 1
            final_action, normalized_action = self._get_oracle_action(obs_t_raw, oracle_idx)
            selected_idx = actor_index

            mean_v, unc = self._oracle_value_stats(obs_t)
            infos = {
                "maps/selected_oracle": float(oracle_idx),
                "maps/selected_uncertainty": float(unc[oracle_idx]),
                "maps/selected_value": float(mean_v[oracle_idx]),
                "maps/uncertainty_avg": float(unc.mean().item()),
                "maps/use_oracle": 1.0,
            }

        return final_action, normalized_action, selected_idx, infos

    @torch.no_grad()
    def get_training_action(self, raw_obs: np.ndarray):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
        mean_v, unc = self._oracle_value_stats(obs_t)
        scores = mean_v + self.maps_beta * unc
        k_star = int(torch.argmax(scores).item())
        final_action, normalized_action = self._get_oracle_action(obs_t_raw, k_star)
        selected_idx = k_star + 1

        infos = {
            "maps/selected_oracle": float(k_star),
            "maps/selected_uncertainty": float(unc[k_star]),
            "maps/selected_value": float(mean_v[k_star]),
            "maps/uncertainty_avg": float(unc.mean().item()),
            "maps/use_oracle": 1.0,
        }
        return final_action, normalized_action, selected_idx, infos

    @torch.no_grad()
    def get_inference_action(self, raw_obs: np.ndarray, eval_oracle=False, oracle_name=None, **kwargs):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        if eval_oracle and oracle_name is not None:
            oracle_actor = self.oracles_actors[oracle_name]
            oracle_idx = self.oracles_names.index(oracle_name)
            std = self.oracle_std_multiplier_by_name[oracle_name]
            action = oracle_actor.get_action(obs_t_raw, std)
            final_action = action * oracle_actor.action_scale + oracle_actor.action_bias
        else:
            obs_t = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
            final_action = self.learner_actor.get_eval_action(obs_t)
            final_action = final_action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action.squeeze(0).detach().cpu().numpy()

    def _compute_oracle_value_loss(self, data: dict, oracle_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = data["observations"]
        next_obs = data["next_observations"]
        rewards = data["rewards"]

        with torch.no_grad():
            next_vs, next_v_infos = self.oracles_values[oracle_name].get_target_value_with_info(next_obs)
            min_idx = next_vs.squeeze(-1).argmin(dim=0)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_v_infos], dim=0)
            next_v_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_vs, pred_v_infos = self.oracles_values[oracle_name].get_value_with_info(obs)
        gamma_n = float(self.gamma) ** int(self.args["nstep"])

        v_loss = 0.0
        for pred_info in pred_v_infos:
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_v_log_probs,
                reward=rewards,
                done=data["dones"],
                gamma=gamma_n,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            v_loss = v_loss + loss.mean()

        v_mean = pred_vs.mean()
        v_uncertainty = pred_vs.std(dim=0).mean()
        return v_loss, v_mean, v_uncertainty

    def _compute_actor_loss(self, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data["observations"]
        next_obs = data["next_observations"]
        rewards = data["rewards"]
        discounts = data["discounts"]

        with torch.no_grad():
            current_vals = []
            next_vals = []
            for name in self.oracles_names:
                v_now = self.oracles_values[name].get_value(obs).mean(dim=0).squeeze(-1)
                v_next = self.oracles_values[name].get_target_value(next_obs).mean(dim=0).squeeze(-1)
                current_vals.append(v_now)
                next_vals.append(v_next)
            v_max = torch.stack(current_vals, dim=0).max(dim=0).values
            v_next_max = torch.stack(next_vals, dim=0).max(dim=0).values
            adv = rewards + discounts * v_next_max - v_max
            adv_pos = F.relu(adv)

        pred_action = self.learner_actor.get_eval_action(obs)
        mse = (pred_action - data["actions"]).pow(2).sum(dim=-1)
        actor_loss = (adv_pos * mse).mean()
        return actor_loss, adv.mean()

    def update(self) -> Dict[str, float]:
        num_blocks = int(self.args["learner_critic_utd_ratio"])
        batch_size = int(self.args["learner_critic_batch_size"])

        actor_data = self._prepare_sampled_batch(
            self.replay_buffer.sample(batch_size)
        )
        actor_mini_batches = reshape_into_blocks(actor_data, num_blocks)

        oracle_mini_batches = {}
        for oracle_idx, name in enumerate(self.oracles_names):
            odata = self._prepare_sampled_batch(
                self.replay_buffer.action_mask_sample(
                    batch_size, oracle_idx + 1, on_policy_ratio=self.replay_ratio
                )
            )
            oracle_mini_batches[name] = reshape_into_blocks(odata, num_blocks)

        info: Dict[str, float] = {}
        for i in range(num_blocks):
            self.update_step += 1

            value_losses = []
            for name in self.oracles_names:
                oracle_sub_batch = {k: v[i] for k, v in oracle_mini_batches[name].items()}
                loss, v_mean, v_uncertainty = self._compiled_compute_oracle_value_loss(oracle_sub_batch, name)
                value_losses.append(loss)
                info[f"oracles/{name}/v_loss"] = float(loss.item())
                info[f"oracles/{name}/v_mean"] = float(v_mean.item())
                info[f"oracles/{name}/v_uncertainty"] = float(v_uncertainty.item())

            if value_losses:
                total_value_loss = torch.stack(value_losses).sum()
                for opt in self.oracle_value_optimizers.values():
                    opt.zero_grad(set_to_none=True)
                total_value_loss.backward()
                for name, opt in self.oracle_value_optimizers.items():
                    nn.utils.clip_grad_norm_(self.oracles_values[name].parameters(), self.args["max_grad_norm"])
                    opt.step()
                    l2normalize_network(self.oracles_values[name])

            for name in self.oracles_names:
                self.oracles_values[name].update_target(self.args["tau"])

            if self.update_step % self.policy_delay == 0:
                actor_sub_batch = {k: v[i] for k, v in actor_mini_batches.items()}
                actor_loss, adv_mean = self._compiled_compute_actor_loss(actor_sub_batch)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.args["policy_max_grad_norm"])
                self.actor_optimizer.step()
                l2normalize_network(self.learner_actor)
                info["maps/adv_mean"] = float(adv_mean.item())
                info["learner/losses/actor_loss"] = float(actor_loss.item())

        return info

    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        self._pre_run(env, replay_buffer, eval_env=eval_env)
        _eval_env = eval_env if eval_env is not None else env

        if len(self.oracles_names) > 0:
            for oracle_name in self.oracles_names:
                self.oracles_actors[oracle_name].eval()
                eval_results = self.evaluate(
                    _eval_env,
                    num_episodes=self.eval_episodes,
                    save_name=f"eval_{oracle_name}",
                    eval_tag=f"oracle_{oracle_name}",
                    log_local_eval=False,
                    save_best_model=False,
                    eval_oracle=True,
                    oracle_name=oracle_name,
                )
                if self.use_wandb:
                    for k, v in eval_results.items():
                        wandb.log({f"eval/{oracle_name}/{k}": v}, step=self.global_step)
                self.oracles_actors[oracle_name].train()

        env_obs, _ = self.env.reset(seed=self.args["seed"])

        roll_in_steps_remaining = 0

        while self.global_step < self.total_timesteps:
            self.global_step += 1
            selection_info: Optional[Dict[str, float]] = None

            if self.normalize_observations:
                self._update_obs_normalizer(np.asarray(env_obs, dtype=np.float32).reshape(1, -1))

            with torch.no_grad():
                if self.global_step < self.learning_starts:
                    actor_cycle_step = (self.global_step - 1) // self.per_oracle_explore_steps
                    actor_index = int(actor_cycle_step % (self.n_oracles + 1))
                    final_action, normalized_action, selected_action_index, selection_info = self.get_training_action_with_index(env_obs, actor_index)
                elif self.global_step >= self.learning_starts:
                    if roll_in_steps_remaining > 0:
                        roll_in_steps_remaining -= 1
                        final_action, normalized_action = self._get_learner_action(
                            self._normalize_obs_tensor(self._obs_to_tensor(env_obs))
                        )
                        selected_action_index = 0
                        selection_info = {
                            "maps/selected_oracle": -1.0,
                            "maps/selected_uncertainty": 0.0,
                            "maps/selected_value": 0.0,
                            "maps/uncertainty_avg": 0.0,
                            "maps/use_oracle": 0.0,
                        }
                    elif self.n_oracles > 0:
                        final_action, normalized_action, selected_action_index, selection_info = self.get_training_action(env_obs)
                    else:
                        final_action, normalized_action = self._get_learner_action(
                            self._normalize_obs_tensor(self._obs_to_tensor(env_obs))
                        )
                        selected_action_index = 0
                        selection_info = {
                            "maps/selected_oracle": -1.0,
                            "maps/uncertainty_avg": 0.0,
                            "maps/use_oracle": 0.0,
                        }
                else:
                    if self.n_oracles > 0:
                        final_action, normalized_action, selected_action_index, selection_info = self.get_training_action(env_obs)
                    else:
                        final_action, normalized_action = self._get_learner_action(
                            self._normalize_obs_tensor(self._obs_to_tensor(env_obs))
                        )
                        selected_action_index = 0
                        selection_info = {
                            "maps/selected_oracle": -1.0,
                            "maps/uncertainty_avg": 0.0,
                            "maps/use_oracle": 0.0,
                        }

            selected_action_index = int(selected_action_index)
            self.action_selection_buffer.append(selected_action_index)
            self.log_local_action_selection(
                global_step=self.global_step,
                selected_action_index=selected_action_index,
            )

            next_env_obs, reward, terminated, truncated, infos = self.env.step(final_action)
            if self.normalize_rewards:
                self._update_reward_normalizer(float(reward), bool(terminated or truncated))

            if isinstance(infos, dict) and "episode" in infos:
                self.log_local_train_episode(
                    global_step=self.global_step,
                    episode_reward=infos["episode"]["r"],
                    episode_length=infos["episode"]["l"],
                )

            if (
                self.use_wandb
                and isinstance(infos, dict)
                and "episode" in infos
                and self.global_step > self.learning_starts
            ):
                wandb.log(
                    {"env/episode_reward": infos["episode"]["r"], "env/episode_length": infos["episode"]["l"]},
                    step=self.global_step,
                )

            self.replay_buffer.add_transition(
                {
                    "observations": env_obs,
                    "next_observations": next_env_obs,
                    "actions": normalized_action,
                    "rewards": float(reward),
                    "dones": bool(terminated),
                    "truncated": bool(truncated),
                    "action_mask": int(selected_action_index),
                    "global_step": int(self.global_step),
                }
            )
            env_obs = next_env_obs

            train_info = None
            if self.global_step >= self.learning_starts:
                train_info = self.update()

            do_log = self.use_wandb and (self.global_step % self.log_every == 0)
            if do_log:
                info_payload = train_info or {}
                info_payload["time/total_timesteps"] = self.global_step
                info_payload["time/fps"] = self.global_step / (time.time() - self.start_time)
                if selection_info is not None:
                    info_payload.update(selection_info)
                wandb.log(info_payload, step=self.global_step)

            self.progress_bar.update(1)

            if (self.global_step + 1) % self.eval_every == 0:
                self.learner_actor.eval()
                eval_results = self.evaluate(_eval_env, num_episodes=self.eval_episodes)
                self.learner_actor.train()
                save_eval_results_to_csv(
                    self.args["checkpoint_dir"],
                    self.args.get("eval_run_name", self.args["run_name"]),
                    eval_results,
                    eval_dir=self.args.get("eval_dir"),
                )
                if self.use_wandb:
                    for k, v in eval_results.items():
                        wandb.log({f"eval/{k}": v}, step=self.global_step)

            if (self.global_step + 1) % self.save_freq == 0:
                save_path = f"{self.args['checkpoint_dir']}/{self.args['run_name']}_{self.global_step}.pt"
                self.save_model(save_path)

            if truncated or terminated:
                env_obs, _ = self.env.reset()
                if self.global_step >= self.learning_starts and self.roll_in_max_k > 0:
                    roll_in_steps_remaining = random.randint(0, self.roll_in_max_k)
                else:
                    roll_in_steps_remaining = 0

        return self._after_run()

    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "learner_actor_state_dict": self.learner_actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "oracles_names": list(self.oracles_names),
            }
        )

        for i, name in enumerate(self.oracles_names):
            checkpoint[f"oracles_actor_state_dict_{i}"] = self.oracles_actors[name].state_dict()
            checkpoint[f"oracles_value_state_dict_{i}"] = self.oracles_values[name].state_dict()
            checkpoint[f"oracles_value_optimizer_state_dict_{i}"] = self.oracle_value_optimizers[name].state_dict()

        checkpoint["rng_state_torch"] = torch.get_rng_state()
        checkpoint["rng_state_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        checkpoint["rng_state_numpy"] = np.random.get_state()
        checkpoint["rng_state_python"] = random.getstate()
        buf = np.array(self.action_selection_buffer, dtype=int) if len(self.action_selection_buffer) > 0 else np.array([], dtype=int)
        checkpoint["action_selection_buffer"] = buf
        checkpoint["action_selection_buffer_maxlen"] = int(self.action_selection_buffer.maxlen or 0)

        if self.obs_normalizer is not None:
            checkpoint["obs_rms"] = self.obs_normalizer.rms.state_dict()
        if self.reward_normalizer is not None:
            checkpoint["reward_norm"] = {
                "G": self.reward_normalizer.G,
                "G_rms": self.reward_normalizer.G_rms.state_dict(),
                "G_r_max": self.reward_normalizer.G_r_max,
            }

        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        self.learner_actor.load_state_dict(checkpoint["learner_actor_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

        saved_names = checkpoint["oracles_names"]
        if list(saved_names) != list(self.oracles_names):
            print("[MAPSAlgo] Warning: oracles_names mismatch between checkpoint and current model. Loading in current order.")

        for i, name in enumerate(self.oracles_names):
            k_actor = f"oracles_actor_state_dict_{i}"
            k_value = f"oracles_value_state_dict_{i}"
            k_vopt = f"oracles_value_optimizer_state_dict_{i}"
            if k_actor in checkpoint:
                self.oracles_actors[name].load_state_dict(checkpoint[k_actor])
            if k_value in checkpoint:
                self.oracles_values[name].load_state_dict(checkpoint[k_value])
            if k_vopt in checkpoint:
                self.oracle_value_optimizers[name].load_state_dict(checkpoint[k_vopt])

        try:
            if "rng_state_torch" in checkpoint:
                torch.set_rng_state(checkpoint["rng_state_torch"])
            if "rng_state_cuda" in checkpoint and checkpoint["rng_state_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
            if "rng_state_numpy" in checkpoint:
                np.random.set_state(checkpoint["rng_state_numpy"])
            if "rng_state_python" in checkpoint:
                random.setstate(checkpoint["rng_state_python"])
        except Exception as e:
            print(f"[MAPSAlgo] Warning: failed to restore RNG state: {e}")

        if "action_selection_buffer" in checkpoint:
            buf = checkpoint["action_selection_buffer"]
            maxlen = int(checkpoint["action_selection_buffer_maxlen"])
            try:
                seq = buf.tolist() if hasattr(buf, "tolist") else list(buf)
            except Exception:
                seq = []
            self.action_selection_buffer = deque(seq, maxlen=maxlen)

        if "obs_rms" in checkpoint and self.obs_normalizer is not None:
            self.obs_normalizer.rms.load_state_dict(checkpoint["obs_rms"])
        if "reward_norm" in checkpoint and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(checkpoint["reward_norm"])
            self.G = self.reward_normalizer.G
            self.G_r_max = self.reward_normalizer.G_r_max

        return checkpoint
