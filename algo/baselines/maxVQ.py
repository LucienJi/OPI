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
from algo.algo_utils import ObservationNormalizer, RewardNormalizer, save_eval_results_to_csv, categorical_td_loss
from data_buffer.replay_buffer import ReplayBuffer
from model.simba import DeterministicSimbaPolicy, SimbaCritics, SimbaValues
from model.simba_base import EPS, l2normalize_network
from model.mlp import OraclePolicyBase


class OracleSelectionAlgo(BaseAlgo):
    def __init__(
        self,
        *,
        selection_mode: str,
        learner_actor: DeterministicSimbaPolicy,
        oracles_actors: Dict[str, OraclePolicyBase],
        oracles_values: Optional[Dict[str, SimbaValues]] = None,
        oracles_critics: Optional[Dict[str, SimbaCritics]] = None,
        args: Dict[str, Any],
        device: torch.device,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)
        self.device = device

        mode = str(selection_mode).lower().strip()
        if mode not in {"max_v", "max_q"}:
            raise ValueError(f"selection_mode must be 'max_v' or 'max_q', got {selection_mode}")
        self.selection_mode = mode

        self.learner_actor = learner_actor.to(device)
        self.oracles_actors = {k: v.to(device) for k, v in dict(oracles_actors).items()}
        self.oracles_names = list(self.oracles_actors.keys())
        self._oracle_actor_list = [self.oracles_actors[name] for name in self.oracles_names]
        self.n_oracles = int(len(self.oracles_names))
        if self.n_oracles <= 0:
            raise ValueError("OracleSelectionAlgo requires at least one oracle actor.")

        if self.selection_mode == "max_v":
            self.oracles_values = {k: v.to(device) for k, v in dict(oracles_values).items()}
            self.oracle_estimators = [self.oracles_values[name] for name in self.oracles_names]
        else:
            self.oracles_critics = {k: v.to(device) for k, v in dict(oracles_critics).items()}
            self.oracle_estimators = [self.oracles_critics[name] for name in self.oracles_names]

        for actor in self._oracle_actor_list:
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)

        self.gamma = float(self.args["discount"])
        self.num_bins = int(self.args["num_bins"])
        self.min_v = float(self.args["min_v"])
        self.max_v = float(self.args["max_v"])
        self.replay_ratio = float(self.args.get("replay_ratio", 1.0))
        if not (0.0 <= self.replay_ratio <= 1.0):
            raise ValueError(f"replay_ratio must be in [0,1], got {self.replay_ratio}")
        self.policy_delay = int(self.args["policy_delay"])
        self.learner_learning_starts = int(self.args["learner_learning_starts"])
        self.max_grad_norm = float(self.args["max_grad_norm"])
        self.policy_max_grad_norm = float(self.args["policy_max_grad_norm"])

        self.min_exploration_noise = float(self.args["min_exploration_noise"])
        self.max_exploration_noise = float(self.args["max_exploration_noise"])
        self.clip_noise = float(self.args["clip_noise"])
        self.target_policy_noise = float(self.args["target_policy_noise"])
        self.task_specific_noise = float(self.args["task_specific_noise"])
        self.noise_clip = float(self.args["noise_clip"])

        _osm = self.args["oracle_std_multiplier"]
        if isinstance(_osm, (list, tuple, np.ndarray)):
            self.oracle_std_multipliers = [float(x) for x in list(_osm)]
            if len(self.oracle_std_multipliers) != self.n_oracles:
                raise ValueError(
                    f"oracle_std_multiplier must have length n_oracles={self.n_oracles}, "
                    f"got {len(self.oracle_std_multipliers)}"
                )
        else:
            self.oracle_std_multipliers = [float(_osm) for _ in range(self.n_oracles)]

        self.oracle_burnin_steps = int(self.args["oracle_burnin_steps"])

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
        self.actor_optimizer = self.learner_actor.set_optimizer(self.args["actor_lr"], weight_decay=actor_wd)
        self.oracle_estimator_optimizers: Dict[str, torch.optim.Optimizer] = {}
        if self.selection_mode == "max_v":
            value_wd = float(self.args["value_weight_decay"])
            for name in self.oracles_names:
                self.oracle_estimator_optimizers[name] = self.oracles_values[name].set_optimizer(
                    self.args["critic_lr"], weight_decay=value_wd
                )
        else:
            critic_wd = float(self.args["critic_weight_decay"])
            for name in self.oracles_names:
                self.oracle_estimator_optimizers[name] = self.oracles_critics[name].set_optimizer(
                    self.args["critic_lr"], weight_decay=critic_wd
                )

        self.log_every = int(self.args["log_every"])
        self.action_selection_buffer = deque(maxlen=int(self.args["action_selection_buffer_size"]))

        self.use_compile = bool(self.args["use_compile"])
        if self.selection_mode == "max_v":
            self._compiled_compute_estimator_loss = self._compute_oracle_value_loss
        else:
            self._compiled_compute_estimator_loss = self._compute_oracle_critic_loss
        self._compiled_compute_actor_loss = self._compute_actor_loss
        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_estimator_loss = self._try_compile(
                self._compiled_compute_estimator_loss, "oracle_estimator_loss", fullgraph=True
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
            print(f"[OracleSelectionAlgo] torch.compile failed for {name}: {e}. Falling back to eager.")
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
    def _sample_oracle_action(self, obs_t_raw: torch.Tensor, oracle_index: int) -> torch.Tensor:
        actor = self._oracle_actor_list[int(oracle_index)]
        std = float(self.oracle_std_multipliers[int(oracle_index)])
        action = actor.get_action(obs_t_raw, std)
        return action

    @torch.no_grad()
    def get_inference_action(self, raw_obs, eval_oracle=False, oracle_name=None, **inference_kwargs):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        if eval_oracle and oracle_name is not None:
            oracle_actor = self.oracles_actors[oracle_name]
            oracle_idx = self.oracles_names.index(oracle_name)
            std = float(self.oracle_std_multipliers[oracle_idx])
            action = oracle_actor.get_action(obs_t_raw, std)
            final_action = action * oracle_actor.action_scale + oracle_actor.action_bias
        else:
            obs_t = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
            action = self.learner_actor.get_eval_action(obs_t)
            final_action = action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def _oracle_values(self, obs_t: torch.Tensor) -> torch.Tensor:
        values = []
        for value_net in self.oracle_estimators:
            vs = value_net.get_value(obs_t)
            v_mean = vs.mean(dim=0).squeeze(-1)
            values.append(v_mean)
        return torch.stack(values, dim=0)

    @torch.no_grad()
    def _oracle_actions_and_q(self, obs_t_raw: torch.Tensor, obs_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = []
        q_vals = []
        for i, critic in enumerate(self.oracle_estimators):
            action = self._sample_oracle_action(obs_t_raw, i)
            qs = critic.get_value(obs_t, action)
            q_mean = qs.mean(dim=0).squeeze(-1)
            actions.append(action)
            q_vals.append(q_mean)
        actions_t = torch.stack(actions, dim=0)
        q_vals_t = torch.stack(q_vals, dim=0)
        return actions_t, q_vals_t

    @torch.no_grad()
    def _select_best_oracle_batch(
        self, obs_t_raw: torch.Tensor, obs_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.selection_mode == "max_v":
            values = self._oracle_values(obs_t)
            best_idx = values.argmax(dim=0)
            actions = []
            for i in range(self.n_oracles):
                actions.append(self._sample_oracle_action(obs_t_raw, i))
            actions_t = torch.stack(actions, dim=0)
        else:
            actions_t, q_vals = self._oracle_actions_and_q(obs_t_raw, obs_t)
            best_idx = q_vals.argmax(dim=0)
        idx = best_idx.view(1, -1, 1).expand(1, best_idx.shape[0], actions_t.shape[-1])
        best_actions = actions_t.gather(0, idx).squeeze(0)
        return best_idx, best_actions

    @torch.no_grad()
    def get_training_action(self, raw_obs: np.ndarray):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw

        selected_action = None
        if self.global_step <= self.oracle_burnin_steps:
            selected_oracle_idx = int(random.randint(0, self.n_oracles - 1))
            selected_action = self._sample_oracle_action(obs_t_raw, selected_oracle_idx)
        else:
            if self.selection_mode == "max_v":
                values = self._oracle_values(obs_t)
                selected_oracle_idx = int(values.argmax(dim=0).item())
                selected_action = self._sample_oracle_action(obs_t_raw, selected_oracle_idx)
            else:
                actions_t, q_vals = self._oracle_actions_and_q(obs_t_raw, obs_t)
                selected_oracle_idx = int(q_vals.argmax(dim=0).item())
                selected_action = actions_t[selected_oracle_idx]
        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        selected_action = self.add_noise_to_action(selected_action, exploration_noise, clip_noise=self.clip_noise)
        actor = self._oracle_actor_list[selected_oracle_idx]
        final_action = selected_action * actor.action_scale + actor.action_bias
        selection_info = {
            "baseline/mode": self.selection_mode,
            "baseline/selected_oracle_idx": float(selected_oracle_idx),
        }
        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            selected_action.squeeze(0).detach().cpu().numpy(),
            selected_oracle_idx,
            selection_info,
        )

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

    def _compute_oracle_critic_loss(self, data: dict, oracle_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = data["observations"]
        obs_raw = data["observations_raw"]
        next_obs = data["next_observations"]
        next_obs_raw = data["next_observations_raw"]
        actions = data["actions"]
        rewards = data["rewards"]

        critic = self.oracles_critics[oracle_name]

        with torch.no_grad():
            idx = self.oracles_names.index(oracle_name)
            next_action = self._sample_oracle_action(next_obs_raw, idx)
            next_qs, next_q_infos = critic.get_target_value_with_info(next_obs, next_action)
            min_idx = next_qs.squeeze(-1).argmin(dim=0)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = critic.get_value_with_info(obs, actions)
        gamma_n = float(self.gamma) ** int(self.args["nstep"])

        q_loss = 0.0
        for pred_info in pred_q_infos:
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=rewards,
                done=data["dones"],
                gamma=gamma_n,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )
            q_loss = q_loss + loss.mean()

        q_mean = pred_qs.mean()
        q_uncertainty = pred_qs.std(dim=0).mean()
        return q_loss, q_mean, q_uncertainty

    def _compute_actor_loss(self, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = data["observations"]
        obs_raw = data["observations_raw"]

        with torch.no_grad():
            best_idx, target_actions = self._select_best_oracle_batch(obs_raw, obs)

        pred_action = self.learner_actor.get_eval_action(obs)
        actor_loss = F.mse_loss(pred_action, target_actions)
        return actor_loss, best_idx.float().mean()

    def update(self) -> Dict[str, float]:
        batch_size = int(self.args["learner_critic_batch_size"])
        actor_data = self._prepare_sampled_batch(self.replay_buffer.sample(batch_size))

        estimator_losses = []
        info: Dict[str, float] = {}
        for oracle_idx, name in enumerate(self.oracles_names):
            oracle_data = self._prepare_sampled_batch(
                self.replay_buffer.action_mask_sample(
                    batch_size,
                    oracle_idx + 1,
                    on_policy_ratio=self.replay_ratio,
                )
            )
            loss, mean_val, uncertainty = self._compiled_compute_estimator_loss(oracle_data, name)
            estimator_losses.append(loss)
            if self.selection_mode == "max_v":
                info[f"oracles/{name}/v_loss"] = float(loss.item())
                info[f"oracles/{name}/v_mean"] = float(mean_val.item())
                info[f"oracles/{name}/v_uncertainty"] = float(uncertainty.item())
            else:
                info[f"oracles/{name}/q_loss"] = float(loss.item())
                info[f"oracles/{name}/q_mean"] = float(mean_val.item())
                info[f"oracles/{name}/q_uncertainty"] = float(uncertainty.item())

        if estimator_losses:
            total_loss = torch.stack(estimator_losses).sum()
            for opt in self.oracle_estimator_optimizers.values():
                opt.zero_grad(set_to_none=True)
            total_loss.backward()
            for name, opt in self.oracle_estimator_optimizers.items():
                if self.selection_mode == "max_v":
                    nn.utils.clip_grad_norm_(self.oracles_values[name].parameters(), self.max_grad_norm)
                    opt.step()
                    l2normalize_network(self.oracles_values[name])
                else:
                    nn.utils.clip_grad_norm_(self.oracles_critics[name].parameters(), self.max_grad_norm)
                    opt.step()
                    l2normalize_network(self.oracles_critics[name])

        for name in self.oracles_names:
            if self.selection_mode == "max_v":
                self.oracles_values[name].update_target(self.args["tau"])
            else:
                self.oracles_critics[name].update_target(self.args["tau"])

        if self.update_step % self.policy_delay == 0:
            actor_loss, avg_best_idx = self._compiled_compute_actor_loss(actor_data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.policy_max_grad_norm)
            self.actor_optimizer.step()
            l2normalize_network(self.learner_actor)
            info["learner/losses/actor_loss"] = float(actor_loss.item())
            info["baseline/avg_selected_oracle_idx"] = float(avg_best_idx.item())

        self.update_step += 1
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

        while self.global_step < self.total_timesteps:
            self.global_step += 1

            if self.normalize_observations:
                self._update_obs_normalizer(np.asarray(env_obs, dtype=np.float32).reshape(1, -1))

            with torch.inference_mode():
                final_action, normalized_action, selected_oracle_idx, selection_info = self.get_training_action(env_obs)

            selected_action_index = int(selected_oracle_idx) + 1
            self.action_selection_buffer.append(int(selected_action_index))
            self.log_local_action_selection(
                global_step=self.global_step,
                selected_action_index=int(selected_action_index),
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
                and self.global_step > self.learner_learning_starts
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
            if self.global_step >= self.learner_learning_starts:
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

        return self._after_run()

    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "learner_actor_state_dict": self.learner_actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "oracles_names": list(self.oracles_names),
                "selection_mode": self.selection_mode,
            }
        )

        for i, name in enumerate(self.oracles_names):
            checkpoint[f"oracles_actor_state_dict_{i}"] = self.oracles_actors[name].state_dict()
            if self.selection_mode == "max_v":
                checkpoint[f"oracles_value_state_dict_{i}"] = self.oracles_values[name].state_dict()
                checkpoint[f"oracles_value_optimizer_state_dict_{i}"] = self.oracle_estimator_optimizers[name].state_dict()
            else:
                checkpoint[f"oracles_critic_state_dict_{i}"] = self.oracles_critics[name].state_dict()
                checkpoint[f"oracles_critic_optimizer_state_dict_{i}"] = self.oracle_estimator_optimizers[name].state_dict()

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
            print("[OracleSelectionAlgo] Warning: oracles_names mismatch between checkpoint and current model. Loading in current order.")

        for i, name in enumerate(self.oracles_names):
            k_actor = f"oracles_actor_state_dict_{i}"
            if k_actor in checkpoint:
                self.oracles_actors[name].load_state_dict(checkpoint[k_actor])
            if self.selection_mode == "max_v":
                k_value = f"oracles_value_state_dict_{i}"
                k_opt = f"oracles_value_optimizer_state_dict_{i}"
                if k_value in checkpoint:
                    self.oracles_values[name].load_state_dict(checkpoint[k_value])
                if k_opt in checkpoint:
                    self.oracle_estimator_optimizers[name].load_state_dict(checkpoint[k_opt])
            else:
                k_critic = f"oracles_critic_state_dict_{i}"
                k_opt = f"oracles_critic_optimizer_state_dict_{i}"
                if k_critic in checkpoint:
                    self.oracles_critics[name].load_state_dict(checkpoint[k_critic])
                if k_opt in checkpoint:
                    self.oracle_estimator_optimizers[name].load_state_dict(checkpoint[k_opt])

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
            print(f"[OracleSelectionAlgo] Warning: failed to restore RNG state: {e}")

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
