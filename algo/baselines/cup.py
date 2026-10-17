import random
import time
from collections import deque
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from algo.base_algo import BaseAlgo
from algo.algo_utils import ObservationNormalizer, RewardNormalizer, categorical_td_loss, reshape_into_blocks, save_eval_results_to_csv
from data_buffer.replay_buffer import ReplayBuffer
from model.simba import DeterministicSimbaPolicy, SimbaCritics
from model.simba_base import EPS, l2normalize_network
from model.mlp import OraclePolicyBase


class CUPAlgo(BaseAlgo):
    def __init__(
        self,
        *,
        learner_actor: DeterministicSimbaPolicy,
        learner_critic: SimbaCritics,
        oracles_actors: Dict[str, OraclePolicyBase],
        args: Dict[str, Any],
        device: torch.device,
    ):
        args = dict(args)
        args["device"] = device
        super().__init__(args=args)
        self.device = device

        self.learner_actor = learner_actor.to(device)
        self.learner_critic = learner_critic.to(device)
        self.oracles_actors = {k: v.to(device) for k, v in dict(oracles_actors).items()}
        self.oracles_names = list(self.oracles_actors.keys())
        self._oracle_actor_list = [self.oracles_actors[name] for name in self.oracles_names]

        for actor in self._oracle_actor_list:
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)

        _osm = self.args["oracle_std_multiplier"]
        n_oracles = len(self.oracles_names)
        if isinstance(_osm, (list, tuple, np.ndarray)):
            self.oracle_std_multipliers = [float(x) for x in list(_osm)]
            assert len(self.oracle_std_multipliers) == n_oracles, (
                f"oracle_std_multiplier must have length n_oracles={n_oracles}, "
                f"got {len(self.oracle_std_multipliers)}"
            )
        else:
            self.oracle_std_multipliers = [float(_osm) for _ in range(n_oracles)]
        self.oracle_std_multiplier_by_name = {
            name: self.oracle_std_multipliers[i] for i, name in enumerate(self.oracles_names)
        }
        self.oracle_std_multiplier = float(self.oracle_std_multipliers[0]) if n_oracles > 0 else float(_osm)

        self.gamma = float(self.args["discount"])
        self.num_bins = int(self.args["num_bins"])
        self.min_v = float(self.args["min_v"])
        self.max_v = float(self.args["max_v"])
        self.task_specific_noise = float(self.args["task_specific_noise"])
        self.target_policy_noise = float(self.args["target_policy_noise"])
        self.policy_delay = int(self.args["policy_delay"])
        self.learner_learning_starts = int(self.args["learner_learning_starts"])
        self.max_grad_norm = float(self.args["max_grad_norm"])
        self.policy_max_grad_norm = float(self.args["policy_max_grad_norm"])

        self.cup_beta_1 = float(self.args["cup_beta_1"])
        self.cup_beta_2 = float(self.args["cup_beta_2"])

        self.min_exploration_noise = float(self.args["min_exploration_noise"])
        self.max_exploration_noise = float(self.args["max_exploration_noise"])
        self.clip_noise = float(self.args["clip_noise"])

        self.normalize_observations = bool(self.args["normalize_observations"])
        self.normalize_rewards = bool(self.args["normalize_rewards"])
        self.normalized_g_max = float(self.args["normalized_g_max"])

        self.log_every = int(self.args["log_every"])
        self.action_selection_buffer = deque(maxlen=int(self.args["action_selection_buffer_size"]))

        actor_wd = float(self.args["actor_weight_decay"])
        critic_wd = float(self.args["critic_weight_decay"])
        self.actor_optimizer = self.learner_actor.set_optimizer(self.args["actor_lr"], weight_decay=actor_wd)
        self.critic_optimizer = self.learner_critic.set_optimizer(self.args["critic_lr"], weight_decay=critic_wd)

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

        self.use_compile = bool(self.args["use_compile"])
        self._compiled_compute_learner_critic_loss = self._compute_learner_critic_loss
        self._compiled_compute_learner_actor_loss = self._compute_learner_actor_loss
        if self.use_compile and hasattr(torch, "compile"):
            self._compiled_compute_learner_critic_loss = self._try_compile(
                self._compiled_compute_learner_critic_loss, "learner_critic_loss", fullgraph=True
            )
            self._compiled_compute_learner_actor_loss = self._try_compile(
                self._compiled_compute_learner_actor_loss, "learner_actor_loss", fullgraph=True
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
            print(f"[CUPAlgo] torch.compile failed for {name}: {e}. Falling back to eager.")
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
    def _get_learner_training_critic(self, obs_t: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qs = self.learner_critic.get_value(obs_t, actions)
        q_mean = qs.mean(dim=0).squeeze(-1)
        q_std = qs.std(dim=0).squeeze(-1)
        return q_mean, q_std

    @torch.no_grad()
    def get_training_action(self, raw_obs, **training_kwargs):
        obs_t_raw = self._obs_to_tensor(raw_obs)
        obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw

        action = self.learner_actor.get_action(obs_t_learner,self.task_specific_noise)
        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        action = self.add_noise_to_action(action, exploration_noise, clip_noise=self.clip_noise)
        final_action = action * self.learner_actor.action_scale + self.learner_actor.action_bias

        return (
            final_action.squeeze(0).detach().cpu().numpy(),
            action.squeeze(0).detach().cpu().numpy(),
            0,
        )

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
            obs_t_learner = self._normalize_obs_tensor(obs_t_raw) if self.normalize_observations else obs_t_raw
            action = self.learner_actor.get_eval_action(obs_t_learner)
            final_action = action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action.squeeze(0).detach().cpu().numpy()

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
                final_action, normalized_action, selected_action_index = self.get_training_action(env_obs)

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

    def update(self):
        self.learner_actor.train()
        self.learner_critic.train()

        num_blocks = int(self.args["learner_critic_utd_ratio"])
        batch_size = int(self.args["learner_critic_batch_size"])

        data = self._prepare_sampled_batch(self.replay_buffer.sample(batch_size))
        mini_batches = reshape_into_blocks(data, num_blocks)

        info = {}
        for i in range(num_blocks):
            self.update_step += 1
            sub_batch = {k: v[i] for k, v in mini_batches.items()}

            critic_info = self.update_critic(sub_batch)
            info.update(critic_info)
            self.update_target_learner_critic()

            if self.update_step % self.policy_delay == 0:
                actor_info = self.update_learner_actor(sub_batch)
                info.update(actor_info)
                self.update_target_learner_actor()
        return info

    def update_critic(self, learner_data: dict):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        q_loss, q_mean, q_uncertainty = self._compiled_compute_learner_critic_loss(learner_data)
        self.critic_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        l2normalize_network(self.learner_critic)
        return {
            "learner/losses/qf_loss": q_loss.item(),
            "learner/values/qf_mean": q_mean.item(),
            "learner/values/qf_uncertainty": q_uncertainty.item(),
        }

    def _compute_learner_critic_loss(self, data: dict):
        with torch.no_grad():
            next_actions = self.learner_actor.get_target_action(data["next_observations"], self.target_policy_noise)
            next_qs, next_q_infos = self.learner_critic.get_target_value_with_info(
                data["next_observations"], next_actions
            )
            min_idx = next_qs.squeeze(-1).argmin(dim=0)
            stacked_log_probs = torch.stack([info["log_prob"] for info in next_q_infos], dim=0)
            next_q_log_probs = stacked_log_probs[min_idx, torch.arange(min_idx.shape[0], device=self.device)]

        pred_qs, pred_q_infos = self.learner_critic.get_value_with_info(data["observations"], data["actions"])
        gamma_n = float(self.gamma) ** int(self.args["nstep"])

        q_loss = 0.0
        for pred_info in pred_q_infos:
            loss = categorical_td_loss(
                pred_log_probs=pred_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=data["rewards"],
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

    def _compute_learner_actor_loss(self, data: dict):
        obs = data["observations"]
        obs_raw = data["observations_raw"]

        learner_action = self.learner_actor.get_eval_action(obs)
        q_pi = self.learner_critic.get_value(obs, learner_action).mean(dim=0).squeeze(-1)
        actor_loss = (-1.0 * q_pi).mean()

        with torch.no_grad():
            candidate_actions: List[torch.Tensor] = [learner_action.detach()]
            for i, oracle_actor in enumerate(self._oracle_actor_list):
                std = float(self.oracle_std_multipliers[i])
                candidate_actions.append(oracle_actor.get_action(obs_raw, std))

            q_values = []
            for cand_action in candidate_actions:
                q = self.learner_critic.get_value(obs, cand_action).mean(dim=0).squeeze(-1)
                q_values.append(q)
            all_q = torch.stack(q_values, dim=0)
            best_idx = all_q.argmax(dim=0)

            idx = best_idx.view(1, -1, 1).expand(1, best_idx.shape[0], learner_action.shape[-1])
            stacked_actions = torch.stack(candidate_actions, dim=0)
            guide_actions = stacked_actions.gather(0, idx).squeeze(0)
            q_g = all_q.gather(0, best_idx.view(1, -1)).squeeze(0)

            advantage = q_g - q_pi.detach()
            reg_weight = self.cup_beta_1 * torch.minimum(advantage, self.cup_beta_2 * torch.abs(q_pi.detach()))
            reg_weight = torch.clamp(reg_weight, min=0.0)

        mse_per = F.mse_loss(learner_action, guide_actions, reduction="none").mean(dim=-1)
        cup_loss = (reg_weight * mse_per).mean()

        oracle_selected_ratio = (best_idx != 0).float().mean()

        return actor_loss, q_pi.mean(), cup_loss, advantage.mean(), reg_weight.mean(), mse_per.mean(), oracle_selected_ratio

    def update_learner_actor(self, data: dict):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        (
            actor_loss,
            q_pi_mean,
            cup_loss,
            adv_mean,
            reg_weight_mean,
            mse_mean,
            oracle_selected_ratio,
        ) = self._compiled_compute_learner_actor_loss(data)

        total_loss = actor_loss + cup_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.policy_max_grad_norm)
        self.actor_optimizer.step()
        l2normalize_network(self.learner_actor)

        return {
            "learner/losses/actor_loss": actor_loss.item(),
            "learner/losses/cup_loss": cup_loss.item(),
            "learner/values/q_pi_mean": q_pi_mean.item(),
            "cup/advantage": adv_mean.item(),
            "cup/reg_weight": reg_weight_mean.item(),
            "cup/mse_loss": mse_mean.item(),
            "cup/oracle_selected_ratio": oracle_selected_ratio.item(),
        }

    def update_target_learner_critic(self):
        with torch.no_grad():
            self.learner_critic.update_target(self.args["tau"])

    def update_target_learner_actor(self):
        with torch.no_grad():
            self.learner_actor.update_target(self.args["tau"])

    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "learner_actor_state_dict": self.learner_actor.state_dict(),
                "learner_critic_state_dict": self.learner_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "oracles_names": list(self.oracles_names),
            }
        )
        for i, name in enumerate(self.oracles_names):
            checkpoint[f"oracles_actor_state_dict_{i}"] = self.oracles_actors[name].state_dict()

        checkpoint["rng_state_torch"] = torch.get_rng_state()
        checkpoint["rng_state_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        checkpoint["rng_state_numpy"] = np.random.get_state()
        checkpoint["rng_state_python"] = random.getstate()

        buf = np.array(self.action_selection_buffer, dtype=int) if len(self.action_selection_buffer) > 0 else np.array([], dtype=int)
        checkpoint["action_selection_buffer"] = buf
        checkpoint["action_selection_buffer_maxlen"] = int(self.action_selection_buffer.maxlen or self.args["action_selection_buffer_size"])

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
        self.learner_critic.load_state_dict(checkpoint["learner_critic_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        saved_names = checkpoint["oracles_names"]
        if list(saved_names) != list(self.oracles_names):
            print("[CUPAlgo] Warning: oracles_names mismatch between checkpoint and current model. Loading in current order.")

        for i, name in enumerate(self.oracles_names):
            k_actor = f"oracles_actor_state_dict_{i}"
            if k_actor in checkpoint:
                self.oracles_actors[name].load_state_dict(checkpoint[k_actor])

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
            print(f"[CUPAlgo] Warning: failed to restore RNG state: {e}")

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
