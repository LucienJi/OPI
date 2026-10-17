import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from algo.base_algo import BaseAlgo
from algo.algo_utils import reshape_into_blocks, save_eval_results_to_csv, categorical_td_loss
from data_buffer.replay_buffer import ReplayBuffer
from model.simba import DeterministicSimbaPolicy, SimbaCritics
from model.simba_base import l2normalize_network
from model.mlp import OraclePolicyBase


class LOKIAlgo(BaseAlgo):
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
        self.oracles_names: List[str] = list(self.oracles_actors.keys())
        self.num_oracles = len(self.oracles_names)

        for actor in self.oracles_actors.values():
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

        self.actor_optimizer = self.learner_actor.set_optimizer(
            self.args["actor_lr"], weight_decay=self.args["actor_weight_decay"]
        )
        self.critic_optimizer = self.learner_critic.set_optimizer(
            self.args["critic_lr"], weight_decay=self.args["critic_weight_decay"]
        )

        self.gamma = float(self.args["discount"])
        self.num_bins = int(self.args["num_bins"])
        self.min_v = float(self.args["min_v"])
        self.max_v = float(self.args["max_v"])
        self.learning_starts = int(self.args["learning_starts"])
        self.learner_critic_utd_ratio = int(self.args["learner_critic_utd_ratio"])
        self.learner_critic_batch_size = int(self.args["learner_critic_batch_size"])
        self.min_exploration_noise = float(self.args["min_exploration_noise"])
        self.max_exploration_noise = float(self.args["max_exploration_noise"])
        self.target_policy_noise = float(self.args["target_policy_noise"])
        self.task_specific_noise = float(self.args["task_specific_noise"])
        self.noise_clip = float(self.args["noise_clip"])
        self.max_grad_norm = float(self.args["max_grad_norm"])
        self.policy_max_grad_norm = float(self.args.get("policy_max_grad_norm", self.args["max_grad_norm"]))
        self.log_every = int(self.args["log_every"])
        self.policy_delay = int(self.args["policy_delay"])

        self.loki_N_min = int(self.args["loki_N_min"])
        self.loki_N_max = int(self.args["loki_N_max"])
        if self.loki_N_max < self.loki_N_min:
            self.loki_N_max = self.loki_N_min
        self.loki_K = random.randint(self.loki_N_min, self.loki_N_max)
        print(f"[LOKIAlgo] Sampled switch step K={self.loki_K} from [{self.loki_N_min}, {self.loki_N_max}]")

        self.use_compile = bool(self.args.get("use_compile", False))
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
                limit = int(self.args.get("dynamo_cache_size_limit", 0))
                if limit > 0:
                    dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), limit)
            except Exception:
                pass
            return torch.compile(fn, fullgraph=fullgraph)
        except Exception as e:
            print(f"[LOKIAlgo] torch.compile failed for {name}: {e}. Falling back to eager.")
            return fn

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

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
    def _oracle_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actions = []
        for i, name in enumerate(self.oracles_names):
            oracle = self.oracles_actors[name]
            std = float(self.oracle_std_multipliers[i])
            a = oracle.get_action(obs, std)
            actions.append(a)
        return torch.stack(actions, dim=0)

    def update_target_learner_critic(self):
        with torch.no_grad():
            self.learner_critic.update_target(self.tau)

    def update_target_learner_actor(self):
        with torch.no_grad():
            self.learner_actor.update_target(self.tau)

    def update(self):
        self.learner_actor.train()
        self.learner_critic.train()

        num_blocks = int(self.learner_critic_utd_ratio)
        batch_size = int(self.learner_critic_batch_size)

        data = self.replay_buffer.sample(batch_size)
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
        act_pred = self.learner_actor.get_eval_action(obs)
        in_imitation = self.global_step < self.loki_K

        if in_imitation:
            with torch.no_grad():
                oracle_actions = self._oracle_actions(obs)
                q_list = []
                for i in range(self.num_oracles):
                    qs = self.learner_critic.get_value(obs, oracle_actions[i])
                    q_mean = qs.mean(dim=0).squeeze(-1)
                    q_list.append(q_mean)
                all_q = torch.stack(q_list, dim=0)
                best_idx = all_q.argmax(dim=0)
                idx = best_idx.view(1, -1, 1).expand(1, best_idx.shape[0], oracle_actions.shape[-1])
                best_actions = oracle_actions.gather(0, idx).squeeze(0).detach()
            actor_loss = F.mse_loss(act_pred, best_actions)
            bc_loss = actor_loss
        else:
            q_pi = self.learner_critic.get_value(obs, act_pred).mean(dim=0).squeeze(-1)
            actor_loss = -(q_pi.mean())
            bc_loss = 0.0

        return actor_loss, q_pi.mean() if not in_imitation else 0.0, bc_loss, in_imitation

    def update_learner_actor(self, data: dict):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        actor_loss, q_pi_mean, bc_loss, in_imitation = self._compiled_compute_learner_actor_loss(data)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.learner_actor.parameters(), self.policy_max_grad_norm)
        self.actor_optimizer.step()
        l2normalize_network(self.learner_actor)

        return {
            "learner/losses/actor_loss": actor_loss,
            "learner/values/q_pi_mean": q_pi_mean,
            "loki/phase": 0.0 if in_imitation else 1.0,
            "loki/bc_loss": bc_loss
        }

    @torch.no_grad()
    def get_training_action(self, raw_obs: np.ndarray):
        obs_t = self._obs_to_tensor(raw_obs)
        action = self.learner_actor.get_action(obs_t, self.task_specific_noise).squeeze(0)
        exploration_noise = self.get_action_noise(
            self.global_step, self.total_timesteps, self.min_exploration_noise, self.max_exploration_noise
        )
        action = self.add_noise_to_action(action, exploration_noise, clip_noise=self.noise_clip)
        final_action = action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action.cpu().numpy(), action.cpu().numpy()

    @torch.no_grad()
    def get_inference_action(self, raw_obs: np.ndarray, eval_oracle=False, oracle_name=None, **kwargs):
        obs_t = self._obs_to_tensor(raw_obs)
        if eval_oracle and oracle_name is not None:
            oracle_actor = self.oracles_actors[oracle_name]
            oracle_idx = self.oracles_names.index(oracle_name)
            std = float(self.oracle_std_multipliers[oracle_idx])
            action = oracle_actor.get_action(obs_t, std).squeeze(0)
            final_action = action * oracle_actor.action_scale + oracle_actor.action_bias
        else:
            action = self.learner_actor.get_eval_action(obs_t).squeeze(0)
            final_action = action * self.learner_actor.action_scale + self.learner_actor.action_bias
        return final_action.cpu().numpy()

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

        env_obs, _ = env.reset(seed=self.args["seed"])

        while self.global_step < self.total_timesteps:
            self.global_step += 1

            final_action, normalized_action = self.get_training_action(env_obs)
            next_obs, reward, terminated, truncated, info = env.step(final_action)
            done = terminated or truncated

            if isinstance(info, dict) and "episode" in info:
                self.log_local_train_episode(
                    global_step=self.global_step,
                    episode_reward=info["episode"]["r"],
                    episode_length=info["episode"]["l"],
                )

            self.replay_buffer.add_transition(
                {
                    "observations": env_obs,
                    "next_observations": next_obs,
                    "actions": normalized_action,
                    "rewards": float(reward),
                    "dones": bool(done),
                    "truncated": bool(truncated),
                    "action_mask": 0,
                    "global_step": int(self.global_step),
                }
            )
            env_obs = next_obs if not done else env.reset()[0]

            train_info = None
            if self.global_step >= self.learning_starts:
                train_info = self.update()

            do_log = self.use_wandb and (self.global_step % self.log_every == 0)
            if do_log:
                info_payload = train_info or {}
                wandb.log(info_payload, step=self.global_step)

            if (self.global_step + 1) % self.eval_every == 0:
                self.learner_actor.eval()
                eval_results = self.evaluate(eval_env or env, num_episodes=self.eval_episodes)
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

            self.progress_bar.update(1)

        return self._after_run()

    def _save_model_checkpoint(self, checkpoint: dict) -> dict:
        checkpoint.update(
            {
                "learner_actor": self.learner_actor.state_dict(),
                "learner_critic": self.learner_critic.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "loki_K": self.loki_K,
            }
        )
        return checkpoint

    def _load_model_checkpoint(self, checkpoint: dict) -> dict:
        if "learner_actor" in checkpoint:
            self.learner_actor.load_state_dict(checkpoint["learner_actor"])
        if "learner_critic" in checkpoint:
            self.learner_critic.load_state_dict(checkpoint["learner_critic"])
        if "actor_opt" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        if "critic_opt" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        if "loki_K" in checkpoint:
            self.loki_K = int(checkpoint["loki_K"])
        return checkpoint
