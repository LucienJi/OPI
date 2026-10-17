import torch
import os
import time
import re
import numpy as np
import wandb
import csv
from tqdm import tqdm
from typing import Optional

from algo.algo_utils import find_latest_checkpoint, cleanup_old_checkpoints, save_frames_as_video_async, get_frame, set_seed
from data_buffer.replay_buffer import ReplayBuffer


class BaseAlgo:
    RESUME_KEEP_CURRENT = (
        "total_timesteps",
        "run_name",
        "checkpoint_dir",
        "replay_dir",
        "eval_dir",
        "eval_root_dir",
        "video_dir",
    )

    def __init__(
        self,
        args,
    ):
        self.args = args
        self.device = args['device']
        self.discount = args['discount']
        self.tau = args['tau']

        self.use_wandb = self.args['use_wandb']
        self.wandb_run = None

        self.global_step = 0
        self.update_step = 0
        self.start_time = None
        self.best_eval_return = -np.inf
        self.best_success_rate = 0.0

        self.progress_bar = None

        self.video_executors = []

        os.makedirs(self.args["checkpoint_dir"], exist_ok=True)
        self.checkpoint_dir = self.args["checkpoint_dir"]
        self.eval_dir = self.args.get("eval_dir", self.checkpoint_dir)
        self.video_dir = self.args.get("video_dir", self.checkpoint_dir)
        eval_every = self.args['eval_every']
        save_freq = self.args['save_freq']
        self.eval_every = eval_every
        self.save_freq = save_freq
        self.eval_episodes = self.args.get("eval_episodes", 10)

        self.local_log_enabled = bool(self.args.get("save_local_logs", True))
        self.save_local_eval_episodes = bool(self.args.get("save_local_eval_episodes", False))
        self.action_selection_log_every = int(self.args.get("action_selection_log_every", 1000))
        self.local_log_dir = None
        self._local_log_paths = {}
        self._action_selection_buffer = []
        if self.local_log_enabled:
            self._init_local_logger()


    def __del__(self):
        self.wait_for_video_threads()

    def wait_for_video_threads(self, timeout=None):
        for executor in self.video_executors:
            if executor is not None:
                executor.shutdown(wait=True)
        self.video_executors = []

    def init_wandb(self, run_id=None):
        if self.use_wandb:
            config = self.args.copy()

            if "wandb_project_name" not in config:
                config["wandb_project_name"] = "OPI"
            if "wandb_entity" not in config:
                config["wandb_entity"] = None

            if "full_config" in config:
                del config["full_config"]

            wandb_group = config.get("wandb_group", config.get("method_name"))
            wandb_name = config.get("wandb_run_name") or config.get("run_name") or f"{wandb_group}-seed{config.get('seed', 42)}"
            tags = self._build_wandb_tags()
            self.wandb_run = wandb.init(
                project=config["wandb_project_name"],
                entity=config["wandb_entity"],
                config=config,
                name=wandb_name,
                id=run_id,
                resume="allow" if run_id else None,
                monitor_gym=False,
                group=wandb_group,
                tags=tags if tags else None,
            )

    def _build_wandb_tags(self):
        tags = []
        full_cfg = self.args.get("full_config", None)

        def _get_by_path(cfg: dict, path: str):
            cur = cfg
            for key in str(path).split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            return cur

        def _format_value(val):
            if val is None:
                return None
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, float):
                return f"{val:g}"
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return None
                if len(val) <= 6:
                    return ",".join([_format_value(v) for v in val if v is not None])
                return f"len{len(val)}"
            return str(val)

        def _add_tag(key, val):
            v = _format_value(val)
            if v is None or v == "":
                return
            tags.append(f"{key}={v}")

        if isinstance(full_cfg, dict):
            method_name = full_cfg.get("method_name")
            if method_name:
                _add_tag("method", method_name)
            env_cfg = full_cfg.get("env", {})
            if isinstance(env_cfg, dict):
                env_name = env_cfg.get("env_name") or env_cfg.get("domain_name")
                task_name = env_cfg.get("task_name")
                if env_name and task_name:
                    _add_tag("env", f"{env_name}/{task_name}")
                elif env_name:
                    _add_tag("env", env_name)
                if env_cfg.get("mode") is not None:
                    _add_tag("mode", env_cfg.get("mode"))

        extra_tags = self.args.get("wandb_tags", None)
        if isinstance(extra_tags, (list, tuple)):
            tags.extend([str(t) for t in extra_tags if t])
        elif isinstance(extra_tags, str):
            tags.extend([t for t in extra_tags.split(",") if t.strip()])

        tag_fields = self.args.get("wandb_tag_fields", None)
        if isinstance(tag_fields, (list, tuple)):
            for path in tag_fields:
                if not path:
                    continue
                value = None
                if isinstance(full_cfg, dict):
                    value = _get_by_path(full_cfg, path)
                if value is None:
                    value = _get_by_path(self.args, path)
                key = str(path).split(".")[-1]
                _add_tag(key, value)
        elif isinstance(tag_fields, str):
            for path in [p.strip() for p in tag_fields.split(",") if p.strip()]:
                value = None
                if isinstance(full_cfg, dict):
                    value = _get_by_path(full_cfg, path)
                if value is None:
                    value = _get_by_path(self.args, path)
                key = str(path).split(".")[-1]
                _add_tag(key, value)

        seen = set()
        deduped = []
        for t in tags:
            if not t or t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        return deduped


    @torch.no_grad()
    def get_training_action(self, raw_obs, **training_kwargs):
        raise NotImplementedError("get_training_action is not implemented")

    @torch.no_grad()
    def get_inference_action(self, raw_obs, **inference_kwargs):
        raise NotImplementedError("get_inference_action is not implemented")

    def evaluate(
        self,
        env,
        num_episodes=10,
        save_name=None,
        eval_tag: Optional[str] = None,
        log_local_eval: bool = True,
        save_best_model: bool = True,
        **eval_kwargs,
    ):
        returns = []
        episode_lengths = []
        if self.progress_bar:
            self.progress_bar.set_description("Evaluating...")
        frames_list = []
        time_steps_list = []
        return_per_step_list = []
        success_list = []
        actions_list = []

        for episode in range(num_episodes):
            frames = []
            time_steps = []
            return_per_step = []
            episode_return = 0
            episode_length = 0
            actions = []
            seeds = self.args['seed']
            episode_seed = seeds + episode
            set_seed(episode_seed)
            env_obs, _ = env.reset(seed=episode_seed)
            done = False

            while not done:
                with torch.no_grad():
                    action = self.get_inference_action(env_obs, **eval_kwargs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    actions.append(action)
                    if self.args["save_video"]:
                        frames.append(get_frame(env, next_obs))
                    done = terminated or truncated
                    episode_return += reward
                    episode_length += 1
                    time_steps.append(episode_length)
                    return_per_step.append(episode_return)
                    env_obs = next_obs
                    if done:
                        if 'is_success' in info:
                            success_list.append(info['is_success'])
                        else:
                            success_list.append(None)
                        break

            returns.append(episode_return)
            episode_lengths.append(episode_length)
            frames_list.append(frames)
            time_steps_list.append(time_steps)
            return_per_step_list.append(return_per_step)
            actions_list.append(np.stack(actions, axis=0))

        if self.progress_bar:
            self.progress_bar.set_description("Training")

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)

        eval_results = {
            'step': self.global_step,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length,
        }
        if len(success_list) > 0 and success_list[0] is not None:
            eval_results['success_rate'] = np.mean(success_list)

        if log_local_eval and self.save_local_eval_episodes:
            self._log_local_eval_episodes(
                returns=returns,
                episode_lengths=episode_lengths,
                success_list=success_list,
                eval_tag=eval_tag,
            )

        if self.args["save_video"] and ((self.global_step + 1) % self.args["save_video_every"] == 0):
            save_name = save_name or f"{self.args['run_name']}_{self.global_step}_eval.mp4"
            video_path = os.path.join(self.video_dir, save_name)
            executor = save_frames_as_video_async(
                frames_list, video_path, fps=self.args.get("save_video_fps", 30),
                downsample=1, time_steps=time_steps_list, success_list=success_list,
                return_per_step_list=return_per_step_list, actions_list=actions_list
            )
            self.video_executors.append(executor)

        if save_best_model:
            has_success = len(success_list) > 0 and success_list[0] is not None
            best_path = os.path.join(self.args["checkpoint_dir"], f"{self.args['run_name']}_best.pt")

            if has_success:
                success_rate = float(np.mean(success_list))
                if (success_rate > self.best_success_rate) or (
                    np.isclose(success_rate, self.best_success_rate) and mean_return > self.best_eval_return
                ):
                    self.best_success_rate = success_rate
                    self.best_eval_return = mean_return
                    self.save_model(best_path)
            else:
                if mean_return > self.best_eval_return:
                    self.best_eval_return = mean_return
                    self.save_model(best_path)

        return eval_results

    def _init_local_logger(self):
        group_name = self._infer_group_name()
        eval_root = self.args.get("eval_root_dir")
        if eval_root is None:
            eval_root = os.path.join(os.path.dirname(self.checkpoint_dir), "eval")
        task_name = self.args.get("task_name")
        seed = self.args.get("seed", "unknown")

        task_dir = os.path.join(eval_root, str(task_name)) if task_name else eval_root
        train_dir = os.path.join(task_dir, "training", group_name)
        eval_dir = os.path.join(task_dir, "eval", group_name)
        selection_dir = os.path.join(task_dir, "actor_selection", group_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(selection_dir, exist_ok=True)

        self.local_log_dir = task_dir
        self._local_log_paths = {
            "train_episodes": os.path.join(train_dir, f"seed{seed}.csv"),
            "eval_episodes": os.path.join(eval_dir, f"seed{seed}.csv"),
            "action_selection": os.path.join(selection_dir, f"seed{seed}.csv"),
        }

    def _infer_group_name(self) -> str:
        name = (
            self.args.get("method_name")
            or self.args.get("algo_name")
            or self.args.get("run_name")
            or "run"
        )
        name = str(name)
        name = re.sub(r"([_-]seed\d+)$", "", name)
        return name

    def _append_local_csv_row(self, path: str, header: list, row: dict):
        if not self.local_log_enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_local_csv_rows(self, path: str, header: list, rows: list):
        if not self.local_log_enabled or not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def log_local_train_episode(self, *, global_step: int, episode_reward: float, episode_length: int):
        if not self.local_log_enabled:
            return
        path = self._local_log_paths.get("train_episodes")
        if path is None:
            return
        header = ["global_step", "episode_reward", "episode_length"]
        row = {
            "global_step": int(global_step),
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
        }
        self._append_local_csv_row(path, header, row)

    def _log_local_eval_episodes(self, *, returns, episode_lengths, success_list, eval_tag: Optional[str] = None):
        if not self.local_log_enabled:
            return
        path = self._local_log_paths.get("eval_episodes")
        if eval_tag:
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(eval_tag))
            seed = self.args.get("seed", "unknown")
            path = os.path.join(self.local_log_dir, f"eval_episodes_{safe_tag}_seed{seed}.csv")
        if path is None:
            return
        has_success = len(success_list) > 0 and success_list[0] is not None
        header = ["global_step", "eval_episode", "return", "length"]
        if has_success:
            header.append("success")
        for idx, (ret, length) in enumerate(zip(returns, episode_lengths)):
            row = {
                "global_step": int(self.global_step),
                "eval_episode": int(idx),
                "return": float(ret),
                "length": int(length),
            }
            if has_success:
                row["success"] = float(success_list[idx])
            self._append_local_csv_row(path, header, row)

    def log_local_action_selection(self, *, global_step: int, selected_action_index: int):
        if not self.local_log_enabled:
            return
        self._action_selection_buffer.append(
            {
                "global_step": int(global_step),
                "selected_action_index": int(selected_action_index),
            }
        )
        if self.action_selection_log_every > 0 and len(self._action_selection_buffer) >= self.action_selection_log_every:
            self._flush_action_selection_logs()

    def _flush_action_selection_logs(self):
        if not self.local_log_enabled or not self._action_selection_buffer:
            return
        path = self._local_log_paths.get("action_selection")
        if path is None:
            self._action_selection_buffer.clear()
            return
        header = ["global_step", "selected_action_index"]
        self._append_local_csv_rows(path, header, self._action_selection_buffer)
        self._action_selection_buffer.clear()


    def _pre_run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        self.env = env
        if eval_env is None:
            eval_env = env
        self.replay_buffer = replay_buffer
        print(f"Found {self.replay_buffer.size()} transitions in the replay buffer")

        if self.args.get("resume", False):
            resume_path = find_latest_checkpoint(self.args["checkpoint_dir"])
            if resume_path and os.path.exists(resume_path):
                self.load_model(resume_path)
                print(f"Resuming training from step {self.global_step}")
            else:
                print("No checkpoint found for resuming, starting from scratch")

        if self.use_wandb:
            wandb_id = self.args.get("wandb_id", None)
            self.init_wandb(run_id=wandb_id)

        self.start_time = time.time()
        self.total_timesteps = self.args["total_timesteps"]
        remaining_timesteps = self.total_timesteps - self.global_step

        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="global_step",
            initial=self.global_step
        )

    def _after_run(self):
        self._flush_action_selection_logs()
        final_path = os.path.join(self.args["checkpoint_dir"], f"{self.args['run_name']}_final.pt")
        self.save_model(final_path)

        self.progress_bar.close()
        self.progress_bar = None

        self.wait_for_video_threads()

        if self.use_wandb:
            wandb.finish()

        print("Training completed.")
        return self.best_eval_return

    def run(self, env, replay_buffer: ReplayBuffer, eval_env=None):
        raise NotImplementedError("run is not implemented")


    def _save_model_checkpoint(self, checkpoint: dict):
        raise NotImplementedError("_save_model_checkpoint is not implemented")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        args_dict = vars(self.args) if hasattr(self.args, "__dict__") else self.args

        checkpoint = {
            "global_step": self.global_step,
            "update_step": self.update_step,
            "args": args_dict,
            "best_eval_return": self.best_eval_return,
        }

        if self.use_wandb and self.wandb_run:
            checkpoint["wandb_id"] = self.wandb_run.id


        checkpoint = self._save_model_checkpoint(checkpoint)


        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
        print(f"Model saved to {path}")

        basename = os.path.basename(path)
        if not (basename.endswith('_best.pt') or basename.endswith('_final.pt') or basename.endswith('_pretrain.pt')):
            cleanup_old_checkpoints(self.checkpoint_dir, keep=self.args.get("keep_checkpoint_num", 10))

    def _load_model_checkpoint(self, checkpoint: dict):
        raise NotImplementedError("_load_model_checkpoint is not implemented")
    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist, skipping loading")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.global_step = checkpoint["global_step"]
        self.update_step = checkpoint.get("update_step", 0)

        if "best_eval_return" in checkpoint:
            self.best_eval_return = checkpoint["best_eval_return"]
        preserved = {key: self.args[key] for key in self.RESUME_KEEP_CURRENT if key in self.args}
        self.args.update(checkpoint["args"])
        for key, value in preserved.items():
            if self.args.get(key) != value:
                print(f"[resume] using current {key}={value!r} (checkpoint had {self.args.get(key)!r})")
            self.args[key] = value
        self.use_wandb = self.args.get("use_wandb", self.use_wandb)
        if self.use_wandb and "wandb_id" in checkpoint:
            self.args["wandb_id"] = checkpoint["wandb_id"]

        checkpoint = self._load_model_checkpoint(checkpoint)
        print(f"Model loaded from {path} at step {self.global_step}")
        return True
