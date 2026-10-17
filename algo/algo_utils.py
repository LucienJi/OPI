import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import h5py
import torch
import random
import re
import math
import glob
try:
    import torchvision.transforms.functional as TF
    import torchvision.transforms as T
except Exception as _e:
    TF = None
    T = None
    _TORCHVISION_IMPORT_ERROR = _e
import copy
from torch.utils._pytree import tree_map
import csv
from typing import Dict, Tuple

def save_eval_results_to_csv(checkpoint_dir, run_name, eval_results, *, eval_dir=None):
    target_dir = eval_dir or checkpoint_dir
    filename = run_name if str(run_name).endswith(".csv") else f"{run_name}.csv"
    csv_path = os.path.join(target_dir, filename)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    current_fieldnames = sorted(eval_results.keys())

    file_exists = os.path.exists(csv_path)
    need_new_header = True

    if file_exists:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            try:
                existing_fieldnames = next(reader)
                if sorted(existing_fieldnames) == current_fieldnames:
                    need_new_header = False
            except StopIteration:
                pass

    if file_exists and need_new_header:
        existing_data = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data.append(row)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=current_fieldnames)
            writer.writeheader()
            for row in existing_data:
                filtered_row = {k: row.get(k, '') for k in current_fieldnames}
                writer.writerow(filtered_row)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=current_fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(eval_results)

    print(f"Evaluation results saved to {csv_path}")

def reshape_into_blocks(batch: dict, num_blocks: int):
    return tree_map(
        lambda x: x.reshape(num_blocks, -1, *x.shape[1:]),
        batch
    )


def move_batch_to_device(batch,device="cpu"):
    for k,v in batch.items():
        if isinstance(v, np.ndarray):
             t = torch.from_numpy(v)
             if t.dtype != torch.float32:
                t = t.float()
             batch[k] = t.to(device, non_blocking=True)
             if batch[k].ndim == 0:
                batch[k] = batch[k].unsqueeze(0)
        elif torch.is_tensor(v):
             batch[k] = v.to(device, non_blocking=True)
             if batch[k].ndim == 0:
                batch[k] = batch[k].unsqueeze(0)
    return batch

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "*.pt")
    ckpts = glob.glob(pattern)

    if not ckpts:
        return None

    latest_step = -1
    latest_ckpt = None

    for ckpt in ckpts:
        if ckpt.endswith('_best.pt') or ckpt.endswith('_final.pt') or ckpt.endswith('_pretrain.pt'):
            continue

        basename = os.path.basename(ckpt)
        match = re.search(r'_(\d+)\.pt$', basename)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_ckpt = ckpt

    if latest_ckpt:
        print(f"Found latest checkpoint: {latest_ckpt} at step {latest_step}")

    return latest_ckpt

def find_final_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "*.pt")
    ckpts = glob.glob(pattern)

    if not ckpts:
        return None

    final_ckpt = None
    for ckpt in ckpts:
        if ckpt.endswith('_final.pt'):
            final_ckpt = ckpt
            break

    if final_ckpt:
        print(f"Found final checkpoint: {final_ckpt}")
        return final_ckpt
    else:
        print("No final checkpoint found")
        return None

def cleanup_old_checkpoints(checkpoint_dir, keep=10):
    if not os.path.exists(checkpoint_dir):
        return

    pattern = os.path.join(checkpoint_dir, "*.pt")
    ckpts = glob.glob(pattern)

    training_ckpts = []
    for ckpt in ckpts:
        if ckpt.endswith('_best.pt') or ckpt.endswith('_final.pt') or ckpt.endswith('_pretrain.pt'):
            continue

        basename = os.path.basename(ckpt)
        match = re.search(r'_(\d+)\.pt$', basename)
        if match:
            step = int(match.group(1))
            training_ckpts.append((step, ckpt))

    if len(training_ckpts) > keep:
        training_ckpts.sort(key=lambda x: x[0])

        for step, ckpt_path in training_ckpts[:-keep]:
            try:
                os.remove(ckpt_path)
                print(f"Removed old checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Warning: Could not remove checkpoint {ckpt_path}: {e}")


def extend_and_repeat(data, dim: int, repeat: int):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(dim).repeat_interleave(repeat, dim=dim)
    elif isinstance(data, dict):
        return {key: extend_and_repeat(value, dim, repeat) for key, value in data.items()}
    else:
        return data


def deep_copy_dict_with_tensors(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, dict):
            result[key] = deep_copy_dict_with_tensors(value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def get_frame(env, obs):
    if "agentview_image" in obs:
        return obs["agentview_image"]
    elif hasattr(env, "render"):
        return env.render()
    else:
        return None

def add_text_to_frame(frame, time_step, success_status, return_per_step=None, actions=None):
    frame_copy = frame.copy()

    if frame_copy.dtype != np.uint8:
        frame_copy = (frame_copy * 255).astype(np.uint8) if frame_copy.max() <= 1.0 else frame_copy.astype(np.uint8)

    if time_step is not None:
        cv2.putText(frame_copy, f"t={time_step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if return_per_step is not None:
        cv2.putText(frame_copy, f"return={return_per_step:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if actions is not None:
        actions = np.round(actions, 2)
        action_str = "[" + ", ".join(f"{a:.2f}" for a in actions) + "]"
        cv2.putText(frame_copy, f"action={action_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    if success_status is None:
        color = (128, 128, 128)
        text = "N/A"
    elif success_status:
        color = (0, 255, 0)
        text = "SUCCESS"
    else:
        color = (0, 0, 255)
        text = "FAILURE"

    cv2.putText(frame_copy, text, (10, frame_copy.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame_copy


def save_frames_as_video(frames, output_path, fps=30, downsample=1, target_size=(128, 128), time_steps=None, success_list=None, return_per_step_list=None, actions_list=None):
    all_frames = []

    for ep_idx, episode_frames in enumerate(frames):
        episode_time_steps = time_steps[ep_idx] if time_steps is not None else None
        episode_success = success_list[ep_idx] if success_list is not None else None
        episode_return_per_step = return_per_step_list[ep_idx] if return_per_step_list is not None else None
        for frame_idx, frame in enumerate(episode_frames):
            current_time_step = episode_time_steps[frame_idx] if episode_time_steps is not None else None
            current_return_per_step = episode_return_per_step[frame_idx] if episode_return_per_step is not None else None
            current_actions = actions_list[ep_idx][frame_idx] if actions_list is not None else None
            processed_frame = np.array(frame)

            if len(processed_frame.shape) == 3 and processed_frame.shape[0] == 3:
                processed_frame = processed_frame.transpose(1, 2, 0)

            if target_size is not None:
                if processed_frame.dtype != np.uint8:
                    processed_frame = (processed_frame * 255).astype(np.uint8) if processed_frame.max() <= 1.0 else processed_frame.astype(np.uint8)
                processed_frame = cv2.resize(processed_frame, target_size, interpolation=cv2.INTER_LINEAR)

            processed_frame = add_text_to_frame(processed_frame, current_time_step, episode_success, current_return_per_step, current_actions)

            all_frames.append(processed_frame)

    if downsample > 1:
        all_frames = all_frames[::downsample]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import imageio, sys
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        with imageio.get_writer(output_path, fps=fps, codec='h264', ffmpeg_params=['-crf', '23', '-preset', 'medium']) as writer:
            for frame in all_frames:
                writer.append_data(frame)
        sys.stderr = old_stderr
    print(f"Video saved to {output_path}")

def save_frames_as_video_async(frames_list, output_path, fps=30, downsample=1, target_size=(256, 256), time_steps=None, success_list=None, return_per_step_list=None, actions_list=None):
    frames_copy = [list(ep) for ep in frames_list]
    time_steps_copy = [list(ep) for ep in time_steps] if time_steps is not None else None
    success_list_copy = list(success_list) if success_list is not None else None
    return_per_step_list_copy = [list(ep) for ep in return_per_step_list] if return_per_step_list is not None else None
    actions_list_copy = list(actions_list) if actions_list is not None else None
    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(save_frames_as_video, frames_copy, output_path, fps, downsample, target_size, time_steps_copy, success_list_copy, return_per_step_list_copy,actions_list_copy)

    print(f"Started saving video to {output_path} in background")
    return executor


def preprocess_obs(obs,norm_stats):
    for key in obs.keys():
        if 'image' in key:
            obs[key] = torch.from_numpy(obs[key].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        elif key in norm_stats.keys():
            obs[key] = torch.from_numpy((obs[key] - norm_stats[key]['mean']) / norm_stats[key]['std']).float().unsqueeze(0)
        else:
            obs[key] = torch.from_numpy(obs[key].copy()).float().unsqueeze(0)
    return obs


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=False)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data

def move_to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, dict):
        return {key: move_to_tensor(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_tensor(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_tensor(item) for item in data)
    else:
        return data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_dict_mean(dict_list):
    if len(dict_list) == 0:
        return {}

    mean_dict = {}
    for key in dict_list[0].keys():
        if not isinstance(dict_list[0][key], torch.Tensor):
            continue
        mean_dict[key] = torch.stack([d[key] for d in dict_list]).mean()
    return mean_dict

def detach_dict(dictionary):
    result = {}
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach()
        else:
            result[k] = v
    return result

def cleanup_ckpt(ckpt_dir, keep=1):
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if len(ckpts) <= keep:
        return

    epoch_nums = []
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch_nums.append((int(match.group(1)), ckpt))

    epoch_nums.sort(reverse=True)

    for _, ckpt in epoch_nums[keep:]:
        os.remove(ckpt)

def get_last_ckpt(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not ckpts:
        return None

    latest_epoch = -1
    latest_ckpt = None
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt

    return latest_ckpt

def cosine_schedule(optimizer, total_steps, eta_min=0.0):
    def lr_lambda(step):
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * step / total_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_norm_stats(dataset_path, num_demos):
    all_states_data = []
    all_action_data = []
    all_force_data  = []

    with h5py.File(dataset_path, 'r') as dataset_file:
        demo_keys = [k for k in dataset_file['data'].keys() if k.startswith('demo_')]
        num_demos_available = len(demo_keys)
        num_demos_to_use = min(num_demos, num_demos_available)

        print(f"Computing robosuite normalization statistics using {num_demos_to_use} demonstrations from {dataset_path}...")

        for i in range(num_demos_to_use):
            demo_key = f'demo_{i}'

            states = dataset_file[f'data/{demo_key}/robot_states'][()].astype(np.float32)
            actions = dataset_file[f'data/{demo_key}/actions'][()].astype(np.float32)
            force = dataset_file[f'data/{demo_key}/force'][()].astype(np.float32)


            all_states_data.append(states)
            all_action_data.append(actions)
            all_force_data.append(force)

    states_array = np.concatenate(all_states_data, axis=0)
    actions_array = np.concatenate(all_action_data, axis=0)
    force_array = np.concatenate(all_force_data, axis=0)

    state_mean = np.mean(states_array, axis=0)
    state_std = np.std(states_array, axis=0)
    state_std = np.clip(state_std, 1e-4, np.inf)

    action_mean = np.mean(actions_array, axis=0)
    action_std = np.std(actions_array, axis=0)
    action_std = np.clip(action_std, 1e-4, np.inf)

    force_mean = np.mean(force_array, axis=0)
    force_std = np.std(force_array, axis=0)
    force_std = np.clip(force_std, 1e-4, np.inf)

    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "force_mean": force_mean,
        "force_std": force_std
    }

    print(f"State Mean shape: {stats['state_mean'].shape},Action Mean shape: {stats['action_mean'].shape},Force Mean shape: {stats['force_mean'].shape}")
    return stats


class RunningMeanStd:
    def __init__(self, shape: tuple = (), dtype=np.float32):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = 0

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


_DEFAULT_EPS: float = 1e-8


class ObservationNormalizer:
    def __init__(self, obs_dim: int, *, dtype=np.float32, eps: float = _DEFAULT_EPS):
        self.obs_dim = int(obs_dim)
        self.eps = float(eps)
        self.rms = RunningMeanStd(shape=(1, self.obs_dim), dtype=dtype)

        self._version: int = 0
        self._tensor_cache: Dict[Tuple[str, str], Tuple[int, torch.Tensor, torch.Tensor]] = {}

    def update(self, obs: np.ndarray):
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.rms.update(x)
        self._version += 1

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        mean = self.rms.mean.squeeze()
        std = np.sqrt(self.rms.var.squeeze() + self.eps)
        return (x - mean) / std

    def _get_mean_std_tensors(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype))
        cached = self._tensor_cache.get(key)
        if cached is not None:
            ver, mean_t, std_t = cached
            if ver == self._version:
                return mean_t, std_t

        mean_np = self.rms.mean.squeeze()
        var_np = self.rms.var.squeeze()
        mean_t = torch.from_numpy(mean_np).to(device=device, dtype=dtype)
        std_t = torch.sqrt(torch.from_numpy(var_np).to(device=device, dtype=dtype) + float(self.eps))
        self._tensor_cache[key] = (self._version, mean_t, std_t)
        return mean_t, std_t

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean_t, std_t = self._get_mean_std_tensors(obs.device, obs.dtype)
        return (obs - mean_t) / std_t

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "obs_dim": self.obs_dim, "eps": self.eps}

    def load_state_dict(self, state: dict):
        if "rms" in state:
            self.rms.load_state_dict(state["rms"])
            self.obs_dim = int(state.get("obs_dim", self.obs_dim))
            self.eps = float(state.get("eps", self.eps))
        else:
            self.rms.load_state_dict(state)
        self._version += 1


class RewardNormalizer:
    def __init__(self, gamma: float, normalized_g_max: float, *, dtype=np.float32, eps: float = _DEFAULT_EPS):
        self.gamma = float(gamma)
        self.normalized_g_max = float(normalized_g_max)
        self.eps = float(eps)

        self.G: float = 0.0
        self.G_rms = RunningMeanStd(shape=(1,), dtype=dtype)
        self.G_r_max: float = 0.0

    def update(self, reward: float, done: bool):
        d = 1.0 if bool(done) else 0.0
        self.G = self.gamma * (1.0 - d) * self.G + float(reward)
        self.G_rms.update(np.array([[self.G]], dtype=np.float32))
        self.G_r_max = max(self.G_r_max, abs(self.G))

    def denominator(self) -> float:
        var_val = self.G_rms.var[0] if isinstance(self.G_rms.var, np.ndarray) else float(self.G_rms.var)
        var_den = float(np.sqrt(var_val + self.eps))
        min_required = float(self.G_r_max / self.normalized_g_max) if self.normalized_g_max > 0 else 0.0
        return float(max(var_den, min_required, self.eps))

    def scale_rewards_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        denom = self.denominator()
        return rewards / float(denom)

    def state_dict(self) -> dict:
        return {"G": self.G, "G_rms": self.G_rms.state_dict(), "G_r_max": self.G_r_max, "gamma": self.gamma, "normalized_g_max": self.normalized_g_max, "eps": self.eps}

    def load_state_dict(self, state: dict):
        self.G = float(state.get("G", 0.0))
        if "G_rms" in state:
            self.G_rms.load_state_dict(state["G_rms"])
        else:
            self.G_rms.load_state_dict(state)
        self.G_r_max = float(state.get("G_r_max", 0.0))
        self.gamma = float(state.get("gamma", self.gamma))
        self.normalized_g_max = float(state.get("normalized_g_max", self.normalized_g_max))
        self.eps = float(state.get("eps", self.eps))


def categorical_td_loss(
    pred_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
    bin_values: torch.Tensor | None = None,
    delta_z: float | None = None,
) -> torch.Tensor:
    B = pred_log_probs.shape[0]
    device = pred_log_probs.device

    reward = reward.view(-1, 1)
    done = done.float().view(-1, 1)

    if bin_values is None:
        bin_values = torch.linspace(min_v, max_v, num_bins, device=device)
    bin_values = bin_values.to(device=device, dtype=pred_log_probs.dtype).view(1, -1)
    target_bin_values = reward + gamma * bin_values * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    if delta_z is None:
        delta_z = (max_v - min_v) / (num_bins - 1)
    b = (target_bin_values - min_v) / delta_z

    l = torch.floor(b).long()
    u = torch.ceil(b).long()

    l = torch.clamp(l, 0, num_bins - 1)
    u = torch.clamp(u, 0, num_bins - 1)

    target_probs = torch.exp(target_log_probs)

    m = torch.zeros_like(pred_log_probs)

    offset = (u.float() + (l == u).float() - b)

    m.scatter_add_(1, l, target_probs * offset)
    m.scatter_add_(1, u, target_probs * (b - l.float()))

    loss = -torch.sum(m.detach() * pred_log_probs, dim=1)

    return loss
