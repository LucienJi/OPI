from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from data_buffer.trajectory_store import TrajectoryStore


class ReplayBuffer:
    def __init__(
        self,
        replay_dir: str,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        capacity: int = 200_000,
        discount: float = 0.99,
        device: Optional[str] = None,
        num_masks: int = 5,
        save_episodes_to_disk: bool = True,
        preload: bool = True,
    ):
        import pathlib

        self.replay_dir = pathlib.Path(replay_dir)
        self.replay_dir.mkdir(exist_ok=True, parents=True)

        self.obs_shape = tuple(observation_shape)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.discount = float(discount)
        self.num_masks = int(num_masks)
        if self.num_masks <= 0:
            raise ValueError("num_masks must be positive")

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_episodes_to_disk = bool(save_episodes_to_disk)

        self.buf = {
            "observations": torch.empty((self.capacity, *self.obs_shape), dtype=torch.float32, device=self.device),
            "next_observations": torch.empty((self.capacity, *self.obs_shape), dtype=torch.float32, device=self.device),
            "actions": torch.empty((self.capacity, self.action_dim), dtype=torch.float32, device=self.device),
            "rewards": torch.empty((self.capacity,), dtype=torch.float32, device=self.device),
            "dones": torch.empty((self.capacity,), dtype=torch.bool, device=self.device),
            "action_mask": torch.empty((self.capacity,), dtype=torch.int32, device=self.device),
            "clean_action": torch.empty((self.capacity, self.action_dim), dtype=torch.float32, device=self.device),
            "global_step": torch.empty((self.capacity,), dtype=torch.int32, device=self.device),
        }

        self.ptr = 0
        self.size_ = 0
        self.total_added = 0

        self.pool_gpu = torch.empty((self.num_masks, self.capacity), dtype=torch.long, device=self.device)
        self.pool_cpu = np.empty((self.num_masks, self.capacity), dtype=np.int32)
        self.pool_count = np.zeros((self.num_masks,), dtype=np.int32)

        self.pool_pos = np.full((self.capacity,), -1, dtype=np.int32)
        self.mask_of_index = np.full((self.capacity,), -1, dtype=np.int32)

        self._ep_obs = []
        self._ep_next_obs = []
        self._ep_act = []
        self._ep_rew = []
        self._ep_done = []
        self._ep_mask = []
        self._ep_clean_act = []
        self._ep_gs = []
        self._ep_trunc = []

        traj_path = self.replay_dir / "trajectories.traj"
        self.store: Optional[TrajectoryStore] = None
        if self.save_episodes_to_disk or traj_path.exists():
            self.store = TrajectoryStore(traj_path)

        if preload and self.store is not None:
            self._preload_from_store()

        print(
            f"ReplayBufferGPUFastSingleEnv init: device={self.device}, capacity={self.capacity}, "
            f"num_masks={self.num_masks}, size={self.size_}"
        )

    def size(self) -> int:
        return int(self.size_)

    def __len__(self) -> int:
        return int(self.size_)

    def _pool_remove_index(self, idx: int) -> None:
        old_mask = int(self.mask_of_index[idx])
        if old_mask < 0:
            return
        pos = int(self.pool_pos[idx])
        cnt = int(self.pool_count[old_mask])
        last_pos = cnt - 1
        if last_pos < 0:
            self.pool_pos[idx] = -1
            self.mask_of_index[idx] = -1
            return

        last_idx = int(self.pool_cpu[old_mask, last_pos])

        self.pool_cpu[old_mask, pos] = last_idx
        self.pool_pos[last_idx] = pos

        self.pool_gpu[old_mask, pos] = int(last_idx)

        self.pool_count[old_mask] = last_pos

        self.pool_pos[idx] = -1
        self.mask_of_index[idx] = -1

    def _pool_add_index(self, idx: int, mask: int) -> None:
        mask = int(mask)
        if not (0 <= mask < self.num_masks):
            raise ValueError(f"action_mask {mask} out of range [0, {self.num_masks-1}]")
        pos = int(self.pool_count[mask])
        if pos >= self.capacity:
            raise RuntimeError(f"Mask pool overflow: mask={mask}, pos={pos}, capacity={self.capacity}")

        self.pool_cpu[mask, pos] = int(idx)
        self.pool_gpu[mask, pos] = int(idx)

        self.pool_pos[idx] = pos
        self.mask_of_index[idx] = mask
        self.pool_count[mask] = pos + 1

    @torch.no_grad()
    def _write_at(self, idx: int, t: Dict[str, Any]) -> None:
        if self.size_ == self.capacity:
            self._pool_remove_index(idx)

        o = t["observations"]
        no = t["next_observations"]
        a = t["actions"]

        if torch.is_tensor(o):
            o_t = o.to(device=self.device, dtype=torch.float32, non_blocking=True).view(self.obs_shape)
        else:
            o_t = torch.from_numpy(np.asarray(o, np.float32).reshape(self.obs_shape)).to(self.device, non_blocking=True)

        if torch.is_tensor(no):
            no_t = no.to(device=self.device, dtype=torch.float32, non_blocking=True).view(self.obs_shape)
        else:
            no_t = torch.from_numpy(np.asarray(no, np.float32).reshape(self.obs_shape)).to(self.device, non_blocking=True)

        if torch.is_tensor(a):
            a_t = a.to(device=self.device, dtype=torch.float32, non_blocking=True).view(self.action_dim)
        else:
            a_t = torch.from_numpy(np.asarray(a, np.float32).reshape(self.action_dim)).to(self.device, non_blocking=True)

        self.buf["observations"][idx].copy_(o_t)
        self.buf["next_observations"][idx].copy_(no_t)
        self.buf["actions"][idx].copy_(a_t)

        self.buf["rewards"][idx] = float(t["rewards"])
        self.buf["dones"][idx] = bool(t["dones"])

        mask = int(t.get("action_mask", 0))
        self.buf["action_mask"][idx] = mask

        ca = t.get("clean_action", a)
        if torch.is_tensor(ca):
            ca_t = ca.to(device=self.device, dtype=torch.float32, non_blocking=True).view(self.action_dim)
        else:
            ca_t = torch.from_numpy(np.asarray(ca, np.float32).reshape(self.action_dim)).to(self.device, non_blocking=True)
        self.buf["clean_action"][idx].copy_(ca_t)

        self.buf["global_step"][idx] = int(t.get("global_step", 0))

        self._pool_add_index(idx, mask)

    def add_transition(self, transition: Dict[str, Any]) -> None:
        idx = int(self.ptr)
        self._write_at(idx, transition)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)
        self.total_added += 1

        self._ep_obs.append(np.asarray(transition["observations"], np.float32).reshape(self.obs_shape))
        self._ep_next_obs.append(np.asarray(transition["next_observations"], np.float32).reshape(self.obs_shape))
        self._ep_act.append(np.asarray(transition["actions"], np.float32).reshape(self.action_dim))
        self._ep_rew.append(float(transition["rewards"]))
        self._ep_done.append(bool(transition["dones"]))
        self._ep_mask.append(int(transition.get("action_mask", 0)))
        self._ep_clean_act.append(np.asarray(transition.get("clean_action", transition["actions"]), np.float32).reshape(self.action_dim))
        self._ep_gs.append(int(transition.get("global_step", 0)))
        self._ep_trunc.append(bool(transition.get("truncated", False)))

        if bool(transition["dones"]) or bool(transition.get("truncated", False)):
            self._finalize_episode()

    def _finalize_episode(self) -> None:
        if len(self._ep_act) == 0:
            return

        if self.save_episodes_to_disk and self.store is not None:
            L = len(self._ep_act)
            episode = {
                "observations": np.asarray(self._ep_obs, np.float32),
                "next_observations": np.asarray(self._ep_next_obs, np.float32),
                "actions": np.asarray(self._ep_act, np.float32).reshape(L, self.action_dim),
                "rewards": np.asarray(self._ep_rew, np.float32),
                "dones": np.asarray(self._ep_done, np.bool_),
                "action_mask": np.asarray(self._ep_mask, np.int32),
                "clean_action": np.asarray(self._ep_clean_act, np.float32).reshape(L, self.action_dim),
                "global_step": np.asarray(self._ep_gs, np.int32),
                "truncated": np.asarray(self._ep_trunc, np.bool_),
            }
            self.store.append_episode(episode)

        self._ep_obs.clear()
        self._ep_next_obs.clear()
        self._ep_act.clear()
        self._ep_rew.clear()
        self._ep_done.clear()
        self._ep_mask.clear()
        self._ep_clean_act.clear()
        self._ep_gs.clear()
        self._ep_trunc.clear()

    def flush_episode(self) -> None:
        self._finalize_episode()

    def close(self) -> None:
        if getattr(self, "store", None) is not None:
            self.store.close()
            self.store = None

    @torch.no_grad()
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size_ <= 0:
            raise RuntimeError("Replay buffer is empty")
        b = int(batch_size)
        idx = torch.randint(0, int(self.size_), (b,), device=self.device)

        dones = self.buf["dones"][idx]
        discounts = (~dones).to(torch.float32) * float(self.discount)

        return {
            "observations": self.buf["observations"][idx],
            "next_observations": self.buf["next_observations"][idx],
            "actions": self.buf["actions"][idx],
            "rewards": self.buf["rewards"][idx],
            "dones": dones,
            "discounts": discounts,
            "action_mask": self.buf["action_mask"][idx],
            "clean_action": self.buf["clean_action"][idx],
            "global_step": self.buf["global_step"][idx],
        }

    @torch.no_grad()
    def action_mask_sample(
        self,
        batch_size: int,
        target_action_mask: int,
        on_policy_ratio: float = 1.0,
        fallback_to_uniform: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.size_ <= 0:
            raise RuntimeError("Replay buffer is empty")

        target = int(target_action_mask)
        if not (0 <= target < self.num_masks):
            raise ValueError(f"target_action_mask {target} out of range [0, {self.num_masks-1}]")

        if not (0.0 <= float(on_policy_ratio) <= 1.0):
            raise ValueError(f"on_policy_ratio must be in [0,1], got {on_policy_ratio}")

        b = int(batch_size)
        n_on = int(b * float(on_policy_ratio))
        n_off = b - n_on

        cnt = int(self.pool_count[target])
        if cnt <= 0:
            if fallback_to_uniform:
                return self.sample(batch_size)
            raise RuntimeError(f"No transitions with action_mask={target} in buffer")

        if n_on <= 0:
            idx = torch.randint(0, int(self.size_), (b,), device=self.device)
        else:
            r = torch.randint(0, cnt, (n_on,), device=self.device)
            idx_on = self.pool_gpu[target, r]
            if n_off > 0:
                idx_off = torch.randint(0, int(self.size_), (n_off,), device=self.device)
                idx = torch.cat([idx_on, idx_off], dim=0)
                idx = idx[torch.randperm(idx.shape[0], device=self.device)]
            else:
                idx = idx_on

        dones = self.buf["dones"][idx]
        discounts = (~dones).to(torch.float32) * float(self.discount)

        return {
            "observations": self.buf["observations"][idx],
            "next_observations": self.buf["next_observations"][idx],
            "actions": self.buf["actions"][idx],
            "rewards": self.buf["rewards"][idx],
            "dones": dones,
            "discounts": discounts,
            "action_mask": self.buf["action_mask"][idx],
            "clean_action": self.buf["clean_action"][idx],
            "global_step": self.buf["global_step"][idx],
        }

    def _preload_from_store(self) -> None:
        if self.store is None:
            return
        for ep in self.store.read_tail_to_fill(self.capacity):
            L = int(ep["actions"].shape[0])
            for i in range(L):
                t = {
                    "observations": ep["observations"][i],
                    "next_observations": ep["next_observations"][i],
                    "actions": ep["actions"][i],
                    "rewards": float(ep["rewards"][i]),
                    "dones": bool(ep["dones"][i]),
                    "action_mask": int(ep["action_mask"][i]) if "action_mask" in ep else 0,
                    "clean_action": ep["clean_action"][i] if "clean_action" in ep else ep["actions"][i],
                    "global_step": int(ep["global_step"][i]) if "global_step" in ep else 0,
                }
                idx = int(self.ptr)
                self._write_at(idx, t)
                self.ptr = (self.ptr + 1) % self.capacity
                self.size_ = min(self.size_ + 1, self.capacity)
                self.total_added += 1
