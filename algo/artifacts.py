import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


def _safe_name(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "unknown"


def _get_cfg_value(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted_key.split("."):
        if node is None:
            return default
        try:
            node = node.get(part)
        except Exception:
            return default
    return node if node is not None else default


def infer_task_name(cfg: Any) -> str:
    explicit = _get_cfg_value(cfg, "artifacts.task_name", None)
    if explicit:
        return _safe_name(explicit)

    domain = _get_cfg_value(cfg, "env.domain_name", None)
    task = _get_cfg_value(cfg, "env.task_name", None)
    if domain and task:
        return _safe_name(f"{domain}_{task}")

    env_name = _get_cfg_value(cfg, "env.env_name", None)
    mode = _get_cfg_value(cfg, "env.mode", None)
    if env_name and mode:
        return _safe_name(f"{env_name}_{mode}")
    if env_name:
        return _safe_name(env_name)

    return "task"


def infer_method_name(cfg: Any) -> str:
    name = _get_cfg_value(cfg, "method_name", None)
    if name is None:
        name = _get_cfg_value(cfg, "training.method_name", None)
    if name is None:
        name = _get_cfg_value(cfg, "algo.method_name", None)
    return _safe_name(name or "method")


def infer_seed(cfg: Any) -> int:
    seed = _get_cfg_value(cfg, "training.seed", None)
    if seed is None:
        seed = _get_cfg_value(cfg, "seed", 42)
    return int(seed)


@dataclass
class ArtifactPaths:
    root_dir: str
    run_dir: str
    checkpoint_dir: str
    eval_root_dir: str
    eval_dir: str
    video_dir: str
    results_root_dir: str
    replay_dir: str


@dataclass
class ArtifactManager:
    task_name: str
    method_name: str
    seed: int
    run_name: str
    wandb_group: str
    wandb_run_name: str
    paths: ArtifactPaths

    @property
    def run_dir(self) -> str:
        return self.paths.run_dir

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ArtifactManager":
        task_name = infer_task_name(cfg)
        method_name = infer_method_name(cfg)
        seed = infer_seed(cfg)

        checkpoint_dir = _get_cfg_value(cfg, "training.checkpoint_dir", None)
        root_dir = _get_cfg_value(cfg, "training.artifacts_root", None)
        if not root_dir:
            root_dir = os.path.dirname(checkpoint_dir) if checkpoint_dir else "./checkpoints"

        eval_root_dir = _get_cfg_value(cfg, "training.eval_root_dir", None) or "./eval"

        root_dir = os.path.abspath(root_dir)
        eval_root_dir = os.path.abspath(eval_root_dir)
        run_dir = os.path.join(root_dir, task_name, method_name, f"seed{seed}")

        paths = ArtifactPaths(
            root_dir=root_dir,
            run_dir=run_dir,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"),
            eval_root_dir=eval_root_dir,
            eval_dir=os.path.join(eval_root_dir, task_name, "eval", method_name),
            video_dir=os.path.join(run_dir, "videos"),
            results_root_dir=os.path.join(run_dir, "results"),
            replay_dir=os.path.join(run_dir, "replay"),
        )

        run_name = f"{method_name}_seed{seed}"
        eval_run_name = f"seed{seed}"
        wandb_group = f"{task_name}/{method_name}"
        wandb_run_name = f"seed{seed}"

        return cls(
            task_name=task_name,
            method_name=method_name,
            seed=seed,
            run_name=run_name,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
            paths=paths,
        )

    def apply_to_cfg(self, cfg: Any) -> None:
        cfg.training.run_name = self.run_name
        cfg.training.checkpoint_dir = self.paths.checkpoint_dir
        cfg.training.eval_root_dir = self.paths.eval_root_dir
        cfg.training.eval_dir = self.paths.eval_dir
        cfg.training.eval_run_name = f"seed{self.seed}"
        cfg.training.video_dir = self.paths.video_dir
        cfg.training.results_root_dir = self.paths.results_root_dir
        cfg.training.replay_dir = self.paths.replay_dir
        cfg.training.wandb_group = self.wandb_group
        cfg.training.wandb_run_name = self.wandb_run_name
        cfg.training.task_name = self.task_name
        cfg.training.method_name = self.method_name

    def ensure_dirs(self) -> None:
        for path in [
            self.paths.run_dir,
            self.paths.checkpoint_dir,
            self.paths.eval_dir,
            self.paths.video_dir,
            self.paths.results_root_dir,
            self.paths.replay_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def write_run_json(self, cfg: Any, *, extra: Optional[Dict[str, Any]] = None) -> str:
        payload: Dict[str, Any] = {
            "task_name": self.task_name,
            "method_name": self.method_name,
            "seed": self.seed,
            "run_name": self.run_name,
            "wandb_group": self.wandb_group,
            "wandb_run_name": self.wandb_run_name,
            "paths": {
                "root_dir": self.paths.root_dir,
                "run_dir": self.paths.run_dir,
                "checkpoint_dir": self.paths.checkpoint_dir,
                "eval_root_dir": self.paths.eval_root_dir,
                "eval_dir": self.paths.eval_dir,
                "video_dir": self.paths.video_dir,
                "results_root_dir": self.paths.results_root_dir,
                "replay_dir": self.paths.replay_dir,
            },
        }
        if extra:
            payload.update(extra)

        payload["resolved_config"] = OmegaConf.to_container(cfg, resolve=True)

        out_path = os.path.join(self.paths.run_dir, "run.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return out_path


def build_artifact_paths(cfg: Any) -> ArtifactManager:
    return ArtifactManager.from_cfg(cfg)


def ensure_artifact_dirs(artifacts: ArtifactManager) -> None:
    artifacts.ensure_dirs()


def update_cfg_with_artifacts(cfg: Any, artifacts: ArtifactManager) -> None:
    OmegaConf.set_struct(cfg, False)
    artifacts.apply_to_cfg(cfg)


def write_run_manifest(run_dir: str, cfg_resolved: Dict[str, Any], artifacts: ArtifactManager) -> str:
    payload: Dict[str, Any] = {
        "task_name": artifacts.task_name,
        "method_name": artifacts.method_name,
        "seed": artifacts.seed,
        "run_name": artifacts.run_name,
        "wandb_group": artifacts.wandb_group,
        "wandb_run_name": artifacts.wandb_run_name,
        "paths": {
            "root_dir": artifacts.paths.root_dir,
            "run_dir": artifacts.paths.run_dir,
            "checkpoint_dir": artifacts.paths.checkpoint_dir,
            "eval_root_dir": artifacts.paths.eval_root_dir,
            "eval_dir": artifacts.paths.eval_dir,
            "video_dir": artifacts.paths.video_dir,
            "results_root_dir": artifacts.paths.results_root_dir,
            "replay_dir": artifacts.paths.replay_dir,
        },
        "resolved_config": cfg_resolved,
    }
    out_path = os.path.join(run_dir, "run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path
