from __future__ import annotations

import importlib
from typing import Literal, Tuple, Type

from .policies_th.oracle_base import TorchBaseOraclePolicy

_POLICY_MODULES: dict[str, tuple[str, str]] = {
    "assembly": ("policies_th.assembly_oracle", "SawyerAssemblyOraclePolicy"),
    "basketball": ("policies_th.basketball_oracle", "SawyerBasketballOraclePolicy"),
    "disassemble": ("policies_th.disassemble_oracle", "SawyerDisassembleOraclePolicy"),
    "drawer-open": ("policies_th.drawer_open_oracle", "SawyerDrawerOpenOraclePolicy"),
    "hammer": ("policies_th.hammer_oracle", "SawyerHammerOraclePolicy"),
    "lever-pull": ("policies_th.lever_pull_oracle", "SawyerLeverPullOraclePolicy"),
    "peg-insert-side": (
        "policies_th.peg_insert_side_oracle",
        "SawyerPegInsertionSideOraclePolicy",
    ),
    "peg-unplug-side": (
        "policies_th.peg_unplug_side_oracle",
        "SawyerPegUnplugSideOraclePolicy",
    ),
    "pick-place": ("policies_th.pick_place_oracle", "SawyerPickPlaceOraclePolicy"),
    "push-wall": ("policies_th.push_wall_oracle", "SawyerPushWallOraclePolicy"),
    "stick-pull": ("policies_th.stick_pull_oracle", "SawyerStickPullOraclePolicy"),
}


def _load_policy_class(env_name: str) -> Type[TorchBaseOraclePolicy]:
    if env_name not in _POLICY_MODULES:
        raise ValueError(f"Unknown env: {env_name}")

    module_path, class_name = _POLICY_MODULES[env_name]
    try:
        module = importlib.import_module(f".{module_path}", package=__package__)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Torch policy for env '{env_name}' is not implemented yet: "
            f"{module_path}.{class_name}"
        ) from exc


def get_oracle_pair(
    env_name: str,
    mode: Literal["spatial", "condition"] = "spatial",
    noise_scales: Tuple[float, float] = (0.1, 1.0),
    grid_size: float = 0.1,
    seed: int = 0,
) -> list[TorchBaseOraclePolicy]:
    policy_class = _load_policy_class(env_name)
    return [
        policy_class(
            mode=mode,
            variant=0,
            noise_scales=noise_scales,
            grid_size=grid_size,
            seed=seed,
        ),
        policy_class(
            mode=mode,
            variant=1,
            noise_scales=noise_scales,
            grid_size=grid_size,
            seed=seed,
        ),
    ]

