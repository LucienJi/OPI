from __future__ import annotations

import importlib
from typing import Literal, Tuple, Type

from .oracle_base import BaseOraclePolicy

_POLICY_MODULES: dict[str, tuple[str, str]] = {
    "assembly": ("policies.assembly_oracle", "SawyerAssemblyOraclePolicy"),
    "basketball": ("policies.basketball_oracle", "SawyerBasketballOraclePolicy"),
    "disassemble": ("policies.disassemble_oracle", "SawyerDisassembleOraclePolicy"),
    "drawer-open": ("policies.drawer_open_oracle", "SawyerDrawerOpenOraclePolicy"),
    "hammer": ("policies.hammer_oracle", "SawyerHammerOraclePolicy"),
    "lever-pull": ("policies.lever_pull_oracle", "SawyerLeverPullOraclePolicy"),
    "peg-insert-side": (
        "policies.peg_insert_side_oracle",
        "SawyerPegInsertionSideOraclePolicy",
    ),
    "peg-unplug-side": (
        "policies.peg_unplug_side_oracle",
        "SawyerPegUnplugSideOraclePolicy",
    ),
    "pick-place": ("policies.pick_place_oracle", "SawyerPickPlaceOraclePolicy"),
    "push-wall": ("policies.push_wall_oracle", "SawyerPushWallOraclePolicy"),
    "stick-pull": ("policies.stick_pull_oracle", "SawyerStickPullOraclePolicy"),
}


def _load_policy_class(env_name: str) -> Type[BaseOraclePolicy]:
    if env_name not in _POLICY_MODULES:
        raise ValueError(f"Unknown env: {env_name}")

    module_path, class_name = _POLICY_MODULES[env_name]
    try:
        module = importlib.import_module(f".{module_path}", package=__package__)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Policy for env '{env_name}' is not implemented yet: "
            f"{module_path}.{class_name}"
        ) from exc


def get_oracle_pair(
    env_name: str,
    mode: Literal["spatial", "condition"] = "spatial",
    noise_scales: Tuple[float, float] = (0.1, 1.0),
    grid_size: float = 0.1,
    seed: int = 0,
) -> list[BaseOraclePolicy]:
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

