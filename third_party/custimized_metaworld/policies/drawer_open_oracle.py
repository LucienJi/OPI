from __future__ import annotations

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import assert_fully_parsed, move

from ..oracle_base import BaseOraclePolicy


class SawyerDrawerOpenOraclePolicy(BaseOraclePolicy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "drwr_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        desired_pos = self._desired_pos(o_d)
        grab_effort = self._grab_effort(o_d)
        desired_pos = self._maybe_add_noise(
            desired_pos, o_d["hand_pos"], self.current_stage_id
        )
        action["delta_pos"] = self._clip_delta_pos(
            move(o_d["hand_pos"], desired_pos, p=self.current_p)
        )
        action["grab_effort"] = grab_effort

        return action.array

    def _desired_pos(
        self, o_d: dict[str, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        self.current_stage_id = -1
        self.current_p = 4.0

        pos_curr = o_d["hand_pos"]
        pos_drwr = o_d["drwr_pos"] + np.array([0.0, 0.0, -0.02])

        # align end effector's Z axis with drawer handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
            self.current_stage_id = 0
            self.current_p = 4.0
            return pos_drwr + np.array([0.0, 0.0, 0.3])
        # drop down to touch drawer handle
        elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
            self.current_stage_id = 1
            self.current_p = 4.0
            return pos_drwr
        # push toward a point just behind the drawer handle
        # also increase p value to apply more force
        else:
            self.current_stage_id = 2
            self.current_p = 50.0
            return pos_drwr + np.array([0.0, -0.06, 0.0])

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        _ = o_d
        return -1.0

