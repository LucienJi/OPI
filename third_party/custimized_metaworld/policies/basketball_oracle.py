from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import assert_fully_parsed, move

from ..oracle_base import BaseOraclePolicy


class SawyerBasketballOraclePolicy(BaseOraclePolicy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "ball_pos": obs[4:7],
            "hoop_x": obs[-3],
            "hoop_yz": obs[-2:],
            "unused_info": obs[7:-3],
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
            move(o_d["hand_pos"], to_xyz=desired_pos, p=25.0)
        )
        action["grab_effort"] = grab_effort
        return action.array

    def _desired_pos(self, o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        self.current_stage_id = -1
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"] + np.array([0.0, 0.0, 0.01])
        # X is given by hoop_pos
        # Y varies between .85 and .9, so we take avg
        # Z is constant at .35
        pos_hoop = np.array([o_d["hoop_x"], 0.875, 0.35])

        if np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > 0.04:
            self.current_stage_id = 0
            return pos_ball + np.array([0.0, 0.0, 0.3])
        elif abs(pos_curr[2] - pos_ball[2]) > 0.025:
            self.current_stage_id = 0
            return pos_ball
        elif abs(pos_ball[2] - pos_hoop[2]) > 0.025:
            self.current_stage_id = 1
            return np.array([pos_curr[0], pos_curr[1], pos_hoop[2]])
        else:
            self.current_stage_id = 1
            return pos_hoop

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_ball = o_d["ball_pos"]
        if (
            np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > 0.04
            or abs(pos_curr[2] - pos_ball[2]) > 0.06 # 0.15
        ):
            return -1.0
        else:
            return 0.6

