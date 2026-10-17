from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import assert_fully_parsed, move

from ..oracle_base import BaseOraclePolicy


class SawyerStickPullOraclePolicy(BaseOraclePolicy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "stick_pos": obs[4:7],
            "unused_2": obs[7:11],
            "obj_pos": obs[11:14],
            "unused_3": obs[14:-3],
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_pow": 3})

        desired_pos = self._desired_pos(o_d)
        grab_effort = self._grab_effort(o_d)
        desired_pos = self._maybe_add_noise(
            desired_pos, o_d["hand_pos"], self.current_stage_id
        )
        action["delta_pos"] = self._clip_delta_pos(
            move(o_d["hand_pos"], to_xyz=desired_pos, p=25.0)
        )
        action["grab_pow"] = grab_effort

        return action.array

    def _desired_pos(self, o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        self.current_stage_id = -1
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + np.array([-0.015, 0.0, 0.03])
        thermos_pos = o_d["obj_pos"] + np.array([-0.015, 0.0, 0.03])
        goal_pos = o_d["goal_pos"] + np.array([-0.05, 0.0, 0.0])

        if abs(stick_pos[0] - thermos_pos[0]) > 0.04:
            if np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02:
                self.current_stage_id = 0
                return stick_pos + np.array([0.0, 0.0, 0.1])
            elif abs(hand_pos[2] - stick_pos[2]) > 0.02:
                self.current_stage_id = 0
                return stick_pos
            elif abs(stick_pos[1] - thermos_pos[1]) > 0.02:
                self.current_stage_id = 0
                return np.array([stick_pos[0], thermos_pos[1], stick_pos[2]])
            elif abs(stick_pos[2] - thermos_pos[2]) > 0.02:
                self.current_stage_id = 1
                return np.array([stick_pos[0], *thermos_pos[1:]])
            else:
                self.current_stage_id = 1
                return thermos_pos
        else:
            self.current_stage_id = 1
            return goal_pos

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + np.array([-0.015, 0.0, 0.03])

        if (
            np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02
            or abs(hand_pos[2] - stick_pos[2]) > 0.06 #0.1
        ):
            return -1.0
        else:
            return +0.7

