import sys
from os import path
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

import math
import random
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from mlgame.utils.enum import get_ai_name

from src.env import SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD
from .base_env import TankManBaseEnv

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree
COMMAND = [
    ["NONE"],
    [AIM_LEFT_CMD],
    [AIM_RIGHT_CMD],
    [SHOOT],
]

class ResupplyEnv(TankManBaseEnv):
    def __init__(
        self,
        green_team_num: int,
        blue_team_num: int,
        frame_limit: int,
        player: Optional[str] = None,
        supply_type: Optional[str] = None,
        randomize: Optional[bool] = False,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(green_team_num, blue_team_num, frame_limit, sound, render_mode)

        self.player_num = green_team_num + blue_team_num
        self.randomize = randomize

        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))
            self.supply_type = np.random.choice(["oil_stations", "bullet_stations"])
        else:
            assert player is not None and supply_type is not None
            assert player in [
                get_ai_name(i) for i in range(self.player_num)
            ], f"{player} is not a valid player id"
            assert supply_type in [
                "oil_stations",
                "bullet_stations",
            ], f"{supply_type} is not a valid supply type"

            self.player = player
            self.supply_type = supply_type

        self._total_angle_segment: int = 360 // DEGREES_PER_SEGMENT
        if type(self._total_angle_segment) is not int:
            raise ValueError("The total angle segment should be an integer, please modify the DEGREES_PER_SEGMENT value.")
        
        # Initialize target position
        self.target_x = random.randint(CELL_PIXEL_SIZE, WIDTH - 2 * CELL_PIXEL_SIZE)
        self.target_y = random.randint(CELL_PIXEL_SIZE, HEIGHT - 2 * CELL_PIXEL_SIZE)

        # gun_angle, angle_to_target
        self._observation_space = Box(low=0, high=self._total_angle_segment - 1, shape=(2,), dtype=np.float32)

        self._action_space = Discrete(len(COMMAND))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))

        # Randomize target position
        self.target_x = random.randint(0, WIDTH)
        self.target_y = random.randint(0, HEIGHT)

        return super().reset(seed=seed, options=options)

    def update_target_position(self):
        delta_x = random.choice([-TANK_SPEED, 0, TANK_SPEED])
        delta_y = random.choice([-TANK_SPEED, 0, TANK_SPEED])

        self.target_x = max(CELL_PIXEL_SIZE, min(WIDTH - 2 * CELL_PIXEL_SIZE, self.target_x + delta_x))
        self.target_y = max(CELL_PIXEL_SIZE, min(HEIGHT - 2 * CELL_PIXEL_SIZE, self.target_y + delta_y))

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    def get_obs(
        self,
        player: str,
        target_x: int,
        target_y: int,
        scene_info: dict,
    ) -> np.ndarray:
        # Get player's data
        player_data = scene_info[player]
        player_x = player_data.get("x", 0)
        player_y = player_data.get("y", 0)
        gun_angle = player_data.get("gun_angle", 0)
        gun_angle_index: int = self._angle_to_index(gun_angle)

        # Calculate angle to the target
        dx = target_x - player_x
        dy = target_y - player_y  # Adjust for screen coordinate system
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)

        # Return gun_angle and normalized angle_to_target
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        if self.player == "1P":
            print("player: " + str(self.player))
            print("gun_angle: " + str(gun_angle))
            print("gun_angle_index: " + str(gun_angle_index))
            print("player_x: " + str(player_x))
            print("player_y: " + str(player_y))
            print("target_x: " + str(target_x))
            print("target_y: " + str(target_y))
            print("angle_to_target: " + str(angle_to_target))
            print("angle_to_target_index: " + str(angle_to_target_index))
            print("obs: " + str(obs))
            print("\n")
        return obs

    def _get_obs(self) -> np.ndarray:
        self.update_target_position()
        return self.get_obs(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def get_reward(self, obs: dict, action: int) -> float:
        angle_reward: float = self.cal_angle_reward(obs, action)
        shoot_reward: float = self.cal_shoot_reward(obs, action)

        total_reward: float = angle_reward + shoot_reward

        if self.player == "1P":
            print("action: " + str(action))
            print("angle_reward: " + str(angle_reward))
            print("shoot_reward: " + str(shoot_reward))
            print("total_reward: " + str(total_reward))
            print("\n")
            print("\n")
        return total_reward

    def cal_angle_reward(self, obs: dict, action: int) -> float:

        angle_reward: float = 0.0

        # the gun angle is point at the right side of the target
        if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 1: # AIM_LEFT_CMD
            angle_reward = 0.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 2:   # AIM_RIGHT_CMD
            angle_reward = 0.0

        # the gun angle is point at the left side of the target
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 2:   # AIM_RIGHT_CMD
            angle_reward = 0.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 1:   # AIM_LEFT_CMD
            angle_reward = 0.0

        # the gun angle is point at the opposite side of the target
        elif obs[0] == (obs[1] + 4) % 8 and (action ==1 or action == 2):
            angle_reward = 0.0
        
        return angle_reward
    
    def cal_shoot_reward(self, obs: dict, action: int) -> float:
        shoot_reward: float = 0.0

        if obs[0] == obs[1] and action == 3:    # SHOOT
            shoot_reward = 0.0
        elif action == 3:
            shoot_reward = 0.0
        elif obs[0] == obs[1]:
            shoot_reward = 0.0
        return shoot_reward
    
    def _get_reward(self, obs: dict, action: int) -> float:
        # Get observation to retrieve the precomputed angle_to_target
        return self.get_reward(obs, action)

    def _is_done(self) -> bool:
        return (
            self._scene_info[self.player]["status"] != "GAME_ALIVE"
            or self._scene_info[self.player]["oil"] == 0
        )

    def _get_commands(self, action: int) -> dict:
        commands = {get_ai_name(id): ["NONE"] for id in range(self.player_num)}
        commands[self.player] = COMMAND[action]
        return commands
    
    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))

if __name__ == "__main__":
    env = ResupplyEnv(3, 3, 100, randomize=True, render_mode="human")
    for _ in range(10):
        env.reset()
        for _ in range(1000):
            obs, reward, terminate, _, _ = env.step(env.action_space.sample())  # type: ignore
            print("Observation:", obs)  # 包含 [gun_angle, angle_to_target]
            print("Reward:", reward)    # Reward 基於對準程度
            env.render()
            if terminate:
                break
    env.close()
