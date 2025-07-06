import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD

WIDTH = 1000  # pixel
HEIGHT = 600  # pixel
TANK_SPEED = 8  # pixel
CELL_PIXEL_SIZE = 50  # pixel
DEGREES_PER_SEGMENT = 45  # degree


BASE_DIR = os.path.dirname(__file__)  # 當前檔案所在的資料夾
MODEL_AIM_PATH = os.path.join(BASE_DIR, "aim_better.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "chase_1_best.zip")
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
# MODEL_DIR = os.path.join(BASE_DIR, "model")
# # 下面模型要改成自己的，記得拉進對的資料夾改名字
# MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim_better.zip")
# MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase_1_best.zip")

COMMAND_AIM = [
    ["NONE"],
    [AIM_LEFT_CMD],
    [AIM_RIGHT_CMD],
    [SHOOT],   
]

COMMAND_CHASE = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [TURN_LEFT_CMD],
    [TURN_RIGHT_CMD],
]

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param side A string like "1P" or "2P" indicates which player the `MLPlay` is for.
        """
        self.side = ai_name
        # print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = 500
        self.target_y = None
        self.target_player = None

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information.
        模型切換邏輯依據資源狀態與與目標距離來決定：
          - 資源不足時（power < 3 或 oil < 20），使用追逐模型（移動，取得補給）。
          - 否則，若與目標距離小則使用瞄準模型進行射擊；若距離大則使用追逐模型移動接近目標。
        """
        self.player = scene_info["id"]
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        self.target_x = scene_info['competitor_info'][0]['x']
        self.target_y = -scene_info['competitor_info'][0]['y']
        self.target_player = scene_info['competitor_info'][0]['id']
        # 每次更新都重新設定目標（或您也可根據其他邏輯決定目標）
        for i in range(3):
            if scene_info['competitor_info'][i]['lives'] == 0:
                self.target_x = scene_info['competitor_info'][(i+1)% 3]['x']
                self.target_y = -scene_info['competitor_info'][(i+1)% 3]['y']
                self.target_player = scene_info['competitor_info'][(i+1)% 3]['id']
            else:
                continue


    # #    print("對手座標：x {}, y {}".format(scene_info['competitor_info'][0]['x'], scene_info['competitor_info'][0]['y']))
        # # print("所有牆的座標：{}".format(scene_info['walls_info']    ))

        if self.target_x is None or self.target_y is None:
            # print("No valid target available.")
            return "RESET"

        # 從 scene_info 取出資源數據（假設 scene_info 中有 power 與 oil 的欄位）
        power = scene_info.get("power", 10)
        oil = scene_info.get("oil", 100)

        # 計算玩家與目標的距離
        distance = self._get_distance(scene_info, self.target_x, self.target_y)

        # 根據資源與距離決定使用哪個模型：
        # 1. 資源不足，始終使用追逐模型
        # 2. 資源充足時，若距離較近（例如 < 100 像素）使用瞄準模型，否則使用追逐模型
        if False and power < 3 or oil < 20:
            obs = self._get_obs_chase()
            action, reward = self.model_chase.predict(obs, deterministic=True)
            # print(reward)
            command = COMMAND_CHASE[action]
            # print("Low resource: using chase model")
        elif distance < 100:
            obs = self._get_obs_aim()
            action, reward = self.model_aim.predict(obs, deterministic=True)
            
            # print(reward)
            command = COMMAND_AIM[action]
            # print("Target is close: using aim model")
        else:
            obs = self._get_obs_chase()
            action, reward = self.model_chase.predict(obs, deterministic=True)
            # print(reward)
            command = COMMAND_CHASE[action]
            # print("Target is far: using chase model")
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        # print(f"Reward: {self.get_reward(obs,action)}")
        # print(f"Obs: {obs}")
        # print(f"Pos is: {player_x}, {player_y}")
        # print(f"Target is : ({self.target_x}, {self.target_y})\n##########Target Play: {self.target_player}")
        # print(f"Distance to target: {distance:.2f}")
        # print(f"Predicted action: {command}")
        self.time += 1
        return command

    
    def get_reward(self, obs: dict, action: int) -> float:
        angle_reward: float = self.cal_angle_reward(obs, action)
        forward_reward: float = self.cal_forward_reward(obs, action)

        total_reward: float = angle_reward + forward_reward

        if self.player == "1P":
            #  print("action: " + str(action))
            #  print("angle_reward: " + str(angle_reward))
            #  print("forward_reward: " + str(forward_reward))
            #  print("total_reward: " + str(total_reward))
            #  print("\n")
            #  print("\n")
            pass
        return total_reward

    # 在不同水平線上時
    def cal_angle_reward(self, obs: dict, action: int) -> float:

        angle_reward: float = 0.0

        # the car angle is point at the right side of the target
        if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 3: # TURN_LEFT_CMD
            angle_reward = 0.8
        elif obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]]:   # TURN_RIGHT_CMD
            angle_reward = -0.8

        # the car angle is point at the left side of the target
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 4:   # TURN_RIGHT_CMD
            angle_reward = 0.8
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]]:   # TURN_LEFT_CMD
            angle_reward = -0.8
        elif obs[0]==obs[1]:
            angle_reward = 0.4
        else:
            angle_reward = -0.4
        
        return angle_reward
    
    def cal_forward_reward(self, obs: dict, action: int) -> float:
        forward_reward: float = 0.0

        if obs[0] == obs[1] and action == 1:    # FORWARD_CMD
            forward_reward = 1
        elif obs[0] == obs[1]: # and action == 2:  # BACKWARD_CMD
            forward_reward = -0.8
        # 轉四下變對面
        elif obs[0] == (obs[1] + 4) % 8 and action == 2:    # BACKWARD_CMD
            forward_reward = 0.8
        elif obs[0] == (obs[1] + 4) % 8 and action == 1:    # FORWARD_CMD
            forward_reward = -0.8
        elif (obs[0] == obs[1] or obs[0] == (obs[1] + 4) % 8) and (action != 1 and action != 2):
            forward_reward = -0.8
        return forward_reward
    
    def reset(self):
        """
        Reset the status.
        """
        # print(f"Resetting Game {self.side}")

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index: int = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        # print("Chase obs: " + str(obs))
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        # print("Aim angle: " + str(angle_to_target))
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        return obs

    def _get_obs_chase(self) -> np.ndarray:
        return self.get_obs_chase(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _get_obs_aim(self) -> np.ndarray:
        return self.get_obs_aim(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _get_distance(self, scene_info: dict, target_x: int, target_y: int) -> float:
        """
        計算玩家與目標之間的歐式距離。
        注意：根據 get_obs_chase 中，玩家的 y 座標為 -scene_info[\"y\"]。
        """
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        dx = target_x - player_x
        dy = target_y - player_y
        return math.hypot(dx, dy)

    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360
        segment_center = (angle + DEGREES_PER_SEGMENT / 2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))

# import random
# from typing import Optional
# import pygame
# import os
# import numpy as np
# from stable_baselines3 import PPO
# from mlgame.utils.enum import get_ai_name
# import math
# from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD
# import random

# WIDTH = 1000 # pixel
# HEIGHT = 600 # pixel
# TANK_SPEED = 8 # pixel
# CELL_PIXEL_SIZE = 50 # pixel
# DEGREES_PER_SEGMENT = 45 # degree

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
# MODEL_DIR = os.path.join(BASE_DIR, "model")
# # 下面模型要改成自己的，記得拉進對的資料夾改名字
# MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim.zip")
# MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase.zip")

# COMMAND_AIM = [
#     ["NONE"],
#     [AIM_LEFT_CMD],
#     [AIM_RIGHT_CMD],
#     [SHOOT],   
# ]

# COMMAND_CHASE = [
#     ["NONE"],
#     [FORWARD_CMD],
#     [BACKWARD_CMD],
#     [TURN_LEFT_CMD],
#     [TURN_RIGHT_CMD],
# ]

# class MLPlay:
#     def __init__(self, ai_name, *args, **kwargs):
#         """
#         Constructor

#         @param side A string like "1P" or "2P" indicates which player the `MLPlay` is for.
#         """
#         self.side = ai_name
        # # print(f"Initial Game {ai_name} ML script")
#         self.time = 0
#         self.player: str = "1P"

#         # Load the trained models
#         self.model_aim = PPO.load(MODEL_AIM_PATH)
#         self.model_chase = PPO.load(MODEL_CHASE_PATH)

#         self.target_x = None
#         self.target_y = None


#     def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
#         """
#         Generate the command according to the received scene information
#         """
#         if scene_info["status"] != "GAME_ALIVE":
#             return "RESET"
#         self._scene_info = scene_info

#         self.target_x = random.randint(0, WIDTH)
#         self.target_y = random.randint(0, HEIGHT)

#         if self.target_x is None or self.target_y is None:
            # # print("No valid target available.")
#             return "RESET"

#         # Randomly switch between model_aim and model_chase
#         if random.choice([True, False]):
#             obs = self._get_obs_aim()
#             action, _ = self.model_aim.predict(obs, deterministic=True)
#             command = COMMAND_AIM[action]
#         else:
#             obs = self._get_obs_chase()
#             action, _ = self.model_chase.predict(obs, deterministic=True)
#             command = COMMAND_CHASE[action]

        # # print(f"Target is : ({self.target_x, self.target_y})")
        # # print(f"Predicted action: {command}")
#         self.time += 1
#         return command


#     def reset(self):
#         """
#         Reset the status
#         """
        # # print(f"Resetting Game {self.side}")

#     def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
#         player_x = scene_info.get("x", 0)
#         player_y = -scene_info.get("y", 0)
#         tank_angle = scene_info.get("angle", 0) + 180
#         tank_angle_index: int = self._angle_to_index(tank_angle)
#         dx = target_x - player_x
#         dy = target_y - player_y
#         angle_to_target = math.degrees(math.atan2(dy, dx))
#         angle_to_target_index: int = self._angle_to_index(angle_to_target)
#         obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        # # print("Chase obs: " + str(obs))
#         return obs

#     def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
#         player_x = scene_info.get("x", 0)
#         player_y = -scene_info.get("y", 0)
#         gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
#         gun_angle_index: int = self._angle_to_index(gun_angle)
#         dx = target_x - player_x
#         dy = target_y - player_y 
#         angle_to_target = math.degrees(math.atan2(dy, dx))
#         angle_to_target_index: int = self._angle_to_index(angle_to_target)
        # # print("Aim angle: " + str(angle_to_target))
#         obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
#         return obs

#     def _get_obs_chase(self) -> np.ndarray:
#         return self.get_obs_chase(
#             self.player,
#             self.target_x,
#             self.target_y,
#             self._scene_info,
#         )

#     def _get_obs_aim(self) -> np.ndarray:
#         return self.get_obs_aim(
#             self.player,
#             self.target_x,
#             self.target_y,
#             self._scene_info,
#         )

#     def _angle_to_index(self, angle: float) -> int:
#         angle = (angle + 360) % 360

#         segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
#         return int(segment_center % (360 // DEGREES_PER_SEGMENT))