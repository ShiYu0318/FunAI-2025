import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD
import random

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree

BASE_DIR = os.path.dirname(__file__)  # 當前檔案所在的資料夾
MODEL_AIM_PATH = os.path.join(BASE_DIR, "aim_2.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "chase_2.zip")

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
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = None
        self.target_y = None

        self.birth_positions = 0
        self.range_threshold = 150  # 設定視野半徑

        self.record_action = []
        self.last_angle= 0
        self.last_action=0

    # def detect_nearby_walls(self, player_x, player_y, walls_info):
    #     nearby_walls = []
    #     for wall in walls_info:
    #         wall_x = wall['x']
    #         wall_y = wall['y']
    #         distance = math.sqrt((wall_x - player_x) ** 2 + (wall_y - player_y) ** 2)
    #         if distance <= self.range_threshold:
    #             nearby_walls.append((wall_x, wall_y))
    #     return nearby_walls

    # def generate_new_target(self, player_x, player_y):
    #     # 隨機產生新目標位置，確保與目前位置距離大於 range_threshold
    #     while True:
    #         target_x = player_x + random.randint(-300, 300)
    #         target_y = player_y + random.randint(-300, 300)
    #         distance = math.sqrt((target_x - player_x) ** 2 + (target_y - player_y) ** 2)
    #         if distance > self.range_threshold:  # 確保新目標遠離目前位置
    #             return target_x, target_y
    
    # def get_closest_enemy(self, scene_info: dict) -> Optional[dict]:
    #     """
    #     遍歷 scene_info["competitor_info"]，選出與我方坦克距離最短的敵人。
    #     如果全部敵人已死亡或列表為空，返回 None。
    #     """
    #     competitors = scene_info.get("competitor_info", [])
    #     closest_enemy = None
    #     min_distance = float("inf")

    #     """
    #     player_info = scene_info.get(self.player, {})
    #     player_x = player_info.get("x", 0)
    #     player_y = player_info.get("y", 0)
    #     """
    #     player_x = scene_info['x']
    #     player_y = scene_info['y']

    #     for enemy in competitors:
    #         if(enemy["lives"] == 0): continue
    #         # 如果需要，可根據 enemy 狀態判斷是否存活，這裡假設敵方若存在數值就是存活
    #         enemy_x = enemy.get("x", 0)
    #         enemy_y = enemy.get("y", 0)
    #         distance = (((enemy_x - player_x)**2 + (enemy_y - player_y)**2)) ** 0.5
    #         if distance < min_distance:
    #             min_distance = distance
    #             closest_enemy = enemy
    #     # print(competitors)
    #     return closest_enemy
    def get_closest_target(self, scene_info: dict, target_key: str) -> Optional[dict]:
        """
        遍歷 scene_info[target_key]，選出與我方坦克距離最短的目標。
        target_key 可以是 "competitor_info" 或 "bullet_stations_info"。

        如果全部目標已死亡或列表為空，返回 None。
        """
        targets = scene_info.get(target_key, [])
        closest_target = None
        min_distance = float("inf")

        player_x = scene_info['x']
        player_y = scene_info['y']

        for target in targets:
            if target_key == "competitor_info" and target.get("lives", 1) == 0:
                continue  # 如果是敵人且已死亡，跳過

            target_x = target.get("x", 0)
            target_y = target.get("y", 0)
            distance = ((target_x - player_x)**2 + (target_y - player_y)**2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_target = target

        return closest_target


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        self._scene_info = scene_info
        
        # 比賽會對應自己的編號
        self.player = scene_info["id"]

        player_x = scene_info['x']
        player_y = scene_info['y']

        # 判斷在哪半邊以便在調整 target_x 時決定 +-
        if scene_info['used_frame'] == 0:
            self.birth_positions = player_x
            print(f"id = {self.player}, 初始位置 x = {self.birth_positions}")


        current_bullets = scene_info['power']
        current_oil = scene_info['oil']

        if current_oil < 30:
            closest_oil_stations = self.get_closest_target(scene_info, "oil_stations_info")
            self.target_x = closest_oil_stations.get("x", 0)
            self.target_y = -closest_oil_stations.get("y", 0)
        elif current_bullets == 0:
            closest_bullet_stations = self.get_closest_target(scene_info, "bullet_stations_info")
            self.target_x = closest_bullet_stations.get("x", 0)
            self.target_y = -closest_bullet_stations.get("y", 0)
        else:
            closest_enemy = self.get_closest_target(scene_info, "competitor_info")
            if closest_enemy is not None:
                self.target_x = closest_enemy.get("x", 0)
                self.target_y = -closest_enemy.get("y", 0)
                if self.birth_positions >= 500:
                    self.target_x += 200
                else:
                    self.target_x -= 200
            else:
                # 如果找不到有效的敵人，則使用預設目標（例如保持原位置或其他策略）
                print("找不到存活的敵人，採取預設策略")
                self.target_x = scene_info[self.player].get("x", 0)
                self.target_y = -scene_info[self.player].get("y", 0)



        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"
        
        distance = ((self.target_x - player_x)**2 + (self.target_y - player_y)**2) ** 0.5

        if distance >= 60 and abs(-self.target_y - player_y) > 7:
            obs = self._get_obs_chase()

            print("Chase obs: " + str(obs))

            action, _ = self.model_chase.predict(obs, deterministic=True)
            
            if(action != self.last_action and (action==3 or action ==4) and (self.last_action==3 or self.last_action==4)):
                action=self.last_action
            self.last_action=action
            command = COMMAND_CHASE[action]
        else:
            command = [SHOOT]

        # cnt = 0
        # for act in self.record_action:
        #     if(act == ['SHOOT'] or act == ['FORWARD'] or act == ['BACKWARD_CMD']):
        #         cnt += 1
        # if(cnt < 5): command = [FORWARD_CMD]

        # print(f"是否正在自轉：{cnt}")
        print(f"dis:{distance}")
        print(f"差距：{abs(-self.target_y - player_y)}")
        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}\n")
        self.time += 1
        
        # self.record_action.append(command)
        # if(len(self.record_action) > 90):
        #     self.record_action.pop(0)

        return command


    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

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
        print("Aim angle: " + str(angle_to_target))
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

    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))
