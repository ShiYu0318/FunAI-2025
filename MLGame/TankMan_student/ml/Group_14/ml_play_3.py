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
MODEL_AIM_PATH = os.path.join(BASE_DIR, "model_3a.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "model_3b.zip")

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

toggle = True

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


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        global toggle
        command = []
        obs = None

        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        # 新增這一行
        self.player = scene_info["id"]
        # 可將 target 先設成（100,100）為遊戲的左上角，確定模型會朝向左上角移動，接著再將 target 改成對手的座標
        # for target in scene_info['competitor_info'] :
        #     tarid = target['id']
        #     if tarid == '4P':
        #         self.target_x = target['x']#random.randint(0, WIDTH)#1000#["teammate"] #
        #         self.target_y = target['y']#random.randint(0, HEIGHT)#600 
        


        closest_competitor = None
        min_distance = float('inf')
        for competitor in scene_info['competitor_info']:
            distance = math.hypot(competitor['x'] - self._scene_info['x'], competitor['y'] - self._scene_info['y'])
            if distance < min_distance:
                min_distance = distance
                closest_competitor = competitor
                self.target_x = competitor['x'] #random.randint(0, WIDTH)#1000#["teammate"] #
                self.target_y = competitor['y']

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        distance = math.hypot(self.target_x - scene_info["x"], self.target_y - scene_info["y"])

        if 100 < scene_info["y"] < 500 and 100 < scene_info["x"] < 900 :
            if toggle == 1 :
                if distance < 280 : #random.choice([True, False]):
                    obs = self._get_obs_aim()
                    if 0 < obs[1] < 4 :
                        command = [AIM_LEFT_CMD]
                    elif obs [1] >= 4 :
                        command = [AIM_RIGHT_CMD]
                    elif obs [1] == 0 :
                        command = random.choices([["SHOOT"], ["NONE"]], weights=[2, 8])[0]

                    # ["NONE"],
                    # [AIM_LEFT_CMD],
                    # [AIM_RIGHT_CMD],

                    action, _ = self.model_aim.predict(obs, deterministic=True)
                    command = random.choices([command,COMMAND_AIM[action]], weights=[8, 2])[0]
                    #command = COMMAND_AIM[action]

                #else :
                    
                    # if 0 < obs[1] < 4 :
                    #     command = [TURN_LEFT_CMD]
                    # elif obs [1] >= 4 :
                    #     command = [TURN_RIGHT_CMD]
                    # elif obs [1] == 0 :
                    #     command = ["NONE"]
                        #command = random.choices([["FORWARD_CMD"], ["NONE"], ["BACKWARD_CMD"]], weights=[7, 3, 5], k=3)[0]
                        # ["NONE"],
                        # [FORWARD_CMD],
                        # [BACKWARD_CMD],
                        # [TURN_LEFT_CMD],
                        # [TURN_RIGHT_CMD],
                    
                    #command = random.choices([command,COMMAND_CHASE[action]], weights=[8, 2])[0]
                    # obs = self._get_obs_chase()
                    # action, _ = self.model_chase.predict(obs, deterministic=True)
                    # command = COMMAND_CHASE[action]
            else :
                obs = self._get_obs_chase()
                if 0 < obs[1] < 4 :
                    command = [TURN_LEFT_CMD]
                elif obs [1] >= 4 :
                    command = [TURN_RIGHT_CMD]
                elif obs [1] == 0 :
                    command = random.choices([["FORWARD_CMD"], ["NONE"], ["BACKWARD_CMD"]], weights=[8, 1, 2], k=3)[0]
                    # ["NONE"],
                    # [FORWARD_CMD],
                    # [BACKWARD_CMD],
                    # [TURN_LEFT_CMD],
                    # [TURN_RIGHT_CMD],

                    action, _ = self.model_chase.predict(obs, deterministic=True)
                    command = random.choices([command,COMMAND_CHASE[action]], weights=[8, 2])[0]
                #command = COMMAND_CHASE[action]
        elif 100 > scene_info["y"]  or scene_info["y"] > 500 or 100 > scene_info["x"] or scene_info["x"] > 900  :
            error = 300
            if toggle == 1 :
                if (100 > scene_info["y"] and 500 > scene_info["x"]) or (300 > scene_info["y"] and 100 > scene_info["x"]):
                    self.target_x = 900 - error
                    self.target_y = 100 + error
                    obs = self._get_obs_chase()
                    if 0 < obs[1] < 4 :
                        command = [TURN_LEFT_CMD]
                    elif obs [1] >= 4 :
                        command = [TURN_RIGHT_CMD]
                    elif obs [1] == 0 :
                        command = [FORWARD_CMD]
                elif (500 < scene_info["y"] and 500 > scene_info['x']) or (300 < scene_info["y"] and 100 > scene_info['x']):
                    self.target_x = 100 + error
                    self.target_y = 100 + error
                    obs = self._get_obs_chase()
                    if 0 < obs[1] < 4 :
                        command = [TURN_LEFT_CMD]
                    elif obs [1] >= 4 :
                        command = [TURN_RIGHT_CMD]
                    elif obs [1] == 0 :
                        command = [FORWARD_CMD]
                elif (500 < scene_info["y"] and 500 < scene_info['x']) or (300 < scene_info["y"] and 900 < scene_info['x']):
                    self.target_x = 100 + error
                    self.target_y = 500 - error
                    obs = self._get_obs_chase()
                    if 0 < obs[1] < 4 :
                        command = [TURN_LEFT_CMD]
                    elif obs [1] >= 4 :
                        command = [TURN_RIGHT_CMD]
                    elif obs [1] == 0 :
                        command = [FORWARD_CMD]
                elif (300 > scene_info["y"] and 900 < scene_info['x']) or (100 > scene_info["y"] and 500 < scene_info['x']):
                    self.target_x = 900 - error
                    self.target_y = 100 + error
                    obs = self._get_obs_chase()
                    if 0 < obs[1] < 4 :
                        command = [TURN_LEFT_CMD]
                    elif obs [1] >= 4 :
                        command = [TURN_RIGHT_CMD]
                    elif obs [1] == 0 :
                        command = [FORWARD_CMD]
            else :
                if (100 > scene_info["y"] and 500 > scene_info["x"]) or (300 > scene_info["y"] and 100 > scene_info["x"]):
                    self.target_x = 900 - error
                    self.target_y = 100 + error
                    
                    if distance < 280 : #random.choice([True, False]):
                        obs = self._get_obs_aim()
                        if 0 < obs[1] < 4 :
                            command = [AIM_LEFT_CMD]
                        elif obs [1] >= 4 :
                            command = [AIM_RIGHT_CMD]
                        elif obs [1] == 0 :
                            command = random.choices([["SHOOT"], ["NONE"]], weights=[2, 8])[0]

                        # ["NONE"],
                        # [AIM_LEFT_CMD],
                        # [AIM_RIGHT_CMD],

                        action, _ = self.model_aim.predict(obs, deterministic=True)
                        command = random.choices([command,COMMAND_AIM[action]], weights=[8, 2])[0]
                        #command = COMMAND_AIM[action]
                elif (500 < scene_info["y"] and 500 > scene_info['x']) or (300 < scene_info["y"] and 100 > scene_info['x']):
                    self.target_x = 100 + error
                    self.target_y = 100 + error
                    
                    if distance < 280 : #random.choice([True, False]):
                        obs = self._get_obs_aim()
                        if 0 < obs[1] < 4 :
                            command = [AIM_LEFT_CMD]
                        elif obs [1] >= 4 :
                            command = [AIM_RIGHT_CMD]
                        elif obs [1] == 0 :
                            command = random.choices([["SHOOT"], ["NONE"]], weights=[2, 8])[0]

                        # ["NONE"],
                        # [AIM_LEFT_CMD],
                        # [AIM_RIGHT_CMD],

                        action, _ = self.model_aim.predict(obs, deterministic=True)
                        command = random.choices([command,COMMAND_AIM[action]], weights=[8, 2])[0]
                        #command = COMMAND_AIM[action]
                elif (500 < scene_info["y"] and 500 < scene_info['x']) or (300 < scene_info["y"] and 900 < scene_info['x']):
                    self.target_x = 100 + error
                    self.target_y = 500 - error
                    
                    if distance < 280 : #random.choice([True, False]):
                        obs = self._get_obs_aim()
                        if 0 < obs[1] < 4 :
                            command = [AIM_LEFT_CMD]
                        elif obs [1] >= 4 :
                            command = [AIM_RIGHT_CMD]
                        elif obs [1] == 0 :
                            command = random.choices([["SHOOT"], ["NONE"]], weights=[2, 8])[0]

                        # ["NONE"],
                        # [AIM_LEFT_CMD],
                        # [AIM_RIGHT_CMD],

                        action, _ = self.model_aim.predict(obs, deterministic=True)
                        command = random.choices([command,COMMAND_AIM[action]], weights=[8, 2])[0]
                        #command = COMMAND_AIM[action]
                elif (300 > scene_info["y"] and 900 < scene_info['x']) or (100 > scene_info["y"] and 500 < scene_info['x']):
                    self.target_x = 900 - error
                    self.target_y = 100 + error
                    
                    if distance < 280 : #random.choice([True, False]):
                        obs = self._get_obs_aim()
                        if 0 < obs[1] < 4 :
                            command = [AIM_LEFT_CMD]
                        elif obs [1] >= 4 :
                            command = [AIM_RIGHT_CMD]
                        elif obs [1] == 0 :
                            command = random.choices([["SHOOT"], ["NONE"]], weights=[2, 8])[0]

                        # ["NONE"],
                        # [AIM_LEFT_CMD],
                        # [AIM_RIGHT_CMD],

                        action, _ = self.model_aim.predict(obs, deterministic=True)
                        command = random.choices([command,COMMAND_AIM[action]], weights=[8, 2])[0]
                        #command = COMMAND_AIM[action]
        
        # if not command:  # 如果 command 是空的，這會評估為 True
        #     command = [TURN_RIGHT_CMD]

        if command == [FORWARD_CMD] :
            command = [BACKWARD_CMD]
        elif command == [BACKWARD_CMD] :
            command = [FORWARD_CMD]

        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}")
        print(obs)
        self.time += 1
        print(self.time)
        toggle = not toggle
        print(toggle)
        print(scene_info.get("gun_angle", 0))
        print(scene_info.get("angle", 0))
        print(scene_info['x'])
        print(scene_info['y'])
        # print("--------")
        # print(scene_info["walls_info"])
        # print("--------")

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
        angle_to_target = math.degrees(math.atan2(dy, dx)) * -1
        print("tank target angel :" + str(angle_to_target))
        if target_x > player_x and target_y < player_y: #1 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - tank_angle
        elif target_x < player_x and target_y < player_y :#2 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - tank_angle
        elif target_x > player_x and target_y > player_y :#4 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - tank_angle
        elif target_x < player_x and target_y > player_y :#3 phrase
            angle_to_target = 180 + angle_to_target
            angle_to_target = angle_to_target - tank_angle

        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print(angle_to_target)
        print(angle_to_target_index)
        print(tank_angle)
        print(tank_angle_index)
        print("Chase obs: " + str(angle_to_target))
        obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)

        gun_angle = scene_info.get("gun_angle", 0) % 360# + scene_info.get("angle", 0) +20
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        angle_to_target = math.degrees(math.atan2(dy, dx)) * -1
        
        if target_x > player_x and target_y < player_y: #1 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - gun_angle
        elif target_x < player_x and target_y < player_y :#2 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - gun_angle
        elif target_x > player_x and target_y > player_y :#4 phrase
            angle_to_target = 180  + angle_to_target
            angle_to_target = angle_to_target - gun_angle
        elif target_x < player_x and target_y > player_y :#3 phrase
            angle_to_target = 180 + angle_to_target
            angle_to_target = angle_to_target - gun_angle
        
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print(angle_to_target)
        print(angle_to_target_index)
        print(gun_angle)
        print(gun_angle_index)
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
#python -m mlgame -f 120 -i ml/ml_play_model.py -i ml/ml_play_model.py -i ml/ml_play_model.py -i ml/ml_play_model.py -i ml/ml_play_model.py -i ml/ml_play_model.py . --green_team_num 3 --blue_team_num 3 --is_manual 1 --frame_limit 400
#python -m mlgame -f 120 -i ml/ml_play_model.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py -i ml/ml_play_manual.py . --green_team_num 3 --blue_team_num 3 --is_manual 1 --frame_limit 400
