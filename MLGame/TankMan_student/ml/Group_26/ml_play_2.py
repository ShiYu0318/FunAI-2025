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

BASE_DIR = os.path.dirname(__file__) # 上層資料夾
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
# MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_DIR = BASE_DIR
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "model_2a.zip")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "model_2b.zip")

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


        self.x = None
        self.y = None
        self.target_x = None
        self.target_y = None
        self.next_action_chase = [] #A queue store the demanded next action if needed
        self.next_action_aim = [] #A queue store the demanded next action if needed
        self.last_too_close_to_wall = False

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
         # 新增這一行
        self.player = scene_info["id"]
        

##############取得目標(坦克)的位置(透過計算所有敵方坦克和自己的距離)################
        self.x = scene_info["x"]
        self.y = scene_info["y"]

        self.target_x, self.target_y, targets_distance, target_index = self.find_min_target(self.x, self.y, "competitor_info", scene_info)
##############################################################################

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        # Randomly switch between model_aim and model_chase
        
        model_choice = False #  random.choice([True, False])

        if targets_distance[target_index] < 250:
            model_choice = True
        else:
            model_choice = False

        #判定撞牆

        #判定是否與敵方在同一直線上及同一直線上是否有友方坦克
        #print(scene_info["bullets_info"])

        #燃料快不足或彈藥不夠
        if scene_info["oil"] < 20:       #True
           self.target_x, self.target_y, targets_distance, target_index = self.find_min_target(self.x, self.y, "oil_stations_info", scene_info) 
           model_choice = False
        elif scene_info["power"] < 2: 
          self.target_x, self.target_y, targets_distance, target_index = self.find_min_target(self.x, self.y, "bullet_stations_info", scene_info)
          model_choice = False

        #self.target_x = 0
        #self.target_y = 0

        if self.next_action_chase:
            command = COMMAND_CHASE[self.next_action_chase.pop(0)]
            print(f"command: {command}")
        elif self.next_action_aim:
            command = COMMAND_AIM[self.next_action_aim.pop(0)]
            print(f"command: {command}")
        else:
            if model_choice:  #True為aim模式，False為chase模式
                obs = self._get_obs_aim()
                action, _ = self.model_aim.predict(obs, deterministic=True)
                command = COMMAND_AIM[action]
                reward = self.get_aim_reward(obs, action)

                # Check if the line of fire is clear

                if action == 3: # SHOOT
                    # Check line-of-fire
                    can_shoot = self.line_of_fire_clear(scene_info)
                    if not can_shoot:
                        # Override by doing "NONE" or adjusting aim
                        action = random.choices([3, 4], weights = [0.5,0.5], k=1)[0] #TURN_LEFT_CMD or TURN_RIGHT_CMD
                        command = COMMAND_CHASE[action]
                    self.next_action_chase.append(1)  # FORWARD_CMD                    
                print(f"Target is : ({self.target_x, self.target_y})")
                print(f"Predicted action: {command}, reward: {reward}")

            else:
                obs = self._get_obs_chase()
                action, _ = self.model_chase.predict(obs, deterministic=True)
                command = COMMAND_CHASE[action]
                reward = self.get_chase_reward(obs, action)
                # action 1 is FORWARD_CMD
                # angle in degrees = scene_info["angle"]
                # next position if we move forward:
                test_x = self.x
                test_y = self.y
                if action == 1:
                    # Convert angle to radians
                    rad_angle = math.radians(scene_info["angle"])
                    test_x -= TANK_SPEED * math.cos(rad_angle) #注意是加號還是減號
                    test_y += TANK_SPEED * math.sin(rad_angle)  # be mindful of screen y-axis
                    
                # Check for collision:
                too_close_to_wall, w_x, w_y = self.is_too_close_to_wall(test_x, test_y, scene_info["walls_info"])
                
                if too_close_to_wall:
 #                   if self.last_too_close_to_wall == False:
                    # If it’s too close, override the model’s action, e.g. turn instead:
                    action = random.choices([2, 3], weights = [0.3,0.7], k=1)[0]  # TURN_LEFT_CMD or pick any rotation #可以再改
                    command = COMMAND_CHASE[action]
                    self.next_action_chase.append(1)  # FORWARD_CMD
                    self.next_action_chase.append(1)  # FORWARD_CMD
                    self.next_action_chase.append(1)  # FORWARD_CMD
                    self.last_too_close_to_wall = True
#                    elif self.last_too_close_to_wall == True:
#                        action = 1 # FORWARD_CMD
#                        command = COMMAND_CHASE[action]
                else:
                    print(f"Target is : ({self.target_x, self.target_y})")
                    print(f"Predicted action: {command}, reward: {reward}")
                    self.last_too_close_to_wall = False



        self.time += 1
        return command


    def find_min_target(self, my_x, my_y, category :str, scene_info: dict) -> tuple:
        """
        Find the nearest target from the given category
        """
        competitors = scene_info[category]
        competitors_distance = []
        for competitor in competitors:
            dx = competitor["x"] - my_x
            dy = competitor["y"] - my_y
            distance = math.sqrt(dx**2 + dy**2)
            competitors_distance.append(distance)
        target_index = competitors_distance.index(min(competitors_distance))
        print(f"Target is a {category}")
        return competitors[target_index]["x"], competitors[target_index]["y"], competitors_distance, target_index

    def is_too_close_to_wall(self, my_x, my_y, walls_info, safe_distance=50) -> tuple: #safe_distance:可更改與牆壁的安全距離
        """
        Check if the tank's next position (x, y) is too close to any wall.
        """
        for wall in walls_info:
            # Each wall is effectively at (wall["x"], wall["y"]).
            # Because walls might be bigger or smaller in your environment, you might
            # want to do a bounding-box check or distance check.
            # For simplicity, we’ll do a direct distance check:
            dist = math.hypot(wall["x"] - my_x, wall["y"] - my_y)
            if dist < safe_distance:
                return True, wall["x"], wall["y"]
        return False, None, None
    def line_of_fire_clear(self, scene_info):
        """
        Check if shooting along gun_angle from (self_x, self_y) would hit a teammate or a wall.
        """
        # Convert gun angle to a direction vector for the bullet
        rad_angle = math.radians(scene_info["gun_angle"])
        teammates_info = scene_info["teammate_info"]
        walls_info = scene_info["walls_info"]
        dir_x = -math.cos(rad_angle) #注意正負號
        dir_y = math.sin(rad_angle)  # Negative if y-axis increases downwards in your game

        # We can sample points along the bullet path up to the bullet’s max range.
        # For demonstration, let's do a few discrete steps:
        max_bullet_range = 300  # or however far bullets travel
        steps = 10
        step_size = max_bullet_range // steps

        #clear result
        for step in range(0, max_bullet_range, step_size):
            check_x = self.x + dir_x * step
            check_y = self.y + dir_y * step

            # Check if we hit the target before the bullet hits the wall or teammates
            dist_to_target = math.hypot(self.target_x - check_x, self.target_y - check_y)
            if dist_to_target < 25:
                return True  
            # 1) Check collision with teammates
            for tm in teammates_info:
                dist_to_teammate = math.hypot(tm["x"] - check_x, tm["y"] - check_y)
                #去除自己
                if tm["x"] == self.x and tm["y"] == self.y:
                    continue
                if dist_to_teammate < 20:  # pick some radius around the tank
                    return False  # line of fire is not clear

            # 2) Check collision with walls
            for w in walls_info:
                dist_to_wall = math.hypot(w["x"] - check_x, w["y"] - check_y)
                if dist_to_wall < 25:  # some threshold
                    return False
        return True  # line of fire is clear
    def get_chase_reward(self, obs: dict, action: int) -> float:
        angle_reward: float = self.cal_direction_reward(obs, action)
        forward_reward: float = self.cal_forward_reward(obs, action)

        total_reward: float = angle_reward + forward_reward

        return total_reward

    def cal_direction_reward(self, obs: dict, action: int) -> float:

        angle_reward: float = 0.0

        # the gun angle is point at the right side of the target
        if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 3: # TURN_LEFT_CMD
            angle_reward = 3.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 4:   # TURN_RIGHT_CMD
            angle_reward = -3.0

        # the gun angle is point at the left side of the target
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 4:   # TURN_RIGHT_CMD
            angle_reward = 3.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 3:   # TURN_LEFT_CMD
            angle_reward = -3.0
        else:
            angle_reward = -2.0
        #7 1      
        #0 1
        return angle_reward
    
    def cal_forward_reward(self, obs: dict, action: int) -> float:
        forward_reward: float = 0.0

        # target is in front of the player
        if obs[0] == obs[1] and action == 1:    # FORWARD_CMD
            forward_reward = 7.0 #或許調低一點?7.0
        elif obs[0] == obs[1] and action == 2:  # BACKWARD_CMD
            forward_reward = -5.0 #或許調低一點?5.0

        # target is behind the player
        elif obs[0] == (obs[1] + 4) % 8 and action == 2:    # BACKWARD_CMD
            forward_reward = 7.0
        elif obs[0] == (obs[1] + 4) % 8 and action == 1:    # FORWARD_CMD
            forward_reward = -5.0
        # target is in front of or behind the player but the player is not moving forward or backward
        elif (obs[0] == obs[1] or obs[0] == (obs[1] + 4) % 8) and (action != 1 or action != 2):
            forward_reward = -3.0 #或許調低一點?3.0
        return forward_reward

    def get_aim_reward(self, obs: dict, action: int) -> float:
        angle_reward: float = self.cal_angle_reward(obs, action)
        shoot_reward: float = self.cal_shoot_reward(obs, action)

        total_reward: float = angle_reward + shoot_reward

        return total_reward

    def cal_angle_reward(self, obs: dict, action: int) -> float:

        angle_reward: float = 0.0

        # the gun angle is point at the right side of the target
        if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 1: # AIM_LEFT_CMD
            angle_reward = 5.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]] and action == 2:   # AIM_RIGHT_CMD
            angle_reward = -5.0

        # the gun angle is point at the left side of the target
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 2:   # AIM_RIGHT_CMD
            angle_reward = 5.0
        elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]] and action == 1:   # AIM_LEFT_CMD
            angle_reward = -5.0

        # the gun angle is point at the opposite side of the target
        elif obs[0] == (obs[1] + 4) % 8 and (action ==1 or action == 2):
            angle_reward = 3.0
        
        return angle_reward
    
    def cal_shoot_reward(self, obs: dict, action: int) -> float:
        shoot_reward: float = 0.0

        if obs[0] == obs[1] and action == 3:    # SHOOT
            shoot_reward = 30.0
        elif action == 3:
            shoot_reward = -30.0
        #same direction but model doesn't shoot
        elif obs[0] == obs[1]:
            shoot_reward = -30.0
        return shoot_reward    
    
    def reset(self):
        """
        Reset the status
        """
        self.x = None
        self.y = None
        self.next_action_chase = [] #A queue store the demanded next action if needed
        self.next_action_aim = [] #A queue store the demanded next action if needed
        self.last_too_close_to_wall = False
        print(f"Resetting Game {self.side}")

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0) #scene_info.get("angle", 0) + 180
        tank_angle_index: int = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y
        angle_to_target = 180 - math.degrees(math.atan2(dy, dx)) #math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        print("Chase obs: " + str(obs))
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)     # -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0)   #scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y  # target_y - player_y
        angle_to_target = 180 - math.degrees(math.atan2(dy, dx))   #
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print(f"Gun angle: {gun_angle}")
        print(f"Gun angle index: {gun_angle_index}")
        print("Aim angle: " + str(angle_to_target))
        print(f"Angle to target index: {angle_to_target_index}")
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        print("Aim obs: " + str(obs))
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