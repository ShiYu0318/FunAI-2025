import pickle
import os
import random


class MLPlay:
    def __init__(self, *args, **kwargs):

        self.search_range = 1000
        self.data = []
        self.all_data = []

        # check if the directory dataset exists
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
            print("Directory 'dataset' created.")

        # check if the file training_data.pkl exists
        if os.path.exists("dataset/training_data.pkl"):
            with open("dataset/training_data.pkl", "rb") as f:
                self.all_data = pickle.load(f)
            print(
                f"Loaded existing data with {len(self.all_data)} entries.")
        else:
            print("No existing data found, starting fresh.")

        self.last_status = None

    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """

        score_vector = [0, 0, 0, 0]  # score for [up, down, left, right]


    #-----------------------------------------------------------------------------

        for food in scene_info["foods"]:

            dx = food["x"] - scene_info["self_x"]
            dy = food["y"] - scene_info["self_y"]

            dis = self.get_distance(scene_info["self_x"], scene_info["self_y"], food["x"], food["y"])
            w = abs(food["score"]) / (dis+1)

            if(scene_info["score_to_pass"] < 80):
                if(food["score"] < 0): continue
            else:
                if(food["type"] == "GARBAGE_3"): w *= scene_info["score_to_pass"] + scene_info["score"] * 5
                elif(food["type"] == "GARBAGE_2"): w *= scene_info["score_to_pass"] + scene_info["score"]
                elif(food["type"] == "FOOD_3"): w *= scene_info["score_to_pass"] / 10
                if(scene_info["score_to_pass"] >= 140 and food["type"] == "GARBAGE_3"): w *= 9999999999

            dir = self.get_direction(dx, dy)

            score_vector[dir] += food["score"] * w


        for i in range(4):
            if(score_vector[i] == 0): 
                score_vector[i] = -9999
            
        if(scene_info["score_to_pass"] >= 140):
            score_vector[2] = -9999
            score_vector[3] = -9999
        
        print(score_vector)

    #-----------------------------------------------------------------------------

        # decide the command according to score_vector
        command = self.decide_command(score_vector)

        print(f"cmd = {command}")
        # collect the data
        row = [score_vector[0], score_vector[1],
               score_vector[2], score_vector[3], command[0]]
        self.data.append(row)

        self.last_status = scene_info["status"]

        # return the command to move squid
        return [command]
    
    def get_direction(self, dx, dy):
        if dx + dy > 0:
                if dx - dy > 0:
                    dir =  3  # R
                else:
                    dir =  1  # D
        else:
            if dx - dy > 0:
                dir =  0  # U
            else:
                dir =  2  # L
        return dir

    def decide_command(self, score_vector):
        """
        Decide the command to move the squid
        """
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        max_idx = score_vector.index(max(score_vector))

        return actions[max_idx]
    
    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")

        if self.last_status == "GAME_PASS":
            self.all_data.extend(self.data)

            with open("dataset/training_data.pkl", "wb") as f:
                pickle.dump(self.all_data, f)
            print(f"Data appended, total {len(self.all_data)} entries saved.")

        self.data.clear()

    def get_distance(self, x1, y1, x2, y2):
        """
        Calculate the distance between two points
        """
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
