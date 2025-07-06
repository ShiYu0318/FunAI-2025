import os
import pickle
import math


class MLPlay:
    def __init__(self, *args, **kwargs):
        print("Initial ml script")

        self.search_range = 1000
        encoder_path = "dataset/knn_encoder.pkl"
        model_path = "dataset/knn_model.pkl"  # Adjust path if needed
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Encoder file not found at {encoder_path}")
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        score_vector = [0, 0, 0, 0]  # score for [up, down, left, right]

        for food in scene_info["foods"]:

            dx = food["x"] - scene_info["self_x"]
            dy = food["y"] - scene_info["self_y"]

            dis = self.get_distance(scene_info["self_x"], scene_info["self_y"], food["x"], food["y"])
            w = abs(food["score"]) / (dis+1)

            if(scene_info["score_to_pass"] < 110):
                if(food["score"] < 0): continue
            else:
                # 第 11 關開始垃圾比較多 將垃圾的權重調高提高閃避能力
                if(food["type"] == "GARBAGE_3"): w *= scene_info["score_to_pass"] + scene_info["score"] * 5
                elif(food["type"] == "GARBAGE_2"): w *= scene_info["score_to_pass"] + scene_info["score"]
                elif(food["type"] == "FOOD_3"): w *= scene_info["score_to_pass"] / 10
                if(scene_info["score_to_pass"] >= 140 and food["type"] == "GARBAGE_3"): w *= 10000

            dir = self.get_direction(dx, dy)

            score_vector[dir] += food["score"] * w


        for i in range(4):
            if(score_vector[i] == 0): 
                score_vector[i] = -9999

        print(score_vector)

        # Feature vector must match training format
        X = [score_vector[0], score_vector[1],
             score_vector[2], score_vector[3]]

        # Predict the numeric label
        pred_label = self.model.predict([X])[0]

        # Convert numeric label back to command if using encoder
        action = self.encoder.inverse_transform([pred_label])[0]

        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        if(action == "U"): ac_idx = 0
        elif(action == "D"): ac_idx = 1
        elif(action == "L"): ac_idx = 2
        elif(action == "R"): ac_idx = 3
        print(action)

        return [actions[ac_idx]]  # Return as a list, e.g., ["UP"]

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

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass


    def get_distance(self, x1, y1, x2, y2):
        """
        Calculate the distance between two points
        """
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
