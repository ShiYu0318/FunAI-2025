import numpy as np
import os
import torch
import sys
sys.path.append(os.path.dirname(__file__))


from Environment import Environment as env
from Network import PolicyNet


class MLPlay:
    def __init__(self,ai_name:str,*args,**kwargs):
        self.other_cars_position = []
        self.coins_pos = []
        self.ai_name = ai_name
        print("Initial ml script")
        print(ai_name)
        print(kwargs)

        self.action_space = [["BRAKE"],["SPEED"],["MOVE_LEFT"],["MOVE_RIGHT"], [""]]
        n_actions = len(self.action_space)
        n_observations = 7 #args


        self.env = env(n_actions, n_observations)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.policy_net = PolicyNet(n_observations, n_actions).to(self.device)

        load_dir = './save'
        model_path = os.path.join(load_dir, "model.pt")

        if os.path.exists(model_path):
            print("Loading the model ... ", end="")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["PolicyNet"])
            print("Done.")
        else:
            print('ERROR: No model saved')

        self.total_reward = 0
        self.step_ctr = 0

        # self.action = 4 #[""] do nothing
        

    def update(self, scene_info: dict,*args,**kwargs):
        with torch.no_grad():
            self.env.set_scene_info(scene_info)
            observation = self.env.reset()
            observation_tensor = torch.tensor(np.expand_dims(observation, axis=0), dtype=torch.float32, device=self.device)

            action = self.policy_net.choose_action(observation_tensor, deterministic=True)
            action = torch.argmax(action)

            observation, reward, done, info = self.env.step(action.cpu().numpy())

        self.total_reward += reward
        self.step_ctr += 1

        if done:
            print("[Evaluation Total reward = {:.6f}, length = {:d}]".format(self.total_reward, self.step_ctr), flush=True)
            self.total_reward = 0
            self.step_ctr = 0

            return "RESET"

        return self.action_space[action] 


    def reset(self):
        """
        Reset the status
        """
        pass
