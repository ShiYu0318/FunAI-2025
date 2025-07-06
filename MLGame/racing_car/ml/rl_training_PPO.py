import numpy as np
import os
import torch
import sys
sys.path.append(os.path.dirname(__file__))
import json

from Environment import Environment as env
from Network import PolicyNet, ValueNet
from PPO import PPO

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

        self.gamma = 0.99
        self.lamb = 0.95

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNet(n_observations, n_actions).to(self.device)
        self.value_net = ValueNet(n_observations).to(self.device)
        self.agent = PPO(self.policy_net, self.value_net, lr=1e-4, max_grad_norm=0.5, ent_weight=0.01, \
            clip_val=0.2, sample_n_epoch=4, sample_mb_size=64, device=self.device)

        load_dir = './save' # './save'
        model_path = os.path.join(load_dir, "model.pt")

        keep_training = True #args
        if keep_training:
            if os.path.exists(model_path):
                print("Loading the model ... ", end="")
                checkpoint = torch.load(model_path)
                self.policy_net.load_state_dict(checkpoint["PolicyNet"])
                # self.value_net.load_state_dict(checkpoint["ValueNet"]).to(self.device)
                print("Done.")
            else:
                print('ERROR: No model saved')


        self.env = env(n_actions, n_observations)

        self.save_dir = "./save"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        log = {"actor loss":[], "critic loss":[], "entropy":[], "mean return":[], "mean length":[]}
        with open("./save/log.txt", "w") as f:
            json.dump(log, f)

        self.save_frq = 200 #args

        self.mean_total_reward = 0
        self.mean_length = 0
        
        self.max_episode = 10000 #args
        self.episode_ctr = 0

        self.max_step = 2048 #args
        self.step_ctr = 0
        
        self.pg_loss = 0
        self.v_loss = 0
        self.ent = 0

        #Storages (state, action, value, reward, a_logp)
        self.mb_states = np.zeros((self.max_step, n_observations), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, n_actions), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)

        self.action = 4 #[""] do nothing
        

    def update(self, scene_info: dict,*args,**kwargs):

        if self.episode_ctr > self.max_episode:
            print("---------------- Training is over --------------------.")
            exit()

        self.env.set_scene_info(scene_info)
        self.observation, reward, done, info = self.env.step(self.action)
        self.mb_rewards[self.step_ctr] = reward

        if done:
            self.train_model()
            self.episode_ctr += 1
            self.step_ctr = 0

            return "RESET"

        self.step_ctr += 1

        self.evaluate_action()
        
        return self.action_space[self.action] 

    def reset(self):
        """
        Reset the status
        """
        if self.episode_ctr != 0 and self.episode_ctr % self.save_frq == 0:
            self.save_model()
        
    
    def train_model(self):
        with torch.no_grad():
            last_value = self.value_net(
                torch.tensor(np.expand_dims(self.observation, axis=0), dtype=torch.float32, device=self.device)
            ).cpu().numpy()

            mb_returns = self.compute_discounted_return(self.mb_rewards[:self.step_ctr], last_value)
        #    mb_returns = self.compute_gae(self.mb_rewards[:self.step_ctr], self.mb_values[:self.step_ctr], last_value)
            mb_advs = mb_returns - self.mb_values[:self.step_ctr]
            # print(mb_advs)
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        with torch.enable_grad():
            self.pg_loss, self.v_loss, self.ent = self.agent.train(self.mb_states[:self.step_ctr], self.mb_actions[:self.step_ctr], \
                self.mb_values[:self.step_ctr], mb_advs, mb_returns, self.mb_a_logps[:self.step_ctr])

        self.mean_total_reward += self.mb_rewards[:self.step_ctr].sum()
        self.mean_length += len(self.mb_states[:self.step_ctr])
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format( \
            self.episode_ctr, self.mb_rewards[:self.step_ctr].sum(), len(self.mb_states[:self.step_ctr])))


    def save_model(self):
        print("\n[{:5d} / {:5d}]".format(self.episode_ctr, self.max_episode))
        print("----------------------------------")
        print("actor loss = {:.6f}".format(self.pg_loss))
        print("critic loss = {:.6f}".format(self.v_loss))
        print("entropy = {:.6f}".format(self.ent))
        print("mean return = {:.6f}".format(self.mean_total_reward / self.save_frq))
        print("mean length = {:.2f}".format(self.mean_length / self.save_frq))
        print("\nSaving the model ... ", end="")
        torch.save({
            "it": self.episode_ctr,
            "PolicyNet": self.policy_net.state_dict(),
            "ValueNet": self.value_net.state_dict()
        }, os.path.join(self.save_dir, "model.pt"))
        
        with open("./save/log.txt", "r+") as f:
            log = json.load(f)
            log["actor loss"].append(self.pg_loss)
            log["critic loss"].append(self.v_loss)
            log["entropy"].append(self.ent)
            log["mean return"].append(self.mean_total_reward / self.save_frq)
            log["mean length"].append(self.mean_length / self.save_frq)
            f.seek(0)
            json.dump(log, f, indent = 4)

        
        # with open("./save/log.json", "w") as f:
        #     json.dump(log, f)

        print("Done.")
      
        self.mean_total_reward = 0
        self.mean_length = 0

    def evaluate_action(self):
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(self.observation, axis=0), dtype=torch.float32, device=self.device)
            action, a_logp = self.policy_net(state_tensor) #self.action = probability of action space
            self.action = torch.argmax(action)

            value = self.value_net(state_tensor)

            action = action.cpu().numpy()[0]
            a_logp = a_logp.cpu().numpy()
            value = value.cpu().numpy()

            self.mb_states[self.step_ctr] = self.observation
            self.mb_actions[self.step_ctr] = action
            self.mb_a_logps[self.step_ctr] = a_logp
            self.mb_values[self.step_ctr] = value
            
    #Compute discounted return
    def compute_discounted_return(self, rewards, last_value):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]

        return returns
    
    #Compute generalized advantage estimation (Optional), discounted rewards - value estimates
    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma*next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma*self.lamb*last_gae_lam

        return advs + values
