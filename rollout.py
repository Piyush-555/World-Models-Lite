import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from gym.envs.box2d.car_racing import *


device = torch.device('cpu')

def Rollout(pca, cont, render=False, return_frames=False):
    assert cont.__name__ == 'global'
    
    env = CarRacing()
    env.render()
    obs = env.reset()
    total_reward = 0
    limit = 800
    action = np.array([1, 0, 0], dtype=np.float64)
    time = 0
    frames = []
    while True:
        if render:
            env.render()
        obs = obs.reshape(1, -1)
        obs = obs / 255
        z = pca.transform(obs)
        z = torch.cat([torch.tensor(z).double(), torch.tensor(action).double().unsqueeze(0)], dim=1).to(device)
        action = cont(z)
        action = action[0]
        action[0] = F.tanh(action[0])
        action[1] = F.sigmoid(action[1])
        action[2] = F.sigmoid(action[2])
        action = action.detach().cpu().numpy()
        obs, reward, done, _ = env.step(action)
        if return_frames:
            frames.append(obs)  # clipping??
        total_reward += reward
        
        if done or time > limit:
            env.close()
            break
        time += 1
    print("Reward:", total_reward)
    if return_frames:
        return total_reward, frames
    return total_reward