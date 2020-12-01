from gym.envs.box2d.car_racing import *
import torchvision.transforms as transforms
import torch
import numpy as np

device = torch.device('cuda')
tensorize = transforms.Compose([transforms.ToTensor(),])

def Rollout(pca, cont, render=False, return_frames=False):
    assert cont.__name__ == 'global'
    
    env = CarRacing()
    env.render()
    obs = env.reset()
    total_reward = 0
    limit = 800
    time = 0
    frames = []
    while True:
        if render:
            env.render()
        obs = obs.reshape(1, -1)
        obs = obs / 255
        z = pca.transform(obs)
        z = tensorize(z).to(device)
        action = cont(z)
        obs, reward, done, _ = env.step(action.squeeze().detach().cpu().numpy())
        if return_frames:
            frames.append(obs)# clipping??
        total_reward += reward
        
        if done or time > limit:
            env.close()
            break
        time += 1
    print("Reward:", total_reward)
    if return_frames:
        return total_reward, frames
    return total_reward