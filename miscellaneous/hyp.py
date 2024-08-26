import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import matplotlib.pyplot as plt
import os
from itertools import product

#THIS CODE IS FOR HYPERPARAMETER TUNING OF THE SYSTEM
#STRICTLY FOR THIS

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_observation(observation, new_shape=(84, 84)):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, new_shape, interpolation=cv2.INTER_AREA)
    return np.array(observation, dtype=np.float32) / 255.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
    else:
        print("Model file not found. Starting from scratch.")

def train_dqn(episodes, batch_size=64, gamma=0.99, lr=0.0001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, model_filepath='qnetwork.pth'):
    env = gym.make('ALE/Pacman-v5', render_mode=None)
    input_shape = (1, 84, 84)
    num_actions = env.action_space.n

    q_network = QNetwork(input_shape, num_actions)
    target_network = QNetwork(input_shape, num_actions)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(10000)
    
    load_model(q_network, model_filepath)
    
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay
    steps_done = 0

    for episode in range(episodes):
        observation = preprocess_observation(env.reset()[0])
        state = np.expand_dims(observation, axis=0)
        terminated = False
        truncated = False
        total_reward = 0
    
        while not terminated and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_network(torch.tensor(state).unsqueeze(0)).argmax().item()
            
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_observation = preprocess_observation(next_observation)
            next_state = np.expand_dims(next_observation, axis=0)
            replay_buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_network(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1
            if steps_done % epsilon_decay == 0:
                epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
            
            if steps_done % 1000 == 0:
                target_network.load_state_dict(q_network.state_dict())
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    save_model(q_network, model_filepath)

    env.close()

    return total_reward  

def hyperparameter_tuning(episode_counts, batch_sizes, gammas, lrs):
    best_params = None
    best_reward = float('-inf')

    for episodes, batch_size, gamma, lr in product(episode_counts, batch_sizes, gammas, lrs):
        print(f"Testing hyperparameters: episodes={episodes}, batch_size={batch_size}, gamma={gamma}, lr={lr}")
        total_reward = train_dqn(episodes, batch_size=batch_size, gamma=gamma, lr=lr)
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = (episodes, batch_size, gamma, lr)
    
    print(f"Best hyperparameters: episodes={best_params[0]}, batch_size={best_params[1]}, gamma={best_params[2]}, lr={best_params[3]}")
    return best_params

if __name__ == '__main__':

    episode_counts = [100]
    batch_sizes = [32, 64, 128]
    gammas = [0.95, 0.99]
    lrs = [0.0001, 0.001]

    best_params = hyperparameter_tuning(episode_counts, batch_sizes, gammas, lrs)


