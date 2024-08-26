import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.optim as optim
import cv2
from matplotlib import pyplot as plt
from gym.wrappers import RecordVideo
from datetime import datetime
import os

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions, h_dimension):
        super(DQN, self).__init__()

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, h_dimension), 
            nn.ReLU(),
            nn.Linear(h_dimension, num_actions)
        )

    def forward(self, x):
        return self.layers_cnn(x)

class DQNAgent:
    def __init__(self,
                 action_space,
                 epsilon=1.0,
                 gamma=0.95,
                 epsilon_min=0.1,
                 epsilon_decay=0.9999,
                 lr=1e-3,
                 memory_len=5000,
                 frames=4,  
                 hidden_dimension=128,
                 device=None):

        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_len = memory_len
        self.lr = lr
        self.memory = deque(maxlen=self.memory_len)
        self.action_space = action_space

        self.target_model = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.model = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def is_explore(self):
        return np.random.rand() <= self.epsilon

    def act(self, state, is_only_random=False, is_only_exploit=False):
        if not is_only_exploit and self.is_explore() or is_only_random:
            action_index = np.random.randint(len(self.action_space))
        else:
            with torch.no_grad():
                q_values = self.model(state)[0] 
                action_index = torch.argmax(q_values).item()

        return self.action_space[action_index]

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []

        for state, action_index, reward, next_state, done in minibatch:
            state = state.to(self.device) 
            next_state = next_state.to(self.device)  

            target = self.model(state)[0]

            train_state.append(target)

            target_copy = target.detach().clone().to(self.device)
            
            if done:
                target_copy[action_index] = reward
            else:
                t = self.target_model(next_state)[0]
                target_copy[action_index] = reward + self.gamma * torch.max(t).item()

            train_target.append(target_copy)

        criterion = nn.MSELoss()
        pred, tru = torch.stack(train_state), torch.stack(train_target)
        loss = criterion(pred, tru)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name):
        self.model = torch.load(name)
        self.target_model = torch.load(name)
        self.model.eval()

    def save_model(self, name):
        torch.save(self.target_model.state_dict(), name)

def preprocess_observation(observation, new_shape=(84, 84)):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, new_shape, interpolation=cv2.INTER_AREA)
    return np.expand_dims(observation, axis=-1)  

class TestTraining:
    def __init__(self, env_name, episodes=1000, batch_size=32, target_update=10, hidden_dimension=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env_name = env_name
        self.episodes = episodes
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        env = gym.make('ALE/Pacman-v5', render_mode=None)
        self.env = env
        self.action_space = list(range(self.env.action_space.n))
        
        self.agent = DQNAgent(
            action_space=self.action_space,
            hidden_dimension=hidden_dimension,
            device=self.device
        )

    def run(self):
        for episode in range(self.episodes):
            observation = preprocess_observation(self.env.reset()[0])
            state = np.expand_dims(observation, axis=0)
            state = torch.tensor(state, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

            done = False
            total_reward = 0

            while not done:
                action = self.agent.act(state)

                next_observation, reward, done = self.env.step(action)  
                next_state = preprocess_observation(next_observation)
                next_state = np.expand_dims(next_state, axis=0)
                next_state = torch.tensor(next_state, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                total_reward += reward

                self.agent.memorize(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

            if episode % self.target_update == 0:
                self.agent.update_target_model()

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.4f}")

        print("Training completed.")

    def save(self, model_name):
        self.agent.save_model(model_name)

    def load(self, model_name):
        self.agent.load_model(model_name)

if __name__ == "__main__":
    trainer = TestTraining(env_name='ALE/Pacman-v5')
    trainer.run()
    trainer.save("pacman_dqn_model.pth")
