import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from collections import deque
import random
import torch.optim as optim
import cv2
import os
from sklearn.linear_model import LinearRegression

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
            nn.Flatten()
        )

        self._calculate_flattened_size(in_channels)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, h_dimension),
            nn.ReLU(),
            nn.Linear(h_dimension, num_actions)
        )

    def _calculate_flattened_size(self, in_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)
            output = self.layers_cnn(dummy_input)
            self.flattened_size = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.layers_cnn(x)
        x = self.fc(x)
        return x

class TD3Agent:

    def __init__(self, action_space, epsilon=1.0, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.9995, lr=0.01, memory_len=10000, frames=1, hidden_dimension=256, device=None, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):


        ##HYPERPARAM TUNING

        #DECAY ==> 0.9999 TO 0.9995
        #gamma ==> 0.95 to 0.98
        #memory_len 50000 to 100000
        #hidden 128 to 256


        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_len = memory_len
        self.lr = lr
        self.memory = deque(maxlen=self.memory_len)
        self.action_space = action_space

        self.critic_1 = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.critic_2 = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.target_critic_1 = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.target_critic_2 = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)

        self.actor = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)
        self.target_actor = DQN(frames, len(self.action_space), hidden_dimension).to(self.device)

        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=self.lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.update_target_models(tau=1.0)

        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def update_target_models(self, tau=None):
        tau = self.tau if tau is None else tau

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, state):

        action_values = self.actor(state)[0].detach().cpu().numpy()
        

        noise = np.random.normal(0, self.policy_noise, size=action_values.shape)
        action_values = np.clip(action_values + noise, -self.noise_clip, self.noise_clip)
       
        action = np.argmax(action_values)
        
        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.total_it += 1

        minibatch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*minibatch)

        state = torch.cat(state).to(self.device)
        action = torch.tensor(action).to(self.device).unsqueeze(1)  
        reward = torch.tensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.cat(next_state).to(self.device)
        done = torch.tensor(done).float().to(self.device).unsqueeze(1)

        next_action = self.target_actor(next_state)[0]

        noise = torch.normal(0, self.policy_noise, size=next_action.shape).clamp(-self.noise_clip, self.noise_clip).to(self.device)
        next_action = (next_action + noise).clamp(-1, 1)

        target_q1 = self.target_critic_1(next_state)
        target_q2 = self.target_critic_2(next_state)
        target_q = torch.min(target_q1, target_q2).squeeze()
        target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state).gather(1, action.long())
        current_q2 = self.critic_2(state).gather(1, action.long())

        loss_q1 = nn.MSELoss()(current_q1, target_q.unsqueeze(1))
        loss_q2 = nn.MSELoss()(current_q2, target_q.unsqueeze(1))

        self.optimizer_critic_1.zero_grad()
        loss_q1.backward(retain_graph=True) 
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        loss_q2.backward() 
        self.optimizer_critic_2.step()


        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state).gather(1, self.actor(state).max(1)[1].unsqueeze(1)).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()


            self.update_target_models()

    def save_model(self, file_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'target_critic_1_state_dict': self.target_critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'target_critic_2_state_dict': self.target_critic_2.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_1_state_dict': self.optimizer_critic_1.state_dict(),
            'optimizer_critic_2_state_dict': self.optimizer_critic_2.state_dict(),
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic_1.load_state_dict(checkpoint['optimizer_critic_1_state_dict'])
        self.optimizer_critic_2.load_state_dict(checkpoint['optimizer_critic_2_state_dict'])

def preprocess_observation(observation, new_shape=(84, 84)):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, new_shape, interpolation=cv2.INTER_AREA)
    return np.array(observation, dtype=np.float32) / 255.0

class TestTraining:
    def __init__(self, env_name, episodes=1000, batch_size=128, target_update=10, hidden_dimension=128, gamma=0.95, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env_name = env_name
        self.episodes = episodes
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.gamma = gamma
        self.lr = lr 

        env = gym.make('ALE/Pacman-v5', render_mode=None)
        self.env = env
        self.action_space = list(range(self.env.action_space.n))
        
        self.agent = TD3Agent(
            action_space=self.action_space,
            gamma=self.gamma,
            lr=self.lr,
            hidden_dimension=hidden_dimension,
            device=self.device
        )

        self.rewards = []

    def calculate_trend(self, rewards):
        X = np.arange(len(rewards)).reshape(-1, 1)
        y = np.array(rewards).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]

    def run(self):
        sumReward = 0
        for episode in range(self.episodes):
            observation = preprocess_observation(self.env.reset()[0])
            state = np.expand_dims(observation, axis=0)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state)
                result = self.env.step(action)
                next_observation = result[0]
                reward = result[1]
                done = result[2]
                next_state = preprocess_observation(next_observation)
                next_state = np.expand_dims(next_state, axis=0)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                total_reward += reward
                self.agent.memorize(state, action, reward, next_state, done)
                state = next_state

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

            sumReward += total_reward
            self.rewards.append(total_reward)
            avg = sumReward / (episode + 1)
            print(f"Episode {episode + 1}/{self.episodes} - Reward: {total_reward:.2f} - Average Reward: {avg:.2f}")
            if (episode + 1) % 500 == 0:
                trend = self.calculate_trend(self.rewards)
                avg_reward = np.mean(self.rewards[-500:])
                print(f"Episode {episode + 1}/{self.episodes} - Reward: {total_reward:.2f} - Average Reward (last 1000 episodes): {avg_reward:.2f} - Reward Trend: {trend:.2f}")

        self.env.close()

    def save(self, model_name):
        self.agent.save_model(model_name)

    def load(self, model_name):
        self.agent.load_model(model_name)

    def validate(self):
        for param_tensor in self.agent.actor.state_dict():
            print(f'{param_tensor}: {self.agent.actor.state_dict()[param_tensor].size()}')

def compare_model_weights(model, checkpoint):
    model_weights = model.state_dict()

    for key in model_weights:
        if key not in checkpoint:
            print(f"Key {key} not found in checkpoint")
            return False
        if not torch.equal(model_weights[key], checkpoint[key]):
            print(f"Mismatch in parameter: {key}")
            return False
    print("All weights match.")
    return True

if __name__ == "__main__":
    model_path = "mspacman_dqn_model.pth"
    pacman = TestTraining('ALE/MsPacman-v5', episodes=10000, batch_size=128)
    
    if os.path.exists(model_path):
        print(f"Loading old model existing at {model_path}...")
        checkpoint = torch.load(model_path)
        pacman.load(model_path)
        pacman.validate()
        print("Comparing model weights...")
        compare_model_weights(pacman.agent.actor, checkpoint)
    else:
        print("No saved model found. Starting training from scratch.")
    
    pacman.run()
    
    print(f"Model saved to {model_path}...")
    pacman.save(model_path)


