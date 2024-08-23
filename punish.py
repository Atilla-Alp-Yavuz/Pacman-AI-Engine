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
import plotly.graph_objs as go
import pandas as pd



class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0  
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):

        if isinstance(priority, (np.ndarray, list)):
            priority = np.isscalar(priority) if isinstance(priority, np.ndarray) else priority[0]
        
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)


    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha  
        self.epsilon = 0.01  

    def push(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions, h_dimension):

        super(DQN, self).__init__()

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),  
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1),  
            nn.ReLU(),
            nn.Flatten()
        )

        self._calculate_flattened_size(in_channels)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, h_dimension),
            nn.ReLU(),
            nn.Linear(h_dimension, h_dimension),
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
    def __init__(
        self, 
        action_space, 
        epsilon=1.0, 
        gamma=0.98, 
        epsilon_min=0.1, 
        epsilon_decay=0.995, 
        lr=0.001, 
        memory_capacity=50000, 
        frames=1, 
        hidden_dimension=512, 
        device=None, 
        tau=0.01, 
        policy_noise=0.1, 
        noise_clip=0.3, 
        policy_freq=2,
        alpha=0.6,  
        beta_start=0.4,  
        beta_frames=100000  
    ):
        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr 
        self.action_space = action_space

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 1  
        self.beta = beta_start
        self.alpha = alpha

        self.memory = PrioritizedReplayBuffer(memory_capacity, alpha)

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

        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            noise = np.random.normal(0, self.policy_noise, size=action_values.shape)
            action_values = np.clip(action_values + noise, -self.noise_clip, self.noise_clip)
            action = np.argmax(action_values)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  

        return action

    def memorize(self, state, action, reward, next_state, done):

        penalty = -10
        if(reward == 0):
            reward+= penalty

        
        state_np = state.cpu().numpy()
        next_state_np = next_state.cpu().numpy()
        action_np = np.array([action], dtype=np.int32)
        reward_np = np.array([reward], dtype=np.float32)
        done_np = np.array([done], dtype=np.bool_)

        sample = (state_np, action_np, reward_np, next_state_np, done_np)

        max_priority = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.memory.push(max_priority, sample)

    def replay(self, batch_size):
        if self.memory.tree.n_entries < batch_size:
            return

        self.total_it += 1
        self.frame_idx += 1
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame_idx / self.beta_frames)

        samples, idxs, is_weights = self.memory.sample(batch_size, self.beta)

        batch = list(zip(*samples))

        states = torch.FloatTensor(np.concatenate(batch[0])).to(self.device)
        actions = torch.LongTensor(np.concatenate(batch[1])).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.concatenate(batch[2])).to(self.device)
        next_states = torch.FloatTensor(np.concatenate(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.concatenate(batch[4])).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device).unsqueeze(1)

        next_actions = self.target_actor(next_states).detach()
        noise = torch.normal(0, self.policy_noise, size=next_actions.shape).clamp(-self.noise_clip, self.noise_clip).to(self.device)
        next_actions = (next_actions + noise).clamp(-1, 1)


        target_q1 = self.target_critic_1(next_states)
        target_q2 = self.target_critic_2(next_states)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q.max(1)[0].unsqueeze(1)

        current_q1 = self.critic_1(states).gather(1, actions)
        current_q2 = self.critic_2(states).gather(1, actions)

        td_errors1 = target_q - current_q1
        td_errors2 = target_q - current_q2

        loss_q1 = (is_weights * td_errors1.pow(2)).mean()
        loss_q2 = (is_weights * td_errors2.pow(2)).mean()
        loss = loss_q1 + loss_q2  

        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()
        loss.backward() 
        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        if self.total_it % self.policy_freq == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic_1(states).gather(1, actor_actions.max(1)[1].unsqueeze(1)).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.update_target_models()


        new_priorities = td_errors1.abs().detach().cpu().numpy() + 1e-6
        for idx, priority in zip(idxs, new_priorities):
            self.memory.update(idx, priority)



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
    def __init__(self, env_name, episodes=1000, batch_size=128, target_update=10, hidden_dimension=512, gamma=0.98, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu',  convergence_window=100, convergence_threshold=0.01):
        self.env_name = env_name
        self.episodes = episodes
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.gamma = gamma
        self.lr = lr 

        env = gym.make('ALE/MsPacman-v5', render_mode=None)
        self.env = env
        self.action_space = list(range(self.env.action_space.n))
        
        self.convergence_window = convergence_window  
        self.convergence_threshold = convergence_threshold  

        self.agent = TD3Agent(
            action_space=self.action_space,
            gamma=self.gamma,
            lr=self.lr,
            hidden_dimension=hidden_dimension,
            device=self.device
        )

        self.rewards = []
        self.rewards_per_episode = []

    def calculate_trend(self, rewards):
        X = np.arange(len(rewards)).reshape(-1, 1)
        y = np.array(rewards).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]



    def calculate_sliding_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def check_convergence(self):
        if len(self.rewards_per_episode) >= self.convergence_window:
            recent_rewards = self.rewards_per_episode[-self.convergence_window:]
            std_recent_rewards = np.std(recent_rewards)
            return std_recent_rewards < self.convergence_threshold
        return False

    def run(self):
        sum_reward = 0
        for episode in range(self.episodes):
            observation = preprocess_observation(self.env.reset()[0])
            state = np.expand_dims(observation, axis=0)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            total_reward = 0
            done = False
            while not done:

                action = self.agent.act(state)
                next_observation, reward, done, _, _ = self.env.step(action)

                penalty  = -10
                if(reward == 0):
                    reward+=penalty

                next_state = preprocess_observation(next_observation)
                next_state = np.expand_dims(next_state, axis=0)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                total_reward += reward
                self.agent.memorize(state, action, reward, next_state, done)
                state = next_state

                if self.agent.memory.tree.n_entries > self.batch_size:
                    self.agent.replay(self.batch_size)

            sum_reward += total_reward
            self.rewards.append(total_reward)
            avg = sum_reward / (episode + 1)
            self.rewards_per_episode.append(avg)
            print(f"Episode {episode + 1}/{self.episodes} - Reward: {total_reward:.2f} - Average Reward: {avg:.2f}")

            if self.check_convergence():
                print("No more significant learning")
                print(f"Convergence reached at episode {episode + 1} with average reward: {avg:.2f}")
                break  

            if (episode + 1) % 250 == 0:
                trend = self.calculate_trend(self.rewards)
                avg_reward = np.mean(self.rewards[-500:])
                print(f"Episode {episode + 1}/{self.episodes} - Reward: {total_reward:.2f} - Average Reward (last 500 episodes): {avg_reward:.2f} - Reward Trend: {trend:.2f}")

            if (episode + 1) % 250 == 0:
                self.plot_rewards((episode + 1) / 250 , self.rewards[-250:])
                print("Graph saved")
            

        self.env.close()

    def plot_rewards(self, i , arr):

        fig = go.Figure()
        tig = go.Figure()
        sag = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(1, len(self.rewards) + 1),
            y=self.rewards,
            mode='lines',
            name='Rewards'
        ))

        fig.update_layout(
            title='Rewards',
            xaxis_title='Episode',
            yaxis_title='Reward',
            template='none',  
            paper_bgcolor='white', 
            plot_bgcolor='white',  
            font=dict(color='black')
        )

        fig.write_image(f"ttt{i}.png")

        tig.add_trace(go.Scatter(
            x=np.arange(1, len(self.rewards_per_episode) + 1),
            y=self.rewards_per_episode,
            mode='lines',
            name='rewards_per_episode'
        ))

        tig.update_layout(
            title='Rewards per Episode',
            xaxis_title='Episode',
            yaxis_title='Reward',
            template='none',  
            paper_bgcolor='white',  
            plot_bgcolor='white',   
            font=dict(color='black')
        )

        tig.write_image(f"rrr{i}.png")

        window_size = 100
        sliding_avg = self.calculate_sliding_average(self.rewards, window_size)

        sag.add_trace(go.Scatter(
            x=np.arange(1, len(sliding_avg) + 1),
            y=sliding_avg,
            mode='lines',
            name=f'Sliding Average (window={window_size})'
        ))

        sag.update_layout(
            title='Sliding Average of Rewards per Episode',
            xaxis_title='Episode',
            yaxis_title='Sliding Average Reward',
            template='none',  
            paper_bgcolor='white',  
            plot_bgcolor='white',   
            font=dict(color='black')
        )

        sag.write_image(f"ppp{i}.png")

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
    model_path = "mspacman_dqn_model_f.pth"
    pacman = TestTraining('ALE/Pacman-v5', episodes=10000, batch_size=128)
    
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
    #pacman.plot_rewards()
    print(f"Model saved to {model_path}...")
    pacman.save(model_path)


