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

#import plotly.graph_objs as go
#import plotly.io as pio

####PLOTTING DOES NOT WORK FOR SOME FUCKİNG REASON
###GET THE RAW DATA AND FEED IT TO ANOTHER PLOTTING FILE

### BİR FONSKİYON YAZ BELLİ DATA POINTLER VERINCE AVGLARIN AVERAGE EĞİLİMİNİNİ VERECEK

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

class DQNAgent:

    def __init__(self, action_space, epsilon=1.0, gamma=0.95, epsilon_min=0.1, epsilon_decay=0.9999, lr=0.001, memory_len=5000, frames=1, hidden_dimension=128, device=None):

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
        flip = np.random.rand() <= self.epsilon
        return flip

    def act(self, state, is_only_random=False, is_only_exploit=False):
        if not is_only_exploit and self.is_explore() or is_only_random:
            action_index = np.random.randint(len(self.action_space))
        else:
            q_values = self.target_model(state)[0]
            action_index = torch.argmax(q_values)

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
                target_copy[action_index] = reward + self.gamma * torch.max(t)

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
        torch.save(self.target_model, name)


def preprocess_observation(observation, new_shape=(84, 84)):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, new_shape, interpolation=cv2.INTER_AREA)
    return np.array(observation, dtype=np.float32) / 255.0


class TestTraining:
    #32 10 128 0.95 1e-3
    #rearrange them : SyntaxError: non-default argument follows default arguments
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
        
        self.agent = DQNAgent(
            action_space=self.action_space,
            gamma = self.gamma,
            lr = self.lr,
            hidden_dimension=hidden_dimension,
            device=self.device
        )
        self.rewards = []

    def run(self):

        sumReward = 0
        for episode in range(self.episodes):
            
            observation = preprocess_observation(self.env.reset()[0])
            state = np.expand_dims(observation, axis=0)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            curr = 0.0
            done = False
            total_reward = 0
            avg = 0.0

            while not done:
                
                action = self.agent.act(state)

                result = self.env.step(action)

                next_observation = result[0]
                reward = result[1]
                done = result[2]
                curr = reward

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

            if episode % self.target_update == 0:
                self.agent.update_target_model()
            

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.4f}, Avg reward per episode: {avg:.4f}")
            
            """
            if (episode + 1) % 100 == 0:    
                self.plot_rewards(episode + 1)"""

        print("Training completed.")

    def save(self, model_name):
        self.agent.save_model(model_name)

    def load(self, model_name):
        self.agent.load_model(model_name)




"""
def plot_rewards(self, current_episode):
    intervals = range(1, current_episode + 1, 1)
    
    fig = go.Figure()

    for i in intervals:
        fig.add_trace(go.Scatter(
            x=list(range(i)),
            y=self.rewards[:i],
            mode='lines',
            name=f'Episode 0-{i}'
        ))

    fig.update_layout(
        title='Reward vs. Episode',
        xaxis_title='Episode',
        yaxis_title='Total Reward',
        legend_title='Episodes',
        template='plotly_white'
    )
    
    # Save the plot as a PNG file
    pio.write_image(fig, f'pacman_rl_{current_episode}.png')


    


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
"""

if __name__ == "__main__":
    
    episode_counts = [100]
    batch_sizes = [32, 64, 128]
    gammas = [0.95, 0.99]
    lrs = [0.0001, 0.001]

    #fine_tune = hyperparameter_tuning(episode_counts, batch_sizes, gammas, lrs)

    #trainer = TestTraining(env_name='ALE/Pacman-v5')
    """
    trainer = TestTraining(
        env_name='ALE/Pacman-v5',
        episodes=fine_tune[0],
        batch_size=fine_tune[1],
        gamma=fine_tune[2],
        lr=fine_tune[3]
    )"""

    trainer = TestTraining(
        env_name='ALE/Pacman-v5',
        episodes=1000,
        batch_size=128,
        gamma=0.95,
        lr=0.001
    )


    trainer.run()
    trainer.save("pacman_dqn_model.pth")

    """    

    def plot_rewards(self, current_episode):

        intervals = range(1, current_episode + 1, 1)
        plt.figure(figsize=(10, 6))
        

        print('Damn')

        for i in intervals:
            print('ssasad')
            plt.plot(range(i), self.rewards[:i], label=f'Episode 0-{i}')

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward vs. Episode')
        
        plt.legend()
        plt.grid(True)
        plt.savefig('pacman_rl_{current_episode}.png')
        #plt.show()
"""
