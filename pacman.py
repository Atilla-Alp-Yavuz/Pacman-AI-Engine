import gymnasium as gym
import time
import mss
import mss.tools
from PIL import Image
import numpy as np
import sys
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


env = gym.make('ALE/Pacman-v5',
               obs_type='rgb',  # ram | rgb | grayscale
               frameskip=4,  # Reduced frame skip for smoother rendering
               mode=None,  # game mode, see Machado et al. 2018
               difficulty=None,  # game difficulty, see Machado et al. 2018
               repeat_action_probability=0.25,  # Sticky action probability
               full_action_space=False,  # Use all actions
               render_mode="human"  # None | human | rgb_array
               )

target_fps = 120
frame_duration = 1.0 / target_fps

arr = []
for i in range(1000):
    action = env.action_space.sample()
    if action not in arr:
        arr.append(action)

print(arr)

start_time = time.time()

episodes = 5

print(env.action_space)

save_dir = '/home/alp_stajyer/Resimler/Screenshots/'

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


model = build_model(state_size, action_size)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DQNAgent(state_size, action_size)

# Training the agent
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Define the reward structure based on game events
        if info['event'] == 'eat_dot':
            reward = 1
        elif info['event'] == 'eat_power_pellet':
            reward = 5
        elif info['event'] == 'eat_ghost':
            reward = 10
        elif info['event'] == 'caught_by_ghost':
            reward = -50
        elif info['event'] == 'lose_life':
            reward = -100
        elif info['event'] == 'complete_level':
            reward = 50
        else:
            reward = 0  # For movements and other events

        total_reward += reward

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {total_reward}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)



