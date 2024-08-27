import torch
import gymnasium as gym
import numpy as np
import plotly.graph_objs as go
from trainPacman import TD3Agent, preprocess_observation  

class Tester:

    def __init__(self, name, models):
        self.env = gym.make(name, render_mode=None)
        self.models = models
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = list(range(self.env.action_space.n))
        self.lenMod = 1

    def load_model(self, agent, model_path):
        agent.load_model(model_path)

    def test(self):
        for name, model_path in self.models.items():
            agent = TD3Agent(action_space=self.action_space, device=self.device)
            self.load_model(agent, model_path)
            total_rewards = []

            for episode in range(100):  
                observation = preprocess_observation(self.env.reset()[0])
                state = np.expand_dims(observation, axis=0)
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                total_reward = 0
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_observation, reward, done, _, _ = self.env.step(action)
                    next_state = preprocess_observation(next_observation)
                    next_state = np.expand_dims(next_state, axis=0)
                    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    total_reward += reward
                    state = next_state
                
                total_rewards.append(total_reward)
                print(f"Pending: {self.lenMod} - Episode {episode + 1}/{100} - Reward: {total_reward:.2f}")

            avg_reward = np.mean(total_rewards)
            self.results[name] = avg_reward
            print(f"Average Reward: {avg_reward}")
            self.lenMod += 1

        return self.results  


if __name__ == "__main__":
    name = 'ALE/MsPacman-v5'

    models = {
        "150_episodes": "mspacman_dqn_model_150.pth",
        "250_episodes": "mspacman_dqn_model_250.pth",
        "500_episodes": "mspacman_dqn_model_500.pth",
        "750_episodes": "mspacman_dqn_model_750.pth",
        "1000_episodes": "mspacman_dqn_model_1000.pth",
    }

    tester = Tester(name, models)
    results = tester.test()

    fig = go.Figure(data=[
        go.Scatter(
            x=list(results.keys()),
            y=list(results.values()),
            mode='markers+lines',  
            marker=dict(color='blue', size=10),
            line=dict(color='blue', width=2)
        )
    ])

    fig.update_layout(
        title='Performance of Models Trained on Different Episodes',
        xaxis_title='Trained Episodes',
        yaxis_title='Average Reward over 100 Runs'
    )

    fig.write_image("results5.png")
    
    fig.show()

    tester.env.close()

