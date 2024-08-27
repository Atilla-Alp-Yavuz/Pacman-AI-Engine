import gymnasium as gym
import time
import random

env = gym.make('ALE/MsPacman-v5', render_mode='human')

state, info = env.reset()

done = False
total_reward = 0
penalty = -1  
previous_reward = 0 

ind = True

#TESTING SOME HARDCODED ACTIONS FOR PILL AND GHOST EATING


def pretrain(total_reward , factor, previous_reward , select):

    print(select)
    if(factor == 0):
        if(total_reward < 60):
            return 2
        elif(total_reward < 100):
            return 4
        elif(total_reward < 140):
            return 2
        elif(total_reward < 200):
            return 1
        elif(total_reward < 230):
            print(total_reward)
            return 3
        elif(select):
            print("PLUUH")
            return 1
        elif(total_reward == 230):
            return 3
        else:
            return 10
    else:
        if(total_reward < 60):
            return 3
        elif(total_reward < 100):
            return 4
        elif(total_reward < 140):
            return 3
        elif(total_reward < 200):
            return 1
        elif(total_reward < 230):
            print(total_reward)
            return 2
        elif(select):
            print("PLUUH")
            return 1
        elif(total_reward == 230):
            return 2
        else:
                return 10

episodes = 3



for t in range(episodes):

    total_reward = 0
    done = False
    select = True 
    k = 0
    previous_reward = 0

    factor = random.randint(0,1)

    while not done:

        if(total_reward == 230):

            k+=1
        
        if( k > 10):
            select = False

        action = pretrain(total_reward , factor , previous_reward , select)
        
        if(action == 10):
            action = env.action_space.sample()

        next_observation, reward, done, _, _ = env.step(action)

        previous_reward = total_reward
        total_reward += reward


    print(f"Episode {t+1} finished with total reward: {total_reward}")

env.close()


