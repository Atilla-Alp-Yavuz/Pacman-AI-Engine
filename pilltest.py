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

"""
def randomshi(total_reward , imp):

    if(imp == 0):

        tot = 0
        next_state, reward, done, truncated, info = env.step(2)
        tot += reward
        total_reward += reward

        while tot < 60:
            next_state, reward, done, truncated, info = env.step(2)
            tot += reward
            total_reward += reward

        while tot < 100:
            next_state, reward, done, truncated, info = env.step(4)
            tot += reward
            total_reward += reward

        while tot < 140:
            next_state, reward, done, truncated, info = env.step(2)
            tot += reward
            total_reward += reward

        next_state, reward, done, truncated, info = env.step(1)
        total_reward += reward
        t = 0

        ###BURADA YUKARI ÇIKIP TÜNELDEN GEÇSE DAHA İYİ OLABİLİR

        while total_reward < 230:
            next_state, reward, done, truncated, info = env.step(3)  
            total_reward += reward
            print(total_reward)
            print(t)

        while(total_reward < 240):
            next_state, reward, done, truncated, info = env.step(1)  
            total_reward += reward
            print(total_reward)
            print(t)
            total_reward+=1

        prev = total_reward

        while(prev == total_reward):
            next_state, reward, done, truncated, info = env.step(3)  
            total_reward += reward


        ind = False

    else:
        
        tot = 0
        next_state, reward, done, truncated, info = env.step(3)
        tot += reward
        total_reward += reward

        while tot < 60:
            next_state, reward, done, truncated, info = env.step(3)
            tot += reward
            total_reward += reward

        while tot < 100:
            next_state, reward, done, truncated, info = env.step(4)
            tot += reward
            total_reward += reward

        while tot < 140:
            next_state, reward, done, truncated, info = env.step(3)
            tot += reward
            total_reward += reward

        next_state, reward, done, truncated, info = env.step(1)
        total_reward += reward
        t = 0

        ###BURADA YUKARI ÇIKIP TÜNELDEN GEÇSE DAHA İYİ OLABİLİR

        while total_reward < 230:
            next_state, reward, done, truncated, info = env.step(2)  
            total_reward += reward
            print(total_reward)
            print(t)

        while(total_reward < 240):
            next_state, reward, done, truncated, info = env.step(1)  
            total_reward += reward
            print(total_reward)
            print(t)
            total_reward+=1

        prev = total_reward

        while(prev == total_reward):
            next_state, reward, done, truncated, info = env.step(2)  
            total_reward += reward


        ind = False


    return ind, total_reward
"""

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
            print("GYATT")

            k+=1
        
        if( k > 10):
            print("Dot fuck em up")
            select = False

        action = pretrain(total_reward , factor , previous_reward , select)
        
        if(action == 10):
            print("Gone")
            action = env.action_space.sample()

        next_observation, reward, done, _, _ = env.step(action)

        previous_reward = total_reward
        total_reward += reward


    print(f"Episode {t+1} finished with total reward: {total_reward}")

env.close()




"""
for t in range(episodes):
    state, info = env.reset()  

    done = False
    ind = True  
    start_position = None  

    while not done:
        truncated = False
        if ind:

            rand = random.randint(0,1)
            ind, total_reward = pretrain(total_reward,rand)
        else:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

          
        if done or truncated:
            ind = False
            break

    print(f"Episode {t+1} finished with total reward: {total_reward}")
    total_reward = 0  

env.close()

print("All episodes finished.")"""
