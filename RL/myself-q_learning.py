import time
import gym
import numpy as np 

env = gym.make('FrozenLake-v0',is_slippery=True)##配置冰面是否光滑

#render = False  
running_reward = None 

Q = np.zeros([env.observation_space.n, env.action_space.n])

##Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
lr = .85
discount = .99
num_episodes = 500
rList = [] 
rAll = 0 
for i in range(num_episodes):
    s = env.reset()
    episode_time = time.time() 
    rAll = 0
    for j in range(100):
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        s1,r,d,_ = env.step(a)
        Q[s,a] = Q[s,a] + lr *(r+discount* np.max(Q[s1,:]-Q[s,a]))
        s = s1 
        rAll += r 
        #print (rAll)
        if d == True:
            break
    rList.append(rAll)
    #print(rList)
    running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
    print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
        (i, num_episodes, rAll, running_reward, time.time() - episode_time))
print(Q)

s=env.reset()
env.render()
for j in range(100):
    a=np.argmax(Q[s,:])
    s1,r,d,_ = env.step(a)
    env.render()
    s = s1
    if  d == True:
        break







