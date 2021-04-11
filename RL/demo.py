import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(10):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print('observation:{}, reward:{}, done:{}, info:{}'.format(observation, reward, done, info))
env.close()