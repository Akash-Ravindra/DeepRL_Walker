from datetime import datetime
import gym
from agent import bipedal_walker_agent

from pathlib import Path

environment_name = "BipedalWalker-v3"
env = gym.make(environment_name)

walker = bipedal_walker_agent(env.observation_space.shape[0], env.action_space.shape[0], None)
episodes = 10

path = Path("/home/eurus/Documents/College/Sem_2/ENPM690/Finals/DoubleDQN/checkpoints/2022-04-30T02-45-08/agent_0.chkpt")
walker.load(path)
walker.exploration_rate = 0.0
for episode in range(1,episodes+1):
    state = env.reset()
    
    done = False
    score = 0
    while not done:
        env.render()
        action = walker.step(state)
        n_state,reward,done,info = env.step(action)
        state = n_state
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))
env.close()   

