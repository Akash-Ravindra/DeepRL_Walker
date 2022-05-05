from datetime import datetime
import gym
from DDPG import Agent

from pathlib import Path


# environment_name = "BipedalWalker-v3"
# environment_name = "Humanoid-v2"
environment_name = "Walker2d-v2"
env = gym.make(environment_name)

# env = gym.wrappers.Monitor(env, Path("/home/eurus/Documents/College/Sem_2/ENPM690/Finals/DDPG/Video/BipedalWalker-v3/"), force=True)

save_dir = None

# agent = Agent(alpha = 0.0001,beta= 0.001,gamma= 0.99, tau=1e-3, batch_size = 256,state_dim = env.observation_space.shape[0],action_dim= env.action_space.shape[0],save_dir=save_dir)
agent = Agent(alpha = 0.0001,beta= 0.001,gamma= 0.99, tau=0.001,batch_size = 256,state_dim = env.observation_space.shape[0],action_dim= env.action_space.shape[0],save_dir=save_dir,layer1 = 800,layer2 = 200)
agent.burnin =  5e5
episodes = 10

path = Path("/home/eurus/Documents/College/Sem_2/ENPM690/Finals/DDPG/local_checkpoint/Walker2d-v2(orginal)/")
agent.actor.load_network(path,24)

for episode in range(1,episodes+1):
    state = env.reset()    
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.act(state,noise = False)
        n_state,reward,done,info = env.step(action[0])
        state = n_state
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))
env.close()   

