from datetime import datetime
import gym
from agent import bipedal_walker_agent

from pathlib import Path

environment_name = "BipedalWalker-v3"
env = gym.make(environment_name)

# episodes = 10

# for episode in range(1,episodes+1):
#     state = env.reset()
    
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action =env.action_space.sample()
#         n_state,reward,done,info = env.step(action)
#         # next_state, reward, done, info = env.step(action=0)
#         # print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode,score))
# env.close()   


episodes = 50000


save_dir = Path('checkpoints')/datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir.mkdir(parents=True)

agent = bipedal_walker_agent(env.observation_space.shape[0], env.action_space.shape[0], save_dir)

# Write a logger to save the training process

for episode in range(1,episodes+1):
    state = env.reset()
    
    done = False
    score = 0
    while not done:
        
        action = agent.step(state)
        
        
        n_state,reward,done,info = env.step(action)
        
        agent.replay_buffer_save(state, action, reward, n_state, done)
        
        _,_ = agent.update()
        
        # next_state, reward, done, info = env.step(action=0)
        
        # print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))
env.close()    