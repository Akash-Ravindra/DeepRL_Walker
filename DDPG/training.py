from datetime import datetime
import gym
from DDPG import Agent
import numpy as np
from tqdm.auto import tqdm, trange
import datetime
from pathlib import Path


import matplotlib.pyplot as plt


def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window) : (t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel("Total Score")
    plt.xlabel("Episode #")
    plt.plot(x, running_avg)
    plt.savefig(filename)


# environment_name = "BipedalWalker-v3"
environment_name = "Humanoid-v2"
# environment_name = "Walker2d-v2"
env = gym.make(environment_name)

save_dir = Path("local_checkpoint") / datetime.datetime.now().strftime(
    "%Y-%m-%dT%H-%M-%S"
)
save_dir.mkdir(parents=True)
score_history = []

# agent = Agent(alpha = 0.0001,beta= 0.001,gamma= 0.99, tau=1e-3, batch_size = 256,state_dim = env.observation_space.shape[0],action_dim= env.action_space.shape[0],save_dir=save_dir)
agent = Agent(
    alpha=0.0001,
    beta=0.001,
    gamma=0.99,
    tau=0.001,
    batch_size=256,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    save_dir=save_dir,
    layer1=800,
    layer2=200,
)
agent.burnin = 1e4
agent.save_every = 1e4

agent.load_network(Path("/home/eurus/Documents/College/Sem_2/ENPM690/Finals/DDPG/local_checkpoint/2022-05-04T18-29-53/"),8)

np.random.seed(0)
episodes = 50000
max_timeSteps = 1500

with trange(episodes) as t:
    for episode in t:
        state = env.reset()
        done = False
        score = 0
        time_step = 0
        while time_step < max_timeSteps:
            time_step += 1
            # action = agent.act(state,False)
            action = agent.act(state, False) + np.random.normal(
                0, 0.1, size=env.action_space.shape[0]
            )
            n_state, reward, done, info = env.step(action[0] * env.action_space.high)
            agent.remember(
                state=state, action=action, reward=reward, next_state=n_state, done=done
            )
            agent.learn()
            state = n_state
            score += reward
            if done:
                break
        score_history.append(score)
        t.set_postfix(
            str=f"Steps:{agent.curr_step} Episode: {episode} Score: {score} Score_avg {np.mean(score_history[-100:])}"
        )
        if episode % 1000 == 0:
            plotLearning(score_history, save_dir/ f"RewardGraph{episode//1000}", window=100)
        if episode% 2500 == 0:
            agent.save_network()
    env.close()
