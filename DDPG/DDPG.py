import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from collections import deque
import random
from pathlib import Path


class ActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros_like(mu)
        self.reset()

    def __call__(self):
        self.x_prev += self.theta * (self.mu - self.x_prev) + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(size=self.mu.shape)
        return self.x_prev

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


class ReplayBuffer:
    def __init__(self, maxlen=100000, batch_size=32, use_cuda=True) -> None:
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        pass

    def save_experience(self, state, action, reward, next_state, done):
        state = (
            torch.FloatTensor(state)
            if not self.use_cuda
            else torch.FloatTensor(state).cuda()
        )
        action = (
            torch.FloatTensor(action)
            if not self.use_cuda
            else torch.FloatTensor(action).cuda()
        )
        reward = (
            torch.FloatTensor(np.array([reward]))
            if not self.use_cuda
            else torch.FloatTensor(np.array([reward])).cuda()
        )
        next_state = (
            torch.FloatTensor(np.array(next_state))
            if not self.use_cuda
            else torch.FloatTensor(np.array(next_state)).cuda()
        )
        done = (
            torch.FloatTensor(np.array([1 - np.float32(done)]))
            if not self.use_cuda
            else torch.FloatTensor(np.array([1 - np.float32(done)])).cuda()
        )

        self.buffer.append((state, action, reward, next_state, done))

    def get_sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return (
            state,
            next_state,
            action.squeeze(),
            reward.squeeze(),
            done.squeeze(),
        )


class Critic(nn.Module):
    # def __init__(
    #     self,
    #     beta,
    #     state_dim,
    #     action_dim,
    #     name,
    #     layer1_size=400,
    #     layer2_size=300,
    #     save_dir=None,
    # ):
    #     super(Critic, self).__init__()
    #     self.use_cuda = torch.cuda.is_available()
    #     self.device = "cuda" if self.use_cuda else "cpu"
    #     self.checkpoint_dir = save_dir
    #     self.name = name

    #     self.layer1 = nn.Linear(state_dim, layer1_size)
    #     f1 = 1 / np.sqrt(self.layer1.weight.data.size()[0])
    #     torch.nn.init.uniform_(self.layer1.weight.data, -f1, f1)
    #     torch.nn.init.uniform_(self.layer1.bias.data, -f1, f1)
    #     self.bn1 = nn.LayerNorm(layer1_size)

    #     self.layer2 = nn.Linear(layer1_size, layer2_size)
    #     f2 = 1 / np.sqrt(self.layer2.weight.data.size()[0])
    #     torch.nn.init.uniform_(self.layer2.weight.data, -f2, f2)
    #     torch.nn.init.uniform_(self.layer2.bias.data, -f2, f2)
    #     self.bn2 = nn.LayerNorm(layer2_size)

    #     self.action_layer = nn.Linear(action_dim, layer2_size)
    #     f3 = 0.003
    #     self.value_func = nn.Linear(layer2_size, 1)
    #     torch.nn.init.uniform_(self.value_func.weight.data, -f3, f3)
    #     torch.nn.init.uniform_(self.value_func.bias.data, -f3, f3)

    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
    #     self.to(self.device)
    #     pass

    # def forward(self, state, action):
    #     state_value = self.layer1(state)
    #     state_value = self.bn1(state_value)
    #     state_value = f.relu(state_value)

    #     state_value = self.layer2(state_value)
    #     state_value = self.bn2(state_value)
    #     state_value = f.relu(state_value)

    #     action_value = f.relu(self.action_layer(action))
    #     value = self.value_func(f.relu(torch.add(state_value, action_value)))
    #     return value

    def __init__(
        self,
        beta,
        state_dim,
        action_dim,
        name,
        layer1_size=400,
        layer2_size=300,
        save_dir=None,
    ):
        super(Critic, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.checkpoint_dir = save_dir
        self.name = name

        self.l1 = torch.nn.Linear(state_dim + action_dim, layer1_size)
        self.l2 = torch.nn.Linear(layer1_size, layer2_size)
        self.l3 = torch.nn.Linear(layer2_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)

    def forward(self, state, action):
        q = f.relu(self.l1(torch.cat([state, action], 1)))
        q = f.relu(self.l2(q))
        return self.l3(q)

    
    def save_network(self,idx):
        torch.save(self.state_dict(), self.checkpoint_dir / f"{self.name}{idx}.ckpt")

    def load_network(self, load_dir,idx = None):
        self.load_state_dict(torch.load(load_dir / f"{self.name}{idx}.ckpt"))


class Actor(nn.Module):
    # def __init__(
    #     self,
    #     alpha,
    #     state_dim,
    #     action_dim,
    #     name,
    #     layer1_size=400,
    #     layer2_size=300,
    #     save_dir=None,
    # ):
    #     super(Actor, self).__init__()
    #     self.use_cuda = torch.cuda.is_available()
    #     self.device = "cuda" if self.use_cuda else "cpu"
    #     self.checkpoint_dir = save_dir
    #     self.name = name

    #     self.layer1 = nn.Linear(state_dim, layer1_size)
    #     f1 = 1 / np.sqrt(self.layer1.weight.data.size()[0])
    #     torch.nn.init.uniform_(self.layer1.weight.data, -f1, f1)
    #     torch.nn.init.uniform_(self.layer1.bias.data, -f1, f1)
    #     self.bn1 = nn.LayerNorm(layer1_size)

    #     self.layer2 = nn.Linear(layer1_size, layer2_size)
    #     f2 = 1 / np.sqrt(self.layer2.weight.data.size()[0])
    #     torch.nn.init.uniform_(self.layer2.weight.data, -f2, f2)
    #     torch.nn.init.uniform_(self.layer2.bias.data, -f2, f2)
    #     self.bn2 = nn.LayerNorm(layer2_size)

    #     f3 = 0.003
    #     self.actions = nn.Linear(layer2_size, action_dim)
    #     torch.nn.init.uniform_(self.actions.weight.data, -f3, f3)
    #     torch.nn.init.uniform_(self.actions.bias.data, -f3, f3)

    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    #     self.to(self.device)
    #     pass

    # def forward(self, state):
    #     x = self.layer1(state)
    #     x = self.bn1(x)
    #     x = f.relu(x)

    #     x = self.layer2(x)
    #     x = self.bn2(x)
    #     x = f.relu(x)

    #     action = torch.tanh(self.actions(x))
    #     return action


    def __init__(
        self,
        alpha,
        state_dim,
        action_dim,
        name,
        layer1_size=400,
        layer2_size=300,
        save_dir=None,
        action_high=1,
    ):
        super(Actor, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.checkpoint_dir = save_dir
        self.name = name
        # self.action_high = action_high

        self.l1 = torch.nn.Linear(state_dim, layer1_size)
        self.l2 = torch.nn.Linear(layer1_size, layer2_size)
        self.l3 = torch.nn.Linear(layer2_size, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        a = f.relu(self.l1(state))
        a = f.relu(self.l2(a))
        return torch.tanh(self.l3(a))

    def save_network(self, idx):
        torch.save(self.state_dict(), self.checkpoint_dir / f"{self.name}{idx}.ckpt")

    def load_network(self, load_dir,idx = None):
        self.load_state_dict(torch.load(load_dir / f"{self.name}{idx}.ckpt"))
        print("Finished loading network")


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        state_dim,
        save_dir,
        tau=0.001,
        gamma=0.95,
        action_dim=4,
        max_len=1000000,
        layer1=400,
        layer2=300,
        batch_size=64,
        action_high = 1
    ):
        self.gamma = gamma
        self.tau = tau
        self.burnin = 1e4
        self.curr_step = 0
        self.learn_every = 5
        self.save_ever = 1e5

        self.memory = ReplayBuffer(maxlen=max_len, batch_size=batch_size, use_cuda=True)
        self.save_dir = save_dir
        self.actor = Actor(
            alpha, state_dim, action_dim, "actor", save_dir=self.save_dir,layer1_size=layer1,layer2_size=layer2
        )
        self.critic = Critic(
            beta, state_dim, action_dim, "critic", save_dir=self.save_dir,layer1_size=layer1,layer2_size=layer2
        )
        self.target_actor = Actor(
            alpha, state_dim, action_dim, "target_actor", save_dir=self.save_dir,layer1_size=layer1,layer2_size=layer2
        )
        self.target_critic = Critic(
            beta, state_dim, action_dim, "target_critic", save_dir=self.save_dir,layer1_size=layer1,layer2_size=layer2
        )

        self.noise = ActionNoise(mu=np.zeros(action_dim))

        # self.update_network(tau=1)
        pass

    def act(self, state, noise=True):
        self.actor.eval()
        state = torch.FloatTensor(state).cuda().unsqueeze(0)
        action = self.actor(state)
        action = action.cpu().data.numpy()
        if noise:
            action += self.noise()
        self.actor.train()
        self.curr_step += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.save_experience(state, action, reward, next_state, done)
        pass

    def learn(self):
        if self.curr_step < self.burnin:
            return
        if self.curr_step % self.save_ever == 0:
            self.save_network()
        if self.curr_step % self.learn_every != 0:
            return

        state, next_state, action, reward, done = self.memory.get_sample()

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_action = self.target_actor(next_state)
        target_valuefunc = self.target_critic(next_state, target_action)
        valuefunc = self.critic(state, action)

        target = reward.squeeze() + (
            target_valuefunc.squeeze() * done.squeeze() * self.gamma
        )

        target = target.unsqueeze(1).float()

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = f.mse_loss(valuefunc, target)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        ac = self.actor(state)
        self.actor.train()
        ac_loss = -self.critic(state, ac).mean()
        ac_loss.backward()
        self.actor.optimizer.step()

        self.update_network()

    def update_network(self, tau=None):
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )
        self.target_actor.load_state_dict(actor_state_dict)

    def save_network(self):
        idx = int(self.curr_step//self.save_ever)
        self.actor.save_network(idx)
        self.critic.save_network(idx)
        self.target_actor.save_network(idx)
        self.target_critic.save_network(idx)

    def load_network(self, load_dir,idx = None):
        self.actor.load_network(load_dir,idx=idx)
        self.critic.load_network(load_dir,idx=idx)
        self.target_actor.load_network(load_dir,idx=idx)
        self.target_critic.load_network(load_dir,idx=idx)
