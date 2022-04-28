import torch
import random, numpy as np
from pathlib import Path

from Qnetwork import Qnet
from collections import deque


class bipedal_walker_agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        ## Neural network to approximate Q-function
        self.net = Qnet(self.state_dim, self.action_dim).float().to(self.device)

        if checkpoint:
            self.load(checkpoint)
            
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0002)

        self.loss_fn = torch.nn.SmoothL1Loss()

    def step(self, state):
        ## random check to see explore or exploit
        if np.random.rand() < self.exploration_rate:
            return np.random.rand(self.action_dim)
        else:
            state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.net(state, model="online").flatten().cpu().data

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_values

        pass

    def predict(self, state):
        pass

    def update(self):
        if self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.replay_buffer_recall()

        
        q_eval = self.net(state, model="online")
        
        with torch.no_grad():
            q_next = self.net(next_state, model="target")
        
        q_target = reward[:,None] + self.gamma*(q_next)*(1-done.float())[:,None]

        

        # Get TD Estimate
        
        
        # Backpropagate loss through Q_online
        loss = self.update_Q_online(q_eval, q_target)

        return (loss)

    def update_Q_online(self, td_eval, td_target) :
        loss = self.loss_fn(td_eval, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def td_estimate(self, state, action):
        current_Q = self.net(state)# Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_Q = self.net(next_state, model="target")
        
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def replay_buffer_save(self, state, action, reward, next_state, done):

        ## Add experience to memory
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
            torch.DoubleTensor(np.array([reward]))
            if not self.use_cuda
            else torch.DoubleTensor(np.array([reward])).cuda()
        )
        next_state = (
            torch.FloatTensor(np.array([next_state]))
            if not self.use_cuda
            else torch.FloatTensor(np.array([next_state])).cuda()
        )
        done = (
            torch.BoolTensor(np.array([done]))
            if not self.use_cuda
            else torch.BoolTensor(np.array([done])).cuda()
        )

        self.memory.append((state, action, reward, next_state, done))
        pass

    def replay_buffer_recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return (
            state.to(self.device),
            next_state.to(self.device),
            action.squeeze().to(self.device),
            reward.squeeze().to(self.device),
            done.squeeze().to(self.device),
        )

    def save(self):
        save_path = self.save_dir / f"agent_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"Agent saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=("cuda" if self.use_cuda else "cpu"))
        exploration_rate = ckp.get("exploration_rate")
        state_dict = ckp.get("model")

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
