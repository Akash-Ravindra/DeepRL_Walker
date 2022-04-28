import torch
from torch import nn
import copy

class Qnet(nn.Module):
    def __init__(self,state_dim, action_dim) -> None:
        super().__init__()
        self.online = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),    
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        for i in self.target.parameters():
            i.requires_grad = False
            
            
    def forward(self, x, model='online'):
        if model == 'online':
            return self.online(x)
        else:
            return self.target(x)

