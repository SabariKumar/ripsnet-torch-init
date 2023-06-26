import torch
import torch.nn as nn
from utils import DeepSetLayer, DeepSetSum

class RipsNet(nn.Module):
    def __init__(self, out_shape = 2500, use_bias = True):
        # out_shape should match the final dimensions of x (ie., the number of pixels in the persistence image)
        super().__init__()
        self.use_bias = use_bias
        self.out_shape = out_shape
        self.rips_net = nn.Sequential(
            DeepSetLayer(600, 30),
            nn.ReLU(),
            DeepSetLayer(30, 20),
            nn.ReLU(),
            DeepSetLayer(20, 10),
            nn.ReLU(),
            DeepSetSum(10),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, self.out_shape),
            nn.Sigmoid())
    
    def forward(self, x):
        out = self.rips_net(x)
        return out
    