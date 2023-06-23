import torch
import torch.nn as nn

# Based on https://colab.research.google.com/drive/1k2AfI6kPJmXK-gN6mi6_EegJQd4SybDz?usp=sharing#scrollTo=_0flZ1L6cWEL

class DeepSetLayer(nn.Module):
    # Converts a (batch, in_size, n) tensor to a (batch, out_size, n) tensor.
    # Replaces the DenseRagged layer from Ripsnet - note that the activation function needs
    # to be applied sequentially afterwards when building the neural net.
    
    def __init__(self, in_blocks, out_blocks, use_bias = True, **kwargs):
        super().__init__()
        
        self.in_blocks = in_blocks
        self.out_blocks = out_blocks
        self.use_bias = use_bias
        
        #Initialization trick from nn.linear
        lim = (in_blocks) ** -0.5 / 2
        
        self.alpha = torch.nn.Parameter(data=rand((out_blocks, in_blocks), -lim, lim))
        self.beta = torch.nn.Parameter(data=rand((out_blocks, in_blocks), -lim, lim))
        if self.use_bias:
            self.gamma = torch.nn.Parameter(data=rand((out_blocks), -lim, lim))
            
    def forward(self, x):
        if self.use_bias:
            return (
                torch.einsum('...jz, ij -> ...iz', x, self.alpha)
                + torch.einsum('...jz, ij -> ...iz', x.sum(axis=-1)[..., None], self.beta)
                + self.gamma[..., None]
            )
        else:
            return (
                torch.einsum('...jz, ij -> ...iz', x, self.alpha)
                + torch.einsum('...jz, ij -> ...iz', x.sum(axis=-1)[..., None], self.beta))
        
class DeepSetSum(nn.Module):
    # Reduces a (batch, blocks, n) tensor to a regular layer of shape (batch, blocks) via projection of 
    # a direct sum of trivial representations (last tensor dimension.)
    
    def __init__(self, blocks, **kwargs):
        super().__init__()
        
        lim = (in_blocks) ** -0.5 / 2
        self.weight = torch.nn.Parameter(data=rand(blocks, -lim, lim))
        self.bias = torch.nn.Parameter(data=rand(blocks, -lim, lim))
        
    def forward(self, x):
        return x.sum(dim = -1) * self.weight + self.bias
    
    