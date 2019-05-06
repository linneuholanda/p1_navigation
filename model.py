import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Input
        =====
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc_units (list): Number of nodes in each layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        fc_units = [state_size] + fc_units + [action_size]
        self.layers = [] 
        for i in range(len(fc_units)-1):
            self.layers.append(nn.Linear(fc_units[i],fc_units[i+1]))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, state):
        """Build a network that maps state -> action values.
        Input
        =====
        state (array-like): Current state.
        """
        x = state
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return x
    
