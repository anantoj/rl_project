import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, in_features, out_features):
        """Initialize baseline DQN model architecture

        Parameters
        ----------
        in_features : int
            Number of input features 
        out_features : int
            Number of output features corresponding to the environment action space
        """
        super().__init__()
        self.in_features = in_features
        self.out_features= out_features
        
        self.input = nn.Linear(in_features=self.in_features,
                             out_features=32)

        self.fc1 = nn.Linear(in_features=32,
                             out_features=64)
        
        self.fc2 = nn.Linear(in_features=64,
                             out_features=128)
        
        self.out = nn.Linear(in_features=128,
                             out_features=self.out_features) 

    def forward(self, t) -> torch.Tensor:
        t = F.relu(self.input(t))
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t