import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionModel(nn.Module):
    def __init__(self, in_features, out_features, image_model):
        """Initialize baseline DQN model architecture

        Parameters
        ----------
        in_features : int
            Number of input features 
        out_features : int
            Number of output features corresponding to the environment action space
        """
        super().__init__()
        self.image_model = image_model
        
        self.in_features = in_features
        self.out_features= out_features
        
        self.input = nn.Linear(in_features=self.in_features,
                             out_features=1*3*64*64)

        if self.image_model.__class__.__name__ == "ResNet":
            self.image_model.fc = nn.Linear(self.image_model.fc.in_features, self.out_features)
        elif self.image_model.__class__.__name__ == "SwinTransformer":
            self.image_model.head = nn.Linear(self.image_model.head.in_features,self.out_features)
        else:
            self.image_model.classifier[len(self.image_model.classifier)-1] = nn.Linear(self.image_model.classifier[len(self.image_model.classifier)-1].in_features, self.out_features)
    

    def forward(self, t) -> torch.Tensor:
        t = self.input(t)
        if len(t.shape) > 1:
            # Replay Memory sample, batch_size > 1
            t = torch.reshape(t,(t.shape[0],3,64,64))
        else:
            t = torch.reshape(t,(1,3,64,64))
        t = self.image_model(t)
        return torch.squeeze(t, 0)