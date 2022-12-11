import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PosModel(nn.Module):
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
        self.out_features = out_features

        self.input = nn.Linear(in_features=self.in_features, out_features=32)

        self.fc1 = nn.Linear(in_features=32, out_features=64)

        self.fc2 = nn.Linear(in_features=64, out_features=128)

        self.out = nn.Linear(in_features=128, out_features=self.out_features)

    def forward(self, t) -> torch.Tensor:
        t = F.relu(self.input(t))
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class VisionModel3L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel3L, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class VisionModel4L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel4L, self).__init__()
        i = 0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)    
        self.bn4 = nn.BatchNorm2d(128)
        i+=4

        out_size = (w-i) * (h-i)*128 
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return self.head(x.view(x.size(0), -1))


class VisionModel10L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel10L, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        i+=2
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        i+=2
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        i+=2
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(32)
        i+=2

        out_size = (w-i) * (h-i)*32
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        
        return self.head(x.view(x.size(0), -1))

class VisionModel6L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel6L, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2

        out_size = (w-i) * (h-i)*64
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        return self.head(x.view(x.size(0), -1))


class VisionModel8L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel8L, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        i+=2
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        i+=2
        
        out_size = (w-i) * (h-i)*16
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        return self.head(x.view(x.size(0), -1))


class VisionModel13L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel13L, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        i+=2
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        i+=2
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        i+=2
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(32)
        i+=2
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.bn11 = nn.BatchNorm2d(64)
        i+=3
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.bn12 = nn.BatchNorm2d(128)
        i+=3
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.bn13 = nn.BatchNorm2d(128)
        i+=3
        out_size = (w-i) * (h-i)*128
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        return self.head(x.view(x.size(0), -1))


class VisionModel15L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel15L, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        i+=2
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        i+=2
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        i+=2
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(32)
        i+=2
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.bn11 = nn.BatchNorm2d(64)
        i+=3
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.bn12 = nn.BatchNorm2d(128)
        i+=3
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.bn13 = nn.BatchNorm2d(128)
        i+=3
        self.conv14 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn14 = nn.BatchNorm2d(64)
        i+=3
        self.conv15 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn15 = nn.BatchNorm2d(32)
        i+=3
        
        out_size = (w-i) * (h-i)*32
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        
        return self.head(x.view(x.size(0), -1))

class VisionModel15LDropout(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionModel15LDropout, self).__init__()
        i=0
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        i+=4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        i+=4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        i+=4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        i+=4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        i+=4
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        i+=2
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        i+=2
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        i+=2
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        i+=2
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(32)
        i+=2
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.bn11 = nn.BatchNorm2d(64)
        i+=3
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.bn12 = nn.BatchNorm2d(128)
        i+=3
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.bn13 = nn.BatchNorm2d(128)
        i+=3
        
        self.conv14 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn14 = nn.BatchNorm2d(64)
        i+=3
        self.conv15 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn15 = nn.BatchNorm2d(32)
        i+=3
        
        self.dropout = nn.Dropout(0.2)

        out_size = (w-i) * (h-i)*32
        self.head = nn.Linear(out_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))

        x = self.dropout(x)
        return self.head(x.view(x.size(0), -1))


def get_vision_model(model_name, h,w,outputs):
    model_dict = {
        "VisionModel3L": VisionModel3L,
        "VisionModel4L": VisionModel4L,
        "VisionModel6L": VisionModel6L,
        "VisionModel8L": VisionModel8L,
        "VisionModel10L": VisionModel10L,
        "VisionModel13L": VisionModel13L,
        "VisionModel15L": VisionModel15L,
        "VisionModel15LDropout": VisionModel15LDropout,
    }


    if model_name not in model_dict:
        raise ValueError("Baseline Model name does not exists")

    model = model_dict[model_name](h,w,outputs)
    
    return model