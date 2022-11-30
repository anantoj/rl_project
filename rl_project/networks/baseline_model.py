import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselinePosModel(nn.Module):
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


class BaselineVisionModel(nn.Module):
    def __init__(self, h , w, outputs):
        super(BaselineVisionModel, self).__init__()
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


class VisionExpand3L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand3L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
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


class VisionExpand4L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand4L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

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
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))

class VisionExpand5L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand5L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)

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
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return self.head(x.view(x.size(0), -1))

class VisionExpand9L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand9L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)

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
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        return self.head(x.view(x.size(0), -1))

class VisionExpand13L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand13L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn13 = nn.BatchNorm2d(128)

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


class VisionExpand21L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand21L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn13 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn15 = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv17 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn17 = nn.BatchNorm2d(16)
        self.conv18 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn18 = nn.BatchNorm2d(16)
        self.conv19 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn21 = nn.BatchNorm2d(128)

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
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        return self.head(x.view(x.size(0), -1))


class VisionExpand37L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand37L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn13 = nn.BatchNorm2d(128)
        
        self.conv14 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn15 = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv17 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn17 = nn.BatchNorm2d(16)
        self.conv18 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn18 = nn.BatchNorm2d(16)
        self.conv19 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn21 = nn.BatchNorm2d(128)

        self.conv22 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn23 = nn.BatchNorm2d(32)
        self.conv24 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn24 = nn.BatchNorm2d(16)
        self.conv25 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn25 = nn.BatchNorm2d(16)
        self.conv26 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn26 = nn.BatchNorm2d(16)
        self.conv27 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn27 = nn.BatchNorm2d(64)
        self.conv28 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn28 = nn.BatchNorm2d(128)
        self.conv29 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn29 = nn.BatchNorm2d(128)
        
        self.conv30 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn30 = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn31 = nn.BatchNorm2d(32)
        self.conv32 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn32 = nn.BatchNorm2d(16)
        self.conv33 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn33 = nn.BatchNorm2d(16)
        self.conv34 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn34 = nn.BatchNorm2d(16)
        self.conv35 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn35 = nn.BatchNorm2d(64)
        self.conv36 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn36 = nn.BatchNorm2d(128)
        self.conv37 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn37 = nn.BatchNorm2d(128)

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
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))

        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn24(self.conv24(x)))
        x = F.relu(self.bn25(self.conv25(x)))
        x = F.relu(self.bn26(self.conv26(x)))
        x = F.relu(self.bn27(self.conv27(x)))
        x = F.relu(self.bn28(self.conv28(x)))
        x = F.relu(self.bn29(self.conv29(x)))
        x = F.relu(self.bn30(self.conv30(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = F.relu(self.bn34(self.conv34(x)))
        x = F.relu(self.bn35(self.conv35(x)))
        x = F.relu(self.bn36(self.conv36(x)))
        x = F.relu(self.bn37(self.conv37(x)))
        return self.head(x.view(x.size(0), -1))


class VisionExpand69L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand69L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(16)
        self.conv10 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn10 = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn13 = nn.BatchNorm2d(128)
        
        self.conv14 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn15 = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn16 = nn.BatchNorm2d(16)
        self.conv17 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn17 = nn.BatchNorm2d(16)
        self.conv18 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn18 = nn.BatchNorm2d(16)
        self.conv19 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn21 = nn.BatchNorm2d(128)

        self.conv22 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn23 = nn.BatchNorm2d(32)
        self.conv24 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn24 = nn.BatchNorm2d(16)
        self.conv25 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn25 = nn.BatchNorm2d(16)
        self.conv26 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn26 = nn.BatchNorm2d(16)
        self.conv27 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn27 = nn.BatchNorm2d(64)
        self.conv28 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn28 = nn.BatchNorm2d(128)
        self.conv29 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn29 = nn.BatchNorm2d(128)
        
        self.conv30 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn30 = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn31 = nn.BatchNorm2d(32)
        self.conv32 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn32 = nn.BatchNorm2d(16)
        self.conv33 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn33 = nn.BatchNorm2d(16)
        self.conv34 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn34 = nn.BatchNorm2d(16)
        self.conv35 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn35 = nn.BatchNorm2d(64)
        self.conv36 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn36 = nn.BatchNorm2d(128)
        self.conv37 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn37 = nn.BatchNorm2d(128)


        self.conv38 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn38 = nn.BatchNorm2d(64)
        self.conv39 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn39 = nn.BatchNorm2d(32)
        self.conv40 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn40 = nn.BatchNorm2d(16)
        self.conv41 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn41 = nn.BatchNorm2d(16)
        self.conv42 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn42 = nn.BatchNorm2d(16)
        self.conv43 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn43 = nn.BatchNorm2d(64)
        self.conv44 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn44 = nn.BatchNorm2d(128)
        self.conv45 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn45 = nn.BatchNorm2d(128)
        
        self.conv46 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn46 = nn.BatchNorm2d(64)
        self.conv47 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn47 = nn.BatchNorm2d(32)
        self.conv48 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn48 = nn.BatchNorm2d(16)
        self.conv49 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn49 = nn.BatchNorm2d(16)
        self.conv50 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn50 = nn.BatchNorm2d(16)
        self.conv51 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn51 = nn.BatchNorm2d(64)
        self.conv52 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn52 = nn.BatchNorm2d(128)
        self.conv53 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn53 = nn.BatchNorm2d(128)

        self.conv54 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn54 = nn.BatchNorm2d(64)
        self.conv55 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn55 = nn.BatchNorm2d(32)
        self.conv56 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn56 = nn.BatchNorm2d(16)
        self.conv57 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn57 = nn.BatchNorm2d(16)
        self.conv58 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn58 = nn.BatchNorm2d(16)
        self.conv59 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn59 = nn.BatchNorm2d(64)
        self.conv60 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn60 = nn.BatchNorm2d(128)
        self.conv61 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.bn61 = nn.BatchNorm2d(128)
        
        self.conv62 = nn.Conv2d(128, 64, kernel_size=4, stride=1)
        self.bn62 = nn.BatchNorm2d(64)
        self.conv63 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.bn63 = nn.BatchNorm2d(32)
        self.conv64 = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        self.bn64 = nn.BatchNorm2d(16)
        self.conv65 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn65 = nn.BatchNorm2d(16)
        self.conv66 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.bn66 = nn.BatchNorm2d(16)
        self.conv67 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn67 = nn.BatchNorm2d(64)
        self.conv68 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn68 = nn.BatchNorm2d(128)
        self.conv69 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.bn69 = nn.BatchNorm2d(128)

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
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn24(self.conv24(x)))
        x = F.relu(self.bn25(self.conv25(x)))
        x = F.relu(self.bn26(self.conv26(x)))
        x = F.relu(self.bn27(self.conv27(x)))
        x = F.relu(self.bn28(self.conv28(x)))
        x = F.relu(self.bn29(self.conv29(x)))
        x = F.relu(self.bn30(self.conv30(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = F.relu(self.bn34(self.conv34(x)))
        x = F.relu(self.bn35(self.conv35(x)))
        x = F.relu(self.bn36(self.conv36(x)))
        x = F.relu(self.bn37(self.conv37(x)))
        x = F.relu(self.bn38(self.conv38(x)))
        x = F.relu(self.bn39(self.conv39(x)))
        x = F.relu(self.bn40(self.conv40(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x = F.relu(self.bn44(self.conv44(x)))
        x = F.relu(self.bn45(self.conv45(x)))
        x = F.relu(self.bn46(self.conv46(x)))
        x = F.relu(self.bn47(self.conv47(x)))
        x = F.relu(self.bn48(self.conv48(x)))
        x = F.relu(self.bn49(self.conv49(x)))
        x = F.relu(self.bn50(self.conv50(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x = F.relu(self.bn54(self.conv54(x)))
        x = F.relu(self.bn55(self.conv55(x)))
        x = F.relu(self.bn56(self.conv56(x)))
        x = F.relu(self.bn57(self.conv57(x)))
        x = F.relu(self.bn58(self.conv58(x)))
        x = F.relu(self.bn59(self.conv59(x)))
        x = F.relu(self.bn60(self.conv60(x)))
        x = F.relu(self.bn61(self.conv61(x)))
        x = F.relu(self.bn62(self.conv62(x)))
        x = F.relu(self.bn63(self.conv63(x)))
        x = F.relu(self.bn64(self.conv64(x)))
        x = F.relu(self.bn65(self.conv65(x)))
        x = F.relu(self.bn66(self.conv66(x)))
        x = F.relu(self.bn67(self.conv67(x)))
        x = F.relu(self.bn68(self.conv68(x)))
        x = F.relu(self.bn69(self.conv69(x)))
        return self.head(x.view(x.size(0), -1))

class VisionExpand6L(nn.Module):
    def __init__(self, h , w, outputs):
        super(VisionExpand6L, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        

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
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        return self.head(x.view(x.size(0), -1))


class BaselineVisionModel6L(nn.Module):
    def __init__(self, h , w, outputs):
        super(BaselineVisionModel6L, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
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
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        return self.head(x.view(x.size(0), -1))

def get_model(model_name, h,w,outputs):
    model_dict = {
        "BaselineVisionModel" : BaselineVisionModel,
        "VisionExpand5L" : VisionExpand5L,
        "BaselineVisionModel6L" : BaselineVisionModel6L,
        "VisionExpand4L": VisionExpand4L,
        "VisionExpand3L": VisionExpand3L,
        "VisionExpand9L": VisionExpand9L,
        "VisionExpand6L": VisionExpand6L,
        "VisionExpand13L": VisionExpand13L,
        "VisionExpand21L": VisionExpand21L,
        "VisionExpand37L": VisionExpand37L,
        "VisionExpand69L": VisionExpand69L,
    }


    if model_name not in model_dict:
        raise ValueError("Baseline Model name does not exists")

    model = model_dict["BaselineVisionModel"](h,w,outputs)
    
    return model