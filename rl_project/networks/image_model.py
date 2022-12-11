import torch
import torch.nn as nn

class VisionModel(nn.Module):
    def __init__(self, image_model, in_features=None, out_features=None, mode="pos"):
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
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        if mode == "pos":
            self.input = nn.Linear(
                in_features=self.in_features, out_features=1 * 3 * 64 * 64
            )

            if self.image_model.__class__.__name__ == "ResNet":
                self.image_model.fc = nn.Linear(
                    self.image_model.fc.in_features, self.out_features
                )
            elif self.image_model.__class__.__name__ == "SwinTransformer":
                self.image_model.head = nn.Linear(
                    self.image_model.head.in_features, self.out_features
                )
            else:
                self.image_model.classifier[
                    len(self.image_model.classifier) - 1
                ] = nn.Linear(
                    self.image_model.classifier[
                        len(self.image_model.classifier) - 1
                    ].in_features,
                    self.out_features,
                )

        elif mode == "img":
            if self.image_model.__class__.__name__ == "ResNet":
                self.image_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False)    
                self.image_model.fc = nn.Linear(
                    self.image_model.fc.in_features, self.out_features
                )
            else:
                # model = torchvision.models.efficientnet_b0()
                first_conv_layer = [nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
                first_conv_layer.extend(list(self.image_model.features))
                self.image_model.features = nn.Sequential(*first_conv_layer )

                self.image_model(torch.randn(1, 6, 60, 135))

                self.image_model.classifier[
                    len(self.image_model.classifier) - 1
                ] = nn.Linear(
                    image_model.classifier[
                        len(image_model.classifier) - 1
                    ].in_features,
                    self.out_features,
                )
                self.image_model(torch.randn(1,6,60,135))
                

    def forward(self, t) -> torch.Tensor:
        if self.mode == "pos":
            t = self.input(t)
            if len(t.shape) > 1:
                # Replay Memory sample, batch_size > 1
                t = torch.reshape(t, (t.shape[0], 3, 64, 64))
            else:
                t = torch.reshape(t, (1, 3, 64, 64))

            t = self.image_model(t)
            return torch.squeeze(t, 0)

        elif self.mode == "img":
            return self.image_model(t)
