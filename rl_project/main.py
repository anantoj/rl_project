from rl_project.trainer import Trainer
from torchvision.models import resnet18, efficientnet_b0

# trainer = Trainer(model=resnet18(), mode="img")
trainer = Trainer(model=efficientnet_b0(), mode="img")
trainer.train()