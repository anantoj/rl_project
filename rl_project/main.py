from rl_project.trainer import Trainer
from torchvision.models import resnet18

trainer = Trainer(model=resnet18(), mode="img")
trainer.train()