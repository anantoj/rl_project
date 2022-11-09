from rl_project.trainer import Trainer
from torchvision.models import resnet18

trainer = Trainer(mode="img")
trainer.train()