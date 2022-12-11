from rl_project.trainer import Trainer
from torchvision.models import efficientnet_b0
import sys

# cartpole_trainer = Trainer(mode="pos")
# cartpole_trainer.train()

# mountaincar_trainer = Trainer(mode="pos", env="MountainCar-v0", target_reward=-110)
# mountaincar_trainer.train()

# acrobot_trainer = Trainer(mode="pos", env="Acrobot-v1", target_reward=-110)
# acrobot_trainer.train()

# acrobot_trainer = Trainer(mode="img", model=efficientnet_b0())
# acrobot_trainer.train()

trainer = Trainer(mode="img", env="CartPole-v1", num_streaks=20, reset_freq=50,reset_percent=0.05, max_timestep=sys.maxsize, baseline_vision_model = "VisionModel15LDropout",num_episodes=2000)
trainer.train()
