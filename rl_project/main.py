from rl_project.trainer import Trainer

# cartpole_trainer = Trainer(mode="pos")
# cartpole_trainer.train()

# mountaincar_trainer = Trainer(mode="pos", env="MountainCar-v0", target_reward=-110)
# mountaincar_trainer.train()

acrobot_trainer = Trainer(mode="pos", env="Acrobot-v1", target_reward=-110)
acrobot_trainer.train()


