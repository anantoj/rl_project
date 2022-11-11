from rl_project.trainer import Trainer

# cartpole_trainer = Trainer(mode="pos")
# cartpole_trainer.train()

mountaincar_trainer = Trainer(mode="pos", env="Acrobot-v1", target_reward=-110)
mountaincar_trainer.train()
