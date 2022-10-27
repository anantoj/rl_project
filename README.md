# Deep Q-Learning Benchmark Trainer

This library is developed as part of my Reinforcement Learning project to evaluate performances of image models in various Deep Q-Learning tasks.
It includes a trainer class that can easily train an image model through an OpenAI Gym environment.
## Installation

```bash
git clone https://github.com/anantoj/rl_project.git
cd rl_project && pip install -e .
```

## How to Use

```py
from rl_project.trainer import Trainer

trainer.train()
```

For image models:
```py
from torchvision.models import resnet18
from rl_project.trainer import Trainer

trainer.train(model=resnet18())
```


## Strategy
The training strategy

## Supported environments
Currently the only supported environments are `CartPole-v1`, `MountainCar-v0`, and `Acrobot-v1`. 
Other environments (even outside of classic control) with `Discrete` inputs should theoritically not cause issues for the trainer, but are disabled for now since they are not yet tested.
You can fork the repo and remove the environment constraint on the `EnvManager` class in `utils.py`, but you have been warned!

## References
The training method implemented in this library is inspired from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html