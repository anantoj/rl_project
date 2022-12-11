# Deep Q-Learning Vision Model Trainer

This library is a Deep Q-Learning trainer, specialized for vision models, in various OpenAI Gym environments. It is developed as part of my Reinforcement Learning project to evaluate performances of image models in various Deep Q-Learning tasks and test out Neurogenesis.

## Installation

```bash
git clone https://github.com/anantoj/rl_project.git
cd rl_project && pip install -e .
```

## How to Use

There are two modes of training: `mode="pos"` and `mode="img"`. The `pos` mode will train your model using positioning or built-in features return from the OpenAI Gym environment. In contrast, the `img` mode will utilize the environment's rendered screen (as RGB array) as inputs to the models.

To train on position features:

```py
from rl_project.trainer import Trainer

trainer = Trainer(mode="pos")
trainer.train()
```

To train on image features:

```py
from rl_project.trainer import Trainer

trainer = Trainer(mode="img", baseline_vision_model = "VisionModel3L")
trainer.train()
```

Use the `reset_freq` and `reset_percent` arguments for Neurogenesis.
```py
trainer = Trainer(mode="img", reset_freq=50,reset_percent=0.05, baseline_vision_model = "VisionModel3L")
```
The code above will apply neurogenesis every 50 episodes by reseting 5% of random weights in the network.

If you intend to use this library in Google Colab or Kaggle, import like this instead:

```py
from rl.project_rl_project.trainer import Trainer
```

## Training Strategy

The training agent utilizes an **ε-greedy strategy** when selecting an action to be applied to the environment. So There is an $ε$ probability chance that a random action will be taken (exploration), there is a $1-ε$ probability that we will follow the policy (the policy network) to determine the action. 

**Experience replay** is also used to aid the Neural Network training, since it has been shown to help better approximate $Q(s, a)$ by mitigating the high, but harmful, correlation between sequential $S A R S'$ pairs (also known as *experience*). In essence, instead of learning the experience sequentially, we will first save the experience in a queue buffer, and then randomly sample from that buffer of experiences to train the network. As a result, we can maintain the independence and de-correlation of training data and not sway the optimizer to the wrong direction.

A separate **Target Network** is also used to act as a copy to the policy network and are used to predict Q-Values to train the policy network. The target network will only be periodically updated every `UPDATE_FREQ` episodes with the policy network weights. This method of using a separate network is shown to make training more stable and prevent catastrophic forgetting in the policy network over a long run (usually resembled by sine-wave-like reward-episode plot).

For Vision Models, the last 2 frames will be stacked
in history instead of taking the subtraction between them.
Furthermore, the model optimization step
after the conclusion of each episode, which differs a lot from
the usual method of applying optimization after each step in
the environment. We found that doing episodal, less frequent
updates far produces faster and more stable convergence for
vision models.

## Supported OpenAI Gym Environments

Currently the only supported environments are `CartPole-v1`, and `Acrobot-v1`. 
Other environments (even outside of classic control) with `Discrete` inputs should theoritically not cause issues for the trainer, but are disabled for now since they are not yet tested.
If you wish to remove the environment constraint, you do so in the `EnvManager` class in `utils.py`.

## Future Improvements

- [ ] Support for non `Discrete` inputs
- [ ] Support other, more state-of-the-art, Reinforcement Learning algorithms especially since DQN is mostly obsolete today
- [ ] Support for Vision Transformer

## References

The training method implemented in this library is inspired from [DQN Tutorial from PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
