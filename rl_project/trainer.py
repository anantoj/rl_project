import gym

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import torch.nn.functional as F

from typing import NamedTuple, Tuple
from collections import namedtuple, deque

from .networks.baseline_model import BaselineModel, BaselineVisionModelV2
from .networks.image_model import VisionModel
from .utils import EnvManager, EpsilonGreedyStrategy, Agent, ReplayMemory, QValues

Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)

# Ignore OpenAI Depracation Warning
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        env="CartPole-v1",
        model=None,
        batch_size=256,
        num_streaks=100,
        target_reward=195,
        max_timestep=500,
        discount_factor=0.999,
        update_freq=10,
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=3000,
        memory_size=100000,
        learning_rate=0.001,
        num_episodes=1000,
        render=False,
        verbose=True,
        mode="pos",
    ):

        self.env = env
        self.model = model

        self.batch_size = batch_size
        self.num_streaks = num_streaks  # number of streaks to indicate completion
        self.target_reward = target_reward  # reward to achieve
        self.max_timestep = max_timestep  #  max timestep per episode for truncation
        self.discount_factor = discount_factor  # discount factor
        self.update_freq = update_freq  # rate of target network update

        # Epsilon Greedy Strategy hyperparameters
        self.eps_start = eps_start  # starting epsilon
        self.eps_end = eps_end  # end epsilon
        self.eps_decay = eps_decay  # epsilon decay rate

        self.memory_size = memory_size  # Replay Memory capacity
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes

        self.render = render
        self.verbose = verbose

        self.mode = mode
        if self.mode not in ["pos", "img"]:
            raise ValueError("Available modes are 'pos' or 'img'")

    def train(self) -> None:

        assert gym.__version__ == "0.25.2", "OpenAI Gym version is not 0.25.2"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = EnvManager(self.env, device, self.mode)
        strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        agent = Agent(strategy, env.get_action_space(), device, self.mode)
        memory = ReplayMemory(self.memory_size)

        # Initialize policy network and target network

        if self.model is None:
            if self.mode == "pos":

                policy_net = BaselineModel(
                    env.num_state_features(), env.get_action_space()
                ).to(device)
                target_net = BaselineModel(
                    env.num_state_features(), env.get_action_space()
                ).to(device)
            elif self.mode == "img":
                policy_net = BaselineVisionModelV2(
                    60, 135, env.get_action_space()
                ).to(device)
                target_net = BaselineVisionModelV2(
                    60, 135, env.get_action_space()
                ).to(device)

        else:
            if self.mode == "pos":
                policy_net = VisionModel(
                    self.model, env.num_state_features(), env.get_action_space()
                ).to(device)
                target_net = VisionModel(
                    self.model, env.num_state_features(), env.get_action_space()
                ).to(device)
            elif self.mode == "img":
                policy_net = VisionModel(
                    self.model, mode="img", out_features=env.get_action_space()
                ).to(device)
                target_net = VisionModel(
                    self.model, mode="img", out_features=env.get_action_space()
                ).to(device)

        # Copy policy network weights for uniformity
        target_net.load_state_dict(policy_net.state_dict())

        # target net only for inference
        target_net.eval()

        optimizer = optim.Adam(params=policy_net.parameters(), lr=self.learning_rate)

        all_rewards = []

        # For each episode:
        for episode in range(self.num_episodes):
            # reset env
            env.reset()
            env.done = False

            # Initialize the starting state.
            if self.mode == "pos":

                state = env.get_state()
            elif self.mode == "img":
                start_screen = env.get_state()
                screens = deque([start_screen] * 2, 2)
                state = torch.cat(list(screens), dim=1)

            timestep = 0
            episode_reward = 0

            while not env.done or timestep < self.max_timestep:
                timestep += 1

                if self.render:
                    env.render()
                # Select an to take action using policy network
                action = agent.select_action(state, policy_net)

                # Apply action and accumulate reward
                reward, done = env.take_action(action)

                # Record state that is the resultant of action taken
                if self.mode == "pos":
                    next_state = env.get_state()
                elif self.mode == "img":
                    screens.append(env.get_state())
                    next_state = torch.cat(list(screens), dim=1)

                episode_reward += reward.item()

                # Save Experience of SARS-d
                memory.push(Experience(state, action, reward, next_state, done))

                state = next_state

                if self.mode == "pos":
                    self.optimize(memory, policy_net, target_net, optimizer)

                # If episode is DONE or TRUNCATED,
                if env.done or timestep >= self.max_timestep:
                    if self.mode == "img":
                        self.optimize(memory, policy_net, target_net, optimizer)

                    all_rewards.append(episode_reward)
                    if self.verbose:
                        print(
                            f"Episode: {len(all_rewards)} | "
                            f"Reward: {episode_reward} | "
                            f"Average reward in {self.num_streaks} episodes : {self.get_average_reward(all_rewards,self.num_streaks)} | "
                            f"current exp rate: {strategy.get_exploration_rate(agent.current_step)}"
                        )
                    break

            # Update target_net with policy_net
            if episode % self.update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Preemptively end training if target is reached
            if (
                len(all_rewards) >= self.num_streaks
                and self.get_average_reward(all_rewards, self.num_streaks)
                >= self.target_reward
            ):
                print(f"Solved {self.env.unwrapped.spec.id} in {episode} episodes!")
                break

    def optimize(self, memory, policy_net, target_net, optimizer):
        if memory.can_provide_sample(self.batch_size):
            experiences = memory.sample(self.batch_size)

            # Convert experience to tensors
            states, actions, rewards, next_states, dones = self.extract_tensors(
                experiences
            )

            # RECALL Q-Learning update formula: Q(S) = Q(S) + a[R + y*Q(S') - Q(S)], where a is lr and y is discount

            # use policy network to calculate state-action values Q(S) for current state S
            current_q_values = QValues.get_current(policy_net, states, actions)

            # use target network to calculate state-action values Q(S') for next state S'
            next_q_values = QValues.get_next(target_net, next_states, dones)

            # R + y*V(S')
            expected_q_values = rewards + (self.discount_factor * next_q_values)

            # Calculate loss between output Q-values and target Q-values. [R + y*Q(S') - Q(S)]
            loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

            # Update policy_net weights from loss
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()  # Q(S) + a[R + y*Q(S') - Q(S)]
        return

    def get_average_reward(self, reward_list: List, num_streaks: int) -> int:
        """Calculates the mean reward from the most recent n-episodes

        Parameters
        ----------
        reward_list : List
            List of all rewards obtained
        num_streaks : int
            Most recent n-episodes to take

        Returns
        -------
        int
            average reward
        """
        if len(reward_list) < num_streaks:
            return round(sum(reward_list) / len(reward_list), 2)

        return sum(reward_list[-num_streaks:]) / num_streaks

    def extract_tensors(self, experiences: NamedTuple) -> Tuple[torch.TensorType]:
        """Convert list of Experience into tensors for each component of SARS-d

        Parameters
        ----------
        experiences : NamedTuple
            Experience namedtuple of (state, action, reward, next_state, done)

        Returns
        -------
        Tuple[torch.TensorType]
            Tuple of size 5 containing SARS-d Tensors with length batch_size sample
        """
        batch = Experience(*zip(*experiences))

        if self.mode == "pos":
            # use stack instead of cat because each state is 1-d array
            t_states = torch.stack(batch.state)
            t_next_state = torch.stack(batch.next_state)

        elif self.mode == "img":
            t_states = torch.cat(batch.state)
            t_next_state = torch.cat(batch.next_state)

        t_actions = torch.cat(batch.action)
        t_rewards = torch.cat(batch.reward)
        t_dones = torch.cat(batch.done)

        return (t_states, t_actions, t_rewards, t_next_state, t_dones)
