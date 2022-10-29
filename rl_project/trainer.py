import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import NamedTuple, Tuple
from collections import namedtuple

from rl_project.networks.baseline_model import BaselineModel
from rl_project.networks.image_model import ImageModel
from rl_project.utils import EnvManager, EpsilonGreedyStrategy, Agent, ReplayMemory, QValues

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state')
)

# Ignore OpenAI Depracation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Trainer():
    def __init__(self, 
            training_mode="pos", env="CartPole-v1", use_baseline=True, model=None,
            batch_size=256, num_streaks=30, max_timestep=500, discount_factor=0.95,
            update_freq=10, eps_start=1, eps_end=0.01, eps_decay=0.001, 
            memory_size=100000, learning_rate=0.001, num_episodes=1000,
            render=False):

        self.training_mode = training_mode
        if self.training_mode not in ["pos", "img"]:
            raise ValueError("Available modes are 'pos' or 'img'")

        self.env = env
        self.use_baseline = use_baseline
        self.model = model
        
        self.batch_size = batch_size
        self.num_streaks = num_streaks # number of streaks to indicate completion
        self.max_timestep = max_timestep #  max timestep per episode for truncation
        self.discount_factor = discount_factor # discount factor
        self.update_freq = update_freq  # rate of target network update

        # Epsilon Greedy Strategy hyperparameters
        self.eps_start = eps_start # starting epsilon
        self.eps_end = eps_end # end epsilon
        self.eps_decay = eps_decay # epsilon decay rate

        self.memory_size = memory_size # Replay Memory capacity
        self.learning_rate = learning_rate
        self.num_episodes= num_episodes

        self.render = render

    def train(self) -> None:

        assert gym.__version__ == "0.25.2", "OpenAI Gym version is not 0.25.2"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = EnvManager("CartPole-v1", device)
        strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        agent = Agent(strategy, env.get_action_space(), device)
        memory = ReplayMemory(self.memory_size)
    
        # Initialize policy network and target network
        
        if self.use_baseline:
            policy_net = BaselineModel(env.num_state_features(), env.get_action_space()).to(device)
            target_net = BaselineModel(env.num_state_features(), env.get_action_space()).to(device)
        else:
            policy_net = ImageModel(env.num_state_features(), env.get_action_space(), self.model).to(device)
            target_net = ImageModel(env.num_state_features(), env.get_action_space(), self.model).to(device)
        

        # Copy policy network weights for uniformity
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(params=policy_net.parameters(), lr=self.learning_rate)

        all_rewards = []

        # For each episode:
        for episode in range(self.num_episodes):

            # reset env
            env.reset()
            env.done = False

            # Initialize the starting state.
            state = env.get_state()

            timestep = 0
            episode_reward = 0
            
            while not env.done or timestep < self.max_timestep:
                timestep+=1
                
                if self.render:
                    env.render()

                # Select an to take action using policy network
                action = agent.select_action(state, policy_net)

                # Apply action and accumulate reward
                reward = env.take_action(action)
                episode_reward += reward.item()
                
                # Record state that is the resultant of action taken
                next_states = env.get_state()
                
                # Save Experience of SARS 
                memory.push(Experience(state, action, reward, next_states))
                state = next_states
                
                # Learn
                if memory.can_provide_sample(self.batch_size):
                    # Extract sample from memory queue if able to
                    experiences = memory.sample(self.batch_size)
                    
                    # Convert experience to tensors
                    states, actions, rewards, next_states = self.extract_tensors(experiences)

                    # RECALL Q-Learning update formula: Q(S) = Q(S) + a[R + y*Q(S') - Q(S)], where a is lr and y is discount

                    # use policy network to calculate state-action values Q(S) for current state S
                    current_q_values = QValues.get_current(policy_net, states, actions)
                    
                    # use target network to calculate state-action values Q(S') for next state S'
                    next_q_values = QValues.get_next(target_net, next_states)

                    # R + y*V(S')
                    expected_q_values = rewards + (self.discount_factor * next_q_values)
                        
                    # Calculate loss between output Q-values and target Q-values. [R + y*Q(S') - Q(S)]
                    loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

                    # Update policy_net weights from loss
                    loss.backward()
                    optimizer.step() # Q(S) + a[R + y*Q(S') - Q(S)]

                    optimizer.zero_grad()

                # If episode is DONE or TRUNCATED, 
                if env.done or timestep >= self.max_timestep:
                    all_rewards.append(episode_reward)
                    print(f"Episode: {len(all_rewards)} | Episode Reward: {episode_reward}")

                    break

            # Update target_net with policy_net
            if episode % self.update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

    def extract_tensors(self, experiences: NamedTuple) -> Tuple[torch.TensorType]:
        """Convert list of Experience into tensors for each component of SARS

        Parameters
        ----------
        experiences : NamedTuple
            Experience namedtuple of (state, action, reward, next_state)

        Returns
        -------
        Tuple[torch.TensorType]
            Tuple of size 4 containing SARS Tensors with length batch_size sample
        """
        batch = Experience(*zip(*experiences))
        
        t_states = torch.stack(batch.state) # use stack instead of cat because each state is array
        t_actions = torch.cat(batch.action)
        t_next_state = torch.stack(batch.next_state)
        t_rewards = torch.cat(batch.reward)

        return (t_states,  t_actions, t_rewards, t_next_state)



