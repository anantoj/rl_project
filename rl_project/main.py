from typing import NamedTuple, Tuple
from collections import namedtuple

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18

from networks.baseline_model import BaselineModel
from networks.image_model import ImageModel
from utils import EnvManager, EpsilonGreedyStrategy, Agent, ReplayMemory, QValues


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state')
)

# Ignore OpenAI Depracation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO: Capitalize Constants

BATCH_SIZE = 256
NUM_STREAKS = 30 # number of streaks to indicate completion
MAX_TIMESTEP = 500 # max timestep per episode for truncation

DISCOUNT_FACTOR = 0.95 # discount factor
UPDATE_FREQ = 10 # rate of target network update

# Epsilon Greedy Strategy hyperparameters
EPS_START = 1 # starting epsilon
EPS_END = 0.01 # end epsilon
EPS_DECAY = 0.001 # epsilon decay rate


MEMORY_SIZE = 100000 # Replay Memory capacity
LEARNING_RATE = 0.001
NUM_EPISODES = 1000


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = EnvManager("CartPole-v1", device)
    strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
    agent = Agent(strategy, env.get_action_space(), device)
    memory = ReplayMemory(MEMORY_SIZE)
 
    # Initialize policy network and target network

    # policy_net = ImageModel(env.num_state_features(), env.get_action_space(), resnet18()).to(device)
    policy_net = BaselineModel(env.num_state_features(), env.get_action_space()).to(device)

    # target_net = ImageModel(env.num_state_features(), env.get_action_space(), resnet18()).to(device)
    target_net = BaselineModel(env.num_state_features(), env.get_action_space()).to(device)

    # Copy policy network weights for uniformity
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

    all_rewards = []

    # For each episode:
    for episode in range(NUM_EPISODES):

        # reset env
        env.reset()
        env.done = False

        # Initialize the starting state.
        state = env.get_state()

        timestep = 0
        episode_reward = 0
        
        while not env.done or timestep < MAX_TIMESTEP:
            timestep+=1
            
            # TODO: Render Option
            # env.render()

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
    
            if memory.can_provide_sample(BATCH_SIZE):

                # Extract sample from memory queue if able to
                experiences = memory.sample(BATCH_SIZE)
                
                # Convert experience to tensors
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)

                target_q_values = rewards + (DISCOUNT_FACTOR * next_q_values)
                    
                # Calculate loss between output Q-values and target Q-values.
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

                # Update policy_net weights from loss
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

            # If episode is DONE or TRUNCATED, 
            if env.done or timestep >= MAX_TIMESTEP:
                all_rewards.append(episode_reward)

                print(f"Episode: {len(all_rewards)} | Episode Reward: {episode_reward}")

                break

        # Update target_net with policy_net
        if episode % UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    pass

def extract_tensors(experiences: NamedTuple) -> Tuple[torch.TensorType]:
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


if __name__ == "__main__":
    assert gym.__version__ == "0.25.2", "OpenAI Gym version is not 0.25.2"
    main()



