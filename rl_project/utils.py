import random
import math
from typing import List
import torch
import torchvision.transforms as T 
import gym
from collections import deque

class ReplayMemory:
    # Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    def __init__(self, capacity: int):
        """Instantiates a queue data structure for storing Experiences.
        Queue is used so that adding new experience will pop an old experience when queue is full

        Parameters
        ----------
        capacity : int
            queue length or capacity
        """
        self.memory = deque([],maxlen=capacity)

    def push(self, experience) -> None: 
        """Push an Experience to memeory queue.

        Parameters
        ----------
        experience : Experience
            SARS Experience namedtuple
        """
        self.memory.append(experience)
    

    def sample(self, batch_size: int) -> List:
        """Returns a sample of size batch_size from memory queue

        Parameters
        ----------
        batch_size : int
            number of samples to be taken

        Returns
        -------
        List
            list with length batch_size containing Experiences from memory
        """

        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        """Checks if memory length can be sampled (queue length greater than batch_size)

        Returns
        -------
        bool
            boolean indicating if memory is greater than batch_size
        """
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy:
    def __init__(self, start:float, end:float, decay:float):
        self.eps_start = start
        self.eps_end = end
        self.eps_decay = decay

    def get_exploration_rate(self, current_step: int) -> float:
        """Returns exploration rate or probability of selection random action.
        Exploration rate decreases as the agent takes more steps.

        Parameters
        ----------
        current_step : int
            total number of steps the agent has taken throughout training. 

        Returns
        -------
        float
            exploration rate
        """

        # Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        return self.eps_end + (self.eps_start - self.eps_end) *math.exp(-1. * current_step * self.eps_decay)

class Agent:
    def __init__(self, strategy, num_actions:int, device):
    
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device

    def select_action(self, state:torch.Tensor, policy_net) -> torch.Tensor:
        """Select an action to be applied to the environment using Epsilon Greedy.
        There is an epsilon % chance that a random action will be taken (exploration).
        There is a 1-epsilon % chance that we will use the policy net to determine the action.

        Parameters
        ----------
        state : torch.Tensor
            _description_
        policy_net : nn.Module
            _description_

        Returns
        -------
        torch.Tensor
            action tensor
        """

        epsilon_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1 # update step to decay epsilon

        
        if random.random() < epsilon_rate: # explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device=self.device) # explore
        else: # exploit
            with torch.no_grad(): 
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(device=self.device)



class EnvManager:
    def __init__(self, env:str, device):

        supported_envs = [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0"
        ]

        if env not in supported_envs:
            raise ValueError(f"{env} environment is currently unsupported.")

        self.device = device
        self.env = gym.make(env, new_step_api=False).unwrapped 
        self.env.reset()
        self.current_state = None
        self.done = False

    def reset(self) -> None:
        """Resets the environment to initial state
        """
        self.current_state = self.env.reset()

    def close(self) -> None:
        """Closes the environment
        """
        self.env.close()

    def render(self, mode:str='human') -> None:
        """Renders environment in PyGame

        Parameters
        ----------
        mode : str, optional
            rendering mode, by default 'human'
        """
        self.env.render(mode)

    def get_action_space(self) -> int:
        """_summary_

        Returns
        -------
        int
            number of actions that can be taken in the environment
        """

        return self.env.action_space.n
        

    def take_action(self, action:torch.Tensor) -> torch.Tensor:
        """Applies an action into the environment, update the current state, 
            and return the rewards received

        Parameters
        ----------
        action : torch.Tensor
            tensor of shape (1) containing the action to be taken

        Returns
        -------
        torch.Tensor
            tensor of shape (1) containing the reward received
        """

        self.current_state, reward, self.done, _, _= self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def get_state(self) -> torch.Tensor:
        """Returns current state

        Returns
        -------
        torch.Tensor
            State tensor with size corresponding to observation space
        """
        if self.done:
            return torch.zeros_like(
              torch.tensor(self.current_state, device=self.device)
            ).float()
        else:
            return torch.tensor(self.current_state, device=self.device).float()

    def num_state_features(self) -> int:
        """Returns the environment observation space

        Returns
        -------
        int
            number of observable variables in the environment
        """
        return self.env.observation_space.shape[0]

  
class QValues:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_current(policy_net, states, actions):
        """Use policy network to calculate state-action values for a batch of state-action pairs

        Parameters
        ----------
        policy_net : nn.Module
            Policy Network
        states : torch.Tensor
            Tensor of size (batch_size, env_obs_space) containing sampled states from memory
        actions : torch.Tensor
            Tensor of size (batch_size) containing sampled corresponding actions from memory

        Returns
        -------
        torch.Tensor
            Tensor of size (batch_size, 1) containing Q-values for each input state-action pair
        """
        q_values = policy_net(states)
        
        return q_values.gather( # only select q values for specific action (eg. if action is 0 then only choose q of index 0)
            dim=1, # action dim
            index=actions.unsqueeze(-1) # select the specific action
        )
        
    

    @staticmethod
    def get_next(target_net, next_states):
        """Use target network to calculate state-action values, specifically for non-terminal next states S'

        Parameters
        ----------
        target_net : nn.Module
            Target Network
        next_states : torch.Tensor
            Tensor of size (batch_size, env_obs_space) containing transitioned states S' from memory

        Returns
        -------
        torch.Tensor
            Tensor of size (batch_size) containing Q-values for each
        """
        # find location of terminal states in S' batch
        terminal_states_location = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool) 
        
        # select non-terminal states
        non_terminal_states_locations = (terminal_states_location == False)
        non_terminal_states = next_states[non_terminal_states_locations]

        # initialize zeros tensor of size (batch_size) 
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)

        # use target net to calculate q values for non-terminal states. Q values for terminal states are 0
        values[non_terminal_states_locations] = target_net(non_terminal_states).max(dim=1)[0].detach()
        return values