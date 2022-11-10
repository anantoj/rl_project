import random
import math
from typing import List, Tuple
import torch
import torchvision.transforms as T
import gym
from collections import deque
import numpy as np
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from torchvision.utils import save_image


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
        self.memory = deque([], maxlen=capacity)

    def push(self, experience) -> None:
        """Push an Experience to memeory queue.

        Parameters
        ----------
        experience : Experience
            SARS-d Experience namedtuple
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
    def __init__(self, start: float, end: float, decay: float):
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
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * current_step / self.eps_decay)

        


class Agent:
    def __init__(self, strategy, num_actions: int, device, mode="pos"):

        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device
        self.mode = mode

    def select_action(self, state: torch.Tensor, policy_net) -> torch.Tensor:
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
        self.current_step += 1  # update step to decay epsilon

        if random.random() < epsilon_rate:  # explore
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
            # action = random.randrange(self.num_actions)
            # return torch.tensor([action]).to(device=self.device)  

        else:  # exploit
            with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                if self.mode == "pos":
                    return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(device=self.device)
                elif self.mode == "img":
                    return policy_net(state).max(1)[1].view(1, 1)
                    # return policy_net(state).argmax(dim=1).to(self.device) 


class EnvManager:
    def __init__(self, env: str, device, mode="pos"):

        supported_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]

        if env not in supported_envs:
            raise ValueError(f"{env} environment is currently unsupported.")

        self.device = device
        self.env = gym.make(env, new_step_api=False).unwrapped
        self.env.reset()
        self.current_state = None
        self.current_screen = None
        self.done = False
        self.mode = mode

    def reset(self) -> None:
        """Resets the environment to initial state"""
        if self.mode == "pos":
            self.current_state = self.env.reset()
        elif self.mode == "img":
            self.env.reset()
            self.current_screen = None

    def close(self) -> None:
        """Closes the environment"""
        self.env.close()

    def render(self, mode: str = "human") -> None:
        """Renders environment in PyGame

        Parameters
        ----------
        mode : str, optional
            rendering mode, by default 'human'
        """
        return self.env.render(mode)

    def get_action_space(self) -> int:
        """_summary_

        Returns
        -------
        int
            number of actions that can be taken in the environment
        """

        return self.env.action_space.n

    def take_action(self, action: torch.Tensor) -> Tuple[torch.TensorType]:
        """Applies an action into the environment, update the current state,
            and return the rewards received

        Parameters
        ----------
        action : torch.Tensor
            tensor of shape (1) containing the action to be taken

        Returns
        -------
        Tuple(torch.Tensor, torch.Tensor)
            Tuple containing reward tensor and done status tensor
        """

        
        if self.mode == "pos":
            self.current_state, reward, self.done, _, _ = self.env.step(action.item())
        elif self.mode == "img":
            _, reward, self.done, _, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device), torch.tensor([self.done], device=self.device)

    def get_state(self) -> torch.Tensor:
        """Returns current state

        Returns
        -------
        torch.Tensor
            State tensor with size corresponding to observation space
        """
        if self.mode == "pos":
            if self.done:
                return torch.zeros_like(
                    torch.tensor(self.current_state, device=self.device)
                ).float()
            else:
                return torch.tensor(self.current_state, device=self.device).float()
        
        elif self.mode == "img":
            return self.get_screen()
            # # if start or terminal state
            # if self.just_starting() or self.done:
            #     # return black screen (image tensor of zeros)
            #     self.current_screen = self.get_processed_screen()
            #     black_screen = torch.zeros_like(self.current_screen)
            #     return black_screen
            # else:
            #     # current screen
            #     s1 = self.current_screen
            #     # call a new screen (the next screen)
            #     s2 = self.get_processed_screen()
            #     # make the next screen is the current screen
            #     self.current_screen = s2
            #     # return the difference between two screens to get the current state
            #     return s2 - s1

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    # Cropping, downsampling (and Grayscaling) image
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((60,135), interpolation=InterpolationMode.BICUBIC),
            T.Grayscale(),  
            T.ToTensor()
        ])
        return transforms(screen).unsqueeze(0).to(self.device)

    def num_state_features(self) -> int:
        """Returns the environment observation space

        Returns
        -------
        int
            number of observable variables in the environment
        """
        return self.env.observation_space.shape[0]

    def just_starting(self) -> bool:
        """Checks if we are in a start state

        Returns
        -------
        bool
            start state status
        """
        return self.current_screen is None


    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        screen = np.ascontiguousarray(screen,dtype=np.float32)
        screen = torch.from_numpy(screen)
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((60,135), interpolation=InterpolationMode.BICUBIC),
            T.Grayscale(),  
            T.ToTensor()
        ])

        return transforms(screen).unsqueeze(0).to(self.device)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]

        return screen

class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        return q_values.gather(  # only select q values for specific action (eg. if action is 0 then only choose q of index 0)
            1,  # action dim
            actions,  # select the specific action
        )

    def get_next(target_net, next_states, dones):
        """Use target network to calculate state-action values, specifically for non-terminal next states S'

        Parameters
        ----------
        target_net : nn.Module
            Target Network
        next_states : torch.Tensor
            Tensor of size (batch_size, env_obs_space) containing transitioned states S' from memory
        dones: torch.Tensor
            Tensor of size (batch_size) denoting terminality of each state in S'

        Returns
        -------
        torch.Tensor
            Tensor of size (batch_size) containing Q-values for each
        """
        # find location of non-terminal states in S' batch
        non_terminal_states_locations = (dones == False)

        # select non-terminal states
        non_terminal_states = next_states[non_terminal_states_locations]

        # initialize zeros tensor of size (batch_size)
        batch_size = next_states.shape[0]

        values = torch.zeros(batch_size).to(QValues.device)

        # use target net to calculate q values for non-terminal states. Q values for terminal states are 0
        values[non_terminal_states_locations] = (
            target_net(non_terminal_states).max(dim=1)[0].detach()
        )
        return values
