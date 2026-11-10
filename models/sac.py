"""
Implementation of RLPD SAC (Soft Actor Critic)
Implementation based on:
https://spinningup.openai.com/en/latest/algorithms/sac.html
https://github.com/ikostrikov/rlpd/blob/main/rlpd/agents/sac/sac_learner.py
https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/agents/continuous/sac.py
"""

from dataclasses import dataclass
from collections import deque
import random

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim:int):
        """
        Q: Takes in state (s) and action (a) and outputs a single scalar representing the expected return
        starting in state s, taking action a, and then following the policy.

        Args:
            state_dim: len(s)
            action_dim: len(a)
        """
        super().__init__()

        self.q_value_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.q_value_network(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        """
        π: Takes in state (s) and outputs mean and log standard deviation for each action.
        Each mean and standard deviation represent a Gaussian Action Distribution.
        
        Args:
            state_dim: len(s)
            action_dim: len(a)
        """
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 2 action heads
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor):
        shared_features = self.backbone(x)
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)

        return mean, log_std


class QNetworkLoss(nn.Module):
    def __init__(self):
        """
        Calculates the loss of a Q network given the Q-network-prediction and the 
        target value.
        L_i = E[ ( Q_i(s,a) - y(r,s',d) )^2 ]
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, q_prediction: torch.Tensor, target_value: torch.Tensor):

        loss = self.mse_loss(q_prediction, target_value)

        return loss


class ActorLoss(nn.Module):
    def __init__(self):
        """
        Calculates the loss of the π Actor network given the weighted entropy term and both
        predicted Q values
        L  = E[ weighted_entropy - min[Q_1(s,a), Q_2(s, a)] ]
        """
        super().__init__()

    def forward(self, weighted_entropy: torch.Tensor, q1_prediction: torch.Tensor,
                q2_prediction: torch.Tensor):

        loss = (weighted_entropy - torch.minimum(q1_prediction, q2_prediction)).mean()

        return loss


class SACAgent():

    @dataclass
    class Transition:
        state: torch.Tensor
        action: torch.Tensor
        next_state: torch.Tensor
        reward: float
        done: int  # 0 or 1

    def __init__(self, state_dim:int, action_dim:int, policy_updates_per_env_step:int=1,
                 replay_buffer_size:int=10e6, batch_size:int=256, learning_rate:int=3e-4,
                 discount_factor:float=0.99):

        self.actor = ActorNetwork(state_dim, action_dim)

        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)

        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        # Set target weights equal to Q
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # Freeze target networks bc they are not optimized directly
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        self.actor_optimizer = torch.optim.adam(self.actor.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.adam(self.q2.parameters(), lr=learning_rate)

        # D: stores state transitions Transition {s,a,s',r,d}
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.policy_updates_per_env_step = policy_updates_per_env_step

        self.batch_size = batch_size

        self.discount_factor = discount_factor

    def select_action(self, state:torch.Tensor):

        # Add in batches (batch_size, state_dim)
        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            # Reparameterization trick
            mean, log_std = self.actor(state)
            noise = torch.randn_like(mean)
            action = torch.tanh(mean + log_std * noise)

        return action.squeeze()  # Remove batch

    def update_policy(self, state:torch.Tensor, action:torch.Tensor, next_state:torch.Tensor, 
                      reward:float, done:int):
        """
        Update the gradients of the policy based on the original state, taken action, 
        and result.

        Args:
            state: s
            action: a
            next_state: s'
            reward: r
            done: d
        """

        # Add transition information to replay buffer
        transition = self.Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
        )
        self.replay_buffer.append(transition)


        for _ in range(self.policy_updates_per_env_step):

            # Randomly sample a batch of transitions
            batch = random.sample(list(self.replay_buffer), self.batch_size)

            # Compute target for Q function
            # target = reward + (1 - self.discount_factor)

            # Update Q function gradients

            # Update policy gradient

            # Update target networks


        return




