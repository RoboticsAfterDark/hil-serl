"""
Implementation of RLPD SAC (Soft Actor Critic)
Implementation based on:
https://spinningup.openai.com/en/latest/algorithms/sac.html
https://github.com/ikostrikov/rlpd/blob/main/rlpd/agents/sac/sac_learner.py
https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/agents/continuous/sac.py
"""

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
    def __init__(self):

        LEARNING_RATE = 3e-4
        BATCH_SIZE = 256

        self.actor = ActorNetwork()
        self.q1 = QNetwork()
        self.q2 = QNetwork()
        self.q1_target = QNetwork()
        self.q2_target = QNetwork()

        self.actor_optimizer = torch.optim.adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = torch.optim.adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = torch.optim.adam(self.q2.parameters(), lr=LEARNING_RATE)


    def step():
        return




